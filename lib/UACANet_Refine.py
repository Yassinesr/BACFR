"""UACANet as a coarse-mask refiner -- faithful to the published architecture.

The UACANet paper (Kim et al., "UACANet: Uncertainty Augmented Context
Attention for Polyp Segmentation") states in Sec. 3.1:

    "PAA-d [predicts] the initial saliency map which serves as an initial
     guidance map, which leads UACA to learn a RESIDUAL saliency map apart
     from the initial map. This helps the consequent UACA focus more on
     uncertain area like boundaries rather than fairly evident region."

That is exactly the refinement formulation. In vanilla UACANet the guidance
map comes from the network's own PAA-d prediction; nothing in the architecture
requires it. Here we ANCHOR that guidance on the external coarse mask:

    guidance   = guidance_scale * (2*m - 1)        # coarse prob -> logit scale
    a5         = a5_paa_d + guidance               # anchored initial map
    a4 = a5 + UACA_res(...)                        # UACA already adds residuals
    ...
    pred       = coarse_logit + sum(learned residuals)

Because each UACA already ends with `out = out + map`, the final prediction is
the coarse mask plus learned corrections -- so the model starts from the coarse
mask instead of re-segmenting each patch blind (which is what made plain
UACANet destroy an already-accurate boundary). Every module (PAA-e, PAA-d,
UACA), the 4-way deep supervision and the loss are unchanged from the paper.

Sample keys: 'image' (B,3,H,W), 'mask' (B,1,H,W in [0,1]) and optional 'gt'.
If 'mask' is absent the model degrades gracefully to vanilla UACANet.
"""
import torch
import torch.nn.functional as F

from .UACANet import UACANet


def _hflip(t):
    return torch.flip(t, dims=[-1])


def _vflip(t):
    return torch.flip(t, dims=[-2])


class UACANet_Refine(UACANet):
    """Vanilla UACANet + coarse mask as the initial guidance map."""

    def __init__(self, channels=256, output_stride=16, pretrained=True,
                 guidance_scale=3.0, **kwargs):
        super().__init__(channels=channels, output_stride=output_stride,
                         pretrained=pretrained)
        # sigmoid(+/-3) = 0.95 / 0.05 -> a near-saturated but not extreme prior.
        self.guidance_scale = float(guidance_scale)

    def forward(self, sample):
        x = sample['image']
        m = sample.get('mask', None)
        y = sample.get('gt', None)
        base_size = x.shape[-2:]

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        x2 = self.context2(x2)
        x3 = self.context3(x3)
        x4 = self.context4(x4)

        f5, a5 = self.decoder(x4, x3, x2)

        # ---- anchor the initial guidance map on the coarse mask ----
        if m is not None:
            g = self.guidance_scale * (2.0 * m - 1.0)
            a5 = a5 + F.interpolate(g, size=a5.shape[-2:],
                                    mode='bilinear', align_corners=False)
        out5 = self.res(a5, base_size)

        f4, a4 = self.attention4(torch.cat([x4, self.ret(f5, x4)], dim=1), a5)
        out4 = self.res(a4, base_size)

        f3, a3 = self.attention3(torch.cat([x3, self.ret(f4, x3)], dim=1), a4)
        out3 = self.res(a3, base_size)

        _, a2 = self.attention2(torch.cat([x2, self.ret(f3, x2)], dim=1), a3)
        out2 = self.res(a2, base_size)

        if y is not None:
            loss5 = self.loss_fn(out5, y)
            loss4 = self.loss_fn(out4, y)
            loss3 = self.loss_fn(out3, y)
            loss2 = self.loss_fn(out2, y)
            loss = loss2 + loss3 + loss4 + loss5
            debug = [out5, out4, out3]
        else:
            loss = 0
            debug = []

        return {'pred': out2, 'loss': loss, 'debug': debug}


class UACANet_Refine_FCT(UACANet_Refine):
    """UACANet_Refine + Flip-Consistency Training.

    Identical FCT scheme to UACANet_FCT, except the COARSE MASK is flipped
    along with the image and GT (it is a model input here, not just context).
    Inference is unchanged, so the same TTA / no-TTA testers apply.
    """

    def __init__(self, channels=256, output_stride=16, pretrained=True,
                 guidance_scale=3.0,
                 fct_weight=0.05, fct_warmup_iters=300, fct_use_vflip=True,
                 use_flip_consistency=True, fct_supervise_flips=True, **kwargs):
        super().__init__(channels=channels, output_stride=output_stride,
                         pretrained=pretrained, guidance_scale=guidance_scale)
        self.fct_weight = float(fct_weight)
        self.fct_warmup_iters = int(fct_warmup_iters)
        self.fct_use_vflip = bool(fct_use_vflip)
        self.use_flip_consistency = bool(use_flip_consistency)
        self.fct_supervise_flips = bool(fct_supervise_flips)
        self.register_buffer('_fct_iter', torch.zeros((), dtype=torch.long))

    def _current_fct_weight(self):
        it = int(self._fct_iter)
        if self.fct_warmup_iters <= 0 or it >= self.fct_warmup_iters:
            return self.fct_weight
        return self.fct_weight * (it / float(self.fct_warmup_iters))

    @staticmethod
    def _stack(views, key, tensors):
        if tensors[0] is not None:
            views[key] = torch.cat(tensors, dim=0)

    def forward(self, sample):
        x = sample['image']
        y = sample.get('gt', None)
        m = sample.get('mask', None)

        if y is None or not self.use_flip_consistency:
            return super().forward(sample)

        lam = self._current_fct_weight()
        self._fct_iter += 1
        B = x.shape[0]

        if self.fct_supervise_flips:
            xs, ys = [x, _hflip(x)], [y, _hflip(y)]
            ms = [m, _hflip(m)] if m is not None else None
            if self.fct_use_vflip:
                xs.append(_vflip(x)); ys.append(_vflip(y))
                if ms is not None:
                    ms.append(_vflip(m))
            s = {'image': torch.cat(xs, 0), 'gt': torch.cat(ys, 0)}
            if ms is not None:
                s['mask'] = torch.cat(ms, 0)
            out = super().forward(s)

            prob = torch.sigmoid(out['pred'])
            p_o = prob[:B]
            cons = F.mse_loss(_hflip(prob[B:2 * B]), p_o)
            if self.fct_use_vflip:
                cons = 0.5 * (cons + F.mse_loss(_vflip(prob[2 * B:3 * B]), p_o))
            out['loss'] = out['loss'] + lam * cons
            out['pred'] = out['pred'][:B]
            out['debug'] = [d[:B] for d in out['debug']]
            return out

        # consistency-only: main loss on the original view (vanilla aug)
        out = super().forward(sample)
        p_o = torch.sigmoid(out['pred'])
        sh = {'image': _hflip(x)}
        if m is not None:
            sh['mask'] = _hflip(m)
        cons = F.mse_loss(_hflip(torch.sigmoid(super().forward(sh)['pred'])), p_o)
        if self.fct_use_vflip:
            sv = {'image': _vflip(x)}
            if m is not None:
                sv['mask'] = _vflip(m)
            cons = 0.5 * (cons + F.mse_loss(
                _vflip(torch.sigmoid(super().forward(sv)['pred'])), p_o))
        out['loss'] = out['loss'] + lam * cons
        return out
