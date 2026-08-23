"""UACANet + Flip-Consistency Training (FCT).

Subclasses vanilla UACANet. During training it triples the batch with
H-flipped (and optionally V-flipped) copies -- each supervised by its own
flipped GT through UACANet's normal deep-supervised loss -- and adds an MSE
consistency loss forcing the un-flipped predictions to agree across views.
Inference (no 'gt' in the sample) is IDENTICAL to vanilla UACANet, so the
same test scripts (run/Test_patch_tta.py for 4-view TTA) work unchanged.

The consistency weight ramps linearly from 0 to fct_weight over the first
fct_warmup_iters iterations; the counter is a registered buffer so it is
saved/restored on resume.
"""
import torch
import torch.nn.functional as F

from .UACANet import UACANet


def _hflip(t):
    return torch.flip(t, dims=[-1])


def _vflip(t):
    return torch.flip(t, dims=[-2])


class UACANet_FCT(UACANet):
    def __init__(self, channels=256, output_stride=16, pretrained=True,
                 fct_weight=0.05, fct_warmup_iters=300, fct_use_vflip=True,
                 use_flip_consistency=True, fct_supervise_flips=True, **kwargs):
        super().__init__(channels=channels, output_stride=output_stride,
                         pretrained=pretrained)
        self.fct_weight = float(fct_weight)
        self.fct_warmup_iters = int(fct_warmup_iters)
        self.fct_use_vflip = bool(fct_use_vflip)
        self.use_flip_consistency = bool(use_flip_consistency)
        # If True (default): the flipped views are ALSO supervised with their
        # flipped GT (extra flip augmentation on top of the pipeline's
        # random_flip). If False: "consistency-only" -- the main deep-supervised
        # loss is computed on the original view only (same augmentation as
        # vanilla UACANet), and the flips are used solely for the consistency
        # MSE. Use False to avoid double flip-augmentation.
        self.fct_supervise_flips = bool(fct_supervise_flips)
        # Saved in state_dict so the warmup ramp resumes correctly.
        self.register_buffer('_fct_iter', torch.zeros((), dtype=torch.long))

    def _current_fct_weight(self):
        it = int(self._fct_iter)
        if self.fct_warmup_iters <= 0 or it >= self.fct_warmup_iters:
            return self.fct_weight
        return self.fct_weight * (it / float(self.fct_warmup_iters))

    def forward(self, sample):
        x = sample['image']
        y = sample.get('gt', None)

        # Inference, or FCT disabled -> plain UACANet.
        if y is None or not self.use_flip_consistency:
            return super().forward(sample)

        lam = self._current_fct_weight()
        self._fct_iter += 1

        if self.fct_supervise_flips:
            # ---- Tripled/doubled batch: all views supervised (extra flip aug) ----
            B = x.shape[0]
            x_h, y_h = _hflip(x), _hflip(y)
            if self.fct_use_vflip:
                x_all = torch.cat([x, x_h, _vflip(x)], dim=0)
                y_all = torch.cat([y, y_h, _vflip(y)], dim=0)
            else:
                x_all = torch.cat([x, x_h], dim=0)
                y_all = torch.cat([y, y_h], dim=0)
            out = super().forward({'image': x_all, 'gt': y_all})
            prob = torch.sigmoid(out['pred'])
            p_o = prob[:B]
            cons = F.mse_loss(_hflip(prob[B:2 * B]), p_o)
            if self.fct_use_vflip:
                cons = 0.5 * (cons + F.mse_loss(_vflip(prob[2 * B:3 * B]), p_o))
            out['loss'] = out['loss'] + lam * cons
            out['pred'] = out['pred'][:B]
            out['debug'] = [d[:B] for d in out['debug']]
            return out

        # ---- Consistency-only: main loss on the ORIGINAL view (same aug as
        #      vanilla); flips used solely for the consistency MSE. ----
        out = super().forward({'image': x, 'gt': y})   # full deep-sup loss on orig
        p_o = torch.sigmoid(out['pred'])
        pred_h = super().forward({'image': _hflip(x)})['pred']  # no gt -> pred only
        cons = F.mse_loss(_hflip(torch.sigmoid(pred_h)), p_o)
        if self.fct_use_vflip:
            pred_v = super().forward({'image': _vflip(x)})['pred']
            cons = 0.5 * (cons + F.mse_loss(_vflip(torch.sigmoid(pred_v)), p_o))
        out['loss'] = out['loss'] + lam * cons
        return out
