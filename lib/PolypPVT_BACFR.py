"""PolypPVT + BACFR-style boosters.

End-to-end PolypPVT (PVT-v2-B2 backbone + CFM/CIM/SAM heads) with the same
component bag that produced the +9% lift on BACFR:

  * Coarse-mask early-fusion (BACFR-style): a small zero-init conv maps the
    coarse mask to a 3-channel residual added to the image input. Optional;
    set use_mask_input=False to train as vanilla PolypPVT-on-patches.
  * HFGate on the deepest backbone feature (PVT stage 4, 512 channels).
  * Dual fg/bg heads on sam_feature (channel=32) with uncertainty-weighted
    BCE on the main head.
  * Flip-consistency training (FCT): batch tripled with [orig, hflip, vflip],
    MSE between sigmoid(pred_orig) and un-flipped sigmoid(pred_{h,v}).

Inference forward signature matches BACFR_Enhanced_v3_3 so Test_patch_tta.py
works unchanged once the config name points at this model.

Constructor accepts the full kwarg surface used by run/Train_patch.py:
    channels, output_stride, pretrained,
    use_mccpb, use_dual_heads, use_boundary_contrast, use_hf_gate,
    use_flip_consistency, edge_dist_mode
Args that don't apply to PolypPVT (output_stride, use_mccpb,
use_boundary_contrast, edge_dist_mode) are accepted and silently ignored.

Required external files:
    lib/pvtv2.py                       (PVT-v2-B2 class)
    pretrained_pth/pvt_v2_b2.pth       (ImageNet pretrained weights)

Sample contract (unchanged from BACFR_Enhanced_v3_3):
    out = model({'image': x, 'mask': coarse_mask, 'gt': y})
    -> {'pred', 'loss', 'debug', 'debug_loss', 'fg_pred', 'bg_pred',
        ...consistency_loss/flip_lambda if FCT...}
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pvtv2 import pvt_v2_b2
from .BACFR_Enhanced_v3 import HFGate, _main_loss_fn


# ============================================================
# PolypPVT building blocks (verbatim from official PolypPVT)
# ============================================================
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        return self.bn(self.conv(x))


class CFM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3
        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
        return self.conv4(x3_2)


class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super().__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        return self.relu(self.conv2(h))


class SAM(nn.Module):
    def __init__(self, num_in=32, plane_mid=16, mids=4, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = mids * mids
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))
        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):
        edge = F.interpolate(edge, (x.size()[-2], x.size()[-1]),
                             mode='bilinear', align_corners=True)
        n = x.size(0)
        edge = torch.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1) \
            if edge.size(1) > 1 else torch.sigmoid(edge)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge

        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1),
                                       x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.softmax(x_proj_reshaped, dim=1)

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_proj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        return x + self.conv_extend(x_state)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


# ============================================================
# PolypPVT + BACFR boosters
# ============================================================
class PolypPVT_BACFR(nn.Module):
    PVT_CKPT_PATH = './pretrained_pth/pvt_v2_b2.pth'

    def __init__(self, channels=32, output_stride=None, pretrained=True,
                 use_mccpb=False,
                 use_dual_heads=False,
                 use_boundary_contrast=False,
                 use_hf_gate=False,
                 use_flip_consistency=True,
                 use_mask_input=True,
                 flip_consistency_max=0.3,
                 edge_dist_mode='cdist',
                 beta_uncertainty=0.5,
                 gamma_consistency=1.0,
                 **kwargs):
        super().__init__()

        # Channel must be 32 to match PolypPVT decoder/SAM. We deliberately
        # ignore the channels kwarg from BACFR-style configs (which use 256)
        # because Translayers, CFM, SAM are all hardwired to channel=32.
        channel = 32

        self.use_dual_heads = use_dual_heads
        self.use_hf_gate = use_hf_gate
        self.use_flip_consistency = use_flip_consistency
        self.use_mask_input = use_mask_input
        self.flip_consistency_max = flip_consistency_max
        self.beta_u = beta_uncertainty
        self.gamma_cons = gamma_consistency

        if use_boundary_contrast:
            print('[PolypPVT_BACFR] use_boundary_contrast not supported; ignoring.')
        if use_mccpb:
            print('[PolypPVT_BACFR] use_mccpb not supported; ignoring.')

        self.register_buffer('current_epoch', torch.zeros(1, dtype=torch.long))

        # ---- PolypPVT scaffolding ----
        self.backbone = pvt_v2_b2()
        if pretrained:
            if os.path.exists(self.PVT_CKPT_PATH):
                save_model = torch.load(self.PVT_CKPT_PATH, map_location='cpu')
                model_dict = self.backbone.state_dict()
                state_dict = {k: v for k, v in save_model.items() if k in model_dict}
                model_dict.update(state_dict)
                self.backbone.load_state_dict(model_dict)
                print(f'[PolypPVT_BACFR] loaded PVT-v2 weights from {self.PVT_CKPT_PATH}')
            else:
                print(f'[PolypPVT_BACFR] WARN: {self.PVT_CKPT_PATH} not found; '
                      f'training PVT-v2 from scratch')

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.cfm = CFM(channel)
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.sam = SAM()

        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_SAM = nn.Conv2d(channel, 1, 1)
        self.out_CFM = nn.Conv2d(channel, 1, 1)

        # ---- BACFR-style coarse-mask early fusion ----
        # Outputs a 3-channel residual at full resolution, added to the RGB
        # input. Last conv is zero-init so the model starts identical to
        # vanilla PolypPVT.
        if use_mask_input:
            self.mask_to_image = nn.Sequential(
                nn.Conv2d(1, 16, 3, 1, 1, bias=False),
                nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 3, 1, 1, bias=False),
                nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, 3, 3, 1, 1, bias=True),
            )
            nn.init.zeros_(self.mask_to_image[-1].weight)
            nn.init.zeros_(self.mask_to_image[-1].bias)

        # ---- HFGate on PVT x4 (512 channels) ----
        if use_hf_gate:
            self.hf_gate = HFGate(512)

        # ---- Dual heads on sam_feature ----
        if use_dual_heads:
            self.fg_head = nn.Conv2d(channel, 1, 1)
            self.bg_head = nn.Conv2d(channel, 1, 1)
            nn.init.zeros_(self.fg_head.weight); nn.init.zeros_(self.fg_head.bias)
            nn.init.zeros_(self.bg_head.weight); nn.init.zeros_(self.bg_head.bias)

        self.loss_fn = _main_loss_fn
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear',
                                                  align_corners=False)

    # ----- epoch / schedule helpers -----
    def set_epoch(self, epoch: int):
        self.current_epoch.fill_(int(epoch))

    def _epoch(self):
        return int(self.current_epoch.item())

    def _lambda_aux(self):
        if not self.use_dual_heads:
            return 0.0
        sched = [0.00, 0.15, 0.30, 0.30, 0.30, 0.30, 0.28, 0.22, 0.12, 0.05]
        return sched[max(0, min(self._epoch(), len(sched) - 1))]

    def _lambda_flip(self):
        if not self.use_flip_consistency:
            return 0.0
        peak = self.flip_consistency_max
        sched = [0.00, peak * 0.33, peak * 0.66, peak, peak, peak,
                 peak, peak, peak * 0.8, peak * 0.5]
        return sched[max(0, min(self._epoch(), len(sched) - 1))]

    # ----- core forward -----
    def _forward_core(self, x_in, mask):
        if self.use_mask_input:
            x_in = x_in + self.mask_to_image(2.0 * mask - 1.0)

        pvt = self.backbone(x_in)
        x1, x2, x3, x4 = pvt[0], pvt[1], pvt[2], pvt[3]

        if self.use_hf_gate:
            x4 = self.hf_gate(x4)

        # CIM
        x1 = self.ca(x1) * x1
        cim_feature = self.sa(x1) * x1

        # CFM
        x2_t = self.Translayer2_1(x2)
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)
        cfm_feature = self.cfm(x4_t, x3_t, x2_t)

        # SAM
        T2 = self.Translayer2_0(cim_feature)
        T2 = self.down05(T2)
        sam_feature = self.sam(cfm_feature, T2)

        prediction1 = self.out_CFM(cfm_feature)
        prediction2 = self.out_SAM(sam_feature)
        return {
            'p1': prediction1, 'p2': prediction2,
            'sam_feat': sam_feature, 'feat_x4': x4,
        }

    # ----- public forward -----
    def forward(self, sample):
        x_in = sample['image']
        mask = sample.get('mask', None)
        if mask is None:
            mask = torch.zeros_like(x_in[:, :1])
        y = sample.get('gt', None)
        base_size = x_in.shape[-2:]

        if y is None or not self.use_flip_consistency:
            return self._forward_no_flip(x_in, mask, y, base_size)

        # ---- Flip-consistency training ----
        B = x_in.shape[0]
        x_h = torch.flip(x_in, dims=[3]); m_h = torch.flip(mask, dims=[3])
        y_h = torch.flip(y, dims=[3])
        x_v = torch.flip(x_in, dims=[2]); m_v = torch.flip(mask, dims=[2])
        y_v = torch.flip(y, dims=[2])

        x_all = torch.cat([x_in, x_h, x_v], dim=0)
        m_all = torch.cat([mask, m_h, m_v], dim=0)
        y_all = torch.cat([y, y_h, y_v], dim=0)

        core = self._forward_core(x_all, m_all)
        p1_up = self.res(core['p1'], base_size)
        p2_up = self.res(core['p2'], base_size)

        if self.use_dual_heads:
            fg_logit = self.fg_head(core['sam_feat'])
            bg_logit = self.bg_head(core['sam_feat'])
            fg_up = self.res(fg_logit, base_size)
            bg_up = self.res(bg_logit, base_size)
            with torch.no_grad():
                u = torch.sigmoid(fg_up) * torch.sigmoid(bg_up)
                u_min = u.amin(dim=(2, 3), keepdim=True)
                u_max = u.amax(dim=(2, 3), keepdim=True)
                u_norm = (u - u_min) / (u_max - u_min).clamp(min=1e-6)
                w_pix = 1.0 + self.beta_u * u_norm
        else:
            fg_up = p2_up * 0.0; bg_up = p2_up * 0.0; w_pix = None

        loss_p1 = self.loss_fn(p1_up, y_all)
        loss_p2 = (self.loss_fn(p2_up, y_all, pixel_weight=w_pix)
                   if self.use_dual_heads else self.loss_fn(p2_up, y_all))
        main_loss = loss_p1 + loss_p2

        lam_aux = self._lambda_aux()
        if self.use_dual_heads and lam_aux > 0:
            loss_fg = self.loss_fn(fg_up, y_all)
            loss_bg = self.loss_fn(bg_up, 1.0 - y_all)
            fg_p = torch.sigmoid(fg_up); bg_p = torch.sigmoid(bg_up)
            loss_cons = ((fg_p + bg_p - 1.0) ** 2).mean()
            aux_loss = lam_aux * (loss_fg + loss_bg + self.gamma_cons * loss_cons)
        else:
            aux_loss = p2_up.new_zeros(())

        # Flip consistency on the main prediction (p2 = SAM head)
        pred_orig = torch.sigmoid(p2_up[:B])
        pred_h = torch.sigmoid(p2_up[B:2 * B])
        pred_v = torch.sigmoid(p2_up[2 * B:3 * B])
        consistency_h = F.mse_loss(torch.flip(pred_h, dims=[3]), pred_orig)
        consistency_v = F.mse_loss(torch.flip(pred_v, dims=[2]), pred_orig)
        consistency = 0.5 * (consistency_h + consistency_v)

        lam_flip = self._lambda_flip()
        flip_loss = lam_flip * consistency

        loss = main_loss + aux_loss + flip_loss
        debug_loss = [loss_p2.item(), loss_p1.item(), 0.0]

        return {
            'pred': p2_up[:B],
            'loss': loss,
            'debug': [p1_up[:B], p2_up[:B]],
            'debug_loss': debug_loss,
            'fg_pred': fg_up[:B],
            'bg_pred': bg_up[:B],
            'consistency_loss': consistency.item(),
            'flip_lambda': lam_flip,
        }

    # ----- inference / no-FCT path -----
    def _forward_no_flip(self, x_in, mask, y, base_size):
        core = self._forward_core(x_in, mask)
        p1_up = self.res(core['p1'], base_size)
        p2_up = self.res(core['p2'], base_size)

        if self.use_dual_heads:
            fg_logit = self.fg_head(core['sam_feat'])
            bg_logit = self.bg_head(core['sam_feat'])
            fg_up = self.res(fg_logit, base_size)
            bg_up = self.res(bg_logit, base_size)
        else:
            fg_up = p2_up * 0.0
            bg_up = p2_up * 0.0

        if y is not None:
            if self.use_dual_heads:
                with torch.no_grad():
                    u = torch.sigmoid(fg_up) * torch.sigmoid(bg_up)
                    u_min = u.amin(dim=(2, 3), keepdim=True)
                    u_max = u.amax(dim=(2, 3), keepdim=True)
                    u_norm = (u - u_min) / (u_max - u_min).clamp(min=1e-6)
                    w_pix = 1.0 + self.beta_u * u_norm
            else:
                w_pix = None

            loss_p1 = self.loss_fn(p1_up, y)
            loss_p2 = (self.loss_fn(p2_up, y, pixel_weight=w_pix)
                       if self.use_dual_heads else self.loss_fn(p2_up, y))
            main_loss = loss_p1 + loss_p2

            lam_aux = self._lambda_aux()
            if self.use_dual_heads and lam_aux > 0:
                loss_fg = self.loss_fn(fg_up, y)
                loss_bg = self.loss_fn(bg_up, 1.0 - y)
                fg_p = torch.sigmoid(fg_up); bg_p = torch.sigmoid(bg_up)
                loss_cons = ((fg_p + bg_p - 1.0) ** 2).mean()
                aux_loss = lam_aux * (loss_fg + loss_bg + self.gamma_cons * loss_cons)
            else:
                aux_loss = p2_up.new_zeros(())

            loss = main_loss + aux_loss
            debug_loss = [loss_p2.item(), loss_p1.item(), 0.0]
        else:
            loss = torch.tensor(0.0, device=p2_up.device)
            debug_loss = []

        return {
            'pred': p2_up,
            'loss': loss,
            'debug': [p1_up, p2_up],
            'debug_loss': debug_loss,
            'fg_pred': fg_up,
            'bg_pred': bg_up,
        }
