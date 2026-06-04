"""BACFR_Enhanced v3.3 — flip-consistency training for patch refinement.

Extends v3.2 with a flip-consistency training objective:
  - Each training batch is internally doubled to include H-flipped + V-flipped copies
  - A consistency loss is added: |sigmoid(pred_orig) - flip_back(sigmoid(pred_flipped))|^2
  - At inference behavior is identical to v3.2 (no flip doubling)

The framing: this primes the model for TTA at inference. Standard TTA on baseline
models gives modest gains; TTA on flip-consistency-trained models exhibits
super-additive behavior because the model has been explicitly trained to be
flip-equivariant.

All v3.2 flags (use_mccpb, use_dual_heads, use_boundary_contrast, use_hf_gate)
remain available. Recommended starting config is the v3.2 winner:
    use_hf_gate=True, use_dual_heads=True, use_flip_consistency=True

Interface unchanged from v3.2:
    out = model({'image': ..., 'mask': ..., 'gt': ...})
    -> {'pred', 'loss', 'debug', 'debug_loss', 'fg_pred', 'bg_pred',
        'consistency_loss'}
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse all building blocks from v3.2.
from .BACFR_Enhanced_v3 import (
    AMCFM, FeatureFusionBlock, DecoderSimple,
    EDGA_v32, HFGate, BoundaryContrastLoss,
    _main_loss_fn, _dice_bce,
)
from .backbones.Res2Net_v1b import res2net50_v1b_26w_4s


class BACFR_Enhanced_v3_3(nn.Module):
    """v3.2 architecture + flip-consistency training.

    During training (when sample['gt'] is provided), the batch is internally
    doubled with H-flipped + V-flipped copies, the model predicts all views,
    and a consistency loss enforces equivariance.

    During inference (no 'gt' in sample), behavior is identical to v3.2.
    """

    def __init__(self, channels=256, output_stride=16, pretrained=True,
                 use_mccpb=False,
                 use_dual_heads=False,
                 use_boundary_contrast=False,
                 use_hf_gate=False,
                 use_flip_consistency=True,      # NEW
                 flip_consistency_max=0.3,       # NEW: peak loss weight
                 edge_dist_mode='cdist',
                 beta_uncertainty=0.5, gamma_consistency=1.0,
                 bc_margin=0.3, bc_push_weight=0.5):
        super().__init__()

        self.use_mccpb = use_mccpb
        self.use_dual_heads = use_dual_heads
        self.use_boundary_contrast = use_boundary_contrast
        self.use_hf_gate = use_hf_gate
        self.use_flip_consistency = use_flip_consistency
        self.flip_consistency_max = flip_consistency_max
        self.beta_u = beta_uncertainty
        self.gamma_cons = gamma_consistency

        self.register_buffer('current_epoch', torch.zeros(1, dtype=torch.long))

        # ---- Identical scaffolding to v3.2 ----
        self.mask_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
        )

        self.resnet = res2net50_v1b_26w_4s(pretrained=pretrained,
                                           output_stride=output_stride)

        if use_hf_gate:
            self.hf_gate = HFGate(2048)

        self.x4_AMCFM = AMCFM(2048, channels, d1=1, d2=2, d3=3)
        self.x3_AMCFM = AMCFM(1024, channels, d1=1, d2=2, d3=3)
        self.x2_AMCFM = AMCFM(512,  channels, d1=1, d2=3, d3=6)

        self.att_x4 = EDGA_v32(channels, reduction=8, use_mccpb=use_mccpb,
                               edge_dist_mode=edge_dist_mode)
        self.att_x3 = EDGA_v32(channels, reduction=8, use_mccpb=use_mccpb,
                               edge_dist_mode=edge_dist_mode)
        self.att_x2 = EDGA_v32(channels, reduction=8, use_mccpb=use_mccpb,
                               edge_dist_mode=edge_dist_mode)

        self.fusion_x3_x4 = FeatureFusionBlock(channels, channels, channels)
        self.fusion_x2_x3 = FeatureFusionBlock(channels, channels, channels)

        self.x4_decoder = DecoderSimple(channels)
        self.x3_decoder = DecoderSimple(channels)
        self.x2_decoder = DecoderSimple(channels)

        if use_dual_heads:
            mid = 64
            self.fg_head = nn.Conv2d(mid, 1, 1)
            self.bg_head = nn.Conv2d(mid, 1, 1)
            nn.init.zeros_(self.fg_head.weight); nn.init.zeros_(self.fg_head.bias)
            nn.init.zeros_(self.bg_head.weight); nn.init.zeros_(self.bg_head.bias)

        if use_boundary_contrast:
            self.bc_loss_module = BoundaryContrastLoss(
                channels=(channels, channels, channels),
                margin=bc_margin,
                push_weight=bc_push_weight,
            )

        self.loss_fn = _main_loss_fn
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear',
                                                  align_corners=False)

    def set_epoch(self, epoch: int):
        self.current_epoch.fill_(int(epoch))

    def _epoch(self):
        return int(self.current_epoch.item())

    def _lambda_aux(self):
        if not self.use_dual_heads:
            return 0.0
        sched = [0.00, 0.15, 0.30, 0.30, 0.30, 0.30, 0.28, 0.22, 0.12, 0.05]
        return sched[max(0, min(self._epoch(), len(sched) - 1))]

    def _lambda_bc(self):
        if not self.use_boundary_contrast:
            return 0.0
        sched = [0.00, 0.05, 0.10, 0.20, 0.20, 0.20, 0.20, 0.10, 0.10, 0.10]
        return sched[max(0, min(self._epoch(), len(sched) - 1))]

    def _lambda_flip(self):
        """Schedule for flip-consistency loss weight."""
        if not self.use_flip_consistency:
            return 0.0
        # Linear ramp 0 -> max over first 3 epochs, then hold
        peak = self.flip_consistency_max
        sched = [0.00, peak * 0.33, peak * 0.66, peak, peak, peak,
                 peak, peak, peak * 0.8, peak * 0.5]
        return sched[max(0, min(self._epoch(), len(sched) - 1))]

    # ====================================================================
    # Core forward: takes image/mask/gt (already concatenated if flip-aug)
    # Returns the four sigmoid'd predictions and decoder features.
    # ====================================================================
    def _forward_core(self, x_in, mask):
        x = self.resnet.conv1(x_in) + self.mask_conv(2.0 * mask - 1.0)
        x = self.resnet.bn1(x); x = self.resnet.relu(x); x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        if self.use_hf_gate:
            x4 = self.hf_gate(x4)

        x2 = self.x2_AMCFM(x2)
        x3 = self.x3_AMCFM(x3)
        x4 = self.x4_AMCFM(x4)

        x4 = self.att_x4(x4, mask)
        need_feat = self.use_dual_heads or self.use_boundary_contrast
        if need_feat:
            out4, f4 = self.x4_decoder(x4, return_feat=True)
        else:
            out4 = self.x4_decoder(x4); f4 = None

        x3 = self.fusion_x3_x4(x4, x3)
        x3 = self.att_x3(x3, torch.sigmoid(out4))
        if need_feat:
            out3, f3 = self.x3_decoder(x3, return_feat=True)
        else:
            out3 = self.x3_decoder(x3); f3 = None

        x2 = self.fusion_x2_x3(x3, x2)
        x2 = self.att_x2(x2, torch.sigmoid(out3))
        if need_feat:
            out2, f2 = self.x2_decoder(x2, return_feat=True)
        else:
            out2 = self.x2_decoder(x2); f2 = None

        return {
            'out2': out2, 'out3': out3, 'out4': out4,
            'f2': f2, 'feat_x4': x4, 'feat_x3': x3, 'feat_x2': x2,
        }

    # ====================================================================
    # Public forward — handles flip-doubling during training
    # ====================================================================
    def forward(self, sample):
        x_in = sample['image']
        mask = sample['mask']
        y = sample.get('gt', None)
        base_size = x_in.shape[-2:]

        # ---- INFERENCE PATH: identical to v3.2 ----
        if y is None or not self.use_flip_consistency:
            return self._forward_no_flip(x_in, mask, y, base_size)

        # ---- TRAINING PATH WITH FLIP CONSISTENCY ----
        B = x_in.shape[0]

        # Build augmented batch: [orig, hflip, vflip]. Each is (B, C, H, W).
        x_h = torch.flip(x_in, dims=[3])
        x_v = torch.flip(x_in, dims=[2])
        m_h = torch.flip(mask, dims=[3])
        m_v = torch.flip(mask, dims=[2])
        y_h = torch.flip(y, dims=[3])
        y_v = torch.flip(y, dims=[2])

        x_all = torch.cat([x_in, x_h, x_v], dim=0)   # (3B, C, H, W)
        m_all = torch.cat([mask, m_h, m_v], dim=0)
        y_all = torch.cat([y, y_h, y_v], dim=0)

        # Single forward pass on the tripled batch
        core = self._forward_core(x_all, m_all)

        out2_all = self.res(core['out2'], base_size)
        out3_all = self.res(core['out3'], base_size)
        out4_all = self.res(core['out4'], base_size)

        # ---- Main deep-supervised loss (averaged over all 3 views) ----
        # Dual-heads pixel weighting computed on the original view only,
        # then broadcast — simpler and avoids weighting the flipped views
        # by their own flipped uncertainty.
        if self.use_dual_heads:
            fg_logit_all = self.fg_head(core['f2'])
            bg_logit_all = self.bg_head(core['f2'])
            fg_up_all = self.res(fg_logit_all, base_size)
            bg_up_all = self.res(bg_logit_all, base_size)
            with torch.no_grad():
                u = torch.sigmoid(fg_up_all) * torch.sigmoid(bg_up_all)
                u_min = u.amin(dim=(2, 3), keepdim=True)
                u_max = u.amax(dim=(2, 3), keepdim=True)
                u_norm = (u - u_min) / (u_max - u_min).clamp(min=1e-6)
                w_pix = 1.0 + self.beta_u * u_norm
        else:
            fg_up_all = out2_all * 0.0
            bg_up_all = out2_all * 0.0
            w_pix = None

        loss4 = self.loss_fn(out4_all, y_all)
        loss3 = self.loss_fn(out3_all, y_all)
        loss2 = self.loss_fn(out2_all, y_all, pixel_weight=w_pix) \
                if self.use_dual_heads else self.loss_fn(out2_all, y_all)
        main_loss = loss2 + loss3 + loss4

        # ---- Aux dual-head losses ----
        lam_aux = self._lambda_aux()
        if self.use_dual_heads and lam_aux > 0:
            loss_fg = self.loss_fn(fg_up_all, y_all)
            loss_bg = self.loss_fn(bg_up_all, 1.0 - y_all)
            fg_p = torch.sigmoid(fg_up_all); bg_p = torch.sigmoid(bg_up_all)
            loss_cons_fg_bg = ((fg_p + bg_p - 1.0) ** 2).mean()
            aux_loss = lam_aux * (loss_fg + loss_bg + self.gamma_cons * loss_cons_fg_bg)
        else:
            aux_loss = out2_all.new_zeros(())

        # ---- Boundary contrast (on original view only to save memory) ----
        lam_bc = self._lambda_bc()
        if self.use_boundary_contrast:
            in_warmup = (int(self.bc_loss_module.warmup_counter.item())
                         < self.bc_loss_module.ema_warmup_iters)
            if lam_bc > 0 or in_warmup:
                feat_x4_orig = core['feat_x4'][:B]
                feat_x3_orig = core['feat_x3'][:B]
                feat_x2_orig = core['feat_x2'][:B]
                bc_val = self.bc_loss_module(
                    [feat_x4_orig, feat_x3_orig, feat_x2_orig], y)
                bc_loss = lam_bc * bc_val
            else:
                bc_loss = out2_all.new_zeros(())
        else:
            bc_loss = out2_all.new_zeros(())

        # ====================================================================
        # NEW: flip-consistency loss
        # ====================================================================
        # Split the tripled output back into the three views.
        # sigmoid space because that's what TTA averages at inference.
        pred_orig  = torch.sigmoid(out2_all[:B])
        pred_h     = torch.sigmoid(out2_all[B:2*B])
        pred_v     = torch.sigmoid(out2_all[2*B:3*B])

        # Un-flip the flipped predictions, then compare to original.
        pred_h_unflipped = torch.flip(pred_h, dims=[3])
        pred_v_unflipped = torch.flip(pred_v, dims=[2])

        consistency_h = F.mse_loss(pred_h_unflipped, pred_orig)
        consistency_v = F.mse_loss(pred_v_unflipped, pred_orig)
        consistency = 0.5 * (consistency_h + consistency_v)

        lam_flip = self._lambda_flip()
        flip_loss = lam_flip * consistency

        # ---- Total loss ----
        loss = main_loss + aux_loss + bc_loss + flip_loss
        debug_loss = [loss2.item(), loss3.item(), loss4.item()]

        # Return predictions for the original view only (matches v3.2 contract)
        return {
            'pred': out2_all[:B],
            'loss': loss,
            'debug': [out4_all[:B], out3_all[:B]],
            'debug_loss': debug_loss,
            'fg_pred': fg_up_all[:B],
            'bg_pred': bg_up_all[:B],
            'consistency_loss': consistency.item(),
            'flip_lambda': lam_flip,
        }

    # ====================================================================
    # Inference path / training path with flip consistency disabled
    # (identical to v3.2 forward, refactored for clarity)
    # ====================================================================
    def _forward_no_flip(self, x_in, mask, y, base_size):
        core = self._forward_core(x_in, mask)

        out2_up = self.res(core['out2'], base_size)
        out3_up = self.res(core['out3'], base_size)
        out4_up = self.res(core['out4'], base_size)

        if self.use_dual_heads:
            fg_logit = self.fg_head(core['f2'])
            bg_logit = self.bg_head(core['f2'])
            fg_up = self.res(fg_logit, base_size)
            bg_up = self.res(bg_logit, base_size)
        else:
            fg_up = out2_up * 0.0
            bg_up = out2_up * 0.0

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

            loss4 = self.loss_fn(out4_up, y)
            loss3 = self.loss_fn(out3_up, y)
            loss2 = self.loss_fn(out2_up, y, pixel_weight=w_pix) \
                    if self.use_dual_heads else self.loss_fn(out2_up, y)
            main_loss = loss2 + loss3 + loss4

            lam_aux = self._lambda_aux()
            if self.use_dual_heads and lam_aux > 0:
                loss_fg = self.loss_fn(fg_up, y)
                loss_bg = self.loss_fn(bg_up, 1.0 - y)
                fg_p = torch.sigmoid(fg_up); bg_p = torch.sigmoid(bg_up)
                loss_cons_fg_bg = ((fg_p + bg_p - 1.0) ** 2).mean()
                aux_loss = lam_aux * (loss_fg + loss_bg + self.gamma_cons * loss_cons_fg_bg)
            else:
                aux_loss = out2_up.new_zeros(())

            lam_bc = self._lambda_bc()
            if self.use_boundary_contrast:
                in_warmup = (int(self.bc_loss_module.warmup_counter.item())
                             < self.bc_loss_module.ema_warmup_iters)
                if lam_bc > 0 or in_warmup:
                    bc_val = self.bc_loss_module(
                        [core['feat_x4'], core['feat_x3'], core['feat_x2']], y)
                    bc_loss = lam_bc * bc_val
                else:
                    bc_loss = out2_up.new_zeros(())
            else:
                bc_loss = out2_up.new_zeros(())

            loss = main_loss + aux_loss + bc_loss
            debug_loss = [loss2.item(), loss3.item(), loss4.item()]
        else:
            loss = torch.tensor(0.0, device=out2_up.device)
            debug_loss = []

        return {
            'pred': out2_up,
            'loss': loss,
            'debug': [out4_up, out3_up],
            'debug_loss': debug_loss,
            'fg_pred': fg_up,
            'bg_pred': bg_up,
        }
