"""BACFR_Enhanced v3.2 — patch-based polyp segmentation refinement.

CRITICAL FIX from v3.1: EDGA now matches the original baseline EXACTLY.
  - Single-head attention via torch.bmm (not multi-head einsum)
  - Hidden dim = in_channels // reduction (32 for channels=256, reduction=8)
  - edge_distance_weighting: weighted_x = x * (1 + dist), then K/V project
    FROM weighted_x, Q projects from unweighted x
  - Edge detection inside torch.no_grad()

Optional additions on top (all zero-init / identity-at-t=0):
  - use_mccpb: MC-CPB additive bias on attention logits
  - use_dual_heads: fg/bg aux heads with product-uncertainty BCE weight
  - use_boundary_contrast: multi-scale prototype BC loss
  - use_hf_gate: zero-init HF gate on backbone layer4

Recommended configs:
  Config B (sanity check - MUST reproduce baseline):
      all flags False. This makes the model behave identically to BACFR.

  Config A (EDGA + additive MC-CPB only):
      use_mccpb=True, others False.

  Config C (full stack):
      all True.

Interface unchanged:
    out = model({'image': ..., 'mask': ..., 'gt': ...})
    -> {'pred', 'loss', 'debug', 'debug_loss', 'fg_pred', 'bg_pred'}
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.Res2Net_v1b import res2net50_v1b_26w_4s

try:
    from .optim.losses import dice_bce_loss as _repo_dice_bce_loss
except Exception:
    _repo_dice_bce_loss = None


# ============================================================
# Loss helpers
# ============================================================
def _dice_bce(pred_logit, target, pixel_weight=None):
    pred = torch.sigmoid(pred_logit)
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
    dice_loss = (1.0 - (2.0 * inter + 1.0) / denom).mean()
    bce = F.binary_cross_entropy_with_logits(pred_logit, target, reduction='none')
    if pixel_weight is not None:
        bce = bce * pixel_weight
    bce_loss = bce.mean()
    return dice_loss + bce_loss


def _main_loss_fn(pred_logit, target, pixel_weight=None):
    if pixel_weight is None and _repo_dice_bce_loss is not None:
        return _repo_dice_bce_loss(pred_logit, target)
    return _dice_bce(pred_logit, target, pixel_weight=pixel_weight)


# ============================================================
# UNCHANGED from baseline (verbatim)
# ============================================================
class GlobalChannelAttentionCollapse(nn.Module):
    def __init__(self, channels_per_branch, branches=5, reduction=8):
        super().__init__()
        self.C = channels_per_branch
        self.branches = branches
        self.total_ch = branches * channels_per_branch
        hidden = self.total_ch // reduction
        self.mlp = nn.Sequential(
            nn.Conv2d(self.total_ch, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, self.total_ch, 1, bias=True),
        )

    def forward(self, x):
        B, BC, H, W = x.shape
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        logits = self.mlp(avg).view(B, self.total_ch) + self.mlp(mx).view(B, self.total_ch)
        attn = torch.sigmoid(logits).view(B, self.total_ch, 1, 1)
        return x * attn


class AMCFM(nn.Module):
    def __init__(self, in_ch, out_ch, d1=1, d2=2, d3=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=d1, dilation=d1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=d2, dilation=d2, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=d3, dilation=d3, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.channel_attn = GlobalChannelAttentionCollapse(
            channels_per_branch=out_ch, branches=5)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch * 2), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch * 2, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[2:]
        x1 = self.conv1(x); x2 = self.conv2(x); x3 = self.conv3(x); x4 = self.conv4(x)
        xp = F.interpolate(self.global_pool(x), size=size, mode='bilinear', align_corners=False)
        xc = torch.cat([x1, x2, x3, x4, xp], dim=1)
        xc = self.channel_attn(xc)
        return self.conv_cat(xc)


class FeatureFusionBlock(nn.Module):
    def __init__(self, in_ch_low, in_ch_high, out_ch):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_ch_low * 2, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, feat_prev, feat_curr):
        if feat_prev.shape[-2:] != feat_curr.shape[-2:]:
            feat_prev = F.interpolate(feat_prev, size=feat_curr.shape[-2:],
                                      mode='bilinear', align_corners=False)
        return self.fusion(torch.cat([feat_prev, feat_curr], dim=1)) + feat_curr


class DecoderSimple(nn.Module):
    """Same as baseline but optionally returns the pre-head feature
    for aux head attachment. Baseline behavior when return_feat=False."""
    def __init__(self, in_channels, mid_channels=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(mid_channels, 1, 1)

    def forward(self, x, return_feat=False):
        f = self.body(x)
        y = self.head(f)
        return (y, f) if return_feat else y


# ============================================================
# MC-CPB: additive attention bias (zero-init)
# ============================================================
class MaskConditionedCPB(nn.Module):
    def __init__(self, meta_dim=32, num_dilations=4):
        super().__init__()
        self.meta_dim = meta_dim

        dils = [1, 2, 4, 8][:num_dilations]
        self.dil_convs = nn.ModuleList([
            nn.Conv2d(1, 1, 3, padding=d, dilation=d, bias=False) for d in dils
        ])
        for c in self.dil_convs:
            nn.init.constant_(c.weight, 1.0 / 9.0)
        self.dil_alphas = nn.Parameter(torch.ones(len(dils)) / len(dils))

        def _mlp():
            m = nn.Sequential(
                nn.Linear(2, meta_dim),
                nn.ReLU(inplace=True),
                nn.Linear(meta_dim, 1),
            )
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)
            return m

        self.pos_mlp = _mlp()
        self.src_mlp = _mlp()
        self.dst_mlp = _mlp()

        self.gate = nn.Parameter(torch.ones(1))
        self._pos_cache = {}

    def _build_coords(self, H, W, device, dtype):
        key = (H, W)
        if key in self._pos_cache:
            c = self._pos_cache[key]
            if c.device == device and c.dtype == dtype:
                return c
        ys = torch.arange(H, device=device, dtype=dtype)
        xs = torch.arange(W, device=device, dtype=dtype)
        try:
            gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        except TypeError:
            gy, gx = torch.meshgrid(ys, xs)
        gy = gy.reshape(-1); gx = gx.reshape(-1)
        dy = gy[:, None] - gy[None, :]
        dx = gx[:, None] - gx[None, :]
        norm = math.log(max(H, W) + 1.0)
        dy_log = torch.sign(dy) * torch.log1p(dy.abs()) / norm
        dx_log = torch.sign(dx) * torch.log1p(dx.abs()) / norm
        coords = torch.stack([dy_log, dx_log], dim=-1)
        self._pos_cache[key] = coords
        return coords

    def forward(self, mask, H, W):
        """Returns bias (B, N, N) for single-head attention."""
        B = mask.shape[0]
        N = H * W
        device, dtype = mask.device, mask.dtype

        s = 4.0 * mask * (1.0 - mask)
        alphas = torch.softmax(self.dil_alphas, dim=0)
        D_raw = 0.0
        for a, c in zip(alphas, self.dil_convs):
            D_raw = D_raw + a * c(s)
        D = 1.0 - torch.sigmoid(D_raw)

        m_flat = mask.reshape(B, N, 1)
        D_flat = D.reshape(B, N, 1)
        mD = torch.cat([m_flat, D_flat], dim=-1)

        src = self.src_mlp(mD).squeeze(-1)   # (B, N)
        dst = self.dst_mlp(mD).squeeze(-1)   # (B, N)

        coords = self._build_coords(H, W, device, dtype)    # (N, N, 2)
        pos = self.pos_mlp(coords).squeeze(-1)              # (N, N)

        # Broadcast to (B, N, N): pos (1,N,N) + dst (B,N,1) + src (B,1,N)
        bias = pos.unsqueeze(0) + dst.unsqueeze(2) + src.unsqueeze(1)
        bias = bias * self.gate
        return bias


# ============================================================
# EDGA — MATCHES BASELINE EXACTLY + optional MC-CPB additive bias
# ============================================================
class EDGA_v32(nn.Module):
    """Faithful reproduction of baseline EdgeDistanceGuidedAttention.

    Matches original:
      - Single-head via torch.bmm, hidden = in_channels // reduction
      - Edge detection inside torch.no_grad()
      - weighted_x = x * (1 + dist), K and V project FROM weighted_x,
        Q projects from unweighted x
      - softmax(attn / sqrt(hidden))
      - out_conv(cat([attended, x]))

    Extension:
      - Optional additive MC-CPB bias on attention logits (zero-init).
      - Edge-distance can use the original batched cdist (exact) or
        a fast dilation-ladder approximation (for memory/speed).
    """
    def __init__(self, in_channels, reduction=8,
                 use_mccpb=False, edge_dist_mode='cdist'):
        super().__init__()
        assert edge_dist_mode in ('cdist', 'ladder')
        self.in_channels = in_channels
        self.hidden = in_channels // reduction
        self.use_mccpb = use_mccpb
        self.edge_dist_mode = edge_dist_mode

        self.query_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden, 1, bias=False),
            nn.BatchNorm2d(self.hidden), nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden, self.hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden), nn.ReLU(inplace=True),
        )
        self.key_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden, 1, bias=False),
            nn.BatchNorm2d(self.hidden), nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden, self.hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden), nn.ReLU(inplace=True),
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
        )

        if use_mccpb:
            self.mccpb = MaskConditionedCPB()

    @torch.no_grad()
    def _edge_distance_cdist(self, mask, eps=1e-6):
        """Verbatim reproduction of baseline edge_distance_map_torch."""
        B, _, H, W = mask.shape
        device = mask.device

        ys = torch.arange(0, H, device=device, dtype=torch.float32)
        xs = torch.arange(0, W, device=device, dtype=torch.float32)
        try:
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        except TypeError:
            yy, xx = torch.meshgrid(ys, xs)
        coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)  # (HW, 2)

        dist_maps = []
        for b in range(B):
            mb = mask[b, 0]
            edge_pts = (mb > 0.5).nonzero(as_tuple=False)
            if edge_pts.numel() == 0:
                dist_maps.append(torch.zeros(H, W, device=device))
                continue
            coords_b = coords.unsqueeze(0)
            edge_b = edge_pts.unsqueeze(0).float()
            d = torch.cdist(coords_b, edge_b)[0]
            min_d = d.min(dim=1)[0].view(H, W)
            max_d = min_d.max().clamp_min(eps)
            norm = min_d / max_d
            weight = 1 - norm
            dist_maps.append(weight)
        return torch.stack(dist_maps, dim=0).unsqueeze(1)

    @torch.no_grad()
    def _edge_distance_ladder(self, mask, max_dist=16, eps=1e-6):
        """Fast approximation of the cdist version using dilation ladder."""
        B, _, H, W = mask.shape
        edge = mask  # already binary in forward()
        dist = torch.full_like(edge, float(max_dist + 1))
        for d in [1, 2, 3, 4, 6, 8, 12, 16]:
            if d > max_dist:
                break
            k = 2 * d + 1
            dilated = F.max_pool2d(edge, k, stride=1, padding=k // 2)
            d_t = edge.new_tensor(float(d))
            dist = torch.where((dilated > 0) & (dist > d_t),
                               d_t.expand_as(dist), dist)
        dist = torch.where(dist > float(max_dist),
                           edge.new_tensor(float(max_dist)).expand_as(dist),
                           dist)
        max_d = dist.amax(dim=(2, 3), keepdim=True).clamp_min(eps)
        norm = dist / max_d
        return 1.0 - norm

    def forward(self, x, mask):
        B, C, H, W = x.shape
        mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)

        # --- EXACT baseline edge-distance computation (no_grad) ---
        with torch.no_grad():
            edge = torch.abs(mask - F.avg_pool2d(mask, 3, 1, 1))
            edge = (edge > 0.01).float()
            if self.edge_dist_mode == 'cdist':
                dist = self._edge_distance_cdist(edge)
            else:
                dist = self._edge_distance_ladder(edge)

        # --- EXACT baseline weighting ---
        weighted_x = x * (1 + dist)

        # --- Q from x, K/V from weighted_x ---
        q = self.query_conv(x).view(B, -1, H * W)             # (B, hidden, N)
        k = self.key_conv(weighted_x).view(B, -1, H * W)      # (B, hidden, N)
        v = self.value_conv(weighted_x).view(B, -1, H * W)    # (B, C, N)

        # Single-head attention logits via bmm
        attn = torch.bmm(q.permute(0, 2, 1), k)               # (B, N, N)
        attn = attn / (q.shape[1] ** 0.5)                     # baseline scaling

        # Optional additive MC-CPB bias (zero-init => identity at t=0)
        if self.use_mccpb:
            attn = attn + self.mccpb(mask, H, W)

        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        out = self.out_conv(torch.cat([out, x], dim=1))
        return out


# ============================================================
# HF gate (zero-init identity)
# ============================================================
class HFGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate_conv = nn.Conv2d(channels, channels, 1, bias=True)
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.zeros_(self.gate_conv.bias)

    def forward(self, x):
        hp = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        g = torch.sigmoid(self.gate_conv(hp))
        return x * (0.5 + g)


# ============================================================
# Boundary-contrast loss
# ============================================================
class BoundaryContrastLoss(nn.Module):
    def __init__(self, channels=(256, 256, 256), margin=0.3, push_weight=0.5,
                 stage_weights=(1.0, 0.5, 0.25), ema_warmup_iters=50,
                 ema_momentum=0.9, max_negatives=1024):
        super().__init__()
        self.margin = margin
        self.push_weight = push_weight
        self.stage_weights = stage_weights
        self.ema_warmup_iters = ema_warmup_iters
        self.ema_momentum = ema_momentum
        self.max_negatives = max_negatives
        self.prototypes = nn.ParameterList([
            nn.Parameter(torch.randn(c) * 0.1) for c in channels
        ])
        self.register_buffer('warmup_counter', torch.zeros(1, dtype=torch.long))

    @staticmethod
    def _boundary_mask(y, kernel=3):
        pad = kernel // 2
        y_bin = (y > 0.5).float()
        dilated = F.max_pool2d(y_bin, kernel, stride=1, padding=pad)
        eroded = -F.max_pool2d(-y_bin, kernel, stride=1, padding=pad)
        return (dilated - eroded).clamp(0, 1)

    @torch.no_grad()
    def _warmup_update(self, proto, feat_b, first):
        if feat_b.numel() == 0:
            return
        mean_feat = F.normalize(feat_b.mean(dim=0), dim=0)
        if first:
            proto.data.copy_(mean_feat)
        else:
            m = self.ema_momentum
            proto.data.mul_(m).add_(mean_feat, alpha=(1.0 - m))
            proto.data.copy_(F.normalize(proto.data, dim=0))

    def forward(self, feats, y):
        is_warmup = bool(self.warmup_counter.item() < self.ema_warmup_iters)
        total = feats[0].new_zeros(())

        for s, (feat, proto, w) in enumerate(zip(feats, self.prototypes, self.stage_weights)):
            B, C, H, W = feat.shape
            y_s = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
            b_mask = self._boundary_mask(y_s, kernel=3)

            feat_n = F.normalize(feat, dim=1)
            feat_flat = feat_n.permute(0, 2, 3, 1).reshape(-1, C)
            b_idx = b_mask.reshape(-1).bool()
            nb_idx = (~b_idx)
            feat_b = feat_flat[b_idx]
            feat_nb = feat_flat[nb_idx]

            if is_warmup:
                self._warmup_update(proto, feat_b,
                                    first=(self.warmup_counter.item() == 0))
                continue
            if feat_b.numel() == 0:
                continue

            proto_n = F.normalize(proto, dim=0)
            cos_b = (feat_b * proto_n).sum(dim=-1)
            pull = (1.0 - cos_b).mean()

            if feat_nb.shape[0] > self.max_negatives:
                idx = torch.randperm(feat_nb.shape[0], device=feat.device)[:self.max_negatives]
                feat_nb = feat_nb[idx]
            if feat_nb.numel() > 0:
                cos_nb = (feat_nb * proto_n).sum(dim=-1)
                push = F.relu(cos_nb - self.margin).mean()
            else:
                push = feat_b.new_zeros(())

            total = total + w * (pull + self.push_weight * push)

        if is_warmup:
            self.warmup_counter += 1
        return total


# ============================================================
# MAIN MODEL — BACFR_Enhanced_v3_2
# ============================================================
class BACFR_Enhanced_v3(nn.Module):
    """Faithful reproduction of baseline BACFR + toggleable additive modules.

    With all flags False, this model is byte-for-byte equivalent to the
    baseline BACFR (modulo parameter name prefixes).
    """

    def __init__(self, channels=256, output_stride=16, pretrained=True,
                 use_mccpb=False,
                 use_dual_heads=False,
                 use_boundary_contrast=False,
                 use_hf_gate=False,
                 edge_dist_mode='cdist',   # 'cdist' (exact) or 'ladder' (fast)
                 beta_uncertainty=0.5, gamma_consistency=1.0,
                 bc_margin=0.3, bc_push_weight=0.5):
        super().__init__()

        self.use_mccpb = use_mccpb
        self.use_dual_heads = use_dual_heads
        self.use_boundary_contrast = use_boundary_contrast
        self.use_hf_gate = use_hf_gate
        self.beta_u = beta_uncertainty
        self.gamma_cons = gamma_consistency

        self.register_buffer('current_epoch', torch.zeros(1, dtype=torch.long))

        # --- Mask fusion stem (unchanged)
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

        # --- AMCFM context (unchanged)
        self.x4_AMCFM = AMCFM(2048, channels, d1=1, d2=2, d3=3)
        self.x3_AMCFM = AMCFM(1024, channels, d1=1, d2=2, d3=3)
        self.x2_AMCFM = AMCFM(512,  channels, d1=1, d2=3, d3=6)

        # --- EDGA decoder blocks matching baseline exactly
        self.att_x4 = EDGA_v32(channels, reduction=8,
                               use_mccpb=use_mccpb,
                               edge_dist_mode=edge_dist_mode)
        self.att_x3 = EDGA_v32(channels, reduction=8,
                               use_mccpb=use_mccpb,
                               edge_dist_mode=edge_dist_mode)
        self.att_x2 = EDGA_v32(channels, reduction=8,
                               use_mccpb=use_mccpb,
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
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    # ----- Epoch-dependent loss weights -----
    def set_epoch(self, epoch: int):
        self.current_epoch.fill_(int(epoch))

    def _epoch(self):
        return int(self.current_epoch.item())

    def _lambda_aux(self):
        if not self.use_dual_heads:
            return 0.0
        sched = [0.00, 0.15, 0.30, 0.30, 0.30, 0.30, 0.28, 0.22, 0.12, 0.05]
        e = max(0, min(self._epoch(), len(sched) - 1))
        return sched[e]

    def _lambda_bc(self):
        if not self.use_boundary_contrast:
            return 0.0
        sched = [0.00, 0.05, 0.10, 0.20, 0.20, 0.20, 0.20, 0.10, 0.10, 0.10]
        e = max(0, min(self._epoch(), len(sched) - 1))
        return sched[e]

    # ----- Forward (mirrors baseline when all flags False) -----
    def forward(self, sample):
        x_in = sample['image']
        mask = sample['mask']
        y = sample.get('gt', None)
        base_size = x_in.shape[-2:]

        # Backbone stem + mask fusion — identical to baseline
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

        # Progressive decoding — exact baseline flow
        x4 = self.att_x4(x4, mask)

        # When aux heads or BC loss are enabled, we need the pre-head feat
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

        if self.use_dual_heads:
            fg_logit = self.fg_head(f2)
            bg_logit = self.bg_head(f2)
        else:
            fg_logit = out2 * 0.0
            bg_logit = out2 * 0.0

        out4_up = self.res(out4, base_size)
        out3_up = self.res(out3, base_size)
        out2_up = self.res(out2, base_size)
        fg_up = self.res(fg_logit, base_size)
        bg_up = self.res(bg_logit, base_size)

        if y is not None:
            # Product-uncertainty pixel weight (only when dual heads active)
            if self.use_dual_heads:
                with torch.no_grad():
                    u = torch.sigmoid(fg_up) * torch.sigmoid(bg_up)
                    u_min = u.amin(dim=(2, 3), keepdim=True)
                    u_max = u.amax(dim=(2, 3), keepdim=True)
                    denom = (u_max - u_min).clamp(min=1e-6)
                    u_norm = (u - u_min) / denom
                    w_pix = 1.0 + self.beta_u * u_norm
            else:
                w_pix = None

            # Main deep-supervised loss — identical to baseline when w_pix is None
            loss4 = self.loss_fn(out4_up, y)
            loss3 = self.loss_fn(out3_up, y)
            loss2 = self.loss_fn(out2_up, y, pixel_weight=w_pix) \
                    if self.use_dual_heads \
                    else self.loss_fn(out2_up, y)
            main_loss = loss2 + loss3 + loss4

            # Aux fg/bg + consistency
            lam_aux = self._lambda_aux()
            if self.use_dual_heads and lam_aux > 0:
                loss_fg = self.loss_fn(fg_up, y)
                loss_bg = self.loss_fn(bg_up, 1.0 - y)
                fg_p = torch.sigmoid(fg_up); bg_p = torch.sigmoid(bg_up)
                loss_cons = ((fg_p + bg_p - 1.0) ** 2).mean()
                aux_loss = lam_aux * (loss_fg + loss_bg + self.gamma_cons * loss_cons)
            else:
                aux_loss = out2.new_zeros(())

            # Boundary contrast — only when active or during EMA warmup
            lam_bc = self._lambda_bc()
            if self.use_boundary_contrast:
                in_warmup = (int(self.bc_loss_module.warmup_counter.item())
                             < self.bc_loss_module.ema_warmup_iters)
                if lam_bc > 0 or in_warmup:
                    bc_val = self.bc_loss_module([x4, x3, x2], y)
                    bc_loss = lam_bc * bc_val
                else:
                    bc_loss = out2.new_zeros(())
            else:
                bc_loss = out2.new_zeros(())

            loss = main_loss + aux_loss + bc_loss
            debug_loss = [loss2.item(), loss3.item(), loss4.item()]
        else:
            loss = torch.tensor(0.0, device=out2.device)
            debug_loss = []

        return {
            'pred': out2_up,
            'loss': loss,
            'debug': [out4_up, out3_up],
            'debug_loss': debug_loss,
            'fg_pred': fg_up,
            'bg_pred': bg_up,
        }
