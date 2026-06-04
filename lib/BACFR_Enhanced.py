import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

from .optim.losses import dice_bce_loss
from .backbones.Res2Net_v1b import res2net50_v1b_26w_4s


# ========================================
# ORIGINAL AMCFM — unchanged from baseline BACFR
# ========================================

class GlobalChannelAttentionCollapse(nn.Module):
    def __init__(self, channels_per_branch, branches=5, reduction=8):
        super().__init__()
        self.C = channels_per_branch
        self.branches = branches
        self.total_ch = branches * channels_per_branch

        hidden = self.total_ch // reduction
        self.mlp = nn.Sequential(
            nn.Conv2d(self.total_ch, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, self.total_ch, kernel_size=1, bias=True),
        )

    def forward(self, x):
        B, BC, H, W = x.shape
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        attn_logits = self.mlp(avg).view(B, self.total_ch) + self.mlp(mx).view(B, self.total_ch)
        attn = torch.sigmoid(attn_logits).view(B, self.total_ch, 1, 1)
        x = x * attn
        return x


class AMCFM(nn.Module):
    def __init__(self, in_ch, out_ch, d1=1, d2=2, d3=3):
        super(AMCFM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=d1, dilation=d1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=d2, dilation=d2, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=d3, dilation=d3, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

        self.channel_attn = GlobalChannelAttentionCollapse(
            channels_per_branch=out_ch, branches=5)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch * 2), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[2:]
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x_pool = self.global_pool(x)
        x_pool = F.interpolate(x_pool, size=size, mode='bilinear', align_corners=False)
        x_cat = torch.cat([x1, x2, x3, x4, x_pool], dim=1)
        x_cat = self.channel_attn(x_cat)
        return self.conv_cat(x_cat)


# ========================================
# MGDLKA — Mask-Guided Deformable Large Kernel Attention
#
# Bottleneck design for memory efficiency:
#   reduce(256→64) → deform_conv_1(64,full) → deform_conv_2(64,full) → expand(64→256)
#
# Key differences from the OOM version:
#   - FULL deformable convs at 64ch instead of DW at 256ch
#   - 3×3 kernels instead of 5×5+7×7
#   - Cross-channel interaction DURING spatial processing (groups=1)
#   - Two sequential 3×3 deformable convs = effective 5×5 RF with
#     content-dependent offsets at each level
#
# Why FULL at bottleneck > DW at full width:
#   out[c,i,j] = Σ_{c_in} Σ_{k} W[c,c_in,k] × x[c_in, i+dy_k, j+dx_k]
#   Each output channel mixes ALL 64 input channels at 9 adaptively-
#   sampled positions. This is true cross-channel spatial mixing —
#   the property that made EDGA's Q·K·V attention effective and that
#   all our DW-based attempts lacked.
# ========================================

class MGDLKA(nn.Module):
    """Mask-Guided Deformable Large Kernel Attention.

    Bottleneck design with FULL (not depth-wise) deformable convolutions
    for cross-channel spatial mixing at reduced memory cost.
    """

    def __init__(self, in_channels, bottleneck=64):
        super().__init__()
        bn = bottleneck

        # Reduce to bottleneck
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, bn, 1, bias=False),
            nn.BatchNorm2d(bn),
            nn.ReLU(inplace=True))

        # Mask-guided offset prediction for first deformable conv
        # 2 × 3 × 3 = 18 offset channels for a 3×3 kernel
        self.offset_conv1 = nn.Sequential(
            nn.Conv2d(bn + 1, bn // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(bn // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn // 2, 2 * 3 * 3, 3, padding=1))

        # First deformable conv: FULL cross-channel, 64→64
        self.deform1 = DeformConv2d(bn, bn, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(bn)

        # Offset prediction for second deformable conv
        # Uses output of first deform + mask for deeper spatial reasoning
        self.offset_conv2 = nn.Sequential(
            nn.Conv2d(bn + 1, bn // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(bn // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn // 2, 2 * 3 * 3, 3, padding=1))

        # Second deformable conv: FULL cross-channel, 64→64
        self.deform2 = DeformConv2d(bn, bn, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(bn)

        # Expand back to full channels
        self.expand = nn.Sequential(
            nn.Conv2d(bn, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels))

        # Output merge (EDGA-compatible: concat[refined, x] → conv)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))

        # Zero-init offsets → deformable convs start as standard 3×3 convs
        nn.init.zeros_(self.offset_conv1[-1].weight)
        nn.init.zeros_(self.offset_conv1[-1].bias)
        nn.init.zeros_(self.offset_conv2[-1].weight)
        nn.init.zeros_(self.offset_conv2[-1].bias)

    def forward(self, x, mask):
        """
        Args:
            x:    features (B, 256, H, W)
            mask: prediction probability [0,1] (B, 1, H, W)
        """
        mask = F.interpolate(mask, x.shape[2:], mode='bilinear', align_corners=False)

        # Reduce to bottleneck
        feat = self.reduce(x)

        # First deformable conv with mask-guided offsets
        off1 = self.offset_conv1(torch.cat([feat, mask], dim=1))
        feat = F.relu(self.bn1(self.deform1(feat, off1)), inplace=True)

        # Second deformable conv: offsets from updated features + mask
        off2 = self.offset_conv2(torch.cat([feat, mask], dim=1))
        feat = F.relu(self.bn2(self.deform2(feat, off2)), inplace=True)

        # Expand back
        feat = self.expand(feat)

        # Merge with original features
        out = self.out_conv(torch.cat([feat, x], dim=1))
        return out


# ========================================
# SUPPORTING MODULES — unchanged from baseline
# ========================================

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
        fuse = torch.cat([feat_prev, feat_curr], dim=1)
        out = self.fusion(fuse) + feat_curr
        return out


class DecoderSimple(nn.Module):
    def __init__(self, in_channels, mid_channels=64):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1))

    def forward(self, x):
        return self.decode(x)


# ========================================
# MAIN MODEL
# ========================================

class BACFR_Enhanced(nn.Module):
    """
    BACFR with original AMCFM context + MGDLKA decoder.

    Context: AMCFM (unchanged from baseline)
    Decoder: MGDLKA — two sequential FULL deformable 3×3 convolutions
             at 64ch bottleneck with mask-guided offset prediction.
    """

    def __init__(self, channels=256, output_stride=16, pretrained=True):
        super(BACFR_Enhanced, self).__init__()

        self.mask_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False))

        self.resnet = res2net50_v1b_26w_4s(
            pretrained=pretrained, output_stride=output_stride)

        # ORIGINAL context encoding
        self.x4_AMCFM = AMCFM(2048, channels, d1=1, d2=2, d3=3)
        self.x3_AMCFM = AMCFM(1024, channels, d1=1, d2=2, d3=3)
        self.x2_AMCFM = AMCFM(512, channels, d1=1, d2=3, d3=6)

        # NOVEL decoder: MGDLKA replaces EDGA
        self.att_x4 = MGDLKA(channels, bottleneck=64)
        self.att_x3 = MGDLKA(channels, bottleneck=64)
        self.att_x2 = MGDLKA(channels, bottleneck=64)

        # Decoders — unchanged from baseline
        self.x4_decoder = DecoderSimple(channels)
        self.x3_decoder = DecoderSimple(channels)
        self.x2_decoder = DecoderSimple(channels)

        # Feature fusion — unchanged from baseline
        self.fusion_x3_x4 = FeatureFusionBlock(channels, channels, channels)
        self.fusion_x2_x3 = FeatureFusionBlock(channels, channels, channels)

        self.loss_fn = dice_bce_loss
        self.res = lambda x, size: F.interpolate(
            x, size=size, mode='bilinear', align_corners=False)

    def forward(self, sample):
        x = sample['image']
        mask = sample['mask']
        base_size = x.shape[-2:]

        if 'gt' in sample.keys():
            y = sample['gt']
        else:
            y = None

        # Backbone + mask fusion — IDENTICAL to baseline
        x = self.resnet.conv1(x) + self.mask_conv((2 * mask - 1))
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        # Context encoding — ORIGINAL AMCFM
        x2 = self.x2_AMCFM(x2)
        x3 = self.x3_AMCFM(x3)
        x4 = self.x4_AMCFM(x4)

        # Decoder — MGDLKA replaces EDGA, same progressive structure
        x4 = self.att_x4(x4, mask)
        out4 = self.x4_decoder(x4)

        x3 = self.fusion_x3_x4(x4, x3)
        x3 = self.att_x3(x3, torch.sigmoid(out4))
        out3 = self.x3_decoder(x3)

        x2 = self.fusion_x2_x3(x3, x2)
        x2 = self.att_x2(x2, torch.sigmoid(out3))
        out2 = self.x2_decoder(x2)

        out4 = self.res(out4, base_size)
        out3 = self.res(out3, base_size)
        out2 = self.res(out2, base_size)

        if y is not None:
            loss4 = self.loss_fn(out4, y)
            loss3 = self.loss_fn(out3, y)
            loss2 = self.loss_fn(out2, y)
            loss = loss2 + loss3 + loss4

            debug_loss = [loss2.item(), loss3.item(), loss4.item()]
        else:
            loss = torch.tensor(0., device=out2.device)
            debug_loss = []

        return {'pred': out2, 'loss': loss, 'debug': [out4, out3],
                'debug_loss': debug_loss}
