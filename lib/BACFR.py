import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from typing import Literal
from .optim.losses import *
from .modules.layers import *
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *

from .backbones.Res2Net_v1b import res2net50_v1b_26w_4s

class FeatureFusionBlock(nn.Module):
    def __init__(self, in_ch_low, in_ch_high, out_ch):
        super().__init__()
        # 融合卷积
        self.fusion = nn.Sequential(
            nn.Conv2d(in_ch_low*2, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_prev, feat_curr):

        # 如果上一层分辨率不同，先上采样
        if feat_prev.shape[-2:] != feat_curr.shape[-2:]:
            feat_prev = F.interpolate(feat_prev, size=feat_curr.shape[-2:], mode='bilinear', align_corners=False)

        fuse = torch.cat([feat_prev, feat_curr], dim=1)
        out = self.fusion(fuse) + feat_curr
        return out

class GlobalChannelAttentionCollapse(nn.Module):
    """
    Treat concatenated (branches * C) channels as a whole, compute channel attention,
    then collapse groups of 'branches' channels into a single channel by weighted sum.

    Input:
        x: [B, branches * C, H, W]
    Output:
        out: [B, C, H, W]

    Args:
        channels_per_branch: C (channels per branch after concat)
        branches: number of branches concatenated (default 5)
        reduction: bottleneck factor for MLP (default 16)
        pool: pooling strategy to form channel descriptors: 'avg', 'avg+max', 'avg+std'
        normalize: 'sigmoid' or 'softmax_branch'
            - 'sigmoid': independent gating for each of the branches*C channels
            - 'softmax_branch': for each original channel index c, normalize the branches' weights
                                 across the branches via softmax (i.e. competition among the branches)
    """
    def __init__(self,
                 channels_per_branch: int,
                 branches: int = 5,
                 reduction: int = 8,
                 ):
        super().__init__()


        self.C = channels_per_branch
        self.branches = branches
        self.total_ch = branches * channels_per_branch


        # MLP: total_ch -> hidden -> total_ch
        hidden = self.total_ch // reduction
        self.mlp = nn.Sequential(
            nn.Conv2d(self.total_ch, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, self.total_ch, kernel_size=1, bias=True),

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, branches*C, H, W]
        returns: [B, branches*C, H, W]
        """
        B, BC, H, W = x.shape
        assert BC == self.total_ch, f"Expected input channels {self.total_ch}, got {BC}"

        # ---- 1) pooled descriptor for each of the total_ch channels ----

        avg = F.adaptive_avg_pool2d(x, 1)
        mx  = F.adaptive_max_pool2d(x, 1)


        # ---- 2) MLP to produce per-channel logits ----
        attn_logits = self.mlp(avg).view(B, self.total_ch)+self.mlp(mx).view(B, self.total_ch)  # [B, total_ch]

        # ---- 3) normalize to get per-channel weights ----
        attn = torch.sigmoid(attn_logits).view(B, self.total_ch, 1, 1)  # [B, total_ch,1,1]

        x = x * attn  # [B, total_ch, H, W]

        return x

class AMCFM(nn.Module):
    def __init__(self, in_ch, out_ch, d1= 1, d2= 2, d3= 3):
        super(AMCFM, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=d1, dilation=d1, bias=False),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=d2, dilation=d2, bias=False),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=d3, dilation=d3, bias=False),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True)
                                    )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.channel_attn = GlobalChannelAttentionCollapse(channels_per_branch=out_ch,branches=5)

        self.conv_cat = nn.Sequential(nn.Conv2d(out_ch * 5, out_ch*2, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(out_ch*2),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(out_ch*2, out_ch, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(out_ch),
                                      nn.ReLU(inplace=True)
                                      )

    def forward(self, x):
        size = x.shape[2:]
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x_pool = self.global_pool(x)
        x_pool = F.interpolate(x_pool, size=size, mode='bilinear', align_corners=False)
        x_cat = torch.cat([x1,x2, x3, x4, x_pool], dim=1)

        x_cat = self.channel_attn(x_cat)

        return self.conv_cat(x_cat)


class EdgeDistanceGuidedAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(EdgeDistanceGuidedAttention, self).__init__()
        hidden_channels = in_channels // reduction

        self.query_conv = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                                        nn.BatchNorm2d(hidden_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(hidden_channels,hidden_channels, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(hidden_channels),
                                        nn.ReLU(inplace=True),
                                        )
        self.key_conv = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                                      nn.BatchNorm2d(hidden_channels),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(hidden_channels),
                                      nn.ReLU(inplace=True),
                                      )
        self.value_conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                        nn.BatchNorm2d(in_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(in_channels),
                                        nn.ReLU(inplace=True),
                                        )


        self.out_conv = nn.Sequential(nn.Conv2d(in_channels *2, in_channels, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(in_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(in_channels),
                                        nn.ReLU(inplace=True),
                                        )

    def edge_distance_map_torch(self, mask, eps=1e-6):
        """
        mask: [B,1,H,W] — binary edge mask (0/1)
        return: [B,1,H,W] — edge=1, far=0
        """
        B, _, H, W = mask.shape
        device = mask.device

        ys = torch.arange(0, H, device=device, dtype=torch.float32)
        xs = torch.arange(0, W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs)
        coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)  # [HW,2]

        dist_maps = []

        for b in range(B):
            mb = mask[b, 0]

            # 提取边缘点坐标
            edge_pts = (mb > 0.5).nonzero(as_tuple=False)

            if edge_pts.numel() == 0:
                dist_maps.append(torch.zeros(H, W, device=device))
                continue

            coords_b = coords.unsqueeze(0)  # [1,HW,2]
            edge_b = edge_pts.unsqueeze(0).float()  # [1,E,2]

            # 欧式距离 transform
            d = torch.cdist(coords_b, edge_b)[0]  # [HW,E]
            min_d = d.min(dim=1)[0].view(H, W)  # [H,W]

            max_d = min_d.max().clamp_min(eps)  # normalize
            norm = min_d / max_d

            weight = 1 - norm  # edge=1 → far=0

            dist_maps.append(weight)

        return torch.stack(dist_maps, dim=0).unsqueeze(1)  # [B,1,H,W]

    def forward(self, x, mask):

        B, C, H, W = x.shape
        mask = F.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=False)
        # ---------- (1) 计算距离图 ----------
        with torch.no_grad():
            edge = torch.abs(mask - F.avg_pool2d(mask, 3, 1, 1))
            edge = (edge > 0.01).float()
            dist = self.edge_distance_map_torch(edge)

        # ---------- (2) 距离加权特征 ----------
        weighted_x = x * (1 + dist)

        # ---------- (3) 生成 Key / Value ----------

        q = self.query_conv(x).view(B, -1, H * W)  # [B, Cq, N]
        k = self.key_conv(weighted_x).view(B, -1, H * W)  # [B, Ck, N]
        v = self.value_conv(weighted_x).view(B, -1, H * W)  # [B, Cv, N]

        attn = torch.bmm(q.permute(0, 2, 1), k)  # [B, N, N]
        attn = F.softmax(attn / (q.shape[1] ** 0.5), dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        out = self.out_conv(torch.cat([out,x],dim=1))
        return out

class DecoderSimple(nn.Module):
    def __init__(self, in_channels, mid_channels=64):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1)  # output logits
        )

    def forward(self, x):
        return self.decode(x)


class BACFR(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channels=256, output_stride=16, pretrained=True):
        super(BACFR, self).__init__()

        self.mask_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )

        self.resnet = res2net50_v1b_26w_4s(pretrained=pretrained, output_stride=output_stride)

        self.x4_AMCFM = AMCFM(2048, 256,d1=1,d2=2,d3=3)
        self.x3_AMCFM = AMCFM(1024, 256,d1=1,d2=2,d3=3)
        self.x2_AMCFM = AMCFM(512, 256,d1=1,d2=3,d3=6)

        self.att_x4 = EdgeDistanceGuidedAttention(256, reduction=8)
        self.att_x3 = EdgeDistanceGuidedAttention(256,reduction=8)
        self.att_x2 = EdgeDistanceGuidedAttention(256,reduction=8)

        self.x4_decoder = DecoderSimple(256)
        self.x3_decoder = DecoderSimple(256)
        self.x2_decoder = DecoderSimple(256)


        self.fusion_x3_x4 = FeatureFusionBlock(256, 256, 256)
        self.fusion_x2_x3 = FeatureFusionBlock(256, 256, 256)


        self.loss_fn = dice_bce_loss

        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, sample):
        x = sample['image']
        mask = sample['mask']

        base_size = x.shape[-2:]

        if 'gt' in sample.keys():
            y = sample['gt']
        else:
            y = None
        #增加处理mask的特征分支,对齐图像的处理分支
        x = self.resnet.conv1(x) + self.mask_conv((2*mask-1))
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)  # 256*64*64
        x2 = self.resnet.layer2(x1)  # 512*32*32
        x3 = self.resnet.layer3(x2)  # 1024*16*16
        x4 = self.resnet.layer4(x3)  # 2048*16*16

        # 金字塔降维，通道维度上与x1特征对齐
        x2 = self.x2_AMCFM(x2)  # 256*32*32
        x3 = self.x3_AMCFM(x3)  # 256*16*16
        x4 = self.x4_AMCFM(x4)  # 256*16*16

        # mask 采样到16*16
        x4 = self.att_x4(x4, mask)  # 256*16*16
        out4 = self.x4_decoder(x4)  # 直接卷积的结果16*16，后面应该+sigmoid

        x3 = self.fusion_x3_x4(x4, x3)
        # CHANGE: Remove threshold (was: (torch.sigmoid(out4) > 0.5).float())
        x3 = self.att_x3(x3, torch.sigmoid(out4))
        out3 = self.x3_decoder(x3)
        
        x2 = self.fusion_x2_x3(x3, x2)
        # CHANGE: Remove threshold (was: (torch.sigmoid(out3) > 0.5).float())
        x2 = self.att_x2(x2, torch.sigmoid(out3))
        out2 = self.x2_decoder(x2)

        out4 = self.res(out4, base_size)
        out3 = self.res(out3, base_size)
        out2 = self.res(out2, base_size)

        if y is not None:
            # self.loss_fn使用F.binary_cross_entropy_with_logits，并且在计算iou之前使用sigmoid
            loss4 = self.loss_fn(out4, y)
            loss3 = self.loss_fn(out3, y)
            loss2 = self.loss_fn(out2, y)
            loss = loss2 + loss3 + loss4

            debug_loss = [loss2.item(), loss3.item(), loss4.item()]
        else:
            loss = torch.tensor(0., device=out2.device)
            debug = []
            debug_loss = []

        return {'pred': out2, 'loss': loss, 'debug': [out4, out3], 'debug_loss': debug_loss}
