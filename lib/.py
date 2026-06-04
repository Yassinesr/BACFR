import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Literal
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

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, d1= 1, d2= 2, d3= 3):
        super(ASPP, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True))


    def forward(self, x):
        size = x.shape[2:]
        x1 = self.conv1(x)
        return x1


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


class UACAPatchNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channels=256, output_stride=16, pretrained=True):
        super(UACAPatchNet, self).__init__()

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

        self.x4_ASPP = ASPP(2048, 256,d1=1,d2=2,d3=3)
        self.x3_ASPP = ASPP(1024, 256,d1=1,d2=2,d3=3)
        self.x2_ASPP = ASPP(512, 256,d1=1,d2=3,d3=6)

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
        x2 = self.x2_ASPP(x2)  # 256*32*32
        x3 = self.x3_ASPP(x3)  # 256*16*16
        x4 = self.x4_ASPP(x4)  # 256*16*16

        # mask 采样到16*16
        x4 = self.att_x4(x4, mask)  # 256*16*16
        out4 = self.x4_decoder(x4)  # 直接卷积的结果16*16，后面应该+sigmoid

        x3 = self.fusion_x3_x4(x4, x3)
        x3 = self.att_x3(x3, (torch.sigmoid(out4) > 0.5).float())
        out3 = self.x3_decoder(x3)  # 直接卷积的结果，后面应该+sigmoid

        x2 = self.fusion_x2_x3(x3, x2)
        x2 = self.att_x2(x2, (torch.sigmoid(out3) > 0.5).float())
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

'''
11.22修改-4
为每一个卷积增加BN和RELU

11.22修改-5
修改backbone为swim-s,主干变大，batchsize设为24

11.24修改6
在修改4的基础上，缩小epoch数，看看是不是还是第一个epoch保存的权重在测试集上最好

11.24修改7
合并两个训练集（0.78和0.95，共80000），训练20个epoch

11.27修改8
准备大改模型
(1)输入卷积层加深，让图像和掩码更好融合
(2)ASPP的膨胀率太大了,根据特征图的大小动态的调整一下，16*16-2,3,4  32*32--2,4,8
(3)更改注意力机制

11.28修改9
注意力机制还是改回去，看看膨胀率，估计提升不大

11.28修改10
不直接将图像和mask从4卷到3，分两步，新增——mask_conv

11.30最终修改
平衡所有模型

12.5-15
在14的基础上改进ASPP
增加通道注意力机制以及动态膨胀率

'''
