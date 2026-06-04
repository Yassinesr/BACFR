import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.modules.layers import *
from utils.utils import *

class reverse_attention(nn.Module):
    def __init__(self, in_channel, channel, depth=3, kernel_size=3):
        super(reverse_attention, self).__init__()
        self.conv_in = conv(in_channel, channel, 1)
        self.conv_mid = nn.ModuleList()
        for i in range(depth):
            self.conv_mid.append(conv(channel, channel, kernel_size))
        self.conv_out = conv(channel, 1, 3 if kernel_size == 3 else 1)

    def forward(self, x, map):
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        rmap = -1 * (torch.sigmoid(map)) + 1
        
        x = rmap.expand(-1, x.shape[1], -1, -1).mul(x)
        x = self.conv_in(x)
        for conv_mid in self.conv_mid:
            x = F.relu(conv_mid(x))
        out = self.conv_out(x)
        out = out + map

        return x, out

class simple_attention(nn.Module):
    def __init__(self, in_channel, channel, depth=3, kernel_size=3):
        super(simple_attention, self).__init__()
        self.conv_in = conv(in_channel, channel, 1)
        self.conv_mid = nn.ModuleList()
        for i in range(depth):
            self.conv_mid.append(conv(channel, channel, kernel_size))
        self.conv_out = conv(channel, 1, 1)

    def forward(self, x, map):
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        amap = torch.sigmoid(map)
        
        x = amap.expand(-1, x.shape[1], -1, -1).mul(x)
        x = self.conv_in(x)
        for conv_mid in self.conv_mid:
            x = F.relu(conv_mid(x))
        out = self.conv_out(x)
        out = out + map

        return x, out

class CA(nn.Module):
    def __init__(self, in_channel, channel):
        super(CA, self).__init__()
        self.channel = channel

        self.conv_query = nn.Sequential(conv(in_channel, channel, 3, relu=True),
                                        conv(channel, channel, 3, relu=True))
        self.conv_key = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                      conv(channel, channel, 1, relu=True))
        self.conv_value = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                        conv(channel, channel, 1, relu=True))

        self.conv_out1 = conv(channel, channel, 3, relu=True)
        self.conv_out2 = conv(in_channel + channel, channel, 3, relu=True)
        self.conv_out3 = conv(channel, channel, 3, relu=True)
        self.conv_out4 = conv(channel, 1, 1)

    def forward(self, x, map):
        b, c, h, w = x.shape
        
        # compute class probability
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        fg = torch.sigmoid(map)
        
        p = fg - .5

        fg = torch.clip(p, 0, 1) # foreground
        bg = torch.clip(-p, 0, 1) # background

        prob = torch.cat([fg, bg], dim=1)

        # reshape feature & prob
        f = x.view(b, h * w, -1)
        prob = prob.view(b, 2, h * w)
        
        # compute context vector
        context = torch.bmm(prob, f).permute(0, 2, 1).unsqueeze(3) # b, 2, c

        # k q v compute
        query = self.conv_query(x).view(b, self.channel, -1).permute(0, 2, 1)
        key = self.conv_key(context).view(b, self.channel, -1)
        value = self.conv_value(context).view(b, self.channel, -1).permute(0, 2, 1)

        # compute similarity map
        sim = torch.bmm(query, key) # b, hw, c x b, c, 2
        sim = (self.channel ** -.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        context = self.conv_out1(context)

        x = torch.cat([x, context], dim=1)
        x = self.conv_out2(x)
        x = self.conv_out3(x)
        out = self.conv_out4(x)
        out = out + map
        return x, out

class UACA(nn.Module):
    def __init__(self, in_channel, channel):
        super(UACA, self).__init__()
        self.channel = channel

        self.conv_query = nn.Sequential(conv(in_channel, channel, 3, relu=True),
                                        conv(channel, channel, 3, relu=True))
        self.conv_key = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                      conv(channel, channel, 1, relu=True))
        self.conv_value = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                        conv(channel, channel, 1, relu=True))

        self.conv_out1 = conv(channel, channel, 3, relu=True)
        self.conv_out2 = conv(in_channel + channel, channel, 3, relu=True)
        self.conv_out3 = conv(channel, channel, 3, relu=True)
        self.conv_out4 = nn.Sequential(conv(channel, channel//2, 3, relu=True),
                                       conv(channel, 1, 1),
                                       )

    def forward(self, x, map):
        b, c, h, w = x.shape
        
        # compute class probability
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        fg = torch.sigmoid(map)
        
        p = fg - .5

        fg = torch.clip(p, 0, 1) # foreground
        bg = torch.clip(-p, 0, 1) # background
        cg = .5 - torch.abs(p) # confusion area

        prob = torch.cat([fg, bg, cg], dim=1)

        # reshape feature & prob
        f = x.view(b, h * w, -1)
        prob = prob.view(b, 3, h * w)
        
        # compute context vector
        context = torch.bmm(prob, f).permute(0, 2, 1).unsqueeze(3) # b, 3, c

        # k q v compute
        query = self.conv_query(x).view(b, self.channel, -1).permute(0, 2, 1)
        key = self.conv_key(context).view(b, self.channel, -1)
        value = self.conv_value(context).view(b, self.channel, -1).permute(0, 2, 1)

        # compute similarity map
        sim = torch.bmm(query, key) # b, hw, c x b, c, 2
        sim = (self.channel ** -.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        context = self.conv_out1(context)

        x = torch.cat([x, context], dim=1)
        x = self.conv_out2(x)  #降一半
        x = self.conv_out3(x) #加深融合
        out = self.conv_out4(x)
        out = out + map
        return x, out


class UACAPatch(nn.Module):
    def __init__(self, in_channel, channel):
        super(UACAPatch, self).__init__()
        self.channel = channel

        self.conv_query = nn.Sequential(conv(in_channel, channel, 3, relu=True),
                                        conv(channel, channel, 3, relu=True))
        self.conv_key = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                      conv(channel, channel, 1, relu=True))
        self.conv_value = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                        conv(channel, channel, 1, relu=True))

        self.conv_out1 = conv(channel, channel, 3, relu=True)
        self.conv_out2 = conv(in_channel + channel, channel, 3, relu=True)
        self.conv_out3 = conv(channel, channel, 3, relu=True)
        self.conv_out4 = conv(channel, 1, 1)

        # 用于边界检测的卷积
        self.edge_conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        # 初始化边界检测核
        kernel = torch.tensor([[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.edge_conv.weight = nn.Parameter(kernel, requires_grad=False)


    def forward(self, x, map):
        b, c, h, w = x.shape

        # compute class probability
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        fg = torch.sigmoid(map)

        # 计算边界区域（不确定区域）
        with torch.no_grad():
            # 创建形态学操作核
            kernel = torch.ones(1, 1, 3, 3).to(map.device)

            # 对前景进行膨胀和腐蚀
            foreground_dilated = map.clone()
            foreground_eroded = map.clone()

            for _ in range(3):
                # 前景膨胀
                foreground_dilated = F.conv2d(foreground_dilated, kernel, padding=1)
                foreground_dilated = (foreground_dilated > 0).float()

                # 前景腐蚀
                foreground_eroded = 1 - F.conv2d(1 - foreground_eroded, kernel, padding=1)
                foreground_eroded = (foreground_eroded > 0).float()

            # 计算三个区域
            definite_fg = foreground_eroded  # 腐蚀后的前景 = 确定前景
            definite_bg = 1 - foreground_dilated  # 膨胀后的背景 = 确定背景
            uncertain = foreground_dilated - foreground_eroded  # 膨胀-腐蚀 = 不确定区域

            # 确保没有负值
            uncertain = torch.clamp(uncertain, 0, 1)


        # reshape feature & prob
        prob = torch.cat([definite_fg, definite_bg, uncertain], dim=1)

        f = x.view(b, h * w, -1)
        prob = prob.view(b, 3, h * w)

        # compute context vector
        context = torch.bmm(prob, f).permute(0, 2, 1).unsqueeze(3)  # b, 3, c

        # k q v compute
        query = self.conv_query(x).view(b, self.channel, -1).permute(0, 2, 1)
        key = self.conv_key(context).view(b, self.channel, -1)
        value = self.conv_value(context).view(b, self.channel, -1).permute(0, 2, 1)

        # compute similarity map
        sim = torch.bmm(query, key)  # b, hw, c x b, c, 2
        sim = (self.channel ** -.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        context = self.conv_out1(context)

        x = torch.cat([x, context], dim=1)
        x = self.conv_out2(x)
        x = self.conv_out3(x)
        out = self.conv_out4(x)

        return out
