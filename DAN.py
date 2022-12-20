import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def make_model():
    return LAN()

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean)

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False
#通道注意力
class CA(nn.Module):
    def __init__(self, channels, reduction):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_ca = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels // reduction, channels, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        out1 = self.avg_pool(x)
        out2 = self.conv_ca(x)
        out = out1 * out2

        return out

# 像素注意力
class PA(nn.Module):
    def __init__(self, channels):
        super(PA, self).__init__()
        self.conv_pa = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)

    def forward(self, x):
        out1 = self.conv_pa(x)
        out2 = self.conv(x)
        out = out1 * out2

        return out


#***********************************************************************

class LAB(nn.Module):

    def __init__(self, channels=64, r=4, K=2, t=30, reduction=8):
        super(LAB, self).__init__()
        self.t = t
        self.K = K

        self.conv_first = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )

        self.conv1x1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        self.conv_last = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Attention Dropout Module  注意力分值权重
        self.ADM = nn.Sequential(
            nn.Linear(channels, channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, self.K, bias=False),
        )

        # attention branch
        self.p_attention = PA(channels)
        self.c_attention = CA(channels, reduction)
        # non-attention branch
        # self.no_attention = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        # 1x1 conv for A2N-M
        # self.non_attention = nn.Conv2d(nf, nf, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x
        a, b, c, d = x.shape

        x = self.conv_first(x)
        x = torch.add(x*0.1, residual)
        # x2 = self.no_attention(x)

        x1 = x#经过初始卷积
        x1 = self.conv1x1(x1)
        # Attention Dropout  注意划分
        y = self.avg_pool(x).view(a, b)
        y = self.ADM(y)
        ax = F.softmax(y / self.t, dim=1)
        #蝶式结构
        x_down1 = self.c_attention(x1)
        x_up1 = torch.add(x1, x_down1*0.1)
        x_up2 = self.p_attention(x_up1)
        x_down2 = torch.add(x_down1,x1*0.1)
        x_down3 = torch.add(x_down2, x_up2*0.1)
        x_up3 = torch.add(x_up2, x_down2*0.1)
      #  attention = self.attention(x)
      #  non_attention = self.non_attention(x)
        #动态系数相加
        x = x_down3 * ax[:, 0].view(a, 1, 1, 1) + x_up3 * ax[:, 1].view(a, 1, 1, 1)
        x = self.lrelu(x)
        out = self.conv_last(x)
        out = torch.add(out*0.1, residual)

        return out

# class LAG(nn.Module):
#     def __init__(self):
#         super(LAG, self).__init__()
#
#         self.group = self.make_layer(LAB, 3)
#         self.tail = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#
#     def make_layer(self, block, num_of_layer):
#         layers = []
#         for _ in range(num_of_layer):
#             layers.append(block())
#         return nn.Sequential(*layers)  # 将按照构造函数中传递的顺序添加到模块中
#
#     def forward(self, x):
#         residual = x
#         out = self.group(x)
#         out = self.tail(x)
#         out = torch.add(out*0.1, residual)
#
#         return  out

class LAN(nn.Module):
    def __init__(self):
        super(LAN, self).__init__()

        in_channels = 3
        channels = 64

        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.sub_mean = MeanShift(rgb_mean, -1)


        # RGB mean for DIV2K
       # rgb_mean = (0.4488, 0.4371, 0.4040)
        #rgb_std = (1.0, 1.0, 1.0)
       # self.sub_mean = common.MeanShift(255.0, rgb_mean, rgb_std)
       # self.add_mean = common.MeanShift(255.0, rgb_mean, rgb_std, 1)

        # head module
        self.head =  nn.Conv2d(in_channels, channels, 3, 1, 1, bias=False)

        # body
        self.body = self.make_layer(LAB, 32)

        # **************************  上采样
        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64 * 9, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(3),  # 缩放的倍数
            # nn.Conv2d(in_channels=64, out_channels=64 * 4, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.PixelShuffle(2),
        )
        # *****************上采样结束


        # tail
        self.tail = nn.Conv2d(channels, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

        self.add_mean = MeanShift(rgb_mean, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)  # 将按照构造函数中传递的顺序添加到模块中

    def forward(self, x):
        # sub mean
        x = self.sub_mean(x)
        # head
        x = self.head(x)
        # body
        res = x
        x = self.body(x)
        res = x + res
        #Upsampler
        res = self.upscale4x(res)
        # tail
        x = self.tail(res)
        # add mean
        out = self.add_mean(x)

        return out
