import nntplib

import torch
from torch import fft, nn
from torch.nn.utils import spectral_norm
from einops import rearrange
from torch.optim import AdamW
from torch import Tensor
from termcolor import colored
from timm.models.layers import trunc_normal_, DropPath
import numpy as np
import sys
from math import pi, sqrt, ceil
from common import *

try:
    from torch import irfft
    from torch import rfft
except ImportError:
    from torch.fft import irfft2
    from torch.fft import rfft2
    def rfft(x, d):
        t = rfft2(x, dim = (-d,-1))
        return torch.stack((t.real, t.imag), -1)
    def irfft(x, d, signal_sizes):
        return irfft2(torch.complex(x[:,:,0], x[:,:,1]), s = signal_sizes, dim = (-d,-1))
# xiugaiblock
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.set_printoptions(precision=2)
def OutImg(x, out_bias='tanh'):
    if out_bias == 'sigmoid':
        return torch.sigmoid(x)
    elif out_bias == 'tanh':
        return (torch.tanh(x) * 0.5) + 0.5
    else:
        return x + float(out_bias)
class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        # self.dwconv = InceptionDWConv2d(in_channels=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)

        # x = self.grn(x)

        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)


        return x
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNeXt(nn.Module):
    def __init__(self, stage_blocks=0, strds=[2, 2, 2, 2], dims=[96, 192, 384, 768],
                 # strds=[5,4,4,2,2],dims=[64,64,64,64,16]
                 in_chans=3, drop_path_rate=0., layer_scale_init_value=1e-6,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stage_num = len(dims)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, stage_blocks * self.stage_num)]
        cur = 0
        for i in range(self.stage_num):
            # Build downsample layers
            if i > 0:
                downsample_layer = nn.Sequential(
                    LayerNorm(dims[i - 1], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i - 1], dims[i], kernel_size=strds[i], stride=strds[i]),
                )
            else:
                downsample_layer = nn.Sequential(
                    nn.Conv2d(in_chans, dims[0], kernel_size=strds[i], stride=strds[i]),
                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
                )
            self.downsample_layers.append(downsample_layer)

            # Build more blocks
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(stage_blocks)]
            )
            self.stages.append(stage)
            cur += stage_blocks

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_list = []
        for i in range(self.stage_num):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            out_list.append(x)
        # return out_list[-1],out_list[0]
        return out_list[-1],out_list
class FuseBlock7(nn.Module):
    def __init__(self, channels):
        super(FuseBlock7, self).__init__()
        # self.fre = nn.Conv2d(channels, channels, 3, 1, 1)
        # self.spa = nn.Conv2d(channels, channels, 3, 1, 1)

        self.fre = nn.Conv2d(channels, channels, 3, 1, 1)
        self.spa = nn.Conv2d(channels, channels, 3, 1, 1)
        # self.fre_att = Attention(dim=channels)
        # self.spa_att = Attention(dim=channels)
        self.fuse = nn.Sequential(nn.Conv2d(2*channels, channels, 3, 1, 1), nn.Conv2d(channels, 2*channels, 3, 1, 1), nn.Sigmoid())

        # self.fuse = nn.Sequential(nn.Conv2d(2 * channels, channels, 1, 1, 1),
        #                           nn.Conv2d(channels, 2 * channels, 1, 1, 1), nn.Sigmoid())


    def forward(self, spa, fre):
        ori = spa
        fre = self.fre(fre)
        spa = self.spa(spa)
        # fre = self.fre_att(fre, spa)+fre
        # spa = self.fre_att(spa, fre)+spa
        fuse = self.fuse(torch.cat((fre, spa), 1))
        fre_a, spa_a = fuse.chunk(2, dim=1)
        spa = spa_a * spa
        fre = fre * fre_a
        res = fre + spa

        res = torch.nan_to_num(res, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return res

def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
   #  print(filter_LL.size())
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    LL = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)

    return LL, LH, HL, HH
def get_wav_two(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
   #  print(filter_LL.size())
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH
class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        # self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

        self.LL, self.LH, self.HL, self.HH = get_wav_two(in_channels,pool=True)
    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


def get_daubechies_wavelet():
    # Daubechies 4 (db2) 一维滤波器系数（需归一化）
    L = np.array([0.4830, 0.8365, 0.2241, -0.1294])  # 低通滤波器
    H = np.array([-0.1294, -0.2241, 0.8365, -0.4830])  # 高通滤波器 (通过 QMF 生成)

    # 归一化
    L = L / np.sqrt(np.sum(L ** 2))
    H = H / np.sqrt(np.sum(H ** 2))
    return L, H
#
# '

#db2
# def get_wav_two(in_channels, wavelet_type='haar', pool=True):
#     """支持不同小波的二维分解"""
#     if wavelet_type == 'haar':
#         # Haar 小波系数（原代码）
#         harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
#         harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
#         harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]
#     elif wavelet_type == 'db2':
#         # Daubechies 4 小波系数
#         L, H = get_daubechies_wavelet()
#         harr_wav_L = L.reshape(1, -1)  # 形状调整为 (1, 4)
#         harr_wav_H = H.reshape(1, -1)
#     else:
#         raise ValueError("Unsupported wavelet type")
#
#     # 生成二维滤波器（外积）
#     harr_wav_LL = np.outer(harr_wav_L, harr_wav_L)
#     harr_wav_LH = np.outer(harr_wav_L, harr_wav_H)
#     harr_wav_HL = np.outer(harr_wav_H, harr_wav_L)
#     harr_wav_HH = np.outer(harr_wav_H, harr_wav_H)
#
#     # 转换为 PyTorch Tensor
#     filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0).float()  # 形状 (1, kernel_size, kernel_size)
#     filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0).float()
#     filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0).float()
#     filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0).float()
#
#     # 选择卷积或反卷积层
#     net = nn.Conv2d if pool else nn.ConvTranspose2d
#     kernel_size = harr_wav_L.shape[1]  # 根据小波长度动态调整（如 Haar 是 2，Daubechies 4 是 4）
#
#     # 创建卷积层（需调整 kernel_size 和 padding）
#     LL = net(in_channels, in_channels, kernel_size=kernel_size,
#              stride=2, padding=kernel_size // 2 - 1,  # 保持输出尺寸为输入的一半
#              bias=False, groups=in_channels)
#     LH = net(in_channels, in_channels, kernel_size=kernel_size,
#              stride=2, padding=kernel_size // 2 - 1, bias=False, groups=in_channels)
#     HL = net(in_channels, in_channels, kernel_size=kernel_size,
#              stride=2, padding=kernel_size // 2 - 1, bias=False, groups=in_channels)
#     HH = net(in_channels, in_channels, kernel_size=kernel_size,
#              stride=2, padding=kernel_size // 2 - 1, bias=False, groups=in_channels)
#
#     # 固定权重
#     for conv in [LL, LH, HL, HH]:
#         conv.weight.requires_grad = False
#     LL.weight.data = filter_LL.expand(in_channels, -1, -1, -1)
#     LH.weight.data = filter_LH.expand(in_channels, -1, -1, -1)
#     HL.weight.data = filter_HL.expand(in_channels, -1, -1, -1)
#     HH.weight.data = filter_HH.expand(in_channels, -1, -1, -1)
#
#     return LL, LH, HL, HH
#
#
# class WavePool(nn.Module):
#     def __init__(self, in_channels, wavelet_type='db2'):
#         super(WavePool, self).__init__()
#         self.LL, self.LH, self.HL, self.HH = get_wav_two(in_channels, wavelet_type=wavelet_type, pool=True)
#
#     def forward(self, x):
#         return self.LL(x), self.LH(x), self.HL(x), self.HH(x)

#db4 db8
# import pywt
#
# def get_wavelet_filters(wavelet_name):
#     wavelet = pywt.Wavelet(wavelet_name)
#     L = wavelet.dec_lo  # 低通滤波器系数
#     H = wavelet.dec_hi  # 高通滤波器系数
#     return L, H
# def get_wav_two(in_channels, wavelet_name='haar', pool=True):
#     """支持任意 Daubechies 小波的二维分解"""
#     # 获取小波滤波器系数
#     if wavelet_name.lower().startswith('db'):
#         # 使用 pywt 自动获取系数
#         L, H = get_wavelet_filters(wavelet_name)
#         L = np.array(L)
#         H = np.array(H)
#     else:
#         # 手动输入其他小波（如 Haar）
#         pass
#
#     # 生成二维滤波器（外积）
#     LL = np.outer(L, L)  # 低通水平 + 低通垂直
#     LH = np.outer(L, H)  # 低通水平 + 高通垂直
#     HL = np.outer(H, L)  # 高通水平 + 低通垂直
#     HH = np.outer(H, H)  # 高通水平 + 高通垂直
#
#     # 转换为 PyTorch Tensor
#     kernel_size = len(L)
#     filter_LL = torch.from_numpy(LL).float().unsqueeze(0)  # 形状 (1, kernel_size, kernel_size)
#     filter_LH = torch.from_numpy(LH).float().unsqueeze(0)
#     filter_HL = torch.from_numpy(HL).float().unsqueeze(0)
#     filter_HH = torch.from_numpy(HH).float().unsqueeze(0)
#
#     # 选择卷积或反卷积层
#     net = nn.Conv2d if pool else nn.ConvTranspose2d
#     padding = kernel_size // 2 - 1  # 保持输出尺寸为输入的一半（需验证）
#
#     # 创建卷积层
#     LL_conv = net(in_channels, in_channels,
#                   kernel_size=kernel_size, stride=2, padding=padding,
#                   bias=False, groups=in_channels)
#     LH_conv = net(in_channels, in_channels,
#                   kernel_size=kernel_size, stride=2, padding=padding,
#                   bias=False, groups=in_channels)
#     HL_conv = net(in_channels, in_channels,
#                   kernel_size=kernel_size, stride=2, padding=padding,
#                   bias=False, groups=in_channels)
#     HH_conv = net(in_channels, in_channels,
#                   kernel_size=kernel_size, stride=2, padding=padding,
#                   bias=False, groups=in_channels)
#
#     # 固定权重并赋值
#     for conv in [LL_conv, LH_conv, HL_conv, HH_conv]:
#         conv.weight.requires_grad = False
#     LL_conv.weight.data = filter_LL.expand(in_channels, -1, -1, -1)
#     LH_conv.weight.data = filter_LH.expand(in_channels, -1, -1, -1)
#     HL_conv.weight.data = filter_HL.expand(in_channels, -1, -1, -1)
#     HH_conv.weight.data = filter_HH.expand(in_channels, -1, -1, -1)
#
#     return LL_conv, LH_conv, HL_conv, HH_conv
#
#
# class WavePool(nn.Module):
#     def __init__(self, in_channels, wavelet_name='db8'):
#         super().__init__()
#         self.LL, self.LH, self.HL, self.HH = get_wav_two(
#             in_channels, wavelet_name=wavelet_name, pool=True
#         )
#
#     def forward(self, x):
#         return self.LL(x), self.LH(x), self.HL(x), self.HH(x)

# def get_wavelet_filters(wavelet_name):
#     """通过 pywt 获取 Symlet 或其他小波的滤波器系数"""
#     wavelet = pywt.Wavelet(wavelet_name)
#     L = wavelet.dec_lo  # 低通滤波器
#     H = wavelet.dec_hi  # 高通滤波器
#     return np.array(L), np.array(H)
#
#
# def get_wav_two(in_channels, wavelet_name='sym4', pool=True):
#     """支持 Symlet 的二维小波分解"""
#     # 获取滤波器系数
#     L, H = get_wavelet_filters(wavelet_name)
#     kernel_size = len(L)
#
#     # 生成二维滤波器（外积）
#     LL = np.outer(L, L)
#     LH = np.outer(L, H)
#     HL = np.outer(H, L)
#     HH = np.outer(H, H)
#
#     # 转换为 PyTorch Tensor
#     filter_LL = torch.from_numpy(LL).float().unsqueeze(0)  # 形状 (1, kernel_size, kernel_size)
#     filter_LH = torch.from_numpy(LH).float().unsqueeze(0)
#     filter_HL = torch.from_numpy(HL).float().unsqueeze(0)
#     filter_HH = torch.from_numpy(HH).float().unsqueeze(0)
#
#     # 选择卷积或反卷积层
#     net = nn.Conv2d if pool else nn.ConvTranspose2d
#     padding = (kernel_size - 2) // 2  # 保证 stride=2 时输出尺寸减半
#
#     # 创建卷积层
#     LL_conv = net(in_channels, in_channels,
#                   kernel_size=kernel_size, stride=2, padding=padding,
#                   bias=False, groups=in_channels)
#     # 同理创建 LH_conv, HL_conv, HH_conv...
#     LH_conv = net(in_channels, in_channels,
#                   kernel_size=kernel_size, stride=2, padding=padding,
#                   bias=False, groups=in_channels)
#     HL_conv = net(in_channels, in_channels,
#                   kernel_size=kernel_size, stride=2, padding=padding,
#                   bias=False, groups=in_channels)
#     HH_conv = net(in_channels, in_channels,
#                   kernel_size=kernel_size, stride=2, padding=padding,
#                   bias=False, groups=in_channels)
#
#     # 固定权重并赋值
#     for conv in [LL_conv, LH_conv, HL_conv, HH_conv]:
#         conv.weight.requires_grad = False
#     LL_conv.weight.data = filter_LL.expand(in_channels, -1, -1, -1)
#     LH_conv.weight.data = filter_LH.expand(in_channels, -1, -1, -1)
#     HL_conv.weight.data = filter_HL.expand(in_channels, -1, -1, -1)
#     HH_conv.weight.data = filter_HH.expand(in_channels, -1, -1, -1)
#
#     # 固定权重并赋值（同之前代码）
#     return LL_conv, LH_conv, HL_conv, HH_conv
#
#
# class WavePool(nn.Module):
#     def __init__(self, in_channels, wavelet_name='sym4'):
#         super().__init__()
#         self.LL, self.LH, self.HL, self.HH = get_wav_two(
#             in_channels, wavelet_name=wavelet_name, pool=True
#         )
#
#     def forward(self, x):
#         return self.LL(x), self.LH(x), self.HL(x), self.HH(x)
#

class WaveUnpool(nn.Module):
    def __init__(self, in_channels, option_unpool='cat5'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = get_wav_two(self.in_channels, pool=False)

    def forward(self, LL, LH, HL, HH, original=None):
        if self.option_unpool == 'sum':
             return self.LH(LH) + self.HL(HL) + self.HH(HH)
           # return self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat5' and original is not None:
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
        else:
            raise NotImplementedError

class CustomModule(nn.Module):
    def __init__(self,ngf):
        super(CustomModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=1, stride=1, bias=True)
        self.ps1 = nn.PixelShuffle(4)
        self.act1 = adaActionF()
        # self.a1 = nn.Parameter(torch.FloatTensor([0.1]))

        self.conv2 = nn.Conv2d(2, ngf*16, kernel_size=1, stride=1, bias=True)
        self.ps2 = nn.PixelShuffle(4)
        self.act2 = adaActionF()
        # self.a2 = nn.Parameter(torch.FloatTensor([0.1]))

    def forward(self, x):
        x = self.act1(self.ps1(self.conv1(x)))
        x = self.act2(self.ps2(self.conv2(x)))
        return x
class Sin(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sin, self).__init__()

    def forward(self, input):
        return torch.sin(input)

class Cos(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Cos, self).__init__()

    def forward(self, input):
        return torch.cos(input)
class RickerWavelet(nn.Module):
    def __init__(self, a=0.5):
        super().__init__()
        self.a = a
    def forward(self, x):
        a = self.a
        return (1 - (x/a)**2) * torch.exp(-(x**2) / (2 * a**2)) / (15 * a * torch.sqrt(torch.tensor(np.pi)))
class adaActionF(nn.Module):
    def __init__(self):
        super().__init__()
        # self.act = nn.GELU()
        # self.a = nn.Parameter(torch.FloatTensor([0.1]))
        self.sin = Sin()
        self.cos = Cos()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.ones(1))
        # self.act = RickerWavelet()
    def forward(self,x):
        # return self.act(10*self.a*x)
        return self.a*self.sin(x)+self.b*self.cos(x)
        # return self.sin(x)
        # return self.act(x)
        # return torch.sin(x)

class FARConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,groups=1,bias=True):
        super(FARConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        H, W = kernel_size
        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, H, W))

        # Create DCT basis
        self.register_buffer('dct_basis', self.create_dct_basis(H, W))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        k = self.groups / (self.weights.size(1) * torch.prod(torch.tensor(self.kernel_size)))
        nn.init.uniform_(self.weights, -torch.sqrt(k), torch.sqrt(k))
        if self.bias is not None:
            nn.init.uniform_(self.bias, -torch.sqrt(k), torch.sqrt(k))
    def create_dct_basis(self, H, W):
        basis = torch.zeros(H, W, H, W)
        for i in range(H):
            for j in range(W):
                ci = math.sqrt((1 if i == 0 else 2) / H)
                cj = math.sqrt((1 if j == 0 else 2) / W)
                for h in range(H):
                    for w in range(W):
                        basis[i, j, h, w] = ci * cj * torch.cos(
                            (2 * torch.tensor(h) + 1) * torch.tensor(i) * torch.pi / (2 * H)) * \
                                            torch.cos((2 * torch.tensor(w) + 1) * torch.tensor(j) * torch.pi / (2 * W))
        return basis.view(H * W, H * W)

    def forward(self, x):
        H, W = self.kernel_size  # You already have H and W from __init__

        # Flatten the DCT basis and the weights to compute the weighted sum of DCT basis
        weight_flat = self.weights.view(self.out_channels * self.in_channels, H * W)
        dct_basis_flat = self.dct_basis.view(H * W, H * W)

        # Compute the weighted sum of DCT basis
        weight_flat_dct = torch.matmul(weight_flat, dct_basis_flat)
        # Reshape back to the original weight shape
        weight_dct = weight_flat_dct.view(self.out_channels, self.in_channels, H, W)

        # Perform the convolution operation
        x = F.conv2d(x, weight_dct, self.bias, self.stride, self.padding)
        return x
class res_NeRVBlock116(nn.Module):
    def __init__(self, **kargs):
        super().__init__()

        ngf, new_ngf, stride, ks ,k = kargs['ngf'], kargs['new_ngf'], kargs['strd'], kargs['ks'],kargs['k']
        self.conv = nn.Conv2d(ngf, new_ngf * stride * stride, (ks,ks), 1, ceil((ks - 1) // 2), bias=True)
        self.up_scale = nn.PixelShuffle(stride)
        # self.act = nn.GELU()
        # self.a = nn.Parameter(torch.FloatTensor([0.1]))
        self.act = adaActionF()
        self.sft_block = ResBlock_SFT(new_ngf, new_ngf, cond_ch=32,k=k,in_act="relu", out_act="gelu", omega=1, args=None)
        # self.g = Sin()
        # self.l = nn.Tanh()
        # self.a = nn.Parameter(torch.rand(1))
    def forward(self, x):
        embed = x[1]
        x = self.conv(x[0])
        x = self.up_scale(x)
        # x = self.act(10*self.a*x)
        # a = torch.sigmoid(self.a)
        # x = a * self.g(5*x) + (1 - a) * self.l(x)
        # x = hat_activation(x)
        x = self.act(x)
        y,embed = self.sft_block((x, embed))
        return y,embed,x
class Decoder_first(nn.Module):
    def __init__(self, ngf):
        super().__init__()
        self.conv = nn.Conv2d(16, ngf, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.act = adaActionF()
        # self.sft_block = ResBlock_SFT(ngf, ngf, cond_ch=32, k=1, in_act="relu", out_act="gelu", omega=1,
        #                               args=None)
        # self.act = nn.GELU()
    def forward(self,x):
        # x = self.conv(x)
        # x = self.act(x)
        # return x
        # embed = x[1]
        x = self.conv(x)
        # x = self.up_scale(x)
        x = self.act(x)
        # x, embed = self.sft_block((F.interpolate(x,scale_factor=5), embed))
        return x
class HighEncoder(nn.Module):
    def __init__(self, input_channels=64,output_channel=2):
        super(HighEncoder, self).__init__()
        # self.conv0 = nn.Conv2d(input_channels, input_channels, kernel_size=2, stride=2)
        self.conv0 = nn.Sequential(nn.Conv2d(3, input_channels, kernel_size=2, stride=2),LayerNorm(input_channels, eps=1e-6, data_format="channels_first"),Block(input_channels, drop_path=0.0,layer_scale_init_value=1e-6))
        self.conv1 = nn.Sequential(LayerNorm(input_channels, eps=1e-6, data_format="channels_first"),nn.Conv2d(input_channels, input_channels, kernel_size=2, stride=2),Block(input_channels, drop_path=0.0,layer_scale_init_value=1e-6))
        self.conv2 = nn.Sequential(LayerNorm(input_channels*2, eps=1e-6, data_format="channels_first"),nn.Conv2d(input_channels*2, input_channels, kernel_size=2, stride=2),Block(input_channels, drop_path=0.0,layer_scale_init_value=1e-6))
        self.conv3 = nn.Sequential(LayerNorm(input_channels*2, eps=1e-6, data_format="channels_first"),nn.Conv2d(input_channels*2, input_channels, kernel_size=2, stride=2),Block(input_channels, drop_path=0.0,layer_scale_init_value=1e-6))
        # self.conv4 = nn.Conv2d(input_channels, input_channels, kernel_size=2, stride=2)
        self.conv5 = nn.Sequential(LayerNorm(input_channels*2, eps=1e-6, data_format="channels_first"),nn.Conv2d(input_channels*2, out_channels=output_channel, kernel_size=2, stride=2),Block(output_channel, drop_path=0.0,layer_scale_init_value=1e-6))


        self.apply(self._init_weights)

        self.pool3 = WavePool(3)
        self.pool64 = WavePool(64)
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
    def forward(self, x):
        # LL, LH, HL, HH = self.pool64(x)
        # x = self.conv1(LH+HL+HH)
        # LL, LH, HL, HH = self.pool64(LL)
        # x = self.conv2(x+LH+HL+HH)
        # LL, LH, HL, HH = self.pool64(LL)
        # x = self.conv3(x+LH+HL+HH)
        # LL, LH, HL, HH = self.pool64(LL)
        # x = self.conv5(x + LH + HL + HH)
        x= self.conv0(x)
        LL, LH, HL, HH = self.pool64(x)
        # high = LH + HL + HH
        x1 = self.conv1(LH + HL + HH)
        LL, LH, HL, HH = self.pool64(LL)
        x = self.conv2(torch.cat([x1,LH + HL + HH],dim=1))
        LL, LH, HL, HH = self.pool64(LL)
        x = self.conv3(torch.cat([x,LH + HL + HH],dim=1))
        LL, LH, HL, HH = self.pool64(LL)
        x = self.conv5(torch.cat([x,LH + HL + HH],dim=1))

        # x0 = self.conv0(x)
        # LL, LH, HL, HH = self.pool64(x)
        # x = self.conv1(x)
        # LL, LH, HL, HH = self.pool64(LL)
        # x = self.conv2(torch.cat([x, LH + HL + HH], dim=1))
        # LL, LH, HL, HH = self.pool64(LL)
        # x = self.conv3(torch.cat([x, LH + HL + HH], dim=1))
        # LL, LH, HL, HH = self.pool64(LL)
        # x = self.conv5(torch.cat([x, LH + HL + HH], dim=1))

        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)

        # LL, LH, HL, HH = self.pool64(x)
        # x = self.res1(LH + HL + HH)
        # LL, LH, HL, HH = self.pool64(LL)
        # x = self.res2(torch.cat([x,LH + HL + HH],dim=1))
        # LL, LH, HL, HH = self.pool64(LL)
        # x = self.res3(torch.cat([x,LH + HL + HH],dim=1))
        # LL, LH, HL, HH = self.pool64(LL)
        # x = self.res4(torch.cat([x,LH + HL + HH],dim=1))

        # LL, LH, HL, HH = self.pool64(x)
        # x = self.unpool64(LL, LH, HL, HH)
        # x = self.conv2(x)
        # LL, LH, HL, HH = self.pool64(x)
        # x = self.unpool64(LL, LH, HL, HH)
        # x = self.conv3(x)
        # LL, LH, HL, HH = self.pool64(x)
        # x = self.unpool64(LL, LH, HL, HH)
        # x = self.conv4(x)
        # LL, LH, HL, HH = self.pool64(x)
        # x = self.unpool64(LL, LH, HL, HH)
        # x=self.conv5(x)
        # #x = self.linear(x)
        return x,x1
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1, bias=True):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.hid_fea = hidden_features
        self.dim = dim

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.layernorm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
    def forward(self, x):
        input = x
        x = self.layernorm(x)
        self.h, self.w = x.shape[2:]
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x+input
class ResBlock_SFT(nn.Module):
    def __init__(self, in_ch, out_ch, cond_ch, factor=1, k=1,in_act="relu", out_act="gelu", omega=1., args=None):
        super().__init__()
        self.k=k
        self.sft0 = SFTLayer(cond_ch, in_ch, factor, in_act, omega, args=args)
        #todo
        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.act = nn.GELU()

        # self.ffn = FeedForward(in_ch)
        # self.sft1 = SFTLayer(cond_ch, out_ch, factor, in_act, omega, args=args)
        # self.conv1 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        # self.up = nn.Sequential(
        #     nn.MaxPool2d(3, 1, 1),
        #     nn.Conv2d(cond_ch, cond_ch * k * k, 1, bias=True),
        #     nn.PixelShuffle(k),
        #     nn.GELU()
        # )


    def forward(self, x):
        # x[0]: fea; x[1]: cond
        cont=x[0]
        embed = F.interpolate(x[1], scale_factor=self.k, mode='bicubic', align_corners=False)
        # embed = self.up(x[1])
        fea = self.sft0((cont,embed))
        #todo
        # fea = self.ffn(fea)
        fea = self.act(self.conv0(fea))

        # fea = self.sft1((fea, embed))
        # fea = self.conv1(fea)
        return x[0] + fea,embed
class SFTLayer(nn.Module):
    def __init__(self, in_ch, out_ch, factor=1, act="relu", omega=1., args=None):
        super().__init__()
        self.SFT_scale_conv0 = nn.Conv2d(in_ch, in_ch//factor, 1)
        self.SFT_scale_conv1 = nn.Conv2d(in_ch//factor, out_ch, 1)
        self.SFT_shift_conv0 = nn.Conv2d(in_ch, in_ch//factor, 1)
        self.SFT_shift_conv1 = nn.Conv2d(in_ch//factor, out_ch, 1)
        self.act = nn.ReLU(True)
        # self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        # self.act = nn.GELU()

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(self.act(self.SFT_scale_conv0(x[1])))
        shift = self.SFT_shift_conv1(self.act(self.SFT_shift_conv0(x[1])))
        return x[0] * (scale + 1) + shift
class HNeRV(nn.Module):
    def __init__(self, fc_dim=95, ks='0_1_5', num_blks='1_1', enc_dim='64_16', enc_strds=[5, 4, 4, 2,2],
                 dec_strds=[5, 4, 4, 2, 2], reduce=1.2, lower_width=12, conv_type=['convnext', 'pshuffel'],training=True):
        super().__init__()
        ks_enc, ks_dec1, ks_dec2 = [int(x) for x in ks.split('_')]
        enc_blks, dec_blks = [int(x) for x in num_blks.split('_')]
        self.training = training
        enc_dim1, enc_dim2 = [int(x) for x in enc_dim.split('_')]
        c_in_list, c_out_list = [enc_dim1] * len(enc_strds), [enc_dim1] * len(enc_strds)
        c_out_list[-1] = enc_dim2
        self.encoder = ConvNeXt(stage_blocks=enc_blks, strds=enc_strds, dims=c_out_list, drop_path_rate=0)

        self.highencoder = HighEncoder()
        # self.highencoder = ConvNeXt(stage_blocks=enc_blks, strds=[4,4,2,2], dims=[64,64,64,2], drop_path_rate=0)
        self.fc_h, self.fc_w = 1, 1
        ch_in = 16
        decoder_layers = []
        ngf = fc_dim
        out_f = int(ngf * self.fc_h * self.fc_w)
        decoder_layer1 = Decoder_first(ngf)
        decoder_layers.append(decoder_layer1)

        #3M 96
        # a = [80, 66, 54, 48, 40]
        #1.5M 68
        # a = [55, 46, 38, 30, 28]
        #0.75M 48
        # a = [40,32,25,21,18]
        # a = [40, 32, 25, 20, 18]
        #0.35M 32
        # a =[26,20,16,14,12]
        up = [1,4,4,2,2]
        for i, strd in enumerate(dec_strds):
            reduction = reduce
            new_ngf = int(max(round(ngf / reduction), lower_width))
            # a = [80,68,56,48,40]
            # new_ngf = a[i]
            #96,80,68,56,48,40
            #
            #

            # new_ngf = a[i]
            for j in range(dec_blks):
                # cur_blk = nn.Sequential(res_NeRVBlock116(dec_block=True, conv_type=conv_type[1], ngf=ngf, new_ngf=new_ngf,
                #                          ks=min(ks_dec1 + 2 * i, ks_dec2), strd=1 if j else strd, bias=True,
                #                          norm='none',
                #                          act='gelu', j=i),
                #                         BasicBlock(new_ngf,new_ngf))
                cur_blk = res_NeRVBlock116(dec_block=True, conv_type=conv_type[1], ngf=ngf, new_ngf=new_ngf,
                                     ks=min(ks_dec1 + 2 * i, ks_dec2), strd=1 if j else strd, bias=True,
                                     norm='none',
                                     act='gelu', k=up[i])


                decoder_layers.append(cur_blk)

                ngf = new_ngf
        # a = [100, 69, 56, 48, 40]
        # self.h2 = CustomModule(a[2])
        # decoder_layers.append(self.h2)
        # self.FuseBlock7 = FuseBlock7(a[2])
        # self.FuseBlock7 = SFTLayer(a[2],a[2])
        # self.FuseBlock7 = FRAN(a[2],a[2])
        # decoder_layers.append(self.FuseBlock7)
        self.stem_high = nn.Sequential(
            nn.Conv2d(2, 32, 1, bias=True),
            adaActionF()
        )
        decoder_layers.append(self.stem_high)
        self.decoder = nn.ModuleList(decoder_layers)
        self.head_layer = nn.Conv2d(new_ngf, 3, 3, 1, 1)
        self.out_bias = 'tanh'

    def forward(self, input):
        img_embed,x0 = self.encoder(input)
        high_embed,x1 = self.highencoder(input)
        high_embed = self.stem_high(high_embed)
        output = self.decoder[0](img_embed)
        for i, layer in enumerate(self.decoder[1:6]):
                output,embed,x = layer((output,high_embed))
                high_embed = embed

        img_out = OutImg(self.head_layer(output), self.out_bias)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return img_out
def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed
def train(local_rank):
    cudnn.benchmark = True
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    # gauss_kernel = get_gaussian_kernel(21).cuda()

    outf = './output/log'
    # data_path = '../data/readysteadygo'
    data_path = '../data/bunny'
    # data_path = './data/DAVIS/blackswan'
    # data_path = './data/lowbunny_x4'
    # crop_list, resize_list = '160_320', '-1'
    crop_list, resize_list = '640_1280', '-1'
    # crop_list, resize_list = '720_1280', '-1'
    # crop_list, resize_list = '960_1920', '-1'

    batchSize = 1
    workers = 4
    datasplit = '1_1_1'

    full_dataset = VideoDataSet(data_path=data_path, crop_list=crop_list, resize_list=resize_list)
    final_size = full_dataset.final_size  # 每帧大小
    full_data_length = len(full_dataset)  # 帧数

    split_num_list = [int(x) for x in datasplit.split('_')]
    train_ind_list, val_ind_list = data_split(list(range(full_data_length)), split_num_list, False, 0)

    train_dataset = Subset(full_dataset, train_ind_list)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True,
                                                   num_workers=workers, pin_memory=True, drop_last=True,
                                                   worker_init_fn=worker_init_fn)
    embed = ''
    ks = '0_1_5'
    num_blks = '1_1'
    enc_strds = [5, 4, 4, 2, 2]
    enc_dim = '64_16'
    conv_type = ['convnext', 'pshuffel']
    act = 'gelu'
    norm = 'none'
    dec_strds = [5, 4, 4, 2, 2]

    fc_hw = '9_16'
    reduce = 1.2
    lower_width = 12
    modelsize = 3
    saturate_stages = -1
    epochs = 300
    total_enc_strds = np.prod(enc_strds)  # 5*4*4*2*2=320
    embed_hw = final_size / total_enc_strds ** 2  # 长和宽各扩大320倍 2x4=8
    enc_dim1, embed_ratio = [float(x) for x in enc_dim.split('_')]
    embed_dim = int(embed_ratio * modelsize * 1e6 / full_data_length / embed_hw) if embed_ratio < 1 else int(
        embed_ratio)
    embed_param = float(
        embed_dim) / total_enc_strds ** 2 * final_size * full_data_length+400*132 \
                   # embed元素个数:embed_hw*embed_dim*full_data_length 8*16*132
    enc_dim = f'{int(enc_dim1)}_{embed_dim}'

    fc_param = (np.prod(enc_strds) // np.prod(dec_strds)) ** 2 * 9

    decoder_size = modelsize * 1e6 - embed_param
    ch_reduce = 1. / reduce
    dec_ks1, dec_ks2 = [int(x) for x in ks.split('_')[1:]]
    fix_ch_stages = len(dec_strds) if saturate_stages == -1 else saturate_stages
    a = ch_reduce * sum([ch_reduce ** (2 * i) * s ** 2 * min((2 * i + dec_ks1), dec_ks2) ** 2 for i, s in
                         enumerate(dec_strds[:fix_ch_stages])])
    b = embed_dim * fc_param
    c = lower_width ** 2 * sum([s ** 2 * min(2 * (fix_ch_stages + i) + dec_ks1, dec_ks2) ** 2 for i, s in
                                enumerate(dec_strds[fix_ch_stages:])])
    fc_dim = int(np.roots([a, b, c - decoder_size]).max())
    # fc_dim = 96
    model = HNeRV()
    decoder_param = (sum([p.data.nelement() for p in model.decoder.parameters()]) / 1e6)
    total_param = decoder_param + embed_param / 1e6
    print(str(model) + '\n' + str(total_param) + '\n')

    # projecth = ProjectionHead(input_dim=4096, output_dim=128, head_type='mlp').cuda()
    # projectl= ProjectionHead(input_dim=128, output_dim=128, head_type='mlp').cuda()

    # mutual = Mutual_info_reg(50, 50).to('cuda')

    model = model.cuda()
    # optimizer = optim.Adam(model.parameters(), weight_decay=0.)
    from optimizer import Adan
    optimizer = Adan(model.parameters(), lr=0.003)

    start = datetime.now()
    psnr_list = []

    img_list = []

    edge = WavePool(3).cuda()

    for epoch in range(0, epochs):
        model.train()
        epoch_start_time = datetime.now()
        pred_psnr_list = []
        msssim_list = []
        lpips_list =[]
        # iterate over dataloader
        device = next(model.parameters()).device
        # for i, (sample,sample2) in enumerate(zip(train_dataloader,train_dataloader2)):
        d_loss = 0
        d_loss_fft = 0
        criteria = torch.nn.L1Loss()
        reg = 1e-6
        orth_loss = data_to_gpu(torch.zeros(1), device)
        for i, sample, in enumerate(train_dataloader):
            img_data, norm_idx, img_idx = data_to_gpu(sample['img'], device), data_to_gpu(sample['norm_idx'],
                                                                                          device), data_to_gpu(
                sample['idx'], device)
            # img_data2 = data_to_gpu(sample2['img'],device)
            cur_input = img_data
            img_out = model(cur_input)
            cur_epoch = (epoch + float(i) / len(train_dataloader)) / epochs
            lr = adjust_lr2(optimizer, cur_epoch,lr=0.003)
            # latent_loss = 0.1*mutual_loss
            optimizer.zero_grad()
            final_loss = loss_fn(img_out, img_data, 'Fusion6_ffl')
            final_loss.backward()
            optimizer.step()
            pred_psnr_list.append(psnr_fn_single(img_out.detach(), img_data))
            msssim_list.append(msssim_fn_single(img_out.detach(), img_data))
            # lpips_list.append(lpips_fn_single(img_out.detach(), img_data))
            if i == len(train_dataloader) - 1:
                pred_psnr = torch.cat(pred_psnr_list).mean()
                train_msssim = torch.cat(msssim_list).mean()
                # train_lpips = torch.cat(lpips_list).mean()
                print_str = '[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e} pred_PSNR: {},ssim:{},lpips:{},loss:{},MIloss:{}'.format(
                    datetime.now().strftime("%Y/%m/%d %H:%M:%S"), local_rank, epoch + 1, epochs, i + 1,
                    len(train_dataloader), lr,
                    RoundTensor(pred_psnr, 2), RoundTensor(train_msssim, 4, False),0,RoundTensor(final_loss,6),0)
                print(print_str, flush=True)
    #             if local_rank in [0, None]:
    #                 with open('{}/rank0_1200_ours_1.5M.txt'.format(outf), 'a') as f:
    #                     f.write(print_str + '\n')
    state_dict = model.state_dict()
    torch.save(state_dict, './weight/model_ha_0708.pth')

import matplotlib.pyplot as plt
def reconstruction_img():
    with torch.no_grad():
        model = HNeRV().cuda()
        model.load_state_dict(torch.load('./weight/model_ha_0708.pth'))
        data_path = '../data/bunny'
        crop_list, resize_list = '640_1280', '-1'
        batchSize = 1
        workers = 4
        datasplit = '1_1_1'

        full_dataset = VideoDataSet(data_path=data_path, crop_list=crop_list, resize_list=resize_list)
        final_size = full_dataset.final_size  # 每帧大小
        full_data_length = len(full_dataset)  # 帧数

        split_num_list = [int(x) for x in datasplit.split('_')]
        train_ind_list, val_ind_list = data_split(list(range(full_data_length)), split_num_list, False, 0)

        train_dataset = Subset(full_dataset, train_ind_list)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True,
                                                       num_workers=workers, pin_memory=True, drop_last=True,
                                                       worker_init_fn=worker_init_fn)
        for i, sample, in enumerate(train_dataloader):
            img_data, norm_idx, img_idx = data_to_gpu(sample['img'], 'cuda'), data_to_gpu(sample['norm_idx'],
                                                                                          'cuda'), data_to_gpu(sample['idx'], 'cuda')
            img_out = model(img_data)
            # for j in range(batchSize):
            #     original = img_data[j]
            #     reconstructed = img_out[j]
            #
            #     # Calculate residual
            #     residual = torch.abs(original - reconstructed)
            #
            #     residual_normalized = (residual - torch.min(residual)) / (torch.max(residual) - torch.min(residual))
            #     # save_image(residual_normalized, f'./visiualexp/residual/residual_{i*2+j}.png')
            #     to_pil = ToPILImage()
            #     residual_image = to_pil(residual_normalized)  # 不需要squeeze，假设residual_normalized是C x H x W的张量
            #
            #     # Convert to grayscale
            #     residual_gray = residual_image.convert('L')
            #
            #     # Save the image
            #     residual_gray.save(f'./visiualexp/residual/residual_gray{i * 2 + j}.png')

                # plt.figure(figsize=(12, 4))
                #
                # plt.subplot(1, 3, 1)
                # plt.title("Original Image")
                # plt.imshow(original)
                # plt.axis('off')
                #
                # plt.subplot(1, 3, 2)
                # plt.title("Reconstructed Image")
                # plt.imshow(reconstructed)
                # plt.axis('off')
                #
                # plt.subplot(1, 3, 3)
                # plt.title("Residual Image")
                # plt.imshow(residual_normalized)
                # plt.axis('off')
                #
                # plt.savefig(f'results_{i}_{j}.png')
                # plt.close()
            # psnr = psnr_fn_single(img_out.detach(), img_data)
            # ssim = msssim_fn_single(img_out.detach(), img_data)
            # save_image(img_out, f'./visiualexp/ours_readysteadygo/img_pred_{i}_psnr_{RoundTensor(psnr, 2, False)}_ssim_{RoundTensor(ssim, 4, False)}.png')
            # save_image(img_out[1], f'./visiualexp/model_high_ffl_100/img_pred_{i*2+1}_5.png')
            # save_image(highout[3][0][0], f'./visiualexp/high_pred_{i * 2}.png')
            # save_image(high_out[1], f'./visiualexp/high_pred_{i * 2 + 1}.png')
            # if i==0:
            #     break
        print("保存完毕!")
if __name__ == '__main__':
    train(None)
    # reconstruction_img()

