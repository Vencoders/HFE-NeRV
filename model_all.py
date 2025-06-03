import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from math import pi, sqrt, ceil
import torch.nn.functional as F
import numpy as np
from matplotlib.path import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms.functional import center_crop, resize
from torchvision.io import read_image
from torch.nn.functional import interpolate
import decord
decord.bridge.set_bridge('torch')
from hnerv_utils import *
import glob

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

# Video dataset
class VideoDataSet(Dataset):
    def __init__(self, args):
        if os.path.isfile(args.data_path):
            self.video = decord.VideoReader(args.data_path)
        else:
            self.video = [os.path.join(args.data_path, x) for x in sorted(os.listdir(args.data_path))]

        # Resize the input video and center crop
        self.crop_list, self.resize_list = args.crop_list, args.resize_list
        # import pdb; pdb.set_trace; from IPython import embed; embed()
        first_frame = self.img_transform(self.img_load(0))
        self.final_size = first_frame.size(-2) * first_frame.size(-1)

    def img_load(self, idx):
        if isinstance(self.video, list):
            img = read_image(self.video[idx])
        else:
            img = self.video[idx].permute(-1, 0, 1)
        return img / 255.

    def img_transform(self, img):
        if self.crop_list != '-1':
            crop_h, crop_w = [int(x) for x in self.crop_list.split('_')[:2]]
            if 'last' not in self.crop_list:
                img = center_crop(img, (crop_h, crop_w))
        if self.resize_list != '-1':
            if '_' in self.resize_list:
                resize_h, resize_w = [int(x) for x in self.resize_list.split('_')]
                img = interpolate(img, (resize_h, resize_w), 'bicubic')
            else:
                resize_hw = int(self.resize_list)
                img = resize(img, resize_hw, 'bicubic')
        if 'last' in self.crop_list:
            img = center_crop(img, (crop_h, crop_w))
        return img

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        tensor_image = self.img_transform(self.img_load(idx))
        norm_idx = float(idx) / len(self.video)
        sample = {'img': tensor_image, 'idx': idx, 'norm_idx': norm_idx}

        return sample

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

def ActivationLayer(act_type):
    if act_type == 'relu':
        act_layer = nn.ReLU(True)
    elif act_type == 'leaky':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == 'leaky01':
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'relu6':
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'sin':
        act_layer = torch.sin
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        act_layer = nn.Softplus()
    elif act_type == 'hardswish':
        act_layer = nn.Hardswish(inplace=True)
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer

def NormLayer(norm_type, ch_width):
    if norm_type == 'none':
        norm_layer = nn.Identity()
    elif norm_type == 'bn':
        norm_layer = nn.BatchNorm2d(num_features=ch_width)
    elif norm_type == 'in':
        norm_layer = nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer

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


# def get_daubechies_wavelet():
#     # Daubechies 4 (db2) 一维滤波器系数（需归一化）
#     L = np.array([0.4830, 0.8365, 0.2241, -0.1294])  # 低通滤波器
#     H = np.array([-0.1294, -0.2241, 0.8365, -0.4830])  # 高通滤波器 (通过 QMF 生成)
#
#     # 归一化
#     L = L / np.sqrt(np.sum(L ** 2))
#     H = H / np.sqrt(np.sum(H ** 2))
#     return L, H
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


class DownConv(nn.Module):
    def __init__(self, **kargs):
        super(DownConv, self).__init__()
        ks, ngf, new_ngf, strd = kargs['ks'], kargs['ngf'], kargs['new_ngf'], kargs['strd']
        if kargs['conv_type'] == 'pshuffel':
            self.downconv = nn.Sequential(
                nn.PixelUnshuffle(strd) if strd != 1 else nn.Identity(),
                nn.Conv2d(ngf * strd ** 2, new_ngf, ks, 1, ceil((ks - 1) // 2), bias=kargs['bias'])
            )
        elif kargs['conv_type'] == 'conv':
            self.downconv = nn.Conv2d(ngf, new_ngf, ks + strd, strd, ceil(ks / 2), bias=kargs['bias'])
        elif kargs['conv_type'] == 'interpolate':
            self.downconv = nn.Sequential(
                nn.Upsample(scale_factor=1. / strd, mode='bilinear', ),
                nn.Conv2d(ngf, new_ngf, ks + strd, 1, ceil((ks + strd - 1) / 2), bias=kargs['bias'])
            )

    def forward(self, x):
        return self.downconv(x)


class UpConv(nn.Module):
    def __init__(self, **kargs):
        super(UpConv, self).__init__()
        ks, ngf, new_ngf, strd = kargs['ks'], kargs['ngf'], kargs['new_ngf'], kargs['strd']
        if kargs['conv_type'] == 'pshuffel':
            self.upconv = nn.Sequential(
                nn.Conv2d(ngf, new_ngf * strd * strd, ks, 1, ceil((ks - 1) // 2), bias=kargs['bias']),
                nn.PixelShuffle(strd) if strd != 1 else nn.Identity(),
            )
        elif kargs['conv_type'] == 'conv':
            self.upconv = nn.ConvTranspose2d(ngf, new_ngf, ks + strd, strd, ceil(ks / 2))
        elif kargs['conv_type'] == 'interpolate':
            self.upconv = nn.Sequential(
                nn.Upsample(scale_factor=strd, mode='bilinear', ),
                nn.Conv2d(ngf, new_ngf, strd + ks, 1, ceil((ks + strd - 1) / 2), bias=kargs['bias'])
            )

    def forward(self, x):
        return self.upconv(x)


class ModConv(nn.Module):
    def __init__(self, **kargs):
        super(ModConv, self).__init__()
        mod_ks, mod_groups, ngf = kargs['mod_ks'], kargs['mod_groups'], kargs['ngf']
        self.mod_conv_multi = nn.Conv2d(ngf, ngf, mod_ks, 1, (mod_ks - 1) // 2,
                                        groups=(ngf if mod_groups == -1 else mod_groups))
        self.mod_conv_sum = nn.Conv2d(ngf, ngf, mod_ks, 1, (mod_ks - 1) // 2,
                                      groups=(ngf if mod_groups == -1 else mod_groups))

    def forward(self, x):
        sum_att = self.mod_conv_sum(x)
        multi_att = self.mod_conv_multi(x)
        return torch.sigmoid(multi_att) * x + sum_att


class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )


class HighEncoder(nn.Module):
    def __init__(self, input_channels=64,output_channel=2):
        super(HighEncoder, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(3, input_channels, kernel_size=2, stride=2),LayerNorm(input_channels, eps=1e-6, data_format="channels_first"),Block(input_channels, drop_path=0.0,layer_scale_init_value=1e-6))
        self.conv1 = nn.Sequential(LayerNorm(input_channels, eps=1e-6, data_format="channels_first"),nn.Conv2d(input_channels, input_channels, kernel_size=2, stride=2),Block(input_channels, drop_path=0.0,layer_scale_init_value=1e-6))
        self.conv2 = nn.Sequential(LayerNorm(input_channels*2, eps=1e-6, data_format="channels_first"),nn.Conv2d(input_channels*2, input_channels, kernel_size=2, stride=2),Block(input_channels, drop_path=0.0,layer_scale_init_value=1e-6))
        self.conv3 = nn.Sequential(LayerNorm(input_channels*2, eps=1e-6, data_format="channels_first"),nn.Conv2d(input_channels*2, input_channels, kernel_size=2, stride=2),Block(input_channels, drop_path=0.0,layer_scale_init_value=1e-6))
        self.conv5 = nn.Sequential(LayerNorm(input_channels*2, eps=1e-6, data_format="channels_first"),nn.Conv2d(input_channels*2, out_channels=output_channel, kernel_size=2, stride=2),Block(output_channel, drop_path=0.0,layer_scale_init_value=1e-6))
        self.apply(self._init_weights)
        self.pool3 = WavePool(3)
        self.pool64 = WavePool(64)
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x= self.conv0(x)
        LL, LH, HL, HH = self.pool64(x)
        x1 = self.conv1(LH + HL + HH)
        LL, LH, HL, HH = self.pool64(LL)
        x = self.conv2(torch.cat([x1,LH + HL + HH],dim=1))
        LL, LH, HL, HH = self.pool64(LL)
        x = self.conv3(torch.cat([x,LH + HL + HH],dim=1))
        LL, LH, HL, HH = self.pool64(LL)
        x = self.conv5(torch.cat([x,LH + HL + HH],dim=1))
        return x

class NeRVBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        conv = UpConv if kargs['dec_block'] else DownConv
        self.conv = conv(ngf=kargs['ngf'], new_ngf=kargs['new_ngf'], strd=kargs['strd'], ks=kargs['ks'],
            conv_type=kargs['conv_type'], bias=kargs['bias'])
        self.norm = NormLayer(kargs['norm'], kargs['new_ngf'])
        self.act = ActivationLayer(kargs['act'])

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

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
    def __init__(self, in_ch, cond_ch, factor=1, k=1):
        super().__init__()
        self.k=k
        self.sft0 = SFTLayer(cond_ch, in_ch, factor)
        self.ffn = FeedForward(in_ch)
    def forward(self, x):
        cont=x[0]
        embed = F.interpolate(x[1], scale_factor=self.k, mode='bicubic', align_corners=False)
        fea = self.sft0((cont,embed))
        fea = self.ffn(fea)
        return x[0] + fea,embed
class Harmonic_NeRVBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()

        ngf, new_ngf, stride, ks ,k = kargs['ngf'], kargs['new_ngf'], kargs['strd'], kargs['ks'],kargs['k']
        self.conv = nn.Conv2d(ngf, new_ngf * stride * stride, (ks,ks), 1, ceil((ks - 1) // 2), bias=True)
        self.up_scale = nn.PixelShuffle(stride)
        self.act = ActivationLayer(kargs['act'])
        self.sft_block = ResBlock_SFT(new_ngf, new_ngf, cond_ch=32,k=k,in_act="relu", out_act="gelu", omega=1, args=None)
    def forward(self, x):
        embed = x[1]
        x = self.conv(x[0])
        x = self.up_scale(x)
        x = self.act(x)
        y,embed = self.sft_block((x, embed))
        return y,embed,x

class Decoder_first(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        self.conv = nn.Conv2d(kargs['ngf'], kargs['new_ngf'], kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.act = ActivationLayer(kargs['act'])
    def forward(self,x):
        x = self.conv(x)
        x = self.act(x)
        return x

class SFTLayer(nn.Module):
    def __init__(self, in_ch, out_ch, factor=1):
        super().__init__()
        self.SFT_scale_conv0 = nn.Conv2d(in_ch, in_ch//factor, 1)
        self.SFT_scale_conv1 = nn.Conv2d(in_ch//factor, out_ch, 1)
        self.SFT_shift_conv0 = nn.Conv2d(in_ch, in_ch//factor, 1)
        self.SFT_shift_conv1 = nn.Conv2d(in_ch//factor, out_ch, 1)
        self.act = nn.ReLU(True)
    def forward(self, x):
        scale = self.SFT_scale_conv1(self.act(self.SFT_scale_conv0(x[1])))
        shift = self.SFT_shift_conv1(self.act(self.SFT_shift_conv0(x[1])))
        return x[0] * (scale + 1) + shift

class FEHNeRV(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed = args.embed
        ks_enc, ks_dec1, ks_dec2 = [int(x) for x in args.ks.split('_')]
        enc_blks, dec_blks = [int(x) for x in args.num_blks.split('_')]
        enc_dim1, enc_dim2 = [int(x) for x in args.enc_dim.split('_')]
        c_in_list, c_out_list = [enc_dim1] * len(args.enc_strds), [enc_dim1] * len(args.enc_strds)
        c_out_list[-1] = enc_dim2

        self.encoder = ConvNeXt(stage_blocks=enc_blks, strds=args.enc_strds, dims=c_out_list, drop_path_rate=0)
        self.highencoder = HighEncoder()
        hnerv_hw = np.prod(args.enc_strds) // np.prod(args.dec_strds)
        self.fc_h, self.fc_w = hnerv_hw, hnerv_hw
        ch_in = enc_dim2

        # BUILD Decoder LAYERS
        decoder_layers = []
        ngf = args.fc_dim
        out_f = int(ngf * self.fc_h * self.fc_w)
        decoder_layer1 = Harmonic_NeRVBlock(dec_block=False, conv_type='conv', ngf=ch_in, new_ngf=out_f, ks=0, strd=1,
                                   bias=True, norm=args.norm, act=args.act)
        decoder_layers.append(decoder_layer1)
        high_dec_strds = [1,4,4,2,2]
        for i, strd in enumerate(args.dec_strds):
            reduction = args.reduce
            new_ngf = int(max(round(ngf / reduction), args.lower_width))
            for j in range(dec_blks):
                cur_blk = Harmonic_NeRVBlock(ngf=ngf, new_ngf=new_ngf,ks=min(ks_dec1 + 2 * i, ks_dec2), strd=1 if j else strd, bias=True,
                                     norm='none',
                                     act='gelu', k=high_dec_strds[i])
                decoder_layers.append(cur_blk)
                ngf = new_ngf
        self.stem_high = nn.Sequential(
            nn.Conv2d(args['high_ch_in'], args['high_embed_dim'], 1, bias=True),
            ActivationLayer(args['act'])
        )
        decoder_layers.append(self.stem_high)
        self.decoder = nn.ModuleList(decoder_layers)
        self.head_layer = nn.Conv2d(new_ngf, 3, 3, 1, 1)
        self.out_bias = 'tanh'

    def forward(self, input):
        img_embed = self.encoder(input)
        high_embed = self.highencoder(input)
        high_embed = self.stem_high(high_embed)
        output = self.decoder[0](img_embed)
        for i, layer in enumerate(self.decoder[1:6]):
                output,embed,x = layer((output,high_embed))
                high_embed = embed
        img_out = OutImg(self.head_layer(output), self.out_bias)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return img_out
    def decoder_params(self):
        decoder_param = (sum([p.data.nelement() for p in self.parameters()]) - sum([p.data.nelement() for p in self.encoder.parameters()])-sum([p.data.nelement() for p in self.highencoder.parameters()])) /1e6
        return decoder_param

class FEHNeRVDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fc_h, self.fc_w = [torch.tensor(x) for x in [model.fc_h, model.fc_w]]
        self.out_bias = model.out_bias
        self.decoder = model.decoder
        self.head_layer = model.head_layer

    def forward(self, img_embed):
        output = self.decoder[0](img_embed)
        n, c, h, w = output.shape
        output = output.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,-1,self.fc_h * h, self.fc_w * w)
        for layer in self.decoder[1:]:
            output = layer(output) 
        output = self.head_layer(output)

        return OutImg(output, self.out_bias)




###################################  Tranform input for denoising or inpainting   ###################################
def RandomMask(height, width, points_num, scale=(0, 1)):
    polygon = [(x, y) for x,y in zip(np.random.randint(height * scale[0], height * scale[1], size=points_num), 
                             np.random.randint(width * scale[0], width * scale[1], size=points_num))]
    poly_path=Path(polygon)

    x, y = np.mgrid[:height, :width]
    coors=np.hstack((x.reshape(-1, 1), y.reshape(-1,1))) # coors.shape is (4000000,2)
    mask = poly_path.contains_points(coors).reshape(height, width)
    return 1 - torch.from_numpy(mask).float()


class TransformInput(nn.Module):
    def __init__(self, args):
        super(TransformInput, self).__init__()
        self.vid = args.vid
        if 'inpaint' in self.vid:
            self.inpaint_size = int(self.vid.split('_')[-1]) // 2

    def forward(self, img):
        inpaint_mask = torch.ones_like(img)
        if 'inpaint' in self.vid:
            gt = img.clone()
            h,w = img.shape[-2:]
            inpaint_mask = torch.ones((h,w)).to(img.device)
            for ctr_x, ctr_y in [(1/2, 1/2), (1/4, 1/4), (1/4, 3/4), (3/4, 1/4), (3/4, 3/4)]:
                ctr_x, ctr_y = int(ctr_x * h), int(ctr_y * w)
                inpaint_mask[ctr_x - self.inpaint_size: ctr_x + self.inpaint_size, ctr_y - self.inpaint_size: ctr_y + self.inpaint_size] = 0
            input = (img * inpaint_mask).clamp(min=0,max=1)
        else:
            input, gt = img, img

        return input, gt, inpaint_mask.detach()


###################################  Code for ConvNeXt   ###################################
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, stage_blocks=0, strds=[2,2,2,2], dims=[96, 192, 384, 768], 
            in_chans=3, drop_path_rate=0., layer_scale_init_value=1e-6,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stage_num = len(dims)
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, stage_blocks*self.stage_num)] 
        cur = 0
        for i in range(self.stage_num):
            # Build downsample layers
            if i > 0:
                downsample_layer = nn.Sequential(
                        LayerNorm(dims[i-1], eps=1e-6, data_format="channels_first"),
                        nn.Conv2d(dims[i-1], dims[i], kernel_size=strds[i], stride=strds[i]),
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
        return out_list[-1]


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
