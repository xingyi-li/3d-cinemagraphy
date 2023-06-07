# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [GN] => PReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, group=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        # if not group:
        #    group = out_channels//16
        self.double_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.GroupNorm(mid_channels // 32, mid_channels),
            nn.PReLU()
        )
        self.double_conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.GroupNorm(out_channels // 32, out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        x1 = self.double_conv1(x)
        x2 = self.double_conv2(x1)
        return x1, x2


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, group=None):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, group)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class ConcatDoubleConv(nn.Module):
    """(convolution => [GN] => PReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, group=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        # if not group:
        #    group = out_channels//16
        self.double_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.GroupNorm(mid_channels // 32, mid_channels),
            nn.PReLU()
        )
        self.double_conv2 = nn.Sequential(
            nn.Conv2d(mid_channels * 2, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.GroupNorm(out_channels // 32, out_channels),
            nn.PReLU()
        )

    def forward(self, x, xc1):
        x1 = self.double_conv1(x)
        x2 = self.double_conv2(torch.cat([xc1, x1], dim=1))
        return x2


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, mid_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConcatDoubleConv(in_channels, out_channels, mid_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConcatDoubleConv(in_channels, out_channels, mid_channels)

    def forward(self, x, xc1, xc2):
        x1 = self.up(x)
        # input is CHW
        diffY = xc1.size()[2] - x1.size()[2]
        diffX = xc1.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2], mode='reflect')
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([xc1, x1], dim=1)
        return self.conv(x, xc2)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        # self.gn = nn.GroupNorm(1, out_channels)
        self.act = nn.Sigmoid()

    def forward(self, x, xc1):
        x1 = self.up(x)
        x2 = torch.cat([xc1, x1], dim=1)
        # return self.act(self.gn(self.conv(x2)))
        return self.act(self.conv(x2))


class ImgDecoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ImgDecoder, self).__init__()
        self.firstconv = nn.Conv2d(in_ch, 32, kernel_size=7, stride=1, padding=3, bias=False)  # 256x256
        self.firstprelu = nn.PReLU()
        self.down1 = Down(32, 64)  # 128x128
        self.down2 = Down(64, 128)  # 64x64
        self.down3 = Down(128, 256)  # 32x32
        self.down4 = Down(256, 512)  # 16x16
        self.conv5 = DoubleConv(512, 512)  # 16x16

        self.up1 = Up(1024, 256, 512)
        self.up2 = Up(512, 128, 256)
        self.up3 = Up(256, 64, 128)
        self.up4 = Up(128, 32, 64)
        self.outc = OutConv(64, out_ch)
        self.alpha = nn.Parameter(torch.tensor(1.))

    def compute_weight_for_two_frame_blending(self, time, disp1, disp2, alpha0, alpha1):
        weight1 = (1 - time) * torch.exp(self.alpha * disp1) * alpha0
        weight2 = time * torch.exp(self.alpha * disp2) * alpha1
        sum_weight = torch.clamp(weight1 + weight2, min=1e-6)
        out_weight1 = weight1 / sum_weight
        out_weight2 = weight2 / sum_weight
        return out_weight1, out_weight2

    def forward(self, x0, x1, time):
        disp0 = x0[:, [-1]]
        disp1 = x1[:, [-1]]
        alpha0 = x0[:, [3]]
        alpha1 = x1[:, [3]]
        w0, w1 = self.compute_weight_for_two_frame_blending(time, disp0, disp1, alpha0, alpha1)
        x = w0 * x0 + w1 * x1

        x0 = self.firstprelu(self.firstconv(x))
        x20, x21 = self.down1(x0)
        x30, x31 = self.down2(x21)
        x40, x41 = self.down3(x31)
        x50, x51 = self.down4(x41)
        x60, x61 = self.conv5(x51)

        xt1 = self.up1(x61, x51, x50)
        xt2 = self.up2(xt1, x41, x40)
        xt3 = self.up3(xt2, x31, x30)
        xt4 = self.up4(xt3, x21, x20)
        target_img = self.outc(xt4, x0)

        return target_img
