import torch.nn as nn
from lib.model.motion.layers.normalization import LinearNoiseLayer
from lib.model.motion.layers.normalization import PartialLinearNoiseLayer
from lib.model.motion.layers.partialconv2d import PartialConv2d


def spectral_conv_function(in_c, out_c, k, p, s, dilation=1, bias=True):
    return nn.utils.spectral_norm(
        nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, stride=s, dilation=dilation, bias=bias)
    )


def conv_function(in_c, out_c, k, p, s, dilation=1, bias=True):
    return nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, stride=s, dilation=dilation, bias=bias)


def spectral_pconv_function(in_c, out_c, k, p, s, dilation=1):
    return nn.utils.spectral_norm(
        PartialConv2d(in_c, out_c, kernel_size=k, padding=p, stride=s, dilation=dilation,
                      bias=True, multi_channel=True, return_mask=True)
    )


def pconv_function(in_c, out_c, k, p, s, dilation=1):
    return PartialConv2d(in_c, out_c, kernel_size=k, padding=p, stride=s, dilation=dilation,
                         bias=True, multi_channel=True, return_mask=True)


def get_conv_layer(opt):
    if "spectral" in opt.norm_G:
        conv_layer_base = spectral_conv_function
    else:
        conv_layer_base = conv_function

    return conv_layer_base


def get_pconv_layer(opt):
    if "spectral" in opt.norm_G:
        conv_layer_base = spectral_pconv_function
    else:
        conv_layer_base = pconv_function
    return conv_layer_base


# Convenience passthrough function
class Identity(nn.Module):
    def forward(self, input):
        return input


# ResNet Blocks
class ResNet_Block(nn.Module):
    def __init__(self, in_c, in_o, opt, downsample=None):
        super().__init__()
        bn_noise1 = LinearNoiseLayer(opt, output_sz=in_c)
        bn_noise2 = LinearNoiseLayer(opt, output_sz=in_o)

        conv_layer = get_conv_layer(opt)

        conv_aa = conv_layer(in_c, in_o, 3, 1, 1)
        conv_ab = conv_layer(in_o, in_o, 3, 1, 1)

        conv_b = conv_layer(in_c, in_o, 1, 0, 1)

        if downsample == "Down":
            norm_downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        elif downsample == "Up":
            norm_downsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        elif downsample:
            norm_downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            norm_downsample = Identity()

        self.ch_a = nn.Sequential(
            bn_noise1,
            nn.ReLU(),
            conv_aa,
            bn_noise2,
            nn.ReLU(),
            conv_ab,
            norm_downsample,
        )
        if downsample or (in_c != in_o):
            self.ch_b = nn.Sequential(conv_b, norm_downsample)
        else:
            self.ch_b = Identity()

    def forward(self, x):
        x_a = self.ch_a(x)
        x_b = self.ch_b(x)

        return x_a + x_b


# ResNet Blocks
class ResNet_Block_Pconv(nn.Module):
    def __init__(self, in_c, in_o, opt, downsample=None, ks=3, activation='Relu', padding=1, stride=1, dilation=1):
        super().__init__()
        if "pbn" in opt.pconv:
            self.bn_noise1 = PartialLinearNoiseLayer(opt, output_sz=in_c)
            self.bn_noise2 = PartialLinearNoiseLayer(opt, output_sz=in_o)
            self.pconv_opt = 'pbn'
        else:
            self.bn_noise1 = LinearNoiseLayer(opt, output_sz=in_c)
            self.bn_noise2 = LinearNoiseLayer(opt, output_sz=in_o)
            self.pconv_opt = 'bn'

        pconv_layer = get_pconv_layer(opt)
        conv_layer = get_conv_layer(opt)

        self.conv_aa = pconv_layer(in_c, in_o, ks, padding, stride, dilation)
        self.conv_ab = pconv_layer(in_o, in_o, ks, padding, stride, dilation)

        if "woresbias" in opt.pconv:
            self.conv_b = conv_layer(in_c, in_o, 1, 0, 1, bias=False)
        else:
            self.conv_b = conv_layer(in_c, in_o, 1, 0, 1)
        if downsample == "Down":
            # self.norm_downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.norm_downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif downsample == "Up":
            # self.norm_downsample = nn.Upsample(scale_factor=2, mode="bilinear")
            self.norm_downsample = nn.Upsample(scale_factor=2, mode="nearest")
        elif downsample:
            # self.norm_downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.norm_downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.norm_downsample = Identity()
        self.downsample = downsample
        self.in_c = in_c
        self.in_o = in_o
        self.debug_opt = opt.pconv
        if activation == "Relu":
            self.activation = nn.ReLU()
        elif activation == "LRelu":
            self.activation = nn.LeakyReLU(0.2)
        elif activation == "PRelu":
            self.activation = nn.PReLU()
        elif activation == "None":
            self.activation = Identity()
        elif activation == False:
            self.activation = Identity()
        else:
            self.activation = nn.ReLU()

    def forward(self, x, mask):
        if "debug" in self.debug_opt:
            print("mask ratio:{}.".format(mask.sum() / (mask.shape[0] * mask.shape[1] * mask.shape[2] * mask.shape[3])))
        # x: [1, 64, 256, 256]
        # mask: [1, 64, 256, 256]
        # x_a
        if "pbn" in self.pconv_opt:
            x_a, mask_a = self.bn_noise1(x, mask)  # [1, 64, 256, 256]
        else:
            x_a = self.bn_noise1(x)
            mask_a = mask

        x_a = self.activation(x_a)
        x_a, mask_a = self.conv_aa(x_a, mask_a)

        if "pbn" in self.pconv_opt:
            x_a, mask_a = self.bn_noise2(x_a, mask_a)
        else:
            x_a = self.bn_noise2(x_a)

        x_a = self.activation(x_a)
        x_a, mask_a = self.conv_ab(x_a, mask_a)
        x_a = self.norm_downsample(x_a)
        mask_a = self.norm_downsample(mask_a)
        # x_b
        if self.downsample or (self.in_c != self.in_o):
            x_b = self.conv_b(x)
            x_b = self.norm_downsample(x_b)
        else:
            x_b = Identity()(x)
        return x_a + x_b, mask_a


# ResNet Blocks
class ResNet_Block_Pconv2(nn.Module):
    def __init__(self, in_c, in_o, opt, downsample=None, ks=3, activation='Relu', padding=1, stride=1, dilation=1):
        super().__init__()
        if "pbn" in opt.pconv:
            self.bn_noise1 = PartialLinearNoiseLayer(opt, output_sz=in_c)
            self.bn_noise2 = PartialLinearNoiseLayer(opt, output_sz=in_o)
            self.pconv_opt = 'pbn'
        else:
            self.bn_noise1 = LinearNoiseLayer(opt, output_sz=in_c)
            self.bn_noise2 = LinearNoiseLayer(opt, output_sz=in_o)
            self.pconv_opt = 'bn'

        pconv_layer = get_pconv_layer(opt)
        conv_layer = get_conv_layer(opt)

        self.conv_aa = pconv_layer(in_c, in_o, ks, padding, stride, dilation)
        self.conv_ab = pconv_layer(in_o, in_o, ks, padding, stride, dilation)

        if "woresbias" in opt.pconv:
            self.conv_b = conv_layer(in_c, in_o, 1, 0, 1, bias=False)
        else:
            self.conv_b = conv_layer(in_c, in_o, 1, 0, 1)
        if downsample == "Down":
            self.norm_downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.norm_downsample_mask = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif downsample == "Up":
            self.norm_downsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.norm_downsample_mask = nn.Upsample(scale_factor=2, mode="nearest")
        else:
            self.norm_downsample = Identity()
            self.norm_downsample_mask = Identity()
        self.downsample = downsample
        self.in_c = in_c
        self.in_o = in_o
        self.debug_opt = opt.pconv
        if activation == "Relu":
            self.activation = nn.ReLU()
        elif activation == "LRelu":
            self.activation = nn.LeakyReLU(0.2)
        elif activation == "PRelu":
            self.activation = nn.PReLU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x, mask):
        if "debug" in self.debug_opt:
            print("mask ratio:{}.".format(mask.sum() / (mask.shape[0] * mask.shape[1] * mask.shape[2] * mask.shape[3])))
        # x: [1, 64, 256, 256]
        # mask: [1, 64, 256, 256]
        # x_a
        if "pbn" in self.pconv_opt:
            x_a, mask_a = self.bn_noise1(x, mask)  # [1, 64, 256, 256]
        else:
            x_a = self.bn_noise1(x)
            mask_a = mask

        x_a = self.activation(x_a)
        x_a, mask_a = self.conv_aa(x_a, mask_a)

        if "pbn" in self.pconv_opt:
            x_a, mask_a = self.bn_noise2(x_a, mask_a)
        else:
            x_a = self.bn_noise2(x_a)

        x_a = self.activation(x_a)
        x_a, mask_a = self.conv_ab(x_a, mask_a)
        x_a = self.norm_downsample(x_a)
        mask_a = self.norm_downsample_mask(mask_a)
        # x_b
        if self.downsample or (self.in_c != self.in_o):
            x_b = self.conv_b(x)
            x_b = self.norm_downsample(x_b)
        else:
            x_b = Identity()(x)
        return x_a + x_b, mask_a


# ResNet Blocks
class ResNet_Block_Pconv3(nn.Module):
    def __init__(self, in_c, in_o, opt, downsample=None, ks=3, activation=None):
        super().__init__()
        if "pbn" in opt.pconv:
            self.bn_noise1 = PartialLinearNoiseLayer(opt, output_sz=in_c)
            self.bn_noise2 = PartialLinearNoiseLayer(opt, output_sz=in_o)
            self.pconv_opt = 'pbn'
        else:
            self.bn_noise1 = LinearNoiseLayer(opt, output_sz=in_c)
            self.bn_noise2 = LinearNoiseLayer(opt, output_sz=in_o)
            self.pconv_opt = 'bn'

        pconv_layer = get_pconv_layer(opt)
        conv_layer = get_conv_layer(opt)

        if downsample == "Up":
            self.conv_aa = pconv_layer(in_c, in_o, ks, 1, 1)
            self.conv_ab = pconv_layer(in_o, in_o * 4, ks, 1, 1)
        else:
            self.conv_aa = pconv_layer(in_c, in_o, ks, 1, 1)
            self.conv_ab = pconv_layer(in_o, in_o, ks, 1, 1)

        if "woresbias" in opt.pconv:
            if downsample == "Up":
                self.conv_b = conv_layer(in_c, in_o * 4, 1, 0, 1, bias=False)
            else:
                self.conv_b = conv_layer(in_c, in_o, 1, 0, 1, bias=False)
        else:
            if downsample == "Up":
                self.conv_b = conv_layer(in_c, in_o * 4, 1, 0, 1)
            else:
                self.conv_b = conv_layer(in_c, in_o, 1, 0, 1)
        if downsample == "Down":
            self.norm_downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.norm_downsample_mask = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif downsample == "Up":
            self.norm_downsample = nn.PixelShuffle(2)
            self.norm_downsample_mask = nn.PixelShuffle(2)
            # self.norm_downsample = nn.Upsample(scale_factor=2, mode="bilinear")
            # self.norm_downsample_mask = nn.Upsample(scale_factor=2, mode="nearest")
        else:
            self.norm_downsample = Identity()
            self.norm_downsample_mask = Identity()
        self.downsample = downsample
        self.in_c = in_c
        self.in_o = in_o
        self.debug_opt = opt.pconv
        if activation == "Relu":
            self.activation = nn.ReLU()
        elif activation == "LRelu":
            self.activation = nn.LeakyReLU(0.2)
        elif activation == "PRelu":
            self.activation = nn.PReLU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x, mask):
        if "debug" in self.debug_opt:
            print("mask ratio:{}.".format(mask.sum() / (mask.shape[0] * mask.shape[1] * mask.shape[2] * mask.shape[3])))
        # x: [1, 64, 256, 256]
        # mask: [1, 64, 256, 256]
        # x_a
        if "pbn" in self.pconv_opt:
            x_a, mask_a = self.bn_noise1(x, mask)  # [1, 64, 256, 256]
        else:
            x_a = self.bn_noise1(x)
            mask_a = mask

        x_a = self.activation(x_a)
        x_a, mask_a = self.conv_aa(x_a, mask_a)

        if "pbn" in self.pconv_opt:
            x_a, mask_a = self.bn_noise2(x_a, mask_a)
        else:
            x_a = self.bn_noise2(x_a)

        x_a = self.activation(x_a)
        x_a, mask_a = self.conv_ab(x_a, mask_a)
        x_a = self.norm_downsample(x_a)
        mask_a = self.norm_downsample_mask(mask_a)

        # x_b
        if self.downsample or (self.in_c != self.in_o):
            x_b = self.conv_b(x)
            x_b = self.norm_downsample(x_b)
        else:
            x_b = Identity()(x)
        return x_a + x_b, mask_a
