import torch
import torch.nn as nn
import torchvision
import numpy as np
from lib.model.motion.layers.blocks import ResNet_Block
from lib.model.motion.layers.blocks import ResNet_Block_Pconv
from lib.model.motion.layers.blocks import ResNet_Block_Pconv2
from lib.model.motion.networks import SPADE

from lib.model.motion.layers.normalization import BatchNorm_StandingStats
from lib.model.motion.configs import get_resnet_arch

from lib.model.motion.layers.normalization import LinearNoiseLayer
from lib.model.motion.layers.normalization import PartialLinearNoiseLayer
from lib.model.motion.layers.partialconv2d import PartialConv2d
import torch.nn.functional as F


def spectral_conv_function(in_c, out_c, k, p, s, bias=True):
    return nn.utils.spectral_norm(
        nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, stride=s, bias=bias)
    )


def conv_function(in_c, out_c, k, p, s, bias=True):
    return nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, stride=s, bias=bias)


def spectral_pconv_function(in_c, out_c, k, p, s):
    return nn.utils.spectral_norm(
        PartialConv2d(in_c, out_c, kernel_size=k, padding=p, stride=s,
                      bias=True, multi_channel=True, return_mask=True)
    )


def pconv_function(in_c, out_c, k, p, s):
    return PartialConv2d(in_c, out_c, kernel_size=k, padding=p, stride=s,
                         bias=True, multi_channel=True, return_mask=True)


def get_conv_layer(config, use_3D=False):
    if "spectral" in config['norm_G']:
        if use_3D:
            conv_layer_base = lambda in_c, out_c, k, s, p: nn.utils.spectral_norm(
                nn.Conv3d(in_c, out_c, kernel_size=k, padding=p, stride=s)
            )
        else:
            conv_layer_base = lambda in_c, out_c, k, s, p: nn.utils.spectral_norm(
                nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, stride=s)
            )
    else:
        if use_3D:
            conv_layer_base = lambda in_c, out_c, k, s, p: nn.Conv3d(
                in_c, out_c, kernel_size=k, padding=p, stride=s
            )
        else:
            conv_layer_base = lambda in_c, out_c, k, s, p: nn.Conv2d(
                in_c, out_c, kernel_size=k, padding=p, stride=s
            )

    return conv_layer_base


def get_pconv_layer(opt):
    if "spectral" in opt.norm_G:
        conv_layer_base = spectral_pconv_function
    else:
        conv_layer_base = pconv_function
    return conv_layer_base


def get_batchnorm_layer(opt):
    norm_G = opt.norm_G.split(":")[1]
    if norm_G == "batch":
        norm_layer = nn.BatchNorm2d
    elif norm_G == "spectral_instance":
        norm_layer = nn.InstanceNorm2d
    elif norm_G == "spectral_batch":
        norm_layer = nn.BatchNorm2d
    elif norm_G == "spectral_batchstanding":
        norm_layer = BatchNorm_StandingStats

    return norm_layer


# VGG architecture, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(
            pretrained=True
        ).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # Normalize the image so that it is in the appropriate range
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


#######################################################################################
#       ResNet models
#######################################################################################
class ResNetEncoder(nn.Module):
    """ Modified implementation of the BigGAN model.
    """

    def __init__(
            self,
            opt,
            channels_in=3,
            channels_out=64,
            downsample=True,
            model_type=None,
    ):
        super().__init__()
        if not model_type:
            arch = get_resnet_arch(opt.refine_model_type, opt, channels_in)
            # opt.refine_model_type == 'resnet_256W8UpDown64'
        else:
            arch = get_resnet_arch(model_type, opt, channels_in)

        gblocks = []
        for l_id in range(1, len(arch["layers_enc"])):
            gblock = ResNet_Block(
                arch["layers_enc"][l_id - 1],
                arch["layers_enc"][l_id],
                opt,
                (downsample and arch["downsample"][l_id - 1]),
            )
            gblocks += [gblock]

        self.gblocks = nn.Sequential(*gblocks)

    def forward(self, x):
        return self.gblocks(x)


class ResNetEncoder_with_Z(nn.Module):
    """ Modified implementation of the BigGAN model.
    """

    def __init__(
            self,
            opt,
            channels_in=3,
            channels_out=64,
            downsample=True,
            model_type=None,
    ):
        super().__init__()
        if not model_type:
            arch = get_resnet_arch(opt.refine_model_type, opt, channels_in)
            # opt.refine_model_type == 'resnet_256W8UpDown64'
        else:
            arch = get_resnet_arch(model_type, opt, channels_in)

        gblocks = []
        for l_id in range(1, len(arch["layers_enc"]) - 1):
            gblock = ResNet_Block(
                arch["layers_enc"][l_id - 1],
                arch["layers_enc"][l_id],
                opt,
                (downsample and arch["downsample"][l_id - 1]),
            )
            gblocks += [gblock]

        gblock = ResNet_Block(
            arch["layers_enc"][len(arch["layers_enc"]) - 2],
            arch["layers_enc"][len(arch["layers_enc"]) - 1] + 1,
            opt,
            (downsample and arch["downsample"][len(arch["layers_enc"]) - 2]),
        )
        gblocks += [gblock]

        self.gblocks = nn.Sequential(*gblocks)
        self.opt = opt

    def forward(self, x):
        output = self.gblocks(x)
        return output[:, :-1, ...], output[:, -1:, ...]


class Exp(nn.Module):
    def forward(self, input):
        return input.exp()


class Identity(nn.Module):
    def forward(self, input):
        return input


class ResNetDecoder(nn.Module):
    """ Modified implementation of the BigGAN model. """

    def __init__(self, opt, channels_in=64, channels_out=3, use_tanh=False):
        super().__init__()

        arch = get_resnet_arch(opt.refine_model_type, opt)

        eblocks = []
        for l_id in range(1, len(arch["layers_dec"])):
            eblock = ResNet_Block(
                arch["layers_dec"][l_id - 1],
                arch["layers_dec"][l_id],
                opt,
                arch["upsample"][l_id - 1],
            )
            eblocks += [eblock]

        self.eblocks = nn.Sequential(*eblocks)

    def forward(self, x):
        return self.eblocks(x)


class ResNetBGDecoder(nn.Module):
    """ Modified implementation of the BigGAN model. """

    def __init__(self, opt, channels_in=64, channels_out=3, use_tanh=True):
        super().__init__()

        arch = get_resnet_arch(opt.bg_refine_model_type, opt)

        eblocks = []
        for l_id in range(1, len(arch["layers_dec"])):
            eblock = ResNet_Block(
                arch["layers_dec"][l_id - 1],
                arch["layers_dec"][l_id],
                opt,
                arch["upsample"][l_id - 1],
            )
            eblocks += [eblock]

        self.eblocks = nn.Sequential(*eblocks)

        if use_tanh:
            self.norm = nn.Tanh()
        else:
            self.norm = Identity()

    def forward(self, x):
        # mask = (x != 0).float()
        return self.norm(self.eblocks(x))


class ResBlockAlphaDecoder(nn.Module):
    def __init__(self, opt, channels_in=64, channels_out=1, use_sigmoid=True, use_relu=False, use_tanh=False):
        super().__init__()
        self.block = ResNet_Block(channels_in,
                                  channels_out,
                                  opt,
                                  )
        if use_sigmoid:
            self.norm = nn.Sigmoid()
        if use_relu:
            self.norm = nn.ReLU()
        if use_tanh:
            self.norm = nn.Tanh()

    def forward(self, x):
        return self.norm(self.block(x))


class ResNetDecoderAlpha(nn.Module):
    """ Modified implementation of the BigGAN model. """

    def __init__(self, opt, channels_in=64, channels_out=3, use_sigmoid=True):
        super().__init__()

        arch = get_resnet_arch("resnet_256W8UpDownAlpha", opt)

        eblocks = []
        for l_id in range(1, len(arch["layers_dec"])):
            eblock = ResNet_Block(
                arch["layers_dec"][l_id - 1],
                arch["layers_dec"][l_id],
                opt,
                arch["upsample"][l_id - 1],
            )
            eblocks += [eblock]

        self.eblocks = nn.Sequential(*eblocks)

        if use_sigmoid:
            self.norm = nn.Sigmoid()

    def forward(self, x):
        # mask = (x != 0).float()
        return self.norm(self.eblocks(x))


class ResNetDecoderPconv(nn.Module):
    """ Modified implementation of the BigGAN model. """

    def __init__(self, opt, channels_in=64, channels_out=3, use_tanh=True):
        super().__init__()

        arch = get_resnet_arch(opt.refine_model_type, opt)

        eblocks = []
        for l_id in range(1, len(arch["layers_dec"])):
            eblock = ResNet_Block_Pconv(
                arch["layers_dec"][l_id - 1],  # opt.ngf,
                arch["layers_dec"][l_id],  # opt.ngf*2
                opt,
                arch["upsample"][l_id - 1],  # "Down"
                arch["ks_dec"][l_id - 1] if "ks_dec" in arch else 3,
                arch["padding_dec"][l_id - 1] if "padding_dec" in arch else 1,
                arch["stride_dec"][l_id - 1] if "stride_dec" in arch else 1,
                arch["activation"][l_id - 1] if "activation" in arch else None,
            )
            eblocks += [eblock]
        self.eblocks = nn.Sequential(*eblocks)

        if use_tanh:
            self.norm = nn.Tanh()
        self.pconv_setting = opt.pconv

    def forward(self, x):
        mask = (x != 0).float()
        if "mask1" in self.pconv_setting:
            mask[:] = 1.0
        x_a, mask_a = self.eblocks[0](x, mask)
        for eblock in self.eblocks[1:]:
            x_a, mask_a = eblock(x_a, mask_a)
        return self.norm(x_a)


class ResNetDecoderPconv2(nn.Module):
    """ Modified implementation of the BigGAN model. """

    def __init__(self, opt, channels_in=64, channels_out=3, use_tanh=False):
        super().__init__()

        arch = get_resnet_arch(opt.refine_model_type, opt)

        eblocks = []
        for l_id in range(1, len(arch["layers_dec"])):
            eblock = ResNet_Block_Pconv2(
                arch["layers_dec"][l_id - 1],  # opt.ngf,
                arch["layers_dec"][l_id],  # opt.ngf*2
                opt,
                arch["upsample"][l_id - 1],  # "Down"
                arch["ks_dec"][l_id - 1] if "ks_dec" in arch else 3,
                arch["activation"][l_id - 1] if "activation" in arch else None,
            )
            eblocks += [eblock]
        self.eblocks = nn.Sequential(*eblocks)

        self.pconv_setting = opt.pconv

    def forward(self, x):
        mask = (x != 0).float()
        if "mask1" in self.pconv_setting:
            mask[:] = 1.0
        x_a, mask_a = self.eblocks[0](x, mask)
        for eblock in self.eblocks[1:]:
            x_a, mask_a = eblock(x_a, mask_a)
        return x_a


#######################################################################################
#       UNet models
#######################################################################################

class Unet4Motion(nn.Module):
    """ Reimplementation of Unet that allows for simpler modifications.
    """

    def __init__(
            self,
            num_filters=32,
            channels_in=3,
            channels_out=2,
            use_tanh=False,
            use_3D=False,
            opt=None,
            up_mode="bilinear",
    ):
        super(Unet4Motion, self).__init__()

        conv_layer = get_conv_layer(opt, use_3D=use_3D)

        self.conv1 = conv_layer(channels_in, num_filters, 4, 2, 1)
        self.conv2 = conv_layer(num_filters, num_filters * 2, 4, 2, 1)
        self.conv3 = conv_layer(num_filters * 2, num_filters * 4, 4, 2, 1)
        self.conv4 = conv_layer(num_filters * 4, num_filters * 8, 4, 2, 1)
        self.conv5 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv6 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv7 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv8 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)

        self.up = nn.Upsample(scale_factor=2, mode=up_mode, align_corners=False)

        self.dconv1 = conv_layer(num_filters * 8, num_filters * 8, 3, 1, 1)
        self.dconv2 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv3 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv4 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv5 = conv_layer(num_filters * 8 * 2, num_filters * 4, 3, 1, 1)
        self.dconv6 = conv_layer(num_filters * 4 * 2, num_filters * 2, 3, 1, 1)
        self.dconv7 = conv_layer(num_filters * 2 * 2, num_filters, 3, 1, 1)
        self.dconv8 = conv_layer(num_filters * 2, channels_out, 3, 1, 1)

        norm_layer = get_batchnorm_layer(opt)

        self.batch_norm = norm_layer(num_filters)
        self.batch_norm2_0 = norm_layer(num_filters * 2)
        self.batch_norm2_1 = norm_layer(num_filters * 2)
        self.batch_norm4_0 = norm_layer(num_filters * 4)
        self.batch_norm4_1 = norm_layer(num_filters * 4)
        self.batch_norm8_0 = norm_layer(num_filters * 8)
        self.batch_norm8_1 = norm_layer(num_filters * 8)
        self.batch_norm8_2 = norm_layer(num_filters * 8)
        self.batch_norm8_3 = norm_layer(num_filters * 8)
        self.batch_norm8_4 = norm_layer(num_filters * 8)
        self.batch_norm8_5 = norm_layer(num_filters * 8)
        self.batch_norm8_6 = norm_layer(num_filters * 8)
        self.batch_norm8_7 = norm_layer(num_filters * 8)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.use_tanh = use_tanh

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 256 x 256
        e1 = self.conv1(input)
        # state size is (num_filters) x 128 x 128
        e2 = self.batch_norm2_0(self.conv2(self.leaky_relu(e1)))
        # state size is (num_filters x 2) x 64 x 64
        e3 = self.batch_norm4_0(self.conv3(self.leaky_relu(e2)))
        # state size is (num_filters x 4) x 32 x 32
        e4 = self.batch_norm8_0(self.conv4(self.leaky_relu(e3)))
        # state size is (num_filters x 8) x 16 x 16
        e5 = self.batch_norm8_1(self.conv5(self.leaky_relu(e4)))
        # state size is (num_filters x 8) x 8 x 8
        e6 = self.batch_norm8_2(self.conv6(self.leaky_relu(e5)))
        # state size is (num_filters x 8) x 4 x 4
        e7 = self.batch_norm8_3(self.conv7(self.leaky_relu(e6)))
        # state size is (num_filters x 8) x 2 x 2
        # No batch norm on output of Encoder
        e8 = self.conv8(self.leaky_relu(e7))

        # Decoder
        # Deconvolution layers:
        # state size is (num_filters x 8) x 1 x 1
        d1_ = self.batch_norm8_4(self.dconv1(self.up(self.relu(e8))))
        # state size is (num_filters x 8) x 2 x 2
        d1 = torch.cat((d1_, e7), 1)
        d2_ = self.batch_norm8_5(self.dconv2(self.up(self.relu(d1))))
        # state size is (num_filters x 8) x 4 x 4
        d2 = torch.cat((d2_, e6), 1)
        d3_ = self.batch_norm8_6(self.dconv3(self.up(self.relu(d2))))
        # state size is (num_filters x 8) x 8 x 8
        d3 = torch.cat((d3_, e5), 1)
        d4_ = self.batch_norm8_7(self.dconv4(self.up(self.relu(d3))))
        # state size is (num_filters x 8) x 16 x 16
        d4 = torch.cat((d4_, e4), 1)
        d5_ = self.batch_norm4_1(self.dconv5(self.up(self.relu(d4))))
        # state size is (num_filters x 4) x 32 x 32
        d5 = torch.cat((d5_, e3), 1)
        d6_ = self.batch_norm2_1(self.dconv6(self.up(self.relu(d5))))
        # state size is (num_filters x 2) x 64 x 64
        d6 = torch.cat((d6_, e2), 1)
        d7_ = self.batch_norm(self.dconv7(self.up(self.relu(d6))))
        # state size is (num_filters) x 128 x 128
        # d7_ = torch.Tensor(e1.data.new(e1.size()).normal_(0, 0.5))
        d7 = torch.cat((d7_, e1), 1)
        d8 = self.dconv8(self.up(self.relu(d7)))
        # state size is (nc) x 256 x 256
        if self.use_tanh:
            output = self.tanh(d8)
        else:
            output = d8
        return output


class SPADEUnet4Motion(nn.Module):
    """ Reimplementation of Unet that allows for simpler modifications.
    """

    def __init__(
            self,
            num_filters=32,
            channels_in=3,
            channels_out=2,
            use_3D=False,
            opt=None,
            up_mode="bilinear",
    ):
        super(SPADEUnet4Motion, self).__init__()

        conv_layer = get_conv_layer(opt, use_3D=use_3D)

        self.conv1 = conv_layer(channels_in, num_filters, 4, 2, 1)
        self.conv2 = conv_layer(num_filters, num_filters * 2, 4, 2, 1)
        self.conv3 = conv_layer(num_filters * 2, num_filters * 4, 4, 2, 1)
        self.conv4 = conv_layer(num_filters * 4, num_filters * 8, 4, 2, 1)
        self.conv5 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv6 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv7 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv8 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)

        # self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up = nn.Upsample(scale_factor=2, mode=up_mode, align_corners=False)

        self.dconv1 = conv_layer(num_filters * 8, num_filters * 8, 3, 1, 1)
        self.dconv2 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv3 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv4 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv5 = conv_layer(num_filters * 8 * 2, num_filters * 4, 3, 1, 1)
        self.dconv6 = conv_layer(num_filters * 4 * 2, num_filters * 2, 3, 1, 1)
        self.dconv7 = conv_layer(num_filters * 2 * 2, num_filters, 3, 1, 1)
        self.dconv8 = conv_layer(num_filters * 2, channels_out, 3, 1, 1)

        norm_layer = nn.InstanceNorm2d

        self.spade_layer = SPADE(norm_layer, num_filters, 6)  # self.batch_norm = norm_layer(num_filters)
        self.spade_layer2_0 = SPADE(norm_layer, num_filters * 2, 6)  # self.batch_norm2_0 = norm_layer(num_filters * 2)
        self.spade_layer2_1 = SPADE(norm_layer, num_filters * 2, 6)  # self.batch_norm2_1 = norm_layer(num_filters * 2)
        self.spade_layer4_0 = SPADE(norm_layer, num_filters * 4, 6)  # self.batch_norm4_0 = norm_layer(num_filters * 4)
        self.spade_layer4_1 = SPADE(norm_layer, num_filters * 4, 6)  # self.batch_norm4_1 = norm_layer(num_filters * 4)
        self.spade_layer8_0 = SPADE(norm_layer, num_filters * 8, 6)  # self.batch_norm8_0 = norm_layer(num_filters * 8)
        self.spade_layer8_1 = SPADE(norm_layer, num_filters * 8, 6)  # self.batch_norm8_1 = norm_layer(num_filters * 8)
        self.spade_layer8_2 = SPADE(norm_layer, num_filters * 8, 6)  # self.batch_norm8_2 = norm_layer(num_filters * 8)
        self.spade_layer8_3 = SPADE(norm_layer, num_filters * 8, 6)  # self.batch_norm8_3 = norm_layer(num_filters * 8)
        self.spade_layer8_4 = SPADE(norm_layer, num_filters * 8, 6)  # self.batch_norm8_4 = norm_layer(num_filters * 8)
        self.spade_layer8_5 = SPADE(norm_layer, num_filters * 8, 6)  # self.batch_norm8_5 = norm_layer(num_filters * 8)
        self.spade_layer8_6 = SPADE(norm_layer, num_filters * 8, 6)  # self.batch_norm8_6 = norm_layer(num_filters * 8)
        self.spade_layer8_7 = SPADE(norm_layer, num_filters * 8, 6)  # self.batch_norm8_7 = norm_layer(num_filters * 8)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 256 x 256
        e1 = self.conv1(input)
        # state size is (num_filters) x 128 x 128
        e2 = self.spade_layer2_0(self.conv2(self.leaky_relu(e1)), input)
        # state size is (num_filters x 2) x 64 x 64
        e3 = self.spade_layer4_0(self.conv3(self.leaky_relu(e2)), input)
        # state size is (num_filters x 4) x 32 x 32
        e4 = self.spade_layer8_0(self.conv4(self.leaky_relu(e3)), input)
        # state size is (num_filters x 8) x 16 x 16
        e5 = self.spade_layer8_1(self.conv5(self.leaky_relu(e4)), input)
        # state size is (num_filters x 8) x 8 x 8
        e6 = self.spade_layer8_2(self.conv6(self.leaky_relu(e5)), input)
        # state size is (num_filters x 8) x 4 x 4
        e7 = self.spade_layer8_3(self.conv7(self.leaky_relu(e6)), input)
        # state size is (num_filters x 8) x 2 x 2
        # No batch norm on output of Encoder
        e8 = self.conv8(self.leaky_relu(e7))

        # Decoder
        # Deconvolution layers:
        # state size is (num_filters x 8) x 1 x 1
        d1_ = self.spade_layer8_4(self.dconv1(self.up(self.relu(e8))), input)
        # state size is (num_filters x 8) x 2 x 2
        d1 = torch.cat((d1_, e7), 1)
        d2_ = self.spade_layer8_5(self.dconv2(self.up(self.relu(d1))), input)
        # state size is (num_filters x 8) x 4 x 4
        d2 = torch.cat((d2_, e6), 1)
        d3_ = self.spade_layer8_6(self.dconv3(self.up(self.relu(d2))), input)
        # state size is (num_filters x 8) x 8 x 8
        d3 = torch.cat((d3_, e5), 1)
        d4_ = self.spade_layer8_7(self.dconv4(self.up(self.relu(d3))), input)
        # state size is (num_filters x 8) x 16 x 16
        d4 = torch.cat((d4_, e4), 1)
        d5_ = self.spade_layer4_1(self.dconv5(self.up(self.relu(d4))), input)
        # state size is (num_filters x 4) x 32 x 32
        d5 = torch.cat((d5_, e3), 1)
        d6_ = self.spade_layer2_1(self.dconv6(self.up(self.relu(d5))), input)
        # state size is (num_filters x 2) x 64 x 64
        d6 = torch.cat((d6_, e2), 1)
        d7_ = self.spade_layer(self.dconv7(self.up(self.relu(d6))), input)
        # state size is (num_filters) x 128 x 128
        # d7_ = torch.Tensor(e1.data.new(e1.size()).normal_(0, 0.5))
        d7 = torch.cat((d7_, e1), 1)
        d8 = self.dconv8(self.up(self.relu(d7)))
        return d8


class SPADEUnet4MaskMotion(nn.Module):
    """ Reimplementation of Unet that allows for simpler modifications.
    """

    def __init__(
            self,
            num_filters=32,
            channels_in=3,
            channels_out=2,
            use_3D=False,
            config=None,
            up_mode="bilinear",
    ):
        super(SPADEUnet4MaskMotion, self).__init__()

        conv_layer = get_conv_layer(config, use_3D=use_3D)

        self.conv1 = conv_layer(channels_in, num_filters, 4, 2, 1)
        self.conv2 = conv_layer(num_filters, num_filters * 2, 4, 2, 1)
        self.conv3 = conv_layer(num_filters * 2, num_filters * 4, 4, 2, 1)
        self.conv4 = conv_layer(num_filters * 4, num_filters * 8, 4, 2, 1)
        self.conv5 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv6 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv7 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv8 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)

        # self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_nearest = nn.Upsample(scale_factor=2, mode="nearest")
        self.up = nn.Upsample(scale_factor=2, mode=up_mode, align_corners=False)

        self.dconv1 = conv_layer(num_filters * 8, num_filters * 8, 3, 1, 1)
        self.dconv2 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv3 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv4 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv5 = conv_layer(num_filters * 8 * 2, num_filters * 4, 3, 1, 1)
        self.dconv6 = conv_layer(num_filters * 4 * 2, num_filters * 2, 3, 1, 1)
        self.dconv7 = conv_layer(num_filters * 2 * 2, num_filters, 3, 1, 1)
        self.dconv8 = conv_layer(num_filters * 2, channels_out, 3, 1, 1)

        if "motion_norm_G" in config.keys():
            motion_norm_G = config['motion_norm_G'].split(":")[1]
            if motion_norm_G == "batch":
                norm_layer = nn.BatchNorm2d
            elif motion_norm_G == "spectral_instance":
                norm_layer = nn.InstanceNorm2d
            elif motion_norm_G == "spectral_batch":
                norm_layer = nn.BatchNorm2d
            elif motion_norm_G == "spectral_batchstanding":
                norm_layer = BatchNorm_StandingStats
        else:
            norm_layer = nn.InstanceNorm2d

        # self.spade_layer = SPADE(norm_layer, num_filters, 6)#self.batch_norm = norm_layer(num_filters)
        # self.spade_layer2_0 = SPADE(norm_layer, num_filters*2, 6)#self.batch_norm2_0 = norm_layer(num_filters * 2)
        # self.spade_layer2_1 = SPADE(norm_layer, num_filters*2, 6)#self.batch_norm2_1 = norm_layer(num_filters * 2)
        # self.spade_layer4_0 = SPADE(norm_layer, num_filters*4, 6)#self.batch_norm4_0 = norm_layer(num_filters * 4)
        # self.spade_layer4_1 = SPADE(norm_layer, num_filters*4, 6)#self.batch_norm4_1 = norm_layer(num_filters * 4)
        # self.spade_layer8_0 = SPADE(norm_layer, num_filters*8, 6)#self.batch_norm8_0 = norm_layer(num_filters * 8)
        # self.spade_layer8_1 = SPADE(norm_layer, num_filters*8, 6)#self.batch_norm8_1 = norm_layer(num_filters * 8)
        # self.spade_layer8_2 = SPADE(norm_layer, num_filters*8, 6)#self.batch_norm8_2 = norm_layer(num_filters * 8)
        # self.spade_layer8_3 = SPADE(norm_layer, num_filters*8, 6)#self.batch_norm8_3 = norm_layer(num_filters * 8)
        # self.spade_layer8_4 = SPADE(norm_layer, num_filters*8, 6)#self.batch_norm8_4 = norm_layer(num_filters * 8)
        # self.spade_layer8_5 = SPADE(norm_layer, num_filters*8, 6)#self.batch_norm8_5 = norm_layer(num_filters * 8)
        # self.spade_layer8_6 = SPADE(norm_layer, num_filters*8, 6)#self.batch_norm8_6 = norm_layer(num_filters * 8)
        # self.spade_layer8_7 = SPADE(norm_layer, num_filters*8, 6)#self.batch_norm8_7 = norm_layer(num_filters * 8)

        self.spade_layer = SPADE(norm_layer, num_filters, channels_in)  # self.batch_norm = norm_layer(num_filters)
        self.spade_layer2_0 = SPADE(norm_layer, num_filters * 2,
                                    channels_in)  # self.batch_norm2_0 = norm_layer(num_filters * 2)
        self.spade_layer2_1 = SPADE(norm_layer, num_filters * 2,
                                    channels_in)  # self.batch_norm2_1 = norm_layer(num_filters * 2)
        self.spade_layer4_0 = SPADE(norm_layer, num_filters * 4,
                                    channels_in)  # self.batch_norm4_0 = norm_layer(num_filters * 4)
        self.spade_layer4_1 = SPADE(norm_layer, num_filters * 4,
                                    channels_in)  # self.batch_norm4_1 = norm_layer(num_filters * 4)
        self.spade_layer8_0 = SPADE(norm_layer, num_filters * 8,
                                    channels_in)  # self.batch_norm8_0 = norm_layer(num_filters * 8)
        self.spade_layer8_1 = SPADE(norm_layer, num_filters * 8,
                                    channels_in)  # self.batch_norm8_1 = norm_layer(num_filters * 8)
        self.spade_layer8_2 = SPADE(norm_layer, num_filters * 8,
                                    channels_in)  # self.batch_norm8_2 = norm_layer(num_filters * 8)
        self.spade_layer8_3 = SPADE(norm_layer, num_filters * 8,
                                    channels_in)  # self.batch_norm8_3 = norm_layer(num_filters * 8)
        self.spade_layer8_4 = SPADE(norm_layer, num_filters * 8,
                                    channels_in)  # self.batch_norm8_4 = norm_layer(num_filters * 8)
        self.spade_layer8_5 = SPADE(norm_layer, num_filters * 8,
                                    channels_in)  # self.batch_norm8_5 = norm_layer(num_filters * 8)
        self.spade_layer8_6 = SPADE(norm_layer, num_filters * 8,
                                    channels_in)  # self.batch_norm8_6 = norm_layer(num_filters * 8)
        self.spade_layer8_7 = SPADE(norm_layer, num_filters * 8,
                                    channels_in)  # self.batch_norm8_7 = norm_layer(num_filters * 8)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 256 x 256
        e1 = self.conv1(input)
        # state size is (num_filters) x 128 x 128
        e2 = self.spade_layer2_0(self.conv2(self.leaky_relu(e1)), input)
        # state size is (num_filters x 2) x 64 x 64
        e3 = self.spade_layer4_0(self.conv3(self.leaky_relu(e2)), input)
        # state size is (num_filters x 4) x 32 x 32
        e4 = self.spade_layer8_0(self.conv4(self.leaky_relu(e3)), input)
        # state size is (num_filters x 8) x 16 x 16
        e5 = self.spade_layer8_1(self.conv5(self.leaky_relu(e4)), input)
        # state size is (num_filters x 8) x 8 x 8
        e6 = self.spade_layer8_2(self.conv6(self.leaky_relu(e5)), input)
        # state size is (num_filters x 8) x 4 x 4
        e7 = self.spade_layer8_3(self.conv7(self.leaky_relu(e6)), input)
        # state size is (num_filters x 8) x 2 x 2
        # No batch norm on output of Encoder
        e8 = self.conv8(self.leaky_relu(e7))

        # Decoder
        # Deconvolution layers:
        # state size is (num_filters x 8) x 1 x 1
        e8 = self.relu(e8)
        e8 = torch.cat([self.up(e8[:, :3, ...]), self.up_nearest(e8[:, 3:4, ...]), self.up(e8[:, 4:, ...])], 1)
        d1_ = self.spade_layer8_4(self.dconv1(e8), input)
        # state size is (num_filters x 8) x 2 x 2
        d1_ = torch.cat([self.up(d1_[:, :3, ...]), self.up_nearest(d1_[:, 3:4, ...]), self.up(d1_[:, 4:, ...])], 1)
        e7 = torch.cat([self.up(e7[:, :3, ...]), self.up_nearest(e7[:, 3:4, ...]), self.up(e7[:, 4:, ...])], 1)
        d1 = torch.cat((d1_, e7), 1)
        d1 = self.relu(d1)
        d2_ = self.spade_layer8_5(self.dconv2(d1), input)
        # state size is (num_filters x 8) x 4 x 4
        d2_ = torch.cat([self.up(d2_[:, :3, ...]), self.up_nearest(d2_[:, 3:4, ...]), self.up(d2_[:, 4:, ...])], 1)
        e6 = torch.cat([self.up(e6[:, :3, ...]), self.up_nearest(e6[:, 3:4, ...]), self.up(e6[:, 4:, ...])], 1)
        d2 = torch.cat((d2_, e6), 1)
        d2 = self.relu(d2)
        d3_ = self.spade_layer8_6(self.dconv3(d2), input)
        # state size is (num_filters x 8) x 8 x 8
        d3_ = torch.cat([self.up(d3_[:, :3, ...]), self.up_nearest(d3_[:, 3:4, ...]), self.up(d3_[:, 4:, ...])], 1)
        e5 = torch.cat([self.up(e5[:, :3, ...]), self.up_nearest(e5[:, 3:4, ...]), self.up(e5[:, 4:, ...])], 1)
        d3 = torch.cat((d3_, e5), 1)
        d3 = self.relu(d3)
        d4_ = self.spade_layer8_7(self.dconv4(d3), input)
        # state size is (num_filters x 8) x 16 x 16
        d4_ = torch.cat([self.up(d4_[:, :3, ...]), self.up_nearest(d4_[:, 3:4, ...]), self.up(d4_[:, 4:, ...])], 1)
        e4 = torch.cat([self.up(e4[:, :3, ...]), self.up_nearest(e4[:, 3:4, ...]), self.up(e4[:, 4:, ...])], 1)
        d4 = torch.cat((d4_, e4), 1)
        d4 = self.relu(d4)
        d5_ = self.spade_layer4_1(self.dconv5(d4), input)
        # state size is (num_filters x 4) x 32 x 32
        d5_ = torch.cat([self.up(d5_[:, :3, ...]), self.up_nearest(d5_[:, 3:4, ...]), self.up(d5_[:, 4:, ...])], 1)
        e3 = torch.cat([self.up(e3[:, :3, ...]), self.up_nearest(e3[:, 3:4, ...]), self.up(e3[:, 4:, ...])], 1)
        d5 = torch.cat((d5_, e3), 1)
        d5 = self.relu(d5)
        d6_ = self.spade_layer2_1(self.dconv6(d5), input)
        # state size is (num_filters x 2) x 64 x 64
        d6_ = torch.cat([self.up(d6_[:, :3, ...]), self.up_nearest(d6_[:, 3:4, ...]), self.up(d6_[:, 4:, ...])], 1)
        e2 = torch.cat([self.up(e2[:, :3, ...]), self.up_nearest(e2[:, 3:4, ...]), self.up(e2[:, 4:, ...])], 1)
        d6 = torch.cat((d6_, e2), 1)
        d6 = self.relu(d6)
        d7_ = self.spade_layer(self.dconv7(d6), input)
        # state size is (num_filters) x 128 x 128
        # d7_ = torch.Tensor(e1.data.new(e1.size()).normal_(0, 0.5))
        d7_ = torch.cat([self.up(d7_[:, :3, ...]), self.up_nearest(d7_[:, 3:4, ...]), self.up(d7_[:, 4:, ...])], 1)
        e1 = torch.cat([self.up(e1[:, :3, ...]), self.up_nearest(e1[:, 3:4, ...]), self.up(e1[:, 4:, ...])], 1)
        d7 = torch.cat((d7_, e1), 1)
        d7 = self.relu(d7)
        d8 = self.dconv8(d7)
        return d8


class Unet(nn.Module):
    """ Reimplementation of Unet that allows for simpler modifications.
    """

    def __init__(
            self,
            num_filters=32,
            channels_in=3,
            channels_out=3,
            use_tanh=False,
            use_3D=False,
            opt=None,
    ):
        super(Unet, self).__init__()

        conv_layer = get_conv_layer(opt, use_3D=use_3D)

        self.conv1 = conv_layer(channels_in, num_filters, 4, 2, 1)
        self.conv2 = conv_layer(num_filters, num_filters * 2, 4, 2, 1)
        self.conv3 = conv_layer(num_filters * 2, num_filters * 4, 4, 2, 1)
        self.conv4 = conv_layer(num_filters * 4, num_filters * 8, 4, 2, 1)
        self.conv5 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv6 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv7 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv8 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.dconv1 = conv_layer(num_filters * 8, num_filters * 8, 3, 1, 1)
        self.dconv2 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv3 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv4 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv5 = conv_layer(num_filters * 8 * 2, num_filters * 4, 3, 1, 1)
        self.dconv6 = conv_layer(num_filters * 4 * 2, num_filters * 2, 3, 1, 1)
        self.dconv7 = conv_layer(num_filters * 2 * 2, num_filters, 3, 1, 1)
        self.dconv8 = conv_layer(num_filters * 2, channels_out, 3, 1, 1)

        norm_layer = get_batchnorm_layer(opt)

        self.batch_norm = norm_layer(num_filters)
        self.batch_norm2_0 = norm_layer(num_filters * 2)
        self.batch_norm2_1 = norm_layer(num_filters * 2)
        self.batch_norm4_0 = norm_layer(num_filters * 4)
        self.batch_norm4_1 = norm_layer(num_filters * 4)
        self.batch_norm8_0 = norm_layer(num_filters * 8)
        self.batch_norm8_1 = norm_layer(num_filters * 8)
        self.batch_norm8_2 = norm_layer(num_filters * 8)
        self.batch_norm8_3 = norm_layer(num_filters * 8)
        self.batch_norm8_4 = norm_layer(num_filters * 8)
        self.batch_norm8_5 = norm_layer(num_filters * 8)
        self.batch_norm8_6 = norm_layer(num_filters * 8)
        self.batch_norm8_7 = norm_layer(num_filters * 8)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 256 x 256
        e1 = self.conv1(input)
        # state size is (num_filters) x 128 x 128
        e2 = self.batch_norm2_0(self.conv2(self.leaky_relu(e1)))
        # state size is (num_filters x 2) x 64 x 64
        e3 = self.batch_norm4_0(self.conv3(self.leaky_relu(e2)))
        # state size is (num_filters x 4) x 32 x 32
        e4 = self.batch_norm8_0(self.conv4(self.leaky_relu(e3)))
        # state size is (num_filters x 8) x 16 x 16
        e5 = self.batch_norm8_1(self.conv5(self.leaky_relu(e4)))
        # state size is (num_filters x 8) x 8 x 8
        e6 = self.batch_norm8_2(self.conv6(self.leaky_relu(e5)))
        # state size is (num_filters x 8) x 4 x 4
        e7 = self.batch_norm8_3(self.conv7(self.leaky_relu(e6)))
        # state size is (num_filters x 8) x 2 x 2
        # No batch norm on output of Encoder
        e8 = self.conv8(self.leaky_relu(e7))

        # Decoder
        # Deconvolution layers:
        # state size is (num_filters x 8) x 1 x 1
        d1_ = self.batch_norm8_4(self.dconv1(self.up(self.relu(e8))))
        # state size is (num_filters x 8) x 2 x 2
        d1 = torch.cat((d1_, e7), 1)
        d2_ = self.batch_norm8_5(self.dconv2(self.up(self.relu(d1))))
        # state size is (num_filters x 8) x 4 x 4
        d2 = torch.cat((d2_, e6), 1)
        d3_ = self.batch_norm8_6(self.dconv3(self.up(self.relu(d2))))
        # state size is (num_filters x 8) x 8 x 8
        d3 = torch.cat((d3_, e5), 1)
        d4_ = self.batch_norm8_7(self.dconv4(self.up(self.relu(d3))))
        # state size is (num_filters x 8) x 16 x 16
        d4 = torch.cat((d4_, e4), 1)
        d5_ = self.batch_norm4_1(self.dconv5(self.up(self.relu(d4))))
        # state size is (num_filters x 4) x 32 x 32
        d5 = torch.cat((d5_, e3), 1)
        d6_ = self.batch_norm2_1(self.dconv6(self.up(self.relu(d5))))
        # state size is (num_filters x 2) x 64 x 64
        d6 = torch.cat((d6_, e2), 1)
        d7_ = self.batch_norm(self.dconv7(self.up(self.relu(d6))))
        # state size is (num_filters) x 128 x 128
        # d7_ = torch.Tensor(e1.data.new(e1.size()).normal_(0, 0.5))
        d7 = torch.cat((d7_, e1), 1)
        d8 = self.dconv8(self.up(self.relu(d7)))
        # state size is (nc) x 256 x 256
        # output = self.tanh(d8)
        return d8


class UNetDecoder16Pconv2(nn.Module):
    """ Reimplementation of Unet that allows for simpler modifications.
    """

    def __init__(self, opt=None, num_filters=64, channels_in=3, channels_out=3):
        super().__init__()
        conv_layer = get_conv_layer(opt, use_3D=False)
        pconv_layer = get_pconv_layer(opt)
        norm_layer = get_batchnorm_layer(opt)

        self.conv1 = pconv_layer(num_filters, num_filters, 7, 3, 2)  # 64
        self.conv2 = pconv_layer(num_filters, num_filters * 2, 5, 2, 2)  # 128
        self.conv3 = pconv_layer(num_filters * 2, num_filters * 4, 3, 1, 2)  # 256
        self.conv4 = pconv_layer(num_filters * 4, num_filters * 8, 3, 1, 2)  # 512
        self.conv5 = pconv_layer(num_filters * 8, num_filters * 8, 3, 1, 2)
        self.conv6 = pconv_layer(num_filters * 8, num_filters * 8, 3, 1, 2)
        self.conv7 = pconv_layer(num_filters * 8, num_filters * 8, 3, 1, 2)
        self.conv8 = pconv_layer(num_filters * 8, num_filters * 8, 3, 1, 2)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # self.up = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dconv1 = pconv_layer(num_filters * 16, num_filters * 8, 3, 1, 1)
        self.dconv2 = pconv_layer(num_filters * 16, num_filters * 8, 3, 1, 1)
        self.dconv3 = pconv_layer(num_filters * 16, num_filters * 8, 3, 1, 1)
        self.dconv4 = pconv_layer(num_filters * 16, num_filters * 8, 3, 1, 1)
        self.dconv5 = pconv_layer(num_filters * 12, num_filters * 4, 3, 1, 1)
        self.dconv6 = pconv_layer(num_filters * 6, num_filters * 2, 3, 1, 1)
        self.dconv7 = pconv_layer(num_filters * 3, num_filters, 3, 1, 1)
        self.dconv8 = pconv_layer(num_filters * 2, channels_out, 3, 1, 1)

        self.batch_norm2 = norm_layer(num_filters * 2)
        self.batch_norm3 = norm_layer(num_filters * 4)
        self.batch_norm4 = norm_layer(num_filters * 8)
        self.batch_norm5 = norm_layer(num_filters * 8)
        self.batch_norm6 = norm_layer(num_filters * 8)
        self.batch_norm7 = norm_layer(num_filters * 8)
        self.batch_norm8 = norm_layer(num_filters * 8)

        self.batch_norm9 = norm_layer(num_filters * 8)
        self.batch_norm10 = norm_layer(num_filters * 8)
        self.batch_norm11 = norm_layer(num_filters * 8)
        self.batch_norm12 = norm_layer(num_filters * 8)
        self.batch_norm13 = norm_layer(num_filters * 4)
        self.batch_norm14 = norm_layer(num_filters * 2)
        self.batch_norm15 = norm_layer(num_filters)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

        if opt.normalize_image:
            self.norm = nn.Tanh()
        else:
            self.norm = nn.Sigmoid()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        mask = (input != 0).float()
        e1, mask1 = self.conv1(input, mask)
        e1 = self.relu(e1)
        e2, mask2 = self.conv2(e1, mask1)
        e2 = self.batch_norm2(e2)
        e2 = self.relu(e2)
        # state size is (num_filters x 2) x 64 x 64
        e3, mask3 = self.conv3(e2, mask2)
        e3 = self.batch_norm3(e3)
        e3 = self.relu(e3)
        # state size is (num_filters x 4) x 32 x 32
        e4, mask4 = self.conv4(e3, mask3)
        e4 = self.batch_norm4(e4)
        e4 = self.relu(e4)
        # state size is (num_filters x 8) x 16 x 16
        e5, mask5 = self.conv5(e4, mask4)
        e5 = self.batch_norm5(e5)
        e5 = self.relu(e5)
        # state size is (num_filters x 8) x 8 x 8
        e6, mask6 = self.conv6(e5, mask5)
        e6 = self.batch_norm6(e6)
        e6 = self.relu(e6)
        # state size is (num_filters x 8) x 4 x 4
        e7, mask7 = self.conv7(e6, mask6)
        e7 = self.batch_norm7(e7)
        e7 = self.relu(e7)
        # state size is (num_filters x 8) x 2 x 2
        e8, mask8 = self.conv8(self.relu(e8), mask7)
        e8 = self.batch_norm8(e8)
        e8 = self.relu(e8)
        # state size is (num_filters x 8) x 1 x 1
        # Decoder
        # Deconvolution layers:
        d = self.up(e8);
        del e8
        dmask = self.up(mask8);
        del mask8
        d = torch.cat((d, e7), 1)
        dmask = torch.cat((dmask, mask7), 1);
        del mask7
        d, dmask = self.dconv1(d, dmask)
        d = self.batch_norm9(d)
        d = self.leaky_relu(d)

        d = self.up(d);
        del e7
        dmask = self.up(dmask);
        d = torch.cat((d, e6), 1)
        dmask = torch.cat((dmask, mask6), 1);
        del mask6
        d, dmask = self.dconv2(d, dmask)
        d = self.batch_norm10(d)
        d = self.leaky_relu(d)

        d = self.up(d);
        del e6
        dmask = self.up(dmask)
        d = torch.cat((d, e5), 1)
        dmask = torch.cat((dmask, mask5), 1)
        d, dmask = self.dconv3(d, dmask)
        d = self.batch_norm11(d)
        d = self.leaky_relu(d)

        d = self.up(d);
        dmask = self.up(dmask)
        d = torch.cat((d, e4), 1);
        del e4
        dmask = torch.cat((dmask, mask4), 1);
        del mask4
        d, dmask = self.dconv4(d, dmask)
        d = self.batch_norm12(d)
        d = self.leaky_relu(d)

        d = self.up(d)
        dmask = self.up(dmask)
        d = torch.cat((d, e3), 1);
        del e3
        dmask = torch.cat((dmask, mask3), 1);
        del mask3
        d, dmask = self.dconv5(d, dmask)
        d = self.batch_norm13(d)
        d = self.leaky_relu(d)

        d = self.up(d)
        dmask = self.up(dmask)
        d = torch.cat((d, e2), 1);
        del e2
        dmask = torch.cat((dmask, mask2), 1);
        del mask2
        d, dmask = self.dconv6(d, dmask)
        d = self.batch_norm14(d)
        d = self.leaky_relu(d)

        d = self.up(d)
        dmask = self.up(dmask)
        d = torch.cat((d, e1), 1);
        del e1
        dmask = torch.cat((dmask, mask1), 1);
        del mask1
        d, dmask = self.dconv7(d, dmask)
        d = self.batch_norm15(d)
        d = self.leaky_relu(d)

        d = self.up(d)
        dmask = self.up(dmask)
        d = torch.cat((d, input), 1);
        del input
        dmask = torch.cat((dmask, mask), 1)
        d, dmask = self.dconv8(d, dmask)
        return self.norm(d)
