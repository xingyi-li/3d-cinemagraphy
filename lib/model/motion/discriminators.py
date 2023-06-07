"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Based on https://github.com/NVlabs/SPADE/blob/master/models/pix2pix_model.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from lib.model.motion.layers.normalization import get_D_norm_layer


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(
            "Network [%s] was created."
            + "Total number of parameters: %.1f million. "
              "To see the architecture, do print(network)."
            % (type(self).__name__, num_params / 1000000)
        )

    def init_weights(self, init_type="normal", gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm2d") != -1:
                if hasattr(m, "weight") and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, "weight") and (
                    classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                if init_type == "normal":
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier":
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "xavier_uniform":
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == "kaiming":
                    init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == "none":  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        "initialization method [%s] is not implemented"
                        % init_type
                    )
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, "init_weights"):
                m.init_weights(init_type, gain)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    def __init__(self, config):
        super().__init__()
        self.config = config

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = config['ndf']
        input_nc = self.compute_D_input_nc(config)

        norm_layer = get_D_norm_layer(config, config['norm_D'])
        sequence = [
            [
                nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, False),
            ]
        ]

        for n in range(1, config['n_layers_D']):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == config['n_layers_D'] - 1 else 2
            sequence += [
                [
                    norm_layer(
                        nn.Conv2d(
                            nf_prev,
                            nf,
                            kernel_size=kw,
                            stride=stride,
                            padding=padw,
                        )
                    ),
                    nn.LeakyReLU(0.2, False),
                ]
            ]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module("model" + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, config):
        # if opt.concat_discriminators:
        #     input_nc = opt.output_nc * 2
        # else:
        input_nc = config['output_nc']
        return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.config['no_ganFeat_loss']
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]


class MultiscaleDiscriminator(BaseNetwork):
    def __init__(self, config):
        super().__init__()
        self.config = config

        for i in range(self.config['num_D']):
            subnetD = self.create_single_discriminator(config)
            self.add_module("discriminator_%d" % i, subnetD)

    def create_single_discriminator(self, config):
        netD = NLayerDiscriminator(config)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(
            input,
            kernel_size=3,
            stride=2,
            padding=[1, 1],
            count_include_pad=False,
        )

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.config['no_ganFeat_loss']

        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


def define_D(config):
    net = MultiscaleDiscriminator(config)
    net.init_weights("xavier", 0.02)
    if torch.cuda.is_available():
        net = net.cuda()
    return net
