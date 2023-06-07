"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Based on https://github.com/NVlabs/SPADE/blob/master/models/pix2pix_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import lib.model.motion.discriminators as discriminators


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(
            self,
            gan_mode,
            target_real_label=1.0,
            target_fake_label=0.0,
            tensor=torch.FloatTensor,
            config=None,
    ):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.config = config
        if gan_mode == "ls":
            pass
        elif gan_mode == "original":
            pass
        elif gan_mode == "w":
            pass
        elif gan_mode == "hinge":
            pass
        else:
            raise ValueError("Unexpected gan_mode {}".format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = (
                    self.Tensor(1).fill_(self.real_label).to(input.device)
                )
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = (
                    self.Tensor(1).fill_(self.fake_label).to(input.device)
                )
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)

        self.zero_tensor = self.zero_tensor.to(input.device)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == "original":  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == "ls":
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == "hinge":
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert (
                    target_is_real
                ), "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(
                    pred_i, target_is_real, for_discriminator
                )
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


class BaseDiscriminator(nn.Module):
    def __init__(self, config, name):
        super().__init__()

        if name == "pix2pixHD":
            self.netD = discriminators.define_D(config)
        elif name == "pix2pixHDorigin":
            from models.networks.networks import define_D
            """
            self.netD = define_D(input_nc=2,
                                 ndf=64,
                                 n_layers_D=3,
                                 norm='instance',
                                 use_sigmoid=False,
                                 num_D=3,
                                 getIntermFeat=False)
            """
            self.netD = define_D(input_nc=2,
                                 ndf=64,
                                 n_layers_D=3,
                                 norm='instance',
                                 use_sigmoid=False,
                                 num_D=2,
                                 getIntermFeat=False)
        self.criterionGAN = GANLoss(
            config['gan_mode'], tensor=torch.FloatTensor, config=config
        )
        self.criterionFeat = torch.nn.L1Loss()
        self.config = config

        self.FloatTensor = (
            torch.cuda.FloatTensor
            if torch.cuda.is_available()
            else torch.FloatTensor
        )

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, fake_image, real_image):

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_image, real_image], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[: tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[: pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def compute_discrimator_loss(self, fake_image, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(fake_image, real_image)

        D_losses["D_Fake"] = self.criterionGAN(
            pred_fake, False, for_discriminator=True
        )
        D_losses["D_real"] = self.criterionGAN(
            pred_real, True, for_discriminator=True
        )
        D_losses["Total Loss"] = sum(D_losses.values()).mean()

        return D_losses

    def compute_generator_loss(self, fake_image, real_image):
        G_losses = {}
        pred_fake, pred_real = self.discriminate(fake_image, real_image)
        # numD2 num_intermediate_outputs5 4 64
        G_losses["GAN"] = self.criterionGAN(
            pred_fake, True, for_discriminator=False
        )

        if not self.config['no_ganFeat_loss']:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(
                        num_intermediate_outputs
                ):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach()
                    )
                    GAN_Feat_loss += (
                            unweighted_loss * self.config['lambda_feat'] / num_D
                    )
            G_losses["GAN_Feat"] = GAN_Feat_loss

        G_losses["Total Loss"] = sum(G_losses.values()).mean()

        return G_losses, fake_image

    def forward(self, fake_image, real_image, mode="generator"):
        if mode == "generator":
            g_loss, generated = self.compute_generator_loss(
                fake_image, real_image
            )
            return g_loss

        elif mode == "discriminator":
            d_loss = self.compute_discrimator_loss(fake_image, real_image)
            return d_loss

    def update_learning_rate(self, curr_epoch):
        restart, new_lrs = self.netD.update_learning_rate(curr_epoch)

        return restart, new_lrs


class DiscriminatorLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Get the losses
        loss_name = config['discriminator_losses']  # 'pix2pixHD'

        self.netD = self.get_loss_from_name(loss_name)

    def get_optimizer(self):
        optimizerD = torch.optim.Adam(
            list(self.netD.parameters()), lr=self.opt.lr * 2, betas=(0, 0.9)
        )
        return optimizerD

    def get_loss_from_name(self, name):
        netD = BaseDiscriminator(self.config, name=name)

        if torch.cuda.is_available():
            return netD.cuda()

        return netD

    def forward(self, pred_img, gt_img):
        losses = [loss(pred_img, gt_img, mode="discriminator") for loss in self.losses]

        loss_dir = {}
        for i, l in enumerate(losses):
            if "Total Loss" in l.keys():
                if "Total Loss" in loss_dir.keys():
                    loss_dir["Total Loss"] = (
                            loss_dir["Total Loss"]
                            + l["Total Loss"] * self.lambdas[i]
                    )
                else:
                    loss_dir["Total Loss"] = l["Total Loss"]

            loss_dir = dict(l, **loss_dir)  # Have loss_dir override l

        return loss_dir

    def run_generator_one_step(self, pred_img, gt_img):
        return self.netD(pred_img, gt_img, mode="generator")

    def run_discriminator_one_step(self, pred_img, gt_img):
        return self.netD(pred_img, gt_img, mode="discriminator")

    def update_learning_rate(self, curr_epoch):
        restart, new_lrs = self.netD.update_learning_rate(curr_epoch)

        return restart, new_lrs


class AlphaDiscriminatorLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        # Get the losses
        loss_name = opt.discriminator_losses  # 'pix2pixHD'

        self.netD_alpha = self.get_loss_from_name(loss_name)

    def get_optimizer(self):
        optimizer_D_alpha = torch.optim.Adam(
            list(self.netD_alpha.parameters()), lr=self.opt.lr * 2, betas=(0, 0.9)
        )
        return optimizer_D_alpha

    def get_loss_from_name(self, name):
        netD_alpha = BaseDiscriminator(self.opt, name=name)

        if torch.cuda.is_available():
            return netD_alpha.cuda()

        return netD_alpha

    def forward(self, pred_img, gt_img):
        losses = [
            loss(pred_img, gt_img, mode="discriminator") for loss in self.losses
        ]

        loss_dir = {}
        for i, l in enumerate(losses):
            if "Total Loss" in l.keys():
                if "Total Loss" in loss_dir.keys():
                    loss_dir["Total Loss"] = (
                            loss_dir["Total Loss"]
                            + l["Total Loss"] * self.lambdas[i]
                    )
                else:
                    loss_dir["Total Loss"] = l["Total Loss"]

            loss_dir = dict(l, **loss_dir)  # Have loss_dir override l

        return loss_dir

    def run_generator_one_step(self, pred_img, gt_img):
        return self.netD_alpha(pred_img, gt_img, mode="generator")

    def run_discriminator_one_step(self, pred_img, gt_img):
        return self.netD_alpha(pred_img, gt_img, mode="discriminator")

    def update_learning_rate(self, curr_epoch):
        restart, new_lrs = self.netD_alpha.update_learning_rate(curr_epoch)

        return restart, new_lrs


class InpaintDiscriminatorLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        # Get the losses
        loss_name = opt.discriminator_losses  # 'pix2pixHD'

        self.netD = self.get_loss_from_name(loss_name)

    def get_optimizer(self):
        optimizerD = torch.optim.Adam(
            list(self.netD.parameters()), lr=self.opt.lr * 2, betas=(0, 0.9)
        )
        return optimizerD

    def get_loss_from_name(self, name):
        netD = BaseDiscriminator(self.opt, name=name)

        if torch.cuda.is_available():
            return netD.cuda()

        return netD

    def forward(self, pred_img, gt_img):
        losses = [
            loss(pred_img, gt_img, mode="discriminator") for loss in self.losses
        ]

        loss_dir = {}
        for i, l in enumerate(losses):
            if "Total Loss" in l.keys():
                if "Total Loss" in loss_dir.keys():
                    loss_dir["Total Loss"] = (
                            loss_dir["Total Loss"]
                            + l["Total Loss"] * self.lambdas[i]
                    )
                else:
                    loss_dir["Total Loss"] = l["Total Loss"]

            loss_dir = dict(l, **loss_dir)  # Have loss_dir override l

        return loss_dir

    def run_generator_one_step(self, pred_img, gt_img):
        return self.netD(pred_img, gt_img, mode="generator")

    def run_discriminator_one_step(self, pred_img, gt_img):
        return self.netD(pred_img, gt_img, mode="discriminator")

    def update_learning_rate(self, curr_epoch):
        restart, new_lrs = self.netD.update_learning_rate(curr_epoch)

        return restart, new_lrs


class MotionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Get the losses
        print(config['motion_losses'])
        print(zip(*[l.split("_") for l in config['motion_losses']]))
        lambdas, loss_names = zip(*[l.split("_") for l in config['motion_losses']])
        lambdas = [float(l) for l in lambdas]
        # loss_names += ("PSNR", "SSIM")

        self.lambdas = lambdas
        self.loss_names = loss_names
        self.losses = nn.ModuleList(
            [self.get_loss_from_name(loss_name) for loss_name in loss_names]
        )

    def get_loss_from_name(self, name):
        if name == "MotionL1":
            loss = MotionL1LossWrapper()
        elif name == "EndPointError":
            loss = MotionEnePointErrorWrapper()

        if torch.cuda.is_available():
            return loss.cuda()

    def forward(self, pred_motion, gt_motion, image=None, mask=None):
        losses = []
        for i, loss_name in enumerate(self.loss_names):
            if ("EP" not in loss_name) and ("Reconstruction" not in loss_name):
                losses.append(self.losses[i](pred_motion, gt_motion))
            else:
                losses.append(self.losses[i](pred_motion, gt_motion, image))

        loss_dir = {}
        for i, l in enumerate(losses):
            if "Total Loss" in l.keys():
                if "Total Loss" in loss_dir.keys():
                    loss_dir["Total Loss"] = (
                            loss_dir["Total Loss"]
                            + l["Total Loss"] * self.lambdas[i]
                    )
                else:
                    loss_dir["Total Loss"] = l["Total Loss"] * self.lambdas[i]

            loss_dir = dict(l, **loss_dir)  # Have loss_dir override l
        return loss_dir


# Wrapper of the L1Loss so that the format matches what is expected
class L1LossWrapper(nn.Module):
    def __init__(self, subname=""):
        super().__init__()
        self.subname = subname

    def forward(self, pred_img, gt_img):
        err = nn.L1Loss()(pred_img, gt_img)
        return {"L1" + self.subname: err, "Total Loss": err}


class MotionL1LossWrapper(nn.Module):
    def forward(self, pred_motion, gt_motion):
        err = nn.L1Loss()(pred_motion, gt_motion)
        return {"MotionL1": err, "Total Loss": err}


class MotionEnePointErrorWrapper(nn.Module):
    def forward(self, pred_motion, gt_motion):
        if pred_motion.shape[1] == 3:
            new_pred_motion = pred_motion[:, :2, :, :] * pred_motion[:, 2:3, :, :]
        else:
            new_pred_motion = pred_motion
        if gt_motion.shape[1] == 3:
            new_gt_motion = gt_motion[:, :2, :, :] * gt_motion[:, 2:3, :, :]
        else:
            new_gt_motion = gt_motion
        # err = nn.MSELoss(reduction='none')(new_pred_motion, new_gt_motion).sum(1).sqrt()
        err = torch.norm(new_pred_motion - new_gt_motion, 2, 1)
        err = err.mean()
        return {"EndPointError": err, "Total Loss": err}
