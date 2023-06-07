import os
import torch
from lib.model.inpaint.networks.resunet import ResUNet
from lib.model.inpaint.networks.img_decoder import ImgDecoder
from lib.utils.general_utils import de_parallel
from lib.model.motion.motion_loss import DiscriminatorLoss


class Namespace:

    def __init__(self, **kwargs):
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def __eq__(self, other):
        if not isinstance(other, Namespace):
            return NotImplemented
        return vars(self) == vars(other)

    def __contains__(self, key):
        return key in self.__dict__


########################################################################################################################
# creation/saving/loading of the model
########################################################################################################################

class SpaceTimeAnimationModel(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        load_opt = not args.no_load_opt
        load_scheduler = not args.no_load_scheduler
        device = torch.device('cuda:{}'.format(args.local_rank))

        # initialize feature extraction network
        feat_in_ch = 4
        if config['spacetime_model']['use_inpainting_mask_for_feature']:
            feat_in_ch += 1
        if config['spacetime_model']['use_depth_for_feature']:
            feat_in_ch += 1
        self.feature_net = ResUNet(in_ch=feat_in_ch, out_ch=config['spacetime_model']['feature_dim']).to(device)
        # initialize decoder
        decoder_in_ch = config['spacetime_model']['feature_dim'] + 4
        decoder_out_ch = 3

        if config['spacetime_model']['use_depth_for_decoding']:
            decoder_in_ch += 1
        if config['spacetime_model']['use_mask_for_decoding']:
            decoder_in_ch += 1

        self.img_decoder = ImgDecoder(in_ch=decoder_in_ch, out_ch=decoder_out_ch).to(device)

        learnable_params = list(self.feature_net.parameters())
        learnable_params += list(self.img_decoder.parameters())

        self.G_learnable_params = learnable_params

        self.optimG = torch.optim.Adam(self.G_learnable_params,
                                       lr=config['train']['lr'],
                                       weight_decay=1e-4,
                                       betas=(0.9, 0.999))

        self.schedG = torch.optim.lr_scheduler.StepLR(self.optimG,
                                                      step_size=config['train']['lrate_decay_steps'],
                                                      gamma=config['train']['lrate_decay_factor'])

        self.netD = DiscriminatorLoss(config['animation_discriminator']).to(device)
        self.D_learnable_params = list(self.netD.parameters())
        self.learnable_params = self.G_learnable_params + self.D_learnable_params

        self.optimD = torch.optim.Adam(self.D_learnable_params,
                                       lr=config['train']['lr_d'],
                                       betas=(config['train']['beta1'], config['train']['beta2']))
        self.schedD = torch.optim.lr_scheduler.StepLR(self.optimD,
                                                      step_size=config['train']['lrate_decay_steps'],
                                                      gamma=config['train']['lrate_decay_factor'])

        out_folder = os.path.join(args.input_dir, 'output')
        self.start_step = self.load_from_ckpt(out_folder,
                                              load_opt=load_opt,
                                              load_scheduler=load_scheduler)

        if args.distributed:
            self.feature_net = torch.nn.parallel.DistributedDataParallel(
                self.feature_net,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
            )

            self.img_decoder = torch.nn.parallel.DistributedDataParallel(
                self.img_decoder,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
            )

            self.netD = torch.nn.parallel.DistributedDataParallel(
                self.netD,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
            )

    def switch_to_eval(self):
        self.feature_net.eval()
        self.img_decoder.eval()
        self.netD.eval()

    def switch_to_train(self):
        self.feature_net.train()
        self.img_decoder.train()
        self.netD.train()

    def save_model(self, filename):
        to_save = {'optimG': self.optimG.state_dict(),
                   'schedG': self.schedG.state_dict(),
                   'optimD': self.optimD.state_dict(),
                   'schedD': self.schedD.state_dict(),
                   'feature_net': de_parallel(self.feature_net).state_dict(),
                   'img_decoder': de_parallel(self.img_decoder).state_dict(),
                   'netD': de_parallel(self.netD).state_dict(),
                   }
        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.args.distributed:
            to_load = torch.load(filename, map_location='cuda:{}'.format(self.args.local_rank))
        else:
            to_load = torch.load(filename)

        if load_opt:
            self.optimG.load_state_dict(to_load['optimG'])
            self.optimD.load_state_dict(to_load['optimD'])
        if load_scheduler:
            self.schedG.load_state_dict(to_load['schedG'])
            self.schedD.load_state_dict(to_load['schedD'])

        self.feature_net.load_state_dict(to_load['feature_net'])
        self.img_decoder.load_state_dict(to_load['img_decoder'])
        self.netD.load_state_dict(to_load['netD'])

    def load_from_ckpt(self, out_folder,
                       load_opt=True,
                       load_scheduler=True,
                       force_latest_ckpt=False):
        '''
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        '''

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f)
                     for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            print('Reloading from {}, starting at step={}'.format(fpath, step))
        else:
            print('No ckpts found, training from scratch...')
            step = 0

        return step
