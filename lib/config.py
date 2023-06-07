import yaml


ANIMATION3D_DEFAULTS = {
    'generator': {
        'div_flow': 1.0,
        'use_mask_as_motion_input': True,
        'use_hint_as_motion_input': True,
        'norm_G': 'sync:spectral_batch',
        'motion_norm_G': 'sync:spectral_instance',
        'motion_losses': ['10.0_EndPointError'],
        'motionH': 256,
        'motionW': 256,
        'W': 256,
    },

    'discriminator': {
        'output_nc': 2,
        'ndf': 64,
        'num_D': 2,
        'norm_D': 'spectralinstance',
        'n_layers_D': 4,
        'discriminator_losses': 'pix2pixHD',
        'gan_mode': 'hinge',
        'no_ganFeat_loss': False,
        'lambda_feat': 10.0
    },
    'animation_discriminator': {
        'output_nc': 3,
        'ndf': 64,
        'num_D': 2,
        'norm_D': 'spectralinstance',
        'n_layers_D': 4,
        'discriminator_losses': 'pix2pixHD',
        'gan_mode': 'hinge',
        'no_ganFeat_loss': False,
        'lambda_feat': 10.0
    },
    'spacetime_model': {
        'use_inpainting_mask_for_feature': False,
        'use_depth_for_feature': False,
        'feature_dim': 32,
        'use_depth_for_decoding': True,
        'use_mask_for_decoding': False,
        'adaptive_pts_radius': True,
        'point_radius': 1.5,
        'vary_pts_radius': True,
    },

    'data': {
        'name': 'eulerian_data_motion_hint',
        'motionH': 768,
        'motionW': 768,
        'W': 768,
        'max_hint': 5
    },

    'train': {
        'batch_size': 1,
        'n_itrs': 250000,
        'optim': 'adam',
        'lr': 1e-4,
        'lrate_decay_steps': 50000,
        'lrate_decay_factor': 0.5,

        'lr_d': 2e-3,
        'beta1': 0,
        'beta2': 0.9,
        'decay_itrs': [-1],
        'decay': 0.5,

        'boundary_crop_ratio': 0,
        'loss_mode': 'lpips'
    },
}


def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    defaults = ANIMATION3D_DEFAULTS
    _merge(defaults, config)
    return config
