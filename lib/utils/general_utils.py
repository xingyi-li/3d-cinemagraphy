import time
import os
import numpy as np
import torch
from datetime import datetime
import shutil

TINY_NUMBER = 1e-6  # float32 only has 7 decimal digits precision


def de_parallel(model):
    return model.module if hasattr(model, 'module') else model


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def dict_to_device(dict_):
    for k in dict_.keys():
        if type(dict_[k]) == torch.Tensor:
            dict_[k] = dict_[k].cuda()

    return dict_


def save_current_code(outdir):
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m_%d-%H:%M:%S")
    src_dir = '.'
    code_out_dir = os.path.join(outdir, 'code')
    os.makedirs(code_out_dir, exist_ok=True)
    dst_dir = os.path.join(code_out_dir, '{}'.format(date_time))
    shutil.copytree(src_dir, dst_dir,
                    ignore=shutil.ignore_patterns('pretrained*', '*logs*', 'out*', '*.png', '*.mp4', 'eval*',
                                                  '*__pycache__*', '*.git*', '*.idea*', '*.zip', '*.jpg',
                                                  'exp', 'debug', 'inpainting_ckpts', 'weights'))


def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None):  # pylint: disable=redefined-builtin
    # assert isinstance(input, torch.Tensor)
    if posinf is None:
        posinf = torch.finfo(input.dtype).max
    if neginf is None:
        neginf = torch.finfo(input.dtype).min
    assert nan == 0
    return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)


def float2uint8(x):
    return (255. * x).astype(np.uint8)


def float2uint16(x):
    return (65535 * x).astype(np.uint16)


def normalize_0_1(x):
    x_min, x_max = x.min(), x.max()
    return (x - x_min) / (x_max - x_min)


def img2mse(x, y, mask=None):
    '''
    :param x: img 1, [(...), 3]
    :param y: img 2, [(...), 3]
    :param mask: optional, [(...)]
    :return: mse score
    '''
    if mask is None:
        return torch.mean((x - y) * (x - y))
    else:
        return torch.sum((x - y) * (x - y) * mask.unsqueeze(-1)) / (torch.sum(mask) * x.shape[-1] + TINY_NUMBER)


mse2psnr = lambda x: -10. * np.log(x + TINY_NUMBER) / np.log(10.)


def img2psnr(x, y, mask=None):
    return mse2psnr(img2mse(x, y, mask).item())


def set_gpu(gpu):
    print('set gpu: {:s}'.format(gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def check_file(path):
    if not os.path.isfile(path):
        raise ValueError('file does not exist: {:s}'.format(path))


def check_path(path):
    if not os.path.exists(path):
        raise ValueError('path does not exist: {:s}'.format(path))


def ensure_path(path, remove=False):
    if os.path.exists(path):
        if remove:
            if input('{:s} exists, remove? ([y]/n): '.format(path)) != 'n':
                shutil.rmtree(path)
                os.makedirs(path)
    else:
        os.makedirs(path)


def count_params(model, return_str=True):
    n_params = 0
    for p in model.parameters():
        n_params += p.numel()
    if return_str:
        if n_params >= 1e6:
            return '{:.1f}M'.format(n_params / 1e6)
        else:
            return '{:.1f}K'.format(n_params / 1e3)
    else:
        return n_params


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.mean = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.mean = self.sum / self.count

    def item(self):
        return self.mean


class Timer(object):
    def __init__(self):
        self.start()

    def start(self):
        self.v = time.time()

    def end(self):
        return time.time() - self.v


def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t > 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)
