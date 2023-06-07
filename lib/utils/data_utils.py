from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np
import cv2
import torch
import av
import lz4framed
import pickle
import random


def load_compressed_tensor(filename):
    retval = None
    with open(filename, mode='rb') as file:
        retval = torch.from_numpy(pickle.loads(lz4framed.decompress(file.read())))
    return retval


class VideoReader(object):
    """
    Wrapper for PyAV

    Read frames from a video file into numpy tensors. Example:

    file = VideoReader(filename)
    video_frames = file[start_frame:end_frame]

    If desired, a table-of-contents (ToC) file can be provided to speed up the loading/seeking time.
    """

    def __init__(self, file, toc=None, format=None):
        if not hasattr(file, 'read'):
            file = str(file)
        self.file = file
        self.format = format
        self._container = None

        with av.open(self.file, format=self.format) as container:
            stream = [s for s in container.streams if s.type == 'video'][0]
            self.bit_rate = stream.bit_rate

            # Build a toc
            if toc is None:
                packet_lengths = []
                packet_ts = []
                for packet in container.demux(stream):
                    if packet.stream.type == 'video':
                        decoded = packet.decode()
                        if len(decoded) > 0:
                            packet_lengths.append(len(decoded))
                            packet_ts.append(decoded[0].pts)
                self._toc = {
                    'lengths': packet_lengths,
                    'ts': packet_ts,
                }
            else:
                self._toc = toc

            self._toc_cumsum = np.cumsum(self.toc['lengths'])
            self._len = self._toc_cumsum[-1]

            # PyAV always returns frames in color, and we make that
            # assumption in get_frame() later below, so 3 is hardcoded here:
            self._im_sz = stream.height, stream.width, 3
            self._time_base = stream.time_base
            self.rate = stream.average_rate

        self._load_fresh_file()

    @staticmethod
    def _next_video_packet(container_iter):
        for packet in container_iter:
            if packet.stream.type == 'video':
                decoded = packet.decode()
                if len(decoded) > 0:
                    return decoded

        raise ValueError("Could not find any video packets.")

    def _load_fresh_file(self):
        if self._container is not None:
            self._container.close()

        if hasattr(self.file, 'seek'):
            self.file.seek(0)

        self._container = av.open(self.file, format=self.format)
        demux = self._container.demux(self._video_stream)
        self._current_packet = self._next_video_packet(demux)
        self._current_packet_no = 0

    @property
    def _video_stream(self):
        return [s for s in self._container.streams if s.type == 'video'][0]

    def __len__(self):
        return self._len

    def __del__(self):
        if self._container is not None:
            self._container.close()

    def __getitem__(self, item):
        if isinstance(item, int):
            item = slice(item, item + 1)

        if item.start < 0 or item.start >= len(self):
            raise IndexError(f"start index ({item.start}) out of range")

        if item.stop < 0 or item.stop > len(self):
            raise IndexError(f"stop index ({item.stop}) out of range")

        return np.stack([self.get_frame(i) for i in range(item.start, item.stop)])

    @property
    def frame_shape(self):
        return self._im_sz

    @property
    def toc(self):
        return self._toc

    def get_frame(self, j):
        # Find the packet this frame is in.
        packet_no = self._toc_cumsum.searchsorted(j, side='right')
        self._seek_packet(packet_no)
        # Find the location of the frame within the packet.
        if packet_no == 0:
            loc = j
        else:
            loc = j - self._toc_cumsum[packet_no - 1]
        frame = self._current_packet[loc]  # av.VideoFrame

        return frame.to_ndarray(format='rgb24')

    def _seek_packet(self, packet_no):
        """Advance through the container generator until we get the packet
        we want. Store that packet in selfpp._current_packet."""
        packet_ts = self.toc['ts'][packet_no]
        # Only seek when needed.
        if packet_no == self._current_packet_no:
            return
        elif (packet_no < self._current_packet_no
              or packet_no > self._current_packet_no + 1):
            self._container.seek(packet_ts, stream=self._video_stream)

        demux = self._container.demux(self._video_stream)
        self._current_packet = self._next_video_packet(demux)
        while self._current_packet[0].pts < packet_ts:
            self._current_packet = self._next_video_packet(demux)

        self._current_packet_no = packet_no


def read_flo(strFile):
    with open(strFile, 'rb') as objFile:
        strFlow = objFile.read()

    assert (np.frombuffer(buffer=strFlow, dtype=np.float32, count=1, offset=0) == 202021.25)

    intWidth = np.frombuffer(buffer=strFlow, dtype=np.int32, count=1, offset=4)[0]
    intHeight = np.frombuffer(buffer=strFlow, dtype=np.int32, count=1, offset=8)[0]

    return np.frombuffer(buffer=strFlow, dtype=np.float32, count=intHeight * intWidth * 2, offset=12).reshape(
        [intHeight, intWidth, 2])


def get_params(config, size=(1920, 1024), crop_size=1024):
    w, h = size
    x = random.randint(0, np.maximum(0, w - crop_size))
    y = random.randint(0, np.maximum(0, h - crop_size))

    flip = random.random() > 0.5
    if config['no_flip']:
        flip = False
    colorjitter = random.random() > 0.5
    if config['use_color_jitter']:
        colorjitter = random.random() > 0.5
    else:
        colorjitter = False
    colorjitter_params = {}
    colorjitter_params['brightness'] = (torch.rand(1) * 0.2 + 1.0).numpy()[0]
    colorjitter_params['contrast'] = (torch.rand(1) * 0.2 + 1.0).numpy()[0]
    colorjitter_params['saturation'] = (torch.rand(1) * 0.2 + 1.0).numpy()[0]
    colorjitter_params['hue'] = (torch.rand(1) * 0.05).numpy()[0]
    # brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05

    return {'crop_pos': (x, y), 'crop_size': crop_size, 'flip': flip, 'colorjitter': colorjitter,
            'colorjitter_params': colorjitter_params}


def __scale_width(img, target_width, method=InterpolationMode.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __colorjitter(img, colorjitter, colorjitter_params):
    if colorjitter:
        brightness = colorjitter_params['brightness']  # 0.2
        contrast = colorjitter_params['contrast']  # 0.2
        saturation = colorjitter_params['saturation']  # 0.2
        hue = colorjitter_params['hue']  # 0.05
        return transforms.ColorJitter(brightness=[brightness, brightness],
                                      contrast=[contrast, contrast],
                                      saturation=[saturation, saturation],
                                      hue=[hue, hue])(img)
    return img


def get_transform(config, size, params, method=InterpolationMode.BICUBIC, normalize=True):
    transform_list = []
    if 'crop' in config['resize_or_crop']:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], params['crop_size'])))
    if 'resize' in config['resize_or_crop']:
        osize = [config['W'], config['W']]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in config['resize_or_crop']:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, config['W'], method)))
    if not config['no_flip']:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    if config['use_color_jitter']:
        transform_list.append(
            transforms.Lambda(lambda img: __colorjitter(img, params['colorjitter'], params['colorjitter_params'])))
    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size

    def __call__(self, img):
        return img[(self.h - self.th) // 2:(self.h + self.th) // 2, (self.w - self.tw) // 2:(self.w + self.tw) // 2, :]


def resize_img_intrinsic(img, intrinsic, w_out, h_out):
    h, w = img.shape[:2]
    if w_out > w:
        interpolation_method = cv2.INTER_LINEAR
    else:
        interpolation_method = cv2.INTER_AREA
    img = cv2.resize(img, (int(w_out), int(h_out)), interpolation=interpolation_method)
    intrinsic[0] *= 1. * w_out / w
    intrinsic[1] *= 1. * h_out / h
    return img, intrinsic


def resize_img(img, ds_factor, w_out=None, h_out=None):
    h, w = img.shape[:2]
    if w_out is None and h_out is None:
        if ds_factor == 1:
            return img
        if ds_factor > 1:
            interpolation_method = cv2.INTER_LINEAR
        else:
            interpolation_method = cv2.INTER_AREA
        img = cv2.resize(img, (int(w * ds_factor), int(h * ds_factor)), interpolation=interpolation_method)
    else:
        if w_out > w:
            interpolation_method = cv2.INTER_LINEAR
        else:
            interpolation_method = cv2.INTER_AREA
        img = cv2.resize(img, (int(w_out), int(h_out)), interpolation=interpolation_method)
    return img


def get_src_tgt_ids(num_frames, max_interval=1):
    assert num_frames > max_interval + 1
    src_id1 = np.random.choice(num_frames - max_interval - 1)
    interval = np.random.randint(low=0, high=max_interval) + 1
    src_id2 = src_id1 + interval + 1
    tgt_id = np.random.randint(src_id1 + 1, src_id2)
    return src_id1, src_id2, tgt_id


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.intrinsics = np.array([[fx, 0, cx, 0],
                                    [0, fy, cy, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def unnormalize_intrinsics(intrinsics, h, w):
    intrinsics[0] *= w
    intrinsics[1] *= h
    return intrinsics


def parse_pose_file(file):
    f = open(file, 'r')
    cam_params = {}
    for i, line in enumerate(f):
        if i == 0:
            video_id = line.replace('https://www.youtube.com/watch?v=', '')[:-1]
            continue
        entry = [float(x) for x in line.split()]
        id = int(entry[0])
        cam_params[id] = Camera(entry)
    return video_id, cam_params


def crop_img(img, factor=16):
    h, w = img.shape[:2]
    ho = h // factor * factor
    wo = w // factor * factor
    img = img[:ho, :wo]
    return img
