import math
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from kornia.filters import box_blur
from kornia.filters.kernels import get_binary_kernel2d
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter


def homogenize_np(coord):
    """
    append ones in the last dimension
    :param coord: [...., 2/3]
    :return: homogenous coordinates
    """
    return np.concatenate([coord, np.ones_like(coord[..., :1])], axis=-1)


def homogenize_pt(coord):
    return torch.cat([coord, torch.ones_like(coord[..., :1])], dim=-1)


def get_coord_grids_pt(h, w, device, homogeneous=False):
    """
    create pxiel coordinate grid
    :param h: height
    :param w: weight
    :param device: device
    :param homogeneous: if homogeneous coordinate
    :return: coordinates [h, w, 2]
    """
    y = torch.arange(0, h).to(device)
    x = torch.arange(0, w).to(device)
    grid_y, grid_x = torch.meshgrid(y, x)
    if homogeneous:
        return torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1)
    return torch.stack([grid_x, grid_y], dim=-1)  # [h, w, 2]


def normalize_for_grid_sample(coords, h, w):
    device = coords.device
    coords_normed = coords / torch.tensor([w-1., h-1.]).to(device) * 2. - 1.
    return coords_normed


def unproject_pts_np(intrinsics, coords, depth):
    if coords.shape[-1] == 2:
        coords = homogenize_np(coords)
    intrinsics = intrinsics.squeeze()[:3, :3]
    coords = np.linalg.inv(intrinsics).dot(coords.T) * depth.reshape(1, -1)
    return coords.T   # [n, 3]


def unproject_pts_pt(intrinsics, coords, depth):
    if coords.shape[-1] == 2:
        coords = homogenize_pt(coords)
    intrinsics = intrinsics.squeeze()[:3, :3]
    coords = torch.inverse(intrinsics).mm(coords.T) * depth.reshape(1, -1)
    return coords.T   # [n, 3]


def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
    """Transforms coordinates in the pixel frame to the camera frame.
    Args:
    depth: [batch, height, width]
    pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
    intrinsics: camera intrinsics [batch, 3, 3]
    is_homogeneous: return in homogeneous coordinates
    Returns:
    Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
    """
    if depth.ndim == 4:
        assert depth.shape[1] == 1
        depth = depth.squeeze(1)
    batch, height, width = depth.shape
    depth = depth.reshape(batch, 1, -1)
    pixel_coords = pixel_coords.reshape(batch, 3, -1)
    cam_coords = torch.inverse(intrinsics).bmm(pixel_coords) * depth
    if is_homogeneous:
        ones = torch.ones_like(depth)
        cam_coords = torch.cat([cam_coords, ones], dim=1)
    cam_coords = cam_coords.reshape(batch, -1, height, width)
    return cam_coords


def transform_pts_in_3D(pts, pose, return_homogeneous=False):
    '''
    :param pts: nx3, tensor
    :param pose: 4x4, tensor
    :return: nx3 or nx4, tensor
    '''
    pts_h = homogenize_pt(pts)
    pose = pose.squeeze()
    assert pose.shape == (4, 4)
    transformed_pts_h = pose.mm(pts_h.T).T  # [n, 4]
    if return_homogeneous:
        return transformed_pts_h
    return transformed_pts_h[..., :3]


def crop_boundary(x, ratio):
    h, w = x.shape[-2:]
    crop_h = int(h * ratio)
    crop_w = int(w * ratio)
    return x[:, :, crop_h:h-crop_h, crop_w:w-crop_w]


def masked_smooth_filter(x, mask, kernel_size=9, sigma=1):
    '''
    :param x: [B, n, h, w]
    :param mask: [B, 1, h, w]
    :return: [B, n, h, w]
    '''
    x_ = x * mask
    x_ = box_blur(x_, (kernel_size, kernel_size), border_type='constant')
    mask_ = box_blur(mask, (kernel_size, kernel_size), border_type='constant')
    x_ = x_ / torch.clamp(mask_, min=1e-6)
    mask_bool = (mask.repeat(1, x.shape[1], 1, 1) > 1e-6).float()
    out = mask_bool * x + (1. - mask_bool) * x_
    return out, mask_


def remove_noise_in_dpt_disparity(disparity, kernel_size=5):
    return median_filter(disparity, size=kernel_size)


def _compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
    r"""Utility function that computes zero padding tuple."""
    computed: List[int] = [(k - 1) // 2 for k in kernel_size]
    return computed[0], computed[1]


def masked_median_blur(input, mask, kernel_size=9):
    assert input.shape == mask.shape
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    padding: Tuple[int, int] = _compute_zero_padding((kernel_size, kernel_size))

    # prepare kernel
    kernel: torch.Tensor = get_binary_kernel2d((kernel_size, kernel_size)).to(input)
    b, c, h, w = input.shape

    # map the local window to single vector
    features: torch.Tensor = F.conv2d(input.reshape(b * c, 1, h, w), kernel, padding=padding, stride=1)
    masks: torch.Tensor = F.conv2d(mask.reshape(b * c, 1, h, w), kernel, padding=padding, stride=1)
    features = features.view(b, c, -1, h, w).permute(0, 1, 3, 4, 2)  # BxCxxHxWx(K_h * K_w)
    min_value, max_value = features.min(), features.max()
    masks = masks.view(b, c, -1, h, w).permute(0, 1, 3, 4, 2)  # BxCxHxWx(K_h * K_w)
    index_invalid = (1 - masks).nonzero(as_tuple=True)
    index_b, index_c, index_h, index_w, index_k = index_invalid
    features[(index_b[::2], index_c[::2], index_h[::2], index_w[::2], index_k[::2])] = min_value
    features[(index_b[1::2], index_c[1::2], index_h[1::2], index_w[1::2], index_k[1::2])] = max_value
    # compute the median along the feature axis
    median: torch.Tensor = torch.median(features, dim=-1)[0]

    return median


def define_camera_path(num_frames, x, y, z, path_type='circle', return_t_only=False):
    generic_pose = np.eye(4)
    tgt_poses = []
    if path_type == 'straight-line':
        corner_points = np.array([[0, 0, 0], [(0 + x) * 0.5, (0 + y) * 0.5, (0 + z) * 0.5], [x, y, z]])
        corner_t = np.linspace(0, 1, len(corner_points))
        t = np.linspace(0, 1, num_frames)
        cs = interp1d(corner_t, corner_points, axis=0, kind='quadratic')
        spline = cs(t)
        xs, ys, zs = [xx.squeeze() for xx in np.split(spline, 3, 1)]
    elif path_type == 'double-straight-line':
        corner_points = np.array([[-x, -y, -z], [0, 0, 0], [x, y, z]])
        corner_t = np.linspace(0, 1, len(corner_points))
        t = np.linspace(0, 1, num_frames)
        cs = interp1d(corner_t, corner_points, axis=0, kind='quadratic')
        spline = cs(t)
        xs, ys, zs = [xx.squeeze() for xx in np.split(spline, 3, 1)]
    elif path_type == 'circle':
        xs, ys, zs = [], [], []
        for frame_id, bs_shift_val in enumerate(np.arange(-2.0, 2.0, (4./num_frames))):
            xs += [np.cos(bs_shift_val * np.pi) * 1 * x]
            ys += [np.sin(bs_shift_val * np.pi) * 1 * y]
            zs += [np.cos(bs_shift_val * np.pi/2.) * 1 * z]
        xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
    elif path_type == 'debug':
        xs = np.array([x, 0, -x, 0, 0])
        ys = np.array([0, y, 0, -y, 0])
        zs = np.array([0, 0, 0, 0, z])
    else:
        raise NotImplementedError

    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
    if return_t_only:
        return np.stack([xs, ys, zs], axis=1)  # [n, 3]
    for xx, yy, zz in zip(xs, ys, zs):
        tgt_poses.append(generic_pose * 1.)
        tgt_poses[-1][:3, -1] = np.array([xx, yy, zz])
    return tgt_poses


def make_circle(n_frames, x_lim, y_lim, z_lim):
    if isinstance(x_lim, (list, tuple)):
        x_lim = x_lim[-1]
    if isinstance(y_lim, (list, tuple)):
        y_lim = y_lim[-1]
    if not isinstance(z_lim, (list, tuple)):
        z_lim = (-z_lim, z_lim)
    assert len(z_lim) == 2, \
        'z_lim must have two values, got {:d}'.format(len(z_lim))

    R = torch.eye(3)
    tics = torch.linspace(-2, 2, n_frames)
    xs = torch.cos(tics * math.pi) * x_lim
    ys = torch.sin(tics * math.pi) * y_lim
    dz = z_lim[1] - z_lim[0]
    zs = (torch.cos(tics * math.pi / 2) + 1) * dz / 2 + z_lim[0]
    ts = torch.stack([-xs, -ys, -zs], -1).unsqueeze(-1)
    Ms = torch.cat([R.repeat(n_frames, 1, 1), ts], -1)
    return Ms


def make_swing(n_frames, x_lim, z_lim):
    return make_circle(n_frames, x_lim, 0, z_lim)


def make_ken_burns(n_frames, x_lim, y_lim, z_lim):
    if not isinstance(x_lim, (list, tuple)):
        x_lim = (-x_lim, x_lim)
    if not isinstance(y_lim, (list, tuple)):
        y_lim = (-y_lim, y_lim)
    if not isinstance(z_lim, (list, tuple)):
        z_lim = (-z_lim, z_lim)
    assert len(x_lim) == 2, \
        'x_lim must have two values, got {:d}'.format(len(x_lim))
    assert len(y_lim) == 2, \
        'y_lim must have two values, got {:d}'.format(len(y_lim))
    assert len(z_lim) == 2, \
        'z_lim must have two values, got {:d}'.format(len(z_lim))

    R = torch.eye(3)
    xs = torch.linspace(x_lim[0], x_lim[1], n_frames)
    ys = torch.linspace(y_lim[0], y_lim[1], n_frames)
    zs = torch.linspace(z_lim[0], z_lim[1], n_frames)
    ts = torch.stack([-xs, -ys, -zs], -1).unsqueeze(-1)
    Ms = torch.cat([R.repeat(n_frames, 1, 1), ts], -1)
    return Ms


def make_zoom(n_frames, z_lim):
    return make_ken_burns(n_frames, 0, 0, z_lim)


def make_dolly_zoom(n_frames, z_lim, fov, ctr_depth):
    Ms = make_zoom(n_frames, z_lim)
    plane_width = math.tan(fov / 2) * ctr_depth
    ctr_depths = ctr_depth + Ms[:, 2, 3]
    fovs = 2 * torch.atan2(plane_width, ctr_depths)
    return Ms, fovs
