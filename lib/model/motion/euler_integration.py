import torch
import torch.nn as nn


def euler_integration(motion, destination_frame, return_all_frames=False):
    """
    This function is provided by Aleksander Hołyński <holynski@cs.washington.edu>
    Repeatedly integrates the Eulerian motion field to get the displacement map to a future frame.

    :param motion: The Eulerian motion field to be integrated.
    :param destination_frame: The number of times the motion field should be integrated.
    :param return_all_frames: Return the displacement maps to all intermediate frames, not only the last frame.
    :return: The displacement map resulting from repeated integration of the motion field.
    """

    assert (motion.dim() == 4)
    b, c, height, width = motion.shape
    assert (b == 1), 'Function only implemented for batch = 1'
    assert (c == 2), f'Input motion field should be Bx2xHxW. Given tensor is: {motion.shape}'

    y, x = torch.meshgrid(
        [torch.linspace(0, height - 1, height, device='cuda'),
         torch.linspace(0, width - 1, width, device='cuda')])
    coord = torch.stack([x, y], dim=0).long()

    destination_coords = coord.clone().float()
    destination_coords_ = destination_coords.clone().float()

    if return_all_frames:
        displacements = torch.zeros(destination_frame + 1, 2, height, width, device='cuda')
        visible_pixels = torch.ones(b + 1, 1, height, width, device='cuda')
    else:
        displacements = torch.zeros(1, 2, height, width, device='cuda')
        visible_pixels = torch.ones(1, 1, height, width, device='cuda')
    invalid_mask = torch.zeros(1, height, width, device='cuda').bool()
    for frame_id in range(1, destination_frame + 1):
        destination_coords = destination_coords + motion[0][:, torch.round(destination_coords[1]).long(),
                                                  torch.round(destination_coords[0]).long()]
        out_of_bounds_x = torch.logical_or(destination_coords[0] > (width - 1), destination_coords[0] < 0)
        out_of_bounds_y = torch.logical_or(destination_coords[1] > (height - 1), destination_coords[1] < 0)
        invalid_mask = torch.logical_or(out_of_bounds_x.unsqueeze(0), invalid_mask)
        invalid_mask = torch.logical_or(out_of_bounds_y.unsqueeze(0), invalid_mask)

        # Set the displacement of invalid pixels to zero, to avoid out-of-bounds access errors
        destination_coords[invalid_mask.expand_as(destination_coords)] = coord[
            invalid_mask.expand_as(destination_coords)].float()
        # DEBUG
        destination_coords_ = destination_coords_ + motion[0][:, torch.round(destination_coords[1]).long(),
                                                    torch.round(destination_coords[0]).long()]
        if return_all_frames:
            displacements[frame_id] = (destination_coords_ - coord.float()).unsqueeze(0)
            # Set the displacements for invalid pixels to be out of bounds.
            displacements[frame_id][invalid_mask] = torch.max(height, width) + 1
            visible_pixels[frame_id] = torch.logical_not(invalid_mask.clone()).float().unsqueeze(0)
        else:
            displacements = (destination_coords_ - coord.float()).unsqueeze(0)
            visible_pixels = torch.logical_not(invalid_mask.clone()).float().unsqueeze(0)
    return displacements, visible_pixels


class EulerIntegration(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, motion, destination_frame, return_all_frames=False, show_visible_pixels=False):
        displacements = torch.zeros(motion.shape).to(motion.device)
        visible_pixels = torch.zeros(motion.shape[0], 1, motion.shape[2], motion.shape[3])
        for b in range(motion.shape[0]):
            displacements[b:b + 1], visible_pixels[b:b + 1] = euler_integration(motion[b:b + 1], destination_frame[b])

        if show_visible_pixels:
            return displacements, visible_pixels
        else:
            return displacements
