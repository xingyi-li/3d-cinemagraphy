# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import torch
import torch.nn as nn
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
)


class PointsRenderer(nn.Module):
    """
    A class for rendering a batch of points. The class should
    be initialized with a rasterizer and compositor class which each have a forward
    function.
    """

    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def to(self, device):
        # Manually move to device rasterizer as the cameras
        # within the class are not of type nn.Module
        self.rasterizer = self.rasterizer.to(device)
        self.compositor = self.compositor.to(device)
        return self

    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        if type(r) == torch.Tensor:
            if r.shape[-1] > 1:
                idx = fragments.idx.clone()
                idx[idx == -1] = 0
                r = r[:, idx.squeeze().long()]
                r = r.permute(0, 3, 1, 2)

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images


def linear_interpolation(data0, data1, time):
    return (1. - time) * data0 + time * data1


def create_pcd_renderer(args, h, w, intrinsics, R=None, T=None, radius=None, device="cuda"):
    # Initialize a camera.
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    if R is None:
        R = torch.eye(3)[None]  # (1, 3, 3)
    if T is None:
        T = torch.zeros(1, 3)  # (1, 3)
    cameras = PerspectiveCameras(R=R, T=T,
                                 device=device,
                                 focal_length=((-fx, -fy),),
                                 principal_point=(tuple(intrinsics[:2, -1]),),
                                 image_size=((h, w),),
                                 in_ndc=False,
                                 )

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters.
    if radius is None:
        radius = args.point_radius / min(h, w) * 2.0
        if args.vary_pts_radius:
            if np.random.choice([0, 1], p=[0.6, 0.4]):
                factor = 1 + (0.2 * (np.random.rand() - 0.5))
                radius *= factor

    raster_settings = PointsRasterizationSettings(
        image_size=(h, w),
        radius=radius,
        points_per_pixel=8,
    )

    # Create a points renderer by compositing points using an alpha compositor (nearer points
    # are weighted more heavily). See [1] for an explanation.
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )
    return renderer
