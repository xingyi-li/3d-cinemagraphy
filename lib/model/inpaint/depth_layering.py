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


import torch
from sklearn.cluster import AgglomerativeClustering


def get_depth_bins(depth=None, disparity=None, num_bins=None):
    """
    :param depth: [1, 1, H, W]
    :param disparity: [1, 1, H, W]
    :return: depth_bins
    """

    assert (disparity is not None) or (depth is not None)
    if disparity is None:
        assert depth.min() > 1e-2
        disparity = 1. / depth

    if depth is None:
        depth = 1. / torch.clamp(disparity, min=1e-2)

    assert depth.shape[:2] == (1, 1) and disparity.shape[:2] == (1, 1)
    disparity_max = disparity.max().item()
    disparity_min = disparity.min().item()
    disparity_feat = disparity[:, :, ::10, ::10].reshape(-1, 1).cpu().numpy()
    disparity_feat = (disparity_feat - disparity_min) / (disparity_max - disparity_min)
    if num_bins is None:
        n_clusters = None
        distance_threshold = 5
    else:
        n_clusters = num_bins
        distance_threshold = None
    result = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=distance_threshold).fit(disparity_feat)
    num_bins = result.n_clusters_ if n_clusters is None else n_clusters
    depth_bins = [depth.min().item()]
    for i in range(num_bins):
        th = (disparity_feat[result.labels_ == i]).min()
        th = th * (disparity_max - disparity_min) + disparity_min
        depth_bins.append(1. / th)

    depth_bins = sorted(depth_bins)
    depth_bins[0] = depth.min() - 1e-6
    depth_bins[-1] = depth.max() + 1e-6
    return depth_bins
