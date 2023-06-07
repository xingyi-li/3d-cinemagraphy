import os
import imageio
import torch.utils.data.distributed
from pytorch3d.structures import Pointclouds
from lib.utils.render_utils import *
from lib.utils.general_utils import normalize_0_1
from lib.model.inpaint.depth_layering import get_depth_bins
from lib.model.inpaint.pcd import create_pcd_renderer
from lib.model.motion.euler_integration import EulerIntegration
from kornia.morphology import erosion


class ImgRenderer():
    def __init__(self, args, config, model, scene_flow_estimator, inpainter, device):
        self.args = args
        self.config = config
        self.model = model
        self.scene_flow_estimator = scene_flow_estimator
        self.inpainter = inpainter
        self.euler_integration = EulerIntegration()
        self.device = device

    def process_data(self, data):
        if 'motion_rgbs' in data.keys():
            self.motion_rgbs = data['motion_rgbs'].to(self.device)
        if 'motions' in data.keys():
            self.motions = data['motions'].to(self.device)
        if 'hints' in data.keys():
            self.hints = data['hints'].to(self.device)
        if 'mask' in data.keys():
            self.mask = data['mask'].to(self.device)
        if 'frames' in data.keys():
            frames = data['frames']
            self.start_frame = frames[0].to(self.device)
            self.middle_frame = frames[1].to(self.device)
            self.end_frame = frames[2].to(self.device)
        if 'frames_depth' in data.keys():
            frames_depth = data['frames_depth']
            self.start_depth = frames_depth[0].to(self.device)
            self.end_depth = frames_depth[1].to(self.device)
        if 'frames_index' in data.keys():
            frames_index = data['frames_index']
            self.start_index = frames_index[0]
            self.middle_index = frames_index[1]
            self.end_index = frames_index[2]
            self.time = ((self.middle_index.float() - self.start_index.float()).float() / (
                    self.end_index.float() - self.start_index.float() + 1.0).float()).item()

        self.src_img = data['src_img'].to(self.device)
        self.h, self.w = self.src_img.shape[-2:]
        self.src_depth = data['src_depth'].to(self.device)
        self.intrinsic = data['intrinsic'].to(self.device)

        self.pose = data['pose'].to(self.device)
        self.scale_shift = data['scale_shift'][0].to(self.device)

        if 'tgt_img' in data.keys():
            self.tgt_img = data['tgt_img'].to(self.device)
        if 'tgt_pose' in data.keys():
            self.tgt_pose = data['tgt_pose'].to(self.device)
        if 'src_mask' in data.keys():
            self.src_mask = data['src_mask'].to(self.device)
        else:
            self.src_mask = torch.ones_like(self.src_depth)

    def feature_extraction(self, rgba_layers, mask_layers, depth_layers):
        rgba_layers_in = rgba_layers.squeeze(1)

        if self.config['spacetime_model']['use_inpainting_mask_for_feature']:
            rgba_layers_in = torch.cat([rgba_layers_in, mask_layers.squeeze(1)], dim=1)

        if self.config['spacetime_model']['use_depth_for_feature']:
            rgba_layers_in = torch.cat([rgba_layers_in, 1. / torch.clamp(depth_layers.squeeze(1), min=1.)], dim=1)
        featmaps = self.model.feature_net(rgba_layers_in)
        return featmaps

    def apply_scale_shift(self, depth, scale, shift):
        disp = 1. / torch.clamp(depth, min=1e-3)
        disp = scale * disp + shift
        return 1 / torch.clamp(disp, min=1e-3 * scale)

    def masked_diffuse(self, x, mask, iter=10, kernel_size=35, median_blur=False):
        if median_blur:
            x = masked_median_blur(x, mask.repeat(1, x.shape[1], 1, 1), kernel_size=5)
        for _ in range(iter):
            x, mask = masked_smooth_filter(x, mask, kernel_size=kernel_size)
        return x, mask

    def compute_weight_for_two_frame_blending(self, time, disp1, disp2, alpha1, alpha2):
        alpha = 4
        weight1 = (1 - time) * torch.exp(alpha * disp1) * alpha1
        weight2 = time * torch.exp(alpha * disp2) * alpha2
        sum_weight = torch.clamp(weight1 + weight2, min=1e-6)
        out_weight1 = weight1 / sum_weight
        out_weight2 = weight2 / sum_weight
        return out_weight1, out_weight2

    def transform_all_pts(self, all_pts, pose):
        all_pts_out = []
        for pts in all_pts:
            pts_out = transform_pts_in_3D(pts, pose)
            all_pts_out.append(pts_out)
        return all_pts_out

    def render_pcd(self, pts, rgbs, feats, mask, side_ids, R=None, t=None, time=0, t_step=None, path_type=None):
        rgb_feat = torch.cat([rgbs, feats], dim=-1)

        num_sides = side_ids.max() + 1
        assert num_sides == 1 or num_sides == 2

        if R is None:
            R = torch.eye(3, device=self.device)
        if t is None:
            t = torch.zeros(3, device=self.device)

        pts_ = (R.mm(pts.T) + t.unsqueeze(-1)).T
        if self.config['spacetime_model']['adaptive_pts_radius']:
            radius = self.config['spacetime_model']['point_radius'] / min(self.h, self.w) * 2.0 * pts[..., -1][None] / \
                     torch.clamp(pts_[..., -1][None], min=1e-6)
        else:
            radius = self.config['spacetime_model']['point_radius'] / min(self.h, self.w) * 2.0

        if self.config['spacetime_model']['vary_pts_radius'] and np.random.choice([0, 1], p=[0.6, 0.4]):
            if type(radius) == torch.Tensor:
                factor = 1 + (0.2 * (torch.rand_like(radius) - 0.5))
            else:
                factor = 1 + (0.2 * (np.random.rand() - 0.5))
            radius *= factor

        if self.config['spacetime_model']['use_mask_for_decoding']:
            rgb_feat = torch.cat([rgb_feat, mask], dim=-1)

        if self.config['spacetime_model']['use_depth_for_decoding']:
            disp = normalize_0_1(1. / torch.clamp(pts_[..., [-1]], min=1e-6))
            rgb_feat = torch.cat([rgb_feat, disp], dim=-1)

        global_out_list = []
        direct_color_out_list = []
        meta = {}
        for j in range(num_sides):
            mask_side = side_ids == j
            renderer = create_pcd_renderer(self.args, self.h, self.w, self.intrinsic.squeeze()[:3, :3],
                                           radius=radius[:, mask_side] if type(radius) == torch.Tensor else radius)
            all_pcd_j = Pointclouds(points=[pts_[mask_side]], features=[rgb_feat[mask_side]])
            global_out_j = renderer(all_pcd_j)
            all_colored_pcd_j = Pointclouds(points=[pts_[mask_side]], features=[rgbs[mask_side]])
            direct_rgb_out_j = renderer(all_colored_pcd_j)

            global_out_list.append(global_out_j)
            direct_color_out_list.append(direct_rgb_out_j)

        w1, w2 = self.compute_weight_for_two_frame_blending(time,
                                                            global_out_list[0][..., [-1]],
                                                            global_out_list[-1][..., [-1]],
                                                            global_out_list[0][..., [3]],
                                                            global_out_list[-1][..., [3]]
                                                            )
        direct_rgb_out = w1 * direct_color_out_list[0] + w2 * direct_color_out_list[-1]
        pred_rgb = self.model.img_decoder(global_out_list[0].permute(0, 3, 1, 2),
                                          global_out_list[-1].permute(0, 3, 1, 2),
                                          time)

        direct_rgb = direct_rgb_out[..., :3].permute(0, 3, 1, 2)
        acc = 0.5 * (global_out_list[0][..., [3]] + global_out_list[-1][..., [3]]).permute(0, 3, 1, 2)
        meta['acc'] = acc

        if self.args.save_frames:
            out_folder = os.path.join(self.args.input_dir, 'output')
            video_out_folder = os.path.join(out_folder, 'visuals', 'demo', self.args.split, self.args.scene_id,
                                            'frames',
                                            f'{path_type}')
            os.makedirs(video_out_folder, exist_ok=True)
            imageio.imwrite(os.path.join(video_out_folder, f'{self.args.scene_id}_pred_rgb_{t_step}.png'),
                            (255. * pred_rgb.detach().cpu().squeeze().permute(1, 2, 0).numpy()).astype(np.uint8))

        return pred_rgb, direct_rgb, meta

    def render_pcd_for_novel_view(self, pts, rgbs, feats, mask, R=None, t=None):
        rgb_feat = torch.cat([rgbs, feats], dim=-1)

        if R is None:
            R = torch.eye(3, device=self.device)
        if t is None:
            t = torch.zeros(3, device=self.device)

        pts_ = (R.mm(pts.T) + t.unsqueeze(-1)).T
        if self.config['spacetime_model']['adaptive_pts_radius']:
            radius = self.config['spacetime_model']['point_radius'] / min(self.h, self.w) * 2.0 * pts[..., -1][None] / \
                     torch.clamp(pts_[..., -1][None], min=1e-6)
        else:
            radius = self.config['spacetime_model']['point_radius'] / min(self.h, self.w) * 2.0

        if self.config['spacetime_model']['vary_pts_radius'] and np.random.choice([0, 1], p=[0.6, 0.4]):
            if type(radius) == torch.Tensor:
                factor = 1 + (0.2 * (torch.rand_like(radius) - 0.5))
            else:
                factor = 1 + (0.2 * (np.random.rand() - 0.5))
            radius *= factor

        if self.config['spacetime_model']['use_mask_for_decoding']:
            rgb_feat = torch.cat([rgb_feat, mask], dim=-1)

        if self.config['spacetime_model']['use_depth_for_decoding']:
            disp = normalize_0_1(1. / torch.clamp(pts_[..., [-1]], min=1e-6))
            rgb_feat = torch.cat([rgb_feat, disp], dim=-1)

        meta = {}

        renderer = create_pcd_renderer(self.args, self.h, self.w, self.intrinsic.squeeze()[:3, :3],
                                       radius=radius)
        all_pcd = Pointclouds(points=[pts_], features=[rgb_feat])
        global_out = renderer(all_pcd)
        all_colored_pcd = Pointclouds(points=[pts_], features=[rgbs])
        direct_rgb_out = renderer(all_colored_pcd)

        pred_rgb = self.model.img_decoder(global_out.permute(0, 3, 1, 2),
                                          global_out.permute(0, 3, 1, 2),
                                          0)

        direct_rgb = direct_rgb_out[..., :3].permute(0, 3, 1, 2)
        acc = 0.5 * (global_out[..., [3]] + global_out[..., [3]]).permute(0, 3, 1, 2)
        meta['acc'] = acc
        return pred_rgb, direct_rgb, meta

    def get_reprojection_mask(self, pts, R, t):
        pts1_ = (R.mm(pts.T) + t.unsqueeze(-1)).T
        mask1 = torch.ones_like(self.src_img[:, :1].reshape(-1, 1))
        mask_renderer = create_pcd_renderer(self.args, self.h, self.w, self.intrinsic.squeeze()[:3, :3],
                                            radius=1.0 / min(self.h, self.w) * 4.)
        mask_pcd = Pointclouds(points=[pts1_], features=[mask1])
        mask = mask_renderer(mask_pcd).permute(0, 3, 1, 2)
        mask = F.max_pool2d(mask, kernel_size=7, stride=1, padding=3)
        return mask

    def get_cropping_ids(self, mask):
        assert mask.shape[:2] == (1, 1)
        mask = mask.squeeze()
        h, w = mask.shape
        mask_mean_x_axis = mask.mean(dim=0)
        x_valid = torch.nonzero(mask_mean_x_axis > 0.5)
        bad = False
        if len(x_valid) < 0.75 * w:
            left, right = 0, w - 1  # invalid
            bad = True
        else:
            left, right = x_valid[0][0], x_valid[-1][0]
        mask_mean_y_axis = mask.mean(dim=1)
        y_valid = torch.nonzero(mask_mean_y_axis > 0.5)
        if len(y_valid) < 0.75 * h:
            top, bottom = 0, h - 1  # invalid
            bad = True
        else:
            top, bottom = y_valid[0][0], y_valid[-1][0]
        assert 0 <= top <= h - 1 and 0 <= bottom <= h - 1 and 0 <= left <= w - 1 and 0 <= right <= w - 1
        return top, bottom, left, right, bad

    def render_depth_from_mdi(self, depth_layers, alpha_layers):
        '''
        :param depth_layers: [n_layers, 1, h, w]
        :param alpha_layers: [n_layers, 1, h, w]
        :return: rendered depth [1, 1, h, w]
        '''
        num_layers = len(depth_layers)
        h, w = depth_layers.shape[-2:]
        layer_id = torch.arange(num_layers, device=self.device).float()
        layer_id_maps = layer_id[..., None, None, None, None].repeat(1, 1, 1, h, w)
        T = torch.cumprod(1. - alpha_layers, dim=0)[:-1]
        T = torch.cat([torch.ones_like(T[:1]), T], dim=0)
        weights = alpha_layers * T
        depth_map = torch.sum(weights * depth_layers, dim=0)
        depth_map = torch.clamp(depth_map, min=1.)
        layer_id_map = torch.sum(weights * layer_id_maps, dim=0)
        return depth_map, layer_id_map

    def render_rgbda_layers_from_one_view(self):
        depth_bins = get_depth_bins(depth=self.src_depth1)
        rgba_layers, depth_layers, mask_layers = \
            self.inpainter.sequential_inpainting(self.src_img1, self.src_depth1, depth_bins)
        coord1 = get_coord_grids_pt(self.h, self.w, device=self.device).float()
        src_depth1 = self.apply_scale_shift(self.src_depth1, self.scale_shift1[0], self.scale_shift1[1])
        pts1 = unproject_pts_pt(self.intrinsic1, coord1.reshape(-1, 2), src_depth1.flatten())

        featmaps = self.feature_extraction(rgba_layers, mask_layers, depth_layers)
        depth_layers = self.apply_scale_shift(depth_layers, self.scale_shift1[0], self.scale_shift1[1])
        num_layers = len(rgba_layers)
        all_pts = []
        all_rgbas = []
        all_feats = []
        all_masks = []
        for i in range(num_layers):
            alpha_i = rgba_layers[i][:, -1] > 0.5
            rgba_i = rgba_layers[i]
            mask_i = mask_layers[i]
            featmap = featmaps[i][None]
            featmap = F.interpolate(featmap, size=(self.h, self.w), mode='bilinear', align_corners=True)
            pts1_i = unproject_pts_pt(self.intrinsic1, coord1.reshape(-1, 2), depth_layers[i].flatten())
            pts1_i = pts1_i.reshape(1, self.h, self.w, 3)
            all_pts.append(pts1_i[alpha_i])
            all_rgbas.append(rgba_i.permute(0, 2, 3, 1)[alpha_i])
            all_feats.append(featmap.permute(0, 2, 3, 1)[alpha_i])
            all_masks.append(mask_i.permute(0, 2, 3, 1)[alpha_i])

        all_pts = torch.cat(all_pts)
        all_rgbas = torch.cat(all_rgbas)
        all_feats = torch.cat(all_feats)
        all_masks = torch.cat(all_masks)
        all_side_ids = torch.zeros_like(all_masks.squeeze(), dtype=torch.long)

        R = self.tgt_pose[0, :3, :3]
        t = self.tgt_pose[0, :3, 3]

        pred_img, direct_rgb_out, meta = self.render_pcd(all_pts, all_pts,
                                                         all_rgbas, all_rgbas,
                                                         all_feats, all_feats,
                                                         all_masks, all_side_ids,
                                                         R, t, 0)

        mask = self.get_reprojection_mask(pts1, R, t)
        t, b, l, r, bad = self.get_cropping_ids(mask)
        gt_img = self.src_img2
        skip = False
        if not skip and not self.args.eval_mode:
            pred_img = pred_img[:, :, t:b, l:r]
            mask = mask[:, :, t:b, l:r]
            direct_rgb_out = direct_rgb_out[:, :, t:b, l:r]
            gt_img = gt_img[:, :, t:b, l:r]
        else:
            skip = True

        res_dict = {
            'src_img1': self.src_img1,
            'src_img2': self.src_img2,
            'pred_img': pred_img,
            'gt_img': gt_img,
            'mask': mask,
            'direct_rgb_out': direct_rgb_out,
            'skip': skip
        }
        return res_dict

    def compute_scene_flow_one_side(self, coord, pose,
                                    rgb1, rgb2,
                                    rgba_layers1, rgba_layers2,
                                    featmaps1, featmaps2,
                                    pts1, pts2,
                                    depth_layers1, depth_layers2,
                                    mask_layers1, mask_layers2,
                                    flow_f, flow_b, kernel,
                                    with_inpainted=False):

        num_layers1 = len(rgba_layers1)
        pts2 = transform_pts_in_3D(pts2, pose).T.reshape(1, 3, self.h, self.w)

        mask_mutual_flow = self.scene_flow_estimator.get_mutual_matches(flow_f, flow_b, th=5, return_mask=True).float()
        mask_mutual_flow = mask_mutual_flow.unsqueeze(1)

        coord1_corsp = coord + flow_f
        coord1_corsp_normed = normalize_for_grid_sample(coord1_corsp, self.h, self.w)
        pts2_sampled = F.grid_sample(pts2, coord1_corsp_normed, align_corners=True,
                                     mode='nearest', padding_mode="border")
        depth2_sampled = pts2_sampled[:, -1:]

        rgb2_sampled = F.grid_sample(rgb2, coord1_corsp_normed, align_corners=True, padding_mode="border")
        mask_layers2_ds = F.interpolate(mask_layers2.squeeze(1), size=featmaps2.shape[-2:], mode='area')
        featmap2 = torch.sum(featmaps2 * mask_layers2_ds, dim=0, keepdim=True)
        context2 = torch.sum(mask_layers2_ds, dim=0, keepdim=True)
        featmap2_sampled = F.grid_sample(featmap2, coord1_corsp_normed, align_corners=True, padding_mode="border")
        context2_sampled = F.grid_sample(context2, coord1_corsp_normed, align_corners=True, padding_mode="border")
        mask2_sampled = F.grid_sample(self.src_mask2, coord1_corsp_normed, align_corners=True, padding_mode="border")

        featmap2_sampled = featmap2_sampled / torch.clamp(context2_sampled, min=1e-6)
        context2_sampled = (context2_sampled > 0.5).float()
        last_pts2_i = torch.zeros_like(pts2.permute(0, 2, 3, 1))
        last_alpha_i = torch.zeros_like(rgba_layers1[0][:, -1], dtype=torch.bool)

        all_pts = []
        all_rgbas = []
        all_feats = []
        all_rgbas_end = []
        all_feats_end = []
        all_masks = []
        all_pts_end = []
        all_optical_flows = []
        for i in range(num_layers1):
            alpha_i = (rgba_layers1[i][:, -1] * self.src_mask1.squeeze(1) * mask2_sampled.squeeze(1)) > 0.5
            rgba_i = rgba_layers1[i]
            mask_i = mask_layers1[i]
            mask_no_mutual_flow = mask_i * context2_sampled
            mask_gau_i = mask_no_mutual_flow * mask_mutual_flow
            mask_no_mutual_flow = erosion(mask_no_mutual_flow, kernel)
            mask_gau_i = erosion(mask_gau_i, kernel)

            featmap1 = featmaps1[i][None]
            featmap1 = F.interpolate(featmap1, size=(self.h, self.w), mode='bilinear', align_corners=True)
            pts1_i = unproject_pts_pt(self.intrinsic1, coord.reshape(-1, 2), depth_layers1[i].flatten())
            pts1_i = pts1_i.reshape(1, self.h, self.w, 3)

            flow_inpainted, mask_no_mutual_flow_ = self.masked_diffuse(flow_f.permute(0, 3, 1, 2),
                                                                       mask_no_mutual_flow,
                                                                       kernel_size=15, iter=7)

            coord_inpainted = coord.clone()
            coord_inpainted_ = coord + flow_inpainted.permute(0, 2, 3, 1)
            mask_no_mutual_flow_bool = (mask_no_mutual_flow_ > 1e-6).squeeze(1)
            coord_inpainted[mask_no_mutual_flow_bool] = coord_inpainted_[mask_no_mutual_flow_bool]

            depth_inpainted = depth_layers1[i].clone()
            depth_inpainted_, mask_gau_i_ = self.masked_diffuse(depth2_sampled, mask_gau_i,
                                                                kernel_size=15, iter=7)
            mask_gau_i_bool = (mask_gau_i_ > 1e-6).squeeze(1)
            depth_inpainted.squeeze(1)[mask_gau_i_bool] = depth_inpainted_.squeeze(1)[mask_gau_i_bool]
            pts2_i = unproject_pts_pt(self.intrinsic2, coord_inpainted.contiguous().reshape(-1, 2),
                                      depth_inpainted.flatten()).reshape(1, self.h, self.w, 3)

            if i > 0:
                mask_wrong_ordering = (pts2_i[..., -1] <= last_pts2_i[..., -1]) * last_alpha_i
                pts2_i[mask_wrong_ordering] = last_pts2_i[mask_wrong_ordering] * 1.01

            rgba_end = mask_gau_i * torch.cat([rgb2_sampled, mask_gau_i], dim=1) + (1 - mask_gau_i) * rgba_i
            feat_end = mask_gau_i * featmap2_sampled + (1 - mask_gau_i) * featmap1
            last_alpha_i[alpha_i] = True
            last_pts2_i[alpha_i] = pts2_i[alpha_i]

            if with_inpainted:
                mask_keep = alpha_i
            else:
                mask_keep = mask_i.squeeze(1).bool()

            all_pts.append(pts1_i[mask_keep])
            all_rgbas.append(rgba_i.permute(0, 2, 3, 1)[mask_keep])
            all_feats.append(featmap1.permute(0, 2, 3, 1)[mask_keep])
            all_masks.append(mask_i.permute(0, 2, 3, 1)[mask_keep])
            all_pts_end.append(pts2_i[mask_keep])
            all_rgbas_end.append(rgba_end.permute(0, 2, 3, 1)[mask_keep])
            all_feats_end.append(feat_end.permute(0, 2, 3, 1)[mask_keep])
            all_optical_flows.append(flow_inpainted.permute(0, 2, 3, 1)[mask_keep])

        return all_pts, all_pts_end, all_rgbas, all_rgbas_end, all_feats, all_feats_end, all_masks, all_optical_flows

    def compute_scene_flow_for_motion(self, coord, pose,
                                      rgb,
                                      rgba_layers,
                                      featmaps, pts, depth_layers,
                                      mask_layers, flow, kernel,
                                      with_inpainted=False):

        num_layers = len(rgba_layers)

        pts = transform_pts_in_3D(pts, pose).T.reshape(1, 3, self.h, self.w)

        coord_corsp = coord
        coord_corsp_normed = normalize_for_grid_sample(coord_corsp, self.h, self.w)

        rgb_sampled = F.grid_sample(rgb, coord_corsp_normed, align_corners=True, padding_mode="border")
        mask_layers_ds = F.interpolate(mask_layers.squeeze(1), size=featmaps.shape[-2:], mode='area')
        featmap = torch.sum(featmaps * mask_layers_ds, dim=0, keepdim=True)
        context = torch.sum(mask_layers_ds, dim=0, keepdim=True)

        featmap_sampled = F.grid_sample(featmap, coord_corsp_normed, align_corners=True, padding_mode="border")
        context_sampled = F.grid_sample(context, coord_corsp_normed, align_corners=True, padding_mode="border")

        featmap_sampled = featmap_sampled / torch.clamp(context_sampled, min=1e-6)
        last_pts_end_i = torch.zeros_like(pts.permute(0, 2, 3, 1))
        last_alpha_end_i = torch.zeros_like(rgba_layers[0][:, -1], dtype=torch.bool)

        all_pts_src = []
        all_rgbas_src = []
        all_feats_src = []
        all_rgbas_end = []
        all_feats_end = []
        all_masks = []
        all_pts_end = []
        all_optical_flows = []
        for i in range(num_layers):
            alpha_i = (rgba_layers[i][:, -1] * self.src_mask.squeeze(1)) > 0.5
            rgba_i = rgba_layers[i]
            mask_i = mask_layers[i]
            mask_no_mutual_flow = erosion(mask_i, kernel)
            mask_gau_i = mask_no_mutual_flow

            featmap = featmaps[i][None]
            featmap = F.interpolate(featmap, size=(self.h, self.w), mode='bilinear', align_corners=True)
            pts_src_i = unproject_pts_pt(self.intrinsic, coord.reshape(-1, 2), depth_layers[i].flatten())
            pts_src_i = pts_src_i.reshape(1, self.h, self.w, 3)

            flow_inpainted, mask_no_mutual_flow_ = self.masked_diffuse(flow.permute(0, 3, 1, 2),
                                                                       mask_no_mutual_flow,
                                                                       kernel_size=15, iter=7)

            coord_inpainted = coord.clone()
            coord_inpainted_ = coord + flow_inpainted.permute(0, 2, 3, 1)
            mask_no_mutual_flow_bool = (mask_no_mutual_flow_ > 1e-6).squeeze(1)
            coord_inpainted[mask_no_mutual_flow_bool] = coord_inpainted_[mask_no_mutual_flow_bool]

            depth_inpainted = depth_layers[i].clone()
            depth_inpainted_, mask_gau_i_ = self.masked_diffuse(depth_inpainted, mask_gau_i,
                                                                kernel_size=15, iter=7)
            mask_gau_i_bool = (mask_gau_i_ > 1e-6).squeeze(1)
            depth_inpainted.squeeze(1)[mask_gau_i_bool] = depth_inpainted_.squeeze(1)[mask_gau_i_bool]

            if self.args.correct_inpaint_depth:
                if i > 0:
                    inpainting_area = alpha_i ^ mask_gau_i_bool
                    depth_inpainted.squeeze(1)[inpainting_area] = torch.where(
                        depth_inpainted.squeeze(1)[inpainting_area] <= last_pts_end_i[..., -1].max(),
                        last_pts_end_i[..., -1].max() * 1.1, depth_inpainted.squeeze(1)[inpainting_area])

            coord_flowed = coord_inpainted.clone().permute(0, 3, 1, 2)[0]

            invalid_mask = torch.zeros(1, self.h, self.w, device='cuda').bool()
            out_of_bounds_x = torch.logical_or(coord_flowed[0] > (self.w - 1), coord_flowed[0] < 0)
            out_of_bounds_y = torch.logical_or(coord_flowed[1] > (self.h - 1), coord_flowed[1] < 0)
            invalid_mask = torch.logical_or(out_of_bounds_x.unsqueeze(0), invalid_mask)
            invalid_mask = torch.logical_or(out_of_bounds_y.unsqueeze(0), invalid_mask)

            pts_end_i = unproject_pts_pt(self.intrinsic, coord_inpainted.contiguous().reshape(-1, 2),
                                         depth_inpainted.flatten()).reshape(1, self.h, self.w, 3)

            if i > 0:
                mask_wrong_ordering = (pts_end_i[..., -1] <= last_pts_end_i[..., -1]) * last_alpha_end_i
                pts_end_i[mask_wrong_ordering] = last_pts_end_i[mask_wrong_ordering] * 1.01

            rgba_end = mask_gau_i * torch.cat([rgba_i[:, :3], mask_gau_i], dim=1) + (1 - mask_gau_i) * rgba_i
            feat_end = mask_gau_i * featmap + (1 - mask_gau_i) * featmap_sampled
            last_alpha_end_i[alpha_i] = True
            last_pts_end_i[alpha_i] = pts_end_i[alpha_i]

            if with_inpainted:
                mask_keep = alpha_i
            else:
                mask_keep = mask_i.squeeze(1).bool()

            mask_keep = mask_keep * ~invalid_mask

            all_pts_src.append(pts_src_i[mask_keep])
            all_rgbas_src.append(rgba_i.permute(0, 2, 3, 1)[mask_keep])
            all_feats_src.append(featmap.permute(0, 2, 3, 1)[mask_keep])
            all_masks.append(mask_i.permute(0, 2, 3, 1)[mask_keep])
            all_pts_end.append(pts_end_i[mask_keep])
            all_rgbas_end.append(rgba_end.permute(0, 2, 3, 1)[mask_keep])
            all_feats_end.append(feat_end.permute(0, 2, 3, 1)[mask_keep])
            all_optical_flows.append(flow_inpainted.permute(0, 2, 3, 1)[mask_keep])

        return all_pts_src, all_pts_end, all_rgbas_src, all_rgbas_end, all_feats_src, \
            all_feats_end, all_masks, all_optical_flows

    def prepare_for_novel_view(self, coord, rgba_layers,
                               featmaps, depth_layers,
                               mask_layers, kernel,
                               with_inpainted=False):

        num_layers = len(rgba_layers)

        all_pts = []
        all_rgbas = []
        all_feats = []
        all_masks = []
        for i in range(num_layers):
            alpha_i = (rgba_layers[i][:, -1] * self.src_mask.squeeze(1)) > 0.5
            rgba_i = rgba_layers[i]
            mask_i = mask_layers[i]

            featmap = featmaps[i][None]
            featmap = F.interpolate(featmap, size=(self.h, self.w), mode='bilinear', align_corners=True)
            pts_i = unproject_pts_pt(self.intrinsic, coord.reshape(-1, 2), depth_layers[i].flatten())
            pts_i = pts_i.reshape(1, self.h, self.w, 3)

            if with_inpainted:
                mask_keep = alpha_i
            else:
                mask_keep = mask_i.squeeze(1).bool()

            all_pts.append(pts_i[mask_keep])
            all_rgbas.append(rgba_i.permute(0, 2, 3, 1)[mask_keep])
            all_feats.append(featmap.permute(0, 2, 3, 1)[mask_keep])
            all_masks.append(mask_i.permute(0, 2, 3, 1)[mask_keep])

        return all_pts, all_rgbas, all_feats, all_masks

    def compute_scene_flow_for_motion_without_layering(self, coord, pts, depth, flow):
        pts_src = pts
        coord_flowed = coord + flow
        pts_end = unproject_pts_pt(self.intrinsic, coord_flowed.contiguous().reshape(-1, 2),
                                   depth.flatten())

        return pts_src, pts_end

    def compute_flow_and_inpaint(self):
        with torch.no_grad():
            image = self.motion_rgbs
            mask = self.mask[None]
            hint = self.hints[None]
            outputs_motion = self.scene_flow_estimator.forward_flow(image, mask, hint)
            flow = outputs_motion["PredMotion"] * mask

            for _ in range(7):
                flow_ = box_blur(flow, (15, 15), border_type='constant')
            flow = flow_ * mask

            flow_scale = [self.w / flow.shape[3], self.h / flow.shape[2]]
            flow = flow * torch.FloatTensor(flow_scale).to(flow.device).view(1, 2, 1, 1)
            flow = F.interpolate(flow, (self.h, self.w), mode='bilinear', align_corners=False)

        depth_bins_src = get_depth_bins(depth=self.src_depth)

        rgba_layers_src, depth_layers_src, mask_layers_src = \
            self.inpainter.sequential_inpainting(self.src_img, self.src_depth, depth_bins_src)

        featmaps_src = self.feature_extraction(rgba_layers_src, mask_layers_src, depth_layers_src)

        depth_layers_src = self.apply_scale_shift(depth_layers_src, self.scale_shift[0], self.scale_shift[1])

        processed_depth_src, layer_id_map_src = self.render_depth_from_mdi(depth_layers_src, rgba_layers_src[:, :, -1:])

        h, w = self.src_img.shape[-2:]
        coord = get_coord_grids_pt(h, w, device=self.device).float()[None]
        pts_src = unproject_pts_pt(self.intrinsic, coord.reshape(-1, 2), processed_depth_src.flatten())

        return coord, flow, pts_src, featmaps_src, rgba_layers_src, depth_layers_src, mask_layers_src

    def only_compute_flow(self):
        with torch.no_grad():
            image = self.motion_rgbs
            mask = self.mask[None]
            hint = self.hints[None]
            outputs_motion = self.scene_flow_estimator.forward_flow(image, mask, hint)
            flow = outputs_motion["PredMotion"] * mask

            for _ in range(7):
                flow_ = box_blur(flow, (15, 15), border_type='constant')
            flow = flow_ * mask

            flow_scale = [self.w / flow.shape[3], self.h / flow.shape[2]]
            flow = flow * torch.FloatTensor(flow_scale).to(flow.device).view(1, 2, 1, 1)
            flow = F.interpolate(flow, (self.h, self.w), mode='bilinear', align_corners=False)

        h, w = self.src_img.shape[-2:]
        coord = get_coord_grids_pt(h, w, device=self.device).float()[None]
        pts_src = unproject_pts_pt(self.intrinsic, coord.reshape(-1, 2), self.src_depth.flatten())

        return coord, flow, pts_src

    def only_inpaint(self, index=None):
        flow = None

        if index is not None:
            depth_bins_src = get_depth_bins(depth=self.src_depth[index][None])

            rgba_layers_src, depth_layers_src, mask_layers_src = \
                self.inpainter.sequential_inpainting(self.src_img[index][None], self.src_depth[index][None],
                                                     depth_bins_src)
        else:
            depth_bins_src = get_depth_bins(depth=self.src_depth)

            rgba_layers_src, depth_layers_src, mask_layers_src = \
                self.inpainter.sequential_inpainting(self.src_img, self.src_depth, depth_bins_src)

        featmaps_src = self.feature_extraction(rgba_layers_src, mask_layers_src, depth_layers_src)

        depth_layers_src = self.apply_scale_shift(depth_layers_src, self.scale_shift[0], self.scale_shift[1])

        processed_depth_src, layer_id_map_src = self.render_depth_from_mdi(depth_layers_src, rgba_layers_src[:, :, -1:])

        h, w = self.src_img.shape[-2:]
        coord = get_coord_grids_pt(h, w, device=self.device).float()[None]
        pts_src = unproject_pts_pt(self.intrinsic, coord.reshape(-1, 2), processed_depth_src.flatten())

        return coord, flow, pts_src, featmaps_src, rgba_layers_src, depth_layers_src, mask_layers_src

    def render_rgbda_layers_with_scene_flow(self, return_pts=False):
        kernel = torch.ones(5, 5, device=self.device)
        batch = {
            'images': self.motion_rgbs,
            'motions': self.motions,
            'hints': self.hints
        }
        with torch.no_grad():
            motion_loss, outputs_motion = self.scene_flow_estimator.forward(batch)
            flow = outputs_motion["PredMotion"] * outputs_motion["MovingMask"]

            flow_scale = [self.w / flow.shape[3], self.h / flow.shape[2]]
            flow = flow * torch.FloatTensor(flow_scale).to(flow.device).view(1, 2, 1, 1)
            flow = F.interpolate(flow, (self.h, self.w), mode='bilinear', align_corners=False)

            flow_f = self.euler_integration(flow, self.middle_index.long() - self.start_index.long())
            flow_b = self.euler_integration(-flow, self.end_index.long() + 1 - self.middle_index.long())
            flow_f = flow_f.permute(0, 2, 3, 1)
            flow_b = flow_b.permute(0, 2, 3, 1)

        depth_bins_src = get_depth_bins(depth=self.src_depth)
        depth_bins1 = get_depth_bins(depth=self.start_depth)
        depth_bins2 = get_depth_bins(depth=self.end_depth)

        rgba_layers_src, depth_layers_src, mask_layers_src = \
            self.inpainter.sequential_inpainting(self.src_img, self.src_depth, depth_bins_src)
        rgba_layers1, depth_layers1, mask_layers1 = \
            self.inpainter.sequential_inpainting(self.start_frame, self.start_depth, depth_bins1)
        rgba_layers2, depth_layers2, mask_layers2 = \
            self.inpainter.sequential_inpainting(self.end_frame, self.end_depth, depth_bins2)

        featmaps_src = self.feature_extraction(rgba_layers_src, mask_layers_src, depth_layers_src)
        featmaps1 = self.feature_extraction(rgba_layers1, mask_layers1, depth_layers1)
        featmaps2 = self.feature_extraction(rgba_layers2, mask_layers2, depth_layers2)

        depth_layers_src = self.apply_scale_shift(depth_layers_src, self.scale_shift[0], self.scale_shift[1])
        depth_layers1 = self.apply_scale_shift(depth_layers1, self.scale_shift[0], self.scale_shift[1])
        depth_layers2 = self.apply_scale_shift(depth_layers2, self.scale_shift[0], self.scale_shift[1])

        processed_depth_src, layer_id_map_src = self.render_depth_from_mdi(depth_layers_src, rgba_layers_src[:, :, -1:])
        processed_depth1, layer_id_map1 = self.render_depth_from_mdi(depth_layers1, rgba_layers1[:, :, -1:])
        processed_depth2, layer_id_map2 = self.render_depth_from_mdi(depth_layers2, rgba_layers2[:, :, -1:])

        assert self.start_frame.shape[-2:] == self.start_frame.shape[-2:]
        h, w = self.start_frame.shape[-2:]
        coord = get_coord_grids_pt(h, w, device=self.device).float()[None]
        pts_src = unproject_pts_pt(self.intrinsic, coord.reshape(-1, 2), processed_depth_src.flatten())
        pts1 = unproject_pts_pt(self.intrinsic, coord.reshape(-1, 2), processed_depth1.flatten())
        pts2 = unproject_pts_pt(self.intrinsic, coord.reshape(-1, 2), processed_depth2.flatten())

        all_pts_src, all_rgbas_src, all_feats_src, all_masks_src = \
            self.prepare_for_novel_view(coord, rgba_layers_src, featmaps_src, depth_layers_src,
                                        mask_layers_src, kernel, with_inpainted=True)

        _, all_pts_f, _, all_rgbas_f, _, all_feats_f, \
            all_masks_f, all_optical_flow_f = \
            self.compute_scene_flow_for_motion(coord, torch.inverse(self.pose),
                                               rgba_layers1, featmaps1, pts1, depth_layers1,
                                               mask_layers1, flow_f, kernel, with_inpainted=True)

        _, all_pts_b, _, all_rgbas_b, _, all_feats_b, \
            all_masks_b, all_optical_flow_b = \
            self.compute_scene_flow_for_motion(coord, torch.inverse(self.pose),
                                               rgba_layers2, featmaps2, pts2, depth_layers2,
                                               mask_layers2, flow_b, kernel, with_inpainted=True)

        all_pts_src = torch.cat(all_pts_src)
        all_rgbas_src = torch.cat(all_rgbas_src)
        all_feats_src = torch.cat(all_feats_src)
        all_masks_src = torch.cat(all_masks_src)
        all_pts_flowed = torch.cat(all_pts_f + all_pts_b)
        all_rgbas_flowed = torch.cat(all_rgbas_f + all_rgbas_b)
        all_feats_flowed = torch.cat(all_feats_f + all_feats_b)
        all_masks = torch.cat(all_masks_f + all_masks_b)
        all_side_ids = torch.zeros_like(all_masks.squeeze(), dtype=torch.long)
        num_pts_2 = sum([len(x) for x in all_pts_b])
        all_side_ids[-num_pts_2:] = 1
        all_optical_flow = torch.cat(all_optical_flow_f + all_optical_flow_b)

        if return_pts:
            return all_pts_src, all_pts_flowed, all_rgbas_src, all_rgbas_flowed, \
                all_feats_src, all_feats_flowed, all_masks, all_side_ids, all_optical_flow

        R = self.tgt_pose[0, :3, :3]
        t = self.tgt_pose[0, :3, 3]

        pred_img, direct_rgb_out, meta = self.render_pcd(all_pts_flowed,
                                                         all_rgbas_flowed,
                                                         all_feats_flowed,
                                                         all_masks, all_side_ids,
                                                         time=self.time)

        pred_novel_view, direct_rgb_novel_view, novel_view_meta = \
            self.render_pcd_for_novel_view(all_pts_src,
                                           all_rgbas_src,
                                           all_feats_src,
                                           all_masks_src, R, t)

        novel_view_mask = self.get_reprojection_mask(pts_src, R, t)
        novel_view_gt_img = self.tgt_img
        t, b, l, r, bad = self.get_cropping_ids(novel_view_mask)

        pred_novel_view = pred_novel_view[:, :, t:b, l:r]
        novel_view_mask = novel_view_mask[:, :, t:b, l:r]
        direct_rgb_novel_view = direct_rgb_novel_view[:, :, t:b, l:r]
        novel_view_gt_img = novel_view_gt_img[:, :, t:b, l:r]

        res_dict = {
            'src_img': self.src_img,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,

            'flow': flow,

            'pred_novel_view': pred_novel_view,
            'direct_rgb_novel_view': direct_rgb_novel_view,
            'novel_view_gt_img': novel_view_gt_img,
            'novel_view_mask': novel_view_mask,

            'pred_img': pred_img,
            'direct_rgb_out': direct_rgb_out,
            'gt_img': self.middle_frame,

            'alpha_layers1': rgba_layers1[:, :, [-1]],
            'alpha_layers2': rgba_layers2[:, :, [-1]],
            'mask_layers1': mask_layers1,
            'mask_layers2': mask_layers2,
        }
        return res_dict

    def get_prediction(self, data):
        # process data first
        self.process_data(data)
        return self.render_rgbda_layers_with_scene_flow()
