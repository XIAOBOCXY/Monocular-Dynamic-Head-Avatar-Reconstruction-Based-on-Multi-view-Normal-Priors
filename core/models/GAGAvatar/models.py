#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import math
import torch
import torch.nn as nn
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer, look_at_view_transform
from pytorch3d.renderer.implicit.harmonic_embedding import HarmonicEmbedding
from pytorch3d.structures import Meshes

from core.models.modules import DINOBase, StyleUNet
from core.libs.flame_model import FLAMEModel
from core.libs.utils_renderer import render_gaussian
from core.libs.utils_perceptual import FacePerceptualLoss

class GAGAvatar(nn.Module):
    def __init__(self, model_cfg=None, **kwargs):
        super().__init__()
        self.base_model = DINOBase(output_dim=256)
        for param in self.base_model.dino_model.parameters():
            param.requires_grad = False
        # dir_encoder
        n_harmonic_dir = 4
        self.direnc_dim = n_harmonic_dir * 2 * 3 + 3
        self.harmo_encoder = HarmonicEmbedding(n_harmonic_dir)
        # pre_trained
        self.head_base = nn.Parameter(torch.randn(5023, 256), requires_grad=True)
        self.gs_generator_g = LinearGSGenerator(in_dim=1024, dir_dim=self.direnc_dim)
        self.gs_generator_l0 = ConvGSGenerator(in_dim=256, dir_dim=self.direnc_dim)
        self.gs_generator_l1 = ConvGSGenerator(in_dim=256, dir_dim=self.direnc_dim)
        self.cam_params = {'focal_x': 12.0, 'focal_y': 12.0, 'size': [512, 512]}
        self.upsampler = StyleUNet(in_size=512, in_dim=32, out_dim=3, out_size=512)
        self.percep_loss = FacePerceptualLoss(loss_type='l1', weighted=True)
        normal_loss_cfg = getattr(model_cfg, 'NORMAL_LOSS', None) if model_cfg is not None else None
        self.use_normal_loss = bool(normal_loss_cfg.ENABLED) if normal_loss_cfg is not None else False
        self.normal_loss_mode = str(getattr(normal_loss_cfg, 'MODE', 'point')).lower() if normal_loss_cfg is not None else 'point'
        self.normal_loss_weight = float(normal_loss_cfg.WEIGHT) if normal_loss_cfg is not None else 0.0
        self.normal_render_size = int(getattr(normal_loss_cfg, 'RENDER_SIZE', 128)) if normal_loss_cfg is not None else 128
        soap_guidance_cfg = getattr(model_cfg, 'SOAP_GUIDANCE', None) if model_cfg is not None else None
        self.use_soap_guidance = bool(soap_guidance_cfg.ENABLED) if soap_guidance_cfg is not None else False
        self.soap_rgb_weight = float(getattr(soap_guidance_cfg, 'RGB_WEIGHT', 0.0)) if soap_guidance_cfg is not None else 0.0
        self.soap_normal_weight = float(getattr(soap_guidance_cfg, 'NORMAL_WEIGHT', 0.0)) if soap_guidance_cfg is not None else 0.0
        self.soap_render_size = int(getattr(soap_guidance_cfg, 'RENDER_SIZE', 128)) if soap_guidance_cfg is not None else 128
        self.soap_elevation = float(getattr(soap_guidance_cfg, 'ELEVATION', 0.0)) if soap_guidance_cfg is not None else 0.0
        self.soap_view_angles = tuple(float(angle) for angle in getattr(
            soap_guidance_cfg, 'VIEW_ANGLES', [0, -90, 180, 90, -45, 45]
        )) if soap_guidance_cfg is not None else ()
        self._plane_face_cache = {}
        if self.use_normal_loss:
            if self.normal_loss_mode not in {'point', 'screen'}:
                raise ValueError(f'Unsupported NORMAL_LOSS.MODE: {self.normal_loss_mode}')
            flame_faces = FLAMEModel(n_shape=300, n_exp=100, scale=1.0, no_lmks=True).get_faces()
            self.register_buffer('flame_faces', flame_faces.long(), persistent=False)
            self.normal_raster_settings = RasterizationSettings(
                image_size=self.normal_render_size,
                blur_radius=0.0,
                faces_per_pixel=1,
                perspective_correct=True,
                cull_backfaces=False,
            )
        if self.use_soap_guidance:
            if len(self.soap_view_angles) == 0:
                raise ValueError('SOAP_GUIDANCE.VIEW_ANGLES must not be empty when SOAP guidance is enabled.')
            if self.soap_normal_weight > 0.0:
                self.soap_raster_settings = RasterizationSettings(
                    image_size=self.soap_render_size,
                    blur_radius=0.0,
                    faces_per_pixel=1,
                    perspective_correct=True,
                    cull_backfaces=False,
                )

    def forward(self, batch, train_frac=1.0, rand=True):
        batch_size = batch['f_image'].shape[0]
        t_image, t_bbox = batch['t_image'], batch['t_bbox']
        f_image, f_planes = batch['f_image'], batch['f_planes']
        t_points, t_transform =  batch['t_points'], batch['t_transform']
        # feature encoding
        output_size = int(math.sqrt(f_planes['plane_points'].shape[1]))
        f_feature0, f_feature1 = self.base_model(f_image, output_size=output_size)
        # dir encoding
        plane_direnc = self.harmo_encoder(f_planes['plane_dirs'])
        # global part
        gs_params_g = self.gs_generator_g(
                torch.cat([
                    self.head_base[None].expand(batch_size, -1, -1), f_feature1[:, None].expand(-1, 5023, -1), 
                ], dim=-1
            ), plane_direnc
        )
        gs_params_g['xyz'] = t_points
        # local part
        gs_params_l0 = self.gs_generator_l0(f_feature0, plane_direnc)
        gs_params_l1 = self.gs_generator_l1(f_feature0, plane_direnc)
        gs_params_l0['xyz'] = f_planes['plane_points'] + gs_params_l0['positions'] * f_planes['plane_dirs'][:, None]
        gs_params_l1['xyz'] = f_planes['plane_points'] + -1 * gs_params_l1['positions'] * f_planes['plane_dirs'][:, None]
        gs_params = {
            k:torch.cat([gs_params_g[k], gs_params_l0[k], gs_params_l1[k]], dim=1) for k in gs_params_g.keys()
        }
        gen_images = render_gaussian(
            gs_params=gs_params, cam_matrix=t_transform, cam_params=self.cam_params
        )['images']
        sr_gen_images = self.upsampler(gen_images)
        results = {
            't_image':t_image, 't_bbox':t_bbox, 't_points': t_points, 
            'p_points': torch.cat([gs_params_l0['xyz'], gs_params_l1['xyz']], dim=1),
            'gen_image': gen_images[:, :3], 'sr_gen_image': sr_gen_images
        }
        if self.use_normal_loss or (self.use_soap_guidance and self.soap_normal_weight > 0.0):
            results['p_points_l0'] = gs_params_l0['xyz']
            results['p_points_l1'] = gs_params_l1['xyz']
            results['plane_size'] = output_size
            results['t_transform'] = t_transform
        if self.use_soap_guidance and 'soap_guidance' in batch:
            results['f_transform'] = batch['f_transform']
            results['soap_guidance'] = batch['soap_guidance']
            if self.soap_rgb_weight > 0.0 and 'f_points' in batch:
                source_gs_params = {
                    k: torch.cat([gs_params_g[k], gs_params_l0[k], gs_params_l1[k]], dim=1) for k in gs_params_g.keys()
                }
                source_gs_params['xyz'] = torch.cat([batch['f_points'], gs_params_l0['xyz'], gs_params_l1['xyz']], dim=1)
                results['soap_gs_params'] = source_gs_params
        return results

    @torch.no_grad()
    def forward_expression(self, batch):
        if not hasattr(self, '_gs_params'):
            batch_size = batch['f_image'].shape[0]
            f_image, f_planes = batch['f_image'], batch['f_planes']
            f_feature0, f_feature1 = self.base_model(f_image)
            # dir encoding
            plane_direnc = self.harmo_encoder(f_planes['plane_dirs'])
            # global part
            gs_params_g = self.gs_generator_g(
                torch.cat([
                        self.head_base[None].expand(batch_size, -1, -1), f_feature1[:, None].expand(-1, 5023, -1), 
                    ], dim=-1
                ), plane_direnc
            )
            gs_params_g['xyz'] = batch['f_image'].new_zeros((batch_size, 5023, 3))
            # local part
            gs_params_l0 = self.gs_generator_l0(f_feature0, plane_direnc)
            gs_params_l1 = self.gs_generator_l1(f_feature0, plane_direnc)
            gs_params_l0['xyz'] = f_planes['plane_points'] + gs_params_l0['positions'] * f_planes['plane_dirs'][:, None]
            gs_params_l1['xyz'] = f_planes['plane_points'] + -1 * gs_params_l1['positions'] * f_planes['plane_dirs'][:, None]
            gs_params = {
                k:torch.cat([gs_params_g[k], gs_params_l0[k], gs_params_l1[k]], dim=1) for k in gs_params_g.keys()
            }
            self._gs_params = gs_params
        gs_params = self._gs_params
        t_image, t_points, t_transform = batch['t_image'], batch['t_points'], batch['t_transform']
        gs_params['xyz'][:, :5023] = t_points
        gen_images = render_gaussian(
            gs_params=gs_params, cam_matrix=t_transform, cam_params=self.cam_params
        )['images']
        sr_gen_images = self.upsampler(gen_images)
        results = {
            't_image':t_image, 'gen_image': gen_images[:, :3], 'sr_gen_image': sr_gen_images,
        }
        return results

    def calc_metrics(self, results):
        loss_fn = nn.functional.l1_loss
        t_image, t_bbox = results['t_image'], results['t_bbox']
        t_bbox = expand_bbox(t_bbox, scale=1.1)
        gen_image, sr_gen_image = results['gen_image'], results['sr_gen_image']
        # pec_loss_0 = self.percep_loss(gen_image, t_image)
        # pec_loss_1 = self.percep_loss(sr_gen_image, t_image)
        img_loss_0 = loss_fn(gen_image, t_image)
        img_loss_1 = loss_fn(sr_gen_image, t_image)
        box_loss_0, bpec_loss_0 = self.calc_box_loss(gen_image, t_image, t_bbox, loss_fn)
        box_loss_1, bpec_loss_1 = self.calc_box_loss(sr_gen_image, t_image, t_bbox, loss_fn)
        pec_loss = (bpec_loss_0 + bpec_loss_1)
        img_loss = (img_loss_0 + img_loss_1) * 0.5
        box_loss = (box_loss_0 + box_loss_1) * 0.5
        point_indices = None
        if self.use_normal_loss and self.normal_loss_mode == 'point':
            point_distances, point_indices = square_distance(
                results['t_points'], results['p_points'], return_indices=True
            )
        else:
            point_distances = square_distance(results['t_points'], results['p_points'])
        point_loss = point_distances.mean()
        loss = {'percep_loss': pec_loss, 'img_loss': img_loss, 'box_loss': box_loss, 'point_loss': point_loss}
        show_metric = {}
        if self.use_normal_loss:
            normal_loss, normal_show = self._calc_normal_loss(results, point_indices)
            loss['normal_loss'] = normal_loss
            show_metric.update(normal_show)
        if self.use_soap_guidance and 'soap_guidance' in results:
            soap_losses, soap_show = self._calc_soap_guidance_loss(results)
            loss.update(soap_losses)
            show_metric.update(soap_show)
        psnr = -10.0 * torch.log10(nn.functional.mse_loss(t_image, sr_gen_image).detach())
        show_metric['psnr'] = psnr.item()
        return loss, show_metric

    def _calc_normal_loss(self, results, point_indices=None):
        if self.normal_loss_mode == 'point':
            return self._calc_point_normal_loss(results, point_indices)
        return self._calc_screen_normal_loss(results)

    def _calc_soap_guidance_loss(self, results):
        soap_losses, soap_show = {}, {}
        view_transforms = self._build_soap_view_transforms(results['f_transform'])
        if self.soap_normal_weight > 0.0:
            soap_normal_loss, soap_normal_show = self._calc_soap_normal_guidance(results, view_transforms)
            soap_losses['soap_normal_loss'] = soap_normal_loss
            soap_show.update(soap_normal_show)
        if self.soap_rgb_weight > 0.0 and 'soap_gs_params' in results:
            soap_rgb_loss, soap_rgb_show = self._calc_soap_rgb_guidance(results, view_transforms)
            soap_losses['soap_rgb_loss'] = soap_rgb_loss
            soap_show.update(soap_rgb_show)
        return soap_losses, soap_show

    def _calc_point_normal_loss(self, results, point_indices):
        pred_normals = torch.cat([
            estimate_grid_normals(results['p_points_l0'], results['plane_size']),
            estimate_grid_normals(results['p_points_l1'], results['plane_size'])
        ], dim=1)
        target_normals = compute_vertex_normals(results['t_points'], self.flame_faces)
        matched_pred_normals = torch.gather(
            pred_normals, 1, point_indices.expand(-1, -1, pred_normals.shape[-1])
        )
        normal_cos = nn.functional.cosine_similarity(
            matched_pred_normals, target_normals, dim=-1
        ).abs()
        normal_loss = (1.0 - normal_cos).mean() * self.normal_loss_weight
        return normal_loss, {'normal_cos': normal_cos.mean().item()}

    def _calc_screen_normal_loss(self, results):
        plane_faces = self._get_dual_plane_faces(results['plane_size'], results['p_points_l0'].device)
        pred_vertices = torch.cat([results['p_points_l0'], results['p_points_l1']], dim=1)
        pred_normals, pred_mask = self._render_normal_map(
            pred_vertices, plane_faces, results['t_transform']
        )
        target_normals, target_mask = self._render_normal_map(
            results['t_points'], self.flame_faces, results['t_transform']
        )
        overlap_weight = (pred_mask & target_mask).float()
        cosine = (pred_normals * target_normals).sum(dim=1, keepdim=True).abs().clamp(0.0, 1.0)
        denom = overlap_weight.sum().clamp(min=1.0)
        normal_loss = ((1.0 - cosine) * overlap_weight).sum() / denom
        normal_cos = (cosine * overlap_weight).sum() / denom
        normal_overlap = overlap_weight.mean()
        return normal_loss * self.normal_loss_weight, {
            'normal_cos': float(normal_cos.item()),
            'normal_overlap': float(normal_overlap.item()),
        }

    def _calc_soap_normal_guidance(self, results, view_transforms):
        soap_guidance = results['soap_guidance']
        view_count = len(self.soap_view_angles)
        if soap_guidance['normals'].shape[1] != view_count:
            raise ValueError(
                f'SOAP normal view count mismatch: expected {view_count}, got {soap_guidance["normals"].shape[1]}'
            )
        plane_faces = self._get_dual_plane_faces(results['plane_size'], results['p_points_l0'].device)
        pred_vertices = torch.cat([results['p_points_l0'], results['p_points_l1']], dim=1)
        batch_size = pred_vertices.shape[0]
        flat_vertices = pred_vertices[:, None].expand(-1, view_count, -1, -1).reshape(
            batch_size * view_count, pred_vertices.shape[1], pred_vertices.shape[2]
        )
        flat_transforms = view_transforms.reshape(batch_size * view_count, 3, 4)
        pred_normals, pred_masks = self._render_normal_map(
            flat_vertices, plane_faces, flat_transforms, raster_settings=self.soap_raster_settings
        )
        pred_normals = pred_normals.reshape(batch_size, view_count, 3, self.soap_render_size, self.soap_render_size)
        pred_masks = pred_masks.reshape(batch_size, view_count, 1, self.soap_render_size, self.soap_render_size).float()
        target_normals = self._resize_multiview_tensor(soap_guidance['normals'], self.soap_render_size)
        target_masks = self._resize_multiview_tensor(soap_guidance['masks'], self.soap_render_size, mode='nearest')
        target_normals = self._rotate_soap_normals(target_normals, target_masks)
        overlap = pred_masks * target_masks
        cosine = (pred_normals * target_normals).sum(dim=2, keepdim=True).abs().clamp(0.0, 1.0)
        denom = overlap.sum().clamp(min=1.0)
        soap_normal_loss = ((1.0 - cosine) * overlap).sum() / denom
        soap_normal_cos = (cosine * overlap).sum() / denom
        return soap_normal_loss * self.soap_normal_weight, {
            'soap_normal_cos': float(soap_normal_cos.item()),
            'soap_normal_overlap': float(overlap.mean().item()),
        }

    def _calc_soap_rgb_guidance(self, results, view_transforms):
        soap_guidance = results['soap_guidance']
        view_count = len(self.soap_view_angles)
        if soap_guidance['images'].shape[1] != view_count:
            raise ValueError(
                f'SOAP RGB view count mismatch: expected {view_count}, got {soap_guidance["images"].shape[1]}'
            )
        batch_size = view_transforms.shape[0]
        flat_transforms = view_transforms.reshape(batch_size * view_count, 3, 4)
        flat_gs_params = expand_gaussian_params(results['soap_gs_params'], view_count)
        soap_render = render_gaussian(
            gs_params=flat_gs_params, cam_matrix=flat_transforms, cam_params=self.cam_params,
        )['images']
        pred_rgb = self.upsampler(soap_render)
        target_size = soap_guidance['images'].shape[-2:]
        pred_rgb = nn.functional.interpolate(
            pred_rgb, size=target_size, mode='bilinear', align_corners=False, antialias=True
        )
        pred_rgb = pred_rgb.reshape(batch_size, view_count, 3, target_size[0], target_size[1])
        target_rgb = soap_guidance['images']
        target_masks = soap_guidance['masks']
        denom = target_masks.expand_as(target_rgb).sum().clamp(min=1.0)
        soap_rgb_loss = ((pred_rgb - target_rgb).abs() * target_masks.expand_as(target_rgb)).sum() / denom
        return soap_rgb_loss * self.soap_rgb_weight, {
            'soap_rgb_l1': float(soap_rgb_loss.item()),
        }

    def _get_dual_plane_faces(self, plane_size, device):
        cached_faces = self._plane_face_cache.get(plane_size)
        if cached_faces is None or cached_faces.device != device:
            plane_faces = build_plane_faces(plane_size, device=device)
            point_count = plane_size * plane_size
            cached_faces = torch.cat([plane_faces, plane_faces + point_count], dim=0)
            self._plane_face_cache[plane_size] = cached_faces
        return cached_faces

    def _render_normal_map(self, vertices, faces, transform_matrix, raster_settings=None):
        if faces.dim() == 2:
            faces = faces.unsqueeze(0).expand(vertices.shape[0], -1, -1)
        raster_settings = self.normal_raster_settings if raster_settings is None else raster_settings
        image_size = raster_settings.image_size if isinstance(raster_settings.image_size, int) else raster_settings.image_size[0]
        pixel_normals, masks = [], []
        for bid in range(vertices.shape[0]):
            cameras = build_cameras(
                transform_matrix=transform_matrix[bid:bid+1],
                focal_x=self.cam_params['focal_x'],
                focal_y=self.cam_params['focal_y'],
                image_size=image_size,
                device=vertices.device,
            )
            mesh_faces = faces[bid]
            mesh = Meshes(verts=[vertices[bid]], faces=[mesh_faces])
            fragments = MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings,
            )(mesh)
            vertex_normals = mesh.verts_normals_padded()[0]
            face_vertex_normals = vertex_normals[mesh_faces]
            normal_map = interpolate_face_attributes(
                fragments.pix_to_face,
                fragments.bary_coords,
                face_vertex_normals,
            )[..., 0, :]
            normal_map = nn.functional.normalize(normal_map, dim=-1, eps=1.0e-6)
            mask = (fragments.pix_to_face[..., 0] >= 0).unsqueeze(1)
            pixel_normals.append(normal_map.permute(0, 3, 1, 2) * mask.float())
            masks.append(mask)
        return torch.cat(pixel_normals, dim=0), torch.cat(masks, dim=0)

    def _build_soap_view_transforms(self, base_transform):
        batch_size = base_transform.shape[0]
        device = base_transform.device
        camera_distance = base_transform[:, :, 3].norm(dim=-1)
        if torch.count_nonzero(camera_distance) == 0:
            camera_distance = torch.full_like(camera_distance, 9.3)
        all_transforms = []
        for angle in self.soap_view_angles:
            azimuth = torch.full_like(camera_distance, angle)
            rotation, translation = look_at_view_transform(
                dist=camera_distance,
                elev=torch.full_like(camera_distance, self.soap_elevation),
                azim=azimuth,
                device=device,
            )
            all_transforms.append(torch.cat([rotation, translation[:, :, None]], dim=-1))
        return torch.stack(all_transforms, dim=1)

    def _rotate_soap_normals(self, normals, masks):
        rotation = build_yaw_rotations(self.soap_view_angles, normals.device).to(dtype=normals.dtype)
        normals = normals.permute(0, 1, 3, 4, 2)
        masks = masks.permute(0, 1, 3, 4, 2)
        normals = (normals * 2.0 - 1.0) * masks
        normals = nn.functional.normalize(normals, dim=-1, eps=1.0e-6) * masks
        normals = torch.einsum('bvhwc,vcd->bvhwd', normals, rotation)
        normals = nn.functional.normalize(normals, dim=-1, eps=1.0e-6) * masks
        return normals.permute(0, 1, 4, 2, 3)

    @staticmethod
    def _resize_multiview_tensor(tensor, target_size, mode='bilinear'):
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        if tensor.shape[-2:] == target_size:
            return tensor
        flat_tensor = tensor.reshape(-1, tensor.shape[2], tensor.shape[3], tensor.shape[4])
        if mode == 'nearest':
            flat_tensor = nn.functional.interpolate(flat_tensor, size=target_size, mode=mode)
        else:
            flat_tensor = nn.functional.interpolate(
                flat_tensor, size=target_size, mode=mode, align_corners=False, antialias=True
            )
        return flat_tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2], target_size[0], target_size[1])

    def configure_optimizers(self, config):
        learning_rate = config.LEARNING_RATE
        print('Learning rate: {}'.format(learning_rate))
        # params
        decay_names = []
        normal_params, decay_params0, decay_params1 = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'style_mlp' in name or 'final_linear' in name:
                decay_names.append(".".join(name.split('.')[:-2]) if len(name.split('.'))> 3 else ".".join(name.split('.')[:-1]))
                decay_params0.append(param)
            elif 'gaussian_conv' in name or ('gs_generator_g' in name and 'feature_layers' not in name):
                # decay_names.append(".".join(name.split('.')[:-2]) if len(name.split('.'))> 3 else ".".join(name.split('.')[:-1]))
                decay_params1.append(param)
            else:
                normal_params.append(param)
        print('Decay params: {}'.format(set(decay_names)))
        # optimizer
        optimizer = torch.optim.Adam([
                {'params': normal_params, 'lr': learning_rate},
                {'params': decay_params0, 'lr': learning_rate*0.1},
                {'params': decay_params1, 'lr': learning_rate},
            ], lr=learning_rate, betas=(0.0, 0.99)
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=config.LR_DECAY_RATE, 
            total_iters=config.LR_DECAY_ITER,
            # verbose=True if self.logger is None else False
        )
        return optimizer, scheduler

    def calc_box_loss(self, image, gt_image, bbox, loss_fn, resize_size=512):
        def _resize(frames, tgt_size):
            frames = nn.functional.interpolate(
                frames, size=(tgt_size, tgt_size), mode='bilinear', align_corners=False, antialias=True
            )
            return frames
        bbox = bbox.clamp(min=0, max=1)
        bbox = (bbox * image.shape[-1]).long()
        pred_croped, gt_croped = [], []
        for idx, box in enumerate(bbox):
            gt_croped.append(_resize(gt_image[idx:idx+1, :, box[1]:box[3], box[0]:box[2]], resize_size))
            pred_croped.append(_resize(image[idx:idx+1, :, box[1]:box[3], box[0]:box[2]], resize_size))
        gt_croped = torch.cat(gt_croped, dim=0)
        pred_croped = torch.cat(pred_croped, dim=0)
        box_fn_loss = loss_fn(pred_croped, gt_croped)
        box_perc_loss = self.percep_loss(pred_croped, gt_croped) * 1e-2
        # box_loss = (box_1_loss + box_2_loss) / 2
        return box_fn_loss, box_perc_loss


class LinearGSGenerator(nn.Module):
    def __init__(self, in_dim=1024, dir_dim=27, **kwargs):
        super().__init__()
        # params
        self.feature_layers = nn.Sequential(
            nn.Linear(in_dim, in_dim//4, bias=True),
            nn.ReLU(),
            nn.Linear(in_dim//4, in_dim//4, bias=True),
            nn.ReLU(),
            nn.Linear(in_dim//4, in_dim//4, bias=True),
            nn.ReLU(),
            nn.Linear(in_dim//4, in_dim//4, bias=True),
        )
        layer_in_dim = in_dim//4 + dir_dim
        self.color_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 32, bias=True),
        )
        self.opacity_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 1, bias=True),
        )
        self.scale_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 3, bias=True)
        )
        self.rotation_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 4, bias=True),
        )

    def forward(self, input_features, plane_direnc):
        input_features = self.feature_layers(input_features)
        plane_direnc = plane_direnc[:, None].expand(-1, input_features.shape[1], -1)
        input_features = torch.cat([input_features, plane_direnc], dim=-1)
        # color
        colors = self.color_layers(input_features)
        colors[..., :3] = torch.sigmoid(colors[..., :3])
        # opacity
        opacities = self.opacity_layers(input_features)
        opacities = torch.sigmoid(opacities)
        # scale
        scales = self.scale_layers(input_features)
        # scales = torch.exp(scales) * 0.01
        scales = torch.sigmoid(scales) * 0.05
        # rotation
        rotations = self.rotation_layers(input_features)
        rotations = nn.functional.normalize(rotations)
        return {'colors':colors, 'opacities':opacities, 'scales':scales, 'rotations':rotations}


class ConvGSGenerator(nn.Module):
    def __init__(self, in_dim=256, dir_dim=27, **kwargs):
        super().__init__()
        out_dim = 32 + 1 + 3 + 4 + 1 # color + opacity + scale + rotation + position
        self.gaussian_conv = nn.Sequential(
            nn.Conv2d(in_dim+dir_dim, in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_dim//2, out_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, input_features, plane_direnc):
        plane_direnc = plane_direnc[:, :, None, None].expand(-1, -1, input_features.shape[2], input_features.shape[3])
        input_features = torch.cat([input_features, plane_direnc], dim=1)
        gaussian_params = self.gaussian_conv(input_features)
        # color
        colors = gaussian_params[:, :32]
        colors[..., :3] = torch.sigmoid(colors[..., :3])
        # opacity
        opacities = gaussian_params[:, 32:33]
        opacities = torch.sigmoid(opacities)
        # scale
        scales = gaussian_params[:, 33:36]
        # scales = torch.exp(scales) * 0.01
        scales = torch.sigmoid(scales) * 0.05
        # rotation
        rotations = gaussian_params[:, 36:40]
        rotations = nn.functional.normalize(rotations)
        # position
        positions = gaussian_params[:, 40:41]
        positions = torch.sigmoid(positions)
        results = {'colors':colors, 'opacities':opacities, 'scales':scales, 'rotations':rotations, 'positions':positions}
        for key in results.keys():
            results[key] = results[key].permute(0, 2, 3, 1).reshape(results[key].shape[0], -1, results[key].shape[1])
        return results


def square_distance(src, dst, return_indices=False):
    import faiss
    assert src.dim() == 3 and dst.dim() == 3, 'Input tensors must be 3-dim.'
    all_indices = []
    for bid in range(src.shape[0]):
        src_np = src[bid].detach().cpu().numpy()
        dst_np = dst[bid].detach().cpu().numpy()
        index = faiss.IndexFlatL2(3)
        index.add(dst_np)
        _, indices = index.search(src_np, 1)
        all_indices.append(torch.tensor(indices))
    indices = torch.stack(all_indices).to(src.device, dtype=torch.long)
    dst_selected = torch.gather(dst, 1, indices.to(src.device).expand(-1, -1, dst.shape[-1]))
    distances = torch.sum((src - dst_selected) ** 2, dim=-1) * 10
    if return_indices:
        return distances, indices
    return distances


def estimate_grid_normals(points, plane_size):
    assert points.dim() == 3, 'Input points must be 3-dim.'
    assert points.shape[1] == plane_size * plane_size, 'Point count does not match plane size.'
    point_grid = points.reshape(points.shape[0], plane_size, plane_size, 3).permute(0, 3, 1, 2)
    dx = point_grid[:, :, :, 2:] - point_grid[:, :, :, :-2]
    dy = point_grid[:, :, 2:, :] - point_grid[:, :, :-2, :]
    dx = nn.functional.pad(dx, (1, 1, 0, 0), mode='replicate') * 0.5
    dy = nn.functional.pad(dy, (0, 0, 1, 1), mode='replicate') * 0.5
    normals = torch.cross(dx.permute(0, 2, 3, 1), dy.permute(0, 2, 3, 1), dim=-1)
    normals = nn.functional.normalize(normals, dim=-1, eps=1.0e-6)
    return normals.reshape(points.shape[0], -1, 3)


def compute_vertex_normals(vertices, faces):
    assert vertices.dim() == 3 and faces.dim() == 2, 'Input tensors must be batched vertices and mesh faces.'
    face_vertices0 = vertices[:, faces[:, 0], :]
    face_vertices1 = vertices[:, faces[:, 1], :]
    face_vertices2 = vertices[:, faces[:, 2], :]
    face_normals = torch.cross(face_vertices1 - face_vertices0, face_vertices2 - face_vertices0, dim=-1)
    vertex_normals = torch.zeros_like(vertices)
    for bid in range(vertices.shape[0]):
        vertex_normals[bid].index_add_(0, faces[:, 0], face_normals[bid])
        vertex_normals[bid].index_add_(0, faces[:, 1], face_normals[bid])
        vertex_normals[bid].index_add_(0, faces[:, 2], face_normals[bid])
    return nn.functional.normalize(vertex_normals, dim=-1, eps=1.0e-6)


def build_plane_faces(plane_size, device=None):
    assert plane_size >= 2, 'plane_size must be at least 2.'
    grid_y, grid_x = torch.meshgrid(
        torch.arange(plane_size - 1, device=device),
        torch.arange(plane_size - 1, device=device),
        indexing='ij',
    )
    top_left = grid_y * plane_size + grid_x
    top_right = top_left + 1
    bottom_left = top_left + plane_size
    bottom_right = bottom_left + 1
    tri0 = torch.stack([top_left, bottom_left, top_right], dim=-1)
    tri1 = torch.stack([top_right, bottom_left, bottom_right], dim=-1)
    return torch.cat([tri0.reshape(-1, 3), tri1.reshape(-1, 3)], dim=0).long()


def build_yaw_rotations(view_angles, device):
    angles = torch.tensor(view_angles, device=device, dtype=torch.float32) * math.pi / 180.0
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    rotation = torch.zeros(len(view_angles), 3, 3, device=device, dtype=torch.float32)
    rotation[:, 0, 0] = cosine
    rotation[:, 0, 2] = sine
    rotation[:, 1, 1] = 1.0
    rotation[:, 2, 0] = -sine
    rotation[:, 2, 2] = cosine
    return rotation


def expand_gaussian_params(gs_params, repeat_count):
    expanded_params = {}
    for key, value in gs_params.items():
        expand_shape = (-1, repeat_count) + tuple(-1 for _ in value.shape[1:])
        expanded_value = value[:, None].expand(*expand_shape)
        expanded_params[key] = expanded_value.reshape(value.shape[0] * repeat_count, *value.shape[1:])
    return expanded_params


def build_cameras(transform_matrix, focal_x, focal_y, image_size, device):
    batch_size = transform_matrix.shape[0]
    screen_size = torch.tensor([image_size, image_size], device=device).float()[None].repeat(batch_size, 1)
    focal_length = torch.tensor([focal_x, focal_y], device=device).float()[None].repeat(batch_size, 1)
    principal_point = torch.zeros(batch_size, 2, device=device).float()
    return PerspectiveCameras(
        principal_point=principal_point,
        focal_length=focal_length,
        image_size=screen_size,
        device=device,
        R=transform_matrix[:, :3, :3],
        T=transform_matrix[:, :3, 3],
    )


def expand_bbox(bbox, scale=1.1):
    xmin, ymin, xmax, ymax = bbox.unbind(dim=-1)
    cenx, ceny = (xmin + xmax) / 2, (ymin + ymax) / 2
    extend_size = torch.sqrt((ymax - ymin) * (xmax - xmin)) * scale
    extend_size = torch.min(extend_size, cenx*2)
    extend_size = torch.min(extend_size, ceny*2)
    extend_size = torch.min(extend_size, (1-cenx)*2)
    extend_size = torch.min(extend_size, (1-ceny)*2)
    xmine, xmaxe = cenx - extend_size / 2, cenx + extend_size / 2
    ymine, ymaxe = ceny - extend_size / 2, ceny + extend_size / 2
    expanded_bbox = torch.stack([xmine, ymine, xmaxe, ymaxe], dim=-1)
    return torch.stack([xmine, ymine, xmaxe, ymaxe], dim=-1)
