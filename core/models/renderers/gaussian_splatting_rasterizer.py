from dataclasses import dataclass
import math
import numpy as np

import torch

import threestudio
from threestudio.models.renderers.base import Rasterizer
from threestudio.utils.typing import *
from threestudio.utils.ops import get_cam_info_gaussian

import gsplat
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from core.utils.helper import fov2focal, focal2fov, get_cam_info_gaussian_batch
from core.utils.point_utils import depth_to_normal_batch


@threestudio.register("gaussian-splatting-rasterizer")
class GaussianSplattingRasterizer(Rasterizer):
    @dataclass
    class Config(Rasterizer.Config):
        backend: str                = 'gsplat'

        depth_type: str             = 'inverse'         # options: 'raw', 'inverse', etc
        depth_norm_radius: float    = 1.0
        depth_min_value: float      = 0.0
        depth_max_value: float      = 2.0
        depth_padding_value: float  = 10. / 255.
        depth_target_min: float     = 50. / 255.
        depth_target_max: float     = 255./ 255.
        depth_blending: bool        = True
        
        depth_type_local: str       = 'norm_radius'
        depth_blending_local: bool  = False

        normal_space: str           = 'camera'          # options: 'camera', 'world'

        znear: float                = 0.01
        zfar: float                 = 100.0

        # unused
        # - radius

    cfg: Config

    def configure(
        self,
        geometry, material, background
    ) -> None:
        super().configure(geometry, material, background)

    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        fovy: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        height: int,
        width: int,
        local: bool = False,
        local_index: Optional[List] = None,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:

        if self.cfg.backend == 'vanilla':
            fg_rgb, opacity, fg_semantic, fg_depth, fg_normal, viewspace_points, radii, local_masks = \
                self.__splatting_batch_vanilla(c2w, fovy, height, width, local, local_index)
        elif self.cfg.backend == 'gsplat':
            fg_rgb, opacity, fg_semantic, fg_depth, fg_normal, viewspace_points, radii, local_masks = \
                self.__splatting_batch_gsplat(c2w, fovy, height, width, local, local_index)
        else:
            raise NotImplementedError

        bg_out      = self.background(local=local, mvp_mtx=mvp_mtx, c2w=c2w, height=height, width=width)
        bg_rgb      = bg_out['rgb']
        bg_depth    = bg_out['depth']
        bg_normal   = bg_out['normal']
        bg_semantic = bg_out['semantic']

        # fusion (foregraound + background)
        depth_blending  = self.cfg.depth_blending_local if local else self.cfg.depth_blending
        comp_rgb        = opacity * fg_rgb + (1. - opacity) * bg_rgb
        comp_depth      = opacity * fg_depth + (1. - opacity) * bg_depth if depth_blending else fg_depth
        comp_normal     = opacity * fg_normal + (1. - opacity) * bg_normal # TODO: Quaternion slerp
        comp_semantic   = opacity * fg_semantic + (1. - opacity) * bg_semantic

        # correctness
        mask                = (fg_depth[...,0] > bg_depth[...,0]) & (bg_depth[...,0] > 0.0)
        comp_rgb[mask]      = bg_rgb[mask]
        comp_depth[mask]    = bg_depth[mask]
        comp_normal[mask]   = bg_normal[mask]
        comp_semantic[mask] = bg_semantic[mask]

        # [0,1]
        depth_type = self.cfg.depth_type_local if local else self.cfg.depth_type

        if depth_type == 'raw':
            pass

        elif depth_type == 'inverse':
            invalid_mask                = comp_depth <= 0.0
            depth_valid                 = comp_depth[~invalid_mask]
            depth_valid                 = 1. / depth_valid  # disparity
            comp_depth[~invalid_mask]   = depth_valid
            comp_depth[invalid_mask]    = 0.0

        elif depth_type == 'inverse_adaptive':
            invalid_mask                = comp_depth <= 0.0
            depth_valid                 = comp_depth[~invalid_mask]
            depth_valid                 = 1. / depth_valid  # disparity
            depth_min                   = depth_valid.min()
            depth_max                   = depth_valid.max()
            depth_valid                 = (depth_valid - depth_min) / (depth_max - depth_min)
            depth_valid                 = depth_valid * \
                                            (self.cfg.depth_target_max - self.cfg.depth_target_min) + \
                                                self.cfg.depth_target_min
            comp_depth[~invalid_mask]   = depth_valid
            comp_depth[invalid_mask]    = self.cfg.depth_padding_value     # not completely black

        elif depth_type == 'reverse_minmax':
            min_val                     = self.cfg.depth_min_value
            max_val                     = self.cfg.depth_max_value

            invalid_mask                = comp_depth <= 0.0
            depth_valid                 = comp_depth[~invalid_mask]

            # reverse
            depth_valid                 = (max_val - depth_valid) / (max_val - min_val)
            depth_valid                 = torch.clamp(depth_valid, 0, 1)

            comp_depth[~invalid_mask]   = depth_valid
            comp_depth[invalid_mask]    = 0.0

        elif depth_type == 'reverse_adaptive':
            # from SceneTex and RichDreamer
            invalid_mask                = comp_depth <= 0.0

            depth_valid                 = comp_depth[~invalid_mask]
            depth_min                   = depth_valid.min()
            depth_max                   = depth_valid.max()

            # reverse
            depth_valid                 = (depth_max - depth_valid) / (depth_max - depth_min)
            depth_valid                 = depth_valid * \
                                            (self.cfg.depth_target_max - self.cfg.depth_target_min) + \
                                                self.cfg.depth_target_min

            comp_depth[~invalid_mask]   = depth_valid
            comp_depth[invalid_mask]    = self.cfg.depth_padding_value     # not completely black

        elif depth_type == 'norm_radius':
            # from RichDreamer
            cam_dist                    = camera_distances.reshape(-1,1,1,1).expand_as(comp_depth)
            min_val                     = cam_dist - self.cfg.depth_norm_radius
            max_val                     = cam_dist + self.cfg.depth_norm_radius

            invalid_mask                = comp_depth <= 0.0
            depth_valid                 = comp_depth[~invalid_mask]
            depth_valid                 = (max_val[~invalid_mask] - depth_valid) / (2 * self.cfg.depth_norm_radius)
            depth_valid                 = torch.clamp(depth_valid, 0, 1)

            comp_depth[~invalid_mask]   = depth_valid
            comp_depth[invalid_mask]    = 0.0

        else:
            raise ValueError(depth_type)

        # [-1,1] -> [0,1]
        comp_normal = (comp_normal + 1.) * 0.5
        
        out = {
            "comp_rgb"          : comp_rgb,
            "opacity"           : opacity,
            "comp_depth"        : comp_depth,
            "comp_normal"       : comp_normal,
            "comp_semantic"     : comp_semantic,
            "viewspace_points"  : viewspace_points,
            "radii"             : radii,
            "local_masks"       : local_masks,
        }

        return out
         
    def __splatting_one(
        self, means3D, opacity, scales, rotations, features, semantic, means2D, c2w, fovx, fovy, height, width,
    ) -> tuple[Float[Tensor, "..."]]:
        world_view_transform, full_proj_transform, camera_center = \
            get_cam_info_gaussian(c2w, fovx, fovy, self.cfg.znear, self.cfg.zfar)

        # Set up rasterization configuration
        rasterizer = GaussianRasterizer(raster_settings=GaussianRasterizationSettings(
            image_height    = height,
            image_width     = width,
            tanfovx         = math.tan(fovx*0.5),
            tanfovy         = math.tan(fovy*0.5),
            bg              = torch.tensor([0.,0.,0.], device=means3D.device),
            scale_modifier  = 1.0,
            viewmatrix      = world_view_transform,
            projmatrix      = full_proj_transform,
            sh_degree       = self.geometry.active_sh_degree if self.geometry.use_sh else 0,
            campos          = camera_center,
            prefiltered     = False,
            debug           = False,
        ))

        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D         = means3D,
            means2D         = means2D,
            shs             = features if self.geometry.use_sh else None,
            colors_precomp  = None if self.geometry.use_sh else features,
            opacities       = opacity,
            scales          = scales,
            rotations       = rotations,
            cov3D_precomp   = None
        )

        with torch.no_grad():
            rendered_semantic, _, _, _ = rasterizer(
                means3D         = means3D,
                means2D         = means2D.detach().clone(),
                shs             = None,
                colors_precomp  = semantic,
                opacities       = opacity,
                scales          = scales,
                rotations       = rotations,
                cov3D_precomp   = None
            )

        # (X,H,W) -> (1,H,W,X)
        rendered_image          = rendered_image.unsqueeze(0).permute(0,2,3,1)
        rendered_alpha          = rendered_alpha.unsqueeze(0).permute(0,2,3,1)
        rendered_depth          = rendered_depth.unsqueeze(0).permute(0,2,3,1)
        rendered_semantic       = rendered_semantic.unsqueeze(0).permute(0,2,3,1)

        mask                    = rendered_alpha[...,0] != 0
        rendered_image[mask]    = rendered_image[mask] / rendered_alpha[mask]
        rendered_depth[mask]    = rendered_depth[mask] / rendered_alpha[mask]
        rendered_semantic[mask] = rendered_semantic[mask] / rendered_alpha[mask]

        # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
        surf_normal = depth_to_normal_batch(
            world_view_transform[None], full_proj_transform[None], rendered_depth, self.cfg.normal_space)
        # remember to multiply with accum_alpha since render_normal is unnormalized.
        # surf_normal = surf_normal * rendered_alpha

        comp_rgb        = rendered_image
        opacity         = rendered_alpha
        comp_semantic   = rendered_semantic
        comp_depth      = rendered_depth
        comp_normal     = surf_normal

        return comp_rgb, opacity, comp_semantic, comp_depth, comp_normal, radii
    
    def __splatting_batch_vanilla(
        self, 
        c2w: Float[Tensor, "B 4 4"], 
        fovy: Float[Tensor, "B"],
        height: int,
        width: int,
        local: bool = False,
        local_index: int = -1,
    ):
        # camera intrinsic
        focal   = fov2focal(fovy, height)
        fovx    = focal2fov(focal, width)

        # Gaussians
        means3D, opacity, scales, rotations, features, semantic, instance, local_mask = \
            self.geometry.get_global_params() if not local else self.geometry.get_local_params(local_index)

        # rendering
        B, N    = c2w.shape[0], means3D.shape[0]
        means2D = torch.zeros((B,N,3), requires_grad=True).to(means3D)
        if self.training: means2D.retain_grad()

        rgb_list, opacity_list, semantic_list, depth_list, normal_list, radii_list = [], [], [], [], [], []
        for i, (_c2w, _fovx, _fovy, _means2D) in enumerate(zip(c2w, fovx, fovy, means2D)):
            _rgb, _opacity, _semantic, _depth, _normal, _radii = self.__splatting_one(
                means3D, opacity, scales, rotations, features, semantic, _means2D,
                _c2w, _fovx, _fovy, height, width)
            
            rgb_list.append(_rgb)
            opacity_list.append(_opacity)
            semantic_list.append(_semantic)
            depth_list.append(_depth)
            normal_list.append(_normal)
            radii_list.append(_radii)

        radii = torch.stack(radii_list).to(means3D) / float(max(width, height))

        return \
            torch.cat(rgb_list), \
            torch.cat(opacity_list), \
            torch.cat(semantic_list), \
            torch.cat(depth_list), \
            torch.cat(normal_list), \
            means2D, \
            radii, \
            local_mask
    
    def __splatting_batch_gsplat(
        self,
        c2w: Float[Tensor, "B 4 4"], 
        fovy: Float[Tensor, "B"], 
        height: int, 
        width: int, 
        local: bool = False, 
        local_index: int = -1
    ):
        # camera intrinsic
        focal       = fov2focal(fovy, height)
        fovx        = focal2fov(focal, width)
        Ks          = torch.zeros((focal.shape[0], 3, 3)).to(focal)
        Ks[:, 0, 0] = focal
        Ks[:, 1, 1] = focal
        Ks[:, 0, 2] = 0.5 * width
        Ks[:, 1, 2] = 0.5 * height
        Ks[:, 2, 2] = 1.0

        # camera pose
        world_view_transform, full_proj_transform, _ = \
            get_cam_info_gaussian_batch(c2w, fovx, fovy, self.cfg.znear, self.cfg.zfar)
        viewmats = world_view_transform.transpose(1, 2)

        # Gaussians
        means3D, opacity, scales, rotations, features, semantic, instance, local_mask = \
            self.geometry.get_global_params() if not local else self.geometry.get_local_params(local_index)

        # rendering
        rendered_results, rendered_alpha, meta = gsplat.rasterization(
            means=means3D, quats=rotations, scales=scales, opacities=opacity.squeeze(dim=-1), colors=features,
            viewmats=viewmats, Ks=Ks, width=width, height=height, near_plane=self.cfg.znear, far_plane=self.cfg.zfar,
            sh_degree=self.geometry.active_sh_degree if self.geometry.use_sh else None,
            packed=False, absgrad=False, 
            rasterize_mode='classic', # options: classic, antialiased
            distributed=False,
            render_mode='RGB+D', # options: RGB+D, RGB+ED
        )
        rendered_image  = rendered_results[...,:-1].clone()
        rendered_depth  = rendered_results[...,-1:].clone()
        means2d         = meta['means2d']
        radii           = meta['radii'].max(dim=-1).values / float(max(width, height))
        
        if self.training: means2d.retain_grad()

        with torch.no_grad():
            rendered_semantic, _, _ = gsplat.rasterization(
                means=means3D, quats=rotations, scales=scales, opacities=opacity.squeeze(dim=-1), colors=semantic,
                viewmats=viewmats, Ks=Ks, width=width, height=height, near_plane=self.cfg.znear, far_plane=self.cfg.zfar,
                sh_degree=None,
                packed=False, absgrad=False, 
                rasterize_mode='classic',
                distributed=False,
                render_mode='RGB'
            )

        mask                    = rendered_alpha[...,0] != 0
        rendered_image[mask]    = rendered_image[mask] / rendered_alpha[mask]
        rendered_depth[mask]    = rendered_depth[mask] / rendered_alpha[mask]
        rendered_semantic[mask] = rendered_semantic[mask] / rendered_alpha[mask]

        # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
        surf_normal = depth_to_normal_batch(
            world_view_transform, full_proj_transform, rendered_depth, self.cfg.normal_space)
        # remember to multiply with accum_alpha since render_normal is unnormalized.
        # surf_normal = surf_normal * rendered_alpha

        comp_rgb        = rendered_image
        opacity         = rendered_alpha
        comp_semantic   = rendered_semantic
        comp_depth      = rendered_depth
        comp_normal     = surf_normal

        return \
            comp_rgb, \
            opacity, \
            comp_semantic, \
            comp_depth, \
            comp_normal, \
            means2d, \
            radii, \
            local_mask