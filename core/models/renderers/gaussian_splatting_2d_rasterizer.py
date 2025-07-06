from dataclasses import dataclass
import math
import numpy as np

import torch

import threestudio
from threestudio.models.renderers.base import Rasterizer
from threestudio.utils.typing import *
from threestudio.utils.ops import get_cam_info_gaussian

from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from core.utils.helper import fov2focal, focal2fov, quaternion_multiply
from core.utils.point_utils import depth_to_normal


@threestudio.register("gaussian-splatting-2d-rasterizer")
class GaussianSplatting2DRasterizer(Rasterizer):
    @dataclass
    class Config(Rasterizer.Config):
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

        object_dropout_rate: float  = 0.0

        # unused
        # - radius

    cfg: Config

    def configure(
        self,
        geometry, material, background
    ) -> None:
        super().configure(geometry, material, background)
        # dropout objects during inference
        self.object_dropout_ids     = []
        # edit objects during inference
        self.object_translations    = [[0.,0.,0.] for _ in range(self.geometry.num_instance)]
        self.object_rotations       = [0. for _ in range(self.geometry.num_instance)]
        self.object_scales          = [1. for _ in range(self.geometry.num_instance)]

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
        fg_rgb, opacity, fg_semantic, fg_depth, fg_normal, fg_surf_normal, \
            depth_distortion, viewspace_points, visibility_filter, radii = \
                self.__splatting_batch(c2w, fovy, height, width, local, local_index)

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
        surf_normal     = opacity * fg_surf_normal + (1. - opacity) * bg_normal
        comp_semantic   = opacity * fg_semantic + (1. - opacity) * bg_semantic

        # correctness
        mask = (fg_depth[...,0] > bg_depth[...,0]) & (bg_depth[...,0] > 0.0)
        comp_rgb[mask]      = bg_rgb[mask]
        comp_depth[mask]    = bg_depth[mask]
        comp_normal[mask]   = bg_normal[mask]
        surf_normal[mask]   = bg_normal[mask]
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

        elif depth_type in 'reverse_minmax':
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
        surf_normal = (surf_normal + 1.) * 0.5
        
        out = {
            "comp_rgb"          : comp_rgb,
            "opacity"           : opacity,
            "comp_depth"        : comp_depth,
            "comp_normal"       : comp_normal,
            "surf_normal"       : surf_normal,
            "comp_semantic"     : comp_semantic,
            "depth_distortion"  : depth_distortion,
            "viewspace_points"  : viewspace_points,
            "visibility_filter" : visibility_filter,
            "radii"             : radii,
        }

        return out
        
    def __splatting(
        self, 
        c2w: Float[Tensor, "4 4"], 
        fovx: float, 
        fovy: float,
        height: int,
        width: int,
        local: bool,
        local_index: int,
    ) -> tuple[Float[Tensor, "..."]]:
        world_view_transform, full_proj_transform, camera_center = \
            get_cam_info_gaussian(c2w, fovx, fovy, self.cfg.znear, self.cfg.zfar)

        means3D, opacity, scales, rotations, features, semantic, instance, local_mask = \
            self.geometry.get_global_params() if not local else self.geometry.get_local_params(local_index)

        # object dropout through masking opacity
        if self.training and (self.cfg.object_dropout_rate >= 0.0):
            opacity_mask = torch.ones_like(opacity)
            for i in range(self.geometry.num_instance):
                opacity_mask[instance==i] = int(np.random.rand() > self.cfg.object_dropout_rate)
            opacity = opacity * opacity_mask
        else:
            opacity_mask = torch.ones_like(opacity)
            for i in self.object_dropout_ids:
                opacity_mask[instance==i] = 0.0
            opacity = opacity * opacity_mask

        if not self.training:
            means3D     = means3D.clone()
            scales      = scales.clone()
            rotations   = rotations.clone()
            for i in range(self.geometry.num_instance):
                selected_index = instance[...,0]==i

                o_s, o_t, o_phi = \
                    self.object_scales[i], self.object_translations[i], self.object_rotations[i]
                o_t = torch.tensor(o_t).to(means3D)
                o_R = torch.tensor([
                    [np.cos(o_phi), -np.sin(o_phi), 0.0],
                    [np.sin(o_phi),  np.cos(o_phi), 0.0],
                    [           0.0,           0.0, 1.0]
                ]).to(means3D)

                o_t_orig = self.geometry.instance_location[i].to(means3D)

                obj_xyz = means3D[selected_index] - o_t_orig
                obj_xyz = (obj_xyz @ o_R.T) * o_s + o_t_orig + o_t
                means3D[selected_index] = obj_xyz

                scales[selected_index] *= o_s

                o_q = torch.tensor([np.cos(o_phi*0.5), 0.0, 0.0, np.sin(o_phi*0.5)]).to(means3D)
                rotations[selected_index] = quaternion_multiply(o_q, rotations[selected_index])
                
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        means2D = torch.zeros_like(
            self.geometry.get_xyz, dtype=means3D.dtype, requires_grad=True, device=means3D.device) + 0
        try:
            means2D.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        raster_settings = GaussianRasterizationSettings(
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
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        rendered_image, radii, allmap = rasterizer(
            means3D         = means3D,
            means2D         = means2D if not local else means2D[local_mask],
            shs             = features if self.geometry.use_sh else None,
            colors_precomp  = None if self.geometry.use_sh else features,
            opacities       = opacity,
            scales          = scales,
            rotations       = rotations,
            cov3D_precomp   = None
        )

        with torch.no_grad():
            rendered_semantic, _, _ = rasterizer(
                means3D         = means3D,
                means2D         = means2D if not local else means2D[local_mask],
                shs             = None,
                colors_precomp  = semantic,
                opacities       = opacity,
                scales          = scales,
                rotations       = rotations,
                cov3D_precomp   = None
            )

        rendered_alpha = allmap[1:2]

        # get normal map [-1,1]
        rendered_normal = allmap[2:5]
        if self.cfg.normal_space == 'world':
            # transform normal from view space to world space
            rendered_normal = (rendered_normal.permute(1,2,0) @ (world_view_transform[:3,:3].T)).permute(2,0,1)
        elif self.cfg.normal_space == 'camera':
            # camera coordinate: OpenCV (up:-y,right:+x,forward:+z) -> OpenGL (up:+y,right:+x,forward:-z)
            # flip x-axis
            rendered_normal *= -1.0

        # get median depth map
        render_depth_median = allmap[5:6]
        # render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

        # get expected depth map
        # render_depth_expected = allmap[0:1]
        # render_depth_expected = (render_depth_expected / rendered_alpha)
        # render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
        
        # get depth distortion map
        depth_distortion = allmap[6:7]

        # psedo surface attributes
        # surf depth is either median or expected by setting depth_ratio to 1 or 0
        # for bounded scene, use median depth, i.e., depth_ratio = 1;
        # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
        # surf_depth = render_depth_expected * (1-self.cfg.depth_ratio) + (self.cfg.depth_ratio) * render_depth_median
        surf_depth = render_depth_median

        # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
        surf_normal = depth_to_normal(world_view_transform, full_proj_transform, surf_depth, self.cfg.normal_space)
        surf_normal = surf_normal.permute(2,0,1)
        # remember to multiply with accum_alpha since render_normal is unnormalized.
        # surf_normal = surf_normal * (rendered_alpha).detach()

        mask                        = rendered_alpha[0] != 0
        rendered_image[:,mask]      = rendered_image[:,mask] / rendered_alpha[:,mask]
        rendered_semantic[:,mask]   = rendered_semantic[:,mask] / rendered_alpha[:,mask]
        rendered_normal[:,mask]     = rendered_normal[:,mask] / rendered_alpha[:,mask]

        comp_rgb        = rendered_image.unsqueeze(0).permute(0,2,3,1)
        opacity         = rendered_alpha.unsqueeze(0).permute(0,2,3,1)
        comp_semantic   = rendered_semantic.unsqueeze(0).permute(0,2,3,1)
        comp_depth      = surf_depth.unsqueeze(0).permute(0,2,3,1)
        comp_normal     = rendered_normal.unsqueeze(0).permute(0,2,3,1)
        surf_normal     = surf_normal.unsqueeze(0).permute(0,2,3,1)

        radii_full = radii
        if local:
            radii_full = torch.zeros((self.geometry.get_xyz.shape[0]), dtype=radii.dtype, device=radii.device)
            radii_full[local_mask] = radii

        depth_distortion    = depth_distortion.unsqueeze(0).permute(0,2,3,1)
        viewspace_points    = means2D
        visibility_filter   = radii_full > 0

        return \
            comp_rgb, \
            opacity, \
            comp_semantic, \
            comp_depth, \
            comp_normal, \
            surf_normal, \
            depth_distortion, \
            viewspace_points, \
            visibility_filter, \
            radii_full
    
    def __splatting_batch(
        self, 
        c2w: Float[Tensor, "B 4 4"], 
        fovy: Float[Tensor, "B"],
        height: int,
        width: int,
        local: bool = False,
        local_index: int = -1,
    ):
        rgb_list, opacity_list, semantic_list, depth_list, normal_list, surf_normal_list = [], [], [], [], [], []
        depth_distortion_list, viewspace_points_list, visibility_filter_list, radii_list = [], [], [], []
        for _c2w, _fovy in zip(c2w, fovy):
            focal = fov2focal(_fovy, height)
            _fovx = focal2fov(focal, width)

            rgb, opacity, semantic, depth, normal, surf_normal, \
                depth_distortion, viewspace_points, visibility_filter, radii = \
                    self.__splatting(_c2w, _fovx, _fovy, height, width, local, local_index)
            
            rgb_list.append(rgb)
            opacity_list.append(opacity)
            semantic_list.append(semantic)
            depth_list.append(depth)
            normal_list.append(normal)
            surf_normal_list.append(surf_normal)

            depth_distortion_list.append(depth_distortion)
            viewspace_points_list.append(viewspace_points)
            visibility_filter_list.append(visibility_filter)
            radii_list.append(radii)

        return \
            torch.cat(rgb_list), \
            torch.cat(opacity_list), \
            torch.cat(semantic_list), \
            torch.cat(depth_list), \
            torch.cat(normal_list), \
            torch.cat(surf_normal_list), \
            torch.cat(depth_distortion_list), \
            viewspace_points_list, \
            visibility_filter_list, \
            radii_list