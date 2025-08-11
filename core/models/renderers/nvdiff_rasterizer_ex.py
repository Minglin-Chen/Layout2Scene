from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseExplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *

from einops import repeat


@threestudio.register("nvdiff-rasterizer-ex")
class NVDiffRasterizerEx(Rasterizer):
    @dataclass
    class Config(VolumeRenderer.Config):
        context_type: str           = "cuda"

        depth_type: str             = 'reverse_adaptive'    # options: 'raw', 'reverse_adaptive', etc
        depth_norm_radius: float    = 1.0
        depth_min_value: float      = 0.0
        depth_max_value: float      = 2.0
        depth_padding_value: float  = 10. / 255.
        depth_target_min: float     = 50. / 255.
        depth_target_max: float     = 255./ 255.
        
        normal_space: str           = 'camera'              # options: 'camera', 'world'

    cfg: Config

    def configure(
        self,
        geometry: BaseExplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())

    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        height: int,
        width: int,
        local: bool = False,
        local_index: Optional[List] = None,
        **kwargs
    ) -> Dict[str, Any]:
        batch_size = mvp_mtx.shape[0]
        
        w2c         = self.c2wtow2c(c2w)

        vertex_pos, face_idx, vertex_nrm, vertex_sem, vertex_uv = \
            self.geometry.get_global_params() if not local else self.geometry.get_local_params(local_index)

        v_pos_clip: Float[Tensor, "B Nv 4"]
        v_pos_clip  = self.ctx.vertex_transform(vertex_pos, mvp_mtx)
        rast, _     = self.ctx.rasterize(v_pos_clip, face_idx, (height, width))

        # opacity
        mask        = rast[..., 3:] > 0
        mask_aa     = self.ctx.antialias(mask.float(), rast, v_pos_clip, face_idx)

        # normal [-1,1] -> [0,1]
        if self.cfg.normal_space == 'world':
            normal, _   = self.ctx.interpolate_one(vertex_nrm, rast, face_idx)
            normal_bg   = torch.zeros_like(normal)
        elif self.cfg.normal_space == 'camera':
            batch_v_nrm = repeat(vertex_nrm, "n c -> b n c", b=batch_size)
            v_nrm       = self.world2camera(batch_v_nrm, w2c)
            normal, _   = self.ctx.interpolate(v_nrm, rast, face_idx)
            normal_bg   = torch.zeros_like(normal)
            normal_bg[...,2] = 1.0
        else:
            raise NotImplementedError
        normal      = F.normalize(normal, dim=-1)
        normal_aa   = torch.lerp((normal_bg+1.0)*0.5, (normal+1.0)*0.5, mask.float())
        normal_aa   = self.ctx.antialias(normal_aa, rast, v_pos_clip, face_idx)
        
        # depth [0,+inf]
        v_pos       = repeat(vertex_pos, "n c -> b n c", b=batch_size)
        v_pos       = self.homo_proj(v_pos, w2c)
        z_depth     = - v_pos[..., 2:]
        depth, _    = self.ctx.interpolate(z_depth, rast, face_idx)
        depth_aa    = self.ctx.antialias(depth, rast, v_pos_clip, face_idx)

        # [0,+inf] -> [0,1]
        if self.cfg.depth_type == 'raw':
            pass

        elif self.cfg.depth_type == 'inverse':
            invalid_mask            = depth_aa <= 0.0
            depth_valid             = depth_aa[~invalid_mask]

            depth_valid             = 1. / depth_valid  # disparity
            depth_aa[~invalid_mask] = depth_valid
            depth_aa[invalid_mask]  = 0.0

        elif self.cfg.depth_type == 'inverse_adaptive':
            invalid_mask            = depth_aa <= 0.0
            depth_valid             = depth_aa[~invalid_mask]
            depth_valid             = 1. / depth_valid  # disparity
            depth_min               = depth_valid.min()
            depth_max               = depth_valid.max()
            depth_valid             = (depth_valid - depth_min) / (depth_max - depth_min)
            depth_aa[~invalid_mask] = depth_valid
            depth_aa[invalid_mask]  = 0.0

        elif self.cfg.depth_type == 'reverse_minmax':
            min_val                 = self.cfg.depth_min_value
            max_val                 = self.cfg.depth_max_value

            invalid_mask            = depth_aa <= 0.0
            depth_valid             = depth_aa[~invalid_mask]

            # reverse
            depth_valid             = (max_val - depth_valid) / (max_val - min_val)
            depth_valid             = torch.clamp(depth_valid, 0, 1)

            depth_aa[~invalid_mask] = depth_valid
            depth_aa[invalid_mask]  = 0.0

        elif self.cfg.depth_type == 'reverse_adaptive':
            # from SceneTex and RichDreamer
            invalid_mask            = depth_aa <= 0.0

            depth_valid             = depth_aa[~invalid_mask]
            depth_min               = depth_valid.min()
            depth_max               = depth_valid.max()

            # reverse
            depth_valid             = (depth_max - depth_valid) / (depth_max - depth_min)
            depth_valid             = depth_valid * \
                                        (self.cfg.depth_target_max - self.cfg.depth_target_min) + \
                                            self.cfg.depth_target_min

            depth_aa[~invalid_mask] = depth_valid
            depth_aa[invalid_mask]  = self.cfg.depth_padding_value  # not completely black

        elif self.cfg.depth_type == 'norm_radius':
            # from RichDreamer
            cam_dist                = camera_distances.reshape(-1,1,1,1).expand_as(depth_aa)
            min_val                 = cam_dist - self.cfg.depth_norm_radius
            max_val                 = cam_dist + self.cfg.depth_norm_radius

            invalid_mask            = depth_aa <= 0.0
            depth_valid             = depth_aa[~invalid_mask]
            depth_valid             = (max_val[~invalid_mask] - depth_valid) / (2 * self.cfg.depth_norm_radius)
            depth_valid             = torch.clamp(depth_valid, 0, 1)

            depth_aa[~invalid_mask] = depth_valid
            depth_aa[invalid_mask]  = 0.0
        
        else:
            raise ValueError(self.cfg.depth_type)

        # semantic [0,1]
        semantic, _ = self.ctx.interpolate_one(vertex_sem, rast, face_idx)
        semantic_aa = self.ctx.antialias(semantic, rast, v_pos_clip, face_idx)

        # rgb
        selector    = mask[..., 0]

        gb_pos, _   = self.ctx.interpolate_one(vertex_pos, rast, face_idx)
        gb_viewdirs = F.normalize(gb_pos - camera_positions[:, None, None, :], dim=-1)
        positions   = gb_pos[selector]

        if self.geometry.cfg.texture_type == 'vertex':
            assert not local
            features    = self.ctx.interpolate_one(self.geometry.v_tex, rast, face_idx)
            features    = features[selector]
            geo_out     = {"features": features}
        elif self.geometry.cfg.texture_type in ['uv', 'field2d']:
            uv_coord, _ = self.ctx.interpolate_one(vertex_uv, rast, face_idx)
            geo_out     = self.geometry(uv_coord[selector])
        elif self.geometry.cfg.texture_type == 'field3d':
            geo_out     = self.geometry(positions)
        else:
            raise NotImplementedError

        rgb_fg = self.material(
            viewdirs=gb_viewdirs[selector],
            positions=positions,
            **geo_out
        )
        gb_rgb_fg = torch.zeros(batch_size, height, width, 3).to(rgb_fg)
        gb_rgb_fg[selector] = rgb_fg

        gb_rgb_bg   = self.background(dirs=gb_viewdirs)
        gb_rgb      = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
        gb_rgb_aa   = self.ctx.antialias(gb_rgb, rast, v_pos_clip, face_idx)
        
        out = {
            "comp_rgb":         gb_rgb_aa,
            "opacity":          mask_aa,
            "comp_depth":       depth_aa,
            "comp_normal":      normal_aa,
            "comp_semantic":    semantic_aa,
        }
        if self.geometry.cfg.texture_type in ['uv', 'field2d']:
            out['uv_coord'] = uv_coord

        return out

    @staticmethod
    def c2wtow2c(c2w: Float[Tensor, "B 4 4"]) -> Float[Tensor, "B 4 4"]:
        """transfer camera2world to world2camera matrix"""

        w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
        w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
        w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
        w2c[:, 3, 3] = 1.0

        return w2c
    
    @staticmethod
    def world2camera(normal, w2c):
        rotate: Float[Tensor, "B 4 4"] = w2c[..., :3, :3]
        camera_normal = normal @ rotate.permute(0, 2, 1)
        # pixel space flip axis so we need built negative y-axis normal
        flip_x = torch.eye(3).to(w2c)
        flip_x[0, 0] = -1

        camera_normal = camera_normal @ flip_x[None, ...]

        return camera_normal
    
    @staticmethod
    def homo_proj(p: Float[Tensor, "B n 3"], matrix: Float[Tensor, "B 4 4"]) -> Float[Tensor, "B n 3"]:
        # NOTE that only at homo_weight = 1

        r_p: Float[Tensor, "B n 3"] = (matrix[:, :3, :3] @ p.transpose(2, 1)).transpose(
            2, 1
        )
        t_p: Float[Tensor, "B 1 3"] = matrix[:, :3, 3][:, None, :]
        return r_p + t_p