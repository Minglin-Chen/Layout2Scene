from dataclasses import dataclass, field
import os
import os.path as osp
from glob import glob
import numpy as np

from einops import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import get_activation, scale_tensor
from threestudio.utils.typing import *

from core.utils.ade20k_protocol import ade20k_label2color
from core.utils.helper import load_mesh, parse_background_meshes, parse_background_mesh
from core.utils.layout_utils import load_layout

import trimesh


@threestudio.register("room-structure-background")
class RoomStructureBackground(BaseBackground):
    @dataclass
    class Config(BaseBackground.Config):
        context_type: str           = 'cuda'

        # geometry
        init_strategy: str          = 'from_layout'
        init_file_path: str         = ''
        init_scale: float           = 1.0

        normal_space: str           = 'camera'      # options: 'camera', 'world'

        # texture (learnable)
        texture_type: str           = 'field'       # options: 'explicit', 'implicit', 'field'
        rgb_as_latents: bool        = False
        rgb_activation: str         = 'scale_-11_01'

        # - params of explicit texture
        subdivide_max_edge: float   = 0.01

        # - params of implicit texture
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        
    cfg: Config

    def configure(self) -> None:
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())

        t2c = {k: [v[0]/255., v[1]/255., v[2]/255.] for k, v in ade20k_label2color.items()}

        # geometry
        if self.cfg.init_strategy == 'from_layout':
            assert os.path.exists(self.cfg.init_file_path)
            _, background_mesh = load_layout(self.cfg.init_file_path)
            vertices, faces, normals, semantics = \
                parse_background_mesh(background_mesh, t2c['ceiling'], t2c['floor'], t2c['wall'])

        elif self.cfg.init_strategy == 'cube':
            # up: +z
            background_mesh = trimesh.creation.box(extents=[2.0 * self.cfg.init_scale] * 3)
            background_mesh.faces[:,[0,2]] = background_mesh.faces[:,[2,0]]
            vertices, faces, normals, semantics = \
                parse_background_mesh(background_mesh, t2c['ceiling'], t2c['floor'], t2c['wall'])
            
        elif self.cfg.init_strategy == 'from_file':
            filepath = glob(osp.join(self.cfg.init_file_path, 'background.*'))[0] \
                if osp.isdir(self.cfg.init_file_path) else self.cfg.init_file_path
            background_mesh = load_mesh(filepath)
            vertices, faces, normals, semantics = \
                parse_background_mesh(background_mesh, t2c['ceiling'], t2c['floor'], t2c['wall'])

        elif self.cfg.init_strategy == 'from_files':
            assert osp.isdir(self.cfg.init_file_path)
            ceiling_meshes, floor_meshes, wall_meshes = [], [], []
            mesh_paths = glob(osp.join(self.cfg.init_file_path, '*'))
            for p in mesh_paths:
                category = osp.basename(p).split('.')[0].split('_')[-1]
                if category.lower() == 'ceiling':
                    ceiling_meshes.append(load_mesh(p))
                elif category.lower() == 'floor':
                    floor_meshes.append(load_mesh(p))
                elif category.lower() == 'wall':
                    wall_meshes.append(load_mesh(p))
                else:
                    raise ValueError
            vertices, faces, normals, semantics = \
                parse_background_meshes(ceiling_meshes, floor_meshes, wall_meshes, t2c['ceiling'], t2c['floor'], t2c['wall'])
            
        else:
            raise ValueError
        
        vertices    = torch.as_tensor(vertices, dtype=torch.float32).contiguous()
        faces       = torch.as_tensor(faces, dtype=torch.long).contiguous()
        normals     = torch.as_tensor(normals, dtype=torch.float32).contiguous()
        semantics   = torch.as_tensor(semantics, dtype=torch.float32).contiguous()
        self.register_buffer("v_pos", vertices)
        self.register_buffer("t_idx", faces)
        self.register_buffer("v_nrm", normals)
        self.register_buffer("v_sem", semantics)

        self.register_buffer("bbox", torch.stack([self.v_pos.min(dim=0)[0], self.v_pos.max(dim=0)[0]]))

        # texture (learnable)
        if self.cfg.texture_type == 'explicit':
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh = mesh.subdivide_to_size(max_edge=self.cfg.subdivide_max_edge)
            
            self.register_buffer("tex_verts", torch.as_tensor(mesh.vertices, dtype=torch.float32))
            self.register_buffer("tex_faces", torch.as_tensor(mesh.faces, dtype=torch.long))
            
            if self.cfg.rgb_as_latents:
                self.tex_rgbs = nn.Parameter(torch.randn((self.tex_verts.shape[0], 4)))
            else:
                self.tex_rgbs = nn.Parameter(torch.randn((self.tex_verts.shape[0], 3))) 

        elif self.cfg.texture_type in ['implicit', 'field']:
            self.encoding = get_encoding(3, self.cfg.pos_encoding_config)
            n_output_dims = 4 if self.cfg.rgb_as_latents else 3
            self.network = get_mlp(
                self.encoding.n_output_dims, 
                n_output_dims, 
                self.cfg.mlp_network_config
            )

        else:
            raise NotImplementedError
        
    def forward(
        self, 
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        height: int,
        width: int,
        **kwargs
    ) -> Dict[str, Any]:
        batch_size  = mvp_mtx.shape[0]

        w2c         = self.c2wtow2c(c2w)

        v_pos_clip: Float[Tensor, "B Nv 4"]
        v_pos_clip  = self.ctx.vertex_transform(self.v_pos, mvp_mtx)
        rast, _     = self.ctx.rasterize(v_pos_clip, self.t_idx, (height, width))

        # opacity
        mask        = rast[..., 3:] > 0
        mask_aa     = self.ctx.antialias(mask.float(), rast, v_pos_clip, self.t_idx)

        # normal [-1,1]
        if self.cfg.normal_space == 'world':
            normal, _   = self.ctx.interpolate_one(self.v_nrm, rast, self.t_idx)
        elif self.cfg.normal_space == 'camera':
            batch_v_nrm = repeat(self.v_nrm, "n c -> b n c", b=batch_size)
            v_nrm       = self.world2camera(batch_v_nrm, w2c)
            normal, _   = self.ctx.interpolate(v_nrm, rast, self.t_idx)
        else:
            raise NotImplementedError
        normal      = F.normalize(normal, dim=-1)
        normal_aa   = self.ctx.antialias(normal, rast, v_pos_clip, self.t_idx)

        # depth [0,+inf]
        v_pos       = repeat(self.v_pos, "n c -> b n c", b=batch_size)
        v_pos       = self.homo_proj(v_pos, w2c)
        z_depth     = - (v_pos[..., 2:] + 1e-9)
        depth, _    = self.ctx.interpolate(z_depth, rast, self.t_idx)
        depth_aa    = self.ctx.antialias(depth, rast, v_pos_clip, self.t_idx)

        # semantic [0,1]
        semantic, _ = self.ctx.interpolate_one(self.v_sem, rast, self.t_idx)
        semantic_aa = self.ctx.antialias(semantic, rast, v_pos_clip, self.t_idx)

        # rgb
        rgb         = self.render_texture(mvp_mtx, height, width)
        
        out = {
            "rgb":      rgb,
            "opacity":  mask_aa,
            "depth":    depth_aa,
            "normal":   normal_aa,
            "semantic": semantic_aa,
        }

        return out
    
    def render_texture(self, mvp_mtx: Float[Tensor, "B 4 4"], height: int, width: int):
        if self.cfg.texture_type == 'explicit':
            v_pos_clip: Float[Tensor, "B Nv 4"]
            v_pos_clip  = self.ctx.vertex_transform(self.tex_verts, mvp_mtx)
            rast, _     = self.ctx.rasterize(v_pos_clip, self.tex_faces, (height, width))

            rgb, _  = self.ctx.interpolate_one(self.tex_rgbs, rast, self.tex_faces)
            rgb     = self.ctx.antialias(rgb, rast, v_pos_clip, self.tex_faces)
            
        elif self.cfg.texture_type == 'implicit':
            v_pos_clip: Float[Tensor, "B Nv 4"] 
            v_pos_clip  = self.ctx.vertex_transform(self.v_pos, mvp_mtx)
            rast, _     = self.ctx.rasterize(v_pos_clip, self.t_idx, (height, width))

            n_faces = self.t_idx.shape[0]
            uvi = rast[...,[0,1,3]]
            uvi[...,-1] /= (n_faces + 1) # i: (0,1), suppose the ray hits atleast one vertice
            enc = self.encoding(uvi.view(-1, 3))
            rgb = self.network(enc).view(*uvi.shape[:-1], 4 if self.cfg.rgb_as_latents else 3)
        
        elif self.cfg.texture_type == 'field':
            v_pos_clip: Float[Tensor, "B Nv 4"] 
            v_pos_clip  = self.ctx.vertex_transform(self.v_pos, mvp_mtx)
            rast, _     = self.ctx.rasterize(v_pos_clip, self.t_idx, (height, width))

            points, _   = self.ctx.interpolate_one(self.v_pos, rast, self.t_idx)
            points      = self.ctx.antialias(points, rast, v_pos_clip, self.t_idx)

            scaled_points = scale_tensor(points, self.bbox, (0, 1))
            enc = self.encoding(scaled_points.view(-1, 3))
            rgb = self.network(enc).view(*scaled_points.shape[:-1], 4 if self.cfg.rgb_as_latents else 3)

        else:
            raise NotImplementedError
        
        if not self.cfg.rgb_as_latents:
            rgb = get_activation(self.cfg.rgb_activation)(rgb)

        return rgb

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