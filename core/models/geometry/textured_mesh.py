import os
import math
from glob import glob
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import BaseExplicitGeometry, contract_to_unisphere
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.typing import *
from core.utils.helper import *
from core.utils.ade20k_protocol import ade20k_label2color
from core.utils.texture_utils import uv_packing, uv_unpacking
from core.utils.mesh_utils import uv_unwarpping_xatlas, compute_vertex_normal

import json
import trimesh


@threestudio.register("textured-mesh")
class TexturedMesh(BaseExplicitGeometry):
    @dataclass
    class Config(BaseExplicitGeometry.Config):
        init_strategy: str      = 'from_file'
        init_file_path: str     = ''
        is_local_space: bool    = False

        # texture (learnable)
        texture_type: str   = 'field3d'   # options: 'vertex', 'uv', 'field3d', 'field2d'
        n_feature_dims: int = 3

        # - params of uv texture
        texture_size: int   = 4096

        # - params of texture field 
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
        super().configure()

        # initialization
        if self.cfg.init_strategy == 'from_scenetex':
            v_pos, t_idx, v_nrm, v_uv, v_sem, v_idx, f_idx, instance_location, instance_size, instance_rotation = \
                self._init_from_scenetex()
        elif self.cfg.init_strategy == 'from_layout':
            v_pos, t_idx, v_nrm, v_uv, v_sem, v_idx, f_idx, instance_location, instance_size, instance_rotation = \
                self._init_from_layout()
        else:
            raise NotImplementedError
        
        self.v_pos              = nn.Parameter(v_pos, requires_grad=False)              # (Nv,3)
        self.t_idx              = nn.Parameter(t_idx, requires_grad=False)              # (Nf,3)
        self.v_nrm              = nn.Parameter(v_nrm, requires_grad=False)              # (Nv,3)
        self.v_uv               = nn.Parameter(v_uv, requires_grad=False)               # (Nv,2)
        self.v_sem              = nn.Parameter(v_sem, requires_grad=False)              # (Nv,3)

        self.v_idx              = nn.Parameter(v_idx, requires_grad=False)              # (No+n,2)
        self.f_idx              = nn.Parameter(f_idx, requires_grad=False)              # (No+n,2)

        self.instance_location  = nn.Parameter(instance_location, requires_grad=False)  # (No,3)
        self.instance_size      = nn.Parameter(instance_size, requires_grad=False)      # (No,3)
        self.instance_rotation  = nn.Parameter(instance_rotation, requires_grad=False)  # (No,3)

        self.num_instance       = self.instance_location.shape[0] + 1

        # texture (learnable)
        if self.cfg.texture_type == 'vertex':
            self.v_tex = nn.Parameter(torch.zeros((self.v_pos.shape[0], self.cfg.n_feature_dims)))
        elif self.cfg.texture_type == 'uv':
            self.texture = nn.Parameter(
                torch.zeros((self.cfg.n_feature_dims, self.cfg.texture_size, self.cfg.texture_size)))
            self.register_buffer(
                "inpaint_mask", torch.ones((1, self.cfg.texture_size, self.cfg.texture_size))
            ) # 1: inpainting 0: keeping
        elif self.cfg.texture_type == 'field3d':
            self.encoding = get_encoding(3, self.cfg.pos_encoding_config)
            self.network = get_mlp(
                self.encoding.n_output_dims, 
                self.cfg.n_feature_dims, 
                self.cfg.mlp_network_config
            )
        elif self.cfg.texture_type == 'field2d':
            self.encoding = get_encoding(2, self.cfg.pos_encoding_config)
            self.network = get_mlp(
                self.encoding.n_output_dims, 
                self.cfg.n_feature_dims, 
                self.cfg.mlp_network_config
            )
        else:
            raise NotImplementedError

        # bounding box
        min_values  = torch.min(self.v_pos, dim=0)[0]
        max_values  = torch.max(self.v_pos, dim=0)[0]
        center      = (max_values + min_values) * 0.5
        extend      = torch.max((max_values - min_values) * 0.5)
        self.bbox: Float[Tensor, "2 3"]
        self.register_buffer(
            "bbox",
            torch.stack([center - 1.1 * extend, center + 1.1 * extend], dim=0)
        )

    def _init_from_scenetex(self):
        scenetex_to_ade20k_label_alias = {
            'cabinet_shelf_desk': 'cabinet'
        }

        # load configuration file
        config_path = os.path.join(self.cfg.init_file_path, 'scene_config.json')
        assert os.path.exists(config_path), config_path
        with open(config_path, 'r') as f:
            objs = json.load(f)

        # load mesh
        mesh_paths  = [ os.path.join(self.cfg.init_file_path, v['type'], os.path.basename(v['path'])) for v in objs.values() ]
        mesh_names  = [ v['name'] for v in objs.values() if v['name'] ]
        meshes      = [ trimesh.load_mesh(p) for p in mesh_paths ]

        v_pos, t_idx, v_nrm, v_uv, v_sem = [], [], [], [], []
        v_idx, f_idx = [], []
        v_offset, f_offset = 0, 0
        for i, (mesh, name) in enumerate(zip(meshes, mesh_names)):
            assert hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv')

            rotation_matrix = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
            mesh.apply_transform(rotation_matrix)

            vertices    = np.array(mesh.vertices)
            faces       = np.array(mesh.faces)
            normals     = np.array(mesh.vertex_normals)
            uv_coords   = np.array(mesh.visual.uv)

            v_idx.append([v_offset, v_offset+vertices.shape[0]])
            f_idx.append([f_offset, f_offset+faces.shape[0]])

            faces       += v_offset
            v_offset    += vertices.shape[0]
            f_offset    += faces.shape[0]

            if name == 'room':
                semantics = np.array([ade20k_label2color['wall']]*vertices.shape[0])
            else:
                name = scenetex_to_ade20k_label_alias[name] \
                    if name in scenetex_to_ade20k_label_alias.keys() else name
                assert name in ade20k_label2color.keys(), name
                semantics = np.array([ade20k_label2color[name]]*vertices.shape[0])
            v_sem.append(semantics / 255.)
            
            v_pos.append(vertices)
            t_idx.append(faces)
            v_nrm.append(normals)
            v_uv.append(uv_packing(uv_coords, i, len(meshes)))

        v_pos   = torch.tensor(np.concatenate(v_pos, axis=0), dtype=torch.float32)
        t_idx   = torch.tensor(np.concatenate(t_idx, axis=0), dtype=torch.long)
        v_nrm   = torch.tensor(np.concatenate(v_nrm, axis=0), dtype=torch.float32)
        v_uv    = torch.tensor(np.concatenate(v_uv, axis=0), dtype=torch.float32)
        v_sem   = torch.tensor(np.concatenate(v_sem, axis=0), dtype=torch.float32)
        v_idx   = torch.tensor(np.array(v_idx), dtype=torch.long)
        f_idx   = torch.tensor(np.array(f_idx), dtype=torch.long)
        
        instance_location   = torch.zeros((v_idx.shape[0]-1, 3), dtype=torch.float32)
        instance_size       = torch.ones((v_idx.shape[0]-1, 3), dtype=torch.float32)
        instance_rotation   = torch.zeros((v_idx.shape[0]-1, 3), dtype=torch.float32)

        return v_pos, t_idx, v_nrm, v_uv, v_sem, v_idx, f_idx, instance_location, instance_size, instance_rotation

    def _init_from_layout(self):
        assert os.path.isdir(self.cfg.init_file_path)

        # load layout
        instance_location, instance_size, instance_rotation, instance_class, instance_prompt = [], [], [], [], []
        with open(os.path.join(self.cfg.init_file_path, '..', 'layout.json'), 'r') as f:
            data = json.load(f)
            bbox, background = data['bbox'], data['background']
        for b in bbox:
            instance_location.append(torch.tensor(b['location']))
            instance_size.append(torch.tensor(b['size']))
            instance_rotation.append(torch.tensor(b['rotation']))
            instance_class.append(b['class'])
            instance_prompt.append(b['prompt'])

        instance_location   = torch.stack(instance_location)
        instance_size       = torch.stack(instance_size)
        instance_rotation   = torch.stack(instance_rotation)

        # object mesh
        mesh_paths  = glob(os.path.join(self.cfg.init_file_path, '*'))
        mesh_paths  = sorted(mesh_paths, key=lambda s: int(os.path.basename(s).split('.')[0]))
        meshes      = [trimesh.load_mesh(p) for p in mesh_paths]
        assert hasattr(meshes[0], 'faces') and (meshes[0].faces is not None) and len(meshes[0].faces) > 0

        # background mesh
        ceiling = trimesh.Trimesh(vertices=background["vertices"], faces=background["faces"]["ceiling"])
        ceiling.remove_unreferenced_vertices()

        floor   = trimesh.Trimesh(vertices=background["vertices"], faces=background["faces"]["floor"])
        floor.remove_unreferenced_vertices()

        wall    = trimesh.Trimesh(vertices=background["vertices"], faces=background["faces"]["walls"])
        wall_meshes = []
        for wall_face in wall.faces:
            wall_mesh = trimesh.Trimesh(vertices=background["vertices"], faces=[wall_face])
            wall_mesh.remove_unreferenced_vertices()
            wall_meshes.append(wall_mesh)
        wall    = trimesh.util.concatenate(wall_meshes)

        meshes          += [ceiling, floor, wall]
        instance_class  += ['ceiling', 'floor', 'wall']

        v_pos, t_idx, v_nrm, v_uv, v_sem = [], [], [], [], []
        v_idx, f_idx = [], []
        v_offset, f_offset = 0, 0
        for i, (mesh, cls) in enumerate(zip(meshes, instance_class)):

            if self.cfg.is_local_space and (cls not in ['ceiling', 'floor', 'wall']):
                mesh.apply_scale(instance_size[i].max().item())

                rotation = trimesh.transformations.rotation_matrix(
                    np.radians(instance_rotation[i][2].item()), [0, 0, 1])
                mesh.apply_transform(rotation)

                mesh.apply_translation(instance_location[i])

            vertices    = torch.tensor(mesh.vertices)
            faces       = torch.tensor(mesh.faces)
            normals     = torch.tensor(mesh.vertex_normals)

            if hasattr(mesh.visual, 'uv') and (mesh.visual.uv is not None):
                uv_coords = torch.tensor(mesh.visual.uv, dtype=torch.float32)
            else:
                threestudio.info(
                    "Perform UV padding on texture maps to avoid seams, may take a while ..."
                )
                vmapping, faces, uv_coords = uv_unwarpping_xatlas(vertices, faces)

                vertices    = vertices[vmapping]
                normals     = normals[vmapping]
            
            v_idx.append([v_offset, v_offset+vertices.shape[0]])
            f_idx.append([f_offset, f_offset+faces.shape[0]])

            faces       += v_offset
            v_offset    += vertices.shape[0]
            f_offset    += faces.shape[0]

            v_pos.append(vertices)
            t_idx.append(faces)
            v_nrm.append(normals)
            v_uv.append(uv_packing(uv_coords, i, len(meshes)))

            assert cls in ade20k_label2color.keys(), cls
            semantics = torch.tensor([ade20k_label2color[cls]]*vertices.shape[0])
            v_sem.append(semantics / 255.)

        v_pos   = torch.cat(v_pos, dim=0).float()
        t_idx   = torch.cat(t_idx, dim=0).long()
        v_nrm   = torch.cat(v_nrm, dim=0).float()
        v_uv    = torch.cat(v_uv, dim=0).float()
        v_sem   = torch.cat(v_sem, dim=0).float()
        v_idx   = torch.tensor(np.array(v_idx), dtype=torch.long)
        f_idx   = torch.tensor(np.array(f_idx), dtype=torch.long)
        
        return v_pos, t_idx, v_nrm, v_uv, v_sem, v_idx, f_idx, instance_location, instance_size, instance_rotation

    def get_global_params(self):
        vertex_pos  = self.v_pos
        face_idx    = self.t_idx
        vertex_nrm  = self.v_nrm
        vertex_sem  = self.v_sem
        vertex_uv   = self.v_uv
        return vertex_pos, face_idx, vertex_nrm, vertex_sem, vertex_uv

    def get_local_params(self, index):
        n_objects       = self.instance_location.shape[0]
        is_background   = index >= n_objects

        if is_background:
            v_start, v_end = self.v_idx[n_objects][0], None
            f_start, f_end = self.f_idx[n_objects][0], None
        else:
            v_start, v_end = self.v_idx[index]
            f_start, f_end = self.f_idx[index]

        vertex_pos      = self.v_pos[v_start:v_end]
        face_idx        = self.t_idx[f_start:f_end] - v_start
        vertex_nrm      = self.v_nrm[v_start:v_end]
        vertex_sem      = self.v_sem[v_start:v_end]
        vertex_uv       = self.v_uv[v_start:v_end]

        if is_background:
            mx, mn  = vertex_pos.max(dim=0)[0], vertex_pos.min(dim=0)[0]
            sz      = (mx - mn).min()
            loc     = (mx + mn) * 0.5
            vertex_pos = (vertex_pos - loc) / sz * 5.
        else:
            loc         = self.instance_location[index]
            rot         = self.instance_rotation[index]
            sz          = self.instance_size[index].max()[None]

            vertex_pos  = translate_rotate_scale(vertex_pos, 1. / sz, torch.deg2rad(-rot), - loc)

            rot_mat     = euler_to_rotation_matrix(torch.deg2rad(-rot))
            vertex_nrm  = vertex_nrm @ rot_mat.T

        return vertex_pos, face_idx, vertex_nrm, vertex_sem, vertex_uv

    def forward(self, points: Float[Tensor, "*N Di"]) -> Dict[str, Float[Tensor, "..."]]:
        if self.cfg.texture_type == 'uv':
            assert points.shape[-1] == 2
            points = points * 2. - 1.               # [0,1] -> [-1,1]
            features = F.grid_sample(
                self.texture[None],                 # (1,C,H,W)
                points.view(-1,2)[None,:, None,:],  # (1,N,1,2)
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=False)
            features = features[0,:,:,0].permute(1,0).view(
                *points.shape[:-1], self.cfg.n_feature_dims
            )
        elif self.cfg.texture_type == 'field3d':
            assert points.shape[-1] == 3
            points_unscaled = points  # points in the original scale
            points = contract_to_unisphere(points_unscaled, self.bbox)  # points normalized to (0, 1)
            enc = self.encoding(points.view(-1, 3))
            features = self.network(enc).view(
                *points.shape[:-1], self.cfg.n_feature_dims
            )
        elif self.cfg.texture_type == 'field2d':
            assert points.shape[-1] == 2
            enc = self.encoding(points.view(-1, 2))
            features = self.network(enc).view(
                *points.shape[:-1], self.cfg.n_feature_dims
            )
        else:
            raise NotImplementedError
        return {"features": features}

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        return self.forward(points)