from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from tqdm import trange

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.exporters.base import Exporter, ExporterOutput
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *
from threestudio.utils.ops import get_activation

from core.utils.texture_utils import uv_packing, uv_unpacking


@threestudio.register("textured-mesh-exporter")
class TexturedMeshExporter(Exporter):
    @dataclass
    class Config(Exporter.Config):
        save_name: str = "model"
        save_suffix: str = "glb"
        save_normal: bool = False
        save_uv: bool = True
        save_texture: bool = True
        texture_size: int = 2048
        context_type: str = "cuda"

        material_strategy: str = 'sample'

        # padding_size: int = 8

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, self.device)

    def __call__(self) -> List[ExporterOutput]:
        n_instance = self.geometry.v_idx.shape[0]

        outputs = []
        for i in trange(n_instance):
            v_pos   = self.geometry.v_pos[self.geometry.v_idx[i,0] : self.geometry.v_idx[i,1]]
            t_idx   = self.geometry.t_idx[self.geometry.f_idx[i,0] : self.geometry.f_idx[i,1]] - self.geometry.v_idx[i,0]
            v_nrm   = self.geometry.v_nrm[self.geometry.v_idx[i,0] : self.geometry.v_idx[i,1]]
            v_uv    = self.geometry.v_uv[self.geometry.v_idx[i,0] : self.geometry.v_idx[i,1]]

            v_uv_ori = uv_unpacking(v_uv, i, n_instance)

            mesh = trimesh.Trimesh(
                vertices = v_pos.detach().cpu().numpy(), 
                faces = t_idx.detach().cpu().numpy(), 
                process = False)
        
            if self.cfg.save_normal:
                mesh.vertex_normals = v_nrm.detach().cpu().numpy()

            if self.cfg.save_uv:
                if not self.cfg.save_texture:
                    mesh.visual = trimesh.visual.TextureVisuals(uv=v_uv_ori.detach().cpu().numpy())
                else:
                    if self.cfg.material_strategy == 'extract':
                        material = self.extract_material(v_pos, v_uv, v_uv_ori, t_idx)
                    elif self.cfg.material_strategy == 'sample':
                        material = self.sample_material(i, n_instance)
                    else:
                        raise NotImplementedError
                        
                    mesh.visual = trimesh.visual.TextureVisuals(
                        uv=v_uv_ori.detach().cpu().numpy(), material=material)
        
            rotation_matrix = trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0])
            mesh.apply_transform(rotation_matrix)

            params = {
                "mesh": mesh
            }

            if self.cfg.save_suffix == 'obj':
                save_name = f"{self.cfg.save_name}_{i}/{self.cfg.save_name}_{i}.{self.cfg.save_suffix}"
            elif self.cfg.save_suffix == 'glb':
                save_name = f"{self.cfg.save_name}_{i}.{self.cfg.save_suffix}"
            else:
                raise ValueError

            outputs.append(
                ExporterOutput(
                    save_name=save_name, 
                    save_type="trimesh", 
                    params=params
                )
            )

        return outputs

    def extract_material(self, v_pos, v_uv, v_uv_ori, t_idx):
        material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )

        # clip space transform
        uv_clip = v_uv_ori * 2.0 - 1.0
        uv_clip[...,1] *= -1.0
        # pad to four component coordinate
        uv_clip4 = torch.cat(
            (
                uv_clip,
                torch.zeros_like(uv_clip[..., 0:1]),
                torch.ones_like(uv_clip[..., 0:1]),
            ),
            dim=-1,
        )
        # rasterize
        rast, _ = self.ctx.rasterize_one(
            uv_clip4, t_idx, (self.cfg.texture_size, self.cfg.texture_size)
        )

        # hole_mask = ~(rast[:, :, 3] > 0)

        def uv_padding(image):
            inpaint_image = (image.detach().cpu().numpy() * 255).astype(np.uint8)
            # uv_padding_size = self.cfg.padding_size
            # inpaint_image = (
            #     cv2.inpaint(
            #         (image.detach().cpu().numpy() * 255).astype(np.uint8),
            #         (hole_mask.detach().cpu().numpy() * 255).astype(np.uint8),
            #         uv_padding_size,
            #         cv2.INPAINT_TELEA,
            #     )
            # )
            return inpaint_image

        if self.geometry.cfg.texture_type == 'vertex':
            raise NotImplementedError

        elif self.geometry.cfg.texture_type in ['uv', 'field2d']:
            uv_coord, _ = self.ctx.interpolate_one(v_uv, rast[None, ...], t_idx)
            geo_out = self.geometry.export(points=uv_coord[0])

        elif self.geometry.cfg.texture_type == 'field3d':
            # Interpolate world space position
            gb_pos, _ = self.ctx.interpolate_one(v_pos, rast[None, ...], t_idx)
            geo_out = self.geometry.export(points=gb_pos[0])
        
        else:
            raise NotImplementedError

        mat_out = self.material.export(**geo_out)

        map_Kd, map_Pm, map_Pr, map_Bump = None, None, None, None
        if "albedo" in mat_out:
            map_Kd = uv_padding(mat_out["albedo"])
        else:
            threestudio.warn(
                "save_texture is True but no albedo texture found, using default white texture"
            )
        if "metallic" in mat_out:
            map_Pm = uv_padding(mat_out["metallic"])
        if "roughness" in mat_out:
            map_Pr = uv_padding(mat_out["roughness"])
        if "bump" in mat_out:
            map_Bump = uv_padding(mat_out["bump"])

        if map_Kd is not None:
            material.baseColorTexture = Image.fromarray(map_Kd)
        if (map_Pm is not None) and (map_Pr is not None):
            material.metallicRoughnessTexture = Image.fromarray(
                np.concatenate([map_Pm, map_Pr], axis=-1))
        if map_Bump is not None:
            assert not self.cfg.save_normal
            material.normalTexture = Image.fromarray(map_Bump)

        return material
    
    def sample_material(self, i, n):
        assert self.geometry.cfg.texture_type == 'field2d'

        material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )

        device = self.geometry.v_pos.device

        coords  = torch.linspace(0., 1., steps=self.cfg.texture_size, device=device)
        xx, yy  = torch.meshgrid(coords, coords, indexing='xy')
        uv      = torch.stack((xx, 1-yy), dim=-1)
        uv      = uv_packing(uv, i, n)

        geo_out = self.geometry.export(points=uv)
        mat_out = self.material.export(**geo_out)

        def uv_padding(image):
            inpaint_image = (image.detach().cpu().numpy() * 255).astype(np.uint8)
            # uv_padding_size = self.cfg.padding_size
            # inpaint_image = (
            #     cv2.inpaint(
            #         (image.detach().cpu().numpy() * 255).astype(np.uint8),
            #         (hole_mask.detach().cpu().numpy() * 255).astype(np.uint8),
            #         uv_padding_size,
            #         cv2.INPAINT_TELEA,
            #     )
            # )
            return inpaint_image

        map_Kd, map_Pm, map_Pr, map_Bump = None, None, None, None
        if "albedo" in mat_out:
            map_Kd = uv_padding(mat_out["albedo"])
        else:
            threestudio.warn(
                "save_texture is True but no albedo texture found, using default white texture"
            )
        if "metallic" in mat_out:
            map_Pm = uv_padding(mat_out["metallic"])
        if "roughness" in mat_out:
            map_Pr = uv_padding(mat_out["roughness"])
        if "bump" in mat_out:
            map_Bump = uv_padding(mat_out["bump"])

        if map_Kd is not None:
            material.baseColorTexture = Image.fromarray(map_Kd)
        if (map_Pm is not None) and (map_Pr is not None):
            material.metallicRoughnessTexture = Image.fromarray(
                np.concatenate([map_Pm, map_Pr], axis=-1))
        if map_Bump is not None:
            assert not self.cfg.save_normal
            material.normalTexture = Image.fromarray(map_Bump)

        return material