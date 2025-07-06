from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *

import trimesh
from core.utils.layout_utils import load_layout


@threestudio.register("layout-rasterizer")
class LayoutRasterizer(Rasterizer):
    @dataclass
    class Config(VolumeRenderer.Config):
        context_type: str = "cuda"

        layout_path: str = ""

        # unused
        # - radius

    cfg: Config

    def configure(self) -> None:
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())

        # load & parse layout
        bbox_mesh, background_mesh = load_layout(self.cfg.layout_path)
        layout_mesh = trimesh.util.concatenate([bbox_mesh, background_mesh])

        vertices    = torch.as_tensor(layout_mesh.vertices, dtype=torch.float32).contiguous()
        faces       = torch.as_tensor(layout_mesh.faces, dtype=torch.float32).contiguous()
        semantics   = torch.as_tensor(layout_mesh.visual.vertex_colors[...,:3]/255, dtype=torch.float32).contiguous()
        self.register_buffer("v_pos", vertices)
        self.register_buffer("t_idx", faces)
        self.register_buffer("v_sem", semantics)

    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        height: int,
        width: int,
        **kwargs
    ) -> Dict[str, Any]:
        # batch_size = mvp_mtx.shape[0]

        v_pos_clip: Float[Tensor, "B Nv 4"]
        v_pos_clip  = self.ctx.vertex_transform(self.v_pos, mvp_mtx)
        rast, _     = self.ctx.rasterize(v_pos_clip, self.t_idx, (height, width))

        # opacity
        # mask        = rast[..., 3:] > 0
        # mask_aa     = self.ctx.antialias(mask.float(), rast, v_pos_clip, self.t_idx)

        # semantic [0,1]
        semantic, _ = self.ctx.interpolate_one(self.v_sem, rast, self.t_idx)
        semantic_aa = self.ctx.antialias(semantic, rast, v_pos_clip, self.t_idx)
        
        return semantic_aa