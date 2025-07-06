import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.utils.typing import *


@threestudio.register("no-background")
class NoBackground(BaseBackground):
    @dataclass
    class Config(BaseBackground.Config):
        color_strategy: str         = 'fixed_color' # options: 'fixed_color', 'random_flip', 'random_image_level', 'random_pixel_level'
        color_value: tuple          = (1.0, 1.0, 1.0)

        require_depth: bool         = True
        require_normal: bool        = True
        require_semantic: bool      = True

        depth_value: float          = 0.
        normal_color: str           = 'blue'
        
    cfg: Config

    def configure(self) -> None:
        self.register_buffer(
            "color_value", torch.as_tensor(self.cfg.color_value, dtype=torch.float32)
        )
        self.register_buffer(
            "depth_value", torch.as_tensor(self.cfg.depth_value, dtype=torch.float32)
        )

    def forward(
        self, 
        c2w: Float[Tensor, "B 4 4"],
        height: int,
        width: int,
        **kwargs
    ) -> Dict[str, Any]:
        batch_size = c2w.shape[0]

        out = {}
        if self.training:
            if self.cfg.color_strategy == 'fixed_color':
                rgb = torch.ones(batch_size, height, width, 3).to(self.color_value) * self.color_value
            elif self.cfg.color_strategy == 'random_flip':
                rgb = torch.tensor([1, 1, 1] if random.random() > 0.5 else [0, 0, 0], dtype=torch.float32)
                rgb = rgb[None, None, None, :].to(self.color_value).expand(batch_size, height, width, 3)
            elif self.cfg.color_strategy == 'random_image_level':
                rgb = torch.rand(batch_size, 1, 1, 3).to(self.color_value).expand(batch_size, height, width, 3)
            elif self.cfg.color_strategy == 'random_pixel_level':
                rgb = torch.rand(batch_size, height, width, 3).to(self.color_value)
            else:
                raise ValueError
        else:
            rgb = torch.ones(batch_size, height, width, 3).to(self.color_value) * self.color_value
        out['rgb'] = rgb.to(c2w)

        if self.cfg.require_depth:
            depth = torch.ones(batch_size, height, width, 1).to(self.depth_value) * self.depth_value
            out['depth'] = depth.to(c2w)

        if self.cfg.require_normal:
            if self.cfg.normal_color == "blue":
                normal = torch.zeros(batch_size, height, width, 3)
                normal[...,2] = 1.0
            elif self.cfg.normal_color == "black":
                normal = - torch.ones(batch_size, height, width, 3)
            elif self.cfg.normal_color == "white":
                normal = torch.ones(batch_size, height, width, 3)
            else:
                raise NotImplementedError
            out['normal'] = normal.to(c2w)

        if self.cfg.require_semantic:
            semantic = torch.zeros(batch_size, height, width, 3)
            out['semantic'] = semantic.to(c2w)

        return out