import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.utils.typing import *


@threestudio.register("global-local-background")
class GlobalLocalBackground(BaseBackground):
    @dataclass
    class Config(BaseBackground.Config):
        global_background_type: str = "no-background"
        global_background: dict     = field(default_factory=dict)

        local_background_type: str  = "no-background"
        local_background: dict      = field(default_factory=dict)

    cfg: Config

    def configure(self) -> None:
        self.global_background = threestudio.find(self.cfg.global_background_type)(
            self.cfg.global_background
        )
        self.local_background = threestudio.find(self.cfg.local_background_type)(
            self.cfg.local_background
        )

    def forward(self, local: bool = False, **kwargs):
        return self.local_background(**kwargs) if local else self.global_background(**kwargs)