import bisect
import math
import random
from dataclasses import dataclass, field

import numpy as np
import pytorch_lightning as pl
import torch
import torch.linalg
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_device
from threestudio.utils.ops import (
    get_full_projection_matrix,
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *

from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)


@dataclass
class GlobalLocalCameraDataModuleConfig:
    global_camera_type: str = ""
    global_camera: dict     = field(default_factory=dict)

    local_camera_type: str  = ""
    local_camera: dict      = field(default_factory=dict)


class GlobalLocalCameraIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg: any) -> None:
        super().__init__()
        self.cfg: GlobalLocalCameraDataModuleConfig = cfg

        global_camera = threestudio.find(self.cfg.global_camera_type)(self.cfg.global_camera)
        global_camera.setup()
        self.global_camera = global_camera.train_dataset
        
        self.local_camera = None
        if self.cfg.local_camera_type not in ["", "none"]:
            local_camera = threestudio.find(self.cfg.local_camera_type)(self.cfg.local_camera)
            local_camera.setup()
            self.local_camera = local_camera.train_dataset

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.global_camera.update_step(epoch, global_step, on_load_weights)
        if self.local_camera is not None:
            self.local_camera.update_step(epoch, global_step, on_load_weights)

    def __iter__(self):
        while True:
            yield {}

    def progressive_view(self, global_step):
        self.global_camera.progressive_view(global_step)
        if self.local_camera is not None:
            self.local_camera.progressive_view(global_step)

    def collate(self, batch) -> Dict[str, Any]:
        ret_dict = self.global_camera.collate(batch)
        if self.local_camera is not None:
            assert 'local' not in ret_dict.keys()
            ret_dict['local'] = self.local_camera.collate(batch)
        return ret_dict


class GlobalLocalCameraDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: GlobalLocalCameraDataModuleConfig = cfg
        assert split in ["val", "test"]

        global_camera = threestudio.find(self.cfg.global_camera_type)(self.cfg.global_camera)
        global_camera.setup()
        self.global_camera = global_camera.val_dataset if split == "val" else global_camera.test_dataset

    def __len__(self):
        return self.global_camera.__len__()
    
    def __getitem__(self, index):
        return self.global_camera.__getitem__(index)
    
    def collate(self, batch):
        return self.global_camera.collate(batch)


@register("global-local-camera-datamodule")
class GlobalLocalCameraDataModule(pl.LightningDataModule):
    cfg: GlobalLocalCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(GlobalLocalCameraDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = GlobalLocalCameraIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = GlobalLocalCameraDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = GlobalLocalCameraDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )