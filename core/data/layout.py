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

import json


@dataclass
class LayoutAwareCameraDataModuleConfig(RandomCameraDataModuleConfig):
    camera_path: str            = ""

    elevation_std: float        = 0.
    azimuth_std: float          = 0.

    test_all: bool              = False

    # unused
    # - elevation_range
    # - azimuth_range
    # - camera_distance_range
    # - center_perturb
    # - light_position_perturb
    # - light_distance_range
    # - eval_elevation_deg
    # - eval_camera_distance
    # - light_sample_strategy
    # - batch_uniform_azimuth


class LayoutAwareCameraIterableDataset(RandomCameraIterableDataset):
    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.cfg: LayoutAwareCameraDataModuleConfig = cfg

        # load cameras
        with open(self.cfg.camera_path, 'r') as f:
            cameras = json.load(f)
        
        # positions
        self.camera_positions       = torch.tensor([c['location'] for c in cameras])

        # elevations & azimuths
        self.elevations             = torch.tensor([math.radians(c['rotation'][0]-90) for c in cameras])
        self.azimuths               = torch.tensor([math.radians(c['rotation'][2]+90) for c in cameras])
        self.elevation_std          = torch.tensor([math.radians(self.cfg.elevation_std)] * self.elevations.shape[0])
        self.azimuth_std            = torch.tensor([math.radians(self.cfg.azimuth_std)] * self.azimuths.shape[0])

        # sampling probabilities
        self.camera_probabilities   = torch.tensor([c['probability'] for c in cameras])

    def collate(self, batch) -> Dict[str, Any]:
        selected = torch.multinomial(self.camera_probabilities, self.batch_size, replacement=True)
        
        # sample elevation
        elevation: Float[Tensor, "B"]       = self.elevations[selected]
        elevation_std: Float[Tensor, "B"]   = self.elevation_std[selected]
        elevation += (torch.rand(self.batch_size) * 2.0 - 1.0) * elevation_std
        elevation_deg: Float[Tensor, "B"]   = elevation / math.pi * 180.0

        # sample azimuth
        azimuth: Float[Tensor, "B"]         = self.azimuths[selected]
        azimuth_std: Float[Tensor, "B"]     = self.azimuth_std[selected]
        azimuth += (torch.rand(self.batch_size) * 2.0 - 1.0) * azimuth_std
        azimuth_deg: Float[Tensor, "B"]     = azimuth / math.pi * 180.0

        # sample camera positions
        camera_positions: Float[Tensor, "B 3"]  = self.camera_positions[selected]

        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.batch_size, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: Float[Tensor, "B 3"] = (
            torch.rand(self.batch_size, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        )
        camera_positions = camera_positions + camera_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B 3"] = (
            torch.randn(self.batch_size, 3) * self.cfg.up_perturb
        )
        up = up + up_perturb

        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(self.batch_size) * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        )
        fovy = fovy_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        lookat: Float[Tensor, "B 3"] = torch.stack(
            [
                torch.cos(elevation) * torch.cos(azimuth),
                torch.cos(elevation) * torch.sin(azimuth),
                torch.sin(elevation),
            ],
            dim=-1,
        )
        right: Float[Tensor, "B 3"] = F.normalize(torch.linalg.cross(lookat, up), dim=-1)
        up = F.normalize(torch.linalg.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, normalize=self.cfg.rays_d_normalize
        )

        self.proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.01, 100.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, self.proj_mtx)
        self.fovy = fovy

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_positions.norm(dim=-1),
            "height": self.height,
            "width": self.width,
            "fovy": self.fovy,
            "proj_mtx": self.proj_mtx,
        }


class LayoutAwareCameraDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: LayoutAwareCameraDataModuleConfig = cfg
        self.split = split

        if split == "val":
            self.n_views = self.cfg.n_val_views
        else:
            self.n_views = self.cfg.n_test_views

        # load cameras
        with open(self.cfg.camera_path, 'r') as f:
            cameras = json.load(f)
        
        # probabilities
        probabilities       = [c['probability'] for c in cameras]
        if self.cfg.test_all and split != "val":
            selected = np.arange(len(probabilities))
        else:
            selected = np.argsort(probabilities)[::-1][:self.n_views]
        self.n_views        = len(selected)

        # positions
        camera_positions    = np.array([c['location'] for c in cameras])[selected]
        camera_positions    = torch.tensor(camera_positions, dtype=torch.float32)

        # elevations & azimuths
        elevation_deg       = np.array([c['rotation'][0]-90 for c in cameras])[selected]
        elevation_deg       = torch.tensor(elevation_deg, dtype=torch.float32)
        azimuth_deg         = np.array([c['rotation'][2]+90 for c in cameras])[selected]
        azimuth_deg         = torch.tensor(azimuth_deg, dtype=torch.float32)

        elevation           = elevation_deg * math.pi / 180
        azimuth             = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        lookat: Float[Tensor, "B 3"] = torch.stack(
            [
                torch.cos(elevation) * torch.cos(azimuth),
                torch.cos(elevation) * torch.sin(azimuth),
                torch.sin(elevation),
            ],
            dim=-1,
        )
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.n_views, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_fovy_deg
        )
        fovy = fovy_deg * math.pi / 180

        right: Float[Tensor, "B 3"] = F.normalize(torch.linalg.cross(lookat, up), dim=-1)
        up = F.normalize(torch.linalg.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
        )
        directions_unit_focal = get_ray_directions(
            H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0
        )
        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
            None, :, :, :
        ].repeat(self.n_views, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, normalize=self.cfg.rays_d_normalize
        )
        self.proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.cfg.eval_width / self.cfg.eval_height, 0.01, 100.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, self.proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx
        self.c2w = c2w
        self.camera_positions = camera_positions
        self.elevation, self.azimuth = elevation, azimuth
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distances = camera_positions.norm(dim=-1)
        self.fovy = fovy

    def __len__(self):
        return self.n_views

    def __getitem__(self, index):
        return {
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.c2w[index],
            "camera_positions": self.camera_positions[index],
            "elevation": self.elevation_deg[index],
            "azimuth": self.azimuth_deg[index],
            "camera_distances": self.camera_distances[index],
            "height": self.cfg.eval_height,
            "width": self.cfg.eval_width,
            "fovy": self.fovy[index],
            "proj_mtx": self.proj_mtx[index],
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch


@register("layout-aware-camera-datamodule")
class LayoutAwareCameraDataModule(pl.LightningDataModule):
    cfg: LayoutAwareCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(LayoutAwareCameraDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = LayoutAwareCameraIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = LayoutAwareCameraDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = LayoutAwareCameraDataset(self.cfg, "test")

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