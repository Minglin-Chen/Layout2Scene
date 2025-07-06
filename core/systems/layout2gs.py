from dataclasses import dataclass, field
import os
import numpy as np

import torch
import torch.nn.functional as F

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.utils.misc import get_device

from core.utils.helper import *


@threestudio.register("layout2gs-system")
class Layout2GS(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        mode: str                               = "appearance" # options: geometry, appearance

        latent_steps: int                       = 0
        geometry_latent_steps: int              = 0
        local_geometry_latent_steps: int        = 0

        # additional renderer
        layout_renderer_type: str               = ""
        layout_renderer: dict                   = field(default_factory=dict)

        # additional prompt processor
        geometry_prompt_processor_type: str     = ""
        geometry_prompt_processor: dict         = field(default_factory=dict)

        # additional guidances
        geometry_guidance_type: str             = ""
        geometry_guidance: dict                 = field(default_factory=dict)

        ctrl_geometry_guidance_type: str        = ""
        ctrl_geometry_guidance: dict            = field(default_factory=dict)

        ctrl_appearance_guidance_type: str      = ""
        ctrl_appearance_guidance: dict          = field(default_factory=dict)

        # local guidance
        local_geometry_guidance_type: str       = ""
        local_geometry_guidance: dict           = field(default_factory=dict)

        local_appearance_guidance_type: str     = ""
        local_appearance_guidance: dict         = field(default_factory=dict)

        # others
        freq: dict                              = field(default_factory=dict)

    cfg: Config
    
    def configure(self) -> None:
        # create geometry, material, background, renderer
        super().configure()

        self.is_gaussian_geometry = self.cfg.geometry_type == 'gaussian-model'

        self.layout_renderer = None
        if self.cfg.layout_renderer_type not in ["", "none"]:
            self.layout_renderer = threestudio.find(self.cfg.layout_renderer_type)(
                self.cfg.layout_renderer
            )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        if self.layout_renderer is not None: 
            render_out["layout"] = self.layout_renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # create prompt processor (only used in training)
        self.prompt_utils = None
        if self.cfg.prompt_processor_type not in ["", "none"]:
            if hasattr(self.cfg.prompt_processor, 'pretrained_model_name_or_path'):
                self.cfg.prompt_processor.pretrained_model_name_or_path = HF_PATH(
                    self.cfg.prompt_processor.pretrained_model_name_or_path
                )
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
            self.prompt_utils = self.prompt_processor()

        self.geometry_prompt_utils = None
        if self.cfg.geometry_prompt_processor_type not in ["", "none"]:
            if hasattr(self.cfg.geometry_prompt_processor, 'pretrained_model_name_or_path'):
                self.cfg.geometry_prompt_processor.pretrained_model_name_or_path = HF_PATH(
                    self.cfg.geometry_prompt_processor.pretrained_model_name_or_path
                )
            self.geometry_prompt_processor = threestudio.find(self.cfg.geometry_prompt_processor_type)(
                self.cfg.geometry_prompt_processor
            )
            self.geometry_prompt_utils = self.geometry_prompt_processor()

        # create guidance (only used in training)
        self.guidance = None
        if self.cfg.guidance_type not in ["", "none"]:
            self.cfg.guidance.pretrained_model_name_or_path = HF_PATH(
                self.cfg.guidance.pretrained_model_name_or_path
            )
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
            print(f'use guidance ({self.cfg.guidance_type})')
            
        self.geometry_guidance = None
        if self.cfg.geometry_guidance_type not in ["", "none"]:
            self.cfg.geometry_guidance.pretrained_model_name_or_path = HF_PATH(
                self.cfg.geometry_guidance.pretrained_model_name_or_path
            )
            self.geometry_guidance = threestudio.find(
                self.cfg.geometry_guidance_type)(self.cfg.geometry_guidance)
            print(f'use geometry guidance ({self.cfg.geometry_guidance_type})')
            
        self.ctrl_geometry_guidance = None
        if self.cfg.ctrl_geometry_guidance_type not in ["", "none"]:
            self.ctrl_geometry_guidance = threestudio.find(
                self.cfg.ctrl_geometry_guidance_type)(self.cfg.ctrl_geometry_guidance)
            print(f'use controllable geometry guidance ({self.cfg.ctrl_geometry_guidance_type})')

        self.ctrl_appearance_guidance = None
        if self.cfg.ctrl_appearance_guidance_type not in ["", "none"]:
            self.ctrl_appearance_guidance = threestudio.find(
                self.cfg.ctrl_appearance_guidance_type)(self.cfg.ctrl_appearance_guidance)
            print(f'use controllable appearance guidance ({self.cfg.ctrl_appearance_guidance_type})')

        self.local_geometry_guidance = None
        if self.cfg.local_geometry_guidance_type not in ["", "none"]:
            self.local_geometry_guidance = threestudio.find(
                self.cfg.local_geometry_guidance_type)(self.cfg.local_geometry_guidance)
            print(f'use local geometry guidance ({self.cfg.local_geometry_guidance_type})')

        self.local_appearance_guidance = None
        if self.cfg.local_appearance_guidance_type not in ["", "none"]:
            self.local_appearance_guidance = threestudio.find(
                self.cfg.local_appearance_guidance_type)(self.cfg.local_appearance_guidance)
            print(f'use local appearance guidance ({self.cfg.local_appearance_guidance_type})')

        if self.is_gaussian_geometry:
            # inspect the initialized point cloud
            # pointcloud_init_path = self.get_save_path(os.path.join('export', 'pointcloud_init.ply'))
            # self.geometry.export(pointcloud_init_path, fmt='pointcloud')

            # learning rate schedule
            def _get_lr_func(param_name):
                attr = f'{param_name}_lr_schedule'
                if not hasattr(self.cfg.optimizer, attr): return None
                cfg_lr_schedule = getattr(self.cfg.optimizer, attr)
                lr_init         = cfg_lr_schedule.lr_init
                lr_final        = cfg_lr_schedule.lr_final
                lr_delay_steps  = getattr(cfg_lr_schedule, 'lr_delay_steps', 0)
                lr_delay_mult   = getattr(cfg_lr_schedule, 'lr_delay_mult', 1)
                lr_max_steps    = getattr(cfg_lr_schedule, 'lr_max_steps', 1000000)

                def expon_lr_func(step):
                    """ from 3DGS """
                    if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
                        # Disable this parameter
                        return 0.0
                    if lr_delay_steps > 0:
                        # A kind of reverse cosine decay.
                        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
                        )
                    else:
                        delay_rate = 1.0
                    t = np.clip(step / lr_max_steps, 0, 1)
                    log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
                    return delay_rate * log_lerp

                return expon_lr_func

            self._xyz_lr_func           = _get_lr_func('_xyz')
            self._features_dc_lr_func   = _get_lr_func('_features_dc')
            self._features_rest_lr_func = _get_lr_func('_features_rest')
            self._scaling_lr_func       = _get_lr_func('_scaling')
            self._rotation_lr_func      = _get_lr_func('_rotation')
            self._opacity_lr_func       = _get_lr_func('_opacity')
            self.background_lr_func     = _get_lr_func('background')

    def training_step(self, batch):
        # rendering
        local, local_index, batch_local = False, None, None
        if 'local' in batch.keys():
            batch_local = batch.pop('local')
            local       = True
            local_index = np.random.randint(self.geometry.num_instance)
            batch_local.update({'local': local, 'local_index': local_index})
            
        out         = self(batch)
        out_local   = self(batch_local) if local else None
        
        if self.is_gaussian_geometry:
            self.viewspace_points       = out['viewspace_points']
            self.radii                  = out['radii']
            self.visibility_filter      = out['visibility_filter']

            if local:
                self.viewspace_points   += out_local['viewspace_points']
                self.radii              += out_local['radii']
                self.visibility_filter  += out_local['visibility_filter']

        # prompt
        prompt_utils = self.prompt_utils \
            if not isinstance(self.prompt_utils, dict) else self.prompt_utils['global']
        
        geometry_prompt_utils = self.geometry_prompt_utils \
            if not isinstance(self.geometry_prompt_utils, dict) else self.geometry_prompt_utils['global']

        local_prompt_utils = self.prompt_utils['local'][local_index] \
            if (self.prompt_utils is not None) and local else None
        local_geometry_prompt_utils = self.geometry_prompt_utils['local'][local_index] \
            if (self.geometry_prompt_utils is not None) and local else None

        # loss
        loss = 0.0

        guidance_eval = (self.cfg.freq.guidance_eval > 0) and (self.true_global_step % self.cfg.freq.guidance_eval == 0)

        loss = self.__guidance_step(batch, out, prompt_utils, loss, guidance_eval)
        loss = self.__geometry_guidance_step(batch, out, geometry_prompt_utils, loss, guidance_eval)
        
        loss = self.__ctrl_geometry_guidance_step(batch, out, geometry_prompt_utils, loss, guidance_eval)
        loss = self.__ctrl_appearance_guidance_step(batch, out, prompt_utils, loss, guidance_eval)

        loss = self.__local_geometry_guidance_step(batch_local, out_local, local_geometry_prompt_utils, loss, guidance_eval)

        # regularization
        if hasattr(self.cfg.loss, 'lambda_sparsity') and self.C(self.cfg.loss.lambda_sparsity) > 0:
            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        if hasattr(self.cfg.loss, 'lambda_opaque') and self.C(self.cfg.loss.lambda_opaque) > 0:
            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        if self.is_gaussian_geometry:
            if hasattr(self.cfg.loss, 'lambda_orient') and self.C(self.cfg.loss.lambda_orient) > 0:
                loss_orient = torch.relu(0.5 - out["comp_normal"][...,2]).sum()
                self.log("train/loss_orient", loss_orient)
                loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            if hasattr(self.cfg.loss, 'lambda_normal_consistency') and self.C(self.cfg.loss.lambda_normal_consistency) > 0:
                comp_normal, surf_normal = out["comp_normal"] * 2. - 1., out["surf_normal"] * 2. - 1.
                loss_normal_consistency = 1 - (comp_normal * surf_normal.detach()).mean()
                self.log("train/loss_normal_consistency", loss_normal_consistency)
                loss += loss_normal_consistency * self.C(self.cfg.loss.lambda_normal_consistency)

            if hasattr(self.cfg.loss, 'lambda_normal_smoothness') and self.C(self.cfg.loss.lambda_normal_smoothness) > 0:
                normal = out["comp_normal"] * 2. - 1.
                dx = normal[:, 1:, :, :] - normal[:, :-1, :, :]  # [B, H-1, W, 3]
                dy = normal[:, :, 1:, :] - normal[:, :, :-1, :]  # [B, H, W-1, 3]
                loss_normal_smoothness = dx.pow(2).mean() + dy.pow(2).mean()
                self.log("train/loss_normal_smoothness", loss_normal_smoothness)
                loss += loss_normal_smoothness * self.C(self.cfg.loss.lambda_normal_smoothness)

            if hasattr(self.cfg.loss, 'lambda_depth_distortion') and self.C(self.cfg.loss.lambda_depth_distortion) > 0:
                loss_depth_distortion = out["depth_distortion"].mean()
                self.log("train/loss_depth_distortion", loss_depth_distortion)
                loss += loss_depth_distortion * self.C(self.cfg.loss.lambda_depth_distortion)

            if hasattr(self.cfg.loss, 'lambda_scaling') and self.C(self.cfg.loss.lambda_scaling) > 0:
                gaussian_scaling = self.geometry.get_scaling
                loss_scaling = gaussian_scaling.abs().sum()
                self.log("train/loss_scaling", loss_scaling)
                loss += loss_scaling * self.C(self.cfg.loss.lambda_scaling)

            if hasattr(self.cfg.loss, 'lambda_out_of_box') and self.C(self.cfg.loss.lambda_out_of_box) > 0:
                loss_out_of_box = 0
                for i, (loc, rot, sz) in enumerate(zip(\
                    self.geometry.instance_location, self.geometry.instance_rotation, self.geometry.instance_size)):
                    instance_mask   = self.geometry.get_instance[...,0] == i
                    instance_xyz    = self.geometry.get_xyz[instance_mask]
                    if self.geometry.cfg.is_local_space:
                        instance_xyz = instance_xyz / (sz / sz.max()) / self.geometry.cfg.local_scale
                    else:
                        instance_xyz = translate_rotate_scale(\
                            instance_xyz, self.geometry.cfg.local_scale / sz, torch.deg2rad(-rot), - loc)
                        
                    loss_out_of_box += F.relu(instance_xyz.abs() - 0.5).sum()
                loss += loss_out_of_box * self.C(self.cfg.loss.lambda_out_of_box)

            require_tv_feature  = hasattr(self.cfg.loss, 'lambda_tv_feature') and self.C(self.cfg.loss.lambda_tv_feature) > 0
            require_tv_rotation = hasattr(self.cfg.loss, 'lambda_tv_rotation') and self.C(self.cfg.loss.lambda_tv_rotation) > 0
            if require_tv_feature or require_tv_rotation:
                loss_tv_feature, loss_tv_rotation = self.geometry.total_variation(require_tv_feature, require_tv_rotation)
                if loss_tv_feature is not None:
                    self.log("train/loss_tv_feature", loss_tv_feature)
                    loss += loss_tv_feature * self.C(self.cfg.loss.lambda_tv_feature)
                if loss_tv_rotation is not None:
                    self.log("train/loss_tv_rotation", loss_tv_rotation)
                    loss += loss_tv_rotation * self.C(self.cfg.loss.lambda_tv_rotation)

        # record
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def __guidance_step(self, batch, out, prompt_utils, loss, guidance_eval):
        if self.guidance is None: return loss

        if self.cfg.mode == 'appearance':
            guidance_out = self.guidance(
                out['comp_rgb'], prompt_utils, **batch, rgb_as_latents=False, guidance_eval=guidance_eval
            )

            for name, value in guidance_out.items():
                if name.startswith("loss_"):
                    self.log(f"train/guidance/{name}", value.item())
                    loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

            if guidance_eval:
                self.guidance_evaluation_save(
                    out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]], 
                    guidance_out["eval"], "guidance"
                )

        elif self.cfg.mode == 'geometry':
            if self.true_global_step < self.cfg.latent_steps:
                guidance_in = torch.cat((
                    out['comp_normal'] * 2.0 - 1.0, out['opacity']), dim=-1)
                rgb_as_latents = True
            else:
                guidance_in = out['comp_normal']
                rgb_as_latents = False

            guidance_out = self.guidance(
                guidance_in, prompt_utils, **batch, rgb_as_latents=rgb_as_latents, guidance_eval=guidance_eval
            )

            for name, value in guidance_out.items():
                if name.startswith("loss_"):
                    self.log(f"train/guidance/{name}", value.item())
                    loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

            if guidance_eval:
                self.guidance_evaluation_save(
                    out["comp_normal"].detach()[: guidance_out["eval"]["bs"]], 
                    guidance_out["eval"], "guidance/normal"
                )
                self.guidance_evaluation_save(
                    out["comp_depth"].detach()[: guidance_out["eval"]["bs"]], 
                    guidance_out["eval"], "guidance/depth"
                )

        else:
            raise NotImplementedError

        return loss

    def __geometry_guidance_step(self, batch, out, prompt_utils, loss, guidance_eval):
        if self.geometry_guidance is None: return loss
        assert self.cfg.mode == 'geometry'

        guidance_in = torch.cat((out['comp_normal'], out['comp_depth']), dim=-1)
        if self.true_global_step < self.cfg.geometry_latent_steps:
            guidance_in     = guidance_in * 2.0 - 1.0
            rgb_as_latents  = True
        else:
            rgb_as_latents  = False

        guidance_out = self.geometry_guidance(
            guidance_in, prompt_utils, **batch, rgb_as_latents=rgb_as_latents, guidance_eval=guidance_eval
        )
 
        for name, value in guidance_out.items():
            if name.startswith("loss_"):
                self.log(f"train/geometry_guidance/{name}", value.item())
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_geometry_")])

        if guidance_eval:
            self.guidance_evaluation_save(
                out["comp_normal"].detach()[: guidance_out["eval"]["bs"]], 
                guidance_out["eval"], "geometry_guidance/normal"
            )
            self.guidance_evaluation_save(
                out["comp_depth"].detach()[: guidance_out["eval"]["bs"]], 
                guidance_out["eval"], "geometry_guidance/depth"
            )

        return loss

    def __collect_inputs(self, out, keys, concat=True): 
        collect_inputs = [out[k] for k in keys]
        return torch.cat(collect_inputs, dim=-1) if concat else collect_inputs

    def __ctrl_geometry_guidance_step(self, batch, out, prompt_utils, loss, guidance_eval):
        if self.ctrl_geometry_guidance is None: return loss
        assert self.cfg.mode == 'geometry'

        guidance_in = torch.cat((out['comp_normal'], out['comp_depth']), dim=-1)
        if self.true_global_step < self.cfg.geometry_latent_steps:
            guidance_in     = guidance_in * 2.0 - 1.0
            rgb_as_latents  = True
        else:
            rgb_as_latents  = False

        guidance_out = self.ctrl_geometry_guidance(
            guidance_in, prompt_utils, 
            self.__collect_inputs(out, self.ctrl_geometry_guidance.cfg.condition_keys, False), 
            **batch, rgb_as_latents=rgb_as_latents, guidance_eval=guidance_eval
        )

        for name, value in guidance_out.items():
            if name.startswith("loss_"):
                self.log(f"train/ctrl_geometry_guidance/{name}", value.item())
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_ctrl_geometry_")])

        if guidance_eval:
            self.guidance_evaluation_save(
                out["comp_normal"].detach()[: guidance_out["eval"]["bs"]], 
                guidance_out["eval"], "ctrl_geometry_guidance/normal"
            )
            self.guidance_evaluation_save(
                out["comp_depth"].detach()[: guidance_out["eval"]["bs"]], 
                guidance_out["eval"], "ctrl_geometry_guidance/depth"
            )

        return loss

    def __ctrl_appearance_guidance_step(self, batch, out, prompt_utils, loss, guidance_eval):
        if self.ctrl_appearance_guidance is None: return loss
        assert self.cfg.mode == 'appearance'

        guidance_out = self.ctrl_appearance_guidance(
            out["comp_rgb"], prompt_utils, 
            self.__collect_inputs(out, self.ctrl_appearance_guidance.cfg.condition_keys, False), 
            **batch, rgb_as_latents=False, guidance_eval=guidance_eval
        )

        for name, value in guidance_out.items():
            if name.startswith("loss_"):
                self.log(f"train/ctrl_appearance_guidance/{name}", value.item())
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_ctrl_appearance_")])

        if guidance_eval:
            self.guidance_evaluation_save(
                out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]], 
                guidance_out["eval"], "ctrl_appearance_guidance"
            )

        return loss

    def __local_geometry_guidance_step(self, batch, out, prompt_utils, loss, guidance_eval):
        if self.local_geometry_guidance is None: return loss
        assert self.cfg.mode == 'geometry'

        guidance_in = torch.cat((out['comp_normal'], out['comp_depth']), dim=-1)
        if self.true_global_step < self.cfg.local_geometry_latent_steps:
            guidance_in     = guidance_in * 2.0 - 1.0
            rgb_as_latents  = True
        else:
            rgb_as_latents  = False

        guidance_out = self.local_geometry_guidance(
            guidance_in, prompt_utils, **batch, 
            rgb_as_latents=rgb_as_latents, guidance_eval=guidance_eval
        )
 
        for name, value in guidance_out.items():
            if name.startswith("loss_"):
                self.log(f"train/local_geometry_guidance/{name}", value.item())
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_local_geometry_")])

        if guidance_eval:
            self.guidance_evaluation_save(
                out["comp_normal"].detach()[: guidance_out["eval"]["bs"]], 
                guidance_out["eval"], "local_geometry_guidance/normal"
            )
            self.guidance_evaluation_save(
                out["comp_depth"].detach()[: guidance_out["eval"]["bs"]], 
                guidance_out["eval"], "local_geometry_guidance/depth"
            )

        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png", 
            (
                [{"type": "rgb", "img": out["comp_rgb"][0], "kwargs": {"data_format": "HWC"}}]
                if 'comp_rgb' in out.keys() else []
            ) 
            + 
            (
                [{"type": "rgb", "img": out["comp_semantic"][0], "kwargs": {"data_format": "HWC"}}]
                if 'comp_semantic' in out.keys() else []
            ) 
            + 
            (
                [{"type": "rgb", "img": out["comp_normal"][0], "kwargs": {"data_format": "HWC"}}]
                if 'comp_normal' in out.keys() else []
            )
            + 
            (
                [{"type": "grayscale", "img": out["comp_depth"][0, :, :, 0], "kwargs": {"cmap": "jet", "data_range": (0, 1)}}]
                if 'comp_depth' in out.keys() else []
            ) 
            + 
            (
                [{"type": "grayscale", "img": out["opacity"][0, :, :, 0], "kwargs": {"cmap": None, "data_range": (0, 1)}}]
                if 'opacity' in out.keys() else []
            )
            ,
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [{"type": "rgb", "img": out["comp_rgb"][0], "kwargs": {"data_format": "HWC"}}]
                if 'comp_rgb' in out.keys() else []
            )
            + 
            (
                [{"type": "rgb", "img": out["comp_semantic"][0], "kwargs": {"data_format": "HWC"}}]
                if 'comp_semantic' in out.keys() else []
            ) 
            +
            (
                [{"type": "rgb", "img": out["comp_normal"][0], "kwargs": {"data_format": "HWC"}}]
                if 'comp_normal' in out.keys() else []
            )
            + 
            (
                [{"type": "grayscale", "img": out["comp_depth"][0, :, :, 0], "kwargs": {"cmap": "jet", "data_range": (0, 1)}}]
                if 'comp_depth' in out.keys() else []
            ) 
            + 
            (
                [{"type": "grayscale", "img": out["opacity"][0, :, :, 0], "kwargs": {"cmap": None, "data_range": (0, 1)}}]
                if 'opacity' in out.keys() else []
            )
            + 
            (
                [{"type": "rgb", "img": out["layout"][0], "kwargs": {"data_format": "HWC"}}]
                if 'layout' in out.keys() else []
            )
            ,
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self) -> None:
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )

        if self.is_gaussian_geometry:
            gaussians_path  = self.get_save_path(
                os.path.join('export', f'gaussians-it{self.true_global_step}-test'))
            mesh_path = self.get_save_path(
                os.path.join('export', f'mesh-it{self.true_global_step}-test'))
            
            self.geometry.export(gaussians_path, fmt='gaussians')
            self.geometry.export(mesh_path, fmt='mesh')

        else:
            if self.cfg.exporter_type not in ["", "none"]:
                exporter: Exporter = threestudio.find(self.cfg.exporter_type)(
                    self.cfg.exporter,
                    geometry=self.geometry,
                    material=self.material,
                    background=self.background,
                )

                exporter_output: List[ExporterOutput] = exporter()
                for out in exporter_output:
                    save_func_name = f"save_{out.save_type}"
                    if not hasattr(self, save_func_name):
                        raise ValueError(f"{save_func_name} not supported by the SaverMixin")
                    save_func = getattr(self, save_func_name)
                    save_func(f"it{self.true_global_step}-export/{out.save_name}", **out.params)

    def guidance_evaluation_save(self, comp_in, guidance_eval_out, save_dir):
        assert comp_in.ndim == 4
        is_rgb = comp_in.shape[-1] == 3
        B, size = comp_in.shape[:2]
        resize = lambda x: F.interpolate(
            x.permute(0, 3, 1, 2), (size, size), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        filename = os.path.join(save_dir, f"it{self.true_global_step}-train.png")

        def merge12(x):
            return x.reshape(-1, *x.shape[2:])
        
        self.save_image_grid(
            filename,
            [
                {
                    "type": "rgb" if is_rgb else \
                        "grayscale",
                    "img": merge12(comp_in)[...,:3] if is_rgb else \
                        merge12(comp_in)[...,-1],
                    "kwargs": {"data_format": "HWC"} if is_rgb else \
                        {"cmap": "jet", "data_range": (0, 1)},
                }
            ]
            + (
                [
                    {
                        "type": "rgb" if is_rgb else \
                            "grayscale",
                        "img": merge12(resize(guidance_eval_out["imgs_noisy"]))[...,:3] if is_rgb else \
                            merge12(resize(guidance_eval_out["imgs_noisy"]))[...,-1],
                        "kwargs": {"data_format": "HWC"} if is_rgb else \
                            {"cmap": "jet", "data_range": (0, 1)},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb" if is_rgb else \
                            "grayscale",
                        "img": merge12(resize(guidance_eval_out["imgs_1step"]))[...,:3] if is_rgb else \
                            merge12(resize(guidance_eval_out["imgs_1step"]))[...,-1],
                        "kwargs": {"data_format": "HWC"} if is_rgb else \
                            {"cmap": "jet", "data_range": (0, 1)},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb" if is_rgb else \
                            "grayscale",
                        "img": merge12(resize(guidance_eval_out["imgs_1orig"]))[...,:3] if is_rgb else \
                            merge12(resize(guidance_eval_out["imgs_1orig"]))[...,-1],
                        "kwargs": {"data_format": "HWC"} if is_rgb else \
                            {"cmap": "jet", "data_range": (0, 1)},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb" if is_rgb else \
                            "grayscale",
                        "img": merge12(resize(guidance_eval_out["imgs_final"]))[...,:3] if is_rgb else \
                            merge12(resize(guidance_eval_out["imgs_final"]))[...,-1],
                        "kwargs": {"data_format": "HWC"} if is_rgb else \
                            {"cmap": "jet", "data_range": (0, 1)},
                    }
                ]
            ),
            name="train_step",
            step=self.true_global_step,
            texts=guidance_eval_out["texts"],
        )

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        if self.is_gaussian_geometry:
            # update learning rate
            if self._xyz_lr_func is not None:
                for group in optimizer.param_groups:
                    if group["name"].split('.')[-1] != '_xyz': continue
                    group['lr'] = self._xyz_lr_func(self.true_global_step)
                    break

            if self._features_dc_lr_func is not None:
                for group in optimizer.param_groups:
                    if group["name"].split('.')[-1] != '_features_dc': continue
                    group['lr'] = self._features_dc_lr_func(self.true_global_step)
                    break
            
            if self._features_rest_lr_func is not None:
                for group in optimizer.param_groups:
                    if group["name"].split('.')[-1] != '_features_rest': continue
                    group['lr'] = self._features_rest_lr_func(self.true_global_step)
                    break
            
            if self._scaling_lr_func is not None:
                for group in optimizer.param_groups:
                    if group["name"].split('.')[-1] != '_scaling': continue
                    group['lr'] = self._scaling_lr_func(self.true_global_step)
                    break

            if self._rotation_lr_func is not None:
                for group in optimizer.param_groups:
                    if group["name"].split('.')[-1] != '_rotation': continue
                    group['lr'] = self._rotation_lr_func(self.true_global_step)
                    break
            
            if self._opacity_lr_func is not None:
                for group in optimizer.param_groups:
                    if group["name"].split('.')[-1] != '_opacity': continue
                    group['lr'] = self._opacity_lr_func(self.true_global_step)
                    break

            if self.background_lr_func is not None:
                for group in optimizer.param_groups:
                    if group["name"].split('.')[-1] != 'background': continue
                    group['lr'] = self.background_lr_func(self.true_global_step)
                    break

        # update parameters
        optimizer.step(closure=optimizer_closure)

        if self.is_gaussian_geometry:
            # update geometry
            self.geometry.update(
                self.true_global_step, 
                optimizer, 
                self.viewspace_points, 
                self.radii, 
                self.visibility_filter)
            
    # SaverMixin extension (used in Exporter)
    def save_trimesh(self, filename, mesh) -> List[str]:
        save_paths: List[str] = []

        save_path = self.get_save_path(filename)
        mesh.export(save_path)

        save_paths.append(save_path)
        return save_paths