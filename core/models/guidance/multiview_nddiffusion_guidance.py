import importlib
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *

from core.utils.helper import HF_PATH


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


# load model
def load_model_from_config(config, ckpt, device, vram_O=True, verbose=False):
    pl_sd = torch.load(ckpt, map_location='cpu')

    if 'global_step' in pl_sd and verbose:
        print(f"Global Step: {pl_sd['global_step']}")

    sd = pl_sd['state_dict']

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("[INFO] missing keys: \n", m)
    if len(u) > 0 and verbose:
        print("[INFO] unexpected keys: \n", u)

    # !!! DO NOT USE IT FOR ND-DIFFUSION
    # # manually load ema and delete it to save GPU memory
    # if model.use_ema:
    #     if verbose:
    #         print("[INFO] loading EMA...")
    #     model.model_ema.copy_to(model.model)
    #     del model.model_ema

    if vram_O:
        # we don't need decoder
        del model.first_stage_model.decoder

    torch.cuda.empty_cache()

    model.eval().to(device)

    return model


@threestudio.register("multiview-nddiffusion-guidance")
class MultiviewNDDiffusionGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_config: str = "nddiffusion/configs/txtcond_mvsd-4-objaverse_finetune_wovae.yaml"
        pretrained_model_name_or_path: str = "nddiffusion/ckpt/nd_mv_ema.ckpt"
        clip_pretrained_model_name_or_path: str = "openai/clip-vit-large-patch14"
        n_view: int = 4
        use_vae_decoder: bool = False
        rotate_z: bool = True
        image_size: int = 256
        vram_O: bool = True
        guidance_scale: float = 50.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = False

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        sqrt_anneal: bool = False  # sqrt anneal proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        trainer_max_steps: int = 25000
        use_img_loss: bool = False  # image-space SDS proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        
        use_sjc: bool = False
        var_red: bool = True
        weighting_strategy: str = "sds"

        view_dependent_prompting: bool = True

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Multiview Normal Depth Diffusion ...")

        self.cfg.pretrained_config                  = HF_PATH(self.cfg.pretrained_config)
        self.cfg.pretrained_model_name_or_path      = HF_PATH(self.cfg.pretrained_model_name_or_path)
        self.cfg.clip_pretrained_model_name_or_path = HF_PATH(self.cfg.clip_pretrained_model_name_or_path)

        self.config = OmegaConf.load(self.cfg.pretrained_config)
        self.config.model.params.cond_stage_config.params.version = \
            self.cfg.clip_pretrained_model_name_or_path

        assert not self.cfg.half_precision_weights
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        self.model = load_model_from_config(
            self.config,
            self.cfg.pretrained_model_name_or_path,
            device=self.device,
            vram_O=self.cfg.vram_O,
        )

        for p in self.model.parameters():
            p.requires_grad_(False)

        if self.cfg.use_sjc:
            # score jacobian chaining use DDPM
            self.scheduler = DDPMScheduler(
                self.config.model.params.timesteps,
                self.config.model.params.linear_start,
                self.config.model.params.linear_end,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
            )
        else:
            self.scheduler = DDIMScheduler(
                self.config.model.params.timesteps,
                self.config.model.params.linear_start,
                self.config.model.params.linear_end,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
            )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )
        if self.cfg.use_sjc:
            # score jacobian chaining need mu
            self.us: Float[Tensor, "..."] = torch.sqrt((1 - self.alphas) / self.alphas)

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded Multiview Normal Depth Diffusion!")

    @torch.amp.autocast('cuda', enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.amp.autocast('cuda', enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(imgs.to(self.weights_dtype))
        )
        return latents.to(input_dtype)

    @torch.amp.autocast('cuda', enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        if self.cfg.use_vae_decoder:
            assert not self.cfg.vram_O
            image = self.model.decode_first_stage(latents)
        else:
            image = F.interpolate(
                latents, (self.cfg.image_size, self.cfg.image_size), mode='bilinear')
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        image: Float[Tensor, "B 3 512 512"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        cameras: Float[Tensor, "B 16"]
    ):
        batch_size = elevation.shape[0]

        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                # pred noise
                x_in = torch.cat([latents_noisy] * 4, dim=0)
                t_in = torch.cat([t] * 4)
                c_in = {
                    "context": text_embeddings,
                    "camera": torch.cat([cameras] * 4),
                    "num_frames": self.cfg.n_view
                }
                with autocast('cuda'):
                    noise_pred = self.model.apply_model(x_in, t_in, c_in) # (4B, 4, 32, 32)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)  # TODO: use torch generator
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                # pred noise
                x_in = torch.cat([latents_noisy] * 2, dim=0)
                t_in = torch.cat([t] * 2)
                c_in = {
                    "context": text_embeddings,
                    "camera": torch.cat([cameras] * 2),
                    "num_frames": self.cfg.n_view
                }
                with autocast('cuda'):
                    noise_pred = self.model.apply_model(x_in, t_in, c_in)

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        grad = w * (noise_pred - noise)
        # image-space SDS proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        if self.cfg.use_img_loss:
            alpha = (self.alphas[t] ** 0.5).view(-1, 1, 1, 1)
            sigma = ((1 - self.alphas[t]) ** 0.5).view(-1, 1, 1, 1)
            latents_denoised = (latents_noisy - sigma * noise_pred) / alpha
            image_denoised = self.decode_latents(latents_denoised)
            grad_img = w * (image - image_denoised) * alpha / sigma
        else:
            grad_img = None

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "cameras": cameras,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
        }

        return grad, grad_img, guidance_eval_utils

    def compute_grad_sjc(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        cameras: Float[Tensor, "B 16"]
    ):
        batch_size = elevation.shape[0]

        sigma = self.us[t]
        sigma = sigma.view(-1, 1, 1, 1)

        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            with torch.no_grad():
                noise = torch.randn_like(latents)
                y = latents
                zs = y + sigma * noise
                scaled_zs = zs / torch.sqrt(1 + sigma**2)
                # pred noise
                x_in = torch.cat([scaled_zs] * 4, dim=0)
                t_in = torch.cat([t] * 4)
                c_in = {
                    "context": text_embeddings,
                    "camera": torch.cat([cameras] * 4),
                    "num_frames": self.cfg.n_view
                }
                with autocast('cuda'):
                    noise_pred = self.model.apply_model(x_in, t_in, c_in) # (4B, 4, 32, 32)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)  # TODO: use torch generator
                y = latents

                zs = y + sigma * noise
                scaled_zs = zs / torch.sqrt(1 + sigma**2)

                # pred noise
                x_in = torch.cat([scaled_zs] * 2, dim=0)
                t_in = torch.cat([t] * 2)
                c_in = {
                    "context": text_embeddings,
                    "camera": torch.cat([cameras] * 2),
                    "num_frames": self.cfg.n_view
                }
                with autocast('cuda'):
                    noise_pred = self.model.apply_model(x_in, t_in, c_in)

                # perform guidance (high scale from paper!)
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

        Ds = zs - sigma * noise_pred

        if self.cfg.var_red:
            grad = -(Ds - y) / sigma
        else:
            grad = -(Ds - zs) / sigma

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "cameras": cameras,
            "t_orig": t,
            "latents_noisy": scaled_zs,
            "noise_pred": noise_pred,
        }

        return grad, guidance_eval_utils

    def get_camera_condition(
        self,
        c2w: Float[Tensor, "B 4 4"],
        distance: Union[float, Float[Tensor, "B"]] = 2.
    ) -> Float[Tensor, "B 16"]:
        camera              = c2w.detach().clone()

        # normalization
        translation         = camera[:, :3, 3]
        translation         = translation / (torch.norm(translation, dim=1, keepdim=True) + 1e-8)
        camera[:, :3, 3]    = translation

        if self.cfg.rotate_z:
            r = R.from_euler("z", -90, degrees=True).as_matrix()
            rotate_mat = torch.eye(4, dtype=camera.dtype, device=camera.device)
            rotate_mat[:3, :3] = torch.from_numpy(r)
            rotate_mat = rotate_mat.unsqueeze(0).repeat(camera.shape[0], 1, 1)
            camera = torch.matmul(rotate_mat, camera)

        if isinstance(distance, torch.Tensor):
            distance = distance.unsqueeze(1)

        camera[:, :3, 3]    = camera[:, :3, 3] * distance
        camera              = camera.flatten(start_dim=1)
        return camera

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        c2w: Float[Tensor, "B 4 4"],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        camera_distances_relative: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        assert batch_size % self.cfg.n_view == 0

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 32 32"]
        rgb_BCHW_256 = F.interpolate(
            rgb_BCHW, (self.cfg.image_size, self.cfg.image_size), mode="bilinear", align_corners=False
        )
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (self.cfg.image_size//8, self.cfg.image_size//8), mode="bilinear", align_corners=False
            )
            # latents = F.adaptive_avg_pool2d(rgb_BCHW, (self.cfg.image_size//8, self.cfg.image_size//8))
        else:
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_256)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size // self.cfg.n_view],
            dtype=torch.long,
            device=self.device,
        ).repeat_interleave(self.cfg.n_view, dim=0)

        # camera conditions
        camera_conditions = self.get_camera_condition(c2w, camera_distances_relative * 2.0)

        if self.cfg.use_sjc:
            grad, guidance_eval_utils = self.compute_grad_sjc(
                latents, t, prompt_utils, elevation, azimuth, camera_distances, camera_conditions
            )
            grad_img = torch.tensor([0.0], dtype=grad.dtype).to(grad.device)
        else:
            grad, grad_img, guidance_eval_utils = self.compute_grad_sds(
                latents,
                rgb_BCHW_256,
                t,
                prompt_utils,
                elevation,
                azimuth,
                camera_distances,
                camera_conditions,
            )

        grad = torch.nan_to_num(grad)

        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        if self.cfg.use_img_loss:
            grad_img = torch.nan_to_num(grad_img)
            if self.grad_clip_val is not None:
                grad_img = grad_img.clamp(-self.grad_clip_val, self.grad_clip_val)
            target_img = (rgb_BCHW_256 - grad_img).detach()
            loss_sds_img = (
                0.5 * F.mse_loss(rgb_BCHW_256, target_img, reduction="sum") / batch_size
            )
            guidance_out["loss_sds_img"] = loss_sds_img

        if guidance_eval:
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out

    @torch.amp.autocast('cuda', enabled=False)
    @torch.no_grad()
    def get_noise_pred(
        self,
        latents_noisy,
        t,
        text_embeddings,
        cameras,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        batch_size = latents_noisy.shape[0]

        if use_perp_neg:
            # pred noise
            x_in = torch.cat([latents_noisy] * 4, dim=0)
            t_in = torch.cat([t.reshape(1)] * x_in.shape[0]).to(self.device)
            c_in = {
                "context": text_embeddings,
                "camera": torch.cat([cameras] * 4),
                "num_frames": self.cfg.n_view
            }
            with autocast('cuda'):
                noise_pred = self.model.apply_model(x_in, t_in, c_in)
                # (4B, 4, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            # pred noise
            x_in = torch.cat([latents_noisy] * 2, dim=0)
            t_in = torch.cat([t.reshape(1)] * x_in.shape[0]).to(self.device)
            c_in = {
                "context": text_embeddings,
                "camera": torch.cat([cameras] * 2),
                "num_frames": self.cfg.n_view
            }
            with autocast('cuda'):
                noise_pred = self.model.apply_model(x_in, t_in, c_in)
            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return noise_pred

    @torch.amp.autocast('cuda', enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        text_embeddings,
        cameras,
        latents_noisy,
        noise_pred,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[
            :bs
        ].unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1)

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(bs // self.cfg.n_view):
            step_output = self.scheduler.step(
                noise_pred[b * self.cfg.n_view : (b + 1) * self.cfg.n_view], 
                t[b * self.cfg.n_view], 
                latents_noisy[b * self.cfg.n_view : (b + 1) * self.cfg.n_view], 
                eta=1
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1)
        imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1)

        latents_final = []
        idxs = idxs.reshape(-1, self.cfg.n_view)[:, 0]
        for b, i in enumerate(idxs):
            latents = latents_1step[b * self.cfg.n_view : (b + 1) * self.cfg.n_view]
            if use_perp_neg:
                text_emb = torch.cat([
                    text_embeddings[b * self.cfg.n_view : (b + 1) * self.cfg.n_view],
                    text_embeddings[(b + len(idxs)) * self.cfg.n_view : (b + len(idxs) + 1) * self.cfg.n_view],
                    text_embeddings[(b + 2 * len(idxs)) * self.cfg.n_view : (b + 2 * len(idxs) + 1) * self.cfg.n_view],
                    text_embeddings[(b + 3 * len(idxs)) * self.cfg.n_view : (b + 3 * len(idxs) + 1) * self.cfg.n_view]
                ])
            else:
                text_emb = torch.cat([
                    text_embeddings[b * self.cfg.n_view : (b + 1) * self.cfg.n_view],
                    text_embeddings[(b + len(idxs)) * self.cfg.n_view : (b + len(idxs) + 1) * self.cfg.n_view],
                ])
            neg_guid = neg_guidance_weights[b * self.cfg.n_view : (b + 1) * self.cfg.n_view] if use_perp_neg else None
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                # pred noise
                noise_pred = self.get_noise_pred(
                    latents, t, text_emb, cameras[b * self.cfg.n_view : (b + 1) * self.cfg.n_view], use_perp_neg, neg_guid
                )
                # get prev latent
                latents = self.scheduler.step(noise_pred, t, latents, eta=0)[
                    "prev_sample"
                ]
            latents_final.append(latents)

        latents_final = torch.cat(latents_final)
        imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)

        return {
            "bs": bs,
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        if self.cfg.sqrt_anneal:
            percentage = (
                float(global_step) / self.cfg.trainer_max_steps
            ) ** 0.5  # progress percentage
            if type(self.cfg.max_step_percent) not in [float, int]:
                max_step_percent = self.cfg.max_step_percent[1]
            else:
                max_step_percent = self.cfg.max_step_percent
            curr_percent = (
                max_step_percent - C(self.cfg.min_step_percent, epoch, global_step)
            ) * (1 - percentage) + C(self.cfg.min_step_percent, epoch, global_step)
            self.set_min_max_steps(
                min_step_percent=curr_percent,
                max_step_percent=curr_percent,
            )
        else:
            self.set_min_max_steps(
                min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
                max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
            )
