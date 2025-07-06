from dataclasses import dataclass, field
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDIMScheduler, DDPMScheduler, UniPCMultistepScheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *


HF_ROOT = os.environ.get('HF_ROOT')
HF_PATH = lambda p: os.path.join(HF_ROOT, p) if (HF_ROOT is not None) and (not os.path.exists(p)) else p


@threestudio.register("controlnet-guidance")
class ControlNetGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 100.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        sqrt_anneal: bool = False  # sqrt anneal proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        trainer_max_steps: int = 25000
        use_img_loss: bool = False # image-space SDS proposed in HiFA: https://hifa-team.github.io/HiFA-site/

        guidance_type: str = "sds"
        var_red: bool = True
        weighting_strategy: str = "sds"

        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        view_dependent_prompting: bool = True

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4

        # controlnet parameters
        controlnet_pretrained_model_name_or_path: str = "lllyasviel/sd-controlnet-scribble"
        controlnet_conditioning_scale: float = 1.0
        controlnet_guess_mode: bool = False

        condition_keys: Optional[list] = field(default_factory=lambda: ["layout"])

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Control Net ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            HF_PATH(self.cfg.pretrained_model_name_or_path),
            controlnet=ControlNetModel.from_pretrained(
                HF_PATH(self.cfg.controlnet_pretrained_model_name_or_path), 
                torch_dtype=self.weights_dtype
            ),
            **pipe_kwargs,
        ).to(self.device)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        cleanup()

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        self.controlnet = self.pipe.controlnet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.controlnet.parameters():
            p.requires_grad_(False)

        if self.cfg.token_merging:
            import tomesd

            tomesd.apply_patch(self.unet, **self.cfg.token_merging_params)

        if self.cfg.guidance_type == 'sjc':
            # score jacobian chaining use DDPM
            self.scheduler = DDPMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
                beta_start=0.00085,
                beta_end=0.0120,
                beta_schedule="scaled_linear",
            )
        else:
            self.scheduler = DDIMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
            )
        # Other scheduler options
        # self.scheduler = self.pipe.scheduler
        # self.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."]           = self.scheduler.alphas.to(self.device)
        self.alphas_cumprod: Float[Tensor, "..."]   = self.scheduler.alphas_cumprod.to(self.device)
        
        if self.cfg.guidance_type == 'sjc':
            # score jacobian chaining need mu
            self.us: Float[Tensor, "..."] = torch.sqrt((1 - self.alphas_cumprod) / self.alphas_cumprod)

        self.grad_clip_val: Optional[float] = None

        self.my_cnt = 0

        threestudio.info(f"Loaded Control Net!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_controlnet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        condition: Float[Tensor, "..."],
        conditioning_scale: float = 1.0,
        guess_mode: bool = False,
    ):
        input_dtype = latents.dtype

        if guess_mode:
            # Infer ControlNet only for the conditional batch
            controlnet_latent_model_input = latents.chunk(2)[0]
            t = t.chunk(2)[0]
            controlnet_prompt_embeds = encoder_hidden_states.chunk(2)[0]
            condition = condition.chunk(2)[0]
        else:
            controlnet_latent_model_input = latents
            controlnet_prompt_embeds = encoder_hidden_states

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            controlnet_latent_model_input.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=controlnet_prompt_embeds.to(self.weights_dtype),
            controlnet_cond=condition.to(self.weights_dtype),
            conditioning_scale=conditioning_scale,
            guess_mode=guess_mode,
            return_dict=False
        )

        if guess_mode:
            # Infered ControlNet only for the conditional batch.
            # To apply the output of ControlNet to both the unconditional and conditional batches,
            # add 0 to the unconditional batch to keep it unchanged.
            down_block_res_samples = [torch.cat([d, torch.zeros_like(d)]) for d in down_block_res_samples]
            mid_block_res_sample = torch.cat([mid_block_res_sample, torch.zeros_like(mid_block_res_sample)])

        return [t.to(input_dtype) for t in down_block_res_samples], mid_block_res_sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype

        if down_block_additional_residuals is not None:
            down_block_additional_residuals = [t.to(self.weights_dtype) for t in down_block_additional_residuals]
        if mid_block_additional_residual is not None:
            mid_block_additional_residual = mid_block_additional_residual.to(self.weights_dtype)

        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        # latents = F.interpolate(
        #     latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        # )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        image: Float[Tensor, "B 3 512 512"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        condition: Optional[Float[Tensor, "B 3 512 512"]],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
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
                latent_model_input = torch.cat([latents_noisy] * 4, dim=0)

                if condition is not None:
                    down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                        latent_model_input,
                        torch.cat([t] * 4),
                        encoder_hidden_states=text_embeddings,
                        condition=torch.cat([condition] * 4),
                        conditioning_scale=self.cfg.controlnet_conditioning_scale,
                        guess_mode=self.cfg.controlnet_guess_mode
                    )
                else:
                    down_block_res_samples, mid_block_res_sample = None, None

                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample
                ) # (4B, 3, 64, 64)

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
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)

                if condition is not None:
                    down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                        latent_model_input,
                        torch.cat([t] * 2),
                        encoder_hidden_states=text_embeddings,
                        condition=torch.cat([condition] * 2),
                        conditioning_scale=self.cfg.controlnet_conditioning_scale,
                        guess_mode=self.cfg.controlnet_guess_mode
                    )
                else:
                    down_block_res_samples, mid_block_res_sample = None, None

                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample
                )

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas_cumprod[t] ** 0.5 * (1 - self.alphas_cumprod[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        grad = w * (noise_pred - noise)
        # image-space SDS proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        if self.cfg.use_img_loss:
            alpha = (self.alphas_cumprod[t] ** 0.5).view(-1, 1, 1, 1)
            sigma = ((1 - self.alphas_cumprod[t]) ** 0.5).view(-1, 1, 1, 1)
            latents_denoised = (latents_noisy - sigma * noise_pred) / alpha
            image_denoised = self.decode_latents(latents_denoised)
            grad_img = w * (image - image_denoised) * alpha / sigma
        else:
            grad_img = None

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
            "condition": condition,
        }

        return grad, grad_img, guidance_eval_utils

    def compute_grad_sjc(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        condition: Float[Tensor, "B 3 512 512"],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
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
                latent_model_input = torch.cat([scaled_zs] * 4, dim=0)

                if condition is not None:
                    down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                        latent_model_input,
                        torch.cat([t] * 4),
                        encoder_hidden_states=text_embeddings,
                        condition=torch.cat([condition] * 4),
                        conditioning_scale=self.cfg.controlnet_conditioning_scale,
                        guess_mode=self.cfg.controlnet_guess_mode
                    )
                else:
                    down_block_res_samples, mid_block_res_sample = None, None
                
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample
                )  # (4B, 3, 64, 64)

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
                latent_model_input = torch.cat([scaled_zs] * 2, dim=0)

                if condition is not None:
                    down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                        latent_model_input,
                        torch.cat([t] * 2),
                        encoder_hidden_states=text_embeddings,
                        condition=torch.cat([condition] * 2),
                        conditioning_scale=self.cfg.controlnet_conditioning_scale,
                        guess_mode=self.cfg.controlnet_guess_mode
                    )
                else:
                    down_block_res_samples, mid_block_res_sample = None, None
                
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample
                )

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
            "t_orig": t,
            "latents_noisy": scaled_zs,
            "noise_pred": noise_pred,
            "condition": condition,
        }

        return grad, guidance_eval_utils

    def compute_grad_isd(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        image: Float[Tensor, "B 3 512 512"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        condition: Optional[Float[Tensor, "B 3 512 512"]],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
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
                latent_model_input = torch.cat([latents_noisy] * 4, dim=0)

                if condition is not None:
                    down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                        latent_model_input,
                        torch.cat([t] * 4),
                        encoder_hidden_states=text_embeddings,
                        condition=torch.cat([condition] * 4),
                        conditioning_scale=self.cfg.controlnet_conditioning_scale,
                        guess_mode=self.cfg.controlnet_guess_mode
                    )
                else:
                    down_block_res_samples, mid_block_res_sample = None, None

                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample
                ) # (4B, 3, 64, 64)

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
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)

                if condition is not None:
                    down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                        latent_model_input,
                        torch.cat([t] * 2),
                        encoder_hidden_states=text_embeddings,
                        condition=torch.cat([condition] * 2),
                        conditioning_scale=self.cfg.controlnet_conditioning_scale,
                        guess_mode=self.cfg.controlnet_guess_mode
                    )
                else:
                    down_block_res_samples, mid_block_res_sample = None, None

                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample
                )

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            delta_cls = noise_pred_text - noise_pred_uncond

        # refer to `VividDreamer: Invariant Score Distillation For Hyper-Realistic Text-to-3D Generation` ECCV'24
        # setup scheduler
        num_inference_steps = 50
        self.scheduler.set_timesteps(num_inference_steps)
        
        # get prev latent
        latents_1step = []
        for b in range(batch_size):
            step_output = self.scheduler.step(
                noise_pred_text[b : b + 1], t[b], latents_noisy[b : b + 1]
            )
            latents_1step.append(step_output["prev_sample"])
        latents_1step = torch.cat(latents_1step)

        # get final latent
        # latents_final = self.get_latents_final(latents_noisy, t, text_embeddings, condition)

        t_prev = t - self.num_train_timesteps // num_inference_steps

        neg_guid = neg_guidance_weights if prompt_utils.use_perp_neg else None
        noise_pred_prev = self.get_noise_pred(
            latents_1step, t_prev, text_embeddings, condition, 
            prompt_utils.use_perp_neg, neg_guid, use_guidance=False
        )

        delta_inv = noise_pred_prev - noise_pred_text

        lmb = (((1.-self.alphas[t_prev]) / self.alphas[t_prev]) / ((1.-self.alphas[t]) / self.alphas[t])) ** 0.5

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas_cumprod[t] ** 0.5 * (1 - self.alphas_cumprod[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        # grad = w * (noise_pred - noise)
        grad = w * (lmb * delta_inv + self.cfg.guidance_scale * delta_cls)
        # image-space SDS proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        if self.cfg.use_img_loss:
            alpha = (self.alphas_cumprod[t] ** 0.5).view(-1, 1, 1, 1)
            sigma = ((1 - self.alphas_cumprod[t]) ** 0.5).view(-1, 1, 1, 1)
            latents_denoised = (latents_noisy - sigma * noise_pred) / alpha
            image_denoised = self.decode_latents(latents_denoised)
            grad_img = w * (image - image_denoised) * alpha / sigma

            
            # import numpy as np
            # import cv2
            # out_dir = "E:/SceneGeneration/Code/experiment/layout2gs/outputs/ctrlroom_layout/layout2gs_scene_texture_ctrlroom"

            # # image_denoised = self.decode_latents(latents_final)

            # # image_input = self.decode_latents(latents)
            # # image_save = (image_denoised[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
            # self.my_cnt += 1
            # cv2.imwrite(f"{out_dir}/{self.my_cnt}_final.png", (image_denoised[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8))
            # # cv2.imwrite(f"{out_dir}/{self.my_cnt}_input.png", (image_input[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8))
            # # print(image_save.shape, image_save.dtype, image_save.min(), image_save.max(), ' <- image save')
            # # grad_img = image - image_denoised
        else:
            grad_img = None

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
            "condition": condition,
        }

        return grad, grad_img, guidance_eval_utils

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        condition: Float[Tensor, "B HH WW 3"],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        rgb_BCHW_512 = F.interpolate(
            rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
        )
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        if condition is not None: 
            condition = condition.permute(0, 3, 1, 2)
            condition = F.interpolate(
                condition, (512, 512), mode="bilinear", align_corners=False
            )
            condition = condition.repeat(1,3,1,1) if condition.shape[1] == 1 else condition

        if self.cfg.guidance_type == 'sjc':
            grad, guidance_eval_utils = self.compute_grad_sjc(
                latents, t, prompt_utils, condition, elevation, azimuth, camera_distances
            )
            grad_img = torch.tensor([0.0], dtype=grad.dtype).to(grad.device)
        elif self.cfg.guidance_type == 'sds':
            grad, grad_img, guidance_eval_utils = self.compute_grad_sds(
                latents, rgb_BCHW_512, t, prompt_utils, condition, elevation, azimuth, camera_distances
            )
        elif self.cfg.guidance_type == 'isd':
            grad, grad_img, guidance_eval_utils = self.compute_grad_isd(
                latents, rgb_BCHW_512, t, prompt_utils, condition, elevation, azimuth, camera_distances
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
            "min_step": float(self.min_step),
            "max_step": float(self.max_step),
        }

        if self.cfg.use_img_loss:
            grad_img = torch.nan_to_num(grad_img)
            if self.grad_clip_val is not None:
                grad_img = grad_img.clamp(-self.grad_clip_val, self.grad_clip_val)
            target_img = (rgb_BCHW_512 - grad_img).detach()
            loss_sds_img = (
                0.5 * F.mse_loss(rgb_BCHW_512, target_img, reduction="sum") / batch_size
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

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_noise_pred(
        self,
        latents_noisy,
        t,
        text_embeddings,
        condition,
        use_perp_neg=False,
        neg_guidance_weights=None,
        use_guidance=True,
    ):
        batch_size = latents_noisy.shape[0]

        if use_perp_neg:
            assert use_guidance
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)

            if condition is not None:
                down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                    latent_model_input,
                    torch.cat([t.reshape(1)] * 4).to(self.device),
                    encoder_hidden_states=text_embeddings,
                    condition=torch.cat([condition] * 4),
                    conditioning_scale=self.cfg.controlnet_conditioning_scale,
                    guess_mode=self.cfg.controlnet_guess_mode
                )
            else:
                down_block_res_samples, mid_block_res_sample = None, None

            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 4).to(self.device),
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample
            ) # (4B, 3, 64, 64)

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
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)

            if condition is not None:
                down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                    latent_model_input,
                    torch.cat([t.reshape(1)] * 2).to(self.device),
                    encoder_hidden_states=text_embeddings,
                    condition=torch.cat([condition] * 2),
                    conditioning_scale=self.cfg.controlnet_conditioning_scale,
                    guess_mode=self.cfg.controlnet_guess_mode
                )
            else:
                down_block_res_samples, mid_block_res_sample = None, None

            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample
            )
            
            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            if use_guidance:
                noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            else:
                noise_pred = noise_pred_text

        return noise_pred

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_latents_final(
        self,
        latents_noisy: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        text_embeddings, 
        condition: Optional[Float[Tensor, "B 3 512 512"]],
        num_inference_steps=50,
        use_perp_neg=False,
        neg_guidance_weights=None,
        use_guidance=True,
    ):
        bs = latents_noisy.shape[0]

        self.scheduler.set_timesteps(num_inference_steps)

        large_enough_idxs = \
            self.scheduler.timesteps.expand([bs, -1]).to(t) > t.unsqueeze(-1) # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]

        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_noisy[b:b+1]
            text_emb = (
                text_embeddings[[b, b+bs, b+2*bs, b+3*bs], ...] 
                if use_perp_neg else 
                text_embeddings[[b, b+bs], ...]
            )
            neg_guid = neg_guidance_weights[b:b+1] if use_perp_neg else None
            for t_val in tqdm(self.scheduler.timesteps[i:], leave=False):
                # pred noise
                noise_pred = self.get_noise_pred(
                    latents, t_val, text_emb, condition[b:b+1], use_perp_neg, neg_guid, use_guidance
                )
                # get prev latent
                latents = self.scheduler.step(noise_pred, t_val, latents, eta=1)[
                    "prev_sample"
                ]
            latents_final.append(latents)
        latents_final = torch.cat(latents_final)

        return latents_final

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        text_embeddings,
        condition,
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
        for b in range(bs):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1], eta=1
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1)
        imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1)

        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1]
            text_emb = (
                text_embeddings[
                    [b, b + len(idxs), b + 2 * len(idxs), b + 3 * len(idxs)], ...
                ]
                if use_perp_neg
                else text_embeddings[[b, b + len(idxs)], ...]
            )
            neg_guid = neg_guidance_weights[b : b + 1] if use_perp_neg else None
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                # pred noise
                noise_pred = self.get_noise_pred(
                    latents, t, text_emb, condition, use_perp_neg, neg_guid
                )
                # get prev latent
                latents = self.scheduler.step(noise_pred, t, latents, eta=1)[
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
