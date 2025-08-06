""" conda activate scenegeneration && python ctrlnet_normalbae.py  """

import torch
import os
import os.path as osp
from huggingface_hub import HfApi
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from controlnet_aux import NormalBaeDetector

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)


HF_ROOT = os.environ.get('HF_ROOT')
HF_PATH = lambda p: p if HF_ROOT is None else osp.join(HF_ROOT, p)


if __name__=='__main__':
    # 0. configuration
    # https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae/resolve/main/images/input.png
    input_path  = "../../../data/controlnet/images/normalbae/input_normalbae.png"
    prompt      = "A head full of roses"
    output_path = "../../../outputs/controlnet/images/normalbae"
    os.makedirs(output_path, exist_ok=True)

    # 1. load & preprocess image
    image = load_image(input_path)

    processor = NormalBaeDetector.from_pretrained(HF_PATH("lllyasviel/Annotators"))
    control_image = processor(image)

    control_image.save(osp.join(output_path, "control_normalbae.png"))

    # 2. generation
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        HF_PATH("runwayml/stable-diffusion-v1-5"),
        controlnet=ControlNetModel.from_pretrained(
            HF_PATH("lllyasviel/control_v11p_sd15_normalbae"),
            torch_dtype=torch.float16),
        torch_dtype=torch.float16
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(33)
    image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]

    image.save(osp.join(output_path, 'image_normalbae_out.png'))
    print('DONE')