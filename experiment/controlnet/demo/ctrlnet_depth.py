""" conda activate scenegeneration && python ctrlnet_depth.py  """

import torch
import os
import os.path as osp
from huggingface_hub import HfApi
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from transformers import pipeline

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)


HF_ROOT = os.environ.get('HF_ROOT')
HF_PATH = lambda p: p if HF_ROOT is None else osp.join(HF_ROOT, p)


if __name__=='__main__':
    # 0. configuration
    # https://huggingface.co/lllyasviel/control_v11p_sd15_depth/resolve/main/images/input.png
    input_path  = "../../../data/controlnet/images/depth/input_depth.png"
    prompt      = "Stormtrooper's lecture in beautiful lecture hall"
    output_path = "../../../outputs/controlnet/images/depth"
    os.makedirs(output_path, exist_ok=True)

    # 1. load & preprocess image
    image = load_image(input_path)
    
    depth_estimator = pipeline('depth-estimation', model=HF_PATH("Intel/dpt-large"))
    image = depth_estimator(image)['depth']
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(image)

    control_image.save(osp.join(output_path, "control_depth.png"))

    # 2. generation
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        HF_PATH("runwayml/stable-diffusion-v1-5"),
        controlnet=ControlNetModel.from_pretrained(
            HF_PATH("lllyasviel/control_v11p_sd15_depth"), 
            torch_dtype=torch.float16), 
        safety_checker=None,
        torch_dtype=torch.float16
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(0)
    image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]

    image.save(osp.join(output_path, 'image_depth_out.png'))
    print('DONE')