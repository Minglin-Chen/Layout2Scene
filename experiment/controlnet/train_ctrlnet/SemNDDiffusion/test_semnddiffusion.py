""" conda activate controlnet && python test_semnddiffusion.py  """

import torch
import os

from PIL import Image
import numpy as np

from diffusers import (
    StableDiffusionControlNetPipeline, 
    ControlNetModel,
    UniPCMultistepScheduler,
)


HF_ROOT = os.environ.get('HF_ROOT')
HF_PATH = lambda p: os.path.join(HF_ROOT, p) if (HF_ROOT is not None) and (not os.path.exists(p)) else p


import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--pretrained_model_name_or_path", 
    type=str, 
    default="nexuslrf/nd-diffusion", 
    help="Normal Depth Diffusion pretrained model name or path")
parser.add_argument(
    "--controlnet_pretrained_model_name_or_path", 
    type=str, 
    default="../../../../checkpoint/SemNDDiffusion", 
    help="ControlNet pretrained model name or path")
parser.add_argument("--input_path", type=str, default="semantic.png", help="data path")
parser.add_argument("--prompt", type=str,default="", help="text prompt")
parser.add_argument("--negative_prompt", type=str, default="", help="negative text prompt")
parser.add_argument("--n", type=int, default=4, help="number of examples")
parser.add_argument("--guidance_scale", type=float, default=7.5, help="number of examples")
parser.add_argument(
    "--controlnet_conditioning_scale", type=float, default=1.0, help="controlnet conditioning scale")
parser.add_argument("--guess_mode", action="store_true", help="guess mode")
parser.add_argument(
    "--output_path", type=str, default="../../../../outputs/semnddiffusion", help="output data path")
args = parser.parse_args()


if __name__=='__main__':
    # 1. build model
    print("Build model")
    weight_dtype = torch.float32 # torch.float32, torch.float16, or torch.bfloat16 
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        HF_PATH(args.pretrained_model_name_or_path),
        controlnet=ControlNetModel.from_pretrained(
            HF_PATH(args.controlnet_pretrained_model_name_or_path),
            torch_dtype=weight_dtype
        ), 
        safety_checker=None,
        torch_dtype=weight_dtype
    ).to('cuda:0')
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # 2. load data
    print("Load data")
    input_path = os.path.join(args.input_path, 'semantic.png') \
        if os.path.isdir(args.input_path) else args.input_path
    semantic = Image.open(input_path).convert('RGB')
    
    # 3. generation
    generator = torch.manual_seed(42)
    with torch.no_grad():
        images = pipe(
            [args.prompt]*args.n,
            negative_prompt=[args.negative_prompt]*args.n,
            guidance_scale=args.guidance_scale,
            num_inference_steps=50,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            guess_mode=args.guess_mode,
            image=[semantic],
            generator=generator).images

    # 4. output
    os.makedirs(args.output_path, exist_ok=True)
    for i, image in enumerate(images):
        image = np.array(image)
        normal, depth = image[...,:3], image[...,-1]
        normal = Image.fromarray(normal)
        depth = Image.fromarray(depth)
        normal.save(os.path.join(args.output_path, f'normal_out_{i}.png'))
        depth.save(os.path.join(args.output_path, f'depth_out_{i}.png'))

    semantic.save(os.path.join(args.output_path, f'semantic_in.png'))

    print('DONE')