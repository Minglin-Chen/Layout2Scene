""" conda activate scenegeneration && python ctrlnet_multi.py  """

import torch
import os
import os.path as osp

from diffusers.utils import load_image
from PIL import Image
import numpy as np

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

from diffusers.pipelines.controlnet import MultiControlNetModel


HF_ROOT = os.environ.get('HF_ROOT')
HF_PATH = lambda p: p if HF_ROOT is None else osp.join(HF_ROOT, p)


def load_images(semantic_path, normal_path, depth_path):
    semantic    = Image.open(semantic_path).convert('RGB')
    semantic    = torch.tensor(np.array(semantic)).permute(2,0,1).float() / 255.
    semantic    = semantic[None]
    print(semantic.shape, semantic.min(), semantic.max(), semantic.dtype)

    normal      = Image.open(normal_path).convert('RGB')
    normal      = torch.tensor(np.array(normal)).permute(2,0,1).float() / 255.
    normal      = normal[None]
    print(normal.shape, normal.min(), normal.max(), normal.dtype)

    depth       = Image.open(depth_path)
    depth       = np.array(depth, dtype=np.uint16)
    depth       = np.bitwise_or(np.right_shift(depth, 3), np.left_shift(depth, 16 - 3))
    depth       = depth.astype(np.float32) / 1000.0
    depth[depth>8] = 8  # raw depth in range [0,8]
    depth       = 1. / depth  # inverse depth
    depth       = np.nan_to_num(depth)
    depth       = (depth - depth.min()) / (depth.max() - depth.min())
    depth       = torch.tensor(np.stack([depth]*3)).float()
    depth       = depth[None]
    print(depth.shape, depth.min(), depth.max(), depth.dtype)

    return semantic, normal, depth

if __name__=='__main__':
    # 0. configuration
    semantic_path   = "../../../data/controlnet/images/multi_ctrlnet/semantic.png"
    normal_path     = "../../../data/controlnet/images/multi_ctrlnet/normal.png"
    depth_path      = "../../../data/controlnet/images/multi_ctrlnet/depth.png"
    prompt          = 'a room'
    output_path     = "../../../outputs/controlnet/images/multi_ctrlnet"
    os.makedirs(output_path, exist_ok=True)

    # 1. load images
    semantic, normal, depth = load_images(semantic_path, normal_path, depth_path)

    controlnet_semantic = ControlNetModel.from_pretrained(HF_PATH("lllyasviel/control_v11p_sd15_seg"), torch_dtype=torch.float16)
    controlnet_normal   = ControlNetModel.from_pretrained(HF_PATH("lllyasviel/control_v11p_sd15_normalbae"), torch_dtype=torch.float16)
    controlnet_depth    = ControlNetModel.from_pretrained(HF_PATH("lllyasviel/control_v11p_sd15_depth"), torch_dtype=torch.float16)

    multicontrolnet     = MultiControlNetModel([controlnet_semantic, controlnet_normal, controlnet_depth])

    # 2. generation
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        HF_PATH("runwayml/stable-diffusion-v1-5"),
        controlnet=multicontrolnet, 
        safety_checker=None,
        torch_dtype=torch.float16
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(33)
    image = pipe(
        prompt, 
        num_inference_steps=30, 
        generator=generator, 
        image=[semantic, normal, depth]).images[0]

    image.save(osp.join(output_path, 'image_out.png'))
    print('DONE')