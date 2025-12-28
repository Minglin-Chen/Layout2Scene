import os
import os.path as osp

import torch
from diffusers import StableDiffusionPipeline


HF_ROOT = os.environ.get('HF_ROOT')
HF_PATH = lambda p: osp.join(HF_ROOT, p) if (HF_ROOT is not None) and (not osp.exists(p)) else p


if __name__=='__main__':

    TEXT_AND_NUM = [
        ['a modern style bedroom', 586],
        ['a Cartoon style bedroom', 1031],
        ['a Bohemian style bedroom', 220],
        ['a Scandinavian style bedroom', 444],
        ['a industrial style bedroom', 742],
        ['a Bohemian style bathroom', 100],
        ['a Scandinavian style office', 100],
        ['a industrial style dining room', 100],
        ['a Chinese style bedroom', 100],
        ['a modern style dining room', 100],
        ['a Midcentury style bedroom', 100],
        ['a Cartoon style living room', 100],
        ['a Bohemian style living room', 100],
        ['a Chinese style living room', 211],
        ['a modern style living room', 800],
        ['a Midcentury style living room', 436],
        ['a Scandinavian style bedroom', 25],
        ['a industrial style dining room', 25],
        ['a Chinese style garage', 25],
        ['a modern style living room', 25],
        ['a Cartoon style bedroom', 724]
    ]
    negative_prompt = 'deformed, extra digit, fewer digits, cropped, worst quality, low quality, smoke, shading, lighting, lumination, shadow, text in image, watermarks'
    
    device = 'cuda:0'
    pipe = StableDiffusionPipeline.from_pretrained(
				HF_PATH("stabilityai/stable-diffusion-2-1-base"), 
				revision="fp16", torch_dtype=torch.float16)
    pipe.to(device)

    output_root = '../../outputs/results/sd21base'
    total = 0
    for ith_scene, (text, num) in enumerate(TEXT_AND_NUM):
        total += num
        output_dir =  osp.join(output_root, f'{ith_scene}_'+text.replace(' ', '_'))
        os.makedirs(output_dir, exist_ok=True)

        for n in range(num):
            print(f'{ith_scene} {n} | {num}')
            image = pipe(
                text, 
                negative_prompt=negative_prompt, 
                num_inference_steps=50).images[0]
            image.save(osp.join(output_dir, f'{n}.jpg'))
    print(total, ' <-total ')