import os
import os.path as osp
import random
from functools import partial
import numpy as np
from PIL import Image

import torch

from datasets import load_dataset
from torchvision import transforms as T
from transformers import AutoTokenizer

HF_ROOT = os.environ.get('HF_ROOT')
HF_PATH = lambda p: p if HF_ROOT is None else osp.join(HF_ROOT, p)


def make_dataset(train_data_dir, resolution, proportion_empty_prompts, tokenizer):
    # dataset
    dataset = load_dataset('json', data_files={
            'train': osp.join(train_data_dir, 'train.jsonl'),
            'test': osp.join(train_data_dir, 'test.jsonl')
        }
    )

    prompt_column = 'type'

    # preprocessing
    def tokenize_prompts(examples, is_train=True):
        prompts = []
        for prompt in examples[prompt_column]:
            if random.random() < proportion_empty_prompts:
                prompts.append("")
            elif isinstance(prompt, str):
                prompts.append(prompt)
            elif isinstance(prompt, (list, np.ndarray)):
                # take a random caption if there are multiple
                prompts.append(random.choice(prompt) if is_train else prompts[0])
            else:
                raise ValueError(
                    f"Prompt column `{prompt_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            prompts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    rgb_transforms = T.Compose([
        T.Resize(resolution, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(resolution),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])
    semantic_transforms = T.Compose([
        T.Resize(resolution, interpolation=T.InterpolationMode.NEAREST_EXACT),
        T.CenterCrop(resolution),
        T.ToTensor(),
    ])
    normal_transforms = T.Compose([
        T.Resize(resolution, interpolation=T.InterpolationMode.NEAREST_EXACT),
        T.CenterCrop(resolution),
        T.ToTensor(),
    ])
    depth_transforms = T.Compose([
        T.Resize(resolution, interpolation=T.InterpolationMode.NEAREST_EXACT),
        T.CenterCrop(resolution),
    ])

    def preprocess_train(examples):
        # RGB image
        rgb_pixel_values = []
        for rgb_path in examples['rgb']:
            rgb = Image.open(osp.join(train_data_dir, rgb_path)).convert('RGB')
            rgb = rgb_transforms(rgb)
            rgb_pixel_values.append(rgb)
        examples['rgb_pixel_values'] = rgb_pixel_values

        # Semantic image
        semantic_pixel_values = []
        for semantic_path in examples['semantic']:
            semantic = Image.open(osp.join(train_data_dir, semantic_path)).convert('RGB')
            semantic = semantic_transforms(semantic)
            semantic_pixel_values.append(semantic)
        examples['semantic_pixel_values'] = semantic_pixel_values

        # Normal image
        normal_pixel_values = []
        for normal_path in examples['normal']:
            normal = Image.open(osp.join(train_data_dir, normal_path)).convert('RGB')
            normal = normal_transforms(normal)
            normal_pixel_values.append(normal)
        examples['normal_pixel_values'] = normal_pixel_values

        # Depth image
        depth_pixel_values = []
        for depth_path in examples['depth']:
            depth = Image.open(osp.join(train_data_dir, depth_path))
            print(depth.mode)
            depth = np.array(depth)
            print(f'- {depth.min()} {depth.max()}, {depth.dtype}')
            depth = np.bitwise_or(np.right_shift(depth, 3), np.left_shift(depth, 16 - 3))
            print(f'- {depth.min()} {depth.max()}')
            depth = depth.astype(np.float32) / 1000.0
            print(f'- {depth.min()} {depth.max()}')
            depth[depth>8] = 8  # raw depth in range [0,8]
            depth = 1. / depth  # inverse depth
            depth = torch.tensor([depth, depth, depth])
            depth = torch.nan_to_num(depth)
            depth = depth_transforms(depth)
            depth_pixel_values.append(depth)
        examples['depth_pixel_values'] = depth_pixel_values

        # Tokenization
        examples['input_ids'] = tokenize_prompts(examples)

        return examples

    max_train_samples = 10
    # dataset["train"] = dataset["train"].shuffle().select(range(max_train_samples))
    dataset["train"] = dataset["train"].shuffle().select(range(1))
    train_dataset = dataset['train'].with_transform(preprocess_train)

    return train_dataset


if __name__=='__main__':
    # configuration
    train_data_dir = '/home/mlchen/data/SunRGBD/SUNRGBD4GEN/'
    resolution = 512
    proportion_empty_prompts = 0.
    pretrained_model_name_or_path = HF_PATH('stabilityai/stable-diffusion-2-1/')

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False,
    )

    # dataset
    train_dataset = make_dataset(train_data_dir, resolution, proportion_empty_prompts, tokenizer)

    # test
    for i, item in enumerate(train_dataset):
        rgb_pixel_values        = item['rgb_pixel_values']
        semantic_pixel_values   = item['semantic_pixel_values']
        normal_pixel_values     = item['normal_pixel_values']
        depth_pixel_values      = item['depth_pixel_values']
        input_ids               = item['input_ids']

        print(f'{i} {osp.dirname(item["rgb"])}')
        print(type(rgb_pixel_values), rgb_pixel_values.shape, rgb_pixel_values.dtype, rgb_pixel_values.min(), rgb_pixel_values.max())
        print(type(semantic_pixel_values), semantic_pixel_values.shape, semantic_pixel_values.dtype, semantic_pixel_values.min(), semantic_pixel_values.max())
        print(type(normal_pixel_values), normal_pixel_values.shape, normal_pixel_values.dtype, normal_pixel_values.min(), normal_pixel_values.max())
        print(type(depth_pixel_values), depth_pixel_values.shape, depth_pixel_values.dtype, depth_pixel_values.min(), depth_pixel_values.max())
        print(type(input_ids), input_ids.shape, input_ids.dtype, input_ids.min(), input_ids.max())
        
    print('DONE')