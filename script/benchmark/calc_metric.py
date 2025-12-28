# conda activate scenegeneration
# python calc_metric.py

import os
import os.path as osp
from glob import glob
import numpy as np
from tqdm import tqdm
import cv2
import torch

# pip install torchmetrics[image]
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


HF_ROOT = os.environ.get('HF_ROOT')
HF_PATH = lambda p: osp.join(HF_ROOT, p) if (HF_ROOT is not None) and (not osp.exists(p)) else p


class CLIPScoreCalculator:

    def __init__(self, model_name_or_path='openai/clip-vit-large-patch14', device='cuda:0'):
        self.device = device
        self.metric = CLIPScore(model_name_or_path=HF_PATH(model_name_or_path)).to(device)

    @torch.no_grad()
    def update(self, images, texts):
        """ 
            images: torch.tensor (N,3,H,W) torch.uint8 
            texts: list of str or str
        """
        self.metric.update(images, texts)

    @torch.no_grad()
    def compute(self):
        """ Returns: float """
        score = self.metric.compute()
        score_val = score.detach().item()
        return score_val


class InceptionScoreCalculator:

    def __init__(self, device='cuda:0'):
        self.device = device
        self.metric = InceptionScore(splits=5).to(device)

    @torch.no_grad()
    def update(self, images):
        """ images: torch.tensor (N,3,H,W) torch.uint8 """
        self.metric.update(images)

    @torch.no_grad()
    def compute(self):
        """ Returns: (float, float) """
        mean, std = self.metric.compute()
        mean_val, std_val = mean.detach().item(), std.detach().item()
        return mean_val, std_val


class FrechetInceptionDistanceCalculator:

    def __init__(self, device='cuda:0'):
        self.device = device
        self.metric = FrechetInceptionDistance(feature=192).to(device)

    @torch.no_grad()
    def update(self, images, real=False):
        """ images: torch.tensor (N,3,H,W) torch.uint8 """
        self.metric.update(images, real=real)

    @torch.no_grad()
    def compute(self):
        """ Returns: float """
        fid = self.metric.compute()
        fid_val = fid.detach().item()
        return fid_val


class KernalInceptionDistanceCalculator:

    def __init__(self, device='cuda:0'):
        self.device = device
        self.metric = KernelInceptionDistance(feature=192).to(device)

    @torch.no_grad()
    def update(self, images, real=False):
        """ images: torch.tensor (N,3,H,W) torch.uint8 """
        self.metric.update(images, real=real)

    @torch.no_grad()
    def compute(self):
        """ Returns: float """
        kid, _ = self.metric.compute()
        kid_val = kid.detach().item()
        return kid_val


def compute_ours():
    result_dir = '../../outputs/results/ours'
    sd21base_dir = '../../outputs/results/sd21base'
    device = 'cuda:0'

    prompt_dict = {
        'bedroom_0000_Midcentury_style': 'a midcentury style bedroom',
        'bedroom_0001_Cartoon_style': 'a Cartoon style bedroom',
        'bedroom_0002_Bohemian_style': 'a Bohemian style bedroom',
        'bedroom_0003_Scandinavian_style': 'a Scandinavian style bedroom',
        'bedroom_0004_industrial_style': 'a industrial style bedroom',
        'hypersim_ai_001_001_Bohemian_style': 'a Bohemian style bathroom',
        'hypersim_ai_001_003_Scandinavian_style': 'a Scandinavian style office',
        'hypersim_ai_001_005_industrial_style': 'a industrial style dining room',
        'hypersim_ai_003_004_Chinese_style': 'a Chinese style bedroom',
        'hypersim_ai_006_010_modern_style': 'a modern style dining room',
        'hypersim_ai_010_005_Midcentury_style': 'a Midcentury style bedroom',
        'hypersim_ai_010_008_Cartoon_style': 'a Cartoon style living room',
        'hypersim_ai_022_005_Bohemian_style': 'a Bohemian style living room',
        'livingroom_8013_Chinese_style': 'a Chinese style living room',
        'livingroom_8016_modern_style': 'a modern style living room',
        'livingroom_8017_Midcentury_style': 'a Midcentury style living room',
        'setthescene_bedroom_Scandinavian_style': 'a Scandinavian style bedroom',
        'setthescene_dining_room_industrial_style': 'a industrial style dining room',
        'setthescene_garage_Chinese_style': 'a Chinese style garage',
        'setthescene_living_room_modern_style': 'a modern style living room',
    }

    clipscore_metric    = CLIPScoreCalculator(device=device)
    is_metric           = InceptionScoreCalculator(device=device)
    fid_metric          = FrechetInceptionDistanceCalculator(device=device)
    kid_metric          = KernalInceptionDistanceCalculator(device=device)

    # generated image
    scene_ids = os.listdir(result_dir)
    for scene_id in scene_ids:
        if not osp.isdir(osp.join(result_dir, scene_id)): continue

        # style_prompt = scene_id.split('_')[-2]
        assert scene_id in prompt_dict.keys(), scene_id
        style_prompt = prompt_dict[scene_id]

        image_paths = glob(osp.join(result_dir, scene_id, '*', '*'))
        for image_path in tqdm(image_paths):
            image = cv2.imread(image_path)
            # crop
            height = image.shape[0]
            image = image[:,:height,:3]
            # cv2.imshow('image', image)
            # cv2.waitKey(1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose((2,0,1))[None] # (1,3,H,W)
            image = torch.from_numpy(image).to(device)
            # print(style_prompt, image.shape, image.dtype, image.min(), image.max())

            clipscore_metric.update(image, style_prompt)
            is_metric.update(image)
            fid_metric.update(image, real=False)
            kid_metric.update(image, real=False)

    # real image
    for image_path in tqdm(glob(osp.join(sd21base_dir, '*', '*.*'))):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2,0,1))[None] # (1,3,H,W)
        image = torch.from_numpy(image).to(device)

        fid_metric.update(image, real=True)
        kid_metric.update(image, real=True)

    # compute
    clipscaore      = clipscore_metric.compute()
    is_mean, is_std = is_metric.compute()
    fid             = fid_metric.compute()
    kid             = kid_metric.compute()

    # output
    with open(osp.join(result_dir, 'metrics.txt'), 'w') as f:
        f.write(f'CLIP Score: {clipscaore}\n')
        f.write(f'IS: {is_mean} ({is_std})\n')
        f.write(f'FID: {fid}\n')
        f.write(f'KID: {kid}\n')

    print(f'<------------------------- Ours ------------------------->')
    print(f'CLIP Score: {clipscaore}')
    print(f'IS: {is_mean} ({is_std})')
    print(f'FID: {fid}')
    print(f'KID: {kid}')


def compute_cc3d():
    result_dir = '../../outputs/results/cc3d'
    sd21base_dir = '../../outputs/results/sd21base'
    device = 'cuda:0'

    prompt_dict = {
        'bedroom_0000': 'a modern style bedroom',
        'bedroom_0001': 'a Cartoon style bedroom',
        'bedroom_0002': 'a Bohemian style bedroom',
        'bedroom_0003': 'a Scandinavian style bedroom',
        'bedroom_0004': 'a industrial style bedroom',
        'hypersim_ai_001_005': 'a industrial style dining room',
        'hypersim_ai_003_004': 'a Chinese style bedroom',
        'hypersim_ai_006_010': 'a modern style dining room',
        'hypersim_ai_010_005': 'a Midcentury style bedroom',
        'hypersim_ai_010_008': 'a Cartoon style living room',
        'hypersim_ai_022_005': 'a Bohemian style living room',
        'livingroom_8013': 'a Chinese style living room',
        'livingroom_8016': 'a modern style living room',
        'livingroom_8017': 'a Midcentury style living room',
        'setthescene_bedroom': 'a Scandinavian style bedroom',
        'setthescene_dining_room': 'a industrial style dining room',
        'setthescene_living_room': 'a modern style living room',
        'fankenstein_bedroom_001': 'a Cartoon style bedroom'
    }

    clipscore_metric    = CLIPScoreCalculator(device=device)
    is_metric           = InceptionScoreCalculator(device=device)
    fid_metric          = FrechetInceptionDistanceCalculator(device=device)
    kid_metric          = KernalInceptionDistanceCalculator(device=device)

    # generated image
    scene_ids = os.listdir(result_dir)
    for scene_id in scene_ids:
        if not osp.isdir(osp.join(result_dir, scene_id)): continue
        
        # style_prompt = scene_id.split('_')[-2]
        assert scene_id in prompt_dict.keys(), scene_id
        style_prompt = prompt_dict[scene_id]

        image_paths = glob(osp.join(result_dir, scene_id, '*', '*'))
        for image_path in tqdm(image_paths):
            image = cv2.imread(image_path)
            # crop
            height = image.shape[0]
            image = image[:,:height,:3]
            # cv2.imshow('image', image)
            # cv2.waitKey(1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose((2,0,1))[None] # (1,3,H,W)
            image = torch.from_numpy(image).to(device)
            # print(style_prompt, image.shape, image.dtype, image.min(), image.max())

            clipscore_metric.update(image, style_prompt)
            is_metric.update(image)
            fid_metric.update(image, real=False)
            kid_metric.update(image, real=False)

    # real image
    for image_path in tqdm(glob(osp.join(sd21base_dir, '*', '*.*'))):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2,0,1))[None] # (1,3,H,W)
        image = torch.from_numpy(image).to(device)

        fid_metric.update(image, real=True)
        kid_metric.update(image, real=True)

    # compute
    clipscaore      = clipscore_metric.compute()
    is_mean, is_std = is_metric.compute()
    fid             = fid_metric.compute()
    kid             = kid_metric.compute()

    # output
    with open(osp.join(result_dir, 'metrics.txt'), 'w') as f:
        f.write(f'CLIP Score: {clipscaore}\n')
        f.write(f'IS: {is_mean} ({is_std})\n')
        f.write(f'FID: {fid}\n')
        f.write(f'KID: {kid}\n')

    print(f'<------------------------- CC3D ------------------------->')
    print(f'CLIP Score: {clipscaore}')
    print(f'IS: {is_mean} ({is_std})')
    print(f'FID: {fid}')
    print(f'KID: {kid}')


def compute_frankenstein():
    result_dir = '../../outputs/results/frankenstein'
    sd21base_dir = '../../outputs/results/sd21base'
    device = 'cuda:0'

    prompt_dict = {
        'bedroom_0000': 'a modern style bedroom',
        'bedroom_0001': 'a Cartoon style bedroom',
        'bedroom_0002': 'a Bohemian style bedroom',
        'bedroom_0003': 'a Scandinavian style bedroom',
        'bedroom_0004': 'a industrial style bedroom',
        'hypersim_ai_001_005': 'a industrial style dining room',
        'hypersim_ai_006_010': 'a modern style dining room',
        'hypersim_ai_010_005': 'a Midcentury style bedroom',
        'hypersim_ai_010_008': 'a Cartoon style living room',
        'hypersim_ai_022_005': 'a Bohemian style living room',
        'livingroom_8013': 'a Chinese style living room',
        'livingroom_8016': 'a modern style living room',
        'livingroom_8017': 'a Midcentury style living room',
        'setthescene_bedroom': 'a Scandinavian style bedroom',
        'setthescene_dining_room': 'a industrial style dining room',
        'setthescene_living_room': 'a modern style living room',
        'fankenstein_bedroom_001': 'a Cartoon style bedroom'
    }

    clipscore_metric    = CLIPScoreCalculator(device=device)
    is_metric           = InceptionScoreCalculator(device=device)
    fid_metric          = FrechetInceptionDistanceCalculator(device=device)
    kid_metric          = KernalInceptionDistanceCalculator(device=device)

    # generated image
    scene_ids = os.listdir(result_dir)
    for scene_id in scene_ids:
        if not osp.isdir(osp.join(result_dir, scene_id)): continue
        
        # style_prompt = scene_id.split('_')[-2]
        assert scene_id in prompt_dict.keys(), scene_id
        style_prompt = prompt_dict[scene_id]

        image_paths = glob(osp.join(result_dir, scene_id, '*'))
        for image_path in tqdm(image_paths):
            image = cv2.imread(image_path)
            # crop
            height = image.shape[0]
            image = image[:,:height,:3]
            # cv2.imshow('image', image)
            # cv2.waitKey(1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose((2,0,1))[None] # (1,3,H,W)
            image = torch.from_numpy(image).to(device)
            # print(style_prompt, image.shape, image.dtype, image.min(), image.max())

            clipscore_metric.update(image, style_prompt)
            is_metric.update(image)
            fid_metric.update(image, real=False)
            kid_metric.update(image, real=False)

    # real image
    for image_path in tqdm(glob(osp.join(sd21base_dir, '*', '*.*'))):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2,0,1))[None] # (1,3,H,W)
        image = torch.from_numpy(image).to(device)

        fid_metric.update(image, real=True)
        kid_metric.update(image, real=True)

    # compute
    clipscaore      = clipscore_metric.compute()
    is_mean, is_std = is_metric.compute()
    fid             = fid_metric.compute()
    kid             = kid_metric.compute()

    # output
    with open(osp.join(result_dir, 'metrics.txt'), 'w') as f:
        f.write(f'CLIP Score: {clipscaore}\n')
        f.write(f'IS: {is_mean} ({is_std})\n')
        f.write(f'FID: {fid}\n')
        f.write(f'KID: {kid}\n')

    print(f'<------------------------- Frankenstein ------------------------->')
    print(f'CLIP Score: {clipscaore}')
    print(f'IS: {is_mean} ({is_std})')
    print(f'FID: {fid}')
    print(f'KID: {kid}')


def compute_gala3d():
    result_dir = '../../outputs/results/gala3d'
    sd21base_dir = '../../outputs/results/sd21base'
    device = 'cuda:0'

    prompt_dict = {
        'bedroom_0000_Midcentury': 'a Midcentury style bedroom',
        'bedroom_0001_Cartoon': 'a Cartoon style bedroom',
        'bedroom_0002_Bohemian': 'a Bohemian style bedroom',
        'bedroom_0003_Scandinavian': 'a Scandinavian style bedroom',
        'bedroom_0004_Industrial': 'a industrial style bedroom',
        'hypersim_ai_001_001_Bohemian': 'a Bohemian style bathroom',
        'hypersim_ai_001_003_Scandinavian': 'a Scandinavian style office',
        'hypersim_ai_001_005_Industrial': 'a industrial style dining room',
        'hypersim_ai_003_004_Chinese': 'a Chinese style bedroom',
        'hypersim_ai_006_010_Modern': 'a modern style dining room',
        'hypersim_ai_010_005_Midcentury': 'a Midcentury style bedroom',
        'hypersim_ai_010_008_Cartoon': 'a Cartoon style living room',
        'hypersim_ai_022_005_Bohemian': 'a Bohemian style living room',
        'livingroom_8013_Chinese': 'a Chinese style living room',
        'livingroom_8016_Modern': 'a modern style living room',
        'livingroom_8017_Midcentury': 'a Midcentury style living room',
        'setthescene_bedroom_Scandinavian': 'a Scandinavian style bedroom',
        'setthescene_dining_room_Industrial': 'a industrial style dining room',
        'setthescene_garage_Chinese': 'a Chinese style garage',
        'setthescene_living_room_Modern': 'a modern style living room',
        'fankenstein_bedroom_001_Cartoon': 'a Cartoon style bedroom',
    }

    clipscore_metric    = CLIPScoreCalculator(device=device)
    is_metric           = InceptionScoreCalculator(device=device)
    fid_metric          = FrechetInceptionDistanceCalculator(device=device)
    kid_metric          = KernalInceptionDistanceCalculator(device=device)

    # generated image
    scene_ids = os.listdir(result_dir)
    for scene_id in scene_ids:
        if not osp.isdir(osp.join(result_dir, scene_id)): continue
        
        # style_prompt = scene_id.split('_')[-2]
        assert scene_id in prompt_dict.keys(), scene_id
        style_prompt = prompt_dict[scene_id]

        image_paths = glob(osp.join(result_dir, scene_id, '*', '*'))
        for image_path in tqdm(image_paths):
            image = cv2.imread(image_path)
            # crop
            height = image.shape[0]
            image = image[:,:height,:3]
            # cv2.imshow('image', image)
            # cv2.waitKey(1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = np.all(image >= 255, axis=2)
            image[mask] = 0 # black background for better metric of GALA3D
            image = image.transpose((2,0,1))[None] # (1,3,H,W)
            image = torch.from_numpy(image).to(device)
            # print(style_prompt, image.shape, image.dtype, image.min(), image.max())

            clipscore_metric.update(image, style_prompt)
            is_metric.update(image)
            fid_metric.update(image, real=False)
            kid_metric.update(image, real=False)

    # real image
    for image_path in tqdm(glob(osp.join(sd21base_dir, '*', '*.*'))):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2,0,1))[None] # (1,3,H,W)
        image = torch.from_numpy(image).to(device)

        fid_metric.update(image, real=True)
        kid_metric.update(image, real=True)

    # compute
    clipscaore      = clipscore_metric.compute()
    is_mean, is_std = is_metric.compute()
    fid             = fid_metric.compute()
    kid             = kid_metric.compute()

    # output
    with open(osp.join(result_dir, 'metrics.txt'), 'w') as f:
        f.write(f'CLIP Score: {clipscaore}\n')
        f.write(f'IS: {is_mean} ({is_std})\n')
        f.write(f'FID: {fid}\n')
        f.write(f'KID: {kid}\n')

    print(f'<------------------------- GALA3D ------------------------->')
    print(f'CLIP Score: {clipscaore}')
    print(f'IS: {is_mean} ({is_std})')
    print(f'FID: {fid}')
    print(f'KID: {kid}')


def compute_scenecraft():
    result_dir = '../../outputs/results/scenecraft'
    sd21base_dir = '../../outputs/results/sd21base'
    device = 'cuda:0'

    prompt_dict = {
        'bedroom_0000': 'a Midcentury style bedroom',
        'bedroom_0001': 'a Cartoon style bedroom',
        'bedroom_0002': 'a Bohemian style bedroom',
        'bedroom_0003': 'a Scandinavian style bedroom',
        'bedroom_0004': 'a industrial style bedroom',
        'hypersim_ai_001_001': 'a Bohemian style bathroom',
        'hypersim_ai_001_003': 'a Scandinavian style office',
        'hypersim_ai_001_005': 'a industrial style dining room',
        'hypersim_ai_003_004': 'a Chinese style bedroom',
        'hypersim_ai_006_010': 'a modern style dining room',
        'hypersim_ai_010_005': 'a Midcentury style bedroom',
        'hypersim_ai_010_008': 'a Cartoon style living room',
        'hypersim_ai_022_005': 'a Bohemian style living room',
        'livingroom_8017': 'a Midcentury style living room',
        'setthescene_bedroom': 'a Bohemian style bedroom',
        'setthescene_dining_room': 'a industrial style dining room',
        'setthescene_garage': 'a Chinese style garage',
        'setthescene_living_room': 'a modern style living room',
        'fankenstein_bedroom_001': 'a Cartoon style bedrooom'
    }

    clipscore_metric    = CLIPScoreCalculator(device=device)
    is_metric           = InceptionScoreCalculator(device=device)
    fid_metric          = FrechetInceptionDistanceCalculator(device=device)
    kid_metric          = KernalInceptionDistanceCalculator(device=device)

    # generated image
    scene_ids = os.listdir(result_dir)
    for scene_id in scene_ids:
        if not osp.isdir(osp.join(result_dir, scene_id)): continue
        
        # style_prompt = scene_id.split('_')[-2]
        assert scene_id in prompt_dict.keys(), scene_id
        style_prompt = prompt_dict[scene_id]

        image_paths = glob(osp.join(result_dir, scene_id, '*', 'rgb', '*'))
        for image_path in tqdm(image_paths):
            image = cv2.imread(image_path)
            # crop
            height = image.shape[0]
            image = image[:,:height,:3]
            # cv2.imshow('image', image)
            # cv2.waitKey(1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose((2,0,1))[None] # (1,3,H,W)
            image = torch.from_numpy(image).to(device)
            # print(style_prompt, image.shape, image.dtype, image.min(), image.max())

            clipscore_metric.update(image, style_prompt)
            is_metric.update(image)
            fid_metric.update(image, real=False)
            kid_metric.update(image, real=False)

    # real image
    for image_path in tqdm(glob(osp.join(sd21base_dir, '*', '*.*'))):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2,0,1))[None] # (1,3,H,W)
        image = torch.from_numpy(image).to(device)

        fid_metric.update(image, real=True)
        kid_metric.update(image, real=True)

    # compute
    clipscaore      = clipscore_metric.compute()
    is_mean, is_std = is_metric.compute()
    fid             = fid_metric.compute()
    kid             = kid_metric.compute()

    # output
    with open(osp.join(result_dir, 'metrics.txt'), 'w') as f:
        f.write(f'CLIP Score: {clipscaore}\n')
        f.write(f'IS: {is_mean} ({is_std})\n')
        f.write(f'FID: {fid}\n')
        f.write(f'KID: {kid}\n')

    print(f'<------------------------- SceneCraft ------------------------->')
    print(f'CLIP Score: {clipscaore}')
    print(f'IS: {is_mean} ({is_std})')
    print(f'FID: {fid}')
    print(f'KID: {kid}')


def compute_setthescene():
    result_dir = '../../outputs/results/setthescene'
    sd21base_dir = '../../outputs/results/sd21base'
    device = 'cuda:0'

    prompt_dict = {
        'bedroom_0000_Midcentury': 'a midcentury style bedroom',
        'bedroom_0001_Cartoon': 'a Cartoon style bedroom',
        'bedroom_0002_Bohemian': 'a Bohemian style bedroom',
        'bedroom_0003_Scandinavian': 'a Scandinavian style bedroom',
        'bedroom_0004_Industrial': 'a industrial style bedroom',
        'hypersim_ai_001_001_Bohemian': 'a Bohemian style bathroom',
        'hypersim_ai_001_005_Industrial': 'a industrial style dining room',
        'hypersim_ai_003_004_Chinese': 'a Chinese style bedroom',
        'hypersim_ai_006_010_Modern': 'a modern style dining room',
        'hypersim_ai_010_005_Midcentury': 'a Midcentury style bedroom',
        'hypersim_ai_010_008_Cartoon': 'a Cartoon style living room',
        'hypersim_ai_022_005_Bohemian': 'a Bohemian style living room',
        'livingroom_8013_Chinese': 'a Chinese style living room',
        'livingroom_8016_Modern': 'a modern style living room',
        'livingroom_8017_Midcentury': 'a Midcentury style living room',
        'setthescene_bedroom_Scandinavian': 'a Scandinavian style bedroom',
        'setthescene_dining_room_Industrial': 'a industrial style dining room',
        'setthescene_garage_Chinese': 'a Chinese style garage',
        'setthescene_living_room_Modern': 'a modern style living room',
        'fankenstein_bedroom_001_Cartoon': 'a Cartoon style bedroom',
    }

    clipscore_metric    = CLIPScoreCalculator(device=device)
    is_metric           = InceptionScoreCalculator(device=device)
    fid_metric          = FrechetInceptionDistanceCalculator(device=device)
    kid_metric          = KernalInceptionDistanceCalculator(device=device)

    # generated image
    scene_ids = os.listdir(result_dir)
    for scene_id in scene_ids:
        if not osp.isdir(osp.join(result_dir, scene_id)): continue
        
        # style_prompt = scene_id.split('_')[-2]
        assert scene_id in prompt_dict.keys(), scene_id
        style_prompt = prompt_dict[scene_id]

        image_paths = glob(osp.join(result_dir, scene_id, '*', '*_foreground.*'))
        for image_path in tqdm(image_paths):
            image = cv2.imread(image_path)
            # crop
            height = image.shape[0]
            image = image[:,:height,:3]
            # cv2.imshow('image', image)
            # cv2.waitKey(1)
            mask = np.all(image <= 0, axis=2)
            image[mask] = 255
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose((2,0,1))[None] # (1,3,H,W)
            image = torch.from_numpy(image).to(device)
            # print(style_prompt, image.shape, image.dtype, image.min(), image.max())

            clipscore_metric.update(image, style_prompt)
            is_metric.update(image)
            fid_metric.update(image, real=False)
            kid_metric.update(image, real=False)

    # real image
    for image_path in tqdm(glob(osp.join(sd21base_dir, '*', '*.*'))):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2,0,1))[None] # (1,3,H,W)
        image = torch.from_numpy(image).to(device)

        fid_metric.update(image, real=True)
        kid_metric.update(image, real=True)

    # compute
    clipscaore      = clipscore_metric.compute()
    is_mean, is_std = is_metric.compute()
    fid             = fid_metric.compute()
    kid             = kid_metric.compute()

    # output
    with open(osp.join(result_dir, 'metrics.txt'), 'w') as f:
        f.write(f'CLIP Score: {clipscaore}\n')
        f.write(f'IS: {is_mean} ({is_std})\n')
        f.write(f'FID: {fid}\n')
        f.write(f'KID: {kid}\n')

    print(f'<------------------------- Setthescene ------------------------->')
    print(f'CLIP Score: {clipscaore}')
    print(f'IS: {is_mean} ({is_std})')
    print(f'FID: {fid}')
    print(f'KID: {kid}')


if __name__=='__main__':
    compute_cc3d()
    compute_frankenstein()
    compute_setthescene()
    compute_scenecraft()
    compute_gala3d()
    compute_ours()