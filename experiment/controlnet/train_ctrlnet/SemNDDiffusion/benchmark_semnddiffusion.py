""" conda activate controlnet && python benchmark_semnddiffusion.py  """

import torch
import os

from diffusers.utils import load_image
from PIL import Image
import numpy as np
import jsonlines
from tqdm import tqdm
try:
    from cv2 import cv2
except:
    import cv2

import torch.nn.functional as F

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
    "--sunrgbd4gen_dataset_root", 
    type=str, 
    default="SUNRGBD4GEN", 
    help="dataset path")
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
parser.add_argument(
    "--n", 
    type=int, 
    default=1, 
    help="number of examples")

args = parser.parse_args()


def load_sunrgbd4gen_data(semantic_path, normal_path, depth_path):
    semantic    = Image.open(semantic_path).convert('RGB')
    semantic    = torch.tensor(np.asarray(semantic)).permute(2,0,1).float() / 255.
    semantic    = semantic[None]
    # print(semantic.shape, semantic.min(), semantic.max(), semantic.dtype)

    normal      = Image.open(normal_path).convert('RGB')
    normal      = torch.tensor(np.asarray(normal)).permute(2,0,1).float() / 255.
    normal      = normal[None]
    # print(normal.shape, normal.min(), normal.max(), normal.dtype)

    depth       = Image.open(depth_path)
    depth       = np.asarray(depth, dtype=np.uint16)
    depth       = np.bitwise_or(np.right_shift(depth, 3), np.left_shift(depth, 16 - 3))
    depth       = depth.astype(np.float32) / 1000.0
    depth[depth>8] = 8  # raw depth in range [0,8]
    depth       = 1. / depth  # inverse depth
    depth       = np.nan_to_num(depth)
    depth       = (depth - depth.min()) / (depth.max() - depth.min())
    depth       = torch.tensor(np.stack([depth]*3)).float()
    depth       = depth[None]
    # print(depth.shape, depth.min(), depth.max(), depth.dtype)

    return semantic, normal, depth


def compute_angular_error(normal_pred, normal_gt, mean=True):
    H, W = normal_pred.shape[:2]
    normal_pred = F.normalize(normal_pred, dim=-1)
    normal_gt   = F.normalize(normal_gt[:H, :W], dim=-1)
    cosine_similarity = (normal_pred * normal_gt).sum(dim=-1)
    cosine_similarity = torch.clamp(cosine_similarity, -1., 1.)
    angle = torch.acos(cosine_similarity)
    angle_degree = torch.rad2deg(angle)
    return angle_degree.mean() if mean else angle_degree


def compute_depth_error(depth_pred, depth_gt):
    H, W = depth_pred.shape[:2]
    depth_gt = depth_gt[:H, :W]
    # depth_pred  = 1. / (depth_pred + 1e-8)
    # depth_gt    = 1. / (depth_gt[:H, :W] + 1e-8)
    MAE = (depth_pred - depth_gt).abs().mean()
    MSE = ((depth_pred - depth_gt)**2).mean()

    return MAE, MSE


if __name__=='__main__':
    # 1. build model
    print("Build model")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        HF_PATH(args.pretrained_model_name_or_path),
        controlnet=ControlNetModel.from_pretrained(
            HF_PATH(args.controlnet_pretrained_model_name_or_path),
            torch_dtype=torch.float16
        ), 
        safety_checker=None,
        torch_dtype=torch.float16
    ).to('cuda:0')
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # 2. load data
    print("Load data")
    angular_error, depth_mae, depth_mse = [], [], []
    with jsonlines.open(os.path.join(args.sunrgbd4gen_dataset_root, 'test.jsonl')) as reader:
        for item in tqdm(reader):
            rgb_path        = os.path.join(args.sunrgbd4gen_dataset_root, item['rgb'])
            semantic_path   = os.path.join(args.sunrgbd4gen_dataset_root, item['semantic'])
            normal_path     = os.path.join(args.sunrgbd4gen_dataset_root, item['normal'])
            depth_path      = os.path.join(args.sunrgbd4gen_dataset_root, item['depth'])
            prompt          = item['description']

            semantic, normal_gt, depth_gt = \
                load_sunrgbd4gen_data(semantic_path, normal_path, depth_path)

            # generation
            generator = torch.manual_seed(42)
            with torch.no_grad():
                images = pipe(
                    [prompt]*args.n, 
                    num_inference_steps=30, 
                    generator=generator, 
                    image=semantic).images

            # metric
            semantic    = semantic[0].permute(1,2,0).cuda()
            normal_gt   = normal_gt[0].permute(1,2,0).cuda()
            normal_gt   = normal_gt * 2. - 1.
            depth_gt    = depth_gt[0,0].cuda()
            for image in images:
                image = np.array(image)
                normal, depth = image[...,:3], image[...,-1]

                normal = torch.tensor(normal, dtype=torch.float32).cuda()
                normal = normal / 255. * 2. - 1.

                depth = torch.tensor(depth, dtype=torch.float32).cuda()
                depth = depth / 255.

                angle_err = compute_angular_error(normal, normal_gt)
                angular_error.append(angle_err.item())

                depth_mae_val, depth_mse_val = compute_depth_error(depth, depth_gt)
                depth_mae.append(depth_mae_val.item())
                depth_mse.append(depth_mse_val.item())

                # cv2.imshow("Semantic", cv2.cvtColor(semantic.cpu().numpy(), cv2.COLOR_RGB2BGR))
                # cv2.imshow("Normal (Prediction)", cv2.cvtColor(normal.cpu().numpy(), cv2.COLOR_RGB2BGR) * 0.5 + 0.5)
                # cv2.imshow("Normal (Ground-Truth)", cv2.cvtColor(normal_gt.cpu().numpy(), cv2.COLOR_RGB2BGR) * 0.5 + 0.5)
                # cv2.imshow("Depth (Prediction)", depth.cpu().numpy())
                # cv2.imshow("Depth (Ground-Truth)", depth_gt.cpu().numpy())
                # cv2.waitKey(10)

            print(f"Angular Error: mean {np.mean(angular_error):.4f}, median {np.median(angular_error):.4f}")
            print(f"Depth Error: MAE {np.mean(depth_mae):.4f}, MSE {np.mean(depth_mse):.4f}, RMSE {np.sqrt(np.mean(depth_mse)):.4f}")

        print(f"Angular Error: mean {np.mean(angular_error):.4f}, median {np.median(angular_error):.4f}")
        print(f"Depth Error: MAE {np.mean(depth_mae):.4f}, MSE {np.mean(depth_mse):.4f}, RMSE {np.sqrt(np.mean(depth_mse)):.4f}")

    print('DONE')