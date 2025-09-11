# conda activate scenegeneration
# python calc_metric.py

import os
import os.path as osp
import torch

# pip install torchmetrics[image]
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.inception import InceptionScore


@torch.no_grad()
def compute_CLIP_score(images, texts, model_name_or_path='openai/clip-vit-large-patch14'):
    """
    Args:
        images: torch.tensor (N,C,H,W) torch.uint8
        texts: list of str
        model_name_or_path: str
    Returns:
        float
    """
    metric = CLIPScore(model_name_or_path=model_name_or_path).cuda()
    score = metric(images, texts)
    score_val = score.detach().item()
    return score_val


@torch.no_grad()
def compute_inception_score(images):
    """
    Args:
        images: torch.tensor (N,3,H,W) torch.uint8
    Returns:
        (float, float)
    """
    metric = InceptionScore().cuda()
    metric.update(images)
    mean, std = metric.compute()
    mean_val, std_val = mean.detach().item(), std.detach().item()
    return mean_val, std_val


if __name__=='__main__':
    HUGGINGFACE_ROOT    = "F:\huggingface"
    CLIP_PATH           = osp.join(HUGGINGFACE_ROOT, "openai/clip-vit-large-patch14")

    images  = torch.randint(0, 255, (2,3,224,224), dtype=torch.uint8).cuda()
    texts   = ["random", "none"]
    
    CS      = compute_CLIP_score(images, texts, CLIP_PATH)
    IS, _   = compute_inception_score(images)

    # output
    print(f"Metric: [CS {CS:0.2f}], [IS {IS:0.2f}]")