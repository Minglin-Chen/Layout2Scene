"""
1. Download the processed Hyerpsim Dataset (of SceneCraft) via:
    huggingface-cli download gzzyyxy/layout_diffusion_hypersim --repo-type dataset --local-dir /path/to/local/hypersim
2. Preprocess the dataset via:
    python build_hypersim4generation.py --dataset_root /path/to/local/hypersim --output_path /path/to/local/HYPERSIM4GEN
"""

import sys
import os
import os.path as osp
import numpy as np
from PIL import Image
import json
try:
    import cv2
except:
    from cv2 import cv2

import torch
from transformers import pipeline
from datasets import load_dataset

from controlnet_aux import NormalBaeDetector

sys.path.append('../../../../')
from core.utils.ade20k_protocol import ade20k_label2color


HF_ROOT = os.environ.get('HF_ROOT')
HF_PATH = lambda p: p if HF_ROOT is None else osp.join(HF_ROOT, p)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_root", 
    type=str, 
    default="layout_diffusion_hypersim", 
    help="dataset path")
parser.add_argument(
    "--output_path", 
    type=str, 
    default="HYPERSIM4GEN", 
    help="output path")

args = parser.parse_args()


nyu40_id2label = {
    0: "unlabeled",
    1: "wall",
    2: "floor",
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    13: "blinds",
    14: "desk",
    15: "shelves",
    16: "curtain",
    17: "dresser",
    18: "pillow",
    19: "mirror",
    20: "floormat",
    21: "clothes",
    22: "ceiling",
    23: "books",
    24: "refrigerator",
    25: "television",
    26: "paper",
    27: "towel",
    28: "showercurtain",
    29: "box",
    30: "whiteboard",
    31: "person",
    32: "nightstand",
    33: "toilet",
    34: "sink",
    35: "lamp",
    36: "bathtub",
    37: "bag",
    38: "otherstructure",
    39: "otherfurniture",
    40: "otherprop",
}

nyu40_to_ade20k_label_alias = {
    'unlabeled':        'base',
    'window':           'windowpane', 
    'bookshelf':        'bookcase', 
    'picture':          'painting',
    'blinds':           'blind', 
    'shelves':          'shelf', 
    'dresser':          'desk',
    'floormat':         'rug',
    'clothes':          'apparel', 
    'books':            'book', 
    'television':       'television receiver', 
    'paper':            'poster', 
    'showercurtain':    'curtain', 
    'whiteboard':       'signboard', 
    'nightstand':       'table', 
    'otherstructure':   'base', 
    'otherfurniture':   'base', 
    'otherprop':        'base'
}

# nyu40_to_ade20k = [
#     40, 0, 3, 10, 7, 19, 23, 15, 14, 8, 62, 22, 45, 63, 33, 24, 18, 33, 57, 27, 28, 
#     92, 5, 67, 50, 89, 100, 81, 18, 41, 43, 12, 15, 65, 47, 36, 37, 115, 40, 40, 40
# ]


def get_semantic(sem_id):
    H, W = sem_id.shape

    semantic = np.zeros((H,W,3), dtype=np.uint8)
    for id in np.unique(sem_id):
        label = nyu40_id2label[id]

        if label in nyu40_to_ade20k_label_alias.keys():
            label = nyu40_to_ade20k_label_alias[label]

        assert label in ade20k_label2color.keys()
        color = np.array(ade20k_label2color[label], dtype=np.uint8)

        semantic[sem_id==id] = color

    return semantic


depth_estimator = None
def get_depth(image):
    global depth_estimator
    if depth_estimator is None:
        depth_estimator = pipeline('depth-estimation', model=HF_PATH("Intel/dpt-large"), device=0)

    depth = depth_estimator(image)['depth']
    return depth


normal_estimator = None
def get_normal(image):

    global normal_estimator
    if normal_estimator is None:
        normal_estimator = NormalBaeDetector.from_pretrained(HF_PATH("lllyasviel/Annotators"))
        normal_estimator.to('cuda:0')

    normal = normal_estimator(image)
    return normal


stable_normal_predictor = None
def stable_normal_forward(image, device='cuda'):
    global stable_normal_predictor
    if stable_normal_predictor is None:
        stable_normal_predictor = torch.hub.load(
            "Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True, 
            local_cache_dir=HF_PATH('Stable-X'))
        stable_normal_predictor.to(device)
        
    assert stable_normal_predictor is not None

    normal_image = stable_normal_predictor(image)
    return normal_image


def wirte_jsonl(data, path):
    with open(path, 'w') as f:
        for entry in data:
            line = json.dumps(entry)
            f.write(line + '\n')
            

if __name__=='__main__':

    dataset_root    = args.dataset_root
    output_path     = args.output_path

    dataset = load_dataset("parquet", data_files=osp.join(dataset_root, "data/*.parquet"), streaming=True)

    meta = []
    for i, batch in enumerate(dataset["train"]):
        print(i)

        item_id = f'{i:05d}'
        dst_root = osp.join(output_path, 'data', item_id)
        os.makedirs(dst_root, exist_ok=True)
        dst_rgb_path        = osp.join(dst_root, 'rgb.jpg')
        dst_depth_path      = osp.join(dst_root, 'depth.png')
        dst_normal_path     = osp.join(dst_root, 'normal.png')
        dst_semantic_path   = osp.join(dst_root, 'semantic.png')

        rgb = batch['target']
        rgb.save(dst_rgb_path)

        depth = get_depth(rgb)
        depth.save(dst_depth_path)

        # normal = get_normal(rgb)
        normal = stable_normal_forward(rgb)
        normal = normal.resize(rgb.size)
        normal.save(dst_normal_path)


        sem_id = np.array(batch['labels'])
        semantic = get_semantic(sem_id)
        semantic = Image.fromarray(semantic)
        semantic.save(dst_semantic_path)
        
        prompt = batch['prompt']

        item_dict = {
            'rgb':          osp.join('data', item_id, 'rgb.jpg'),
            'depth':        osp.join('data', item_id, 'depth.png'),
            'normal':       osp.join('data', item_id, 'normal.png'),
            'semantic':     osp.join('data', item_id, 'semantic.png'),
            'description':  prompt
        }

        meta.append(item_dict)

        # visualization
        # cv2.imshow("RGB", cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR))
        # cv2.imshow("Depth", np.array(depth))
        # cv2.imshow("Normal", cv2.cvtColor(np.array(normal), cv2.COLOR_RGB2BGR))
        # cv2.imshow("Semantic", cv2.cvtColor(np.array(semantic), cv2.COLOR_RGB2BGR))
        # cv2.waitKey(10)

    total_num_items = len(meta)
    train_data, test_data = meta[:int(total_num_items * 0.8)], meta[int(total_num_items * 0.8):]
    wirte_jsonl(train_data, osp.join(output_path, 'train.jsonl'))
    wirte_jsonl(test_data, osp.join(output_path, 'test.jsonl'))

    print('DONE')