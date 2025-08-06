"""
1. Download the processed ScanNet++ Dataset (of SceneCraft) via:
    huggingface-cli download gzzyyxy/layout_diffusion_scannetpp_voxel0.2 --repo-type dataset --local-dir /path/to/local
2. Preprocess the dataset via:
    python build_scannetpp4generation.py --dataset_root /path/to/local --output_path /path/to/local/SCANNETPP4GEN
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
from transformers import pipeline, AutoImageProcessor, UperNetForSemanticSegmentation
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
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
    default="layout_diffusion_scannetpp_voxel0.2", 
    help="dataset path")
parser.add_argument(
    "--output_path", 
    type=str, 
    default="SCANNETPP4GEN", 
    help="output path")

args = parser.parse_args()


scenecraft_scannetpp_id2label = {
    0: "wall",
    1: "ceiling",
    2: "floor",
    3: "cabinet",
    4: "bed",
    5: "chair",  
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "counter",
    12: "desk",
    13: "curtain",
    14: "refrigerator",
    15: "television",
    16: "whiteboard",
    17: "toilet",
    18: "sink",
    19: "bathtub",
    20: "doorframe",
    21: "ceiling lamp",
    22: "blinds",
    23: "office chair",
    24: "ceiling light",
    25: "monitor",
    26: "SPLIT",
    27: "object",
    28: "split",
    29: "shelf",
    30: "fake ceiling",
    31: "storage cabinet",
    32: "window frame",
    33: "pipe",
    34: "structure",
    35: "storage rack",
    36: "heater",
    37: "box",
    38: "kitchen cabinet",
    39: "plant",
    40: "tv",
    41: "door frame",
    42: "shower wall",
    43: "computer tower",
    44: "windowsill",
    45: "blanket",
    46: "books",
    47: "linked retractable seats",
    48: "wardrobe",
    49: "trash can",
    50: "window sill",
    51: "cable tray",
    52: "machine",
    53: "jacket",
    54: "trash bin",
    55: "remove",
    56: "carpet",
    57: "kitchen counter",
    58: "pillar",
    59: "air duct",
    60: "bag",
    61: "REMOVE",
    62: "towel",
    63: "window blind",
    64: "electrical duct",
    65: "keyboard",
    66: "stool",
    67: "clothes",
    68: "pillow",
    69: "roof",
    70: "joined tables",
    71: "backpack",
    72: "blackboard",
    73: "suitcase",
    74: "rack",
    75: "fume hood",
    76: "microwave",
    77: "office table",
    78: "pedestal fan",
    79: "rug",
    80: "pipe storage rack",
    81: "blind",
    82: "ladder",
    83: "printer",
    84: "air conditioner",
    85: "cupboard",
    86: "shower curtain",
    87: "windowframe",
    88: "book",
    89: "suspended ceiling",
    90: "rolling cart",
    91: "bottle",
    92: "construction materials",
    93: "glass wall",
    94: "beam",
    95: "crate",
    96: "picture",
    97: "mirror",
    98: "pc",
    99: "milling machine",
    100: "bin",
    101: "bench",
    102: "paper",
    103: "basket",
    104: "oven",
    105: "tv stand",
    106: "power socket",
    107: "ceiling beam",
    108: "stove",
    109: "trolley",
    110: "telephone",
    111: "shoes",
    112: "hydraulic press",
    113: "laboratory cabinet",
    114: "painting",
    115: "lab machine",
    116: "shower floor",
    117: "raised floor",
    118: "sofa chair",
    119: "medical appliance",
    120: "standing lamp",
    121: "objects",
    122: "laptop",
    123: "file cabinet",
    124: "arm chair",
    125: "electrical control panel",
    126: "washing machine",
    127: "bucket",
    128: "shoe rack"
}

scenecraft_scannetpp_to_ade20k_label_alias = {
    'window':                   'windowpane', 
    'bookshelf':                'bookcase', 
    'television':               'television receiver', 
    'whiteboard':               'bulletin board', 
    'doorframe':                'door', 
    'ceiling lamp':             'ceiling', 
    'blinds':                   'blind', 
    'office chair':             'chair', 
    'ceiling light':            'ceiling', 
    'SPLIT':                    'step', 
    'object':                   'case',
    'split':                    'step', 
    'fake ceiling':             'ceiling', 
    'storage cabinet':          'cabinet', 
    'window frame':             'windowpane', 
    'pipe':                     'pot', 
    'structure':                'building', 
    'storage rack':             'shelf', 
    'heater':                   'microwave', 
    'kitchen cabinet':          'cabinet', 
    'tv':                       'television receiver', 
    'door frame':               'door', 
    'shower wall':              'shower', 
    'computer tower':           'computer', 
    'windowsill':               'windowpane', 
    'books':                    'book', 
    'linked retractable seats': 'chair', 
    'trash can':                'ashcan', 
    'window sill':              'windowpane', 
    'cable tray':               'tray', 
    'machine':                  'arcade machine',
    'jacket':                   'apparel',
    'trash bin':                'ashcan', 
    'remove':                   'step', 
    'carpet':                   'rug', 
    'kitchen counter':          'countertop', 
    'pillar':                   'column', 
    'air duct':                 'building', 
    'REMOVE':                   'step', 
    'window blind':             'blind', 
    'electrical duct':          'building', 
    'keyboard':                 'computer', 
    'clothes':                  'apparel', 
    'roof':                     'ceiling', 
    'joined tables':            'table', 
    'backpack':                 'bag', 
    'blackboard':               'bulletin board', 
    'suitcase':                 'case', 
    'rack':                     'shelf', 
    'fume hood':                'hood', 
    'office table':             'table',
    'pedestal fan':             'fan',
    'pipe storage rack':        'shelf', 
    'ladder':                   'stairway', 
    'printer':                  'computer', 
    'air conditioner':          'radiator', 
    'cupboard':                 'cabinet', 
    'shower curtain':           'curtain', 
    'windowframe':              'windowpane', 
    'suspended ceiling':        'ceiling', 
    'rolling cart':             'truck', 
    'construction materials':   'building', 
    'glass wall':               'glass', 
    'beam':                     'ceiling', 
    'crate':                    'box', 
    'picture':                  'base', 
    'pc':                       'computer', 
    'milling machine':          'case', 
    'bin':                      'box', 
    'paper':                    'poster', 
    'tv stand':                 'television receiver', 
    'power socket':             'case', 
    'ceiling beam':             'ceiling', 
    'trolley':                  'truck', 
    'telephone':                'computer', 
    'shoes':                    'case',
    'hydraulic press':          'case', 
    'laboratory cabinet':       'cabinet', 
    'lab machine':              'case', 
    'shower floor':             'floor', 
    'raised floor':             'floor', 
    'sofa chair':               'sofa', 
    'medical appliance':        'case', 
    'standing lamp':            'lamp', 
    'objects':                  'case', 
    'laptop':                   'computer', 
    'file cabinet':             'cabinet', 
    'arm chair':                'chair', 
    'electrical control panel': 'arcade machine', 
    'washing machine':          'washer', 
    'bucket':                   'barrel', 
    'shoe rack':                'shelf'
}

# scenecraft_scannetpp_to_ade20k = [
#     0, 5, 3, 10, 7, 19, 23, 15, 14, 8, 62, 45, 33, 18, 50, 89, 144, 65, 47, 37, 14, 5, 63, 19, 5, 
#     143, 121, 55, 121, 24, 5, 10, 8, 125, 1, 24, 124, 41, 10, 17, 89, 14, 145, 74, 8, 131, 67, 19, 
#     35, 138, 8, 137, 78, 92, 138, 121, 28, 70, 42, 1, 115, 121, 81, 63, 1, 74, 110, 92, 57, 5, 15, 
#     115, 144, 55, 24, 133, 124, 15, 139, 28, 24, 63, 59, 74, 146, 10, 18, 8, 67, 5, 83, 98, 1, 147, 
#     5, 41, 40, 27, 74, 55, 41, 69, 100, 112, 118, 89, 55, 5, 71, 83, 74, 55, 55, 10, 22, 55, 3, 3, 
#     23, 55, 36, 55, 74, 10, 19, 78, 107, 111, 24
# ]

ade20k_id2color = np.asarray([
    # [0, 0, 0],
    [120, 120, 120],
    [180, 120, 120],
    [6, 230, 230],
    [80, 50, 50],
    [4, 200, 3],
    [120, 120, 80],
    [140, 140, 140],
    [204, 5, 255],
    [230, 230, 230],
    [4, 250, 7],
    [224, 5, 255],
    [235, 255, 7],
    [150, 5, 61],
    [120, 120, 70],
    [8, 255, 51],
    [255, 6, 82],
    [143, 255, 140],
    [204, 255, 4],
    [255, 51, 7],
    [204, 70, 3],
    [0, 102, 200],
    [61, 230, 250],
    [255, 6, 51],
    [11, 102, 255],
    [255, 7, 71],
    [255, 9, 224],
    [9, 7, 230],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [7, 255, 224],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [255, 122, 8],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
    [31, 255, 0],
    [255, 31, 0],
    [255, 224, 0],
    [153, 255, 0],
    [0, 0, 255],
    [255, 71, 0],
    [0, 235, 255],
    [0, 173, 255],
    [31, 0, 255],
    [11, 200, 200],
    [255, 82, 0],
    [0, 255, 245],
    [0, 61, 255],
    [0, 255, 112],
    [0, 255, 133],
    [255, 0, 0],
    [255, 163, 0],
    [255, 102, 0],
    [194, 255, 0],
    [0, 143, 255],
    [51, 255, 0],
    [0, 82, 255],
    [0, 255, 41],
    [0, 255, 173],
    [10, 0, 255],
    [173, 255, 0],
    [0, 255, 153],
    [255, 92, 0],
    [255, 0, 255],
    [255, 0, 245],
    [255, 0, 102],
    [255, 173, 0],
    [255, 0, 20],
    [255, 184, 184],
    [0, 31, 255],
    [0, 255, 61],
    [0, 71, 255],
    [255, 0, 204],
    [0, 255, 194],
    [0, 255, 82],
    [0, 10, 255],
    [0, 112, 255],
    [51, 0, 255],
    [0, 194, 255],
    [0, 122, 255],
    [0, 255, 163],
    [255, 153, 0],
    [0, 255, 10],
    [255, 112, 0],
    [143, 255, 0],
    [82, 0, 255],
    [163, 255, 0],
    [255, 235, 0],
    [8, 184, 170],
    [133, 0, 255],
    [0, 255, 92],
    [184, 0, 255],
    [255, 0, 31],
    [0, 184, 255],
    [0, 214, 255],
    [255, 0, 112],
    [92, 255, 0],
    [0, 224, 255],
    [112, 224, 255],
    [70, 184, 160],
    [163, 0, 255],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [255, 0, 143],
    [0, 255, 235],
    [133, 255, 0],
    [255, 0, 235],
    [245, 0, 255],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 41, 255],
    [0, 255, 204],
    [41, 0, 255],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [122, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [0, 133, 255],
    [255, 214, 0],
    [25, 194, 194],
    [102, 255, 0],
    [92, 0, 255],
])


def get_semantic(sem_id):
    H, W = sem_id.shape

    semantic = np.zeros((H,W,3), dtype=np.uint8)
    for id in np.unique(sem_id):
        label = scenecraft_scannetpp_id2label[id]

        if label in scenecraft_scannetpp_to_ade20k_label_alias.keys():
            label = scenecraft_scannetpp_to_ade20k_label_alias[label]

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


semantic_processor = None
semantic_segmentor = None
@torch.no_grad()
def get_semantic_detail(image):
    global semantic_processor, semantic_segmentor
    if (semantic_processor is None) and (semantic_segmentor is None):
        semantic_processor = AutoImageProcessor.from_pretrained(HF_PATH("openmmlab/upernet-convnext-small"))
        semantic_segmentor = UperNetForSemanticSegmentation.from_pretrained(HF_PATH("openmmlab/upernet-convnext-small"))
        semantic_segmentor.to('cuda:0')

    pixel_values    = semantic_processor(image, return_tensors="pt").pixel_values
    segmentation    = semantic_segmentor(pixel_values.to('cuda:0'))
    segmentation    = semantic_processor.post_process_semantic_segmentation(segmentation, target_sizes=[image.size[::-1]])[0]
    segmentation    = segmentation.cpu().numpy()
    color_seg       = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8) # height, width, 3
    for label, color in enumerate(ade20k_id2color):
        color_seg[segmentation == label, :] = color
    color_seg       = color_seg.astype(np.uint8)
    color_seg       = Image.fromarray(color_seg)
    return color_seg


segformer_processor = None
segformer_model = None
@torch.no_grad()
def segformer_forward(image):
    global segformer_processor, segformer_model
    if (segformer_processor is None) or (segformer_model is None):
        segformer_processor = SegformerImageProcessor.from_pretrained(HF_PATH("nvidia/segformer-b0-finetuned-ade-512-512"))
        segformer_model     = SegformerForSemanticSegmentation.from_pretrained(HF_PATH("nvidia/segformer-b0-finetuned-ade-512-512"))
        segformer_model.to('cuda:0')

    pixel_values    = segformer_processor(image, return_tensors="pt").pixel_values
    segmentation    = segformer_model(pixel_values.to('cuda:0'))
    # segmentation    = segmentation.logits[0]
    
    segmentation    = segformer_processor.post_process_semantic_segmentation(segmentation, target_sizes=[image.size[::-1]])[0]
    # segmentation    = torch.argmax(segmentation, dim=0)
    segmentation    = segmentation.cpu().numpy()
    color_seg       = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8) # height, width, 3
    for label, color in enumerate(ade20k_id2color):
        color_seg[segmentation == label, :] = color
    color_seg       = color_seg.astype(np.uint8)
    color_seg       = Image.fromarray(color_seg)
    return color_seg


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
        dst_rgb_path                = osp.join(dst_root, 'rgb.jpg')
        dst_depth_path              = osp.join(dst_root, 'depth.png')
        dst_normal_path             = osp.join(dst_root, 'normal.png')
        dst_semantic_path           = osp.join(dst_root, 'semantic.png')
        dst_semantic_detail_path    = osp.join(dst_root, 'semantic_detail.png')

        rgb = batch['target']
        rgb.save(dst_rgb_path)

        depth = get_depth(rgb)
        depth.save(dst_depth_path)

        normal = get_normal(rgb)
        normal = normal.resize(rgb.size)
        normal.save(dst_normal_path)

        sem_id = np.array(batch['labels'])
        semantic = get_semantic(sem_id)
        semantic = Image.fromarray(semantic)
        semantic.save(dst_semantic_path)
        
        # semantic_detail = get_semantic_detail(rgb)
        # semantic_detail.save(dst_semantic_detail_path)

        # semantic_detail = segformer_forward(rgb)
        # semantic_detail.save(dst_semantic_detail_path)
        # segformer_forward(rgb)

        prompt = batch['prompt']

        item_dict = {
            'rgb':              osp.join('data', item_id, 'rgb.jpg'),
            'depth':            osp.join('data', item_id, 'depth.png'),
            'normal':           osp.join('data', item_id, 'normal.png'),
            'semantic':         osp.join('data', item_id, 'semantic.png'),
            # 'semantic_detail':  osp.join('data', item_id, 'semantic_detail.png'),
            'description':      prompt
        }

        meta.append(item_dict)

        # visualization
        # cv2.imshow("RGB", cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR))
        # cv2.imshow("Depth", np.array(depth))
        # cv2.imshow("Normal", cv2.cvtColor(np.array(normal), cv2.COLOR_RGB2BGR))
        # cv2.imshow("Semantic", cv2.cvtColor(np.array(semantic), cv2.COLOR_RGB2BGR))
        # cv2.imshow("Semantic Detail", cv2.cvtColor(np.array(semantic_detail), cv2.COLOR_RGB2BGR))
        # cv2.waitKey(10)

    # total_num_items = len(meta)
    # train_data, test_data = meta[:int(total_num_items * 0.8)], meta[int(total_num_items * 0.8):]
    # wirte_jsonl(train_data, osp.join(output_path, 'train.jsonl'))
    # wirte_jsonl(test_data, osp.join(output_path, 'test.jsonl'))

    print('DONE')