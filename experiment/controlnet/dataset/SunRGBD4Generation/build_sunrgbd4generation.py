import os
import os.path as osp
import shutil
from glob import glob
from tqdm import tqdm
import json
import numpy as np
import scipy
from PIL import Image

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel

from ade20k_protocol import ade20k_label2color

HF_ROOT = os.environ.get('HF_ROOT')
HF_PATH = lambda p: p if HF_ROOT is None else osp.join(HF_ROOT, p)


def gather_sunrgbd_paths(root):
    paths = [
        osp.dirname(p) for p in \
            glob(osp.join(root, '*/*/*', 'seg.mat')) + \
            glob(osp.join(root, '*/*/*/*/*', 'seg.mat'))]
    
    rgb_paths, depth_paths, seg_paths = [], [], []
    for p in paths:
        rgb_path    = glob(osp.join(p, 'image', '*.jpg'))[0]
        depth_path  = glob(osp.join(p, 'depth_bfx', '*.png'))[0]
        seg_path    = osp.join(p, 'seg.mat')

        rgb_paths.append(rgb_path)
        depth_paths.append(depth_path)
        seg_paths.append(seg_path)

    return rgb_paths, depth_paths, seg_paths


def parse_mat_file(path):
    segmentation_data = scipy.io.loadmat(path)
    assert 'seglabel' in segmentation_data.keys()
    labels = segmentation_data['seglabel']
    assert 'names' in segmentation_data.keys()
    names = segmentation_data['names']
    return labels, names


clip_model, clip_processor = None, None
word_candidate_embeds = None
def find_similar_word(word, pretrained_model_name_or_path='openai/clip-vit-large-patch14'):
    global clip_model, clip_processor, word_candidate_embeds

    def _get_word_embedding(word):
        assert (clip_model is not None) and (clip_processor is not None) 
        # Process the input word
        inputs = clip_processor(text=word, return_tensors="pt", padding=True, truncation=True)
        
        # Get the word embeddings from the model
        with torch.no_grad():
            outputs = clip_model.get_text_features(**inputs)
            
        return outputs
    
    def _cosine_similarity(a, b):
        # Calculate cosine similarity
        a = a / a.norm(dim=-1, keepdim=True)
        b = b / b.norm(dim=-1, keepdim=True)
        return (a @ b.T).squeeze(dim=1)


    if (clip_model is None) or (clip_processor is None):
        clip_model      = CLIPModel.from_pretrained(pretrained_model_name_or_path)
        clip_processor  = CLIPProcessor.from_pretrained(pretrained_model_name_or_path)
        # clip_model.to(device)

    assert (clip_model is not None) and (clip_processor is not None)
    if word_candidate_embeds is None:
        candidates              = ade20k_label2color.keys()
        candidate_embeds        = [_get_word_embedding(w) for w in tqdm(candidates)]
        word_candidate_embeds   = torch.cat(candidate_embeds, dim=0)

    w_embed     = _get_word_embedding(word)
    similarity  = _cosine_similarity(word_candidate_embeds, w_embed)
    index       = torch.argmax(similarity)
    ret_word    = list(ade20k_label2color.keys())[index]

    return ret_word


category_mapping = {}
def get_semantic(labels, names, clip_pretrained_model_name_or_path='openai/clip-vit-large-patch14'):
    height, width = labels.shape[:2]
    semantic = np.zeros((height, width, 3), dtype=np.uint8)
    for i, name in enumerate(names[0]):
        name = name[0]

        if name not in ade20k_label2color.keys():
            if name not in category_mapping.keys():
                # handle missing category name
                category_mapping[name] = find_similar_word(
                    name, clip_pretrained_model_name_or_path)
            name = category_mapping[name]

        assert name in ade20k_label2color.keys()
        c = ade20k_label2color[name]
        semantic[labels==i+1,:] = c
        
    return semantic


blip2_processor, blip2_model = None, None
def blip2_forward(image, prompt=None, pretrained_model_name_or_path='Salesforce/blip2-opt-2.7b', device='cuda'):
    # prompt format: "Question: ...? Answer:"
    global blip2_processor, blip2_model
    if (blip2_processor is None) or (blip2_model is None):
        blip2_processor = Blip2Processor.from_pretrained(pretrained_model_name_or_path)
        blip2_model = Blip2ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16)
        blip2_model.to(device)

    assert (blip2_processor is not None) and (blip2_model is not None)

    inputs = blip2_processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
    generated_ids = blip2_model.generate(**inputs)
    generated_text = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text


stable_normal_predictor = None
def stable_normal_forward(image, weight_path, device='cuda'):
    global stable_normal_predictor
    if stable_normal_predictor is None:
        stable_normal_predictor = torch.hub.load(
            "Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True, local_cache_dir=weight_path)
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
    BLIP2_PATH          = HF_PATH('Salesforce/blip2-opt-2.7b')
    STABLE_NORMAL_PATH  = HF_PATH('Stable-X/stable-normal-v0-1')
    CLIP_PATH           = HF_PATH('openai/clip-vit-large-patch14')

    # configuration
    sunrgbd_root        = '/home/mlchen/data/SunRGBD/SUNRGBD/'
    sunrgbd4gen_root    = '/home/mlchen/data/SunRGBD/SUNRGBD4GEN/'

    # load all paths
    rgb_paths, depth_paths, seg_paths = gather_sunrgbd_paths(sunrgbd_root)
    total_num_items = len(rgb_paths)

    meta = []
    for i, (rgb_path, depth_path, seg_path) in enumerate(zip(rgb_paths, depth_paths, seg_paths)):
        print(f'[{i:05d} | {total_num_items}] {osp.dirname(rgb_path)}')

        item_id = f'{i:05d}'
        dst_root = osp.join(sunrgbd4gen_root, 'data', item_id)
        # assert not osp.exists(dst_root)
        if not osp.exists(dst_root): os.makedirs(dst_root)
        dst_rgb_path        = osp.join(dst_root, 'rgb.jpg')
        dst_depth_path      = osp.join(dst_root, 'depth.png')
        dst_normal_path     = osp.join(dst_root, 'normal.png')
        dst_semantic_path   = osp.join(dst_root, 'semantic.png')

        shutil.copy2(rgb_path, dst_rgb_path)
        shutil.copy2(depth_path, dst_depth_path)

        image = Image.open(rgb_path)

        normal = stable_normal_forward(image, STABLE_NORMAL_PATH, device='cuda:0')
        normal.save(dst_normal_path)

        labels, names = parse_mat_file(seg_path)
        semantic = get_semantic(labels, names, CLIP_PATH)
        Image.fromarray(semantic).save(dst_semantic_path)
        
        description = blip2_forward(
            image, 
            pretrained_model_name_or_path=BLIP2_PATH,
            device='cuda:1')

        room_type = blip2_forward(
            image, 
            'Question: What is the room type in the picture? Answer:', 
            pretrained_model_name_or_path=BLIP2_PATH,
            device='cuda:1')

        item_dict = {
            'rgb':          osp.join('data', item_id, 'rgb.jpg'),
            'depth':        osp.join('data', item_id, 'depth.png'),
            'normal':       osp.join('data', item_id, 'normal.png'),
            'semantic':     osp.join('data', item_id, 'semantic.png'),
            'description':  description,
            'type':         room_type,
        }

        meta.append(item_dict)

        # # debug
        # if i > 0: break

    with open(osp.join(sunrgbd4gen_root, 'category_mapping.json'), 'w') as f:
        json.dump(category_mapping, f)

    train_data, test_data = meta[:10000], meta[10000:]
    wirte_jsonl(train_data, 'train.jsonl')
    wirte_jsonl(test_data, 'test.jsonl')

    print('DONE')