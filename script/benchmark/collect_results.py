
import os
import os.path as osp
from glob import glob
import shutil

if __name__=='__main__':

    outputs_dir = '../../outputs'
    exp_names = [
        'scene_texture_generation_Bohemian',
        'scene_texture_generation_Cartoon',
        'scene_texture_generation_Chinese',
        'scene_texture_generation_industrial',
        'scene_texture_generation_Midcentury',
        'scene_texture_generation_modern',
        'scene_texture_generation_Scandinavian'
    ]

    collected_dir = osp.join(outputs_dir, 'all_models')

    in_dirs = []
    for exp_name in exp_names:
        in_dirs += list(glob(osp.join(outputs_dir, exp_name, '*', 'save', '*-export')))
    
    for in_dir in in_dirs:
        scene_id = osp.basename(osp.dirname(osp.dirname(in_dir)))
        print(scene_id)

        out_dir = osp.join(collected_dir, scene_id)
        os.makedirs(out_dir, exist_ok=True)

        shutil.copytree(in_dir, out_dir, dirs_exist_ok=True)

    print(f'num: {len(in_dirs)}')

    print('DONW')