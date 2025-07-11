import os
import os.path as osp
import numpy as np
from glob import glob
from omegaconf import OmegaConf
import trimesh
import json

import sys
sys.path.append('../../../')
from core.utils.ade20k_protocol import ade20k_id2label

setthescene_to_ade20k_label_alias = {
    'nightstand':       'table', 
    'fuel_barrel':      'barrel',
    'chest':            'chest of drawers',
    'sofa_for_one':     'sofa',
    'sofa_for_three':   'sofa',
}


if __name__=='__main__':
    setthescene_data_root = '../../../data/setthescene'
    config_paths = glob(osp.join(setthescene_data_root, "demo_configs/*.yaml"))

    for cfg_path in config_paths:
        cfg = OmegaConf.load(cfg_path)

        bbox_data, background_data = [], []
        for proxy_name, proxy_center, proxy_rotation in zip(
            cfg.scene_proxies.proxy_to_nerf_ids, 
            cfg.scene_proxies.proxy_centers, 
            cfg.scene_proxies.proxy_rotations_clockwise, 
        ):
            index       = cfg.scene_nerfs.nerf_ids.index(proxy_name)
            shape_scale = cfg.scene_nerfs.shape_scales[index]
            shape_path  = cfg.scene_nerfs.shape_paths[index]

            shape = trimesh.load(osp.join(setthescene_data_root, shape_path))
            
            shape.apply_translation(-shape.bounding_box.centroid)

            # scale_matrix = np.eye(4)
            # scale_matrix[:3, :3] *= shape_scale
            # shape.apply_transform(scale_matrix)
            
            shape.apply_translation(proxy_center)
            
            rotation_matrix = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
            shape.apply_transform(rotation_matrix)

            if proxy_name == 'room':
                pass
            else:
                bbox    = shape.bounds

                loc     = (bbox[0] + bbox[1]) / 2
                size    = bbox[1] - bbox[0]

                if proxy_name in setthescene_to_ade20k_label_alias.keys():
                    proxy_name = setthescene_to_ade20k_label_alias[proxy_name]
                assert proxy_name in list(ade20k_id2label.values()), proxy_name

                bbox_data.append({
                    "class": proxy_name,
                    "prompt": proxy_name,
                    "location": [loc[0], -loc[1], loc[2]],
                    "size": [size[1], size[0], size[2]],
                    "rotation": [0, 0, -proxy_rotation+90]
                })

        # output
        output_path = osp.join(
            setthescene_data_root, 
            osp.basename(cfg_path).split('.')[0], 
            'layout.json'
        )
        os.makedirs(osp.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({"bbox": bbox_data, "background": background_data}, f, indent=4)