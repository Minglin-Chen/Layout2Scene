import os
import os.path as osp
import numpy as np
import yaml

import trimesh
import json


if __name__=='__main__':
    # configuration
    # - bedroom: 
    #   hypersim_ai_010_005, setthescene_bedroom
    # - livingroom: 
    #   hypersim_ai_001_005, hypersim_ai_006_010, hypersim_ai_010_008, hypersim_ai_022_005, 
    #   setthescene_dining_room, setthescene_living_room

    scene_names = [\
        'hypersim_ai_010_005', 'setthescene_bedroom', \
        'hypersim_ai_001_005', 'hypersim_ai_006_010', 'hypersim_ai_010_008', 'hypersim_ai_022_005', \
        'setthescene_dining_room', 'setthescene_living_room', \
    ]
    scene_types = [
        'bedrooms', 'bedrooms',
        'living_rooms', 'living_rooms', 'living_rooms', 'living_rooms', \
        'living_rooms', 'living_rooms'
    ]

    for scene_name, scene_type in zip(scene_names, scene_types):
        layout_path = f'../../../data/layout/{scene_name}/layout.json'
        output_path = f'layout_setthescene_format/{scene_name}'
        os.makedirs(output_path, exist_ok=True)

        print(scene_name)

        if scene_type == 'bedrooms':
            scene_prompt = 'a modern style bedroom'
        elif scene_type == 'living_rooms':
            scene_prompt = 'a modern style living room'
        else:
            raise NotImplementedError

        # load layout
        with open(layout_path, 'r') as f:
            layout = json.load(f)
            bbox, background = layout['bbox'], layout['background']

        proxy_to_nerf_ids, proxy_centers, proxy_rotations_clockwise = [], [], []
        nerf_ids, shape_scales, shape_paths, nerf_texts = [], [], [], []
        # objects
        obj_count = {}
        for b in bbox:
            b_class = b['class']
            if b_class not in obj_count.keys():
                b_name = b_class
                obj_count[b_class] = 1
            else:
                b_name = b_class + f"_{obj_count[b_class]}"
                obj_count[b_class] += 1

            shape = trimesh.creation.box(extents=[b['size'][1], b['size'][2], b['size'][0]])
            shape_path = osp.join(output_path, b_name+'.obj')
            shape.export(shape_path)

            proxy_to_nerf_ids.append(b_name)
            proxy_centers.append([b['location'][0], b['location'][2], b['location'][1]])
            proxy_rotations_clockwise.append(-b['rotation'][2]+90)

            nerf_ids.append(b_name)
            shape_scales.append(max(b['size']))
            shape_paths.append(shape_path)
            nerf_texts.append(b['prompt'])

        # room
        room_mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=np.array(background['vertices'])[...,[1,2,0]], faces=np.array(background['faces']['ceiling'])),
            trimesh.Trimesh(vertices=np.array(background['vertices'])[...,[1,2,0]], faces=np.array(background['faces']['floor'])),
            trimesh.Trimesh(vertices=np.array(background['vertices'])[...,[1,2,0]], faces=np.array(background['faces']['walls'])),
        ])
        room_bbox       = room_mesh.bounds
        room_position   = (room_bbox[0] + room_bbox[1]) / 2
        room_size       = room_bbox[1] - room_bbox[0]
        room_mesh.apply_translation(-room_position)
        room_path = osp.join(output_path, 'room.obj')
        room_mesh.export(room_path)

        proxy_to_nerf_ids.append('room')
        proxy_centers.append([float(room_position[2]), float(room_position[1]), float(room_position[0])])
        proxy_rotations_clockwise.append(90)

        nerf_ids.append('room')
        shape_scales.append(max(room_size.tolist()))
        shape_paths.append(room_path)
        nerf_texts.append('walls of a room')

        # save
        config_dict = {
            'log': { 
                'exp_name': scene_name, 
                'eval_only': False 
            },
            'guide': { 
                'text': scene_prompt 
            },
            'scene_proxies': {
                'proxy_to_nerf_ids': proxy_to_nerf_ids,
                'proxy_centers': proxy_centers,
                'proxy_rotations_clockwise': proxy_rotations_clockwise,
            },
            'scene_nerfs': {
                'nerf_ids': nerf_ids,
                'shape_scales': shape_scales,
                'shape_paths': shape_paths,
                'nerf_texts': nerf_texts,
            },
            'optim': {
                'iters': 15000,
                'device': 'cuda:0'
            }
        }

        config_path = osp.join(output_path, 'config.yaml')
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, allow_unicode=True)

    print('DONE')