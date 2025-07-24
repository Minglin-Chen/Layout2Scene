import os
import os.path as osp
import json
import numpy as np
import yaml


def write_config(template_path, output_path, scene_prompt, prompt, center, edge, ori, scene_name):
    with open(template_path, 'r') as f:
        data = yaml.safe_load(f)

    data['scene']       = scene_prompt
    data['prompt']      = prompt
    data['center']      = center
    data['edge']        = edge
    data['ori']         = ori
    data['outdir']      = scene_name
    data['save_path']   = scene_name

    os.makedirs(osp.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)


def transform(p, mn, mx):
    minx, miny, minz = mn
    maxx, maxy, maxz = mx

    extentx = maxx - minx
    extenty = maxy - miny
    if extentx > extenty:
        maxy += (extentx - extenty) * 0.5
        miny -= (extentx - extenty) * 0.5
    else:
        maxx += (extenty - extentx) * 0.5
        minx -= (extenty - extentx) * 0.5
    assert (maxx - minx) == (maxy - miny)
    extent = maxx - minx
    
    p[...,0] = (p[...,0] - minx) / extent   # (0,1)
    p[...,1] = (p[...,1] - miny) / extent   # (0,1)
    p[...,2] = (p[...,2] - minz) / extent
    p[...,:2] = (p[...,:2] - 0.5) * 2.
    p[...,2] = p[...,2] * 2.

    scale = 2. / extent 
    return p, scale


def convert(layout_path, scene_type, scene_name, template_path, output_path):
    # load layout
    with open(layout_path, 'r') as f:
        layout = json.load(f)
        bbox, background = layout['bbox'], layout['background']

    bg_vertices = np.array(background['vertices'])
    mn, mx      = bg_vertices.min(axis=0), bg_vertices.max(axis=0)

    scene_prompt = 'A ' + scene_type
    prompt, center, edge, ori = [], [], [], []

    for b in bbox:
        b_class = b['class']
        prompt.append('A ' + b_class)
        
        location, scale = transform(np.array(b['location']), mn, mx)
        center.append([
            float(location[0]), 
            float(location[2]), 
            float(-location[1])
        ])

        edge.append([
            float(b['size'][0] * scale), 
            float(b['size'][2] * scale), 
            float(b['size'][1] * scale)
        ])

        ori.append(float(b['rotation'][2]))

    write_config(template_path, output_path, scene_prompt, prompt, center, edge, ori, scene_name)


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
        'bedroom', 'bedroom',
        'livingroom', 'livingroom', 'livingroom', 'livingroom', \
        'livingroom', 'livingroom'
    ]

    for scene_name, scene_type in zip(scene_names, scene_types):
        layout_path     = f'../../../data/layout/{scene_name}/layout.json'
        output_path     = f'layout_gala3d_format/{scene_name}.yaml'
        template_path   = 'template/gala3d_template.yaml'

        print(scene_name)
        convert(layout_path, scene_type, scene_name, template_path, output_path)
    