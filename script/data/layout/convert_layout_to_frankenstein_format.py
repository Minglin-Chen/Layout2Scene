import os
import os.path as osp
import json
import numpy as np

try:
    import cv2
except:
    from cv2 import cv2


def transform(p, mn, mx, res, s=0.8):
    minx, miny = mn
    maxx, maxy = mx

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
    p = 1. - p
    p = (p - 0.5) * s + 0.5

    p       = (p * res).astype(np.int32)
    scale   = s / extent * res
    return p, scale


def convert(layout_path, scene_type, scene_scale, output_path, vis=True):

    if scene_type == 'bedroom':
        resolution, n_label = 40, 3
        label2sem = {
            'wall':     (1.0, 0.0, 0.0),
            'bed':      (0.0, 1.0, 0.0),
            'cabinet':  (0.0, 0.0, 1.0),
        }
        alias = {
            'table':    'cabinet',
            'wardrobe': 'cabinet',
        }
    elif scene_type == 'livingroom':
        resolution, n_label = 56, 5
        label2sem = {
            'wall':     (1.0, 0.0, 0.0, 0.0, 0.0),
            'cabinet':  (0.0, 1.0, 0.0, 0.0, 0.0),
            'chair':    (0.0, 0.0, 1.0, 0.0, 0.0),
            'sofa':     (0.0, 0.0, 0.0, 1.0, 0.0),
            'table':    (0.0, 0.0, 0.0, 0.0, 1.0),
        }
        alias = {
            'shelf':    'cabinet',
            'counter':  'cabinet',
            'bookcase': 'cabinet'
        }
    else:
        raise NotImplementedError

    # load layout
    with open(layout_path, 'r') as f:
        layout = json.load(f)
        bbox, background = layout['bbox'], layout['background']

    bg_vertices         = np.array(background['vertices'])[...,:2]
    bg_ceiling_faces    = background['faces']['ceiling']
    minxy, maxxy        = bg_vertices.min(axis=0), bg_vertices.max(axis=0)

    # 2D layout
    layout_topdown = np.zeros((resolution, resolution, n_label), dtype=np.float32)

    # draw wall
    bg_vertices, _ = transform(bg_vertices, minxy, maxxy, resolution, scene_scale)
    for f in bg_ceiling_faces:
        mask = np.zeros((resolution, resolution), dtype=np.uint8)
        cv2.fillPoly(mask, [bg_vertices[f].reshape((-1,1,2))], color=(255.))
        layout_topdown[mask!=0] = np.array(label2sem['wall'])

    # draw othres
    for b in bbox:
        b_class = b['class']
        if b_class in alias.keys(): b_class = alias[b_class]
        if b_class not in label2sem.keys():
            print(b_class, ' is not supported!')
            continue
        
        center, scale   = transform(np.array(b['location'][:2]), minxy, maxxy, resolution, scene_scale)
        center          = (float(center[0]), float(center[1]))
        size            = (int(b['size'][0] * scale), int(b['size'][1] * scale))
        angle           = - b['rotation'][2]
        box             = cv2.boxPoints((center, size, angle))
        box             = np.int32(box)

        mask = np.zeros((resolution, resolution), dtype=np.uint8)
        cv2.fillPoly(mask, [box], color=(255.))
        layout_topdown[mask!=0] = np.array(label2sem[b_class])

    if vis:
        if scene_type == 'bedroom':
            cv2.imshow('layout_topdown', cv2.resize(layout_topdown, dsize=None, fx=10., fy=10.))
        elif scene_type == 'livingroom':
            cv2.imshow('layout_topdown 0', cv2.resize(layout_topdown[...,[0,1,2]], dsize=None, fx=10., fy=10.))
            cv2.imshow('layout_topdown 1', cv2.resize(layout_topdown[...,[0,3,4]], dsize=None, fx=10., fy=10.))
        else:
            raise ValueError
        
        cv2.waitKey()
    
    layout_maps = np.stack([layout_topdown])

    os.makedirs(output_path, exist_ok=True)
    np.save(osp.join(output_path, 'layout_maps.npy'), layout_maps)


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
        layout_path = f'../../../data/layout/{scene_name}/layout.json'
        output_path = f'layout_frankenstein_format/{scene_type}/{scene_name}'
        scene_scale = 0.6

        print(scene_name)
        convert(layout_path, scene_type, scene_scale, output_path, vis=True)
    