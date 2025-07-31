import os
import os.path as osp
import json
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix, translation_matrix

try:
    import cv2
except:
    from cv2 import cv2

# refer `https://github.com/sherwinbahmani/threed_front_rendering/blob/HEAD/scripts/create_camera_positions.py`
LABEL2ID_BEDROOMS = {
    'cabinet': 2,
    'chair': 3,
    'bed': 7,
    'shelf': 12,
    'sofa': 14,
    'table': 16,
    'television receiver': 17,
    'wardrobe': 18,
}

LABEL2ID_LIVING_ROOMS = {
    'bookcase': 1,
    'cabinet': 2,
    'chair': 4,
    'shelf': 14,
    'sofa': 16,
    'table': 18,
    'television receiver': 19,
}


def transform(p, mn, mx, res, s):
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
    
    p[...,0]    = (p[...,0] - minx) / extent   # (0,1)
    p[...,1]    = (p[...,1] - miny) / extent   # (0,1)
    p[...,2]    = (p[...,2] - minz) / extent
    p[...,:2]   = (p[...,:2] - 0.5) * s
    p[...,2]    = p[...,2] * s - 0.5

    x, y, z     = p[...,0], p[...,1], p[...,2]
    p           = np.stack([x,-z,y], axis=-1)

    p       = p * res
    scale   = s / extent * res
    return p, scale


def convert(layout_path, scene_name, scene_type, scene_scale, output_path):
    # load layout
    with open(layout_path, 'r') as f:
        layout = json.load(f)
        bbox, background = layout['bbox'], layout['background']

    bg_vertices         = np.array(background['vertices'])
    bg_ceiling_faces    = background['faces']['ceiling']
    mn, mx              = bg_vertices.min(axis=0), bg_vertices.max(axis=0)
    
    # 2D layout
    resolution = 256
    layout_topdown = np.zeros((1, resolution, resolution), dtype=np.float32)

    # draw wall
    bg_vertices, _ = transform(bg_vertices, mn, mx, resolution, scene_scale)
    for f in bg_ceiling_faces:
        mask = np.zeros((resolution, resolution), dtype=np.uint8)
        poly = bg_vertices[...,[0,2]][f].reshape((-1,1,2))
        poly = poly + resolution * 0.5
        poly = poly.astype(np.int32)
        cv2.fillPoly(mask, [poly], color=(255.))
        layout_topdown[:,mask!=0] = 1.0
    layout_topdown = layout_topdown.transpose([0,2,1])

    # cv2.imshow('layout topdown', layout_topdown[0])
    # cv2.waitKey()

    # draw othres
    class_labels, translations, sizes, angles = [], [], [], []

    if scene_type == 'bedrooms':
        label2id = LABEL2ID_BEDROOMS
    elif scene_type == 'living_rooms':
        label2id = LABEL2ID_LIVING_ROOMS
    else:
        raise NotImplementedError

    for b in bbox:
        b_class = b['class']
        if b_class not in label2id.keys():
            print(b_class, ' is not supported!')
            continue
        if scene_type == 'bedrooms':
            label = [0] * 21
        elif scene_type == 'living_rooms':
            label = [0] * 24
        else:
            raise NotImplementedError
        label[label2id[b_class]] = 1
        
        center, scale   = transform(np.array(b['location']), mn, mx, 1., scene_scale)
        size            = np.array([b['size'][0], b['size'][2], b['size'][1]]) * scale
        center          = center - size * 0.5

        class_labels.append(label)
        translations.append(center)
        sizes.append(size)
        angles.append(np.deg2rad(b['rotation'][2]))

    class_labels    = np.array(class_labels, dtype=np.float32)
    translations    = np.array(translations, dtype=np.float32)
    sizes           = np.array(sizes, dtype=np.float32)
    angles          = np.array(angles, dtype=np.float32)[:,None]

    # camera
    cameras_path = osp.join(osp.dirname(layout_path), 'cameras.json')
    assert osp.exists(cameras_path)
    camera_coords, target_coords = [], []
    with open(cameras_path, 'r') as f:
        cameras = json.load(f)
    for cam in cameras:
        camera_coords.append(cam['location'])

        elev = np.deg2rad(cam['rotation'][0]-90)
        azim = np.deg2rad(cam['rotation'][2]+90)

        lookat = [
            np.cos(elev) * np.cos(azim),
            np.cos(elev) * np.sin(azim),
            np.sin(elev)
        ]
        target_coord = [
            cam['location'][0] + lookat[0],
            cam['location'][1] + lookat[1],
            cam['location'][2] + lookat[2],
        ]
        target_coords.append(target_coord)

    camera_coords = np.array(camera_coords)
    target_coords = np.array(target_coords)
    camera_coords = transform(camera_coords, mn, mx, 1., scene_scale)[0]
    target_coords = transform(target_coords, mn, mx, 1., scene_scale)[0]
    camera_coords[...,1] *= -1.0
    target_coords[...,1] *= -1.0
    camera_coords = np.stack((camera_coords[...,2], camera_coords[...,1], camera_coords[...,0]), axis=-1)
    target_coords = np.stack((target_coords[...,2], target_coords[...,1], target_coords[...,0]), axis=-1)

    # images
    image_output_path = osp.join(output_path, 'images', scene_name)
    os.makedirs(image_output_path, exist_ok=True)
    fake_image = np.zeros((256,256,4), dtype=np.uint8)
    for i in range(camera_coords.shape[0]):
        cv2.imwrite(osp.join(image_output_path, f'{i:04d}.png'), fake_image)
        break

    # labels
    label_output_path = osp.join(output_path, 'labels', scene_name, 'boxes.npz')
    os.makedirs(osp.dirname(label_output_path), exist_ok=True)
    np.savez(
        label_output_path,
        class_labels=class_labels,
        translations=translations,
        sizes=sizes,
        angles=angles,
        room_layout=layout_topdown,
        camera_coords=camera_coords,
        target_coords=target_coords,
    )

    # print('class_labels ', class_labels.shape, class_labels.dtype, type(class_labels))
    # print('translations ', translations.shape, translations.dtype, type(translations))
    # print('sizes ', sizes.shape, sizes.dtype, type(sizes))
    # print('angles ', angles.shape, angles.dtype, type(angles))
    # print('room_layout ', layout_topdown.shape, layout_topdown.dtype, type(layout_topdown))
    # print('camera_coords ', camera_coords.shape, camera_coords.dtype, type(camera_coords))
    # print('target_coords ', target_coords.shape, target_coords.dtype, type(target_coords))


def create_transformed_box(location, size, angle_deg, axis='z'):
    box = trimesh.creation.box(extents=size)

    # rotation axis
    if isinstance(axis, str):
        axis_dict = {'x': [1,0,0], 'y': [0,1,0], 'z': [0,0,1]}
        axis_vec = axis_dict.get(axis.lower())
        assert axis_vec is not None
    else:
        axis_vec = axis

    # transformation
    R = rotation_matrix(np.deg2rad(angle_deg), direction=axis_vec)
    T = translation_matrix(location)
    transform = T @ R
    box.apply_transform(transform)

    return box


def create_camera_mesh(camera_origin, camera_target, axis_length=0.1):
    # lookat
    direction = camera_target - camera_origin
    direction /= np.linalg.norm(direction)

    # camera position
    sphere = trimesh.creation.icosphere(radius=0.02)
    sphere.apply_translation(camera_origin)

    # camera arrow
    # cam_arrow = trimesh.creation.arrow(
    #     origin=camera_origin,
    #     direction=direction,
    #     shaft_radius=0.005,
    #     head_radius=0.01,
    #     head_length=0.05
    # )

    # camera axis
    # cam_axis = trimesh.creation.axis(origin=camera_origin, axis_length=axis_length)

    # camera
    camera = trimesh.util.concatenate([
        sphere,
        # cam_arrow,
        # cam_axis
    ])

    return camera


def load_cc3d_layout(path='boxes.npz'):
    # load
    layout          = np.load(path)
    class_labels    = layout['class_labels']
    translations    = layout['translations']
    sizes           = layout['sizes']
    angles          = layout['angles']
    room_layout     = layout['room_layout']
    camera_coords   = layout['camera_coords']
    target_coords   = layout['target_coords']

    # print('class_labels', class_labels.shape, class_labels.dtype, type(class_labels))
    # print('translations', translations.shape, translations.dtype, type(translations))
    # print('sizes', sizes.shape, sizes.dtype, type(sizes))
    # print('angles', angles.shape, angles.dtype, type(angles))
    # print('room_layout ', room_layout.shape, room_layout.dtype, type(room_layout))
    # print('camera_coords ', camera_coords.shape, camera_coords.dtype, type(camera_coords))
    # print('target_coords ', target_coords.shape, target_coords.dtype, type(target_coords))

    # layout
    layout_meshes = []
    for loc, size, angle in zip(translations, sizes, angles):
        angle_deg   = np.rad2deg(angle[0])
        loc         = loc + size * 0.5
        loc[1]      *= -1.0
        box_mesh    = create_transformed_box(loc, size, angle_deg, 'y')
        layout_meshes.append(box_mesh)
    trimesh.util.concatenate(layout_meshes).export('layout.ply')

    # camera
    cam_pos_list, cam_tgt_list = [], []
    n_cam = camera_coords.shape[0]
    for i, (cam_pos, cam_tgt) in enumerate(zip(camera_coords, target_coords)):
        pos = trimesh.creation.icosphere(radius=0.01)
        pos.visual.vertex_colors = [int(255 * i / n_cam), 255, 255, 255]  # RGBA
        pos.apply_translation(cam_pos)
        cam_pos_list.append(pos)

        tgt = trimesh.creation.icosphere(radius=0.01)
        tgt.visual.vertex_colors = [int(255 * i / n_cam), 255, 255, 255]  # RGBA
        tgt.apply_translation(cam_tgt)
        cam_tgt_list.append(tgt)

    trimesh.util.concatenate(cam_pos_list).export('camera_position.ply')
    trimesh.util.concatenate(cam_tgt_list).export('camera_target.ply')

    # objects
    resolution      = 256
    layout_topdown  = np.zeros((resolution, resolution), dtype=np.float32)
    for loc, sz, angle in zip(translations, sizes, angles):
        angle_deg   = np.rad2deg(angle[0])
        loc         = loc + sz * 0.5

        center      = (float((loc[0] + 0.5) * resolution), float((loc[2] + 0.5) * resolution))
        size        = (int(sz[0] * resolution), int(sz[2] * resolution))
        box         = cv2.boxPoints((center, size, angle_deg))
        box         = np.int32(box)

        mask = np.zeros((resolution, resolution), dtype=np.uint8)
        cv2.fillPoly(mask, [box], color=(255.))
        layout_topdown[mask!=0] = 1.0

    cv2.imshow('room layout', room_layout[0])
    cv2.imshow('layout topdown', layout_topdown)
    cv2.waitKey(10)


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
        output_path = f'layout_cc3d_format/{scene_type}/{scene_name}'
        scene_scale = 0.8

        print(scene_name)
        convert(layout_path, scene_name, scene_type, scene_scale, output_path)

    # debug
    # load_cc3d_layout()
    # load_cc3d_layout('layout_cc3d_format/living_rooms/setthescene_living_room/labels/setthescene_living_room/boxes.npz')