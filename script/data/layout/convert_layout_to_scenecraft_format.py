import os
import os.path as osp
import numpy as np
import json
import pymeshlab
import trimesh
from collections import defaultdict
import math


nyu40id = [
    'unlabeled',
    'wall',
    'floor',
    'cabinet',
    'bed',
    'chair',
    'sofa',
    'table',
    'door',
    'window',
    'bookshelf',
    'picture',
    'counter',
    'blinds',
    'desk',
    'shelves',
    'curtain',
    'dresser',
    'pillow',
    'mirror',
    'floormat',
    'clothes',
    'ceiling',
    'books',
    'refrigerator',
    'television',
    'paper',
    'towel',
    'showercurtain',
    'box',
    'whiteboard',
    'person',
    'nightstand',
    'toilet',
    'sink',
    'lamp',
    'bathtub',
    'bag',
    'otherstructure',
    'otherfurniture',
    'otherprop',
]


ade20k_to_nyu40_label_alias = {
    'windowpane':           'window',
    'shelf':                'cabinet',
    'wardrobe':             'cabinet',
    'painting':             'picture',
    'television receiver':  'television',
    'bookcase':             'bookshelf',
    'book':                 'books',
    'rug':                  'otherprop',
    'car':                  'unlabeled',
    'barrel':               'unlabeled',
    'chest of drawers':     'cabinet',
}


def face_normal(vertices, face):
    v0, v1, v2 = vertices[face][:3]
    return np.cross(v1 - v0, v2 - v0)

def triangles_to_quads(vertices, faces, angle_threshold=1.0):
    edge_to_faces = defaultdict(list)
    for i, face in enumerate(faces):
        for e in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
            edge = tuple(sorted(e))
            edge_to_faces[edge].append(i)

    visited = set()
    quads = []
    tris_left = []

    cos_thresh = np.cos(np.radians(angle_threshold))

    for i, face in enumerate(faces):
        if i in visited: continue
        merged = False
        n1 = face_normal(vertices, face)
        n1 = n1 / np.linalg.norm(n1)

        for e in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
            edge = tuple(sorted(e))
            if len(edge_to_faces[edge]) == 2:
                f2 = edge_to_faces[edge][1] if edge_to_faces[edge][0] == i else edge_to_faces[edge][0]
                if f2 not in visited:
                    n2 = face_normal(vertices, faces[f2])
                    n2 = n2 / np.linalg.norm(n2)
                    if np.dot(n1, n2) > cos_thresh:
                        verts = list(set(faces[i]) | set(faces[f2]))
                        if len(verts) == 4:
                            quads.append(verts)
                            visited.add(i)
                            visited.add(f2)
                            merged = True
                            break
        if not merged:
            tris_left.append(face.tolist())

    return np.array(quads), np.array(tris_left)

def quad_to_box_xywall(vertices, quad):
    verts = np.array([vertices[i] for i in quad])

    center = verts.mean(axis=0)

    verts_xy = verts[:, :2]

    edges = np.roll(verts_xy, -1, axis=0) - verts_xy
    edge_lengths = np.linalg.norm(edges, axis=1)
    longest_edge_idx = np.argmax(edge_lengths)
    long_vec = edges[longest_edge_idx]

    yaw = np.arctan2(long_vec[1], long_vec[0])
    yaw_deg = np.rad2deg(yaw)

    length = edge_lengths[longest_edge_idx]
    width = 0.01

    height = verts[:, 2].max() - verts[:, 2].min()

    return center, yaw_deg, (width, length, height)

def convert_background(background):
    bg_vertices = np.array(background['vertices'])
    mn, mx      = bg_vertices.min(axis=0), bg_vertices.max(axis=0)

    wall_faces  = np.array(background['faces']['walls'])
    wall_quad, wall_tri = triangles_to_quads(bg_vertices, wall_faces)
    # assert len(wall_tri) == 0

    bboxes, labels, thetas = [], [], []

    for wall_quad in wall_quad:
        wall_pos, wall_rot, wall_size = \
            quad_to_box_xywall(bg_vertices, wall_quad)
        wall_pos    = wall_pos.tolist()
        wall_size   = list(wall_size)

        wall_rot    = float(wall_rot)
        wall_rot    = abs(wall_rot)
        assert wall_rot < 360.
        if abs(wall_rot - 90) < 1.:
            wall_size = [wall_size[1], wall_size[0], wall_size[2]]
        elif abs(wall_rot - 270) < 1.:
            wall_size = [wall_size[1], wall_size[0], wall_size[2]]

        bboxes.append(wall_pos + wall_size)
        labels.append(f"{nyu40id.index('wall')}")
        # thetas.append(wall_rot)
        thetas.append(0.0)

    ceiling_pos = [(mx[0]+mn[0])*0.5, (mx[1]+mn[1])*0.5, mx[2]+0.005]
    floor_pos   = [(mx[0]+mn[0])*0.5, (mx[1]+mn[1])*0.5, mn[2]-0.005]
    ceiling_size = floor_size = [mx[1]-mn[1], mx[0]-mn[0], 0.01]

    bboxes.append(ceiling_pos + ceiling_size)
    labels.append(f"{nyu40id.index('ceiling')}")
    thetas.append(0.0)

    bboxes.append(floor_pos + floor_size)
    labels.append(f"{nyu40id.index('floor')}")
    thetas.append(0.0)

    return bboxes, labels, thetas

normalize = lambda v: v / np.linalg.norm(v)

def get_camera_to_world(loc, euler_deg):
    elevation   = math.radians(euler_deg[0] - 90)
    azimuth     = math.radians(euler_deg[2] + 90)

    up = np.array([0., 0., 1.])
    lookat = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation)
    ])
    right = normalize(np.cross(lookat, up))
    up = normalize(np.cross(right, lookat))
    c2w = np.concatenate(
        [np.stack([right, up, -lookat], axis=-1), np.array(loc)[:,None]],
        axis=-1
    )
    c2w = np.concatenate([c2w, np.zeros_like(c2w[:1])], axis=0)
    c2w[3,3] = 1.

    return c2w

def convert(layout_path, scene_type, scene_name, output_path):
    camera_path = osp.join(osp.dirname(layout_path), 'cameras.json')
    assert osp.exists(camera_path)

    # load layout
    with open(layout_path, 'r') as f:
        layout = json.load(f)
        bbox, background = layout['bbox'], layout['background']

    bboxes, labels, thetas = convert_background(background)

    for b in bbox:
        theta = abs(float(b['rotation'][2]))
        assert theta < 360.

        loc     = b['location']
        size    = b['size']
        if abs(theta) < 1.:
            size = [size[1], size[0], size[2]]
        elif abs(theta - 180) < 1.:
            size = [size[1], size[0], size[2]]

        bboxes.insert(0, loc+size)

        b_class = b['class']
        if b_class in ade20k_to_nyu40_label_alias.keys():
            b_class = ade20k_to_nyu40_label_alias[b_class]
        assert b_class in nyu40id, b_class

        labels.insert(0, f"{nyu40id.index(b_class)}")

        # thetas.insert(0, theta)
        thetas.insert(0, 0.0)

    layout_output = {
        'bboxes': bboxes,
        'labels': labels,
        'thetas': thetas,
    }

    os.makedirs(output_path, exist_ok=True)
    with open(osp.join(output_path, 'layout.json'), 'w') as f:
        json.dump(layout_output, f, indent=2)

    # load camera
    with open(camera_path, 'r') as f:
        cameras = json.load(f)
    
    n_cameras = len(cameras)

    keyframes = []
    camera_path = []
    for i_cam, camera in enumerate(cameras):
        c2w = get_camera_to_world(camera['location'], camera['rotation'])

        # if i_cam % 10 == 0:
        keyframes.append({
            "matrix": f"{c2w.transpose(1,0).reshape(-1).tolist()}",
            "fov": 55,
            "aspect": 1,
            "properties": f'[[\"FOV\",{55}],[\"NAME\",\"Camera {i_cam}\"],[\"TIME\",{i_cam/n_cameras}]]'
        })

        camera_path.append({
            'camera_to_world': c2w.reshape(-1).tolist(),
            'fov': 55,
            'aspect': 1
        })

    camera_json = {
        'keyframes': keyframes,
        'camera_type': "perspective",
        "render_height": 512,
        "render_width": 512,
        'camera_path': camera_path,
        "fps": 10,
        "seconds": n_cameras//10,
        "smoothness_value": 0.0,
        "is_cycle": False,
        "crop": None
    }

    os.makedirs(output_path, exist_ok=True)
    with open(osp.join(output_path, 'cameras.json'), 'w') as f:
        json.dump(camera_json, f, indent=2)


if __name__=='__main__':
    # configuration
    # error in 'livingroom_8013', 'livingroom_8016'
    scene_names = [
        'setthescene_bedroom', 'setthescene_dining_room', 'setthescene_garage', 'setthescene_living_room',
        'hypersim_ai_001_001', 'hypersim_ai_001_003', 'hypersim_ai_001_005', 'hypersim_ai_003_004', 'hypersim_ai_006_010', 'hypersim_ai_010_005', 'hypersim_ai_010_008', 'hypersim_ai_022_005', \
        'bedroom_0000', 'bedroom_0001', 'bedroom_0002', 'bedroom_0003', 'bedroom_0004', 'livingroom_8017', \
        'fankenstein_bedroom_001',
    ]
    scene_types = [
        'bedroom', 'dining room', 'garage', 'living room', \
        'bathroom', 'office', 'dining room', 'bedroom', 'dining room', 'bedroom', 'living room', 'living room', \
        'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'living room', \
        'bedroom',
    ]

    for scene_name, scene_type in zip(scene_names, scene_types):
        layout_path = f'../../../data/layout/{scene_name}/layout.json'
        output_path = f'../../../relatedworks/SceneCraft/data/custom/{scene_name}'

        print(scene_name)
        convert(layout_path, scene_name, scene_type, output_path)

    print('DONE')