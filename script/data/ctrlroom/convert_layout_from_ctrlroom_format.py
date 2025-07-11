import os
import os.path as osp
from glob import glob
import math
import json
from scipy.spatial import Delaunay

import sys
sys.path.append('../../../')
from core.utils.ade20k_protocol import ade20k_label2color


CLASS_MAPPING = {
    'window':       'windowpane',
    'night stand':  'table',
    'picture':      'painting',
    'shelves':      'shelf',
    'dresser':      'wardrobe',
    'fridge':       'refrigerator',
    'television':   'television receiver',
}


class CtrlRoomLayout:

    def __init__(self, path, index=0):

        room_paths = sorted(glob(osp.join(path, '*', '*', 'img', '*')))
        assert index >= 0 and index < len(room_paths), len(room_paths)
        print(f'The {index}-th room of {len(room_paths)} rooms')

        room_path   = room_paths[index]
        room_id     = osp.basename(room_path).split('.')[0]
        room_type   = osp.basename(osp.dirname(osp.dirname(room_path)))
        split       = osp.basename(osp.dirname(osp.dirname(osp.dirname(room_path))))
        print(f'split: {split} type: {room_type} id: {room_id}')

        bbox3d_path = osp.join(osp.dirname(osp.dirname(room_path)), 'bbox_3d', f'{room_id}.json')
        wall_path   = osp.join(osp.dirname(osp.dirname(room_path)), 'quad_walls', f'{room_id}.json')

        bboxes  = self.__load_json(bbox3d_path)['objects']
        walls   = self.__load_json(wall_path)['walls']

        ################################
        # bboxes
        ################################
        self.classes, self.locations, self.sizes, self.rotations = [], [], [], []
        for bbox in bboxes:
            bbox_class = bbox['class']
            bbox_class = CLASS_MAPPING[bbox_class] if bbox_class in CLASS_MAPPING.keys() else bbox_class
            assert bbox_class in ade20k_label2color.keys(), bbox_class

            if bbox_class in ['lamp']: continue

            self.classes.append(bbox_class)
            self.locations.append(bbox['center'])
            self.sizes.append(bbox['size'])
            self.rotations.append([math.degrees(v) for v in bbox['angles']])

        ################################
        # backgrounds
        ################################
        # all vertices
        vertices = set()
        for wall in walls:
            assert wall['class'] == 'wall'
            for vertex in wall['corners']:
                vertices.add(tuple(vertex))
        self.background_vertices = list(vertices)

        # ceiling & floor vertices
        z_coords = [v[2] for v in self.background_vertices]
        z_max, z_min = max(z_coords), min(z_coords)
        ceiling_vertices = [v for v in self.background_vertices if abs(v[2]-z_max) < 1e-4]
        floor_vertices = [v for v in self.background_vertices if abs(v[2]-z_min) < 1e-4]

        # ceiling faces
        ceiling_vertex_indices = [self.background_vertices.index(tuple(v)) for v in ceiling_vertices]
        ceiling_vertices_2d = [(v[0], v[1]) for v in ceiling_vertices]
        tri = Delaunay(ceiling_vertices_2d)
        self.backgrond_ceiling_faces = []
        for face in tri.simplices.tolist():
            self.backgrond_ceiling_faces.append([ceiling_vertex_indices[vertex_idx] for vertex_idx in face[::-1]])

        # floor faces
        floor_vertex_indices = [self.background_vertices.index(tuple(v)) for v in floor_vertices]
        floor_vertices_2d = [(v[0], v[1]) for v in floor_vertices]
        tri = Delaunay(floor_vertices_2d)
        self.backgrond_floor_faces = []
        for face in tri.simplices.tolist():
            self.backgrond_floor_faces.append([floor_vertex_indices[vertex_idx] for vertex_idx in face])

        # wall faces
        walls_faces = []
        for wall in walls:
            face = [self.background_vertices.index(tuple(v)) for v in wall['corners']]
            walls_faces.append(face[::-1])
        self.backgrond_walls_faces = walls_faces

    def __load_json(self, p):
        with open(p, 'r') as f:
            data = json.load(f)
        return data

    def export(self, path):
        bbox_data = []
        for bbox_class, location, size, rotation in \
            zip(self.classes, self.locations, self.sizes, self.rotations):
            bbox_data.append({
                "class": bbox_class,
                "prompt": bbox_class,
                "location": location,
                "size": [size[1], size[0], size[2]],
                "rotation": [rotation[0], rotation[1], rotation[2]+90]
            })

        background_data = {
            "vertices": self.background_vertices,
            "faces": {
                "ceiling": self.backgrond_ceiling_faces,
                "floor": self.backgrond_floor_faces,
                "walls": self.backgrond_walls_faces,
            }
        }

        with open(path, 'w') as f:
            json.dump({"bbox": bbox_data, "background": background_data}, f, indent=4)


if __name__=='__main__':
    # 0. configuration
    source_dataset_root = 'E:/data/Alibaba/new_text2pano'
    layout_output_root = '../../../data/layout_ctrlroom'

    # bedroom: 0,1,2,3,4
    # living room: 8017,8016,8015,8014,8013
    index = 0
    
    # 1. load
    layout = CtrlRoomLayout(source_dataset_root, index=index)

        # 2. export
        output_path = osp.join(layout_output_root, f'ctrlroom_{index:04d}', 'layout.json')
        if not osp.exists(osp.dirname(output_path)): os.makedirs(osp.dirname(output_path))
        layout.export(output_path)

    print('DONE')