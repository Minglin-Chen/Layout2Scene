
import math
import json
import trimesh
import numpy as np

from .ade20k_protocol import ade20k_label2color


def load_layout(path):
    with open(path, 'r') as f:
        data = json.load(f)
    bboxes, background = data["bbox"], data["background"]

    # bboxes
    bbox_meshes = []
    for bbox in bboxes:
        m = trimesh.creation.box()

        m.apply_scale(bbox["size"])
        rotation_matrix = trimesh.transformations.euler_matrix(
            *[math.radians(v) for v in bbox["rotation"]]
        )
        m.apply_transform(rotation_matrix)
        m.apply_translation(bbox["location"])
        
        cls = bbox["class"]
        assert cls in ade20k_label2color.keys(), cls
        color = ade20k_label2color[cls]
        vertex_colors = np.array(
            [[color[0]/255., color[1]/255., color[2]/255., 1.0]] * len(m.vertices)
        )
        m.visual.vertex_colors = vertex_colors

        bbox_meshes.append(m)

    bbox_mesh = trimesh.util.concatenate(bbox_meshes)

    # background
    background_vertices = background["vertices"]

    background_ceiling = trimesh.Trimesh(
        vertices=background_vertices, faces=background["faces"]["ceiling"])
    background_ceiling.remove_unreferenced_vertices()
    background_ceiling.visual.vertex_colors = np.array(
        [[
            ade20k_label2color["ceiling"][0] / 255., 
            ade20k_label2color["ceiling"][1] / 255., 
            ade20k_label2color["ceiling"][2] / 255., 
            1.0
        ]] * len(background_ceiling.vertices)
    )

    background_floor = trimesh.Trimesh(
        vertices=background_vertices, faces=background["faces"]["floor"])
    background_floor.remove_unreferenced_vertices()
    background_floor.visual.vertex_colors = np.array(
        [[
            ade20k_label2color["floor"][0] / 255., 
            ade20k_label2color["floor"][1] / 255., 
            ade20k_label2color["floor"][2] / 255., 
            1.0
        ]] * len(background_floor.vertices)
    )

    background_walls = trimesh.Trimesh(
        vertices=background_vertices, faces=background["faces"]["walls"])
    background_walls.remove_unreferenced_vertices()
    background_walls.visual.vertex_colors = np.array(
        [[
            ade20k_label2color["wall"][0] / 255., 
            ade20k_label2color["wall"][1] / 255., 
            ade20k_label2color["wall"][2] / 255., 
            1.0
        ]] * len(background_walls.vertices)
    )

    background_mesh = trimesh.util.concatenate([
        background_ceiling, background_floor, background_walls
    ])
    # background_mesh.merge_vertices()

    return bbox_mesh, background_mesh