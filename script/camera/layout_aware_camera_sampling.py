import os
import os.path as osp
import numpy as np
import json
import trimesh
from tqdm import tqdm
import math
import pysdf

import torch
import torch.nn.functional as F
import torch.optim


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--layout_path", type=str, default="layout.json")
parser.add_argument("--camera_fov_deg", type=float, default=60)
parser.add_argument("--output_path", type=str, default="cameras.json")
args = parser.parse_args()


def load_layout(path):
    with open(path, 'r') as f:
        data = json.load(f)
    bboxes, background = data["bbox"], data["background"]

    # bboxes
    bbox_meshes, bbox_locations = [], []
    for bbox in bboxes:
        m = trimesh.creation.box()

        m.apply_scale(bbox["size"])
        rotation_matrix = trimesh.transformations.euler_matrix(
            *[math.radians(v) for v in bbox["rotation"]]
        )
        m.apply_transform(rotation_matrix)
        m.apply_translation(bbox["location"])

        bbox_meshes.append(m)
        bbox_locations.append(bbox["location"])
    
    bbox_mesh = trimesh.util.concatenate(bbox_meshes)
    bbox_locations = np.stack(bbox_locations)

    # background
    background_vertices = background["vertices"]

    background_ceiling = trimesh.Trimesh(
        vertices=background_vertices, faces=background["faces"]["ceiling"])
    background_ceiling.remove_unreferenced_vertices()

    background_floor = trimesh.Trimesh(
        vertices=background_vertices, faces=background["faces"]["floor"])
    background_floor.remove_unreferenced_vertices()

    background_walls = trimesh.Trimesh(
        vertices=background_vertices, faces=background["faces"]["walls"])
    background_walls.remove_unreferenced_vertices()

    background_mesh = trimesh.util.concatenate([
        background_ceiling, background_floor, background_walls
    ])
    background_mesh.merge_vertices()

    return bbox_mesh, bbox_locations, background_mesh


def camera_location_sampling_grid(bounds, unit=None):
    if unit is None:
        unit = max(
            (bounds[1][0] - bounds[0][0]),
            (bounds[1][1] - bounds[0][1]),
            (bounds[1][2] - bounds[0][2])
        ) / 20
    x = np.arange(bounds[0][0]+unit*0.5, bounds[1][0], unit)
    y = np.arange(bounds[0][1]+unit*0.5, bounds[1][1], unit)
    z = np.arange(bounds[0][2]+unit*0.5, bounds[1][2], unit)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    locations = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    return locations


def camera_orientation_calculation(
    camera_locations, 
    bbox_locations, 
    camera_fov_deg, 
    gamma=1.0,
    n_steps=10000,
    tolerance=1e-8,
    patience=10,
    device='cuda:0'
):
    camera_locations    = torch.tensor(camera_locations, dtype=torch.float32, device=device)
    bbox_locations      = torch.tensor(bbox_locations, dtype=torch.float32, device=device)
    # camera_fov_rad      = math.radians(camera_fov_deg)

    # (n_camera, n_bbox, 3)
    v_cam2box           = F.normalize(bbox_locations[None,:,:] - camera_locations[:,None,:], dim=-1)

    # weights (n_camera, n_bbox)
    d_cam2box           = torch.norm(bbox_locations[None,:,:] - camera_locations[:,None,:], dim=-1)
    # weights             = 1. / d_cam2box
    weights             = torch.exp(gamma * d_cam2box)

    # variable
    n_camera            = v_cam2box.shape[0]
    v_lookat            = torch.rand(n_camera, 3, device=device, requires_grad=True)
    optimizer           = torch.optim.SGD([v_lookat], lr=0.1, weight_decay=0.0)

    # optimization loop
    prev_loss           = None
    no_change_count     = 0
    for _ in tqdm(range(n_steps)):
        # (n_camera, n_bbox)
        cosine_similarity = torch.sum(v_cam2box * F.normalize(v_lookat[:,None,:], dim=-1), dim=-1)

        # loss = weights * torch.sigmoid(cosine_similarity - math.cos(camera_fov_rad * 0.5))
        loss_focus  = torch.sum(weights * (1. - cosine_similarity))
        # loss_reg    = torch.sum(v_lookat[:,2] ** 2)
        # loss_total  = loss_focus + loss_reg
        loss_total = loss_focus

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # finish condition
        if prev_loss is not None:
            if abs(loss_total.item() - prev_loss) < tolerance:
                no_change_count += 1
            else:
                no_change_count = 0

            if no_change_count >= patience:
                break

        prev_loss = loss_total.item()

    # (n_camera, 3)
    v_lookat = F.normalize(v_lookat, dim=-1)

    return v_lookat.detach().cpu().numpy()


if __name__=='__main__':

    # 1. load layout
    bbox_mesh, bbox_locations, background_mesh = load_layout(args.layout_path)

    # 2. camera location sampling
    # 2.1 camera location sampling within the bounds
    bounds = background_mesh.bounds
    camera_locations = camera_location_sampling_grid(bounds)

    z_min, z_max = bounds[0,2], bounds[1,2]
    z_min_valid = (z_max - z_min) * 0.5 + z_min
    z_max_valid = (z_max - z_min) * 0.7 + z_min
    mask = (camera_locations[:,2] >= z_min_valid) & (camera_locations[:,2] <= z_max_valid)
    camera_locations = camera_locations[mask]

    # 2.2 camera distance to the nearest plane
    # compute signed-distance-field (SDF)
    bg_sdf_fn           = pysdf.SDF(background_mesh.vertices, background_mesh.faces)
    obj_sdf_fn          = pysdf.SDF(bbox_mesh.vertices, bbox_mesh.faces)
    # - note that:
    # --> positive values when the points inside the mesh
    # --> negative values when the points outside the mesh
    bg_sdf              = bg_sdf_fn(camera_locations)
    camera_locations    = camera_locations[bg_sdf > 0.0]  # inside
    bg_sdf              = bg_sdf[bg_sdf > 0.0]

    obj_sdf             = obj_sdf_fn(camera_locations)
    camera_locations    = camera_locations[obj_sdf < 0.0] # outside
    bg_sdf              = bg_sdf[obj_sdf < 0.0]
    obj_sdf             = obj_sdf[obj_sdf < 0.0]

    # camera_distances    = np.minimum(bg_sdf, -obj_sdf)
    camera_distances    = - obj_sdf

    # 2.3 importance sampling
    probabilities       = camera_distances / camera_distances.max()

    # 3. camera rotation calculation
    lookat              = camera_orientation_calculation(camera_locations, bbox_locations, args.camera_fov_deg)

    # camera specification: forward -z, right +x, up +y
    elevations          = np.arccos(-lookat[:,2])
    azimuths            = np.arctan2(-lookat[:,0], lookat[:,1])
    elevations          = np.rad2deg(elevations)
    azimuths            = np.rad2deg(azimuths)

    rotations           = np.stack([elevations, np.zeros_like(elevations), azimuths], axis=-1)

    # 4. export
    cameras = []
    for loc, rot, prob in zip(camera_locations, rotations, probabilities):
        cameras.append({
            "location": loc.tolist(),
            "rotation": rot.tolist(),
            "fov": args.camera_fov_deg,
            "probability": float(prob),
        })

    os.makedirs(osp.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(cameras, f, indent=4)
