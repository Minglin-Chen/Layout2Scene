import os
import os.path as osp
import numpy as np
from glob import glob
import json
from tqdm import tqdm
from omegaconf import OmegaConf


def parse_blenderproc_camera(path):
    # load
    with open(osp.join(path, 'blenderproc.json'), 'r') as f:
        data = json.load(f)
    
    c2ws = [ np.array(m["matrix"]) for m in data.values() ]

    # convert
    cameras = []
    for c2w in c2ws:
        # right   = c2w[:3,0]
        # up      = c2w[:3,1]
        lookat  = c2w[:3,2]
        loc     = c2w[:3,3]

        # camera specification: forward -z, right +x, up +y
        elevation   = np.arccos(-lookat[2])
        azimuth     = np.arctan2(-lookat[0], lookat[1])
        elevation   = np.rad2deg(elevation)
        azimuth     = np.rad2deg(azimuth)

        rot         = [elevation, 0, azimuth]

        cameras.append({
            "location": [loc[0], -loc[1], loc[2]],
            "rotation": rot,
            "probability": 1.0
        })
    
    return cameras


def parse_sphere_camera(path):
    sphere_cameras = OmegaConf.load(osp.join(path, 'sphere.yaml'))

    dist_linspace = np.linspace(
        sphere_cameras.dist.min,
        sphere_cameras.dist.max,
        1 if sphere_cameras.dist.min == sphere_cameras.dist.max else sphere_cameras.dist.num_linspace,
    )
    elev_linspace = np.linspace(
        sphere_cameras.elev.min,
        sphere_cameras.elev.max,
        1 if sphere_cameras.elev.min == sphere_cameras.elev.max else sphere_cameras.elev.num_linspace,
    )
    azim_linspace = np.linspace(
        sphere_cameras.azim.min,
        sphere_cameras.azim.max,
        1 if sphere_cameras.azim.min == sphere_cameras.azim.max else sphere_cameras.azim.num_linspace,
    )
    at = np.array(sphere_cameras.at)
    at = [at[0,0], -at[0,2], at[0,1]]

    combinations = np.array(np.meshgrid(dist_linspace, azim_linspace, elev_linspace)).T.reshape(-1, 3)
    dist_list = combinations[:, 0].tolist()
    azim_list = combinations[:, 1].tolist()
    elev_list = combinations[:, 2].tolist()

    cameras = []
    for d, azim, elev in zip(dist_list, azim_list, elev_list):

        loc = [
            at[0] - d * np.cos(np.deg2rad(elev)) * np.cos(np.deg2rad(azim)),
            at[1] - d * np.cos(np.deg2rad(elev)) * np.sin(np.deg2rad(azim)),
            at[2] - d * np.sin(np.deg2rad(elev))
        ]

        cameras.append({
            "location": loc,
            "rotation": [elev+90, 0.0, azim-90],
            "probability": 1.0
        })

    return cameras


if __name__=='__main__':
    scenetex_data_root = '../../../data/scenetex/3D-FRONT_preprocessed'
    camera_paths = glob(osp.join(scenetex_data_root, "*/*/*/cameras"))
    use_sphere_cameras = True

    for p in tqdm(camera_paths):
        cameras = parse_sphere_camera(p) if use_sphere_cameras else parse_blenderproc_camera(p)

        # output
        output_path = osp.join(osp.dirname(p), 'cameras.json')
        os.makedirs(osp.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(cameras, f, indent=4)