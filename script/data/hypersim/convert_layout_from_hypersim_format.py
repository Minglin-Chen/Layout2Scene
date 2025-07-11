import os
import os.path as osp
from glob import glob
import numpy as np
import h5py
import pandas as pd
import json
from scipy.spatial.transform import Rotation as R


def load_h5py(p):
    with h5py.File(p, 'r') as f:
        assert 'dataset' in f.keys()
        data = f['dataset'][:]
    return data


nyu40_to_ade20k_label_alias = {
    'unlabeled':        'base',
    'window':           'windowpane', 
    'bookshelf':        'bookcase', 
    'picture':          'painting',
    'blinds':           'curtain', 
    'shelves':          'shelf', 
    'dresser':          'desk',
    'floormat':         'rug',
    'clothes':          'apparel', 
    'books':            'book', 
    'television':       'television receiver', 
    'paper':            'poster', 
    'showercurtain':    'curtain', 
    'whiteboard':       'signboard', 
    'nightstand':       'table', 
    'otherstructure':   'base', 
    'otherfurniture':   'base', 
    'otherprop':        'base'
}


if __name__=='__main__':
    # configuration
    hypersim_data_root  = '../../../data/hypersim'
    hypersim_repo_path  = '../../../dependencies/ml-hypersim'
    nyu40id_path        = 'nyu40id.txt'
    output_root         = '../../../data/hypersim_layout/'

    # NYU protocal
    nyu40_id2label = list(map(lambda x: x.strip(), open(nyu40id_path, "r").readlines()))

    # parse
    scene_paths = glob(osp.join(hypersim_data_root, 'ai_*', '_detail'))
    for scene_path in scene_paths:
        scene_id = osp.basename(osp.dirname(scene_path))

        metadata_scene_file     = os.path.join(hypersim_data_root, scene_id, "_detail", "metadata_scene.csv")
        metadata_scene          = pd.read_csv(metadata_scene_file, index_col="parameter_name")
        meters_per_asset_unit   = metadata_scene.loc["meters_per_asset_unit"].values[0]

        #############################################
        # layout
        #############################################
        bbox_size_path      = osp.join(scene_path, 'mesh', 'metadata_semantic_instance_bounding_box_object_aligned_2d_extents.hdf5')
        bbox_rotation_path  = osp.join(scene_path, 'mesh', 'metadata_semantic_instance_bounding_box_object_aligned_2d_orientations.hdf5')
        bbox_location_path  = osp.join(scene_path, 'mesh', 'metadata_semantic_instance_bounding_box_object_aligned_2d_positions.hdf5')
        bbox_size           = load_h5py(bbox_size_path) * meters_per_asset_unit         # Mx3
        bbox_rotation       = load_h5py(bbox_rotation_path)                             # Mx3x3
        bbox_location       = load_h5py(bbox_location_path) * meters_per_asset_unit     # Mx3

        anno_path   = osp.join(hypersim_repo_path, "evermotion_dataset/scenes", scene_id, "_detail/mesh")
        si_file     = osp.join(anno_path, "mesh_objects_si.hdf5")   # NYU40 semantic label for each object ID
        sii_file    = osp.join(anno_path, "mesh_objects_sii.hdf5")  # semantic instance ID for each object ID
        si          = load_h5py(si_file)
        sii         = load_h5py(sii_file)

        bbox_label_ids = list(
            map(lambda i: si[sii == i][0] if np.any(sii == i) and i != 0 else -1, range(len(bbox_location)))
        )
        bbox_label_ids = np.array(bbox_label_ids)

        mask            = np.logical_or(np.any(np.isinf(bbox_size), axis=-1), bbox_label_ids == -1)
        bbox_size       = bbox_size[~mask]
        bbox_rotation   = bbox_rotation[~mask]
        bbox_location   = bbox_location[~mask]
        bbox_label_ids  = bbox_label_ids[~mask]

        bbox_data = []
        for loc, size, rot_matrix, label_id in zip(bbox_location, bbox_size, bbox_rotation, bbox_label_ids):
            label = nyu40_id2label[label_id]

            if label in nyu40_to_ade20k_label_alias.keys():
                label = nyu40_to_ade20k_label_alias[label]

            if label in ['base', 'lamp']: continue

            r = R.from_matrix(rot_matrix)
            euler_angles_deg = r.as_euler('xyz', degrees=True)

            bbox_data.append({
                "class":        label,
                "prompt":       label,
                "location":     loc.tolist(),
                "size":         size.tolist(),
                "rotation":     euler_angles_deg.tolist(),
            })

        layout_output_path = osp.join(
            output_root, f'hypersim_{scene_id}', 'layout.json'
        )
        os.makedirs(osp.dirname(layout_output_path), exist_ok=True)
        with open(layout_output_path, 'w') as f:
            json.dump({"bbox": bbox_data}, f, indent=4)

        #############################################
        # cameras
        #############################################
        camera_indics_path      = osp.join(scene_path, 'cam_00', 'camera_keyframe_frame_indices.hdf5')
        camera_lookat_path      = osp.join(scene_path, 'cam_00', 'camera_keyframe_look_at_positions.hdf5')
        camera_orientation_path = osp.join(scene_path, 'cam_00', 'camera_keyframe_orientations.hdf5')
        camera_position_path    = osp.join(scene_path, 'cam_00', 'camera_keyframe_positions.hdf5')
        
        camera_indics           = load_h5py(camera_indics_path)                             # K
        camera_lookat           = load_h5py(camera_lookat_path)                             # Kx3
        camera_orientation      = load_h5py(camera_orientation_path)                        # Kx3x3
        camera_position         = load_h5py(camera_position_path) * meters_per_asset_unit   # Kx3

        cameras = []
        for lookat, rot_matrix, loc in zip(camera_lookat, camera_orientation, camera_position):
            
            r = R.from_matrix(rot_matrix)
            euler_angles_deg = r.as_euler('xyz', degrees=True)

            cameras.append({
                "location": loc.tolist(),
                "rotation": euler_angles_deg.tolist(),
                "fov": 60,
                "probability": 1.
            })

        cameras_output_path = osp.join(
            output_root, f'hypersim_{scene_id}', 'cameras.json'
        )
        os.makedirs(osp.dirname(cameras_output_path), exist_ok=True)
        with open(cameras_output_path, 'w') as f:
            json.dump(cameras, f, indent=4)

    print('DONE')