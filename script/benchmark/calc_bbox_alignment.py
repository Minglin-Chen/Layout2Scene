import argparse
import os
import os.path as osp
from glob import glob
import json

import numpy as np
import trimesh


def bbox3d_iou(box1, box2):
    """Axis-aligned 3D BBox IoU
    Args:
        box1, box2: [x_min, y_min, z_min, x_max, y_max, z_max]
    Return:
        iou: float
    """
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    z_min = max(box1[2], box2[2])
    x_max = min(box1[3], box2[3])
    y_max = min(box1[4], box2[4])
    z_max = min(box1[5], box2[5])

    inter_dx = max(0.0, x_max - x_min)
    inter_dy = max(0.0, y_max - y_min)
    inter_dz = max(0.0, z_max - z_min)
    inter_vol = inter_dx * inter_dy * inter_dz

    vol1 = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    vol2 = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])

    union_vol = vol1 + vol2 - inter_vol

    iou = inter_vol / union_vol if union_vol > 0 else 0.0
    return iou


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, help='Input directory')
    args = parser.parse_args()

    layout_paths = glob(osp.join(args.input_dir, '*', 'save', 'export', 'layout.json'))

    iou_list = []
    for layout_path in layout_paths:

        with open(layout_path, 'r') as f:
            bboxes_dict = json.load(f)['bbox']

        mesh_paths = glob(osp.join(osp.dirname(layout_path), 'volfusion-it*-test', '*'))
        mesh_paths = sorted(mesh_paths, key=lambda s: int(osp.basename(s).split('.')[0]))
        print('Mesh paths: ', mesh_paths)

        for bbox_dict, mesh_path in zip(bboxes_dict, mesh_paths):
            size_gt = np.array(bbox_dict['size'])
            size_gt = size_gt / size_gt.max()
            bbox_gt = [
                -size_gt[0]*0.5, 
                -size_gt[1]*0.5, 
                -size_gt[2]*0.5, 
                size_gt[0]*0.5, 
                size_gt[1]*0.5, 
                size_gt[2]*0.5
            ]

            bbox_pred = trimesh.load(mesh_path).bounds
            bbox_pred = bbox_pred[0].tolist() + bbox_pred[1].tolist()

            iou = bbox3d_iou(bbox_pred, bbox_gt)
            iou_list.append(iou)
    
    print('IoU: ', iou_list)
    miou = np.mean(iou_list)
    print(f'Mean IOU: {miou:.4f}')
