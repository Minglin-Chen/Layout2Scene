








import os
import os.path as osp
import argparse
import json

from glob import glob


def compute_iou_3d(box1, box2):
    """
    Parameters:
        box1: [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        box2: [[x_min, y_min, z_min], [x_max, y_max, z_max]]
    Returns: float
    """

    # 计算交集区域的坐标
    x_left   = max(box1[0], box2[0])
    y_top    = max(box1[1], box2[1])
    x_right  = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # 如果没有交集，IoU为0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 交集面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 各自的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 并集面积
    union_area = box1_area + box2_area - intersection_area

    # IoU
    iou = intersection_area / union_area
    return iou

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Volfusion output path')
    args = parser.parse_args()

    # load layout
    layout_path = osp.join(args.path, '..', 'layout.json')
    with open(layout_path, 'r') as f:
        bboxes = json.load(f)['bbox']

    # 
    for i, bbox in enumerate(bboxes):
        mesh_path = osp.join(args.path, f'{i}.ply')
        assert osp.exists(mesh_path)

