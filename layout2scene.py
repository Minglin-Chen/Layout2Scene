# conda activate layout2scene
# python layout2scene.py

import argparse
import os
import os.path as osp
import json
from glob import glob


# helpers
HF_ROOT = os.environ.get('HF_ROOT')
HF_PATH = lambda p: p if HF_ROOT is None else osp.join(HF_ROOT, p)

def ADD_PYTHONPATH(p):
    pythonpath = os.environ.get('PYTHONPATH', '')
    paths = pythonpath.split(os.pathsep) if pythonpath else []
    if p in paths: return
    paths.append(p)
    os.environ['PYTHONPATH'] = os.pathsep.join(paths)

def COUNT_OBJECTS(p):
    with open(p, 'r') as f:
        n = len(json.load(f)['bbox'])
    return n

def find_latest_path(path, mode='gaussians'):
    exp_folder = sorted(os.listdir(path), key=lambda v: v.split('@')[-1])[-1]
    path = os.path.join(path, exp_folder, 'save', 'export')
    assert osp.exists(path), path
    iter_folders = [f for f in os.listdir(path) if f.startswith(mode)]
    iter_folder = sorted(iter_folders, key=lambda v: int(v.split('-')[1][2:]))[-1]
    return os.path.join(path, iter_folder)

def find_export_path(output_path, scene_name, mode='refined', multigpu=True):
    if multigpu:
        candidates = glob(osp.join(output_path, scene_name, 'save', 'export', mode+'-it*-test'))
        assert len(candidates) == 1, candidates
        path = candidates[0]
    else:
        path = find_latest_path(output_path, mode)
    return path


# configuration
PROJECT_ROOT        = osp.join('./')
DEPENDENCIES_ROOT   = osp.join(PROJECT_ROOT, 'dependencies')
THREESTUDIO_ROOT    = osp.join(DEPENDENCIES_ROOT, 'threestudio')
SCRIPT_CAMERA_ROOT  = osp.join(PROJECT_ROOT, 'script', 'camera')
ADD_PYTHONPATH(PROJECT_ROOT)
ADD_PYTHONPATH(THREESTUDIO_ROOT)
ADD_PYTHONPATH(SCRIPT_CAMERA_ROOT)

TEXDIFFUSION_PATH   = osp.join(PROJECT_ROOT, "checkpoint/LayoutTexDiffusion/multicontrolnet")


# generate
parser = argparse.ArgumentParser(description="Layout2Scene")
parser.add_argument('--layout', type=str, default=None, help='Path to the layout json file.')
parser.add_argument('--type', type=str, default='bedroom', help='Scene type (e.g., bedroom, living room).')
parser.add_argument('--style', type=str, default='Bohemian style', help='Style of the scene.')
parser.add_argument('--camera', type=str, default=None, help='Path to the camera json file.')
parser.add_argument('--output', type=str, default=None, help='Path to the output folder.')
parser.add_argument('--gpu', type=str, default='0,1,2,3,4,5,6,7', help='GPU device ids, separated by comma.')
args = parser.parse_args()

layout_path = args.layout if args.layout is not None else osp.join(PROJECT_ROOT, "data/layout/hypersim_ai_010_005/layout.json")
camera_path = args.camera if args.camera is not None else osp.join(osp.dirname(layout_path), "cameras.json")
output_root = args.output if args.output is not None else osp.join(PROJECT_ROOT, "outputs")
gpu_str     = args.gpu
n_gpus      = len(gpu_str.split(','))

num_objects = COUNT_OBJECTS(layout_path)
scene_name  = osp.basename(osp.dirname(layout_path))


###################################
# Layout-aware Camera Sampling
###################################
if not osp.exists(camera_path): 
    os.system(f'python -m layout_aware_camera_sampling --layout_path {layout_path} --output_path {camera_path}')


###################################
# Scene - Geometry
###################################
config_path = 'configs/scene_geometry.yaml'

print(f'Scene geometry generation')
max_steps = max(1000 * num_objects // n_gpus, 1000)
os.system(f'\
    torchrun --master_port 29509 --nproc_per_node={n_gpus} launch_layout2scene.py \
        --config {config_path} --train --gpu {gpu_str} tag={scene_name} exp_root_dir="{output_root}" \
        data.global_camera.camera_path="{camera_path}" \
        system.geometry.init_file_path="{layout_path}" \
        system.background.global_background.init_file_path="{layout_path}" \
        system.layout_renderer.layout_path="{layout_path}" \
        system.geometry_prompt_processor.local_prompt_init_file_path="{layout_path}" \
        system.geometry.densify_interval={int(max_steps*0.1)} \
        system.local_geometry_guidance.min_step_percent=[0,0.98,0.02,{max_steps}] \
        system.local_geometry_guidance.max_step_percent=[0,0.98,0.02,{max_steps}] \
        system.optimizer._xyz_lr_schedule.lr_max_steps={int(max_steps*0.6)} \
        system.optimizer._scaling_lr_schedule.lr_max_steps={int(max_steps*0.6)} \
        trainer.max_steps={max_steps} \
    '
)


###################################
# Scene - Mesh Extraction
###################################
gaussian_file_path  = find_export_path(osp.join(output_root, 'scene_gs_geometry'), scene_name, 'gaussians', n_gpus > 1)
volfusion_file_path = osp.join(osp.dirname(gaussian_file_path), osp.basename(gaussian_file_path).replace('gaussians', 'volfusion'))
refined_file_path   = osp.join(osp.dirname(volfusion_file_path), osp.basename(volfusion_file_path).replace('volfusion', 'refined'))

print(f'Mesh extraction ({gaussian_file_path} -> {refined_file_path})')
os.system(f'python -m extract_mesh --input_path {gaussian_file_path} --output_path {volfusion_file_path} --layout_path {layout_path} --is_local_space --device {gpu_str}')
os.system(f'blender -b --python refine_mesh_bl.py -- {volfusion_file_path} {refined_file_path}')


###################################
# Scene - Appearance (Texture)
###################################
config_path             = 'configs/scene_appearance.yaml'

style_prompt            = args.style
global_prompt           = style_prompt + ' ' + args.type
local_additional_prompt = style_prompt
local_prompt_layout     = style_prompt + ', ' + 'minimalist interior room layout with only walls, floor, ceiling'

max_steps = 30000

print(f'Scene appearance generation')
os.system(f'\
    torchrun --master_port 29509 --nproc_per_node={n_gpus} launch_layout2scene.py \
        --config {config_path} --train --gpu {gpu_str} tag={scene_name+"_"+style_prompt.replace(" ", "_")} exp_root_dir="{output_root}" \
        data.global_camera.camera_path="{camera_path}" \
        system.geometry.init_file_path="{refined_file_path}" \
        system.ctrl_appearance_guidance.controlnet_pretrained_model_name_or_path=[{TEXDIFFUSION_PATH}] \
        system.prompt_processor.global_prompt_processor.prompt="{global_prompt}" \
        system.prompt_processor.local_prompt_init_file_path="{layout_path}" \
        system.prompt_processor.local_additional_prompt="{local_additional_prompt}" \
        system.prompt_processor.local_prompt_layout="{local_prompt_layout}" \
        trainer.max_steps="{max_steps}" \
')
