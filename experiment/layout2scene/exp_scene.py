# conda activate scenegeneration
# python exp_scene.py

import os
import os.path as osp
import json


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


# configuration
PROJECT_ROOT        = osp.join('../../')
DEPENDENCIES_ROOT   = osp.join(PROJECT_ROOT, 'dependencies')
THREESTUDIO_ROOT    = osp.join(DEPENDENCIES_ROOT, 'threestudio')
NDDIFFUSION_ROOT    = osp.join(DEPENDENCIES_ROOT, 'normal-depth-diffusion')
SCRIPT_CAMERA_ROOT  = osp.join(PROJECT_ROOT, 'script', 'camera')
ADD_PYTHONPATH(PROJECT_ROOT)
ADD_PYTHONPATH(DEPENDENCIES_ROOT)
ADD_PYTHONPATH(THREESTUDIO_ROOT)
ADD_PYTHONPATH(SCRIPT_CAMERA_ROOT)

SEMGEODIFFUSION_PATH    = osp.join(PROJECT_ROOT, "checkpoint/SemGeoDiffusion/SemGeoDiffusion-SD21Base-20000/multicontrolnet")
SEMNDDIFFUSION_PATH     = osp.join(PROJECT_ROOT, "checkpoint/SemNDDiffusion/SemNDDiffusion_ScanNetpp_noprompt")

scene_name      = "setthescene_bedroom"
layout_path     = osp.join(PROJECT_ROOT, f"data/layout/{scene_name}/layout.json")
num_objects     = COUNT_OBJECTS(layout_path)
camera_path     = osp.join(PROJECT_ROOT, f"data/layout/{scene_name}/cameras.json")
output_root     = osp.join(PROJECT_ROOT, "outputs")

gpu_str         = '0,1,2,3'
n_gpu           = len(gpu_str.split(','))


###################################
# Layout-aware Camera Sampling
###################################
if not osp.exists(camera_path):
    command = f'python -m layout_aware_camera_sampling --layout_path {layout_path} --output_path {camera_path}'
    print(command)
    os.system(command)


###################################
# Scene - Geometry
###################################
config_path = 'configs/scene_gs_geometry.yaml'

max_steps = int(1000*num_objects)
command = f'\
    python -m launch_layout2gs --config {config_path} --train --gpu {gpu_str} tag={scene_name} exp_root_dir="{output_root}" \
        data.global_camera.camera_path="{camera_path}" \
        system.geometry.init_file_path="{layout_path}" \
        system.background.global_background.init_file_path="{layout_path}" \
        system.layout_renderer.layout_path="{layout_path}" \
        system.geometry_prompt_processor.local_prompt_init_file_path="{layout_path}" \
        system.ctrl_geometry_guidance.controlnet_pretrained_model_name_or_path="[{SEMNDDIFFUSION_PATH}]" \
        system.geometry.densify_interval={int(max_steps*0.1)} \
        system.geometry.opacity_reset_interval={int(max_steps*0.1)} \
        system.ctrl_geometry_guidance.min_step_percent=[0,0.98,0.02,{max_steps}] \
        system.ctrl_geometry_guidance.max_step_percent=[0,0.98,0.02,{max_steps}] \
        system.local_geometry_guidance.min_step_percent=[0,0.98,0.02,{max_steps}] \
        system.local_geometry_guidance.max_step_percent=[0,0.98,0.02,{max_steps}] \
        system.optimizer._xyz_lr_schedule.lr_max_steps={int(max_steps*0.6)} \
        system.optimizer._scaling_lr_schedule.lr_max_steps={int(max_steps*0.6)} \
        trainer.max_steps={max_steps} \
        '
print(command)
os.system(command)


###################################
# Scene - Mesh Extraction
###################################
# gaussian_file_path = find_latest_path(os.path.join(output_root, 'scene_gs_geometry'), 'gaussians')
gaussian_file_path = os.path.join(
    output_root, 'scene_gs_geometry', scene_name, 'save', 'export', f'gaussians-it{max_steps}-test')
volfusion_file_path = osp.join(
    osp.dirname(gaussian_file_path), 
    osp.basename(gaussian_file_path).replace('gaussians', 'volfusion'))
command = f'\
    python -m extract_mesh \
        --input_path {gaussian_file_path} \
        --output_path {volfusion_file_path} \
        --is_local_space \
    '
print(command)
os.system(command)


###################################
# Scene - Appearance (Texture)
###################################
config_path         = 'configs/scene_texture_generation.yaml'

stype_prompt        = 'modern style'
a_prompt            = 'best quality, high quality, extremely detailed, good geometry, high-res photo'
prompt              = stype_prompt + ', ' + a_prompt


# volfusion_file_path = find_latest_path(os.path.join(output_root, 'scene_gs_geometry'), 'volfusion')
volfusion_file_path = os.path.join(\
    output_root, 'scene_gs_geometry', scene_name, 'save', 'export', f'volfusion-it{max_steps}-test')
command = f'\
    python -m launch_layout2gs --config {config_path} --train --gpu {gpu_str} tag={scene_name} exp_root_dir="{output_root}" \
        data.camera_path="{camera_path}" \
        system.geometry.init_file_path="{volfusion_file_path}" \
        system.layout_renderer.layout_path="{layout_path}" \
        system.prompt_processor.prompt="{prompt}" \
        system.ctrl_geometry_guidance.max_step_percent=[{5000//n_gpu},0.98,0.02,{(5000//n_gpu)+1}] \
        system.optimizer.args.lr={0.001*n_gpu} \
        trainer.max_steps="{30000//n_gpu}" \
        '
print(command)
os.system(command)


###################################
# Scene - Appearance
###################################
# init_file_path = find_latest_path(os.path.join(output_root, 'scene_gs_geometry'), 'gaussians')

# a_prompt = 'best quality, high quality, extremely detailed, good geometry, high-res photo'

# config_path = 'configs/scene_gs_appearance.yaml'
# prompt      = 'a Lego style living room' + ', ' + a_prompt
# command = f'\
#     python -m launch_layout2gs --config {config_path} --train --gpu 0 exp_root_dir="{output_root}" \
#         data.camera_path="{camera_path}" \
#         system.geometry.init_file_path="{init_file_path}" \
#         system.background.init_file_path="{layout_path}" \
#         system.layout_renderer.layout_path="{layout_path}" \
#         system.prompt_processor.prompt="{prompt}" \
#         '
# print(command)
# os.system(command)


###################################
# Scene
###################################
# config_path     = 'configs/scene_gs.yaml'

# prompt_geometry = 'a bedroom'

# command = f'\
#     python -m launch_layout2gs --config {config_path} --train --gpu 0 exp_root_dir="{output_root}" \
#         data.camera_path="{camera_path}" \
#         system.geometry.init_file_path="{layout_path}" \
#         system.background.init_file_path="{layout_path}" \
#         system.layout_renderer.layout_path="{layout_path}" \
#         system.prompt_processor.prompt="{prompt_geometry}" \
#         '
# print(command)
# os.system(command)


###################################
# Scene - Visualization
###################################
# OUTPUT_PATH = '../../outputs/scene_gs_appearance/modern_style@20250524-210823'
# CONFIG_PATH = osp.join(OUTPUT_PATH, 'configs/parsed.yaml')
# WEIGHT_PATH = osp.join(OUTPUT_PATH, 'ckpts/last.ckpt')
# command = f'python -m launch_imgui --config {CONFIG_PATH} --gpu 0 system.weights="{WEIGHT_PATH}"'
# print(command)
# os.system(command)