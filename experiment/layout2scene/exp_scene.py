# conda activate layout2scene
# python exp_scene.py

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
PROJECT_ROOT        = osp.join('../../')
DEPENDENCIES_ROOT   = osp.join(PROJECT_ROOT, 'dependencies')
THREESTUDIO_ROOT    = osp.join(DEPENDENCIES_ROOT, 'threestudio')
SCRIPT_CAMERA_ROOT  = osp.join(PROJECT_ROOT, 'script', 'camera')
ADD_PYTHONPATH(PROJECT_ROOT)
ADD_PYTHONPATH(THREESTUDIO_ROOT)
ADD_PYTHONPATH(SCRIPT_CAMERA_ROOT)

GEODIFFUSION_PATH   = osp.join(PROJECT_ROOT, "checkpoint/LayoutGeoDiffusion/SemNDDiffusion_SunRGBD")
TEXDIFFUSION_PATH   = osp.join(PROJECT_ROOT, "checkpoint/LayoutTexDiffusion/SemGeoDiffusion-SD21Base-10000/multicontrolnet")

all_scane_names = [
    'hypersim_ai_010_005' # , 'bedroom_0000'
    # 'setthescene_bedroom', 'setthescene_dining_room', 'setthescene_garage', 'setthescene_living_room',
    # 'hypersim_ai_001_001', 'hypersim_ai_001_003', 'hypersim_ai_001_005', 'hypersim_ai_003_004',
    # 'hypersim_ai_006_010', 'hypersim_ai_010_005', 'hypersim_ai_010_008', 'hypersim_ai_022_005',
    # 'bedroom_0000', 'bedroom_0001', 'bedroom_0002', 'bedroom_0003', 'bedroom_0004',
    # 'livingroom_8013', 'livingroom_8016', 'livingroom_8017',
    # 'fankenstein_bedroom_001',
]


for scene_name in all_scane_names:
    layout_path     = osp.join(PROJECT_ROOT, f"data/layout/{scene_name}/layout.json")
    camera_path     = osp.join(PROJECT_ROOT, f"data/layout/{scene_name}/cameras.json")
    output_root     = osp.join(PROJECT_ROOT, "outputs")
    num_objects     = COUNT_OBJECTS(layout_path)

    gpu_str         = '0,1,2,3,4,5,6,7'
    n_gpus          = len(gpu_str.split(','))

    ###################################
    # Layout-aware Camera Sampling
    ###################################
    if not osp.exists(camera_path): os.system(f'python -m layout_aware_camera_sampling --layout_path {layout_path} --output_path {camera_path}')

    ###################################
    # Scene - Geometry
    ###################################
    config_path = 'configs/scene_gs_geometry.yaml'

    print(f'Scene geometry generation')
    # max_steps = max(1000 * num_objects // n_gpus, 1000)
    # os.system(f'\
    #     torchrun --master_port 29509 --nproc_per_node={n_gpus} launch_layout2scene.py \
    #         --config {config_path} --train --gpu {gpu_str} tag={scene_name} exp_root_dir="{output_root}" \
    #         data.global_camera.camera_path="{camera_path}" \
    #         system.geometry.init_file_path="{layout_path}" \
    #         system.background.global_background.init_file_path="{layout_path}" \
    #         system.layout_renderer.layout_path="{layout_path}" \
    #         system.geometry_prompt_processor.local_prompt_init_file_path="{layout_path}" \
    #         system.ctrl_geometry_guidance.controlnet_pretrained_model_name_or_path="[{GEODIFFUSION_PATH}]" \
    #         system.geometry.densify_interval={int(max_steps*0.1)} \
    #         system.ctrl_geometry_guidance.min_step_percent=[0,0.98,0.50,{max_steps}] \
    #         system.ctrl_geometry_guidance.max_step_percent=[0,0.98,0.50,{max_steps}] \
    #         system.local_geometry_guidance.min_step_percent=[0,0.98,0.02,{max_steps}] \
    #         system.local_geometry_guidance.max_step_percent=[0,0.98,0.02,{max_steps}] \
    #         system.optimizer._xyz_lr_schedule.lr_max_steps={int(max_steps*0.6)} \
    #         system.optimizer._scaling_lr_schedule.lr_max_steps={int(max_steps*0.6)} \
    #         trainer.max_steps={max_steps} \
    #     '
    # )

    ###################################
    # Scene - Mesh Extraction
    ###################################
    gaussian_file_path  = find_export_path(osp.join(output_root, 'scene_gs_geometry'), scene_name, 'gaussians', n_gpus > 1)
    volfusion_file_path = osp.join(osp.dirname(gaussian_file_path), osp.basename(gaussian_file_path).replace('gaussians', 'volfusion'))
    refined_file_path   = osp.join(osp.dirname(volfusion_file_path), osp.basename(volfusion_file_path).replace('volfusion', 'refined'))

    print(f'Mesh extraction ({gaussian_file_path} -> {refined_file_path})')
    # os.system(f'python -m extract_mesh --input_path {gaussian_file_path} --output_path {volfusion_file_path} --layout_path {layout_path} --is_local_space --device {gpu_str}')
    # os.system(f'blender -b --python refine_mesh_bl.py -- {volfusion_file_path} {refined_file_path}')

    ###################################
    # Scene - Appearance (Texture)
    ###################################
    config_path             = 'configs/scene_texture_generation.yaml'

    style_prompt            = 'Bohemian style'
    global_prompt           = style_prompt + ' ' + 'bedroom' + #', ' + 'best quality, high quality, extremely detailed, good geometry, high-res photo, \
                                                               #         ultra realistic, without shadows, without lightning effects, neutral lighting, \
                                                               #         diffuse light, studio light'
    local_additional_prompt = style_prompt
    local_prompt_layout     = style_prompt + ', ' + 'minimalist interior room layout with only walls, floor, ceiling'

    gpu_str = '0,1,2,3,6,7'
    n_gpus  = len(gpu_str.split(','))

    max_steps = 30000

    print(f'Scene texture generation')
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
