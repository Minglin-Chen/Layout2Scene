# conda activate scenegeneration
# python exp_gs.py

import os
import os.path as osp


# helpers
HF_ROOT = os.environ.get('HF_ROOT')
HF_PATH = lambda p: p if HF_ROOT is None else osp.join(HF_ROOT, p)

def ADD_PYTHONPATH(p):
    pythonpath = os.environ.get('PYTHONPATH', '')
    paths = pythonpath.split(os.pathsep) if pythonpath else []
    if p in paths: return
    paths.append(p)
    os.environ['PYTHONPATH'] = os.pathsep.join(paths)


# configuration
PROJECT_ROOT        = osp.join('../../')
DEPENDENCIES_ROOT   = osp.join(PROJECT_ROOT, 'dependencies')
THREESTUDIO_ROOT    = osp.join(DEPENDENCIES_ROOT, 'threestudio')
ADD_PYTHONPATH(PROJECT_ROOT)
ADD_PYTHONPATH(DEPENDENCIES_ROOT)
ADD_PYTHONPATH(THREESTUDIO_ROOT)


output_root = osp.join(PROJECT_ROOT, 'outputs')


######################################################################
# Object
######################################################################
config_path = 'configs/object_gs.yaml'
prompt      = 'an icecream'

command = f'\
    python -m launch_layout2gs --config {config_path} --train --gpu 0 exp_root_dir="{output_root}" \
        system.prompt_processor.prompt="{prompt}" \
        '
print(command)
os.system(command)


######################################################################
# Object - Geometry (Multiview)
######################################################################
config_path = 'configs/object_gs_geometry_mv.yaml'
prompt      = 'Majestic Peacock Throne, golden opulence, feathers adorned with jewels, royal symbolism, 3D asset'

command = f'\
    python -m launch_layout2gs --config {config_path} --train --gpu 0 exp_root_dir="{output_root}" \
        system.geometry_prompt_processor.prompt="{prompt}" \
        '
print(command)
os.system(command)


######################################################################
# Object - Geometry
######################################################################
config_path = 'configs/object_gs_geometry.yaml'
prompt      = 'a pineapple'

command = f'\
    python -m launch_layout2gs --config {config_path} --train --gpu 0 exp_root_dir="{output_root}" \
        system.geometry_prompt_processor.prompt="{prompt}" \
        '
print(command)
os.system(command)


######################################################################
# Object - Appearance
######################################################################
def find_latest_gaussian_path(path):
    exp_folder = sorted(os.listdir(path), key=lambda v: v.split('@')[-1])[-1]
    path = os.path.join(path, exp_folder, 'save', 'export')
    assert osp.exists(path), path
    iter_folders = [f for f in os.listdir(path) if f.startswith("gaussians")]
    iter_folder = sorted(iter_folders, key=lambda v: int(v.split('-')[1][2:]))[-1]
    return os.path.join(path, iter_folder)

init_file_path = find_latest_gaussian_path(os.path.join(output_root, 'object_gs_geometry_mv'))

config_path = 'configs/object_gs_appearance.yaml'
prompt      = 'Majestic Peacock Throne, golden opulence, feathers adorned with jewels, royal symbolism, 3D asset'
command = f'\
    python -m launch_layout2gs --config {config_path} --train --gpu 0 exp_root_dir="{output_root}" \
        system.prompt_processor.prompt="{prompt}" \
        system.geometry.init_file_path="{init_file_path}" \
        '
print(command)
os.system(command)