import os
from glob import glob
from dataclasses import dataclass, field
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import BaseExplicitGeometry
from threestudio.utils.typing import *
from threestudio.utils.misc import get_device
from core.utils.helper import *
from core.utils.ade20k_protocol import ade20k_label2color
from core.utils.gaussian_utils import save_gaussians, load_gaussians
from core.utils.gaussian2mesh import gaussian2mesh
from core.utils.point_utils import sampling_cube, sampling_ball

import json
import trimesh

from .sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2


@threestudio.register("gaussian-model")
class GaussianModel(BaseExplicitGeometry):
    @dataclass
    class Config(BaseExplicitGeometry.Config):
        is_2dgs: bool                   = False
        is_local_space: bool            = False
        local_scale: float              = 1.0

        init_strategy: str              = 'from_layout'
        init_file_path: str             = ''

        init_point_strategy: str        = 'random_cube'
        init_num_points: int            = 5000
        init_color_strategy: str        = 'random'
        init_rotation_strategy: str     = 'random'
        init_scaling_strategy: str      = 'nearest_neighbor'
        init_scaling_value: float       = 0.1
        init_opacity_value: float       = 0.1

        use_sh: bool                    = True
        sh_degree: int                  = 0

        densify_from_step: int          = 0
        densify_until_step: int         = -1
        densify_interval: int           = 50
        opacity_reset_interval: int     = -1

        normalize_grad: bool            = True
        densify_grad_threshold: float   = 0.01   
        opacity_threshold: float        = 0.01
        view_size_threshold: float      = 0.0
        world_size_threshold: float     = 0.0

        percent_dense: float            = 0.08

        prune_out_of_box: bool          = False

    cfg: Config

    def configure(self) -> None:
        super().configure()
        
        self.scaling_activation         = torch.exp
        self.rotation_activation        = F.normalize
        self.opacity_activation         = torch.sigmoid

        self.scaling_inverse_activation = torch.log
        self.opacity_inverse_activation = lambda x: torch.log(x/(1-x))

        self.use_sh                     = self.cfg.use_sh

        # initialization
        init_func = getattr(self, '_init_' + self.cfg.init_strategy, None)
        assert init_func is not None, \
            f'{self.__class__.__name__} init method ({self.cfg.init_strategy}) is unimplemented!'

        xyz, features_dc, features_rest, scaling, rotation, opacity, semantic, \
            instance, instance_location, instance_size, instance_rotation, instance_class, instance_prompt = \
                init_func()
        
        # Gaussians
        self._xyz               = nn.Parameter(xyz, requires_grad=True)                 # (N,3)
        self._features_dc       = nn.Parameter(features_dc, requires_grad=True)         # (N,1,3) or (N,3)
        self._features_rest     = nn.Parameter(features_rest, requires_grad=True)       # (N,(self.cfg.sh_degree+1)**2-1,3) or (N,o)
        self._scaling           = nn.Parameter(scaling, requires_grad=True)             # (N,2)
        self._rotation          = nn.Parameter(rotation, requires_grad=True)            # (N,4)
        self._opacity           = nn.Parameter(opacity, requires_grad=True)             # (N,1)

        self._semantic          = nn.Parameter(semantic, requires_grad=False)           # (N,3)
        self._instance          = nn.Parameter(instance, requires_grad=False)           # (N,1)
        self.instance_location  = nn.Parameter(instance_location, requires_grad=False)  # (K,3)
        self.instance_size      = nn.Parameter(instance_size, requires_grad=False)      # (K,3)
        self.instance_rotation  = nn.Parameter(instance_rotation, requires_grad=False)  # (K,3)
        self.instance_class     = instance_class                                        # (K)
        self.instance_prompt    = instance_prompt                                       # (K)

        self.num_instance       = int(instance.max().item()) + 1

        self.optimizable_keys   = ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity']

        # update
        self.xyz_grad_accum     = torch.zeros((self.get_xyz.shape[0])).to(get_device())
        self.denom              = torch.zeros((self.get_xyz.shape[0])).to(get_device())
        self.max_radii2D        = torch.zeros((self.get_xyz.shape[0])).to(get_device())

    # initialization
    def _init_from_sampling(self):
        if self.cfg.init_point_strategy == 'random_cube':
            xyz = sampling_cube(self.cfg.init_num_points)
        elif self.cfg.init_point_strategy == 'random_ball':
            xyz = sampling_ball(self.cfg.init_num_points)
        elif self.cfg.init_point_strategy == 'random_sphere':
            xyz = sampling_ball(self.cfg.init_num_points, is_sphere=True)
        else:
            raise ValueError
        xyz *= self.cfg.radius

        if self.cfg.init_color_strategy == 'random':
            rgb = SH2RGB(torch.rand((xyz.shape[0], 3)) / 255.)
        else:
            raise ValueError
        
        if self.use_sh:
            self.active_sh_degree   = 0
            self.max_sh_degree      = self.cfg.sh_degree
            # (N,(self.max_sh_degree+1)**2,3)
            features                = torch.zeros((xyz.shape[0], (self.max_sh_degree+1)**2, 3))
            features[:,0]           = RGB2SH(rgb)
            features_dc             = features[:,:1].contiguous()
            features_rest           = features[:,1:].contiguous()
        else:
            assert self.cfg.sh_degree == 0
            # (N,3)
            features_dc             = rgb[:,:3].contiguous()
            features_rest           = torch.zeros((rgb.shape[0], 0))
    
        if self.cfg.init_scaling_strategy == 'nearest_neighbor':
            dist2   = torch.clamp_min(distCUDA2(xyz.cuda()), 0.0000001)
            dist    = torch.sqrt(dist2)[...,None].repeat(1, 2 if self.cfg.is_2dgs else 3)
            scaling = self.scaling_inverse_activation(dist)
        elif self.cfg.init_scaling_strategy == 'fixed_value':
            scaling = self.scaling_inverse_activation(
                self.cfg.init_scaling_value * torch.ones((xyz.shape[0], 2 if self.cfg.is_2dgs else 3)))
        else:
            raise ValueError

        if self.cfg.init_rotation_strategy == 'random':
            rotation = torch.rand((xyz.shape[0], 4))
        elif self.cfg.init_rotation_strategy == 'radial':
            rotation = get_quaternion_from_vectors(torch.tensor([[0.,0.,1.]]), F.normalize(xyz))
        else:
            raise ValueError

        opacity = self.opacity_inverse_activation(self.cfg.init_opacity_value * torch.ones((xyz.shape[0], 1)))
        semantic = torch.zeros((xyz.shape[0], 3))
        
        instance            = torch.zeros((xyz.shape[0], 1))
        instance_location   = torch.tensor([[0., 0., 0.]])
        instance_size       = torch.tensor([[2., 2., 2.]])
        instance_rotation   = torch.tensor([[0., 0., 0.]])
        instance_class      = ["none"]
        instance_prompt     = ["3D asset"]

        return xyz, features_dc, features_rest, scaling, rotation, opacity, semantic, \
            instance, instance_location, instance_size, instance_rotation, instance_class, instance_prompt

    def _init_from_gaussians(self):
        assert os.path.isdir(self.cfg.init_file_path)

        # Gaussians
        xyz, features_dc, features_rest, opacity, scaling, rotation, max_sh_degree = \
            load_gaussians(os.path.join(self.cfg.init_file_path, "gaussians.ply"))
        
        xyz             = torch.tensor(xyz, dtype=torch.float32)
        features_dc     = torch.tensor(features_dc, dtype=torch.float32)
        features_rest   = torch.tensor(features_rest, dtype=torch.float32)
        opacity         = torch.tensor(opacity, dtype=torch.float32)
        scaling         = torch.tensor(scaling, dtype=torch.float32)
        rotation        = torch.tensor(rotation, dtype=torch.float32)

        self.active_sh_degree   = max_sh_degree
        self.max_sh_degree      = max_sh_degree
        if (max_sh_degree == 0) and (not self.use_sh):
            features_dc     = SH2RGB(features_dc[...,0])            # (N,3)
            features_rest   = torch.zeros((features_dc.shape[0],0)) # (N,0)
        else:
            self.use_sh     = True
            features_dc     = features_dc.transpose(1,2).contiguous().float()   # (N,1,3)
            features_rest   = features_rest.transpose(1,2).contiguous().float() # (N,(self.max_sh_degree+1)**2-1,3)

        # semantic & instance
        semantic = torch.load(os.path.join(self.cfg.init_file_path, "semantic.pth"))
        instance = torch.load(os.path.join(self.cfg.init_file_path, "instance.pth"))

        # layout
        instance_location, instance_size, instance_rotation, instance_class, instance_prompt = [], [], [], [], []
        with open(os.path.join(self.cfg.init_file_path, "layout.json"), 'r') as f:
            bbox = json.load(f)['bbox']
        for b in bbox:
            instance_location.append(torch.tensor(b['location']))
            instance_size.append(torch.tensor(b['size']))
            instance_rotation.append(torch.tensor(b['rotation']))
            instance_class.append(b['class'])
            instance_prompt.append(b['prompt'])
        
        instance_location   = torch.stack(instance_location)
        instance_size       = torch.stack(instance_size)
        instance_rotation   = torch.stack(instance_rotation)

        return xyz, features_dc, features_rest, scaling, rotation, opacity, semantic, \
            instance, instance_location, instance_size, instance_rotation, instance_class, instance_prompt

    def _init_from_layout(self):
        assert os.path.exists(self.cfg.init_file_path)

        instance_location, instance_size, instance_rotation, instance_class, instance_prompt = [], [], [], [], []

        # load layout
        with open(self.cfg.init_file_path, 'r') as f:
            bbox = json.load(f)['bbox']
        for b in bbox:
            instance_location.append(torch.tensor(b['location']))
            instance_size.append(torch.tensor(b['size']))
            instance_rotation.append(torch.tensor(b['rotation']))
            instance_class.append(b['class'])
            instance_prompt.append(b['prompt'])

        instance_location   = torch.stack(instance_location)
        instance_size       = torch.stack(instance_size)
        instance_rotation   = torch.stack(instance_rotation)

        # gaussians
        if self.cfg.is_local_space:
            n_pts       = [self.cfg.init_num_points for _ in instance_size]
        else:
            volumes     = torch.tensor([torch.prod(s) for s in instance_size])
            n_pts       = torch.ceil(volumes / volumes.sum() * self.cfg.init_num_points).long()
            n_pts[-1]   = self.cfg.init_num_points - n_pts[:-1].sum()
        
        xyz, scaling, rotation, semantic, instance = [], [], [], [], []
        for i, (n, loc, sz, rot, cls) in enumerate(zip(n_pts, instance_location, instance_size, instance_rotation, instance_class)):
            if self.cfg.init_point_strategy == 'random_cube':
                p = sampling_cube(n)
            elif self.cfg.init_point_strategy == 'random_ball':
                p = sampling_ball(n)
            elif self.cfg.init_point_strategy == 'random_sphere':
                p = sampling_ball(n, is_sphere=True)
            else:
                raise ValueError

            # [-1,1]^3 -> fit the instance size
            p = p * 0.5 * sz / sz.max() * self.cfg.radius * self.cfg.local_scale

            if self.cfg.init_rotation_strategy == 'random':
                rot = torch.rand((n, 4))
            elif self.cfg.init_rotation_strategy == 'radial':
                rot = get_quaternion_from_vectors(torch.tensor([[0.,0.,1.]]), F.normalize(p))
            else:
                raise ValueError
            rotation.append(rot)

            if not self.cfg.is_local_space:
                p = p / self.cfg.local_scale
                p = scale_rotate_translate(p, sz.max()[None], torch.deg2rad(rot), loc)
                rotation[-1] = quaternion_multiply(euler_to_quaternion(rot), rotation[-1])
            xyz.append(p)

            if self.cfg.init_scaling_strategy == 'nearest_neighbor':
                dist2   = torch.clamp_min(distCUDA2(p.cuda()), 0.0000001)
                dist    = torch.sqrt(dist2)[...,None].repeat(1, 2 if self.cfg.is_2dgs else 3)
                s       = self.scaling_inverse_activation(dist)
            elif self.cfg.init_scaling_strategy == 'fixed_value':
                s       = self.scaling_inverse_activation(
                    self.cfg.init_scaling_value * torch.ones((p.shape[0], 2 if self.cfg.is_2dgs else 3)))
            else:
                raise ValueError
            scaling.append(s)

            assert cls in ade20k_label2color.keys(), cls
            s_color = ade20k_label2color[cls]
            s = torch.ones((n, 3)) * torch.tensor([[s_color[0]/255., s_color[1]/255., s_color[2]/255.]])
            semantic.append(s)

            instance.append(torch.ones((n, 1)) * i)

        xyz         = torch.cat(xyz)
        scaling     = torch.cat(scaling)
        rotation    = torch.cat(rotation)
        semantic    = torch.cat(semantic)
        instance    = torch.cat(instance)

        if self.cfg.init_color_strategy == 'random':
            rgb = SH2RGB(torch.rand((xyz.shape[0], 3)) / 255.)
        else:
            raise ValueError

        if self.cfg.use_sh:
            self.active_sh_degree   = 0
            self.max_sh_degree      = self.cfg.sh_degree
            # (N,(self.max_sh_degree+1)**2,3)
            features                = torch.zeros((xyz.shape[0], (self.max_sh_degree+1)**2, 3))
            features[:,0]           = RGB2SH(rgb)
            features_dc             = features[:,:1].contiguous()
            features_rest           = features[:,1:].contiguous()
        else:
            assert self.cfg.sh_degree == 0
            # (N,3)
            features_dc             = rgb[:,:3].contiguous()
            features_rest           = torch.zeros((rgb.shape[0], 0))

        opacity = self.opacity_inverse_activation(self.cfg.init_opacity_value * torch.ones((xyz.shape[0], 1)))

        return xyz, features_dc, features_rest, scaling, rotation, opacity, semantic, \
            instance, instance_location, instance_size, instance_rotation, instance_class, instance_prompt

    def _init_from_mesh(self):
        if os.path.isfile(self.cfg.init_file_path):
            mesh = trimesh.load(self.cfg.init_file_path)
            assert (mesh.faces is not None) and len(mesh.faces) > 0
            xyz, face_index, rgb = trimesh.sample.sample_surface(mesh, self.cfg.init_num_points, sample_color=True)

            xyz = torch.from_numpy(xyz).float()
            xyz *= self.cfg.radius

            if self.cfg.init_color_strategy == 'random':
                rgb = SH2RGB(torch.rand((xyz.shape[0], 3)) / 255.)
            elif self.cfg.init_color_strategy == 'from_mesh':
                rgb = torch.from_numpy(rgb).float()[:,:-1] / 255.
            else:
                raise ValueError
            
            if self.use_sh:
                self.active_sh_degree   = 0
                self.max_sh_degree      = self.cfg.sh_degree
                # (N,(self.max_sh_degree+1)**2,3)
                features                = torch.zeros((xyz.shape[0], (self.max_sh_degree+1)**2, 3))
                features[:,0]           = RGB2SH(rgb)
                features_dc             = features[:,:1].contiguous()
                features_rest           = features[:,1:].contiguous()
            else:
                assert self.cfg.sh_degree == 0
                # (N,3)
                features_dc             = rgb[:,:3].contiguous()
                features_rest           = torch.zeros((rgb.shape[0], 0))
        
            if self.cfg.init_scaling_strategy == 'nearest_neighbor':
                dist2   = torch.clamp_min(distCUDA2(xyz.cuda()), 0.0000001)
                dist    = torch.sqrt(dist2)[...,None].repeat(1, 2 if self.cfg.is_2dgs else 3)
                scaling = self.scaling_inverse_activation(dist)
            elif self.cfg.init_scaling_strategy == 'fixed_value':
                scaling = self.scaling_inverse_activation(
                    self.cfg.init_scaling_value * torch.ones((xyz.shape[0], 2 if self.cfg.is_2dgs else 3)))
            else:
                raise ValueError

            if self.cfg.init_rotation_strategy == 'random':
                rotation = torch.rand((xyz.shape[0], 4))
            elif self.cfg.init_rotation_strategy == 'radial':
                rotation = get_quaternion_from_vectors(torch.tensor([[0.,0.,1.]]), F.normalize(xyz))
            elif self.cfg.init_rotation_strategy == 'from_mesh':
                normals = torch.from_numpy(mesh.face_normals[face_index]).float()
                rotation = get_quaternion_from_vectors(torch.tensor([[0.,0.,1.]]), F.normalize(normals))
            else:
                raise ValueError

            opacity = self.opacity_inverse_activation(self.cfg.init_opacity_value * torch.ones((xyz.shape[0], 1)))
            semantic = torch.zeros((xyz.shape[0], 3))
            
            instance            = torch.zeros((xyz.shape[0], 1))
            instance_location   = torch.tensor([[0., 0., 0.]])
            instance_size       = torch.tensor([[2., 2., 2.]])
            instance_rotation   = torch.tensor([[0., 0., 0.]])
            instance_class      = ["none"]
            instance_prompt     = ["3D asset"]

        elif os.path.isdir(self.cfg.init_file_path):
            # load layout
            instance_location, instance_size, instance_rotation, instance_class, instance_prompt = [], [], [], [], []
            with open(os.path.join(self.cfg.init_file_path, '..', 'layout.json'), 'r') as f:
                bbox = json.load(f)['bbox']
            for b in bbox:
                instance_location.append(torch.tensor(b['location']))
                instance_size.append(torch.tensor(b['size']))
                instance_rotation.append(torch.tensor(b['rotation']))
                instance_class.append(b['class'])
                instance_prompt.append(b['prompt'])

            instance_location   = torch.stack(instance_location)
            instance_size       = torch.stack(instance_size)
            instance_rotation   = torch.stack(instance_rotation)
            
            # mesh
            mesh_paths  = glob(os.path.join(self.cfg.init_file_path, '*'))
            mesh_paths  = sorted(mesh_paths, key=lambda s: int(os.path.basename(s).split('.')[0]))
            meshes      = [trimesh.load_mesh(p) for p in mesh_paths]
            assert hasattr(meshes[0], 'faces') and (meshes[0].faces is not None) and len(meshes[0].faces) > 0
            
            if self.cfg.is_local_space:
                n_pts       = [self.cfg.init_num_points for _ in instance_size]
            else:
                volumes     = torch.tensor([torch.prod(s) for s in instance_size])
                n_pts       = torch.ceil(volumes / volumes.sum() * self.cfg.init_num_points).long()
                n_pts[-1]   = self.cfg.init_num_points - n_pts[:-1].sum()
            
            xyz, features_dc, features_rest, scaling, rotation, semantic, instance = [], [], [], [], [], [], []

            for i, (mesh, n, cls) in enumerate(zip(meshes, n_pts, instance_class)):
                # load & sample mesh
                if hasattr(mesh.visual, 'uv') and (mesh.visual.uv is not None):
                    p, face_index, c = trimesh.sample.sample_surface(mesh, n, sample_color=True)
                else:
                    p, face_index = trimesh.sample.sample_surface(mesh, n)
                    c = np.zeros((p.shape[0], 4))

                p = torch.from_numpy(p).float()
                xyz.append(p)

                if self.cfg.init_rotation_strategy == 'random':
                    rot = torch.rand((n, 4))
                elif self.cfg.init_rotation_strategy == 'radial':
                    rot = get_quaternion_from_vectors(torch.tensor([[0.,0.,1.]]), F.normalize(p))
                elif self.cfg.init_rotation_strategy == 'from_mesh':
                    normals = torch.from_numpy(mesh.face_normals[face_index]).float()
                    rot = get_quaternion_from_vectors(torch.tensor([[0.,0.,1.]]), F.normalize(normals))
                else:
                    raise ValueError
                rotation.append(rot)

                if self.cfg.init_color_strategy == 'random':
                    rgb = SH2RGB(torch.rand((xyz.shape[0], 3)) / 255.)
                elif self.cfg.init_color_strategy == 'from_mesh':
                    rgb = torch.from_numpy(c).float()[:,:-1] / 255.
                else:
                    raise ValueError

                if self.use_sh:
                    self.active_sh_degree   = 0
                    self.max_sh_degree      = self.cfg.sh_degree
                    # (N,(self.max_sh_degree+1)**2,3)
                    features                = torch.zeros((xyz.shape[0], (self.max_sh_degree+1)**2, 3))
                    features[:,0]           = RGB2SH(rgb)
                    f_dc                    = features[:,:1].contiguous()
                    f_rest                  = features[:,1:].contiguous()
                else:
                    assert self.cfg.sh_degree == 0
                    # (N,3)
                    f_dc                    = rgb[:,:3].contiguous()
                    f_rest                  = torch.zeros((rgb.shape[0], 0))
                features_dc.append(f_dc)
                features_rest.append(f_rest)

                if self.cfg.init_scaling_strategy == 'nearest_neighbor':
                    dist2   = torch.clamp_min(distCUDA2(p.cuda()), 0.0000001)
                    dist    = torch.sqrt(dist2)[...,None].repeat(1, 2 if self.cfg.is_2dgs else 3)
                    s       = self.scaling_inverse_activation(dist)
                elif self.cfg.init_scaling_strategy == 'fixed_value':
                    s       = self.scaling_inverse_activation(
                        self.cfg.init_scaling_value * torch.ones((p.shape[0], 2 if self.cfg.is_2dgs else 3)))
                else:
                    raise ValueError
                scaling.append(s)

                assert cls in ade20k_label2color.keys(), cls
                s_color = ade20k_label2color[cls]
                s = torch.ones((n, 3)) * torch.tensor([[s_color[0]/255., s_color[1]/255., s_color[2]/255.]])
                semantic.append(s)

                instance.append(torch.ones((n, 1)) * i)

            xyz             = torch.cat(xyz)
            features_dc     = torch.cat(features_dc)
            features_rest   = torch.cat(features_rest)
            scaling         = torch.cat(scaling)
            rotation        = torch.cat(rotation)
            semantic        = torch.cat(semantic)
            instance        = torch.cat(instance)

        else:
            raise ValueError
        
        return xyz, features_dc, features_rest, scaling, rotation, opacity, semantic, \
            instance, instance_location, instance_size, instance_rotation, instance_class, instance_prompt

    # parameters
    @property
    def get_xyz(self): 
        return self._xyz

    @property
    def get_features(self): 
        return torch.cat((self._features_dc, self._features_rest), dim=1)
    
    @property
    def get_semantic(self): 
        return self._semantic
    
    @property
    def get_instance(self):
        return self._instance

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_global_params(self):
        global_xyz      = self.get_xyz.clone()
        global_opacity  = self.get_opacity
        global_scaling  = self.get_scaling.clone()
        global_rotation = self.get_rotation.clone()
        global_features = self.get_features
        global_semantic = self.get_semantic
        global_instance = self.get_instance

        if self.cfg.is_local_space: 
            for i, (loc, rot, sz) in enumerate(zip(self.instance_location, self.instance_rotation, self.instance_size)):
                instance_mask = global_instance[...,0] == i

                global_xyz[instance_mask]       = \
                    scale_rotate_translate(global_xyz[instance_mask] / self.cfg.local_scale, sz.max()[None], torch.deg2rad(rot), loc)

                global_scaling[instance_mask]   = \
                    global_scaling[instance_mask] / self.cfg.local_scale * sz.max() 

                global_rotation[instance_mask]  = \
                    quaternion_multiply(euler_to_quaternion(rot), global_rotation[instance_mask])

        return \
            global_xyz, \
            global_opacity, \
            global_scaling, \
            global_rotation, \
            global_features, \
            global_semantic, \
            global_instance, \
            None

    def get_local_params(self, index):
        local_mask = self.get_instance[...,0] == index

        # Gaussian params
        local_xyz       = self.get_xyz[local_mask]
        local_opacity   = self.get_opacity[local_mask]
        local_scaling   = self.get_scaling[local_mask] 
        local_rotation  = self.get_rotation[local_mask]
        local_features  = self.get_features[local_mask]
        local_semantic  = self.get_semantic[local_mask]
        local_instance  = self.get_instance[local_mask]

        if not self.cfg.is_local_space:
            # local transformation params
            loc = self.instance_location[index]
            rot = self.instance_rotation[index]
            sz  = self.instance_size[index].max()[None]

            # transform to local space
            local_xyz       = translate_rotate_scale(local_xyz, self.cfg.local_scale / sz, torch.deg2rad(-rot), - loc)
            local_scaling   = local_scaling * self.cfg.local_scale / sz
            local_rotation  = quaternion_multiply(euler_to_quaternion(-rot), local_rotation)

        return \
            local_xyz, \
            local_opacity, \
            local_scaling, \
            local_rotation, \
            local_features, \
            local_semantic, \
            local_instance, \
            local_mask

    def oneup_sh_degree(self):
        if self.use_sh and (self.active_sh_degree < self.max_sh_degree):
            self.active_sh_degree += 1

    # optimization
    @torch.no_grad()
    def gradient_synchronization(self):
        if (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()): return

        for k in self.optimizable_keys:
            param = getattr(self, k)
            if param.grad is not None:
                torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
                param.grad.data /= torch.distributed.get_world_size()

    # scheduler
    def configure_schedulers(self, cfg):

        def _get_lr_func(param_name):
            attr = f'{param_name}_lr_schedule'
            if not hasattr(cfg, attr): return None
            cfg_lr_schedule = getattr(cfg, attr)
            lr_init         = cfg_lr_schedule.lr_init
            lr_final        = cfg_lr_schedule.lr_final
            lr_delay_steps  = getattr(cfg_lr_schedule, 'lr_delay_steps', 0)
            lr_delay_mult   = getattr(cfg_lr_schedule, 'lr_delay_mult', 1)
            lr_max_steps    = getattr(cfg_lr_schedule, 'lr_max_steps', 1000000)

            def expon_lr_func(step):
                """ from 3DGS """
                if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
                    # Disable this parameter
                    return 0.0
                if lr_delay_steps > 0:
                    # A kind of reverse cosine decay.
                    delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                        0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
                    )
                else:
                    delay_rate = 1.0
                t = np.clip(step / lr_max_steps, 0, 1)
                log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
                return delay_rate * log_lerp

            return expon_lr_func
        
        self._xyz_lr_func           = _get_lr_func('_xyz')
        self._features_dc_lr_func   = _get_lr_func('_features_dc')
        self._features_rest_lr_func = _get_lr_func('_features_rest')
        self._scaling_lr_func       = _get_lr_func('_scaling')
        self._rotation_lr_func      = _get_lr_func('_rotation')
        self._opacity_lr_func       = _get_lr_func('_opacity')
        self.background_lr_func     = _get_lr_func('background')

    def scheduler_step(self, optimizer, step):

        if self._xyz_lr_func is not None:
            for group in optimizer.param_groups:
                if group["name"].split('.')[-1] != '_xyz': continue
                group['lr'] = self._xyz_lr_func(step)
                break

        if self._features_dc_lr_func is not None:
            for group in optimizer.param_groups:
                if group["name"].split('.')[-1] != '_features_dc': continue
                group['lr'] = self._features_dc_lr_func(step)
                break
        
        if self._features_rest_lr_func is not None:
            for group in optimizer.param_groups:
                if group["name"].split('.')[-1] != '_features_rest': continue
                group['lr'] = self._features_rest_lr_func(step)
                break
        
        if self._scaling_lr_func is not None:
            for group in optimizer.param_groups:
                if group["name"].split('.')[-1] != '_scaling': continue
                group['lr'] = self._scaling_lr_func(step)
                break

        if self._rotation_lr_func is not None:
            for group in optimizer.param_groups:
                if group["name"].split('.')[-1] != '_rotation': continue
                group['lr'] = self._rotation_lr_func(step)
                break
        
        if self._opacity_lr_func is not None:
            for group in optimizer.param_groups:
                if group["name"].split('.')[-1] != '_opacity': continue
                group['lr'] = self._opacity_lr_func(step)
                break

        if self.background_lr_func is not None:
            for group in optimizer.param_groups:
                if group["name"].split('.')[-1] != 'background': continue
                group['lr'] = self.background_lr_func(step)
                break

    # density control
    @torch.no_grad()
    def density_control(self, true_global_step, optimizer, viewspace_points, radii, image_sizes):
        if (self.cfg.densify_until_step < 0) or (true_global_step < self.cfg.densify_until_step):
            for _viewspace_points, _radii, _image_size in zip(viewspace_points, radii, image_sizes):
                if _viewspace_points.grad is None: continue
                self.__add_densification_stats(_viewspace_points, _radii, _image_size)

            # densify and prune
            if true_global_step > self.cfg.densify_from_step:
                if true_global_step % self.cfg.densify_interval == 0:
                    self._densify_and_prune(optimizer)

            # opacity reset
            if (self.cfg.opacity_reset_interval > 0) and \
                (true_global_step % self.cfg.opacity_reset_interval == 0) and \
                (true_global_step != 0):
                self.reset_opacity(optimizer)

    def __add_densification_stats(self, viewspace_points, radii, image_size):
        grads = viewspace_points.absgrad.clone() if hasattr(viewspace_points, 'absgrad') else \
                viewspace_points.grad.clone()
        
        if self.cfg.normalize_grad:
            n_cam, height, width = grads.shape[0], *image_size
            grads[..., 0] *= height / 2.0 * n_cam
            grads[..., 1] *= width / 2.0 * n_cam

        selected    = radii > 0.                # (n_cam, n_point)
        gs_ids      = torch.where(selected)[1]  # (nnz)
        grads       = grads[selected]           # (nnz, 2 or 3)

        local_grads = torch.zeros_like(self.xyz_grad_accum)
        local_grads.index_add_(0, gs_ids, grads.norm(dim=-1))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(local_grads, op=torch.distributed.ReduceOp.SUM)
        self.xyz_grad_accum += local_grads

        local_denom = torch.zeros_like(self.denom)
        local_denom.index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(local_denom, op=torch.distributed.ReduceOp.SUM)
        self.denom          += local_denom

        self.max_radii2D[gs_ids] = torch.maximum(self.max_radii2D[gs_ids], radii[selected])
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.max_radii2D, op=torch.distributed.ReduceOp.MAX)

    def reset_opacity(self, optimizer):
        new_opacity = self.opacity_inverse_activation(
                        torch.min(self.get_opacity, self.cfg.init_opacity_value * torch.ones_like(self.get_opacity)))
        optimizable_tensors = self._replace_tensor_to_optimizer(optimizer, new_opacity, "_opacity")
        self._opacity = optimizable_tensors["_opacity"]

    def _replace_tensor_to_optimizer(self, optimizer, tensor, name):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if group["name"].split('.')[-1] == name:
                stored_state = optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"].split('.')[-1]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, optimizer, prune_mask):
        valid_mask          = ~prune_mask
        optimizable_tensors = self._prune_optimizer(optimizer, valid_mask)

        self._xyz           = optimizable_tensors['_xyz']
        self._features_dc   = optimizable_tensors['_features_dc']
        self._features_rest = optimizable_tensors['_features_rest']
        self._rotation      = optimizable_tensors['_rotation']
        self._opacity       = optimizable_tensors['_opacity']
        self._scaling       = optimizable_tensors['_scaling']

        self._semantic      = nn.Parameter(self._semantic[valid_mask].requires_grad_(False))
        self._instance      = nn.Parameter(self._instance[valid_mask].requires_grad_(False))

        self.xyz_grad_accum = self.xyz_grad_accum[valid_mask]
        self.denom          = self.denom[valid_mask]
        self.max_radii2D    = self.max_radii2D[valid_mask]

    def _prune_optimizer(self, optimizer, mask):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if group["name"].split('.')[-1] not in self.optimizable_keys: continue  
            assert len(group["params"]) == 1

            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                if isinstance(optimizer, torch.optim.Adam) or isinstance(optimizer, torch.optim.AdamW):
                    stored_state["exp_avg"]         = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"]      = stored_state["exp_avg_sq"][mask]
                elif isinstance(optimizer, threestudio.systems.optimizers.Adan):
                    stored_state["exp_avg"]         = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"]      = stored_state["exp_avg_sq"][mask]
                    stored_state["exp_avg_diff"]    = stored_state["exp_avg_diff"][mask]
                    stored_state["neg_pre_grad"]    = stored_state["neg_pre_grad"][mask]
                else:
                    raise NotImplementedError
                
                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"].split('.')[-1]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"].split('.')[-1]] = group["params"][0]

        return optimizable_tensors

    def densify_postfix(
        self, 
        optimizer, 
        new_xyz, 
        new_features_dc, 
        new_features_rest, 
        new_semantic, 
        new_instance,
        new_scaling, 
        new_rotation, 
        new_opacity,
        new_xyz_grad_accum,
        new_denom,
        new_max_radii2D):
        optimizable_tensors = self._concat_tensors_to_optimizer(optimizer, {
            '_xyz':             new_xyz,
            '_features_dc':     new_features_dc,
            '_features_rest':   new_features_rest,
            '_scaling':         new_scaling,
            '_rotation':        new_rotation,
            '_opacity':         new_opacity,
        })

        self._xyz           = optimizable_tensors['_xyz']
        self._features_dc   = optimizable_tensors['_features_dc']
        self._features_rest = optimizable_tensors['_features_rest']
        self._rotation      = optimizable_tensors['_rotation']
        self._opacity       = optimizable_tensors['_opacity']
        self._scaling       = optimizable_tensors['_scaling']

        self._semantic      = nn.Parameter(torch.cat((self._semantic, new_semantic), dim=0).requires_grad_(False))
        self._instance      = nn.Parameter(torch.cat((self._instance, new_instance), dim=0).requires_grad_(False))

        self.xyz_grad_accum = torch.cat((self.xyz_grad_accum, new_xyz_grad_accum), dim=0)
        self.denom          = torch.cat((self.denom, new_denom), dim=0)
        self.max_radii2D    = torch.cat((self.max_radii2D, new_max_radii2D), dim=0)

    def _concat_tensors_to_optimizer(self, optimizer, dict_tensors):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if group["name"].split('.')[-1] not in self.optimizable_keys: continue
            assert len(group["params"]) == 1

            extension_tensor = dict_tensors[group["name"].split('.')[-1]]
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                if isinstance(optimizer, torch.optim.Adam) or isinstance(optimizer, torch.optim.AdamW):
                    stored_state["exp_avg"]         = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"]      = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                elif isinstance(optimizer, threestudio.systems.optimizers.Adan):
                    stored_state["exp_avg"]         = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"]      = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_diff"]    = torch.cat((stored_state["exp_avg_diff"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["neg_pre_grad"]    = torch.cat((stored_state["neg_pre_grad"], torch.zeros_like(extension_tensor)), dim=0)
                else:
                    raise NotImplementedError

                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"].split('.')[-1]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"].split('.')[-1]] = group["params"][0]

        return optimizable_tensors

    def _densify_and_clone(self, optimizer, xyz_grad_avg):
        selected = torch.where(xyz_grad_avg >= self.cfg.densify_grad_threshold, True, False)
        selected = torch.logical_and(selected, self.get_scaling.max(dim=-1).values <= self.cfg.percent_dense)

        new_xyz             = self._xyz[selected]
        new_features_dc     = self._features_dc[selected]
        new_features_rest   = self._features_rest[selected]
        new_semantic        = self._semantic[selected]
        new_instance        = self._instance[selected]
        new_scaling         = self._scaling[selected]
        new_rotation        = self._rotation[selected]
        new_opacity         = self._opacity[selected]
        new_xyz_grad_accum  = self.xyz_grad_accum[selected]
        new_denom           = self.denom[selected]
        new_max_radii2D     = self.max_radii2D[selected]

        self.densify_postfix(
            optimizer, 
            new_xyz, 
            new_features_dc, 
            new_features_rest, 
            new_semantic, 
            new_instance,
            new_scaling, 
            new_rotation,
            new_opacity,
            new_xyz_grad_accum,
            new_denom,
            new_max_radii2D)

    def _densify_and_split(self, optimizer, xyz_grad_avg, N=2):
        n_pts = self.get_xyz.shape[0]
        padded_xyz_grad_avg = torch.zeros((n_pts)).to(xyz_grad_avg)
        padded_xyz_grad_avg[:xyz_grad_avg.shape[0]] = xyz_grad_avg

        selected = torch.where(padded_xyz_grad_avg >= self.cfg.densify_grad_threshold, True, False)
        selected = torch.logical_and(selected, self.get_scaling.max(dim=-1).values > self.cfg.percent_dense)

        stds    = self.get_scaling[selected].repeat(N,1)
        if self.cfg.is_2dgs:
            stds = torch.cat([stds, torch.zeros_like(stds[:,:1])], dim=-1)
        means   = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)

        rots                = quaternion_to_rotation_matrix(self.get_rotation[selected]).repeat(N,1,1)
        new_xyz             = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected].repeat(N, 1)
        new_features_dc     = self._features_dc[selected].repeat(N,1,1) if self.use_sh else self._features_dc[selected].repeat(N,1)
        new_features_rest   = self._features_rest[selected].repeat(N,1,1) if self.use_sh else self._features_rest[selected].repeat(N,1)
        new_semantic        = self._semantic[selected].repeat(N,1)
        new_instance        = self._instance[selected].repeat(N,1)
        new_scaling         = self.scaling_inverse_activation(self.get_scaling[selected].repeat(N,1) / (0.8*N))
        new_rotation        = self._rotation[selected].repeat(N,1)
        new_opacity         = self._opacity[selected].repeat(N,1)
        new_xyz_grad_accum  = self.xyz_grad_accum[selected].repeat(N)
        new_denom           = self.denom[selected].repeat(N)
        new_max_radii2D     = self.max_radii2D[selected].repeat(N)

        self.densify_postfix(
            optimizer, 
            new_xyz, 
            new_features_dc, 
            new_features_rest, 
            new_semantic,
            new_instance, 
            new_scaling, 
            new_rotation, 
            new_opacity,
            new_xyz_grad_accum,
            new_denom,
            new_max_radii2D)
        
        prune_mask = torch.cat((selected, torch.zeros(N * selected.sum(), device=selected.device, dtype=bool)))
        self.prune_points(optimizer, prune_mask)

    def _densify_and_prune(self, optimizer):
        xyz_grad_avg = self.xyz_grad_accum / self.denom.clamp_min(1)

        self._densify_and_clone(optimizer, xyz_grad_avg)
        self._densify_and_split(optimizer, xyz_grad_avg)

        prune_mask = (self.get_opacity < self.cfg.opacity_threshold).squeeze()

        if self.cfg.view_size_threshold > 0.:
            big_points_vs = self.max_radii2D > self.cfg.view_size_threshold
            prune_mask = torch.logical_or(prune_mask, big_points_vs)

        if self.cfg.world_size_threshold > 0.:
            big_points_ws = self.get_scaling.max(dim=-1).values > self.cfg.world_size_threshold
            prune_mask = torch.logical_or(prune_mask, big_points_ws)

        if self.cfg.prune_out_of_box:
            inbox_mask = torch.zeros_like(prune_mask)
            for i, (loc, rot, sz) in enumerate(zip(self.instance_location, self.instance_rotation, self.instance_size)):
                instance_mask   = self._instance[...,0] == i
                instance_xyz    = self._xyz[instance_mask]
                if self.cfg.is_local_space:
                    instance_xyz = instance_xyz / (sz / sz.max()) / self.cfg.local_scale
                else:
                    instance_xyz = translate_rotate_scale(instance_xyz, self.cfg.local_scale / sz, torch.deg2rad(-rot), - loc)
                inbox_mask[instance_mask] = (instance_xyz.abs() < 0.5).all(dim=-1)
            prune_mask = torch.logical_or(prune_mask, ~inbox_mask)

        self.prune_points(optimizer, prune_mask)

        self.xyz_grad_accum = torch.zeros((self.get_xyz.shape[0])).to(self._xyz)
        self.denom          = torch.zeros((self.get_xyz.shape[0])).to(self._xyz)
        self.max_radii2D    = torch.zeros((self.get_xyz.shape[0])).to(self._xyz)

        torch.cuda.empty_cache()

    # regularization
    def total_variation(self, return_feature=True, return_rotation=True):
        xyz = self.get_xyz  # (n,3)

        with torch.no_grad():
            dist2 = (xyz[:,None,:] - xyz[None,:,:]).pow(exponent=2).sum(dim=-1)
            knn_dist2, knn_indics = torch.topk(dist2, k=10, largest=False, sorted=False)
        n, m = knn_indics.shape

        # 
        loss_feature = None
        if return_feature:
            features = self.get_features.reshape(n,-1)
            with torch.no_grad():
                knn_features = features[knn_indics.reshape(-1)].reshape(n,m,-1)

            loss_feature = torch.sum((features[:,None,:] - knn_features).pow_(exponent=2))

        # 
        loss_rotation = None
        if return_rotation:
            rotation = self.get_rotation
            with torch.no_grad():
                knn_rotation = rotation[knn_indics.reshape(-1)].reshape(n,m,-1)

            loss_rotation = torch.sum((rotation[:,None,:] - knn_rotation).pow_(exponent=2))

        return loss_feature, loss_rotation

        # with torch.no_grad():
        #     nn_dist, nn_idx = self.knn(xyz[None], xyz[None])
        #     # (n,m), (n,m) where m is the number of neighbors
        #     nn_dist, nn_idx = nn_dist[0], nn_idx[0]
        
        # print(nn_dist, nn_idx)
        # n, m = nn_idx.shape

        # features        = self.get_features.reshape(n,-1)
        # # with torch.no_grad():
        # nn_features     = features[nn_idx.reshape(-1)].reshape(n, m, -1).detach()
        # print(features.shape, nn_features.shape, " <---------------")
        # print(nn_features)
        # loss_feature    = torch.mean(features[:,None,:].repeat(1,m,1) - nn_features)
        # print(loss_feature.shape)
        # # loss_feature    = (features[:,None,:] - nn_features) ** 2
        # # loss_feature    = loss_feature.mean()
        # print(loss_feature.item())

    # ouptut
    def export(self, path, fmt='pointcloud'):
        assert fmt in ['pointcloud', 'gaussians', 'mesh']

        if fmt == 'pointcloud':
            os.makedirs(os.path.dirname(path), exist_ok=True)
            vertices        = self._xyz.detach().cpu().numpy()
            vertex_colors   = SH2RGB(self._features_dc[:,0].detach().cpu().numpy()) if self.use_sh else \
                                self._features_dc.detach().cpu().numpy()
            trimesh.Trimesh(vertices=vertices, vertex_colors=vertex_colors).export(path)

        elif fmt == 'gaussians':
            os.makedirs(path, exist_ok=True)

            # Gaussians
            save_gaussians(
                path=os.path.join(path, "gaussians.ply"),
                xyz=self._xyz,
                features_dc=self._features_dc.permute((0,2,1)) if self.use_sh else RGB2SH(self._features_dc[:,:,None]),
                features_rest=self._features_rest.permute((0,2,1)) if self.use_sh else RGB2SH(self._features_rest[:,:,None]),
                scaling=self._scaling,
                rotation=self._rotation,
                opacity=self._opacity
            )

            # semantic & instance
            torch.save(self._semantic, os.path.join(path, "semantic.pth"))
            torch.save(self._instance, os.path.join(path, "instance.pth"))

            # layout
            bbox = []
            for loc, sz, rot, cla, prompt in \
                zip(self.instance_location, self.instance_size, self.instance_rotation, self.instance_class, self.instance_prompt):

                bbox.append({
                    "class": cla,
                    "prompt": prompt,
                    "location": loc.detach().cpu().numpy().tolist(),
                    "size": sz.detach().cpu().numpy().tolist(),
                    "rotation": rot.detach().cpu().numpy().tolist()
                })

            with open(os.path.join(path, "layout.json"), 'w') as f:
                json.dump({"bbox": bbox}, f)

        elif fmt == 'mesh':
            os.makedirs(path, exist_ok=True)

            for i in range(self.num_instance):
                instance_mask = self._instance[...,0] == i
                mesh = gaussian2mesh(
                    gaussian_params = {
                        "xyz":      self._xyz[instance_mask],
                        "scaling":  self._scaling[instance_mask],
                        "rotation": self._rotation[instance_mask],
                        "opacity":  self._opacity[instance_mask],
                    }
                )
                mesh.export(os.path.join(path, f"{i}.ply"))

        else:
            raise NotImplementedError
        
        if self.cfg.init_strategy == 'from_layout':
            shutil.copy2(self.cfg.init_file_path, osp.join(osp.dirname(path), 'layout.json'))