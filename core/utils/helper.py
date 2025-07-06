import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

import threestudio
from threestudio.utils.typing import *

import trimesh

HF_ROOT = os.environ.get('HF_ROOT')
HF_PATH = lambda p: osp.join(HF_ROOT, p) if (HF_ROOT is not None) and (not osp.exists(p)) else p


def get_quaternion_from_vectors(
    v1: Num[Tensor, "N 3"], 
    v2: Num[Tensor, "N 3"]
):
    # quaternion representing the rotation from vector v1 to vector v2
    theta   = torch.acos(torch.sum(v1*v2, dim=1, keepdim=True))
    axis    = torch.sin(theta * 0.5) * F.normalize(torch.cross(v1, v2))
    w       = torch.cos(theta * 0.5)
    q       = torch.cat([w, axis], dim=1)
    return q

def euler_to_quaternion(euler: torch.Tensor, convention: str = 'XYZ', degree: bool = True) -> torch.Tensor:
    assert convention == 'XYZ'

    euler_rad = torch.deg2rad(euler) if degree else euler
    roll, pitch, yaw = euler_rad.unbind(-1)

    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([w,x,y,z], dim=-1)

def quaternion_to_rotation_matrix(q: Num[Tensor, "N 4"]):
    r, x, y, z = q.unbind(-1)

    R = torch.zeros((q.shape[0], 3, 3)).to(q)
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)

    return R

def quaternion_multiply(q1: Num[Tensor, "N 4"], q2: Num[Tensor, "N 4"]):
    r1, x1, y1, z1 = q1.unbind(-1)
    r2, x2, y2, z2 = q2.unbind(-1)

    r = r1*r2 - x1*x2 - y1*y2 - z1*z2
    x = r1*x2 + x1*r2 + y1*z2 - z1*y2
    y = r1*y2 - x1*z2 + y1*r2 + z1*x2
    z = r1*z2 + x1*y2 - y1*x2 + z1*r2

    return torch.stack([r,x,y,z], dim=-1)

def fov2focal(fov, pixels):
    return 0.5 * pixels / math.tan(0.5 * fov)

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def euler_to_rotation_matrix(euler: torch.Tensor, convention: str = "XYZ", degree: bool = False) -> torch.Tensor:
    assert (euler.dim() > 0) and (euler.shape[-1] == 3)
    
    euler_rad = torch.deg2rad(euler) if degree else euler

    angles = euler_rad.unbind(dim=-1)
    if convention == "ZYX":
        # yaw (Z), pitch (Y), roll (X)
        yaw, pitch, roll = angles
    elif convention == "XYZ":
        # roll (X), pitch (Y), yaw (Z)
        roll, pitch, yaw = angles
    else:
        raise ValueError
    
    # Z
    cos_z = torch.cos(yaw)
    sin_z = torch.sin(yaw)
    zeros = torch.zeros_like(yaw)
    ones = torch.ones_like(yaw)
    R_z = torch.stack([
        cos_z, -sin_z, zeros,
        sin_z,  cos_z, zeros,
        zeros,  zeros, ones
    ], dim=-1).reshape(*euler_rad.shape[:-1], 3, 3)
    
    # Y
    cos_y = torch.cos(pitch)
    sin_y = torch.sin(pitch)
    R_y = torch.stack([
         cos_y, zeros,  sin_y,
         zeros, ones,   zeros,
        -sin_y, zeros,  cos_y
    ], dim=-1).reshape(*euler_rad.shape[:-1], 3, 3)
    
    # X
    cos_x = torch.cos(roll)
    sin_x = torch.sin(roll)
    R_x = torch.stack([
        ones,  zeros,   zeros,
        zeros, cos_x,  -sin_x,
        zeros, sin_x,   cos_x
    ], dim=-1).reshape(*euler_rad.shape[:-1], 3, 3)
    
    # rotation
    if convention == "ZYX":
        rotation_matrix = R_x @ R_y @ R_z
    elif convention == "XYZ":
        rotation_matrix = R_z @ R_y @ R_x
    
    return rotation_matrix

def scale_rotate_translate(xyz: torch.Tensor, scale: torch.Tensor, euler: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    N = xyz.shape[0]

    if scale.ndim <= 1: scale = scale[None].repeat((N,1))
    if euler.ndim <= 1: euler = euler[None].repeat((N,1))
    if translation.ndim <= 1: translation = translation[None].repeat((N,1))

    rot_mat = euler_to_rotation_matrix(euler)

    xyz = xyz * scale
    xyz = torch.bmm(rot_mat, xyz[...,None])[...,0]
    xyz = xyz + translation

    return xyz

def translate_rotate_scale(xyz: torch.Tensor, scale: torch.Tensor, euler: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    N = xyz.shape[0]

    if scale.ndim <= 1: scale = scale[None].repeat((N,1))
    if euler.ndim <= 1: euler = euler[None].repeat((N,1))
    if translation.ndim <= 1: translation = translation[None].repeat((N,1))

    rot_mat = euler_to_rotation_matrix(euler)

    xyz = xyz + translation
    xyz = torch.bmm(rot_mat, xyz[...,None])[...,0]
    xyz = xyz * scale

    return xyz

def load_mesh(path: str):
    assert osp.exists(path), path

    ext = os.path.splitext(path)[1].lower()
    assert ext in ['.glb', '.ply'], ext

    mesh = trimesh.load(path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()

    if ext == '.glb':
        rotation = trimesh.transformations.rotation_matrix(angle=np.pi*0.5, direction=[1, 0, 0])
        mesh = mesh.apply_transform(rotation)
    
    return mesh

def load_meshes(paths: list):
    vertices, faces = [], []
    n_verts = 0
    for p in paths:
        mesh = trimesh.load(p)
        v, f = np.array(mesh.vertices), np.array(mesh.faces)
        vertices.append(v)
        faces.append(f + n_verts)
        n_verts += v.shape[0]
    vertices = np.concatenate(vertices)
    faces = np.concatenate(faces)

    return trimesh.Trimesh(vertices=vertices, faces=faces)

def parse_background_meshes(
    ceiling_meshes = [],
    floor_meshes = [],
    wall_meshes = [],
    ceiling_semantic = (120/255, 120/255, 80/255),
    floor_semantic = (80/255, 50/255, 50/255),
    wall_semantic = (120/255, 120/255, 120/255)
):
    vertices, faces, normals, semantics = [], [], [], []

    n_verts = 0
    # ceiling
    for m in ceiling_meshes:
        v, f, n = np.array(m.vertices), np.array(m.faces), np.array(m.vertex_normals)
        vertices.append(v)
        faces.append(f + n_verts)
        normals.append(n)
        semantics.append([ceiling_semantic] * v.shape[0])
        n_verts += v.shape[0]
    # floor
    for m in floor_meshes:
        v, f, n = np.array(m.vertices), np.array(m.faces), np.array(m.vertex_normals)
        vertices.append(v)
        faces.append(f + n_verts)
        normals.append(n)
        semantics.append([floor_semantic] * v.shape[0])
        n_verts += v.shape[0]
    # wall
    for m in wall_meshes:
        v, f, n = np.array(m.vertices), np.array(m.faces), np.array(m.vertex_normals)
        vertices.append(v)
        faces.append(f + n_verts)
        normals.append(n)
        semantics.append([wall_semantic] * v.shape[0])
        n_verts += v.shape[0]
    
    assert (len(vertices)>0) and (len(faces)>0) and (len(normals)>0) and (len(semantics)>0)
    vertices    = np.concatenate(vertices)
    faces       = np.concatenate(faces)
    normals     = np.concatenate(normals)
    semantics   = np.concatenate(semantics)
    return vertices, faces, normals, semantics

def parse_background_mesh(
    background_mesh, 
    ceiling_semantic = (120/255, 120/255, 80/255),
    floor_semantic = (80/255, 50/255, 50/255),
    wall_semantic = (120/255, 120/255, 120/255)
):
    z_coords = [v[2] for v in background_mesh.vertices]
    z_max, z_min = max(z_coords), min(z_coords)

    # separate all faces into ceiling, floor and wall categories
    ceiling_faces, floor_faces, walls_faces = [], [], []
    for f in background_mesh.faces:
        v = background_mesh.vertices[f]

        all_verts_max_z = all(abs(i[2] - z_max) < 1e-4 for i in v)
        all_verts_min_z = all(abs(i[2] - z_min) < 1e-4 for i in v)

        if all_verts_max_z:
            ceiling_faces.append(f)
        elif all_verts_min_z:
            floor_faces.append(f)
        else:
            walls_faces.append(f)

    ceiling_meshes = []
    for ceiling_face in ceiling_faces:
        ceiling_mesh = trimesh.Trimesh(vertices=background_mesh.vertices, faces=[ceiling_face])
        ceiling_mesh.remove_unreferenced_vertices()
        ceiling_meshes.append(ceiling_mesh)
    
    floor_meshes = []
    for floor_face in floor_faces:
        floor_mesh = trimesh.Trimesh(vertices=background_mesh.vertices, faces=[floor_face])
        floor_mesh.remove_unreferenced_vertices()
        floor_meshes.append(floor_mesh)

    walls_meshes = []
    for walls_face in walls_faces:
        walls_mesh = trimesh.Trimesh(vertices=background_mesh.vertices, faces=[walls_face])
        walls_mesh.remove_unreferenced_vertices()
        walls_meshes.append(walls_mesh)

    return parse_background_meshes(ceiling_meshes, floor_meshes, walls_meshes, ceiling_semantic, floor_semantic, wall_semantic)