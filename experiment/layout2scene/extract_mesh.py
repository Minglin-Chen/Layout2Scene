""" python -m gs2mesh """
import os
import os.path as osp
import json
from PIL import Image
import open3d as o3d
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from threestudio.utils.ops import get_cam_info_gaussian
from core.utils.gaussian_utils import load_gaussians
from core.utils.helper import *
from core.utils.point_utils import depth_to_normal

import gsplat
import xatlas


def load_gaussians_from_layout(path: str, device: str):
    torch.cuda.set_device(device)

    scaling_activation  = torch.exp
    rotation_activation = F.normalize
    opacity_activation  = torch.sigmoid

    assert osp.isdir(path), path

    # Gaussians
    xyz, features_dc, features_rest, opacity, scaling, rotation, max_sh_degree = \
        load_gaussians(os.path.join(path, "gaussians.ply"))

    xyz             = torch.tensor(xyz, dtype=torch.float32, device="cuda")
    features_dc     = torch.tensor(features_dc, dtype=torch.float32, device="cuda")
    features_rest   = torch.tensor(features_rest, dtype=torch.float32, device="cuda")
    opacity         = torch.tensor(opacity, dtype=torch.float32, device="cuda")
    scaling         = torch.tensor(scaling, dtype=torch.float32, device="cuda")
    rotation        = torch.tensor(rotation, dtype=torch.float32, device="cuda")

    assert (features_dc.ndim == 3) and (features_rest.ndim == 3)
    features_dc     = features_dc.transpose(1,2).contiguous().float()   # (N,1,3)
    features_rest   = features_rest.transpose(1,2).contiguous().float() # (N,(self.max_sh_degree+1)**2-1,3)

    # semantic & instance
    semantic = torch.load(os.path.join(path, "semantic.pth")).to(device)
    instance = torch.load(os.path.join(path, "instance.pth")).to(device)

    # layout
    instance_location, instance_size, instance_rotation, instance_class, instance_prompt = [], [], [], [], []
    with open(os.path.join(path, "layout.json"), 'r') as f:
        bbox = json.load(f)['bbox']
    for b in bbox:
        instance_location.append(torch.tensor(b['location'], device="cuda"))
        instance_size.append(torch.tensor(b['size'], device="cuda"))
        instance_rotation.append(torch.tensor(b['rotation'], device="cuda"))
        instance_class.append(b['class'])
        instance_prompt.append(b['prompt'])
    
    instance_location   = torch.stack(instance_location)
    instance_size       = torch.stack(instance_size)
    instance_rotation   = torch.stack(instance_rotation)

    features    = torch.cat((features_dc, features_rest), dim=1)
    scaling     = scaling_activation(scaling)
    rotation    = rotation_activation(rotation)
    opacity     = opacity_activation(opacity)

    return xyz, features, scaling, rotation, opacity, semantic, max_sh_degree, \
        instance, instance_location, instance_size, instance_rotation, instance_class, instance_prompt


def get_cameras(
    elevations: Float[Tensor, "B"], 
    azimuths: Float[Tensor, "B"], 
    radius: float, 
    is_degree: bool = True
) -> Float[Tensor, "B 4 4"]:
    if is_degree:
        elevations  = torch.deg2rad(elevations)
        azimuths    = torch.deg2rad(azimuths)

    positions = torch.stack([
        - radius * torch.cos(elevations) * torch.cos(azimuths),
        - radius * torch.cos(elevations) * torch.sin(azimuths),
        - radius * torch.sin(elevations)
    ], dim=-1)

    up      = torch.as_tensor([[0,0,1]], dtype=torch.float32)
    lookat  = torch.stack([
        torch.cos(elevations) * torch.cos(azimuths),
        torch.cos(elevations) * torch.sin(azimuths),
        torch.sin(elevations)
    ], dim=-1)
    right   = F.normalize(torch.linalg.cross(lookat, up), dim=-1)
    up      = F.normalize(torch.linalg.cross(right, lookat), dim=-1)
    
    c2w3x4  = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), positions[:, :, None]],
        dim=-1,
    )
    c2w     = torch.cat(
        [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
    )
    c2w[:, 3, 3] = 1.0
    return c2w


@torch.no_grad()
def gs_render_3d(
    xyz, features, opacity, scaling, rotation, sh_degree,
    c2w, fovx, fovy, height, width, znear=0.01, zfar=100.0
):
    # camera intrinsic
    focal       = fov2focal(fovy, height)
    Ks          = torch.zeros((3, 3)).to(xyz)
    Ks[0, 0] = focal
    Ks[1, 1] = focal
    Ks[0, 2] = 0.5 * width
    Ks[1, 2] = 0.5 * height
    Ks[2, 2] = 1.0

    # camera pose
    world_view_transform, full_proj_transform, camera_center = \
        get_cam_info_gaussian(c2w, fovx, fovy, znear, zfar)
    viewmats = world_view_transform.transpose(0, 1)

    # rendering
    rendered_results, rendered_alpha, meta = gsplat.rasterization(
        means=xyz, quats=rotation, scales=scaling, opacities=opacity.squeeze(dim=-1), colors=features,
        viewmats=viewmats[None], Ks=Ks[None], width=width, height=height, near_plane=znear, far_plane=zfar,
        sh_degree=sh_degree,
        packed=False, absgrad=False, 
        rasterize_mode='classic', # options: classic, antialiased
        distributed=False,
        render_mode='RGB+D', # options: RGB+D, RGB+ED
    )
    rendered_results    = rendered_results[0].permute((2,0,1))
    rendered_alpha      = rendered_alpha[0].permute((2,0,1))
    rendered_image      = rendered_results[:-1].clone()
    rendered_depth      = rendered_results[-1:].clone()

    normal = depth_to_normal(
        world_view_transform, full_proj_transform, rendered_depth, 'camera')
    normal = normal.permute(2,0,1)
    normal = normal * rendered_alpha

    return rendered_image, rendered_alpha, rendered_depth, normal


def render_multiview_images(
    is_2dgs, xyz, features, scaling, rotation, opacity, sh_degree, c2ws, fovx, fovy, height, width
):
    rgbs, masks, depths, normals = [], [], [], []
    for c2w in c2ws:
        if is_2dgs:
            raise NotImplementedError
        else:
            rgb, mask, depth, normal = gs_render_3d(
                xyz, features, opacity, scaling, rotation, sh_degree, c2w, fovx, fovy, height, width
            )
            
        rgbs.append(rgb)
        masks.append(mask)
        depths.append(depth)
        normals.append(normal)

    return rgbs, masks, depths, normals


def save_image(image, path):
    assert image.ndim == 3
    image = image.permute((1,2,0)).detach().cpu().numpy()
    np.clip(image, 0.0, 1.0)
    image = (image * 255.).astype(np.uint8)

    if image.shape[-1] == 1:
        image = image[:,:,0]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(image).save(path)


def extract_mesh_from_tsdf_volume(
    rgbs, depths, c2ws, fovx, fovy, height, width, scale, uv_unwarp=False,
    voxel_size=0.004, sdf_trunc=0.02, depth_trunc=30, impl='uniform_tsdf_volume'
):
    volume = None
    if impl == 'uniform_tsdf_volume':
        interval = scale * 0.2
        length = scale + interval
        volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=length, # voxel_length = length / resolution
            resolution=128, # voxel_length = length / resolution
            # voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,

        )
    elif impl == 'scalable_tsdf_volume':
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
    else:
        raise ValueError

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    fx, fy = fov2focal(fovx, width), fov2focal(fovy, height)
    cx, cy = width * 0.5, height * 0.5
    intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

    # for rgb, depth, c2w in tqdm(zip(rgbs, depths, c2ws), desc="TSDF integration progress"):
    print('Depth fusion')
    for rgb, depth, c2w in zip(rgbs, depths, c2ws):
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.asarray(np.clip(rgb.permute(1,2,0).cpu().numpy(), 0.0, 1.0) * 255, order="C", dtype=np.uint8)),
            o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
            depth_trunc = depth_trunc, 
            convert_rgb_to_intensity = False,
            depth_scale = 1.0
        )
        c2w = c2w.detach().cpu().numpy().copy()
        c2w[:3,-1] += length * 0.5
        c2w = np.linalg.inv(c2w)
        c2w = np.array([
            [1,  0,  0, 0], 
            [0, -1,  0, 0], 
            [0,  0, -1, 0], 
            [0,  0,  0, 1]
        ], dtype=np.float32) @ c2w
        volume.integrate(rgbd, intrinsic=intrinsic, extrinsic=c2w)

    print('Mesh extraction')
    o3d_mesh        = volume.extract_triangle_mesh()
    vertices        = np.asarray(o3d_mesh.vertices)
    vertices        = vertices - length * 0.5
    faces           = np.asarray(o3d_mesh.triangles)
    vertex_normals  = np.asarray(o3d_mesh.vertex_normals) if o3d_mesh.has_vertex_normals() else None

    if uv_unwarp:
        print('UV unwarping (may take a while ...)')
        vmapping, faces, uv = xatlas.parametrize(vertices, faces)
        vertices            = vertices[vmapping]
        vertex_normals      = vertex_normals[vmapping] if vertex_normals is not None else vertex_normals

    tri_mesh            = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=vertex_normals, process=False)

    if uv_unwarp:
        tri_mesh.visual.uv  = uv

    return tri_mesh


def mesh_extraction(
    device, indics, 
    input_path, output_path, 
    is_2dgs, is_local_space, local_scale, 
    height, width, fovy_deg, radius, 
    uv_unwarp, to_world
):
    # 1. load Gaussians
    xyz, features, scaling, rotation, opacity, _, max_sh_degree, \
        instance, instance_location, instance_size, instance_rotation, _, instance_prompt = \
            load_gaussians_from_layout(input_path, device)

    # 2. compute poses & intrinsics
    elevations              = torch.linspace(-60, 60, 10)
    azimuths                = torch.linspace(0, 360, 33)[:-1]
    elevations, azimuths    = torch.meshgrid(elevations, azimuths, indexing='ij')
    elevations, azimuths    = elevations.reshape(-1), azimuths.reshape(-1)
    c2ws                    = get_cameras(elevations, azimuths, radius)

    fovy                    = torch.deg2rad(torch.tensor(fovy_deg))
    focal                   = fov2focal(fovy, height)
    fovx                    = focal2fov(focal, width)

    # 3. extraction
    meshes = []
    for index in indics:
        local_mask = instance[...,0] == index

        # Gaussian params
        local_xyz       = xyz[local_mask]
        local_opacity   = opacity[local_mask]
        local_scaling   = scaling[local_mask]
        local_rotation  = rotation[local_mask]
        local_features  = features[local_mask]

        # local transformation params
        loc = instance_location[index]
        rot = instance_rotation[index]
        sz  = instance_size[index].max()[None]
        
        if not is_local_space:
            # transform to local space
            local_xyz       = translate_rotate_scale(local_xyz, local_scale / sz, torch.deg2rad(-rot), - loc)
            local_scaling   = local_scaling * local_scale / sz
            local_rotation  = quaternion_multiply(euler_to_quaternion(-rot), local_rotation)

        # render multiview images
        print('Render multiview depth maps')
        rgbs, masks, depths, normals = render_multiview_images(
            is_2dgs, local_xyz, local_features, local_scaling, local_rotation, local_opacity, max_sh_degree,
            c2ws, fovx, fovy, height, width)

        # extract mesh
        mesh = extract_mesh_from_tsdf_volume(rgbs, depths, c2ws, fovx, fovy, height, width, local_scale, uv_unwarp)
        if to_world:
            mesh.apply_scale(sz.item())
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(rot[2].item()), [0, 0, 1]))
            mesh.apply_translation(loc.cpu().numpy().tolist())
        mesh_path = os.path.join(output_path, f'{index}.ply')
        os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
        mesh.export(mesh_path)

        # transform to scene
        if not to_world:
            mesh.apply_scale(sz.item())
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(rot[2].item()), [0, 0, 1]))
            mesh.apply_translation(loc.cpu().numpy().tolist())
        meshes.append(mesh)

        # # debug
        # for i, (rgb, mask, depth, normal) in enumerate(zip(rgbs, masks, depths, normals)):
        #     save_image(rgb, os.path.join(output_path, 'images', f'{index}', f'rgb_{i}.jpg'))
        #     save_image(mask, os.path.join(output_path, 'images', f'{index}', f'mask_{i}.jpg'))

        #     depth = (depth - depth.min()) / (depth.max() - depth.min())
        #     save_image(depth, os.path.join(output_path, 'images', f'{index}', f'depth_{i}.jpg'))

        #     normal = (normal + 1.0) * 0.5
        #     save_image(normal, os.path.join(output_path, 'images', f'{index}', f'normal_{i}.jpg'))

    return meshes
    

def background_extraction(background):
    # background mesh
    ceiling = trimesh.Trimesh(vertices=background["vertices"], faces=background["faces"]["ceiling"][::-1])
    ceiling.remove_unreferenced_vertices()

    floor   = trimesh.Trimesh(vertices=background["vertices"], faces=background["faces"]["floor"][::-1])
    floor.remove_unreferenced_vertices()

    wall    = trimesh.Trimesh(vertices=background["vertices"], faces=background["faces"]["walls"][::-1])
    wall_meshes = []
    for wall_face in wall.faces:
        wall_mesh = trimesh.Trimesh(vertices=background["vertices"], faces=[wall_face])
        wall_mesh.remove_unreferenced_vertices()
        wall_meshes.append(wall_mesh)
    wall    = trimesh.util.concatenate(wall_meshes)

    background_mesh = trimesh.util.concatenate([ceiling, floor, wall])
    return background_mesh


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='gaussians-it4000-test', help='input path')
    parser.add_argument('--output_path', type=str, default='gs2mesh-it4000-test', help='output path')
    parser.add_argument('--layout_path', type=str, default='', help='layout path')
    # Gaussian
    parser.add_argument('--is_2dgs', action='store_true', help='2DGS or 3DGS')
    parser.add_argument('--is_local_space', action='store_true', help='parse the file in local space')
    parser.add_argument('--local_scale', type=float, default=1.0, help='local scale')
    # Camera poses & intrinsic
    parser.add_argument('--height', type=int, default=1024, help='camera height')
    parser.add_argument('--width', type=int, default=1024, help='camera width')
    parser.add_argument('--fovy_deg', type=float, default=45.0, help='camera y-axis fov (degree)')
    parser.add_argument('--radius', type=float, default=3.0, help='camera radius')
    # Others
    parser.add_argument("--uv_unwarp", action="store_true", help="enable UV unwarpping")
    parser.add_argument("--to_world", action="store_true", help="transform mesh to world space")
    parser.add_argument('--device', type=str, default='0', help='GPU device')
    args = parser.parse_args()

    # load layout
    assert osp.isfile(args.layout_path)
    with open(os.path.join(args.layout_path), 'r') as f:
        data = json.load(f)
        bbox, background = data['bbox'], data['background']

        background_mesh = background_extraction(background)
        background_mesh.export(os.path.join(args.output_path, '..', f'background-{osp.basename(args.output_path)}.ply'))

        num_instance = len(bbox)
        
    # multi-processsing for mesh extraction
    def distribute(num_instance, num_device):
        indics_list = [[] for _ in range(num_device)]
        for i in range(num_instance):
            indics_list[i%num_device].append(i)
        return indics_list
    
    device_list     = [f'cuda:{s}' for s in args.device.split(',')]
    num_device      = len(device_list)
    indics_list     = distribute(num_instance, num_device)

    with mp.get_context('spawn').Pool(processes=num_device) as pool:
        meshes_list = pool.starmap(
            partial(
                mesh_extraction,
                input_path=args.input_path, 
                output_path=args.output_path,
                is_2dgs=args.is_2dgs,
                is_local_space=args.is_local_space,
                local_scale=args.local_scale,
                height=args.height,
                width=args.width,
                fovy_deg=args.fovy_deg,
                radius=args.radius,
                uv_unwarp=args.uv_unwarp,
                to_world=args.to_world,
            ), zip(device_list, indics_list))

    scene_meshes = []
    for meshes in meshes_list: 
        scene_meshes += meshes

    # scene_meshes.append(background_mesh)
    trimesh.util.concatenate(scene_meshes).export(
        os.path.join(args.output_path, '..', f'scene-{osp.basename(args.output_path)}.ply'))
    background_mesh.export(
        os.path.join(args.output_path, '..', f'background-{osp.basename(args.output_path)}.ply'))