import numpy as np
import torch

def depths_to_points(world_view_transform, full_proj_transform, depthmap, space='world'):
    assert space in ['world', 'camera']

    assert depthmap.ndim == 3
    H, W = depthmap.shape[-2:]

    c2w = (world_view_transform.T).inverse()
    projection_matrix = c2w.T @ full_proj_transform
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().to(depthmap).T
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W).float(), torch.arange(H).float(), indexing='xy')
    grid_x, grid_y = grid_x.to(depthmap), grid_y.to(depthmap)
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    if space == 'world':
        rays_d = points @ intrins.float().inverse().T @ c2w[:3,:3].T
        rays_o = c2w[:3,3]
    else:
        rays_d = points @ intrins.float().inverse().T
        rays_o = torch.zeros((3)).to(depthmap)

    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(world_view_transform, full_proj_transform, depth, space='world'):
    points = depths_to_points(
        world_view_transform, full_proj_transform, depth, space).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    if space == 'camera':
        # camera coordinate: OpenCV (up:-y,right:+x,forward:+z) -> OpenGL (up:+y,right:+x,forward:-z)
        # flip x-axis
        output *= -1.0
    return output

def sampling_cube(n):
    # [-1,1]^3
    point = torch.rand((n, 3)) * 2. - 1.
    return point

def sampling_ball(n, is_sphere=False):
    phis        = torch.rand((n,)) * 2 * np.pi
    costheta    = torch.rand((n,)) * 2 - 1
    thetas      = torch.acos(costheta)
    r           = 1. if is_sphere else torch.rand((n,)) ** (1./3.)
    # [-1,1]^3
    point       = torch.stack((
                    r * torch.sin(thetas) * torch.cos(phis), 
                    r * torch.sin(thetas) * torch.sin(phis), 
                    r * torch.cos(thetas)), dim=1)
    return point