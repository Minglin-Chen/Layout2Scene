import math
import torch
import torch.nn.functional as F
import mcubes
import trimesh
import pymeshlab as pml
from typing import Callable
from .gaussian_utils import load_gaussians


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float).to(L)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3)).to(r)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

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


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float).to(s)
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2] if s.shape[1] == 3 else s.amin(1)

    L = R @ L
    return L


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm


def gaussian_3d_coeff(xyzs, covs):
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = covs[:, 0], covs[:, 1], covs[:, 2], covs[:, 3], covs[:, 4], covs[:, 5]

    # eps must be small enough !!!
    inv_det = 1 / (a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24)
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f) - x * y * inv_b - x * z * inv_c - y * z * inv_e

    power[power > 0] = -1e10 # abnormal values... make weights 0
        
    return torch.exp(power)


def decimate_mesh(
    verts, faces, target, backend="pymeshlab", remesh=False, optimalplacement=True
):
    # optimalplacement: default is True, but for flat mesh must turn False to prevent spike artifect.

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    if backend == "pyfqmr":
        import pyfqmr

        solver = pyfqmr.Simplify()
        solver.setMesh(verts, faces)
        solver.simplify_mesh(target_count=target, preserve_border=False, verbose=False)
        verts, faces, normals = solver.getMesh()
    else:
        m = pml.Mesh(verts, faces)
        ms = pml.MeshSet()
        ms.add_mesh(m, "mesh")  # will copy!

        # filters
        # ms.meshing_decimation_clustering(threshold=pml.PercentageValue(1))
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target), optimalplacement=optimalplacement
        )

        if remesh:
            # ms.apply_coord_taubin_smoothing()
            ms.meshing_isotropic_explicit_remeshing(
                iterations=3, targetlen=pml.PercentageValue(1)
            )

        # extract mesh
        m = ms.current_mesh()
        verts = m.vertex_matrix()
        faces = m.face_matrix()

    print(
        f"[INFO] mesh decimation: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}"
    )

    return verts, faces


def clean_mesh(
    verts,
    faces,
    v_pct=1,
    min_f=64,
    min_d=20,
    repair=True,
    remesh=True,
    remesh_size=0.01,
):
    # verts: [N, 3]
    # faces: [N, 3]

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, "mesh")  # will copy!

    # filters
    ms.meshing_remove_unreferenced_vertices()  # verts not refed by any faces

    if v_pct > 0:
        ms.meshing_merge_close_vertices(
            threshold=pml.PercentageValue(v_pct)
        )  # 1/10000 of bounding box diagonal

    ms.meshing_remove_duplicate_faces()  # faces defined by the same verts
    ms.meshing_remove_null_faces()  # faces with area == 0

    if min_d > 0:
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pml.PercentageValue(min_d)
        )

    if min_f > 0:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_f)

    if repair:
        # ms.meshing_remove_t_vertices(method=0, threshold=40, repeat=True)
        ms.meshing_repair_non_manifold_edges(method=0)
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

    if remesh:
        # ms.apply_coord_taubin_smoothing()
        ms.meshing_isotropic_explicit_remeshing(
            iterations=3, targetlen=pml.PureValue(remesh_size)
        )

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(
        f"[INFO] mesh cleaning: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}"
    )

    return verts, faces


@torch.no_grad()
def gaussian2mesh(
    gaussian_params: dict, 
    scaling_activation_fn: Callable=torch.exp,
    rotation_activation_fn: Callable=F.normalize,
    opacity_activation_fn: Callable=torch.sigmoid,
    resolution: int=128, 
    num_blocks: int=16,
    relax_ratio: float=1.5,
    density_thresh: float=1.0,
    decimate_target: float=1e5
):
    device = gaussian_params['xyz'].device

    block_size = 2. / num_blocks
    assert resolution % block_size == 0
    split_size = resolution // num_blocks

    # filter low opacity gaussians
    opacity     = opacity_activation_fn(gaussian_params['opacity'])
    mask        = (opacity > 0.005).squeeze(1)

    xyz         = gaussian_params['xyz'][mask]
    scaling     = scaling_activation_fn(gaussian_params['scaling'])[mask]
    rotation    = rotation_activation_fn(gaussian_params['rotation'])[mask]
    opacity     = opacity[mask]

    # normalize to ~ [-1, 1]
    mn, mx  = xyz.amin(0), xyz.amax(0)
    center  = (mn + mx) * 0.5
    scale   = 1.8 / (mx - mn).amax().item()

    xyz     = (xyz - center) * scale
    scaling = scaling * scale

    cov     = build_covariance_from_scaling_rotation(scaling, 1.0, rotation)

    # tile
    occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

    X   = torch.linspace(-1, 1, resolution).split(split_size)
    Y   = torch.linspace(-1, 1, resolution).split(split_size)
    Z   = torch.linspace(-1, 1, resolution).split(split_size)

    # loop blocks (assume max size of gaussian is small than relax_ratio * block_size !!!)
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)
                # sample points [M, 3]
                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                # in-tile gaussians mask
                vmin, vmax = pts.amin(0), pts.amax(0)
                vmin -= block_size * relax_ratio
                vmax += block_size * relax_ratio
                mask = (xyz < vmax).all(-1) & (xyz > vmin).all(-1)
                # if hit no gaussian, continue to next block
                if not mask.any(): continue

                mask_xyzs = xyz[mask] # [L, 3]
                mask_covs = cov[mask] # [L, 6]
                mask_opas = opacity[mask].view(1, -1) # [L, 1] --> [1, L]

                # query per point-gaussian pair.
                g_pts = pts.unsqueeze(1).repeat(1, mask_covs.shape[0], 1) - mask_xyzs.unsqueeze(0) # [M, L, 3]
                g_covs = mask_covs.unsqueeze(0).repeat(pts.shape[0], 1, 1) # [M, L, 6]

                # batch on gaussian to avoid OOM
                batch_g = 1024
                val = 0
                for start in range(0, g_covs.shape[1], batch_g):
                    end = min(start + batch_g, g_covs.shape[1])
                    w = gaussian_3d_coeff(g_pts[:, start:end].reshape(-1, 3), g_covs[:, start:end].reshape(-1, 6)).reshape(pts.shape[0], -1) # [M, l]
                    val += (mask_opas[:, start:end] * w).sum(-1)
                
                occ[xi * split_size: xi * split_size + len(xs), 
                    yi * split_size: yi * split_size + len(ys), 
                    zi * split_size: zi * split_size + len(zs)] = val.reshape(len(xs), len(ys), len(zs))
    
    # marching cube
    vertices, triangles = mcubes.marching_cubes(occ.detach().cpu().numpy(), density_thresh)
    vertices = vertices / (resolution - 1.0) * 2.0 - 1.0

    # transform back to the original space
    vertices = vertices / scale + center.detach().cpu().numpy()
    
    # post-processing
    # vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=0.015)
    if decimate_target > 0 and triangles.shape[0] > decimate_target:
        vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)

    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    return mesh


if __name__=='__main__':
    gaussians_path = 'gaussians.ply'
    xyz, features_dc, features_extra, opacities, scales, rots, max_sh_degree = load_gaussians(gaussians_path)

    mesh = gaussian2mesh(
        gaussian_params = {
            "xyz": torch.from_numpy(xyz).float().cuda(),
            "scaling": torch.from_numpy(scales).float().cuda(),
            "rotation": torch.from_numpy(rots).float().cuda(),
            "opacity": torch.from_numpy(opacities).float().cuda(),
        }
    )

    mesh.unwrap()

    mesh.export("mesh.obj")