import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from threestudio.utils.ops import dot


def uv_unwarpping_xatlas(
    v_pos: torch.Tensor,
    t_pos_idx: torch.Tensor,
    xatlas_chart_options: dict = {},
    xatlas_pack_options: dict = {}
):
    assert isinstance(v_pos, torch.Tensor) and isinstance(t_pos_idx, torch.Tensor)
    device = v_pos.device

    # torch.Tensor -> np.ndarray
    v_pos       = v_pos.detach().cpu().numpy()
    t_pos_idx   = t_pos_idx.detach().cpu().numpy()

    # uv unwarpping
    import xatlas

    vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

    # atlas = xatlas.Atlas()
    # atlas.add_mesh(v_pos, t_pos_idx)

    # co = xatlas.ChartOptions()
    # po = xatlas.PackOptions()
    # for k, v in xatlas_chart_options.items():
    #     setattr(co, k, v)
    # for k, v in xatlas_pack_options.items():
    #     setattr(po, k, v)
    # atlas.generate(co, po)
    # vmapping, indices, uvs = atlas.get_mesh(0)

    # np.ndarray -> torch.Tensor 
    vmapping = (
        torch.from_numpy(
            vmapping.astype(np.uint64, casting="same_kind").view(np.int64)
        )
        .to(device)
        .long()
    )
    indices = (
        torch.from_numpy(
            indices.astype(np.uint64, casting="same_kind").view(np.int64)
        )
        .to(device)
        .long()
    )
    uvs = torch.from_numpy(uvs).to(device).float()

    return vmapping, indices, uvs


def compute_vertex_normal(vertices, faces):
    i0, i1, i2 = faces[:,0], faces[:,1], faces[:,2]
    v0, v1, v2 = vertices[i0,:], vertices[i2,:], vertices[i2,:]

    face_normals = torch.cross(v1 - v0, v2 - v0)

    # Splat face normals to vertices
    v_nrm = torch.zeros_like(vertices)
    v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    v_nrm = torch.where(
        dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.as_tensor([0.0, 0.0, 1.0]).to(v_nrm)
    )
    v_nrm = F.normalize(v_nrm, dim=1)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_nrm))

    return v_nrm