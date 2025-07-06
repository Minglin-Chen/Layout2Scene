import torch
import numpy as np
import math


def uv_packing(uv, i, n, spacing=0.05):
    if isinstance(uv, torch.Tensor):
        uv = uv.clone()
    elif isinstance(uv, np.ndarray):
        uv = uv.copy()
    else:
        raise NotImplementedError

    # m is the samllest integer satisfying m^2 >= n
    m = math.ceil(math.sqrt(n)) 
    i_row, i_col = i // m, i % m

    # [0, 1] -> [spacing, 1. - 2. * spacing]
    uv[...,0] = uv[...,0] * (1. - 2. * spacing) + spacing
    uv[...,1] = uv[...,1] * (1. - 2. * spacing) + spacing

    # [spacing, 1. - 2. * spacing] -> [min_i, max_i]
    uv[...,0] = uv[...,0] / m + i_col / m
    uv[...,1] = uv[...,1] / m + i_row / m
    
    return uv


def uv_unpacking(uv, i, n, spacing=0.05):
    if isinstance(uv, torch.Tensor):
        uv = uv.clone()
    elif isinstance(uv, np.ndarray):
        uv = uv.copy()
    else:
        raise NotImplementedError

    # m is the samllest integer satisfying m^2 >= n
    m = math.ceil(math.sqrt(n)) 
    i_row, i_col = i // m, i % m

    # [min_i, max_i] -> [spacing, 1. - 2. * spacing]
    uv[...,0] = uv[...,0] * m - i_col
    uv[...,1] = uv[...,1] * m - i_row

    # [spacing, 1. - 2. * spacing] -> [0, 1]
    uv[...,0] = (uv[...,0] - spacing) / (1. - 2. * spacing)
    uv[...,1] = (uv[...,1] - spacing) / (1. - 2. * spacing)

    return uv