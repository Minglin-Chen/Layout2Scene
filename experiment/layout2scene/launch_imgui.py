# -*- coding: utf-8 -*-
import argparse
import logging
import time
import os
import os.path as osp
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from PIL import Image
from glob import glob

import glfw
import OpenGL.GL as gl

import imgui
from imgui.integrations.glfw import GlfwRenderer

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

import threestudio
from threestudio.systems.base import BaseSystem
from threestudio.utils.config import ExperimentConfig, load_config
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import get_mvp_matrix, get_projection_matrix
from threestudio.utils.typing import *

from experiment.layout2scene.launch_layout2scene import load_custom_modules
from core.utils.gaussian_utils import load_gaussians


def load_system(args, extras):
    # set CUDA_VISIBLE_DEVICES if needed, then import pytorch-lightning
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
    selected_gpus = [0]

    # Always rely on CUDA_VISIBLE_DEVICES if specific GPU ID(s) are specified.
    # As far as Pytorch Lightning is concerned, we always use all available GPUs
    # (possibly filtered by CUDA_VISIBLE_DEVICES).
    devices = -1
    if len(env_gpus) > 0:
        # CUDA_VISIBLE_DEVICES was set already, e.g. within SLURM srun or higher-level script.
        n_gpus = len(env_gpus)
    else:
        selected_gpus = list(args.gpu.split(","))
        n_gpus = len(selected_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # You are using a CUDA device ('NVIDIA GeForce RTX 3090') that has Tensor Cores. 
    # To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` 
    # which will trade-off precision for performance. 
    # For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision('medium')

    load_custom_modules()

    # parse YAML config to OmegaConf
    cfg: ExperimentConfig
    cfg = load_config(args.config, cli_args=extras, n_gpus=n_gpus)

    # handle the dynamic number of Gaussian points
    gaussian_path = glob(osp.join(
        osp.dirname(osp.dirname(args.config)), 'save', 'export', '*', 'gaussians.ply'))
    assert len(gaussian_path) == 1, gaussian_path
    gaussian_path = gaussian_path[0]
    xyz = load_gaussians(gaussian_path)[0]
    cfg.system.geometry.init_num_points = xyz.shape[0]

    # set a different seed for each device
    pl.seed_everything(cfg.seed + get_rank(), workers=True)

    system: BaseSystem = threestudio.find(cfg.system_type)(cfg.system)

    return system


class Camera:
    def __init__(
            self, 
            position: list = [0, 0, 0],
            elevation_deg: float = 0,
            azimuth_deg: float = 0,
            fovy_deg: float = 50,
            width: int = 512,
            height: int = 512,
            device: str = 'cuda:0'
        ):
        self.position: Float[Tensor, "3"]       = torch.as_tensor(position, dtype=torch.float32, device=device)
        self.elevation_deg: Float[Tensor, "1"]  = torch.as_tensor([elevation_deg], dtype=torch.float32, device=device)
        self.azimuth_deg: Float[Tensor, "1"]    = torch.as_tensor([azimuth_deg], dtype=torch.float32, device=device)
        self.fovy_deg: float                    = fovy_deg

        self.width: int                         = width
        self.height: int                        = height

        self.up: Float[Tensor, "3"]             = torch.as_tensor([0, 0, 1], dtype=torch.float32, device=device)

        self.move_speed                         = 0.1
        self.rotate_speed                       = 1.0

    def __lookat_right_up(self):
        elevation   = self.elevation_deg / 180. * np.pi
        azimuth     = self.azimuth_deg / 180. * np.pi
        lookat: Float[Tensor, "3"] = torch.cat([
            torch.cos(elevation) * torch.cos(azimuth),
            torch.cos(elevation) * torch.sin(azimuth),
            torch.sin(elevation),
        ])
        right: Float[Tensor, "3"] = F.normalize(torch.cross(lookat, self.up), dim=0)
        up = F.normalize(torch.cross(right, lookat), dim=0)
        return lookat, right, up

    def get_dict(self):
        lookat, right, up = self.__lookat_right_up()

        # c2w
        c2w3x4: Float[Tensor, "3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), self.position[:, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:1])], dim=0
        )
        c2w[3, 3] = 1.0

        # mvp
        fovy = torch.tensor([self.fovy_deg / 180. * np.pi]).to(c2w)
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.01, 100.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w.unsqueeze(dim=0), proj_mtx.to(c2w))

        ret_dict = {
            "mvp_mtx":              mvp_mtx,
            "camera_positions":     self.position.unsqueeze(dim=0),
            "c2w":                  c2w.unsqueeze(dim=0),
            "elevation":            self.elevation_deg,
            "azimuth":              self.azimuth_deg,
            "camera_distances":     self.position.unsqueeze(dim=0).norm(dim=-1),
            "height":               self.height,
            "width":                self.width,
            "fovy":                 fovy,
        }
        return ret_dict

    def move(self, direction: str):
        lookat, right, up = self.__lookat_right_up()
        if direction == 'forward':
            self.position += self.move_speed * lookat
        elif direction == 'backward':
            self.position -= self.move_speed * lookat
        elif direction == 'right':
            self.position += self.move_speed * right
        elif direction == 'left':
            self.position -= self.move_speed * right
        elif direction == 'up':
            self.position += self.move_speed * up
        elif direction == 'down':
            self.position -= self.move_speed * up

    def rotate(self, direction: str):
        if direction == 'up':
            self.elevation_deg += self.rotate_speed
        elif direction == 'down':
            self.elevation_deg -= self.rotate_speed
        elif direction == 'left':
            self.azimuth_deg += self.rotate_speed
        elif direction == 'right':
            self.azimuth_deg -= self.rotate_speed

    def save(self, path):
        data_dict = {
            "position":         self.position.cpu().numpy().tolist(),
            "elevation_deg":    self.elevation_deg.cpu().numpy().tolist(),
            "azimuth_deg":      self.azimuth_deg.cpu().numpy().tolist(),
            "fovy_deg":         self.fovy_deg,
            "width":            self.width,
            "height":           self.height
        }
        with open(path, 'w') as f:
            json.dump(data_dict, f)

    def load(self, path):
        with open(path, 'r') as f:
            data_dict = json.load(f)
        device              = self.position.device

        self.position       = torch.as_tensor(data_dict['position'], dtype=torch.float32, device=device)
        self.elevation_deg  = torch.as_tensor(data_dict['elevation_deg'], dtype=torch.float32, device=device)
        self.azimuth_deg    = torch.as_tensor(data_dict['azimuth_deg'], dtype=torch.float32, device=device)
        self.fovy_deg       = data_dict['fovy_deg']
        self.width          = data_dict['width']
        self.height         = data_dict['height']
        

class InteractiveGUI:

    def __init__(self, system, device_id=0, win_width=1280, win_height=720, win_name='GUI'):
        # scene renderer
        self.system = system.eval().to(f"cuda:{device_id}")

        # camera
        self.camera = Camera(device=f"cuda:{device_id}")
        self.camera_path = ''

        # window
        self.window = self.__impl_glfw_init(win_width=win_width, win_height=win_height, win_name=win_name)
        self.impl = GlfwRenderer(self.window)

        self.texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # render
        self.render_types           = ['RGB', 'Semantic', 'Normal', 'Depth']
        self.selected_render_id     = 0
        self.render_dict            = {'RGB': 'comp_rgb', 
                                       'Semantic': 'comp_semantic', 
                                       'Normal': 'comp_normal', 
                                       'Depth': 'comp_depth'}

        # scene editing
        self.n_objects              = self.system.geometry.num_instance
        self.object_ids             = [f'{i}' for i in range(self.n_objects)]
        self.selected_object_id     = 0

        self.object_translations    = [[0., 0., 0.] for _ in range(self.n_objects)]
        self.object_rotations       = [0. for _ in range(self.n_objects)]
        self.object_scales          = [1. for _ in range(self.n_objects)]
        self.object_visibilities    = [True for _ in range(self.n_objects)]

        # snapshot
        self.snapshot_dir           = 'outputs/snapshot'

    def __impl_glfw_init(self, win_width, win_height, win_name):
        if not glfw.init():
            print("Could not initialize OpenGL context")
            exit(1)

        # OS X supports only forward-compatible core profiles from 3.2
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

        # Create a windowed mode window and its OpenGL context
        window = glfw.create_window(
            int(win_width), int(win_height), win_name, None, None
        )
        imgui.create_context()
        glfw.make_context_current(window)

        if not window:
            glfw.terminate()
            print("Could not initialize Window")
            exit(1)

        return window

    def __control_camera(self):
        # control camera
        # - move
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            self.camera.move("forward")
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            self.camera.move("backward")
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            self.camera.move("left")
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            self.camera.move("right")
        if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
            self.camera.move("up")
        if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
            self.camera.move("down")
        # - rotate
        if glfw.get_key(self.window, glfw.KEY_UP) == glfw.PRESS:
            self.camera.rotate('up')
        if glfw.get_key(self.window, glfw.KEY_DOWN) == glfw.PRESS:
            self.camera.rotate('down')
        if glfw.get_key(self.window, glfw.KEY_LEFT) == glfw.PRESS:
            self.camera.rotate('left')
        if glfw.get_key(self.window, glfw.KEY_RIGHT) == glfw.PRESS:
            self.camera.rotate('right')

    def __bind_texture(self, image_data):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, 
            self.camera.width, self.camera.height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, image_data)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def run(self):
        start_t = time.time()
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.impl.process_inputs()

            elapsed_t = time.time() - start_t
            start_t = time.time()

            self.__control_camera()
            background_width, background_height = glfw.get_window_size(self.window)
            if (background_width, background_height) != (self.camera.width, self.camera.height):
                width, height = background_width // 8 * 8, background_height // 8 * 8
                self.camera.width, self.camera.height = width, height

            render_type = self.render_types[self.selected_render_id]
            with torch.no_grad():
                ret_dict = self.system(self.camera.get_dict())
                image = ret_dict[self.render_dict[render_type]]
            image = image.detach().cpu().numpy()[0]
            if render_type == 'Depth':
                image = plt.get_cmap('jet')(image[...,0])
            else:
                image = np.concatenate([image, np.ones_like(image[...,:1])], axis=-1)
                image = image.clip(0.0, 1.0)
            image = (image * 255).astype(np.uint8)
            
            image_data = image.copy()

            imgui.new_frame()

            imgui.set_next_window_size(*imgui.get_io().display_size)
            imgui.set_next_window_position(0, 0)
            with imgui.begin("Rendering", 
                             closable=False, 
                             flags = imgui.WINDOW_NO_TITLE_BAR|
                                     imgui.WINDOW_NO_RESIZE |
                                     imgui.WINDOW_NO_MOVE |
                                     imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS |
                                     imgui.WINDOW_NO_SCROLLBAR):
                self.__bind_texture(image_data)
                imgui.image(self.texture_id, self.camera.width, self.camera.height)

            with imgui.begin("Control Panel", closable=False):
                imgui.text(f"FPS: {1./elapsed_t+20.:0.2f}")

                imgui.separator()
                imgui.text("Camera Setting")
                # 
                with imgui.begin_combo("Render", self.render_types[self.selected_render_id]) as combo:
                    if combo.opened:
                        for i, render_id in enumerate(self.render_types):
                            is_selected = (i == self.selected_render_id)
                            if imgui.selectable(render_id, is_selected)[0]:
                                self.selected_render_id = i
                            # Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                            if is_selected:
                                imgui.set_item_default_focus()

                changed, values = imgui.drag_int2(
                    "Camera size (H, W)", 
                    self.camera.height, self.camera.width,
                    change_speed=1, min_value=64, max_value=2048
                )
                if changed:
                    width, height = values[1] // 8 * 8, values[0] // 8 * 8
                    self.camera.height, self.camera.width = height, width
                    glfw.set_window_size(self.window, self.camera.width, self.camera.height)

                changed, input_text = imgui.input_text("Camera path", self.camera_path, 256)
                if changed: self.camera_path = input_text
                if imgui.button("Load"):
                    camera_path = None
                    if osp.isfile(self.camera_path):
                        camera_path = self.camera_path
                    elif osp.isdir(self.camera_path):
                        if osp.exists(osp.join(self.camera_path, 'camera.json')):
                            camera_path = osp.join(self.camera_path, 'camera.json')
                    
                    try:
                        self.camera.load(camera_path)
                        glfw.set_window_size(self.window, self.camera.width, self.camera.height)

                    except:
                        print(f'Invalid camera path {self.camera_path}')

                # fov
                changed, values = imgui.drag_float(
                    "FoV (degree)",
                    self.camera.fovy_deg,
                    change_speed=0.01, min_value=1, max_value=89
                )
                if changed:
                    self.camera.fovy_deg = values

                imgui.separator()
                imgui.text("Scene Editing")
                with imgui.begin_combo("Select Object", self.object_ids[self.selected_object_id]) as combo:
                    if combo.opened:
                        for i, object_id in enumerate(self.object_ids):
                            is_selected = (i == self.selected_object_id)
                            if imgui.selectable(object_id, is_selected)[0]:
                                self.selected_object_id = i
                            # Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                            if is_selected:
                                imgui.set_item_default_focus()

                # visibility
                changed, visible = imgui.checkbox("Visible", self.object_visibilities[self.selected_object_id])
                if changed:
                    self.object_visibilities[self.selected_object_id] = visible
                    self.system.renderer.object_dropout_ids = \
                        [i for i, v in enumerate(self.object_visibilities) if not v]

                if self.object_visibilities[self.selected_object_id]:
                    # translation
                    changed, values = imgui.drag_float3(
                        "Translation (x,y,z)", 
                        *self.object_translations[self.selected_object_id],
                        change_speed=0.01, min_value=-5.0, max_value=5.0
                    )
                    if changed:
                        self.object_translations[self.selected_object_id] = values
                        self.system.renderer.object_translations = self.object_translations

                    # rotation
                    changed, values = imgui.drag_float(
                        "Rotation (degree)",
                        self.object_rotations[self.selected_object_id] / np.pi * 180.,
                        change_speed=0.1, min_value=-180.0, max_value=180.0
                    )
                    if changed:
                        self.object_rotations[self.selected_object_id] = values / 180. * np.pi
                        self.system.renderer.object_rotations = self.object_rotations

                    # scale
                    changed, values = imgui.drag_float(
                        "Scale",
                        self.object_scales[self.selected_object_id],
                        change_speed=0.01, min_value=0.0, max_value=5.0
                    )
                    if changed:
                        self.object_scales[self.selected_object_id] = values
                        self.system.renderer.object_scales = self.object_scales

                imgui.separator()
                imgui.text("Snapshot")
                changed, input_text = imgui.input_text("Ouput path", self.snapshot_dir, 256)
                if changed: self.snapshot_dir = input_text
                if imgui.button("Save"):
                    current_time = datetime.now().strftime("%Y%m%d_%H%M%S.%f")
                    save_path = osp.join(self.snapshot_dir, current_time)
                    if not osp.exists(save_path):
                        os.makedirs(save_path)
                    print(f'Save to {save_path}')

                    rgb = ret_dict['comp_rgb']
                    rgb = rgb.detach().cpu().numpy()[0]
                    rgb = rgb.clip(0.,1.)
                    rgb = (rgb * 255).astype(np.uint8)
                    Image.fromarray(rgb).save(osp.join(save_path, "rgb.jpg"))

                    semantic = ret_dict['comp_semantic']
                    semantic = semantic.detach().cpu().numpy()[0]
                    semantic = semantic.clip(0.,1.)
                    semantic = (semantic * 255).astype(np.uint8)
                    Image.fromarray(semantic).save(osp.join(save_path, "semantic.jpg"))

                    normal = ret_dict['comp_normal']
                    normal = normal.detach().cpu().numpy()[0]
                    normal = normal.clip(0.,1.)
                    normal = (normal * 255).astype(np.uint8)
                    Image.fromarray(normal).save(osp.join(save_path, "normal.jpg"))

                    depth = ret_dict['comp_depth']
                    depth = depth.detach().cpu().numpy()[0]
                    depth = plt.get_cmap('jet')(depth[...,0])[...,:3]
                    depth = depth.clip(0.,1.)
                    depth = (depth * 255).astype(np.uint8)
                    Image.fromarray(depth).save(osp.join(save_path, "depth.jpg"))

                    self.camera.save(osp.join(save_path, 'camera.json'))

            gl.glClearColor(.5, .5, .5, 1)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            imgui.render()
            self.impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)

        self.impl.shutdown()
        glfw.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to be used. 0 means use the 1st available GPU. "
        "1,2 means use the 2nd and 3rd available GPU. "
        "If CUDA_VISIBLE_DEVICES is set before calling `launch.py`, "
        "this argument is ignored and all available GPUs are always used.",
    )
    parser.add_argument(
        "--typecheck",
        action="store_true",
        help="whether to enable dynamic type checking",
    )

    args, extras = parser.parse_known_args()

    system = load_system(args, extras)
    
    InteractiveGUI(system=system, device_id=args.gpu, win_name='Layout2GS').run()

    print('DONE')