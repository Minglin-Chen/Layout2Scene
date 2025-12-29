""" blender -b --python refine_mesh_bl.py -- <input_dir> <output_dir> """

import bpy
import os
import sys
import argparse
from mathutils import Vector


SUPPORTED_EXTENSIONS = ['.obj', '.ply', '.glb', '.gltf']


def import_mesh(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.obj':
        bpy.ops.import_scene.obj(filepath=filepath)
    elif ext == '.ply':
        if bpy.app.version >= (4, 0, 0):
            bpy.ops.wm.ply_import(filepath=filepath)
        else:
            bpy.ops.import_mesh.ply(filepath=filepath)
    elif ext in ['.glb', '.gltf']:
        bpy.ops.import_scene.gltf(filepath=filepath)
    else:
        raise ValueError(f"Unsupported format: {ext}")


def export_mesh(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.obj':
        bpy.ops.export_scene.obj(filepath=filepath, use_selection=True)
    elif ext == '.ply':
        if bpy.app.version >= (4, 0, 0):
            bpy.ops.wm.ply_export(filepath=filepath)
        else:
            bpy.ops.export_mesh.ply(filepath=filepath, use_selection=True)
    elif ext in ['.glb', '.gltf']:
        bpy.ops.export_scene.gltf(filepath=filepath, export_format='GLB' if filepath=='.glb' else 'GLTF_SEPARATE', use_selection=True)
    else:
        raise ValueError(f"Unsupported format: {ext}")


def remesh_and_uv(obj, voxel_factor=0.1, uv_margin=0.02):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # voxel size
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    xs = [v[0] for v in bbox]
    ys = [v[1] for v in bbox]
    zs = [v[2] for v in bbox]
    L = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
    voxel_size = L * voxel_factor

    # Remesh
    remesh_mod = obj.modifiers.new(name="Remesh", type='REMESH')
    remesh_mod.mode = 'VOXEL'
    remesh_mod.voxel_size = voxel_size
    remesh_mod.use_smooth_shade = True
    bpy.ops.object.modifier_apply(modifier=remesh_mod.name)

    # UV unwrap
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=66, island_margin=uv_margin)
    bpy.ops.object.mode_set(mode='OBJECT')

    return obj


def batch_process(input_dir, output_dir, voxel_factor=0.1, uv_margin=0.02):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False, confirm=False)

        input_path = os.path.join(input_dir, filename)
        import_mesh(input_path)

        imported_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
        for obj in imported_objs:
            remesh_and_uv(obj, voxel_factor=voxel_factor, uv_margin=uv_margin)

        output_path = os.path.join(output_dir, filename)
        export_mesh(output_path)


if __name__ == '__main__':
    argv = sys.argv
    if "--" not in argv:
        raise ValueError("Please specify argumentation: -- <args>")

    argv = argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="input direcotory")
    parser.add_argument("output_dir", type=str, help="output directory")
    parser.add_argument("--voxel_factor", type=float, default=0.02, help="voxel_size = bbox_max_edge * voxel_factor")
    parser.add_argument("--uv_margin", type=float, default=0.02, help="UV unwrap margin")

    args = parser.parse_args(argv)

    batch_process(args.input_dir, args.output_dir, args.voxel_factor, args.uv_margin)
