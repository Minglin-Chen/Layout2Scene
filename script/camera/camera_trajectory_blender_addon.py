bl_info = {
    "name": "Camera Trajectory",
    "author": "Minglin Chen",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Camera Trajectory",
    "description": "Design camera trajectory",
    "category": "Animation",
}


import bpy
from mathutils import Quaternion
import os
import json
import math


class VIEW3D_PT_CameraTrajectoryPanel(bpy.types.Panel):
    bl_label = "Camera Trajectory"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Camera Trajectory"

    def draw(self, context):
        layout = self.layout

        # Export
        box = layout.box()
        box.operator("view.save_camera_trajectory", text="Export Camera Trajectory")


class VIEW3D_OT_SaveCameraTrajectory(bpy.types.Operator):
    bl_idname = "view.save_camera_trajectory"
    bl_label = "Save Camera Trajectory"
    bl_options = {"REGISTER"}
    
    filepath: bpy.props.StringProperty(subtype="FILE_PATH", default="") # type: ignore
    filter_glob: bpy.props.StringProperty(default="*.json", options={'HIDDEN'}) # type: ignore

    def execute(self, context):
        cam = context.active_object
        if not cam or cam.type != 'CAMERA':
            self.report({'ERROR'}, "No active camera selected!")
            return {'CANCELLED'}

        start_frame = context.scene.frame_start
        end_frame   = context.scene.frame_end

        cameras = []
        for frame in range(int(start_frame), int(end_frame) + 1):
            scene = context.scene
            scene.frame_set(frame)

            matrix = cam.matrix_world.copy()

            location = matrix.translation
            loc_x, loc_y, loc_z = location.x, location.y, location.z

            rotation_euler = matrix.to_euler('XYZ')
            rot_0, rot_1, rot_2 = rotation_euler.x, rotation_euler.y, rotation_euler.z
                
            cameras.append({
                'location': [loc_x, loc_y, loc_z],
                'rotation': [math.degrees(rot_0), math.degrees(rot_1), math.degrees(rot_2)],
                "probability": float(1.0),
            })

        try:
            with open(self.filepath, 'w') as f:
                json.dump(cameras, f, indent=4)
            self.report({'INFO'}, f"The camera trajectory has beed saved to {self.filepath}")
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        
        return {'FINISHED'}

    def evaluate_fcurve(self, obj, data_path, index, frame):
        if not obj.animation_data or not obj.animation_data.action:
            return getattr(obj, data_path)[index]
        
        for fcurve in obj.animation_data.action.fcurves:
            if fcurve.data_path == data_path and fcurve.array_index == index:
                return fcurve.evaluate(frame)
        
        return getattr(obj, data_path)[index]
    
    def invoke(self, context, event):
        blend_filepath = bpy.data.filepath
        if blend_filepath:
            directory = os.path.dirname(blend_filepath)
            self.filepath = os.path.join(directory, "cameras.json")
        else:
            self.filepath = os.path.join(os.path.expanduser("~"), "cameras.json")
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


def register():
    bpy.utils.register_class(VIEW3D_PT_CameraTrajectoryPanel)
    bpy.utils.register_class(VIEW3D_OT_SaveCameraTrajectory)


def unregister():
    bpy.utils.unregister_class(VIEW3D_PT_CameraTrajectoryPanel)
    bpy.utils.unregister_class(VIEW3D_OT_SaveCameraTrajectory)


if __name__ == "__main__":
    register()