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

        cam = context.active_object
        if (not cam) or (cam.type != 'CAMERA') or (not cam.animation_data) or (not cam.animation_data.action):
            layout.label(text=f"Number of Keyframes: UNAVAILABLE")
        else:
            layout.label(text=f"Number of Keyframes: {self.count_keyframes(cam)//6}")

        layout.operator("view.add_keyframe", text="Add Keyframe")

        # Input/Output
        box = layout.box()
        box.label(text="Export")
        col = box.column(align=True)
        col.operator("view.save_camera_trajectory", text="Save")

    def count_keyframes(self, cam):
        return sum([len(fcurve.keyframe_points) for fcurve in cam.animation_data.action.fcurves])


class VIEW3D_OT_AddKeyframe(bpy.types.Operator):
    """Adds a keyframe for the active camera based on current viewport perspective"""
    bl_idname = "view.add_keyframe"
    bl_label = "Add Keyframe"
    
    def execute(self, context):
        # Get the current viewport camera or active camera
        if context.space_data.region_3d.view_perspective == 'CAMERA':
            # If we're in camera view, use that camera
            cam = context.scene.camera
            if not cam:
                self.report({'ERROR'}, "No active camera in scene!")
                return {'CANCELLED'}
        else:
            # Otherwise use the active object (if it's a camera)
            cam = context.active_object
            if not cam or cam.type != 'CAMERA':
                self.report({'ERROR'}, "No active camera selected!")
                return {'CANCELLED'}
        
        # Get current viewport matrix
        world2camera            = context.space_data.region_3d.view_matrix
        camera2world            = world2camera.inverted()
        
        keyframe_location       = camera2world.translation
        # keyframe_quaternion     = camera2world.to_3x3().to_quaternion()
        keyframe_rotation_euler = camera2world.to_euler('XYZ')
        
        # Set camera location and rotation to match viewport
        cam.location = keyframe_location
        
        # cam.rotation_mode = 'QUATERNION'
        # cam.rotation_quaternion = keyframe_quaternion
        cam.rotation_mode = 'XYZ'
        cam.rotation_euler = keyframe_rotation_euler
        
        # Insert keyframes
        cam.keyframe_insert(data_path="location")
        # cam.keyframe_insert(data_path="rotation_quaternion")
        cam.keyframe_insert(data_path="rotation_euler")
        
        self.report({'INFO'}, f"Added keyframe for camera '{cam.name}'")
        return {'FINISHED'}


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

            loc_x = self.evaluate_fcurve(cam, "location", 0, frame)
            loc_y = self.evaluate_fcurve(cam, "location", 1, frame)
            loc_z = self.evaluate_fcurve(cam, "location", 2, frame)
            
            rot_0 = self.evaluate_fcurve(cam, "rotation_euler", 0, frame)
            rot_1 = self.evaluate_fcurve(cam, "rotation_euler", 1, frame)
            rot_2 = self.evaluate_fcurve(cam, "rotation_euler", 2, frame)
            
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
    bpy.utils.register_class(VIEW3D_OT_AddKeyframe)
    bpy.utils.register_class(VIEW3D_OT_SaveCameraTrajectory)


def unregister():
    bpy.utils.unregister_class(VIEW3D_PT_CameraTrajectoryPanel)
    bpy.utils.unregister_class(VIEW3D_OT_AddKeyframe)
    bpy.utils.unregister_class(VIEW3D_OT_SaveCameraTrajectory)


if __name__ == "__main__":
    register()