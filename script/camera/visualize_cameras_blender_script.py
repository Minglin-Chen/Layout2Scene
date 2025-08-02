import bpy
import math
import matplotlib.pyplot as plt


cameras = [
    {
        "location": [
            0.0,
            0.0,
            0.0
        ],
        "rotation": [
            0.0,
            0.0,
            0.0
        ],
        "fov": 60,
        "probability": 0.5
    },
]


def create_pure_color_material(mat_name="PureColorMaterial", color=(0.5,0.5,0.5,1.0)):
    new_mat = bpy.data.materials.new(name=mat_name)
    new_mat.use_nodes = True
    nodes = new_mat.node_tree.nodes
    links = new_mat.node_tree.links

    for node in nodes: nodes.remove(node)

    principled_bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    material_output_node = nodes.new(type="ShaderNodeOutputMaterial")

    links.new(principled_bsdf_node.outputs["BSDF"], material_output_node.inputs["Surface"])

    principled_bsdf_node.inputs["Base Color"].default_value = color
    return new_mat


def create_wireframe_material(mat_name="WireframeMaterial", color=(0.5,0.5,0.5,1.0)):
    new_mat = bpy.data.materials.new(name=mat_name)
    new_mat.use_nodes = True
    nodes = new_mat.node_tree.nodes
    links = new_mat.node_tree.links
    nodes.clear()

    principled_bsdf_node = nodes.new('ShaderNodeBsdfPrincipled')
    material_output_node = nodes.new('ShaderNodeOutputMaterial')
    wireframe_node = nodes.new('ShaderNodeWireframe')

    links.new(wireframe_node.outputs['Fac'], principled_bsdf_node.inputs['Alpha'])
    links.new(principled_bsdf_node.outputs['BSDF'], material_output_node.inputs['Surface'])

    principled_bsdf_node.inputs["Base Color"].default_value = color
    return new_mat


def create_camera(location, rotation, p):
    cmap = plt.get_cmap("jet")
    color = cmap(p)

    # # Camera location
    # bpy.ops.mesh.primitive_uv_sphere_add(
    #     segments=32, ring_count=16, radius=0.05, location=location
    # )
    # cam_loc = bpy.context.active_object
    # cam_loc.data.shade_smooth()

    # material = create_pure_color_material(color=color)
    # cam_loc.data.materials.clear()
    # cam_loc.data.materials.append(material)
    # cam_loc.active_material = material

    # Camera model
    cam_mesh = bpy.data.meshes.new("Camera")
    cam_mesh.from_pydata(
        [
            [ 0.0, 0.0, 0.0],
            [-0.5, 0.5,-1.0],
            [ 0.5, 0.5,-1.0],
            [ 0.5,-0.5,-1.0],
            [-0.5,-0.5,-1.0],
            [ 0.0, 0.7,-1.0]
        ],
        [],
        [
            [0,1,2],
            [0,2,3],
            [0,3,4],
            [0,4,1],
            [5,1,2]
        ]
    )
    cam_obj = bpy.data.objects.new("Camera", cam_mesh)
    bpy.context.collection.objects.link(cam_obj)

    cam_obj.location = location
    cam_obj.rotation_euler = (
        math.radians(rotation[0]),
        math.radians(rotation[1]),
        math.radians(rotation[2])
    )
    cam_obj.scale = (0.2, 0.2, 0.2)

    # material = create_wireframe_material(color=color)
    material = create_pure_color_material(color=color)
    cam_obj.data.materials.clear()
    cam_obj.data.materials.append(material)
    cam_obj.active_material = material


if __name__=='__main__':
    # 1. Create collection
    original_layer_collection = bpy.context.view_layer.active_layer_collection
    collection = bpy.data.collections.get("Cameras", None)
    if collection is not None:
        for obj in collection.objects[:]:
            bpy.data.objects.remove(obj)
        bpy.data.collections.remove(collection)

    collection = bpy.data.collections.new("Cameras")
    bpy.context.scene.collection.children.link(collection)

    layer_collection = bpy.context.view_layer.layer_collection.children["Cameras"]
    bpy.context.view_layer.active_layer_collection = layer_collection

    # 2. Place cameras
    for cam in cameras:
        create_camera(cam["location"], cam["rotation"], cam["probability"])

    # 3. Restore collection
    bpy.context.view_layer.active_layer_collection = original_layer_collection