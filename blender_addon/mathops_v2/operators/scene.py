import uuid

import bpy
from bpy.props import EnumProperty, StringProperty
from bpy.types import Menu, Operator

from .. import properties, runtime
from ..nodes import sdf_tree


_PRIMITIVE_ICONS = {
    "sphere": "MESH_UVSPHERE",
    "box": "MESH_CUBE",
    "cylinder": "MESH_CYLINDER",
    "torus": "MESH_TORUS",
}

_EMPTY_DISPLAY_TYPES = {
    "sphere": "PLAIN_AXES",
    "box": "PLAIN_AXES",
    "cylinder": "PLAIN_AXES",
    "torus": "PLAIN_AXES",
}


def _configure_proxy_defaults(obj, primitive_type):
    settings = obj.mathops_v2_sdf
    settings.enabled = True
    settings.proxy_id = uuid.uuid4().hex
    settings.source_tree_name = ""
    settings.source_node_name = ""
    settings.primitive_type = primitive_type
    settings.radius = 0.5
    settings.size = (0.5, 0.5, 0.5)
    settings.height = 1.0
    settings.major_radius = 0.75
    settings.minor_radius = 0.25

    obj.rotation_mode = "XYZ"
    obj.empty_display_type = _EMPTY_DISPLAY_TYPES.get(primitive_type, "PLAIN_AXES")
    obj.empty_display_size = 0.01
    obj.show_name = False
    obj.show_in_front = False
    obj.hide_render = True


def _link_proxy_object(context, primitive_type):
    label = dict((identifier, name) for identifier, name, _description in properties.PRIMITIVE_ITEMS)[primitive_type]
    obj = bpy.data.objects.new(f"SDF {label}", None)
    _configure_proxy_defaults(obj, primitive_type)
    obj.location = context.scene.cursor.location.copy()
    collection = getattr(context, "collection", None)
    if collection is None:
        collection = context.scene.collection
    collection.objects.link(obj)
    for selected in context.selected_objects:
        selected.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    return obj


class MATHOPS_V2_OT_add_sdf_proxy(Operator):
    bl_idname = "mathops_v2.add_sdf_proxy"
    bl_label = "Add SDF Proxy"
    bl_description = "Add an SDF empty proxy and insert it into the scene graph"
    bl_options = {"REGISTER", "UNDO"}

    primitive_type: EnumProperty(name="Primitive", items=properties.PRIMITIVE_ITEMS, default="sphere")

    def execute(self, context):
        obj = _link_proxy_object(context, self.primitive_type)
        sdf_tree.add_proxy_to_tree(context.scene, obj)
        runtime.clear_error()
        runtime.tag_redraw(context)
        return {"FINISHED"}


class MATHOPS_V2_OT_new_sdf_graph(Operator):
    bl_idname = "mathops_v2.new_sdf_graph"
    bl_label = "New SDF Graph"
    bl_description = "Create a fresh MathOPS SDF node graph for the scene"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        sdf_tree.new_scene_tree(context.scene)
        sdf_tree.focus_scene_tree(context, create=False)
        runtime.clear_error()
        runtime.tag_redraw(context)
        return {"FINISHED"}


class MATHOPS_V2_OT_edit_sdf_graph(Operator):
    bl_idname = "mathops_v2.edit_sdf_graph"
    bl_label = "Edit SDF Graph"
    bl_description = "Focus the scene SDF graph in a Node Editor"
    bl_options = {"REGISTER"}

    def execute(self, context):
        if not sdf_tree.focus_scene_tree(context, create=True):
            self.report({"INFO"}, "Open a Node Editor and switch it to MathOPS SDF")
        return {"FINISHED"}


class MATHOPS_V2_OT_create_node_proxy(Operator):
    bl_idname = "mathops_v2.create_node_proxy"
    bl_label = "Create Node Proxy"
    bl_description = "Create an SDF proxy empty for this node"
    bl_options = {"REGISTER", "UNDO"}

    tree_name: StringProperty(default="")
    node_name: StringProperty(default="")
    primitive_type: EnumProperty(name="Primitive", items=properties.PRIMITIVE_ITEMS, default="sphere")

    def execute(self, context):
        tree = bpy.data.node_groups.get(self.tree_name)
        if tree is None or getattr(tree, "bl_idname", "") != runtime.TREE_IDNAME:
            self.report({"ERROR"}, "SDF node tree not found")
            return {"CANCELLED"}

        node = tree.nodes.get(self.node_name)
        if node is None or getattr(node, "bl_idname", "") != runtime.OBJECT_NODE_IDNAME:
            self.report({"ERROR"}, "SDF initializer node not found")
            return {"CANCELLED"}

        obj = _link_proxy_object(context, str(getattr(node, "primitive_type", self.primitive_type) or self.primitive_type))
        sdf_tree.attach_proxy_to_node(node, obj, adopt_proxy=False)
        runtime.clear_error()
        runtime.tag_redraw(context)
        return {"FINISHED"}


class VIEW3D_MT_mathops_v2_add(Menu):
    bl_idname = "VIEW3D_MT_mathops_v2_add"
    bl_label = "MathOPS SDF"

    def draw(self, _context):
        layout = self.layout
        for primitive_type, label, _description in properties.PRIMITIVE_ITEMS:
            operator = layout.operator(
                MATHOPS_V2_OT_add_sdf_proxy.bl_idname,
                text=label,
                icon=_PRIMITIVE_ICONS.get(primitive_type, "EMPTY_AXIS"),
            )
            operator.primitive_type = primitive_type


def _draw_add_menu(self, _context):
    self.layout.separator()
    self.layout.menu(VIEW3D_MT_mathops_v2_add.bl_idname, icon="META_BALL")


classes = (
    MATHOPS_V2_OT_add_sdf_proxy,
    MATHOPS_V2_OT_new_sdf_graph,
    MATHOPS_V2_OT_edit_sdf_graph,
    MATHOPS_V2_OT_create_node_proxy,
    VIEW3D_MT_mathops_v2_add,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_MT_add.append(_draw_add_menu)


def unregister():
    try:
        bpy.types.VIEW3D_MT_add.remove(_draw_add_menu)
    except Exception:
        pass
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
