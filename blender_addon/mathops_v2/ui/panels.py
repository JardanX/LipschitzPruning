import bpy
from bpy.types import Panel

from .. import runtime
from ..nodes import sdf_tree
from ..operators.scene import MATHOPS_V2_OT_edit_sdf_graph, MATHOPS_V2_OT_new_sdf_graph


def _using_engine(context):
    return getattr(context.scene.render, "engine", "") == runtime.ENGINE_ID


class MATHOPS_V2_PT_render_settings(Panel):
    bl_label = "MathOPS V2"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    @classmethod
    def poll(cls, context):
        return _using_engine(context)

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mathops_v2
        summary = sdf_tree.scene_summary(context.scene)

        layout.prop(settings, "viewport_preview")
        row = layout.row(align=True)
        row.prop(settings, "node_tree", text="Graph")
        row.operator(MATHOPS_V2_OT_new_sdf_graph.bl_idname, text="", icon="ADD")
        layout.operator(MATHOPS_V2_OT_edit_sdf_graph.bl_idname, icon="NODETREE")

        layout.separator()
        layout.label(text=f"Graph: {summary['tree_name']}")
        layout.label(text=f"Proxy Empties: {summary['proxy_count']}")

        layout.separator()
        layout.prop(settings, "max_steps")
        layout.prop(settings, "max_distance")
        layout.prop(settings, "surface_epsilon")
        layout.prop(settings, "light_direction")

        if runtime.last_error_message:
            layout.separator()
            layout.label(text=runtime.last_error_message, icon="ERROR")


class MATHOPS_V2_PT_object_proxy(Panel):
    bl_label = "MathOPS SDF Proxy"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"

    @classmethod
    def poll(cls, context):
        return runtime.is_sdf_proxy(getattr(context, "object", None))

    def draw(self, context):
        layout = self.layout
        obj = context.object
        settings = obj.mathops_v2_sdf
        tree = sdf_tree.get_scene_tree(context.scene, create=False)
        node = sdf_tree.find_initializer_node(tree, obj=obj, proxy_id=str(settings.proxy_id or ""))

        layout.label(text="Initializer node owns this SDF")
        if node is None:
            layout.label(text="Waiting for graph sync", icon="INFO")
        else:
            layout.label(text=f"Node: {node.name}")
            column = layout.column()
            column.enabled = False
            column.prop(node, "primitive_type")
            primitive_type = str(node.primitive_type or "sphere")
            if primitive_type == "sphere":
                column.prop(node, "radius")
            elif primitive_type == "box":
                column.prop(node, "size")
            elif primitive_type == "cylinder":
                column.prop(node, "radius")
                column.prop(node, "height")
            elif primitive_type == "torus":
                column.prop(node, "major_radius")
                column.prop(node, "minor_radius")
            column.prop(node, "sdf_location")
            column.prop(node, "sdf_rotation", text="Rotation")
            column.prop(node, "sdf_scale")

        layout.separator()
        layout.operator(MATHOPS_V2_OT_edit_sdf_graph.bl_idname, icon="NODETREE")


class MATHOPS_V2_PT_graph_sidebar(Panel):
    bl_label = "MathOPS"
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "UI"
    bl_category = "MathOPS"

    @classmethod
    def poll(cls, context):
        space = getattr(context, "space_data", None)
        return space is not None and getattr(space, "tree_type", "") == runtime.TREE_IDNAME

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mathops_v2
        active_node = getattr(context, "active_node", None)

        row = layout.row(align=True)
        row.prop(settings, "node_tree", text="Scene Graph")
        row.operator(MATHOPS_V2_OT_new_sdf_graph.bl_idname, text="", icon="ADD")

        if active_node is not None and getattr(active_node, "bl_idname", "") == runtime.OBJECT_NODE_IDNAME:
            layout.separator()
            layout.label(text="Active Initializer")
            layout.prop(active_node, "primitive_type", text="Type")
            layout.prop(active_node, "sdf_location")
            layout.prop(active_node, "sdf_rotation", text="Rotation")
            layout.prop(active_node, "sdf_scale")

        if runtime.last_error_message:
            layout.separator()
            layout.label(text=runtime.last_error_message, icon="ERROR")


classes = (
    MATHOPS_V2_PT_render_settings,
    MATHOPS_V2_PT_object_proxy,
    MATHOPS_V2_PT_graph_sidebar,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
