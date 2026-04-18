import bpy
from bpy.types import Panel

from .. import runtime
from ..nodes import sdf_tree
from ..operators.scene import MATHOPS_V2_OT_edit_sdf_graph, MATHOPS_V2_OT_new_sdf_graph


def _using_engine(context):
    return getattr(context.scene.render, "engine", "") == runtime.ENGINE_ID


def _rendered_shading(context):
    space = getattr(context, "space_data", None)
    shading = getattr(space, "shading", None)
    if shading is None or getattr(shading, "type", "") != "RENDERED":
        return None
    return shading


def _ensure_scene_tree(context):
    return sdf_tree.ensure_scene_tree(context.scene)


def _draw_proxy_inspection(layout, context):
    obj = context.object
    settings = obj.mathops_v2_sdf
    scene_settings = context.scene.mathops_v2
    tree = _ensure_scene_tree(context)
    node = sdf_tree.find_initializer_node(tree, obj=obj, proxy_id=str(settings.proxy_id or ""))

    layout.label(text="Initializer node owns this SDF")
    info_box = layout.box()
    info_col = info_box.column()
    info_col.enabled = False
    info_col.prop(settings, "source_tree_name", text="Graph")
    info_col.prop(settings, "source_node_name", text="Node")
    info_col.prop(settings, "proxy_id", text="Proxy ID")

    if node is None:
        layout.label(text="Waiting for graph sync", icon="INFO")
    else:
        layout.label(text=f"Graph: {node.id_data.name}", icon="NODETREE")
        layout.label(text=f"Node: {node.name}")
        transform_source = sdf_tree.transform_source_node(node)
        if transform_source is not None:
            layout.label(text=f"Transform: {transform_source.name}", icon="CONSTRAINT")
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
    shading_box = layout.box()
    shading_box.label(text="Viewport Shading")
    shading_box.template_icon_view(scene_settings, "custom_matcap", show_labels=False, scale=4.0, scale_popup=3.0)

    layout.separator()
    layout.operator(MATHOPS_V2_OT_edit_sdf_graph.bl_idname, icon="NODETREE")


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
        tree = _ensure_scene_tree(context)
        summary = sdf_tree.scene_summary(context.scene)

        layout.prop(settings, "viewport_preview")
        row = layout.row(align=True)
        row.prop(settings, "node_tree", text="Graph")
        row.operator(MATHOPS_V2_OT_new_sdf_graph.bl_idname, text="", icon="ADD")
        layout.operator(MATHOPS_V2_OT_edit_sdf_graph.bl_idname, icon="NODETREE")

        layout.separator()
        layout.label(text=f"Graph: {summary['tree_name']}")
        layout.label(text=f"Proxy Empties: {summary['proxy_count']}")
        if tree is not None:
            layout.label(text=f"Active Graph: {tree.name}", icon="NODETREE")

        layout.separator()
        layout.prop(settings, "max_steps")
        layout.prop(settings, "max_distance")
        layout.prop(settings, "surface_epsilon")

        layout.separator()
        layout.prop(settings, "culling_enabled")
        pruning_col = layout.column()
        pruning_col.enabled = settings.culling_enabled
        pruning_col.prop(settings, "pruning_grid_level")
        pruning_col.prop(settings, "debug_shading")
        if settings.debug_shading != "SHADED":
            pruning_col.prop(settings, "colormap_max")

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
        _draw_proxy_inspection(self.layout, context)


class MATHOPS_V2_PT_object_proxy_data(Panel):
    bl_label = "MathOPS SDF Proxy"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "data"

    @classmethod
    def poll(cls, context):
        return runtime.is_sdf_proxy(getattr(context, "object", None))

    def draw(self, context):
        _draw_proxy_inspection(self.layout, context)


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
        _ensure_scene_tree(context)
        active_node = getattr(context, "active_node", None)

        row = layout.row(align=True)
        row.prop(settings, "node_tree", text="Scene Graph")
        row.operator(MATHOPS_V2_OT_new_sdf_graph.bl_idname, text="", icon="ADD")

        if active_node is not None and getattr(active_node, "bl_idname", "") == runtime.OBJECT_NODE_IDNAME:
            layout.separator()
            layout.label(text="Active Initializer")
            layout.prop(active_node, "primitive_type", text="Type")
            transform_source = sdf_tree.transform_source_node(active_node)
            if transform_source is not None:
                layout.label(text=f"Transform: {transform_source.name}", icon="CONSTRAINT")
            layout.prop(active_node, "sdf_location")
            layout.prop(active_node, "sdf_rotation", text="Rotation")
            layout.prop(active_node, "sdf_scale")

        if runtime.last_error_message:
            layout.separator()
            layout.label(text=runtime.last_error_message, icon="ERROR")


def draw_shading_popover(self, context):
    if not _using_engine(context):
        return

    shading = _rendered_shading(context)
    if shading is None:
        return

    layout = self.layout
    settings = context.scene.mathops_v2
    layout.label(text="MathOPS")
    layout.template_icon_view(settings, "custom_matcap", show_labels=False, scale=4.0, scale_popup=3.0)
    layout.prop(settings, "gamma")
    if hasattr(shading, "show_specular_highlight"):
        layout.prop(shading, "show_specular_highlight", text="Specular Lighting")


classes = (
    MATHOPS_V2_PT_render_settings,
    MATHOPS_V2_PT_object_proxy,
    MATHOPS_V2_PT_object_proxy_data,
    MATHOPS_V2_PT_graph_sidebar,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_PT_shading.append(draw_shading_popover)


def unregister():
    try:
        bpy.types.VIEW3D_PT_shading.remove(draw_shading_popover)
    except Exception:
        pass
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
