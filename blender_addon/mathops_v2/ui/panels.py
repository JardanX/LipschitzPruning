import bpy
from bpy.types import Panel

from .. import properties, runtime
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
    properties.ensure_scene_defaults(context.scene, context)
    settings = runtime.scene_settings(context.scene)
    return None if settings is None else getattr(settings, "node_tree", None)


def _linked_socket_text(socket):
    if socket is None or not socket.is_linked or not socket.links:
        return ""
    link = socket.links[0]
    source_socket = getattr(link, "from_socket", None)
    source_node = getattr(link, "from_node", None)
    if source_node is None:
        return ""
    source_name = source_node.name
    socket_name = "" if source_socket is None else str(getattr(source_socket, "name", "") or "")
    return source_name if not socket_name else f"{source_name} · {socket_name}"


def _draw_socket_input(layout, socket, text=None):
    if socket is None:
        return
    label = text or socket.name
    if socket.is_linked:
        row = layout.row(align=True)
        row.label(text=label)
        row.label(text=_linked_socket_text(socket), icon="LINKED")
        return
    if hasattr(socket, "default_value"):
        layout.prop(socket, "default_value", text=label)
        return
    layout.label(text=label)


def _format_vec(values):
    return ", ".join(f"{float(component):.3f}" for component in values)


def _draw_object_node(layout, node):
    box = layout.box()
    box.label(text="Initializer", icon="OBJECT_DATA")
    box.prop(node, "name", text="Node")
    box.prop(node, "primitive_type", text="Type")
    transform_socket = sdf_tree.node_input_socket(node, "Transform")
    if transform_socket is not None and transform_socket.is_linked:
        _draw_socket_input(box, transform_socket)
    else:
        box.label(text="Transform: viewport handle")
    for socket_name in sdf_tree.object_parameter_socket_names(node):
        _draw_socket_input(box, sdf_tree.node_input_socket(node, socket_name))
    if getattr(node, "target", None) is not None:
        box.prop(node.target, "name", text="Handle")
    box.prop(node, "proxy_id", text="Proxy ID")


def _draw_make_transform_node(layout, node):
    box = layout.box()
    box.label(text="Make Transform", icon="CONSTRAINT")
    box.prop(node, "name", text="Node")
    _draw_socket_input(box, sdf_tree.node_input_socket(node, "Location"))
    _draw_socket_input(box, sdf_tree.node_input_socket(node, "Rotation"))
    _draw_socket_input(box, sdf_tree.node_input_socket(node, "Scale"))


def _draw_break_transform_node(layout, node):
    box = layout.box()
    box.label(text="Break Transform", icon="DRIVER")
    box.prop(node, "name", text="Node")
    transform_socket = sdf_tree.node_input_socket(node, "Transform")
    if transform_socket is not None and transform_socket.is_linked:
        _draw_socket_input(box, transform_socket)
    else:
        box.label(text="Transform input not connected", icon="INFO")
    location, rotation, scale = sdf_tree.break_transform_values(node)
    preview = box.column()
    preview.enabled = False
    preview.label(text=f"Location: {_format_vec(location)}")
    preview.label(text=f"Rotation: {_format_vec(rotation)}")
    preview.label(text=f"Scale: {_format_vec(scale)}")


def _draw_csg_node(layout, node):
    box = layout.box()
    box.label(text="CSG", icon="MOD_BOOLEAN")
    box.prop(node, "name", text="Node")
    box.prop(node, "operation", text="Operation")
    _draw_socket_input(box, sdf_tree.node_input_socket(node, "Blend"))


def _draw_mirror_node(layout, node):
    box = layout.box()
    box.label(text="Mirror", icon="MOD_MIRROR")
    box.prop(node, "name", text="Node")
    _draw_socket_input(box, sdf_tree.node_input_socket(node, "SDF"))
    row = box.row(align=True)
    row.prop(node, "mirror_x", toggle=True)
    row.prop(node, "mirror_y", toggle=True)
    row.prop(node, "mirror_z", toggle=True)
    box.prop(node, "blend", text="Blend")
    box.prop(node, "origin_object", text="Origin")


def _draw_array_node(layout, node):
    box = layout.box()
    box.label(text="Array", icon="MOD_ARRAY")
    box.prop(node, "name", text="Node")
    _draw_socket_input(box, sdf_tree.node_input_socket(node, "SDF"))
    box.prop(node, "array_mode", text="Mode")
    if node.array_mode == sdf_tree._ARRAY_MODE_GRID:
        row = box.row(align=True)
        row.prop(node, "count_x")
        row.prop(node, "count_y")
        row.prop(node, "count_z")
        box.prop(node, "spacing")
    else:
        box.prop(node, "radial_count")
        box.prop(node, "radius")
        box.prop(node, "origin_object", text="Origin")
    box.prop(node, "blend", text="Blend")


def _draw_inspector_node(layout, node):
    node_idname = getattr(node, "bl_idname", "")
    if node_idname == runtime.OBJECT_NODE_IDNAME:
        _draw_object_node(layout, node)
        return
    if node_idname == sdf_tree.TRANSFORM_NODE_IDNAME:
        _draw_make_transform_node(layout, node)
        return
    if node_idname == sdf_tree.BREAK_TRANSFORM_NODE_IDNAME:
        _draw_break_transform_node(layout, node)
        return
    if node_idname == runtime.CSG_NODE_IDNAME:
        _draw_csg_node(layout, node)
        return
    if node_idname == sdf_tree.MIRROR_NODE_IDNAME:
        _draw_mirror_node(layout, node)
        return
    if node_idname == sdf_tree.ARRAY_NODE_IDNAME:
        _draw_array_node(layout, node)


def _draw_node_sections(layout, nodes):
    for node in nodes:
        _draw_inspector_node(layout, node)


def _draw_proxy_inspection(layout, context):
    obj = context.object
    settings = obj.mathops_v2_sdf
    tree = _ensure_scene_tree(context)
    node = sdf_tree.find_initializer_node(tree, obj=obj, proxy_id=str(settings.proxy_id or ""))

    header = layout.box()
    header.label(text="Node Inspector", icon="NODETREE")

    if node is None:
        header.label(text="Waiting for graph sync", icon="INFO")
        source_tree_name = str(settings.source_tree_name or "")
        source_node_name = str(settings.source_node_name or "")
        if source_tree_name:
            header.label(text=f"Graph: {source_tree_name}")
        if source_node_name:
            header.label(text=f"Node: {source_node_name}")
    else:
        header.label(text=f"Graph: {node.id_data.name}")
        header.label(text=f"Root: {node.name}")
        _draw_node_sections(layout, sdf_tree.inspector_related_nodes(node))


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
        layout.prop(settings, "disable_surface_shading")
        layout.separator()
        layout.prop(settings, "culling_enabled")
        pruning_col = layout.column()
        pruning_col.enabled = settings.culling_enabled
        pruning_col.prop(settings, "pruning_grid_level")

        cone_toggle_col = layout.column()
        cone_toggle_col.enabled = settings.culling_enabled
        cone_toggle_col.prop(settings, "cone_prepass_enabled")
        if settings.cone_prepass_enabled:
            cone_col = layout.column()
            cone_col.enabled = settings.culling_enabled
            cone_col.prop(settings, "cone_aperture")
            cone_col.prop(settings, "cone_steps")

        layout.prop(settings, "debug_shading")
        if settings.debug_shading in {"PRUNING_ACTIVE", "PRUNING_FIELD"}:
            layout.prop(settings, "colormap_max")

        if runtime.last_error_message:
            layout.separator()
            layout.label(text=runtime.last_error_message, icon="ERROR")


class MATHOPS_V2_PT_object_proxy(Panel):
    bl_label = "Node Inspector"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"

    @classmethod
    def poll(cls, context):
        return runtime.is_sdf_proxy(getattr(context, "object", None))

    def draw(self, context):
        _draw_proxy_inspection(self.layout, context)


class MATHOPS_V2_PT_object_proxy_data(Panel):
    bl_label = "Node Inspector"
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

        if active_node is not None:
            layout.separator()
            layout.label(text="Active Node")
            if getattr(active_node, "bl_idname", "") == runtime.OBJECT_NODE_IDNAME:
                _draw_node_sections(layout, sdf_tree.inspector_related_nodes(active_node))
            else:
                _draw_node_sections(layout, (active_node,))

        if runtime.last_error_message:
            layout.separator()
            layout.label(text=runtime.last_error_message, icon="ERROR")


def draw_shading_popover(self, context):
    if not _using_engine(context):
        return

    properties.ensure_scene_defaults(context.scene, context)
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
