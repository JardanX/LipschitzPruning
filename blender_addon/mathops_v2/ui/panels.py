from pathlib import Path

import bpy
from bpy.types import Panel

from .. import runtime, sdf_nodes, sdf_proxies, viewport_interaction
from .. import operators
from ..render import bridge


def _proxy_object(context):
    obj = getattr(context, "object", None)
    if (
        obj is None
        or obj.type != "EMPTY"
        or not getattr(obj, "mathops_v2_sdf_proxy", False)
    ):
        return None
    return obj


def _proxy_primitive_id(obj):
    return str(
        getattr(obj, "mathops_v2_sdf_node_id", "")
        or getattr(obj, "mathops_v2_sdf_proxy_id", "")
        or ""
    )


def _linked_proxy_node(context):
    obj = _proxy_object(context)
    scene = getattr(context, "scene", None)
    settings = getattr(scene, "mathops_v2_settings", None)
    primitive_id = "" if obj is None else _proxy_primitive_id(obj)
    if (
        scene is None
        or settings is None
        or not getattr(settings, "use_sdf_nodes", False)
    ):
        return primitive_id, None, None
    try:
        tree = sdf_nodes.get_selected_tree(settings, create=True, ensure=True)
    except Exception:
        return primitive_id, None, None
    return primitive_id, tree, sdf_nodes.find_primitive_node(tree, primitive_id)


def _format_scalar(value):
    return f"{float(value):.3f}"


def _format_vector(values):
    return ", ".join(f"{float(component):.3f}" for component in values)


def _surface_output_socket(node):
    for socket in getattr(node, "outputs", ()):
        if getattr(socket, "bl_idname", "") == sdf_nodes.SOCKET_IDNAME:
            return socket
    return None


def _driver_node_prop(node, socket_name, fallback_prop):
    source_node = sdf_nodes.node_input_source_node(node, socket_name)
    if source_node is None:
        return node, fallback_prop, node.name, True

    prop_name = {
        sdf_nodes.VALUE_NODE_IDNAME: "value",
        sdf_nodes.VECTOR_NODE_IDNAME: "value",
        sdf_nodes.COLOR_NODE_IDNAME: "value",
    }.get(getattr(source_node, "bl_idname", ""))
    if prop_name is None:
        return None, None, source_node.name, False
    return source_node, prop_name, source_node.name, True


def _draw_driver_prop(layout, label, node, socket_name, fallback_prop):
    target_node, prop_name, source_name, editable = _driver_node_prop(
        node, socket_name, fallback_prop
    )
    box = layout.box()
    box.label(text=f"{label} <- {source_name}", icon="NODE")
    if editable and prop_name and hasattr(target_node, prop_name):
        box.prop(target_node, prop_name, text=label)
    else:
        box.label(text="Driven by a non-inline graph node", icon="INFO")


def _affecting_csg_nodes(node):
    affecting = []
    seen = set()
    current = node
    csg_idnames = {
        sdf_nodes.CSG_NODE_IDNAME,
        sdf_nodes.UNION_NODE_IDNAME,
        sdf_nodes.SUBTRACT_NODE_IDNAME,
        sdf_nodes.INTERSECT_NODE_IDNAME,
    }
    while current is not None:
        current_key = current.as_pointer()
        if current_key in seen:
            break
        seen.add(current_key)

        output_socket = _surface_output_socket(current)
        if output_socket is None:
            break
        parent = None
        for link in output_socket.links:
            to_node = getattr(link, "to_node", None)
            if getattr(to_node, "bl_idname", "") in csg_idnames:
                parent = to_node
                break
        if parent is None:
            break
        affecting.append(parent)
        current = parent
    return affecting


def _node_source_label(node, socket_name, fallback):
    source = sdf_nodes.node_input_source_node(node, socket_name)
    return fallback if source is None else source.name


def _draw_effective_line(layout, label, value_text, source_text):
    layout.label(text=f"{label}: {value_text}")
    layout.label(text=f"From: {source_text}", icon="NODETREE")


class _MathOPSV2Panel:
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    @classmethod
    def poll(cls, context):
        return context.engine == runtime.ENGINE_ID


class MATHOPS_V2_PT_render_settings(_MathOPSV2Panel, Panel):
    bl_label = "MathOPS-v2"
    bl_idname = "MATHOPS_V2_PT_render_settings"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False


class MATHOPS_V2_PT_scene(_MathOPSV2Panel, Panel):
    bl_label = "Scene"
    bl_parent_id = "MATHOPS_V2_PT_render_settings"
    bl_idname = "MATHOPS_V2_PT_scene"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = context.scene.mathops_v2_settings
        tree = None
        scene_path = None
        metadata = None
        if settings.use_sdf_nodes:
            try:
                tree = sdf_nodes.get_selected_tree(settings, create=True, ensure=True)
            except Exception:
                tree = None
            if tree is not None:
                scene_cache = runtime.generated_scene_cache.get(tree.name_full)
                if scene_cache is not None:
                    scene_path = Path(scene_cache["path"])
                    metadata = scene_cache["metadata"]
        else:
            scene_cache = bridge.graph_scene_cache(settings)
            if scene_cache is not None:
                scene_path = Path(scene_cache["path"])
                metadata = scene_cache["metadata"]

        layout.prop(settings, "use_sdf_nodes")
        layout.prop(settings, "template_scene", text="Template")
        if settings.use_sdf_nodes:
            row = layout.row(align=True)
            row.label(
                text=tree.name if tree is not None else "Pending", icon="NODETREE"
            )
            row.operator(
                sdf_nodes.MATHOPS_V2_OT_new_scene_sdf_tree.bl_idname,
                text="",
                icon="ADD",
            )
            layout.operator(
                sdf_nodes.MATHOPS_V2_OT_edit_scene_sdf_tree.bl_idname,
                text="Edit SDF Graph",
            )

        box = layout.box()
        box.label(
            text=f"Resolved Scene: {scene_path.name if scene_path is not None else 'Pending'}"
        )
        if not settings.use_sdf_nodes:
            box.label(
                text="Template scenes render through MathOPS-v2 only", icon="INFO"
            )
        elif tree is None:
            box.label(text="Scene graph is initializing", icon="INFO")
        else:
            box.label(text=f"Source Tree: {tree.name}", icon="NODETREE")
        if metadata is None:
            if settings.use_sdf_nodes and tree is None:
                return
            box.label(
                text=runtime.last_error_message or "Scene metadata unavailable",
                icon="ERROR",
            )
        else:
            box.label(text=f"Nodes: {metadata['node_count']}")
            box.label(
                text=(
                    f"Bounds: ({metadata['aabb_min'][0]:.2f}, {metadata['aabb_min'][1]:.2f}, {metadata['aabb_min'][2]:.2f}) "
                    f"to ({metadata['aabb_max'][0]:.2f}, {metadata['aabb_max'][1]:.2f}, {metadata['aabb_max'][2]:.2f})"
                )
            )


class MATHOPS_V2_PT_quality(_MathOPSV2Panel, Panel):
    bl_label = "Quality"
    bl_parent_id = "MATHOPS_V2_PT_render_settings"
    bl_idname = "MATHOPS_V2_PT_quality"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        settings = context.scene.mathops_v2_settings
        col = layout.column()
        col.prop(settings, "culling_enabled")
        col.prop(settings, "final_grid_level")
        col.prop(settings, "num_samples")
        col.prop(settings, "viewport_max_dim")
        col.prop(settings, "gamma")


class MATHOPS_V2_PT_viewport(_MathOPSV2Panel, Panel):
    bl_label = "Viewport"
    bl_parent_id = "MATHOPS_V2_PT_render_settings"
    bl_idname = "MATHOPS_V2_PT_viewport"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        settings = context.scene.mathops_v2_settings
        col = layout.column()
        col.prop(settings, "viewport_preview")
        col.prop(settings, "shading_mode")
        if settings.shading_mode == "HEATMAP":
            col.prop(settings, "colormap_max")


class MATHOPS_V2_PT_bounds(_MathOPSV2Panel, Panel):
    bl_label = "Bounds"
    bl_parent_id = "MATHOPS_V2_PT_render_settings"
    bl_idname = "MATHOPS_V2_PT_bounds"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        settings = context.scene.mathops_v2_settings
        col = layout.column()
        col.prop(settings, "use_scene_bounds")
        if settings.use_scene_bounds:
            col.prop(settings, "dynamic_aabb")
        col.prop(settings, "show_aabb_overlay")
        if not settings.use_scene_bounds:
            col.prop(settings, "aabb_min")
            col.prop(settings, "aabb_max")


class MATHOPS_V2_PT_debug(_MathOPSV2Panel, Panel):
    bl_label = "Debug"
    bl_parent_id = "MATHOPS_V2_PT_render_settings"
    bl_idname = "MATHOPS_V2_PT_debug"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        settings = context.scene.mathops_v2_settings

        col = layout.column()
        col.prop(settings, "demo_anim_speed")
        row = col.row(align=True)
        if runtime.demo_anim_running:
            row.operator(
                operators.MATHOPS_V2_OT_stop_demo_anim.bl_idname, text="Stop Demo Anim"
            )
        else:
            row.operator(
                operators.MATHOPS_V2_OT_start_demo_anim.bl_idname, text="Play Demo Anim"
            )
        col.label(text="Viewport-only debug sphere", icon="INFO")


class MATHOPS_V2_PT_view3d_tools(Panel):
    bl_label = "MathOPS-v2"
    bl_idname = "MATHOPS_V2_PT_view3d_tools"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MathOPS-v2"

    @classmethod
    def poll(cls, context):
        settings = getattr(getattr(context, "scene", None), "mathops_v2_settings", None)
        return settings is not None and getattr(settings, "use_sdf_nodes", False)

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.mathops_v2_settings
        node = viewport_interaction.active_primitive_node(scene)
        record = sdf_proxies.active_record(scene)

        layout.operator(
            viewport_interaction.MATHOPS_V2_OT_pick_sdf.bl_idname, icon="EYEDROPPER"
        )
        layout.prop(settings, "viewport_transform_mode", expand=True)
        layout.operator(
            viewport_interaction.MATHOPS_V2_OT_duplicate_sdf.bl_idname,
            icon="DUPLICATE",
        )
        layout.label(text="Shortcuts: G/R/S, X/Y/Z, Shift+D")
        if record is None and node is None:
            layout.label(text="Active primitive: none", icon="INFO")
            layout.label(text="Pick in viewport to activate a primitive")
            return

        if node is not None:
            primitive_type = sdf_nodes.primitive_type_from_node(node) or "primitive"
            layout.label(text=f"Active primitive: {primitive_type}", icon="NODE")
            col = layout.column(align=True)
            col.label(text="Transform in viewport", icon="ORIENTATION_GIMBAL")
            col.prop(node, "color")
            if primitive_type == "box":
                col.prop(node, "size")
                col.prop(node, "bevel")
            else:
                col.prop(node, "radius")
                if primitive_type in {"cylinder", "cone"}:
                    col.prop(node, "height")
        elif record is not None:
            layout.label(
                text=f"Active primitive: {record.primitive_type}", icon="MESH_CUBE"
            )
            col = layout.column(align=True)
            col.label(text="Transform in viewport", icon="ORIENTATION_GIMBAL")
            col.prop(record, "color")
            if record.primitive_type == "box":
                col.prop(record, "size")
                col.prop(record, "bevel")
            else:
                col.prop(record, "radius")
                if record.primitive_type in {"cylinder", "cone"}:
                    col.prop(record, "height")


class MATHOPS_V2_PT_proxy_data(Panel):
    bl_label = "MathOPS-v2 SDF"
    bl_idname = "MATHOPS_V2_PT_proxy_data"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "data"

    @classmethod
    def poll(cls, context):
        del cls
        return _proxy_object(context) is not None

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = False
        layout.use_property_decorate = False

        obj = _proxy_object(context)
        primitive_id, tree, node = _linked_proxy_node(context)
        settings = getattr(getattr(context, "scene", None), "mathops_v2_settings", None)

        if settings is None or not getattr(settings, "use_sdf_nodes", False):
            layout.label(
                text="Enable Use Nodes to inspect the linked SDF node", icon="INFO"
            )
            return

        row = layout.row(align=True)
        row.label(
            text=f"Affecting Node: {node.name if node is not None else 'Missing'}",
            icon="NODE",
        )
        op = row.operator(
            sdf_nodes.MATHOPS_V2_OT_view_sdf_node.bl_idname,
            text="View Linked Node",
            icon="ZOOM_SELECTED",
        )
        op.primitive_id = primitive_id

        if tree is None:
            layout.label(text="Scene graph unavailable", icon="ERROR")
            return
        if node is None:
            layout.label(text="No linked graph node found for this empty", icon="ERROR")
            return

        primitive_type = sdf_nodes.primitive_type_from_node(node) or str(
            obj.get("sdf_type", "primitive")
        )
        layout.label(text=f"Primitive Type: {primitive_type}")
        layout.label(text="Showing direct drivers only", icon="FILTER")

        box = layout.box()
        box.label(text=f"Quick Edit: {node.name}")
        box.label(text="Transform in viewport", icon="ORIENTATION_GIMBAL")
        _draw_driver_prop(box, "Color", node, "Color", "color")
        if primitive_type == "box":
            _draw_driver_prop(box, "Size", node, "Size", "size")
            _draw_driver_prop(box, "Bevel", node, "Bevel", "bevel")
        else:
            _draw_driver_prop(box, "Radius", node, "Radius", "radius")
            if primitive_type in {"cylinder", "cone"}:
                _draw_driver_prop(box, "Height", node, "Height", "height")

        csg_nodes = _affecting_csg_nodes(node)
        if csg_nodes:
            ops_box = layout.box()
            ops_box.label(text="Direct Operation")
            csg_node = csg_nodes[0]
            csg_box = ops_box.box()
            csg_box.label(text=csg_node.name, icon="NODE")
            if getattr(csg_node, "bl_idname", "") == sdf_nodes.CSG_NODE_IDNAME:
                csg_box.prop(csg_node, "blend_mode", text="Operation")
            else:
                csg_box.label(text=f"Operation: {csg_node.bl_label}")
            _draw_driver_prop(
                csg_box,
                "Blend Radius",
                csg_node,
                "Blend Radius",
                "blend_radius",
            )
            if len(csg_nodes) > 1:
                ops_box.label(
                    text=f"{len(csg_nodes) - 1} downstream operation nodes hidden",
                    icon="INFO",
                )


def draw_shading_popover(self, context):
    if context.engine != runtime.ENGINE_ID:
        return

    space = getattr(context, "space_data", None)
    shading = getattr(space, "shading", None)
    if shading is None or shading.type != "RENDERED":
        return

    settings = context.scene.mathops_v2_settings
    layout = self.layout
    layout.label(text="Lighting")
    layout.template_icon_view(
        settings, "custom_matcap", show_labels=False, scale=4.0, scale_popup=3.0
    )
    if hasattr(shading, "show_specular_highlight"):
        layout.prop(shading, "show_specular_highlight", text="Specular Lighting")


classes = (
    MATHOPS_V2_PT_render_settings,
    MATHOPS_V2_PT_scene,
    MATHOPS_V2_PT_quality,
    MATHOPS_V2_PT_viewport,
    MATHOPS_V2_PT_bounds,
    MATHOPS_V2_PT_debug,
    MATHOPS_V2_PT_view3d_tools,
    MATHOPS_V2_PT_proxy_data,
)


def register():
    for cls in classes:
        import bpy

        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_PT_shading.append(draw_shading_popover)


def unregister():
    try:
        bpy.types.VIEW3D_PT_shading.remove(draw_shading_popover)
    except Exception:
        pass
    for cls in reversed(classes):
        import bpy

        bpy.utils.unregister_class(cls)
