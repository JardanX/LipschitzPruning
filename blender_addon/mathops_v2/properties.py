from contextlib import contextmanager

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

from . import runtime, sdf_nodes
from .render import bridge, matcap


_record_updates_suppressed = 0


@contextmanager
def suppress_record_updates():
    global _record_updates_suppressed
    _record_updates_suppressed += 1
    try:
        yield
    finally:
        _record_updates_suppressed -= 1


def _mark_record_dirty(record, context):
    if _record_updates_suppressed > 0:
        return
    record_ptr = record.as_pointer()
    for scene in bpy.data.scenes:
        records = getattr(scene, "mathops_v2_primitives", ())
        for candidate in records:
            if candidate.as_pointer() != record_ptr:
                continue
            settings = getattr(scene, "mathops_v2_settings", None)
            if settings is None:
                return
            try:
                tree = sdf_nodes.get_selected_tree(settings, create=False, ensure=False)
            except Exception:
                return
            from . import sdf_proxies

            record_id = str(getattr(record, "primitive_id", "") or "")
            node = None
            if tree is not None and record_id:
                node = sdf_nodes.find_primitive_node(tree, record_id)
            if node is not None:
                sdf_nodes.mark_tree_transform_dirty(tree)
                try:
                    sdf_proxies.sync_primitive_node_update(scene, node, context)
                except Exception:
                    sdf_nodes.mark_tree_dirty(tree)
                    try:
                        sdf_proxies.sync_from_graph(context)
                    except Exception:
                        pass
            else:
                sdf_nodes.mark_tree_dirty(tree)
                try:
                    sdf_proxies.sync_from_graph(context)
                except Exception:
                    pass
            _redraw_viewports(context)
            return


def _redraw_viewports(context):
    if context is not None:
        bridge.force_redraw_viewports(context)


def _reset_bounds_state():
    runtime.dynamic_aabb_state.clear()
    runtime.current_effective_aabb = None


def _sync_auto_bounds(settings):
    if not getattr(settings, "use_scene_bounds", False):
        return
    if getattr(settings, "use_sdf_nodes", False):
        try:
            tree = sdf_nodes.get_selected_tree(settings, create=False, ensure=False)
        except Exception:
            return
        scene_cache = bridge.graph_scene_cache(settings)
        metadata = scene_cache["metadata"] if scene_cache is not None else None
    else:
        metadata = bridge.safe_scene_metadata(bridge.resolve_scene_path(settings))
    if metadata is None:
        return
    settings.aabb_min = metadata["aabb_min"]
    settings.aabb_max = metadata["aabb_max"]


def _update_template_scene(self, context):
    _reset_bounds_state()
    _sync_auto_bounds(self)
    runtime.last_error_message = ""
    _redraw_viewports(context)


def _update_use_sdf_nodes(self, context):
    _reset_bounds_state()
    if self.use_sdf_nodes and context is not None:
        sdf_nodes.ensure_scene_tree(context.scene, self)
        try:
            sdf_nodes.focus_scene_tree(context, create=True)
        except Exception:
            pass
    _sync_auto_bounds(self)
    runtime.last_error_message = ""
    _redraw_viewports(context)


def _poll_sdf_tree(_self, tree):
    return getattr(tree, "bl_idname", "") == sdf_nodes.TREE_IDNAME


def _update_final_grid_level(self, context):
    value = max(2, min(8, int(self.final_grid_level)))
    if value % 2 != 0:
        value -= 1
    if self.final_grid_level != value:
        self.final_grid_level = value
        return
    _redraw_viewports(context)


def _update_bounds_mode(self, context):
    _reset_bounds_state()
    _sync_auto_bounds(self)
    _redraw_viewports(context)


def _update_manual_bounds(self, context):
    if self.use_scene_bounds:
        return
    _reset_bounds_state()
    _redraw_viewports(context)


def _update_viewport(self, context):
    try:
        from . import sdf_proxies

        sdf_proxies.sync_native_transform_gizmo(context)
    except Exception:
        pass
    _redraw_viewports(context)


def _update_matcap(self, context):
    matcap.sync_viewport_matcap(context, getattr(self, "custom_matcap", ""))
    _redraw_viewports(context)


_PRIMITIVE_ITEMS = (
    ("sphere", "Sphere", "SDF sphere"),
    ("box", "Box", "SDF box"),
    ("cylinder", "Cylinder", "SDF cylinder"),
    ("cone", "Cone", "SDF cone"),
)


class MathOPSV2PrimitiveItem(bpy.types.PropertyGroup):
    primitive_id: StringProperty(default="")
    primitive_type: EnumProperty(
        items=_PRIMITIVE_ITEMS, default="sphere", update=_mark_record_dirty
    )
    location: FloatVectorProperty(
        size=3, default=(0.0, 0.0, 0.0), update=_mark_record_dirty
    )
    rotation: FloatVectorProperty(
        size=3, default=(0.0, 0.0, 0.0), update=_mark_record_dirty
    )
    scale: FloatVectorProperty(
        size=3, default=(1.0, 1.0, 1.0), update=_mark_record_dirty
    )
    color: FloatVectorProperty(
        size=3,
        subtype="COLOR",
        min=0.0,
        max=1.0,
        default=(0.8, 0.8, 0.8),
        update=_mark_record_dirty,
    )
    size: FloatVectorProperty(
        size=3, min=0.001, default=(1.0, 1.0, 1.0), update=_mark_record_dirty
    )
    radius: FloatProperty(min=0.0, default=0.5, update=_mark_record_dirty)
    height: FloatProperty(min=0.0, default=1.0, update=_mark_record_dirty)
    bevel: FloatProperty(min=0.0, default=0.0, update=_mark_record_dirty)


class MathOPSV2Settings(bpy.types.PropertyGroup):
    use_sdf_nodes: BoolProperty(
        name="Use Nodes",
        default=True,
        description="Render a scene-owned SDF node graph instead of a bundled template",
        update=_update_use_sdf_nodes,
    )
    template_scene: EnumProperty(
        name="Template Scene",
        items=[
            (identifier, label, description)
            for identifier, label, _filename, description in runtime.TEMPLATE_SCENES
        ],
        default="TREES",
        update=_update_template_scene,
    )
    sdf_node_tree: PointerProperty(
        name="SDF Tree",
        type=bpy.types.NodeTree,
        poll=_poll_sdf_tree,
        update=_update_template_scene,
    )
    shader_dir: StringProperty(
        name="Shader Dir",
        subtype="DIR_PATH",
        default=str(bridge.addon_dir()),
        description="Developer override for the compiled shader directory",
    )
    final_grid_level: IntProperty(
        name="Grid Level",
        default=8,
        min=2,
        max=8,
        description="Even pruning grid level. Higher values improve culling precision at a memory cost",
        update=_update_final_grid_level,
    )
    num_samples: IntProperty(
        name="Samples",
        default=1,
        min=1,
        max=64,
        description="Shading samples per pixel",
        update=_update_viewport,
    )
    viewport_max_dim: IntProperty(
        name="Viewport Max Dim",
        default=0,
        min=0,
        max=8192,
        description="Maximum internal viewport render dimension, 0 keeps full region resolution",
        update=_update_viewport,
    )
    gamma: FloatProperty(
        name="Gamma",
        default=1.2,
        min=0.5,
        max=4.0,
        description="Display gamma applied by the renderer",
        update=_update_viewport,
    )
    viewport_preview: BoolProperty(
        name="Viewport Preview",
        default=True,
        description="Draw the MathOPS-v2 viewport preview in Rendered mode",
        update=_update_viewport,
    )
    viewport_transform_mode: EnumProperty(
        name="Transform Mode",
        items=(
            ("TRANSLATE", "Move", "Move the active SDF primitive"),
            ("ROTATE", "Rotate", "Rotate the active SDF primitive"),
            ("SCALE", "Scale", "Scale the active SDF primitive"),
        ),
        default="TRANSLATE",
        update=_update_viewport,
    )
    demo_anim_speed: FloatProperty(
        name="Demo Anim Speed",
        default=4.0,
        min=0.0,
        max=4.0,
        description="Playback speed for the viewport-only debug sphere",
        update=_update_viewport,
    )
    dynamic_aabb: BoolProperty(
        name="Dynamic Bounds",
        default=True,
        description="Allow auto bounds to grow and shrink around scene content during viewport rendering",
        update=_update_bounds_mode,
    )
    show_aabb_overlay: BoolProperty(
        name="Show Bounds Overlay",
        default=False,
        description="Draw the current effective bounds in the 3D viewport",
        update=_update_viewport,
    )
    culling_enabled: BoolProperty(
        name="Enable Culling",
        default=True,
        description="Skip branches outside the active bounds to reduce shading work",
        update=_update_viewport,
    )
    use_scene_bounds: BoolProperty(
        name="Auto Bounds",
        default=True,
        description="Use scene-derived bounds instead of manual bounds",
        update=_update_bounds_mode,
    )
    colormap_max: IntProperty(
        name="Heatmap Max",
        default=25,
        min=1,
        max=64,
        description="Upper range for heatmap shading",
        update=_update_viewport,
    )
    aabb_min: FloatVectorProperty(
        name="Bounds Min",
        size=3,
        default=(-1.0, -1.0, -1.0),
        precision=3,
        update=_update_manual_bounds,
    )
    aabb_max: FloatVectorProperty(
        name="Bounds Max",
        size=3,
        default=(1.0, 1.0, 1.0),
        precision=3,
        update=_update_manual_bounds,
    )
    shading_mode: EnumProperty(
        name="Shading",
        items=(
            ("SHADED", "Shaded", "Matcap surface shading"),
            ("HEATMAP", "Heatmap", "Evaluation heatmap"),
            ("NORMALS", "Normals", "Surface normal debug"),
            ("AO", "AO", "Compatibility alias for shaded mode"),
        ),
        default="SHADED",
        update=_update_viewport,
    )
    custom_matcap: EnumProperty(
        name="Matcap",
        description="Matcap used by viewport shaded mode",
        items=matcap.get_matcaps_enum,
        update=_update_matcap,
    )


classes = (
    MathOPSV2PrimitiveItem,
    MathOPSV2Settings,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mathops_v2_settings = PointerProperty(type=MathOPSV2Settings)
    bpy.types.Scene.mathops_v2_scene_id = StringProperty(default="", options={"HIDDEN"})
    bpy.types.Scene.mathops_v2_primitives = CollectionProperty(
        type=MathOPSV2PrimitiveItem
    )
    bpy.types.Scene.mathops_v2_active_primitive_id = StringProperty(
        default="", options={"HIDDEN"}
    )
    bpy.types.Scene.mathops_v2_handle_object_name = StringProperty(
        default="", options={"HIDDEN"}
    )
    bpy.types.Object.mathops_v2_sdf_proxy = BoolProperty(
        default=False, options={"HIDDEN"}
    )
    bpy.types.Object.mathops_v2_sdf_proxy_id = StringProperty(
        default="", options={"HIDDEN"}
    )
    bpy.types.Object.mathops_v2_sdf_node_id = StringProperty(
        default="", options={"HIDDEN"}
    )
    bpy.types.Object.mathops_v2_sdf_handle = BoolProperty(
        default=False, options={"HIDDEN"}
    )
    bpy.types.Object.mathops_v2_sdf_handle_id = StringProperty(
        default="", options={"HIDDEN"}
    )


def unregister():
    del bpy.types.Object.mathops_v2_sdf_handle_id
    del bpy.types.Object.mathops_v2_sdf_handle
    del bpy.types.Object.mathops_v2_sdf_node_id
    del bpy.types.Object.mathops_v2_sdf_proxy_id
    del bpy.types.Object.mathops_v2_sdf_proxy
    del bpy.types.Scene.mathops_v2_handle_object_name
    del bpy.types.Scene.mathops_v2_active_primitive_id
    del bpy.types.Scene.mathops_v2_primitives
    del bpy.types.Scene.mathops_v2_scene_id
    del bpy.types.Scene.mathops_v2_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
