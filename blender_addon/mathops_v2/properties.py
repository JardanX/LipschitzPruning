import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

from . import runtime, sdf_nodes
from .render import bridge, matcap


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
        tree = getattr(settings, "sdf_node_tree", None)
        if tree is None or getattr(tree, "bl_idname", "") != sdf_nodes.TREE_IDNAME:
            return
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
    _redraw_viewports(context)


def _update_matcap(self, context):
    matcap.sync_viewport_matcap(context, getattr(self, "custom_matcap", ""))
    _redraw_viewports(context)


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
        default=True,
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


classes = (MathOPSV2Settings,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mathops_v2_settings = PointerProperty(type=MathOPSV2Settings)


def unregister():
    del bpy.types.Scene.mathops_v2_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
