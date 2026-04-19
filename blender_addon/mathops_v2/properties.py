import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, PointerProperty, StringProperty

from . import runtime
from .nodes import sdf_tree
from .render import matcap


PRIMITIVE_ITEMS = (
    ("sphere", "Sphere", "Sphere SDF proxy"),
    ("box", "Box", "Box SDF proxy"),
    ("cylinder", "Cylinder", "Cylinder SDF proxy"),
    ("torus", "Torus", "Torus SDF proxy"),
)


def _tag_redraw(_self=None, context=None):
    runtime.clear_error()
    runtime.note_interaction()
    runtime.tag_redraw(context)


def _poll_tree(_self, tree):
    return getattr(tree, "bl_idname", "") == runtime.TREE_IDNAME


def _update_custom_matcap(self, context):
    matcap.sync_viewport_matcap(context, getattr(self, "custom_matcap", ""))
    _tag_redraw(self, context)


def _preferred_matcap_identifier(context):
    items = matcap.get_matcaps_enum(None, context)
    for item in items:
        identifier = str(item[0] or "")
        if identifier and identifier != "NONE":
            return identifier
    return ""


def _valid_matcap_identifiers(context):
    items = matcap.get_matcaps_enum(None, context)
    return {str(item[0] or "") for item in items if str(item[0] or "") and str(item[0] or "") != "NONE"}


def ensure_scene_defaults(scene, context=None):
    settings = runtime.scene_settings(scene)
    if settings is None:
        return False

    changed = False
    tree = getattr(settings, "node_tree", None)
    if tree is None or getattr(tree, "bl_idname", "") != runtime.TREE_IDNAME:
        tree = sdf_tree.ensure_scene_tree(scene)
        if getattr(settings, "node_tree", None) != tree:
            settings.node_tree = tree
        changed = True

    if context is None:
        context = getattr(bpy, "context", None)
    if context is not None:
        valid_matcaps = _valid_matcap_identifiers(context)
        preferred_matcap = _preferred_matcap_identifier(context)
        current_matcap = str(getattr(settings, "custom_matcap", "") or "")
        if preferred_matcap and current_matcap not in valid_matcaps:
            try:
                settings.custom_matcap = preferred_matcap
                changed = True
            except Exception:
                pass

    return changed


def _deferred_ensure_scene_defaults():
    scenes = getattr(bpy.data, "scenes", None)
    if scenes is None:
        return None
    context = getattr(bpy, "context", None)
    for scene in scenes:
        ensure_scene_defaults(scene, context=context)
    return None


@persistent
def _on_load_post(_dummy):
    _deferred_ensure_scene_defaults()


class MathOPSV2ObjectSettings(bpy.types.PropertyGroup):
    enabled: BoolProperty(default=False, update=_tag_redraw)
    proxy_id: StringProperty(default="", options={"HIDDEN"})
    source_tree_name: StringProperty(default="", options={"HIDDEN"})
    source_node_name: StringProperty(default="", options={"HIDDEN"})
    primitive_type: EnumProperty(items=PRIMITIVE_ITEMS, default="sphere", update=_tag_redraw)
    radius: FloatProperty(name="Radius", default=0.5, min=0.001, update=_tag_redraw)
    size: FloatVectorProperty(
        name="Half Size",
        size=3,
        default=(0.5, 0.5, 0.5),
        min=0.001,
        subtype="XYZ",
        update=_tag_redraw,
    )
    height: FloatProperty(name="Height", default=1.0, min=0.001, update=_tag_redraw)
    major_radius: FloatProperty(name="Major Radius", default=0.75, min=0.001, update=_tag_redraw)
    minor_radius: FloatProperty(name="Minor Radius", default=0.25, min=0.001, update=_tag_redraw)


class MathOPSV2SceneSettings(bpy.types.PropertyGroup):
    viewport_preview: BoolProperty(
        name="Viewport Preview",
        default=True,
        description="Draw the SDF raymarch in rendered view",
        update=_tag_redraw,
    )
    node_tree: PointerProperty(name="SDF Graph", type=bpy.types.NodeTree, poll=_poll_tree, update=_tag_redraw)
    max_steps: bpy.props.IntProperty(name="Max Steps", default=96, min=8, max=512, update=_tag_redraw)
    max_distance: FloatProperty(name="Max Distance", default=200.0, min=1.0, update=_tag_redraw)
    surface_epsilon: FloatProperty(name="Surface Epsilon", default=0.0015, min=0.0001, max=0.1, precision=5, update=_tag_redraw)
    mesh_resolution: bpy.props.IntProperty(
        name="Mesh Resolution",
        default=48,
        min=8,
        max=256,
        description="Dual contouring cells along the longest scene axis",
        update=_tag_redraw,
    )
    mesh_algorithm: EnumProperty(
        name="Mesh Algorithm",
        items=(
            ("DUAL_CONTOURING", "Dual Contouring", "CPU dual contouring mesh extraction"),
            ("ISO_SIMPLEX", "Iso Simplex", "CPU simplex-based iso-surface extraction"),
        ),
        default="DUAL_CONTOURING",
        update=_tag_redraw,
    )
    mesh_smooth_shading: BoolProperty(
        name="Smooth Shading",
        default=True,
        description="Apply smooth shading to generated CPU meshes; disable for flat shading",
        update=_tag_redraw,
    )
    gamma: FloatProperty(
        name="Gamma",
        default=1.2,
        min=0.5,
        max=4.0,
        description="Display gamma applied to viewport shading",
        update=_tag_redraw,
    )
    disable_surface_shading: BoolProperty(
        name="Disable Surface Shading",
        default=False,
        description="Use a fixed normal direction instead of estimating hit normals, to measure shading cost",
        update=_tag_redraw,
    )
    light_direction: FloatVectorProperty(
        name="Light Direction",
        size=3,
        default=(0.45, 0.55, 0.7),
        subtype="DIRECTION",
        update=_tag_redraw,
    )
    culling_enabled: BoolProperty(
        name="Pruning Enabled",
        default=True,
        description="Use Lipschitz pruning to accelerate the raymarch",
        update=_tag_redraw,
    )
    pruning_grid_level: bpy.props.IntProperty(
        name="Pruning Grid Level",
        default=8,
        min=2,
        max=8,
        description="Even grid exponent used by the pruning hierarchy (4 = 16^3 cells)",
        update=_tag_redraw,
    )
    debug_shading: EnumProperty(
        name="Debug Shading",
        items=(
            ("SHADED", "Shaded", "Normal shaded raymarch"),
            ("PRUNING_ACTIVE", "Pruning Active", "Visualize active node counts per cell"),
            ("PRUNING_FIELD", "Pruning Field", "Visualize the far-field cell values"),
            ("STEP_COUNT", "Step Count", "Number of fine march steps per pixel"),
        ),
        default="SHADED",
        update=_tag_redraw,
    )
    colormap_max: bpy.props.IntProperty(
        name="Heatmap Max",
        default=25,
        min=1,
        max=64,
        description="Upper range for heatmap shading",
        update=_tag_redraw,
    )
    custom_matcap: EnumProperty(
        name="Matcap",
        description="Viewport matcap used by MathOPS UI",
        items=matcap.get_matcaps_enum,
        update=_update_custom_matcap,
    )
    outline_color: FloatVectorProperty(
        name="Outline Color",
        size=3,
        subtype="COLOR",
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0),
        description="Base color for unselected SDF outlines",
        update=_tag_redraw,
    )
    outline_opacity: FloatProperty(
        name="Outline Opacity",
        min=0.0,
        max=1.0,
        default=1.0,
        subtype="FACTOR",
        description="Opacity used for all SDF outlines",
        update=_tag_redraw,
    )
    show_grid: BoolProperty(
        name="Grid",
        description="Show the custom grid in orthographic views",
        default=True,
        update=_tag_redraw,
    )
    show_floor: BoolProperty(
        name="Floor",
        description="Show the custom ground plane in perspective views",
        default=True,
        update=_tag_redraw,
    )
    show_axis_x: BoolProperty(
        name="X",
        description="Show the X axis line",
        default=True,
        update=_tag_redraw,
    )
    show_axis_y: BoolProperty(
        name="Y",
        description="Show the Y axis line",
        default=True,
        update=_tag_redraw,
    )
    show_axis_z: BoolProperty(
        name="Z",
        description="Show the Z axis line",
        default=False,
        update=_tag_redraw,
    )
    grid_scale_rectangular: FloatProperty(
        name="Scale",
        description="Grid scale for the rectangular grid",
        min=0.001,
        max=1000.0,
        default=1.0,
        precision=3,
        update=_tag_redraw,
    )
    grid_subdivisions_rectangular: bpy.props.IntProperty(
        name="Subdivisions",
        description="Number of rectangular grid subdivisions",
        min=1,
        max=100,
        default=1,
        update=_tag_redraw,
    )
    grid_scale_radial: FloatProperty(
        name="Scale",
        description="Grid scale for the radial grid",
        min=0.001,
        max=1000.0,
        default=1.0,
        precision=3,
        update=_tag_redraw,
    )
    grid_subdivisions_radial: bpy.props.IntProperty(
        name="Subdivisions",
        description="Number of radial grid subdivisions",
        min=1,
        max=100,
        default=12,
        update=_tag_redraw,
    )
    grid_type: EnumProperty(
        name="Grid Type",
        description="Type of custom grid to draw",
        items=(
            ("RECTANGULAR", "Rectangular", "Standard rectangular grid"),
            ("RADIAL", "Radial", "Radial grid with concentric circles"),
        ),
        default="RECTANGULAR",
        update=_tag_redraw,
    )
    grid_overlay_initialized: BoolProperty(default=False, options={"HIDDEN"})
    native_overlay_initialized: BoolProperty(default=False, options={"HIDDEN"})
    native_show_floor: BoolProperty(default=True, options={"HIDDEN"})
    native_show_ortho_grid: BoolProperty(default=True, options={"HIDDEN"})
    native_show_axis_x: BoolProperty(default=True, options={"HIDDEN"})
    native_show_axis_y: BoolProperty(default=True, options={"HIDDEN"})
    native_show_axis_z: BoolProperty(default=False, options={"HIDDEN"})
    native_grid_scale: FloatProperty(default=1.0, min=0.001, max=1000.0, options={"HIDDEN"})
    native_grid_subdivisions: bpy.props.IntProperty(default=1, min=1, max=100, options={"HIDDEN"})


classes = (
    MathOPSV2ObjectSettings,
    MathOPSV2SceneSettings,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.mathops_v2_sdf = PointerProperty(type=MathOPSV2ObjectSettings)
    bpy.types.Scene.mathops_v2 = PointerProperty(type=MathOPSV2SceneSettings)
    scenes = getattr(bpy.data, "scenes", None)
    context = getattr(bpy, "context", None)
    if scenes is not None:
        for scene in scenes:
            ensure_scene_defaults(scene, context=context)
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)
    bpy.app.timers.register(_deferred_ensure_scene_defaults, first_interval=0.0)


def unregister():
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    try:
        if bpy.app.timers.is_registered(_deferred_ensure_scene_defaults):
            bpy.app.timers.unregister(_deferred_ensure_scene_defaults)
    except Exception:
        pass
    del bpy.types.Scene.mathops_v2
    del bpy.types.Object.mathops_v2_sdf
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
