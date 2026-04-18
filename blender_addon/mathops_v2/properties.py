import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, PointerProperty, StringProperty

from . import runtime
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
    gamma: FloatProperty(
        name="Gamma",
        default=1.2,
        min=0.5,
        max=4.0,
        description="Display gamma applied to viewport shading",
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


classes = (
    MathOPSV2ObjectSettings,
    MathOPSV2SceneSettings,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.mathops_v2_sdf = PointerProperty(type=MathOPSV2ObjectSettings)
    bpy.types.Scene.mathops_v2 = PointerProperty(type=MathOPSV2SceneSettings)


def unregister():
    del bpy.types.Scene.mathops_v2
    del bpy.types.Object.mathops_v2_sdf
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
