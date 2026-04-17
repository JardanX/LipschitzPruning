import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, PointerProperty, StringProperty

from . import runtime


PRIMITIVE_ITEMS = (
    ("sphere", "Sphere", "Sphere SDF proxy"),
    ("box", "Box", "Box SDF proxy"),
    ("cylinder", "Cylinder", "Cylinder SDF proxy"),
    ("torus", "Torus", "Torus SDF proxy"),
)


def _tag_redraw(_self=None, context=None):
    runtime.clear_error()
    runtime.tag_redraw(context)


def _poll_tree(_self, tree):
    return getattr(tree, "bl_idname", "") == runtime.TREE_IDNAME


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
    light_direction: FloatVectorProperty(
        name="Light Direction",
        size=3,
        default=(0.45, 0.55, 0.7),
        subtype="DIRECTION",
        update=_tag_redraw,
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
