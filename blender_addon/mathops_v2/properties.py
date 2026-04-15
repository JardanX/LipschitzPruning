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

from . import runtime
from .render import bridge


class MathOPSV2Settings(bpy.types.PropertyGroup):
    scene_source: EnumProperty(
        name="Scene Source",
        items=(
            ("TEMPLATE", "Template", "Use one of the bundled template scenes"),
            ("CUSTOM", "Custom", "Use a custom JSON scene file"),
        ),
        default="TEMPLATE",
    )
    template_scene: EnumProperty(
        name="Template Scene",
        items=[
            (identifier, label, description)
            for identifier, label, _filename, description in runtime.TEMPLATE_SCENES
        ],
        default="TREES",
    )
    scene_path: StringProperty(name="Scene JSON", subtype="FILE_PATH")
    shader_dir: StringProperty(
        name="Shader Dir", subtype="DIR_PATH", default=str(bridge.addon_dir())
    )
    image_name: StringProperty(name="Image Name", default="MathOPS-v2 Render")
    final_grid_level: IntProperty(name="Final Grid Level", default=8, min=2, max=8)
    num_samples: IntProperty(name="Samples", default=1, min=1, max=64)
    gamma: FloatProperty(name="Gamma", default=1.2, min=0.5, max=4.0)
    viewport_preview: BoolProperty(name="Viewport Preview", default=True)
    culling_enabled: BoolProperty(name="Enable Pruning", default=True)
    recompute_pruning: BoolProperty(name="Recompute Pruning", default=True)
    use_scene_bounds: BoolProperty(name="Use Scene Bounds", default=True)
    colormap_max: IntProperty(name="Colormap Max", default=25, min=1, max=64)
    aabb_min: FloatVectorProperty(
        name="AABB Min", size=3, default=(-1.0, -1.0, -1.0), precision=3
    )
    aabb_max: FloatVectorProperty(
        name="AABB Max", size=3, default=(1.0, 1.0, 1.0), precision=3
    )
    shading_mode: EnumProperty(
        name="Shading",
        items=(
            ("SHADED", "Shaded", "Shaded lighting"),
            ("HEATMAP", "Heatmap", "Evaluation heatmap"),
            ("NORMALS", "Normals", "Normal debug"),
            ("AO", "AO", "Ambient occlusion"),
        ),
        default="SHADED",
    )
    last_scene_name: StringProperty(name="Last Scene", default="")
    last_scene_path: StringProperty(name="Last Scene Path", default="")
    last_node_count: IntProperty(name="Last Node Count", default=0, min=0)
    last_render_ms: FloatProperty(name="Render", default=0.0)
    last_tracing_ms: FloatProperty(name="Tracing", default=0.0)
    last_culling_ms: FloatProperty(name="Culling", default=0.0)
    last_eval_grid_ms: FloatProperty(name="Eval Grid", default=0.0)
    last_pruning_mem_gb: FloatProperty(name="Pruning VRAM", default=0.0)
    last_tracing_mem_gb: FloatProperty(name="Tracing VRAM", default=0.0)
    last_active_ratio: FloatProperty(name="Active Ratio", default=0.0, min=0.0, max=1.0)
    last_tmp_ratio: FloatProperty(name="Temp Ratio", default=0.0, min=0.0, max=1.0)
    last_error: StringProperty(name="Last Error", default="")
    show_console: BoolProperty(name="Show Console", default=True)


classes = (MathOPSV2Settings,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mathops_v2_settings = PointerProperty(type=MathOPSV2Settings)


def unregister():
    del bpy.types.Scene.mathops_v2_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
