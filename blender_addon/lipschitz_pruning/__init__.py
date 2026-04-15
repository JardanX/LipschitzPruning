bl_info = {
    "name": "Lipschitz Pruning",
    "author": "OpenCode",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "Render > Lipschitz Pruning",
    "category": "Render",
    "description": "Thin Blender bridge for the Vulkan Lipschitz pruning renderer",
}

from pathlib import Path

import bpy
import numpy as np
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Operator, Panel, PropertyGroup


_native_module = None
_renderer = None
_renderer_key = None


def _addon_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_native_module():
    global _native_module
    if _native_module is None:
        from . import lipschitz_pruning_native as native_module

        _native_module = native_module
    return _native_module


def _get_render_size(scene):
    scale = scene.render.resolution_percentage / 100.0
    width = max(1, int(scene.render.resolution_x * scale))
    height = max(1, int(scene.render.resolution_y * scale))
    return width, height


def _camera_target(camera):
    origin = camera.matrix_world.translation
    forward = -(camera.matrix_world.to_3x3().col[2])
    target = origin + forward
    return tuple(origin), tuple(target)


def _shading_mode_value(native_module, shading_mode):
    return {
        "SHADED": native_module.SHADING_MODE_SHADED,
        "HEATMAP": native_module.SHADING_MODE_HEATMAP,
        "NORMALS": native_module.SHADING_MODE_NORMALS,
        "AO": native_module.SHADING_MODE_AO,
    }[shading_mode]


def _get_renderer(settings, width, height):
    global _renderer, _renderer_key

    native_module = _load_native_module()
    grid_level = max(2, min(8, settings.final_grid_level))
    if grid_level % 2 != 0:
        grid_level -= 1
    shader_dir = (
        bpy.path.abspath(settings.shader_dir)
        if settings.shader_dir
        else str(_addon_dir())
    )
    key = (
        shader_dir,
        width,
        height,
        grid_level,
        settings.shading_mode,
        settings.culling_enabled,
        settings.num_samples,
        settings.gamma,
    )

    if _renderer is None or _renderer_key != key:
        _renderer = native_module.Renderer(
            shader_dir,
            width,
            height,
            grid_level,
            _shading_mode_value(native_module, settings.shading_mode),
            settings.culling_enabled,
            settings.num_samples,
            settings.gamma,
        )
        _renderer_key = key

    return _renderer


class LipschitzPruningSettings(PropertyGroup):
    scene_path: StringProperty(name="Scene JSON", subtype="FILE_PATH")
    shader_dir: StringProperty(
        name="Shader Dir", subtype="DIR_PATH", default=str(_addon_dir())
    )
    image_name: StringProperty(name="Image Name", default="LipschitzPruning Render")
    final_grid_level: IntProperty(name="Final Grid Level", default=8, min=2, max=8)
    num_samples: IntProperty(name="Samples", default=1, min=1, max=64)
    gamma: FloatProperty(name="Gamma", default=1.2, min=0.5, max=4.0)
    culling_enabled: BoolProperty(name="Enable Pruning", default=True)
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


class LIPSCHITZ_PRUNING_OT_render(Operator):
    bl_idname = "lipschitz_pruning.render"
    bl_label = "Render Lipschitz Scene"
    bl_options = {"REGISTER"}

    def execute(self, context):
        settings = context.scene.lipschitz_pruning_settings
        scene_path = bpy.path.abspath(settings.scene_path)

        if not scene_path:
            self.report({"ERROR"}, "Set a JSON scene path")
            return {"CANCELLED"}

        if not Path(scene_path).is_file():
            self.report({"ERROR"}, f"Scene file not found: {scene_path}")
            return {"CANCELLED"}

        camera = context.scene.camera
        if camera is None:
            self.report({"ERROR"}, "Set an active scene camera")
            return {"CANCELLED"}

        width, height = _get_render_size(context.scene)
        try:
            renderer = _get_renderer(settings, width, height)
            renderer.load_scene_file(scene_path)

            camera_position, camera_target = _camera_target(camera)
            rgba = renderer.render_rgba(camera_position, camera_target)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        pixels = np.frombuffer(rgba, dtype=np.uint8).astype(np.float32) / 255.0
        pixels = pixels.reshape((height, width, 4))[::-1].copy()

        image = bpy.data.images.get(settings.image_name)
        if image is None:
            image = bpy.data.images.new(
                settings.image_name, width=width, height=height, alpha=True
            )
        elif image.size[0] != width or image.size[1] != height:
            image.scale(width, height)

        image.pixels.foreach_set(pixels.reshape(-1))
        image.update()

        timings = renderer.last_timings()
        self.report(
            {"INFO"},
            f"Render {timings['render_ms']:.2f} ms, tracing {timings['tracing_ms']:.2f} ms",
        )
        return {"FINISHED"}


class LIPSCHITZ_PRUNING_PT_panel(Panel):
    bl_label = "Lipschitz Pruning"
    bl_idname = "LIPSCHITZ_PRUNING_PT_panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.lipschitz_pruning_settings

        layout.prop(settings, "scene_path")
        layout.prop(settings, "shader_dir")
        layout.prop(settings, "image_name")
        layout.prop(settings, "final_grid_level")
        layout.prop(settings, "num_samples")
        layout.prop(settings, "gamma")
        layout.prop(settings, "culling_enabled")
        layout.prop(settings, "shading_mode")
        layout.operator(LIPSCHITZ_PRUNING_OT_render.bl_idname, icon="RENDER_STILL")


classes = (
    LipschitzPruningSettings,
    LIPSCHITZ_PRUNING_OT_render,
    LIPSCHITZ_PRUNING_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.lipschitz_pruning_settings = PointerProperty(
        type=LipschitzPruningSettings
    )


def unregister():
    global _renderer, _renderer_key, _native_module
    del bpy.types.Scene.lipschitz_pruning_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    _renderer = None
    _renderer_key = None
    _native_module = None
