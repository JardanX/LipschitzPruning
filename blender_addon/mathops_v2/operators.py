import traceback

import bpy
from bpy.props import BoolProperty, StringProperty
from bpy.types import Operator

from . import runtime
from .render import bridge


class MATHOPS_V2_OT_clear_console(Operator):
    bl_idname = "mathops_v2.clear_console"
    bl_label = "Clear Console"
    bl_options = {"REGISTER"}

    def execute(self, context):
        runtime.clear_debug_log()
        runtime.debug_log("Console cleared")
        return {"FINISHED"}


class MATHOPS_V2_OT_load_template_scene(Operator):
    bl_idname = "mathops_v2.load_template_scene"
    bl_label = "Load Template Scene"
    bl_options = {"REGISTER"}

    template_id: StringProperty(name="Template")
    frame_camera: BoolProperty(name="Frame Camera", default=True)

    def execute(self, context):
        settings = context.scene.mathops_v2_settings
        settings.scene_source = "TEMPLATE"
        settings.template_scene = self.template_id
        settings.use_scene_bounds = True

        scene_path = bridge.template_scene_path(self.template_id)
        if not scene_path.is_file():
            self.report({"ERROR"}, f"Template scene not found: {scene_path}")
            return {"CANCELLED"}

        metadata = bridge.sync_scene_metadata(settings, scene_path)
        if self.frame_camera:
            bridge.frame_camera_to_aabb(
                context.scene, metadata["aabb_min"], metadata["aabb_max"]
            )
            runtime.debug_log(f"Framed scene camera to {scene_path.name}")

        context.scene.render.engine = runtime.ENGINE_ID
        bridge.force_redraw_viewports(context)
        runtime.debug_log(
            f"Loaded template {scene_path.stem}. Template scenes are external JSON assets; "
            "they do not create Blender viewport objects. Use Render Still or Render to Image."
        )
        self.report({"INFO"}, f"Loaded {scene_path.stem}")
        return {"FINISHED"}


class MATHOPS_V2_OT_refresh_scene(Operator):
    bl_idname = "mathops_v2.refresh_scene"
    bl_label = "Refresh Scene"
    bl_options = {"REGISTER"}

    frame_camera: BoolProperty(name="Frame Camera", default=False)

    def execute(self, context):
        settings = context.scene.mathops_v2_settings
        scene_path = bridge.resolve_scene_path(settings)
        if not scene_path.is_file():
            self.report({"ERROR"}, f"Scene file not found: {scene_path}")
            return {"CANCELLED"}

        metadata = bridge.sync_scene_metadata(settings, scene_path)
        if self.frame_camera:
            bridge.frame_camera_to_aabb(
                context.scene, metadata["aabb_min"], metadata["aabb_max"]
            )
            runtime.debug_log(f"Reframed scene camera to {scene_path.name}")

        context.scene.render.engine = runtime.ENGINE_ID
        bridge.force_redraw_viewports(context)
        runtime.debug_log(f"Refreshed scene source {scene_path}")
        self.report({"INFO"}, f"Refreshed {scene_path.name}")
        return {"FINISHED"}


class MATHOPS_V2_OT_render_image(Operator):
    bl_idname = "mathops_v2.render_image"
    bl_label = "Render to Image"
    bl_options = {"REGISTER"}

    def execute(self, context):
        settings = context.scene.mathops_v2_settings
        try:
            runtime.debug_log("Render to Image requested")
            rgba, width, height, _scene_path = bridge.render_rgba(context.scene)
        except Exception as exc:
            bridge.set_last_error(settings, str(exc))
            traceback.print_exc()
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        pixels = bridge.rgba_to_pixels(rgba, width, height)
        image = bpy.data.images.get(settings.image_name)
        if image is None:
            image = bpy.data.images.new(
                settings.image_name, width=width, height=height, alpha=True
            )
        elif image.size[0] != width or image.size[1] != height:
            image.scale(width, height)

        image.pixels.foreach_set(pixels.reshape(-1))
        image.update()
        context.scene.render.engine = runtime.ENGINE_ID
        self.report({"INFO"}, f"Render {runtime.last_render_stats['render_ms']:.2f} ms")
        return {"FINISHED"}


classes = (
    MATHOPS_V2_OT_clear_console,
    MATHOPS_V2_OT_load_template_scene,
    MATHOPS_V2_OT_refresh_scene,
    MATHOPS_V2_OT_render_image,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
