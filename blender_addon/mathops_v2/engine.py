import traceback

import gpu
import numpy as np
import bpy
from bpy.types import RenderEngine
from gpu_extras.presets import draw_texture_2d

from . import runtime
from .render import bridge, gpu_viewport


class MathOPSV2RenderEngine(RenderEngine):
    bl_idname = runtime.ENGINE_ID
    bl_label = "MathOPS-v2"
    bl_use_preview = False
    bl_use_postprocess = False
    bl_use_eevee_viewport = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gpu_viewport = gpu_viewport.MathOPSV2GPUViewport()
        self._viewport_texture = None
        self._viewport_signature = None
        self._viewport_render_size = (0, 0)
        self._logged_ortho_warning = False

    def __del__(self):
        if getattr(self, "_gpu_viewport", None) is not None:
            self._gpu_viewport.free()

    def _upload_viewport_texture(self, rgba, width, height):
        pixels = bridge.rgba_to_pixels(rgba, width, height)
        flat = np.ascontiguousarray(pixels.reshape(-1), dtype=np.float32)
        buf = gpu.types.Buffer("FLOAT", len(flat), flat)
        if self._viewport_texture is not None:
            try:
                self._viewport_texture.free()
            except Exception:
                pass
        self._viewport_texture = gpu.types.GPUTexture(
            (width, height), format="RGBA32F", data=buf
        )
        self._viewport_render_size = (width, height)

    def _draw_viewport_texture(self, scene, region):
        if self._viewport_texture is None:
            return
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("NONE")
        gpu.state.depth_mask_set(False)
        self.bind_display_space_shader(scene)
        draw_texture_2d(self._viewport_texture, (0, 0), region.width, region.height)
        self.unbind_display_space_shader()
        gpu.state.blend_set("NONE")

    def _viewport_signature_for(
        self,
        context,
        settings,
        scene_path,
        width,
        height,
        camera_position,
        camera_target_value,
        camera_up,
        fov_y,
    ):
        scene_mtime = scene_path.stat().st_mtime_ns if scene_path.is_file() else 0
        return (
            str(scene_path),
            scene_mtime,
            width,
            height,
            context.region.width,
            context.region.height,
            bridge.grid_level(settings),
            settings.viewport_preview,
            settings.shading_mode,
            settings.culling_enabled,
            settings.recompute_pruning,
            settings.num_samples,
            round(float(settings.gamma), 4),
            settings.use_scene_bounds,
            tuple(round(float(v), 4) for v in settings.aabb_min),
            tuple(round(float(v), 4) for v in settings.aabb_max),
            bridge.camera_signature(camera_position, camera_target_value),
            tuple(round(float(v), 4) for v in camera_up),
            round(float(fov_y), 4),
        )

    def render(self, depsgraph):
        scene = depsgraph.scene
        settings = scene.mathops_v2_settings
        width, height = bridge.get_render_size(scene)

        try:
            runtime.debug_log("F12 render requested")
            rgba, width, height, _scene_path = bridge.render_rgba(scene)
            pixels = bridge.rgba_to_pixels(rgba, width, height).reshape((-1, 4))

            result = self.begin_result(0, 0, width, height)
            result.layers[0].passes["Combined"].rect = pixels
            self.end_result(result)
            self.update_stats(
                "MathOPS-v2", f"Render {runtime.last_render_stats['render_ms']:.2f} ms"
            )
        except Exception as exc:
            bridge.set_last_error(settings, str(exc))
            traceback.print_exc()

            result = self.begin_result(0, 0, width, height)
            result.layers[0].passes["Combined"].rect = np.zeros(
                (width * height, 4), dtype=np.float32
            )
            self.end_result(result)
            self.update_stats("MathOPS-v2", f"Render failed: {exc}")

    def view_update(self, context, depsgraph):
        return None

    def view_draw(self, context, depsgraph):
        scene = depsgraph.scene
        settings = scene.mathops_v2_settings
        region = context.region
        region3d = getattr(context.space_data, "region_3d", None)

        if not settings.viewport_preview:
            return None

        if region3d is None:
            return None

        if not (region3d.is_perspective or region3d.view_perspective == "CAMERA"):
            if not self._logged_ortho_warning:
                runtime.debug_log("Viewport preview needs a perspective or camera view")
                self._logged_ortho_warning = True
            return None

        self._logged_ortho_warning = False

        scene_path = bridge.resolve_scene_path(settings)
        if not scene_path.is_file():
            bridge.set_last_error(settings, f"Scene file not found: {scene_path}")
            return None

        if self._gpu_viewport.draw(context, depsgraph):
            self.update_stats(
                "MathOPS-v2",
                f"Viewport {runtime.last_render_stats['render_ms']:.2f} ms (Blender GPU)",
            )
            return None

        width, height = bridge.get_viewport_render_size(context, settings)
        camera_position, camera_target_value, camera_up, fov_y = bridge.view_camera(
            context, scene
        )
        signature = self._viewport_signature_for(
            context,
            settings,
            scene_path,
            width,
            height,
            camera_position,
            camera_target_value,
            camera_up,
            fov_y,
        )

        if signature != self._viewport_signature or self._viewport_texture is None:
            try:
                runtime.debug_log(
                    f"Viewport rerender: {scene_path.name}, size={width}x{height}, region={region.width}x{region.height}"
                )
                rgba, width, height, _scene_path = bridge.render_rgba(
                    scene,
                    width=width,
                    height=height,
                    camera_position=camera_position,
                    camera_target_value=camera_target_value,
                    camera_up=camera_up,
                    fov_y=fov_y,
                    background_alpha=0.0,
                    background_color=(0.0, 0.0, 0.0),
                    interactive=True,
                )
                self._upload_viewport_texture(rgba, width, height)
                self._viewport_signature = signature
            except Exception as exc:
                bridge.set_last_error(settings, str(exc))
                traceback.print_exc()
                if self._viewport_texture is None:
                    return None

        self._draw_viewport_texture(scene, region)
        self.update_stats(
            "MathOPS-v2", f"Viewport {runtime.last_render_stats['render_ms']:.2f} ms"
        )
        return None


classes = (MathOPSV2RenderEngine,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
