import traceback

import gpu
import numpy as np
import bpy
from bpy.types import RenderEngine

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
        self._logged_ortho_warning = False

    def __del__(self):
        if getattr(self, "_gpu_viewport", None) is not None:
            self._gpu_viewport.free()

    def _viewport_status_text(self, label: str) -> str:
        stats = runtime.last_render_stats
        return (
            f"{label} {stats['frame_ms']:.2f} ms | "
            f"trace+shade {stats['shader_ms']:.2f} | "
            f"cull {stats['culling_ms']:.2f} | "
            f"upload {stats['upload_ms']:.2f}"
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
            self.update_stats("MathOPS-v2", self._viewport_status_text("Viewport"))
            return None

        backend = "UNKNOWN"
        try:
            backend = gpu.platform.backend_type_get()
        except Exception:
            pass
        if not runtime.last_error_message:
            bridge.set_last_error(
                settings,
                (
                    "Viewport exact GPU path is unavailable on "
                    f"{backend}. OpenGL is currently required because Blender's Python GPU API does not expose storage buffer bindings."
                ),
            )
        self.update_stats("MathOPS-v2", "Viewport unavailable")
        return None


classes = (MathOPSV2RenderEngine,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
