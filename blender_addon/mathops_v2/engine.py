import traceback
from pathlib import Path

import gpu
import numpy as np
import bpy
from bpy.types import RenderEngine
from gpu_extras.batch import batch_for_shader

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
        self._aabb_shader = None

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

    def _ensure_aabb_shader(self):
        if self._aabb_shader is not None:
            return self._aabb_shader
        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.push_constant("MAT4", "viewProjectionMatrix")
        shader_info.push_constant("VEC4", "color")
        shader_info.vertex_in(0, "VEC3", "position")
        shader_info.fragment_out(0, "VEC4", "FragColor")
        shader_info.vertex_source(
            "void main(){  gl_Position = viewProjectionMatrix * vec4(position, 1.0);}"
        )
        shader_info.fragment_source("void main(){  FragColor = color;}")
        self._aabb_shader = gpu.shader.create_from_info(shader_info)
        return self._aabb_shader

    def _draw_aabb_overlay(self, context, settings):
        if not getattr(settings, "show_aabb_overlay", True):
            return
        if runtime.current_effective_aabb is None:
            return
        aabb_min, aabb_max = runtime.current_effective_aabb
        points = bridge.renderer_aabb_edge_points(aabb_min, aabb_max)
        if not points:
            return
        shader = self._ensure_aabb_shader()
        batch = batch_for_shader(shader, "LINES", {"position": points})
        gpu.state.depth_test_set("NONE")
        gpu.state.depth_mask_set(False)
        gpu.state.blend_set("ALPHA")
        shader.uniform_float(
            "viewProjectionMatrix", context.region_data.perspective_matrix
        )
        shader.uniform_float("color", (1.0, 0.35, 0.1, 1.0))
        batch.draw(shader)
        gpu.state.blend_set("NONE")

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
            bridge.set_last_error(str(exc))
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

        scene_cache = bridge.graph_scene_cache(settings, create=True)
        scene_path = (
            Path(scene_cache["path"])
            if scene_cache is not None
            else bridge.resolve_scene_path(settings, create=True)
        )
        if scene_cache is None and not scene_path.is_file():
            if not runtime.last_error_message:
                bridge.set_last_error(f"Scene file not found: {scene_path}")
            return None

        if self._gpu_viewport.draw(context, depsgraph):
            self._draw_aabb_overlay(context, settings)
            self.update_stats("MathOPS-v2", self._viewport_status_text("Viewport"))
            return None

        backend = "UNKNOWN"
        try:
            backend = gpu.platform.backend_type_get()
        except Exception:
            pass
        if not runtime.last_error_message:
            bridge.set_last_error(
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
