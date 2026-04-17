import bpy
from bpy.types import RenderEngine

from .. import runtime
from ..render.gpu_viewport import MathOPSV2GPUViewport


class MathOPSV2RenderEngine(RenderEngine):
    bl_idname = runtime.ENGINE_ID
    bl_label = "MathOPS V2"
    bl_use_preview = False
    bl_use_postprocess = False
    bl_use_gpu_context = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._viewport = MathOPSV2GPUViewport()

    def __del__(self):
        if getattr(self, "_viewport", None) is not None:
            self._viewport.free()

    def render(self, depsgraph):
        scene = depsgraph.scene
        width = max(1, int(scene.render.resolution_x * scene.render.resolution_percentage / 100.0))
        height = max(1, int(scene.render.resolution_y * scene.render.resolution_percentage / 100.0))
        result = self.begin_result(0, 0, width, height)
        result.layers[0].passes["Combined"].rect = [runtime.BACKGROUND_COLOR] * (width * height)
        self.end_result(result)
        self.update_stats("MathOPS V2", "Viewport GPU path only")

    def view_update(self, context, depsgraph):
        del context
        del depsgraph
        return None

    def view_draw(self, context, depsgraph):
        settings = runtime.scene_settings(depsgraph.scene)
        if settings is None or not settings.viewport_preview:
            return None

        region_data = getattr(context, "region_data", None)
        if region_data is None:
            return None
        if not (region_data.is_perspective or region_data.view_perspective == "CAMERA"):
            runtime.set_error("MathOPS viewport preview requires a perspective or camera view")
            self.update_stats("MathOPS V2", "Perspective view required")
            return None

        try:
            compiled = self._viewport.draw(context, depsgraph)
            stats = f"Prims {compiled['primitive_count']} | Ops {compiled['instruction_count']}"
            if compiled.get("pruning_active"):
                stats += f" | Cull {compiled['pruning_cells']} cells {compiled.get('pruning_ms', 0.0):.2f}ms"
            elif compiled.get("pruning_pending"):
                stats += " | Cull pending"
            self.update_stats(
                "MathOPS V2",
                stats,
            )
        except Exception as exc:
            runtime.set_error(str(exc))
            self.update_stats("MathOPS V2", f"Error: {exc}")
        return None


classes = (MathOPSV2RenderEngine,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
