import time

import bpy
from bpy.types import Operator

from . import runtime
from .render import bridge


def _demo_anim_tick():
    if not runtime.demo_anim_running:
        runtime.demo_anim_timer_registered = False
        return None
    bridge.force_redraw_viewports()
    return 1.0 / 30.0


class MATHOPS_V2_OT_start_demo_anim(Operator):
    bl_idname = "mathops_v2.start_demo_anim"
    bl_label = "Play Demo Anim"
    bl_description = "Start the viewport-only debug sphere animation"
    bl_options = {"REGISTER"}

    def execute(self, context):
        runtime.demo_anim_running = True
        runtime.demo_anim_start_time = time.perf_counter()
        if not runtime.demo_anim_timer_registered:
            bpy.app.timers.register(_demo_anim_tick, first_interval=0.0)
            runtime.demo_anim_timer_registered = True
        bridge.force_redraw_viewports(context)
        runtime.debug_log("Started demo animation")
        return {"FINISHED"}


class MATHOPS_V2_OT_stop_demo_anim(Operator):
    bl_idname = "mathops_v2.stop_demo_anim"
    bl_label = "Stop Demo Anim"
    bl_description = "Stop the viewport-only debug sphere animation"
    bl_options = {"REGISTER"}

    def execute(self, context):
        runtime.demo_anim_running = False
        bridge.force_redraw_viewports(context)
        runtime.debug_log("Stopped demo animation")
        return {"FINISHED"}


classes = (
    MATHOPS_V2_OT_start_demo_anim,
    MATHOPS_V2_OT_stop_demo_anim,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
