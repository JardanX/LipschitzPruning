bl_info = {
    "name": "MathOPS-v2",
    "author": "OpenCode",
    "version": (0, 3, 0),
    "blender": (4, 2, 0),
    "location": "Render > MathOPS-v2",
    "category": "Render",
    "description": "Custom Blender render engine bridge for the MathOPS-v2 Vulkan renderer",
}

from . import engine, operators, properties, runtime, sdf_nodes, ui
from .render import bridge


modules = (
    sdf_nodes,
    properties,
    operators,
    engine,
    ui,
)


def register():
    runtime.reset_runtime()
    for module in modules:
        if hasattr(module, "register"):
            module.register()
    if hasattr(sdf_nodes, "post_register"):
        sdf_nodes.post_register()
    bridge.register_compat_panels()
    runtime.debug_log("MathOPS-v2 addon registered")


def unregister():
    if hasattr(sdf_nodes, "pre_unregister"):
        sdf_nodes.pre_unregister()
    bridge.unregister_compat_panels()
    bridge.close_renderer()
    for module in reversed(modules):
        if hasattr(module, "unregister"):
            module.unregister()
    runtime.reset_runtime()
