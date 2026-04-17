bl_info = {
    "name": "MathOPS V2",
    "author": "OpenCode",
    "version": (0, 1, 0),
    "blender": (5, 0, 0),
    "location": "Render, Node Editor, Shift+A",
    "category": "Render",
    "description": "Simple SDF raymarcher with proxy empties and a custom SDF node graph",
}

from . import engine, nodes, operators, properties, ui


modules = (
    properties,
    nodes,
    operators,
    engine,
    ui,
)


def register():
    for module in modules:
        if hasattr(module, "register"):
            module.register()


def unregister():
    for module in reversed(modules):
        if hasattr(module, "unregister"):
            module.unregister()
