from . import overlay_patch, panels, polygon_overlay


def register():
    polygon_overlay.register()
    panels.register()
    overlay_patch.register()


def unregister():
    overlay_patch.unregister()
    panels.unregister()
    polygon_overlay.unregister()
