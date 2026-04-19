from . import overlay_patch, panels


def register():
    panels.register()
    overlay_patch.register()


def unregister():
    overlay_patch.unregister()
    panels.unregister()
