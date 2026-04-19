from . import matcap, overlay


def register():
    overlay.register()
    return None


def unregister():
    overlay.unregister()
    matcap.clear_cache()
    return None
