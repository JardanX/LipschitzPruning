from . import matcap, mesh_renderer, overlay


def register():
    overlay.register()
    return None


def unregister():
    overlay.unregister()
    matcap.clear_cache()
    return None
