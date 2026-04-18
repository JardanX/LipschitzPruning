from . import matcap


def register():
    return None


def unregister():
    matcap.clear_cache()
    return None
