from . import scene_sync, sdf_tree


def register():
    sdf_tree.register()
    scene_sync.register()


def unregister():
    scene_sync.unregister()
    sdf_tree.unregister()
