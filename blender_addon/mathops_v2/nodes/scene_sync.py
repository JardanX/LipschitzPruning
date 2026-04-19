import uuid

import bpy
from bpy.app.handlers import persistent

from .. import runtime
from . import sdf_tree


_sync_running = False


def _proxy_settings(obj):
    return runtime.object_settings(obj)


def _ensure_unique_proxy_ids(scene):
    seen = set()
    changed = False
    for obj in sdf_tree.list_proxy_objects(scene):
        settings = _proxy_settings(obj)
        if settings is None:
            continue
        proxy_id = str(settings.proxy_id or "")
        if not proxy_id or proxy_id in seen:
            settings.proxy_id = uuid.uuid4().hex
            proxy_id = settings.proxy_id
            changed = True
        seen.add(proxy_id)
    return changed


def _proxy_by_id(scene):
    mapping = {}
    for obj in sdf_tree.list_proxy_objects(scene):
        settings = _proxy_settings(obj)
        if settings is None:
            continue
        proxy_id = str(settings.proxy_id or "")
        if proxy_id:
            mapping[proxy_id] = obj
    return mapping


def _sync_proxy_transform_updates(scene, depsgraph):
    tree = sdf_tree.get_scene_tree(scene, create=False)
    if tree is None:
        return False

    proxy_updates = []
    mirror_origin_updates = []
    fallback_required = False
    for update in getattr(depsgraph, "updates", ()):
        id_data = getattr(update, "id", None)
        if isinstance(id_data, bpy.types.Object):
            if runtime.is_sdf_proxy(id_data):
                proxy_updates.append(id_data)
            elif sdf_tree.mirror_origin_referenced(tree, id_data):
                mirror_origin_updates.append(id_data)
            continue
        if id_data is scene:
            continue
        fallback_required = True

    if not proxy_updates and not mirror_origin_updates:
        return False

    changed = False
    for obj in proxy_updates:
        settings = _proxy_settings(obj)
        proxy_id = "" if settings is None else str(settings.proxy_id or "")
        node = sdf_tree.find_initializer_node(tree, obj=obj, proxy_id=proxy_id)
        if node is None:
            return False
        changed = sdf_tree.sync_proxy_to_node(node) or changed
        sdf_tree.sync_node_to_proxy(node, include_transform=False)

    if changed or mirror_origin_updates:
        runtime.mark_scene_transform_dirty(scene)
        runtime.note_interaction()
        runtime.tag_redraw()
    return not fallback_required


def ensure_scene_graph(scene):
    if scene is None:
        return
    tree = sdf_tree.ensure_scene_tree(scene)
    sdf_tree.ensure_graph_output(tree)
    changed = _ensure_unique_proxy_ids(scene)
    changed = sdf_tree.ensure_unique_initializer_ids(tree) or changed

    proxies_by_id = _proxy_by_id(scene)
    for node in sdf_tree.initializer_nodes(tree):
        sdf_tree.ensure_object_node_id(node)
        target = getattr(node, "target", None)
        if runtime.safe_pointer(target) and runtime.is_sdf_proxy(target):
            continue
        if not bool(getattr(node, "use_proxy", False)):
            continue
        proxy = proxies_by_id.get(str(getattr(node, "proxy_id", "") or ""))
        if proxy is None:
            continue
        with sdf_tree.suppress_object_node_updates():
            node.target = proxy
        changed = True

    valid_target_pointers = {runtime.safe_pointer(obj) for obj in sdf_tree.list_proxy_objects(scene)}
    changed = sdf_tree.prune_tree(tree, valid_target_pointers) or changed

    for node in sdf_tree.initializer_nodes(tree):
        target = getattr(node, "target", None)
        if target is None or not runtime.is_sdf_proxy(target):
            continue
        changed = sdf_tree.sync_proxy_to_node(node) or changed
        sdf_tree.sync_node_to_proxy(node, include_transform=False)

    existing_targets = set()
    for node in tree.nodes:
        if getattr(node, "bl_idname", "") != runtime.OBJECT_NODE_IDNAME:
            continue
        target = getattr(node, "target", None)
        if target is None:
            continue
        target_pointer = runtime.safe_pointer(target)
        if target_pointer:
            existing_targets.add(target_pointer)

    added = False
    for obj in sdf_tree.list_proxy_objects(scene):
        obj_ptr = runtime.safe_pointer(obj)
        if obj_ptr in existing_targets:
            continue
        sdf_tree.add_proxy_to_tree(scene, obj)
        existing_targets.add(obj_ptr)
        added = True

    if added or changed:
        runtime.note_interaction()
        runtime.tag_redraw()


def ensure_all_scene_graphs():
    scenes = getattr(bpy.data, "scenes", None)
    if scenes is None:
        return
    for scene in scenes:
        ensure_scene_graph(scene)


def _deferred_ensure_all_scene_graphs():
    ensure_all_scene_graphs()
    return None


@persistent
def _on_load_post(_dummy):
    ensure_all_scene_graphs()


@persistent
def _on_depsgraph_update_post(scene, depsgraph):
    global _sync_running
    if _sync_running:
        return
    _sync_running = True
    try:
        if _sync_proxy_transform_updates(scene, depsgraph):
            return
        ensure_scene_graph(scene)
    finally:
        _sync_running = False


def register():
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)
    if _on_depsgraph_update_post not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update_post)
    bpy.app.timers.register(_deferred_ensure_all_scene_graphs, first_interval=0.0)


def unregister():
    if _on_depsgraph_update_post in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update_post)
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
