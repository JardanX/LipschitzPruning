import uuid

import bpy
from bpy.app.handlers import persistent

from .. import runtime
from . import sdf_tree


_sync_running = False
_node_selection_signature = None
_proxy_selection_signature = None
_scene_proxy_signatures = {}


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


def _scene_proxy_map(scene):
    mapping = {}
    for obj in sdf_tree.list_proxy_objects(scene):
        obj_key = runtime.object_key(obj)
        if obj_key:
            mapping[obj_key] = obj
    return mapping


def _store_scene_proxy_signature(scene, proxy_map=None):
    scene_key = runtime.scene_key(scene)
    if scene_key == 0:
        return
    if proxy_map is None:
        proxy_map = _scene_proxy_map(scene)
    _scene_proxy_signatures[scene_key] = frozenset(proxy_map.keys())


def _sync_scene_proxy_membership(scene):
    if scene is None:
        return False

    proxy_map = _scene_proxy_map(scene)
    scene_key = runtime.scene_key(scene)
    if scene_key == 0:
        return False
    current_signature = frozenset(proxy_map.keys())
    previous_signature = _scene_proxy_signatures.get(scene_key)
    if previous_signature == current_signature:
        return False

    added_keys = current_signature.difference(previous_signature or frozenset())
    removed = previous_signature is not None and bool(previous_signature.difference(current_signature))
    if not added_keys and not removed:
        _store_scene_proxy_signature(scene, proxy_map)
        return False

    tree = sdf_tree.ensure_scene_tree(scene)
    sdf_tree.ensure_graph_output(tree)
    changed = _ensure_unique_proxy_ids(scene)
    changed = sdf_tree.deduplicate_initializer_nodes(tree) or changed
    changed = sdf_tree.ensure_unique_initializer_ids(tree) or changed

    if removed:
        changed = sdf_tree.prune_tree(tree, set(proxy_map.keys())) or changed

    existing_targets = set()
    for node in tree.nodes:
        if getattr(node, "bl_idname", "") != runtime.OBJECT_NODE_IDNAME:
            continue
        target_pointer = runtime.object_key(getattr(node, "target", None))
        if target_pointer:
            existing_targets.add(target_pointer)

    added_proxies = []
    for proxy_key in added_keys:
        obj = proxy_map.get(proxy_key)
        if obj is None or proxy_key in existing_targets:
            continue
        sdf_tree.add_proxy_to_tree(scene, obj)
        existing_targets.add(proxy_key)
        added_proxies.append(obj)
        changed = True

    _select_nodes_for_proxies(tree, added_proxies)

    _store_scene_proxy_signature(scene, proxy_map)
    if changed:
        runtime.note_interaction()
        runtime.tag_redraw()
    return changed


def _should_import_proxy(tree, obj, include_managed):
    if not runtime.is_sdf_proxy(obj):
        return False
    if include_managed:
        return True
    settings = _proxy_settings(obj)
    if settings is None:
        return False
    source_tree_name = str(getattr(settings, "source_tree_name", "") or "")
    source_node_name = str(getattr(settings, "source_node_name", "") or "")
    if source_tree_name or source_node_name:
        return False
    return True


def _view_layer_object(view_layer, obj):
    if obj is None or view_layer is None:
        return None
    try:
        layer_object = view_layer.objects.get(obj.name)
    except Exception:
        layer_object = None
    if runtime.object_key(layer_object) != runtime.object_key(obj):
        return None
    return layer_object


def _set_node_selection_for_proxies(tree, proxies, active_proxy=None):
    global _node_selection_signature

    if tree is None:
        return False, None

    selected_proxy_keys = set()
    for obj in proxies:
        if not runtime.is_sdf_proxy(obj):
            continue
        try:
            if not bool(obj.select_get()):
                continue
        except RuntimeError:
            continue
        proxy_key = runtime.object_key(obj)
        if proxy_key:
            selected_proxy_keys.add(proxy_key)
    if not selected_proxy_keys:
        return False, None

    active_proxy_key = runtime.object_key(active_proxy)

    changed = False
    active_node = None
    for node in sdf_tree.initializer_nodes(tree):
        target_key = runtime.object_key(getattr(node, "target", None))
        should_select = target_key in selected_proxy_keys
        if bool(getattr(node, "select", False)) != should_select:
            node.select = should_select
            changed = True
        if should_select and target_key == active_proxy_key:
            active_node = node
        elif should_select and active_node is None:
            active_node = node
    if active_node is not None and getattr(tree.nodes, "active", None) != active_node:
        tree.nodes.active = active_node
        changed = True
    if changed:
        _node_selection_signature = None
        runtime.tag_redraw()
    return changed, active_node


def _select_nodes_for_proxies(tree, proxies, active_proxy=None):
    changed, _active_node = _set_node_selection_for_proxies(tree, proxies, active_proxy=active_proxy)
    return changed


def _node_editor_space(area):
    for space in area.spaces:
        if space.type == "NODE_EDITOR":
            return space
    return None


def _active_object_nodes_from_editor(context):
    window = getattr(context, "window", None)
    screen = None if window is None else getattr(window, "screen", None)
    if screen is None:
        return None, None

    ordered_areas = list(screen.areas)
    active_area = getattr(context, "area", None)
    if active_area in ordered_areas:
        ordered_areas.remove(active_area)
        ordered_areas.insert(0, active_area)

    for area in ordered_areas:
        if area.type != "NODE_EDITOR":
            continue
        space = _node_editor_space(area)
        if space is None or getattr(space, "tree_type", "") != runtime.TREE_IDNAME:
            continue
        tree = getattr(space, "node_tree", None)
        if tree is None or getattr(tree, "bl_idname", "") != runtime.TREE_IDNAME:
            continue
        selected_nodes = [
            node
            for node in sdf_tree.initializer_nodes(tree)
            if bool(getattr(node, "select", False)) and runtime.is_sdf_proxy(getattr(node, "target", None))
        ]
        if selected_nodes:
            return tree, selected_nodes
        active_node = getattr(tree.nodes, "active", None)
        if getattr(active_node, "bl_idname", "") == runtime.OBJECT_NODE_IDNAME and runtime.is_sdf_proxy(getattr(active_node, "target", None)):
            return tree, [active_node]
    return None, None


def _selected_proxy_targets_from_scene(scene, view_layer):
    if scene is None or view_layer is None:
        return [], None
    selected_targets = []
    seen = set()
    for obj in scene.objects:
        if not runtime.is_sdf_proxy(obj):
            continue
        if _view_layer_object(view_layer, obj) is None:
            continue
        try:
            if not bool(obj.select_get()):
                continue
        except RuntimeError:
            continue
        proxy_key = runtime.object_key(obj)
        if not proxy_key or proxy_key in seen:
            continue
        seen.add(proxy_key)
        selected_targets.append(obj)
    active_target = runtime.object_identity(getattr(view_layer.objects, "active", None))
    if runtime.object_key(active_target) not in seen:
        active_target = selected_targets[0] if selected_targets else None
    return selected_targets, active_target


def _sync_node_editor_selection():
    global _node_selection_signature, _proxy_selection_signature

    context = getattr(bpy, "context", None)
    if context is None:
        return 0.1

    tree, object_nodes = _active_object_nodes_from_editor(context)
    if tree is None or not object_nodes:
        _node_selection_signature = None
        return 0.1

    targets = []
    target_pointers = set()
    for node in object_nodes:
        target = runtime.object_identity(getattr(node, "target", None))
        target_pointer = runtime.object_key(target)
        if not target_pointer or not runtime.is_sdf_proxy(target) or target_pointer in target_pointers:
            continue
        target_pointers.add(target_pointer)
        targets.append(target)
    if not targets:
        _node_selection_signature = None
        return 0.1

    active_node = getattr(tree.nodes, "active", None)
    active_target = runtime.object_identity(getattr(active_node, "target", None)) if active_node in object_nodes else targets[0]
    if not runtime.is_sdf_proxy(active_target):
        active_target = targets[0]

    signature = (
        runtime.safe_pointer(tree),
        tuple(sorted(target_pointers)),
        runtime.object_key(active_target),
    )
    if signature == _node_selection_signature:
        return 0.1
    _node_selection_signature = signature

    scene = getattr(context, "scene", None)
    view_layer = getattr(context, "view_layer", None)
    if scene is None or view_layer is None:
        return 0.1

    for obj in scene.objects:
        if not runtime.is_sdf_proxy(obj):
            continue
        if _view_layer_object(view_layer, obj) is None:
            continue
        should_select = runtime.object_key(obj) in target_pointers
        if bool(obj.select_get()) != should_select:
            try:
                obj.select_set(should_select)
            except RuntimeError:
                continue
    if runtime.is_sdf_proxy(active_target) and _view_layer_object(view_layer, active_target) is not None:
        try:
            view_layer.objects.active = active_target
        except RuntimeError:
            pass
    _proxy_selection_signature = (
        runtime.scene_key(scene),
        tuple(sorted(target_pointers)),
        runtime.object_key(active_target),
    )
    runtime.tag_redraw(context)
    return 0.1


def _sync_proxy_editor_selection():
    global _proxy_selection_signature

    context = getattr(bpy, "context", None)
    if context is None:
        return 0.1

    scene = getattr(context, "scene", None)
    view_layer = getattr(context, "view_layer", None)
    if scene is None or view_layer is None:
        _proxy_selection_signature = None
        return 0.1

    tree = sdf_tree.get_scene_tree(scene, create=False)
    if tree is None:
        _proxy_selection_signature = None
        return 0.1

    selected_targets, active_target = _selected_proxy_targets_from_scene(scene, view_layer)
    if not selected_targets:
        _proxy_selection_signature = None
        return 0.1

    target_keys = tuple(sorted(runtime.object_key(obj) for obj in selected_targets if runtime.object_key(obj)))
    if not target_keys:
        _proxy_selection_signature = None
        return 0.1

    signature = (
        runtime.scene_key(scene),
        target_keys,
        runtime.object_key(active_target),
    )
    if signature == _proxy_selection_signature:
        return 0.1
    _proxy_selection_signature = signature

    changed, active_node = _set_node_selection_for_proxies(tree, selected_targets, active_proxy=active_target)
    if active_node is None:
        return 0.1

    selected_nodes = [
        node
        for node in sdf_tree.initializer_nodes(tree)
        if runtime.object_key(getattr(node, "target", None)) in set(target_keys)
    ]
    sdf_tree.select_nodes_in_editor(context, selected_nodes, active_node=active_node, reveal=True)
    return 0.1


def _sync_proxy_transform_updates(scene, depsgraph):
    tree = sdf_tree.get_scene_tree(scene, create=False)
    if tree is None:
        return False

    proxy_updates = []
    warp_origin_updates = []
    for update in getattr(depsgraph, "updates", ()):
        id_data = getattr(update, "id", None)
        if isinstance(id_data, bpy.types.Object):
            id_data = runtime.object_identity(id_data)
            if runtime.is_sdf_proxy(id_data):
                proxy_updates.append(id_data)
            elif sdf_tree.warp_origin_referenced(tree, id_data):
                warp_origin_updates.append(id_data)

    if not proxy_updates and not warp_origin_updates:
        return False

    changed = False
    pending_proxy_adds = []
    for obj in proxy_updates:
        node = sdf_tree.find_initializer_node(tree, obj=obj)
        if node is None:
            pending_proxy_adds.append(obj)
            continue
        if runtime.object_key(getattr(node, "target", None)) != runtime.object_key(obj):
            with sdf_tree.suppress_object_node_updates():
                node.target = obj
        changed = sdf_tree.sync_proxy_to_node(node) or changed
        sdf_tree.sync_node_to_proxy(node, include_transform=False)

    if pending_proxy_adds:
        _ensure_unique_proxy_ids(scene)
        added_proxies = []
        for obj in pending_proxy_adds:
            node = sdf_tree.find_initializer_node(tree, obj=obj)
            if node is None:
                tree, node = sdf_tree.add_proxy_to_tree(scene, obj)
                added_proxies.append(obj)
                changed = True
        _select_nodes_for_proxies(tree, added_proxies)

    if changed or warp_origin_updates:
        runtime.mark_scene_transform_dirty(scene)
        runtime.note_interaction()
        runtime.tag_redraw()
    return changed or bool(warp_origin_updates)


def ensure_scene_graph(scene, include_managed_proxy_imports=False):
    if scene is None:
        return
    tree = sdf_tree.ensure_scene_tree(scene)
    sdf_tree.ensure_graph_output(tree)
    changed = _ensure_unique_proxy_ids(scene)
    changed = sdf_tree.deduplicate_initializer_nodes(tree) or changed
    changed = sdf_tree.ensure_unique_initializer_ids(tree) or changed

    proxies_by_id = _proxy_by_id(scene)
    for node in sdf_tree.initializer_nodes(tree):
        sdf_tree.ensure_object_node_id(node)
        target = getattr(node, "target", None)
        if runtime.object_key(target) and runtime.is_sdf_proxy(target):
            continue
        if not bool(getattr(node, "use_proxy", False)):
            continue
        proxy = proxies_by_id.get(str(getattr(node, "proxy_id", "") or ""))
        if proxy is None:
            continue
        with sdf_tree.suppress_object_node_updates():
            node.target = proxy
        changed = True

    valid_target_pointers = {runtime.object_key(obj) for obj in sdf_tree.list_proxy_objects(scene)}
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
        target_pointer = runtime.object_key(target)
        if target_pointer:
            existing_targets.add(target_pointer)

    added = False
    for obj in sdf_tree.list_proxy_objects(scene):
        if not _should_import_proxy(tree, obj, include_managed_proxy_imports):
            continue
        obj_ptr = runtime.object_key(obj)
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
        ensure_scene_graph(scene, include_managed_proxy_imports=True)
        _store_scene_proxy_signature(scene)


def _deferred_ensure_all_scene_graphs():
    ensure_all_scene_graphs()
    return None


@persistent
def _on_load_post(_dummy):
    ensure_all_scene_graphs()


@persistent
def _on_depsgraph_update_post(scene, depsgraph):
    global _sync_running
    scene = runtime.scene_identity(scene)
    if _sync_running:
        return
    _sync_running = True
    try:
        _sync_proxy_transform_updates(scene, depsgraph)
        _sync_scene_proxy_membership(scene)
    finally:
        _sync_running = False


def register():
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)
    if _on_depsgraph_update_post not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update_post)
    bpy.app.timers.register(_deferred_ensure_all_scene_graphs, first_interval=0.0)
    bpy.app.timers.register(_sync_proxy_editor_selection, first_interval=0.1)
    bpy.app.timers.register(_sync_node_editor_selection, first_interval=0.1)


def unregister():
    _scene_proxy_signatures.clear()
    if _on_depsgraph_update_post in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update_post)
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    try:
        if bpy.app.timers.is_registered(_sync_proxy_editor_selection):
            bpy.app.timers.unregister(_sync_proxy_editor_selection)
    except Exception:
        pass
    try:
        if bpy.app.timers.is_registered(_sync_node_editor_selection):
            bpy.app.timers.unregister(_sync_node_editor_selection)
    except Exception:
        pass
