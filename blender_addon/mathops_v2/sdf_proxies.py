import uuid
from contextlib import contextmanager
import time

import bpy
from bpy.app.handlers import persistent
from bpy.types import Menu, Operator
from mathutils import Euler, Matrix, Vector

from . import runtime, sdf_nodes


_PROP_TYPE_KEY = "sdf_type"
_PROP_COLOR_KEY = "sdf_color"
_PROP_SIZE_KEY = "sdf_size"
_PROP_RADIUS_KEY = "sdf_radius"
_PROP_HEIGHT_KEY = "sdf_height"
_PROP_BEVEL_KEY = "sdf_bevel"

_PRIMITIVE_ORDER = ("sphere", "box", "cylinder", "cone")
_PRIMITIVE_LABELS = {
    "sphere": "SDF Sphere",
    "box": "SDF Box",
    "cylinder": "SDF Cylinder",
    "cone": "SDF Cone",
}
_PRIMITIVE_ICONS = {
    "sphere": "MESH_UVSPHERE",
    "box": "CUBE",
    "cylinder": "MESH_CYLINDER",
    "cone": "MESH_CONE",
}
_ID_LIST_SEPARATOR = ","
_DEFAULT_COLOR = (0.8, 0.8, 0.8)
_DEFAULT_SIZE = (1.0, 1.0, 1.0)
_DEFAULTS = {
    "sphere": {
        _PROP_RADIUS_KEY: 0.5,
        _PROP_HEIGHT_KEY: 1.0,
        _PROP_SIZE_KEY: _DEFAULT_SIZE,
        _PROP_BEVEL_KEY: 0.0,
    },
    "box": {
        _PROP_RADIUS_KEY: 0.5,
        _PROP_HEIGHT_KEY: 1.0,
        _PROP_SIZE_KEY: _DEFAULT_SIZE,
        _PROP_BEVEL_KEY: 0.0,
    },
    "cylinder": {
        _PROP_RADIUS_KEY: 0.35,
        _PROP_HEIGHT_KEY: 1.0,
        _PROP_SIZE_KEY: _DEFAULT_SIZE,
        _PROP_BEVEL_KEY: 0.0,
    },
    "cone": {
        _PROP_RADIUS_KEY: 0.35,
        _PROP_HEIGHT_KEY: 1.0,
        _PROP_SIZE_KEY: _DEFAULT_SIZE,
        _PROP_BEVEL_KEY: 0.0,
    },
}
_sync_state = {}
_sync_in_progress = False
_graph_to_proxy_sync_suppressed = 0
_pending_cleanup_scenes = set()
_pending_record_node_sync = {}
_proxy_sync_handlers_suppressed = 0
_HANDLE_NAME = "MathOPS-v2 SDF Handle"
_BLENDER_TO_RENDERER_BASIS = Matrix(
    (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
)
_RENDERER_TO_BLENDER_BASIS = _BLENDER_TO_RENDERER_BASIS.inverted()


def _scene_records(scene):
    return getattr(scene, "mathops_v2_primitives", ())


def _has_scene_store(scene):
    return len(_scene_records(scene)) > 0


def _find_record(scene, primitive_id):
    for index, record in enumerate(_scene_records(scene)):
        if record.primitive_id == primitive_id:
            return record, index
    return None, -1


def _record_props(record):
    return {
        _PROP_TYPE_KEY: record.primitive_type,
        _PROP_COLOR_KEY: tuple(float(v) for v in record.color),
        _PROP_SIZE_KEY: tuple(float(v) for v in record.size),
        _PROP_RADIUS_KEY: float(record.radius),
        _PROP_HEIGHT_KEY: float(record.height),
        _PROP_BEVEL_KEY: float(record.bevel),
    }


def _record_matrix(record):
    return Matrix.LocRotScale(
        Vector(record.location),
        Euler(record.rotation, "XYZ"),
        Vector(record.scale),
    )


def _record_state(record):
    props = _record_props(record)
    return {
        "type": props[_PROP_TYPE_KEY],
        "matrix": _matrix_signature(_record_matrix(record)),
        "color": _signature_vector(props[_PROP_COLOR_KEY]),
        "size": _signature_vector(props[_PROP_SIZE_KEY]),
        "radius": round(float(props[_PROP_RADIUS_KEY]), 6),
        "height": round(float(props[_PROP_HEIGHT_KEY]), 6),
        "bevel": round(float(props[_PROP_BEVEL_KEY]), 6),
    }


def _copy_props_to_record(record, props):
    record.primitive_type = props[_PROP_TYPE_KEY]
    record.color = props[_PROP_COLOR_KEY]
    record.size = props[_PROP_SIZE_KEY]
    record.radius = props[_PROP_RADIUS_KEY]
    record.height = props[_PROP_HEIGHT_KEY]
    record.bevel = props[_PROP_BEVEL_KEY]


def _record_update_guard():
    from . import properties

    return properties.suppress_record_updates()


def _ensure_record(scene, primitive_id, primitive_type=None):
    record, _index = _find_record(scene, primitive_id)
    if record is not None:
        return record
    with _record_update_guard():
        record = scene.mathops_v2_primitives.add()
        record.primitive_id = primitive_id
        if primitive_type is not None:
            _copy_props_to_record(
                record, _proxy_defaults(_sanitize_primitive_type(primitive_type))
            )
            record.location = (0.0, 0.0, 0.0)
            record.rotation = (0.0, 0.0, 0.0)
            record.scale = (1.0, 1.0, 1.0)
    return record


def _record_from_object(record, obj):
    with _record_update_guard():
        props = _read_proxy_properties(obj)
        _copy_props_to_record(record, props)
        loc, rot, scale = obj.matrix_world.decompose()
        record.location = tuple(float(v) for v in loc)
        record.rotation = tuple(float(v) for v in rot.to_euler("XYZ"))
        record.scale = tuple(float(v) for v in scale)


def _record_to_handle(record, obj):
    obj.rotation_mode = "XYZ"
    obj.matrix_world = _record_matrix(record)
    _write_proxy_properties(obj, _record_props(record))
    obj.empty_display_type = "PLAIN_AXES"
    obj.empty_display_size = 0.01


def _new_record(scene, primitive_type, source=None):
    with _record_update_guard():
        record = scene.mathops_v2_primitives.add()
        record.primitive_id = uuid.uuid4().hex
        if source is None:
            props = _proxy_defaults(primitive_type)
            _copy_props_to_record(record, props)
            record.location = (0.0, 0.0, 0.0)
            record.rotation = (0.0, 0.0, 0.0)
            record.scale = (1.0, 1.0, 1.0)
        else:
            _copy_props_to_record(record, _record_props(source))
            record.location = tuple(float(v) for v in source.location)
            record.rotation = tuple(float(v) for v in source.rotation)
            record.scale = tuple(float(v) for v in source.scale)
    return record


def _active_record(scene):
    primitive_id = getattr(scene, "mathops_v2_active_primitive_id", "")
    record, _index = _find_record(scene, primitive_id)
    if record is not None:
        return record
    records = _scene_records(scene)
    if len(records) > 0:
        return records[0]
    return None


def active_record(scene):
    active_obj = getattr(bpy.context, "active_object", None)
    if _is_proxy_object(active_obj):
        active_name = getattr(active_obj, "name", "")
        scene_object = None if scene is None else scene.objects.get(active_name)
        if _is_proxy_object(scene_object):
            record_id = str(
                getattr(scene_object, "mathops_v2_sdf_node_id", "")
                or getattr(scene_object, "mathops_v2_sdf_proxy_id", "")
                or ""
            )
            record, _index = _find_record(scene, record_id)
            if record is not None:
                if getattr(scene, "mathops_v2_active_primitive_id", "") != record_id:
                    scene.mathops_v2_active_primitive_id = record_id
                return record
    return _active_record(scene)


def _set_active_record(scene, primitive_id):
    scene.mathops_v2_active_primitive_id = primitive_id or ""


def _remove_record(scene, primitive_id):
    _record, index = _find_record(scene, primitive_id)
    if index < 0:
        return False
    scene.mathops_v2_primitives.remove(index)
    if scene.mathops_v2_active_primitive_id == primitive_id:
        records = _scene_records(scene)
        scene.mathops_v2_active_primitive_id = (
            records[min(index, len(records) - 1)].primitive_id if len(records) else ""
        )
    return True


def _is_handle_object(obj):
    return bool(obj is not None and obj.type == "EMPTY" and obj.mathops_v2_sdf_handle)


def _canonical_handle(scene):
    handle_name = getattr(scene, "mathops_v2_handle_object_name", "")
    if handle_name:
        obj = scene.objects.get(handle_name)
        if _is_handle_object(obj):
            return obj
    for obj in scene.objects:
        if _is_handle_object(obj):
            return obj
    return None


def _handle_objects(scene):
    return [obj for obj in scene.objects if _is_handle_object(obj)]


def _remove_handle(scene):
    handle = _canonical_handle(scene)
    scene.mathops_v2_handle_object_name = ""
    if handle is not None:
        bpy.data.objects.remove(handle, do_unlink=True)


def _create_handle(scene, context=None):
    obj = bpy.data.objects.new(_HANDLE_NAME, None)
    target_collection = (
        scene.collection if context is None else _active_collection(context)
    )
    target_collection.objects.link(obj)
    obj.mathops_v2_sdf_handle = True
    scene.mathops_v2_handle_object_name = obj.name
    obj.empty_display_type = "PLAIN_AXES"
    obj.empty_display_size = 0.01
    return obj


def _enable_native_transform_gizmo(context, mode):
    space = getattr(context, "space_data", None)
    if space is None or getattr(space, "type", "") != "VIEW_3D":
        return
    try:
        space.show_gizmo = True
    except Exception:
        pass
    try:
        enabled = str(mode or "TRANSLATE").upper()
        if hasattr(space, "show_gizmo_object_translate"):
            space.show_gizmo_object_translate = enabled == "TRANSLATE"
        if hasattr(space, "show_gizmo_object_rotate"):
            space.show_gizmo_object_rotate = enabled == "ROTATE"
        if hasattr(space, "show_gizmo_object_scale"):
            space.show_gizmo_object_scale = enabled == "SCALE"
    except Exception:
        pass


def sync_native_transform_gizmo(context):
    scene = None if context is None else getattr(context, "scene", None)
    if scene is None:
        return
    try:
        from . import viewport_interaction

        mode = viewport_interaction._transform_mode(scene)
    except Exception:
        mode = "TRANSLATE"
    _enable_native_transform_gizmo(context, mode)


def _handle_targets(scene):
    try:
        from . import viewport_interaction
    except Exception:
        return None, None, None, None
    settings = getattr(scene, "mathops_v2_settings", None)
    if settings is None:
        return None, None, None, None
    transform_mode = viewport_interaction._transform_mode(scene)
    if transform_mode not in {"TRANSLATE", "ROTATE", "SCALE"}:
        return None, None, None, None
    targets = viewport_interaction._selected_targets(scene)
    if not targets:
        return None, None, None, None
    rotation_matrix, _origin, _axes, _up_hints = viewport_interaction._target_axes(
        scene
    )
    center = viewport_interaction._selection_center(scene, targets)
    return viewport_interaction, targets, center, rotation_matrix


def _handle_matrix(center, rotation_matrix):
    matrix = rotation_matrix.to_4x4()
    matrix.translation = Vector((float(center.x), float(center.y), float(center.z)))
    return matrix


def _set_handle_transform(handle, center, rotation_matrix):
    matrix = _handle_matrix(center, rotation_matrix)
    if _matrix_signature(handle.matrix_world) == _matrix_signature(matrix):
        return False
    handle.rotation_mode = "XYZ"
    handle.matrix_world = matrix
    return True


def _record_from_matrix(record, matrix):
    with _record_update_guard():
        loc, rot, scale = matrix.decompose()
        record.location = tuple(float(v) for v in loc)
        record.rotation = tuple(float(v) for v in rot.to_euler("XYZ"))
        record.scale = tuple(max(abs(float(v)), 0.001) for v in scale)


def _node_from_blender_matrix(node, matrix):
    renderer_matrix = _blender_to_renderer_matrix(matrix)
    loc, rot, scale = renderer_matrix.decompose()
    node.sdf_location = tuple(float(component) for component in loc)
    node.sdf_rotation = tuple(float(component) for component in rot.to_euler("XYZ"))
    node.sdf_scale = tuple(max(abs(float(component)), 0.001) for component in scale)


def _record_from_node(record, node):
    with _record_update_guard():
        _copy_props_to_record(record, _node_properties(node))
        loc, rot, scale = _node_to_blender_matrix(node).decompose()
        record.location = tuple(float(v) for v in loc)
        record.rotation = tuple(float(v) for v in rot.to_euler("XYZ"))
        record.scale = tuple(max(abs(float(v)), 0.001) for v in scale)


def _target_matrix(kind, target):
    if kind == "record":
        return _record_matrix(target)
    if kind == "node":
        return _node_to_blender_matrix(target)
    return Matrix.Identity(4)


def _set_target_matrix(kind, target, matrix):
    if kind == "record":
        _record_from_matrix(target, matrix)
    elif kind == "node":
        _node_from_blender_matrix(target, matrix)


def _mark_selection_dirty(scene):
    settings = getattr(scene, "mathops_v2_settings", None)
    if settings is None:
        return
    try:
        tree = sdf_nodes.get_selected_tree(settings, create=False, ensure=False)
    except Exception:
        tree = None
    if tree is not None:
        sdf_nodes.mark_tree_transform_dirty(tree)


def _sync_selected_targets_from_handle(scene, handle):
    viewport_interaction, targets, center, rotation_matrix = _handle_targets(scene)
    if viewport_interaction is None or handle is None or center is None or not targets:
        return False
    base_matrix = _handle_matrix(center, rotation_matrix)
    handle_matrix = handle.matrix_world.copy()
    if _matrix_signature(base_matrix) == _matrix_signature(handle_matrix):
        return False
    delta_matrix = handle_matrix @ base_matrix.inverted_safe()
    for kind, target in targets:
        _set_target_matrix(kind, target, delta_matrix @ _target_matrix(kind, target))
    _mark_selection_dirty(scene)
    return True


def ensure_active_handle(scene, context=None):
    del context
    _remove_handle(scene)
    return None


def set_active_primitive(scene, primitive_id, context=None, extend=False):
    _set_active_record(scene, primitive_id)
    _remove_handle(scene)
    if context is None or getattr(context, "scene", None) != scene:
        return None
    obj = find_proxy_object(scene, primitive_id)
    if obj is None:
        _sync_store_scene(scene)
        obj = find_proxy_object(scene, primitive_id)
    if obj is None:
        sync_native_transform_gizmo(context)
        return None
    if not extend:
        _select_only(context, obj)
    else:
        obj.select_set(True)
        context.view_layer.objects.active = obj
    sync_native_transform_gizmo(context)
    return obj


def _migrate_legacy_proxies(scene, context=None):
    proxies = [obj for obj in scene.objects if _is_proxy_object(obj)]
    if not proxies or _has_scene_store(scene):
        return False
    active_obj = None
    if context is not None:
        active_obj = getattr(getattr(context, "view_layer", None), "objects", None)
        active_obj = None if active_obj is None else active_obj.active
    for obj in proxies:
        record = _new_record(
            scene, _sanitize_primitive_type(obj.get(_PROP_TYPE_KEY, "sphere"))
        )
        _record_from_object(record, obj)
        if scene.mathops_v2_active_primitive_id == "" or obj == active_obj:
            scene.mathops_v2_active_primitive_id = record.primitive_id
    for obj in proxies:
        bpy.data.objects.remove(obj, do_unlink=True)
    _remove_handle(scene)
    return True


def duplicate_active_primitive(scene, source_object=None, context=None):
    del source_object
    duplicated_ids = duplicate_primitives(
        scene, selected_primitive_ids(scene, context), context
    )
    if not duplicated_ids:
        return None
    record, _index = _find_record(scene, duplicated_ids[-1])
    return record


def _scene_key(scene):
    return getattr(scene, "name_full", scene.name)


def encode_primitive_ids(primitive_ids):
    return _ID_LIST_SEPARATOR.join(
        str(primitive_id) for primitive_id in primitive_ids if str(primitive_id or "")
    )


def decode_primitive_ids(encoded_ids):
    if not encoded_ids:
        return []
    seen = set()
    decoded = []
    for primitive_id in str(encoded_ids).split(_ID_LIST_SEPARATOR):
        primitive_id = str(primitive_id or "").strip()
        if not primitive_id or primitive_id in seen:
            continue
        seen.add(primitive_id)
        decoded.append(primitive_id)
    return decoded


def _sync_key(scene, obj):
    return (_scene_key(scene), obj.mathops_v2_sdf_proxy_id)


def _is_proxy_object(obj):
    return bool(obj is not None and obj.type == "EMPTY" and obj.mathops_v2_sdf_proxy)


def find_proxy_object(scene, node_id):
    if not node_id:
        return None
    for obj in getattr(scene, "objects", []):
        if _is_proxy_object(obj) and obj.mathops_v2_sdf_node_id == node_id:
            return obj
    return None


def active_primitive_id(scene, context=None):
    ctx = context if context is not None else bpy.context
    if getattr(ctx, "scene", None) == scene:
        active_object = getattr(ctx, "active_object", None)
        if _is_proxy_object(active_object):
            proxy_id = _proxy_record_id(active_object)
            if proxy_id:
                return proxy_id
    node = sdf_nodes.active_primitive_node(scene, create=False)
    if node is not None:
        return sdf_nodes.primitive_node_token(node)
    return str(getattr(scene, "mathops_v2_active_primitive_id", "") or "")


def selected_primitive_ids(scene, context=None):
    ctx = context if context is not None else bpy.context
    selected_ids = []
    seen = set()
    if getattr(ctx, "scene", None) == scene:
        selected_objects = list(getattr(ctx, "selected_objects", ()) or ())
        for obj in selected_objects:
            if not _is_proxy_object(obj):
                continue
            proxy_id = _proxy_record_id(obj)
            if not proxy_id or proxy_id in seen:
                continue
            seen.add(proxy_id)
            selected_ids.append(proxy_id)
        if selected_ids:
            active_id = active_primitive_id(scene, ctx)
            if active_id in selected_ids:
                selected_ids = [
                    primitive_id
                    for primitive_id in selected_ids
                    if primitive_id != active_id
                ] + [active_id]
            return selected_ids

    try:
        tree = _ensure_scene_tree(scene, ensure=False)
    except Exception:
        tree = None
    if tree is not None:
        for node in tree.nodes:
            if not getattr(node, "select", False) or not sdf_nodes.is_primitive_node(
                node
            ):
                continue
            node_id = sdf_nodes.primitive_node_token(node)
            if node_id in seen:
                continue
            seen.add(node_id)
            selected_ids.append(node_id)
        if selected_ids:
            active_id = active_primitive_id(scene, ctx)
            if active_id in selected_ids:
                selected_ids = [
                    primitive_id
                    for primitive_id in selected_ids
                    if primitive_id != active_id
                ] + [active_id]
            return selected_ids

    active_id = active_primitive_id(scene, ctx)
    return [active_id] if active_id else []


def select_primitives(scene, primitive_ids, context=None, active_id=None):
    ids = decode_primitive_ids(encode_primitive_ids(primitive_ids))
    if not ids:
        return False

    active_id = str(active_id or ids[-1] or "")
    try:
        tree = _ensure_scene_tree(scene, ensure=False)
    except Exception:
        tree = None

    if tree is not None:
        for tree_node in tree.nodes:
            tree_node.select = False
        active_node = None
        for primitive_id in ids:
            node = sdf_nodes.find_primitive_node(tree, primitive_id)
            if node is None:
                continue
            node.select = True
            if primitive_id == active_id or active_node is None:
                active_node = node
        if active_node is not None:
            tree.nodes.active = active_node

    ctx = context if context is not None else bpy.context
    if getattr(ctx, "scene", None) == scene:
        active_object = None
        proxy_objects = {}
        for primitive_id in ids:
            obj = find_proxy_object(scene, primitive_id)
            if obj is not None:
                proxy_objects[primitive_id] = obj
        if proxy_objects:
            view_layer = getattr(ctx, "view_layer", None)
            objects = (
                None if view_layer is None else getattr(view_layer, "objects", None)
            )
            if objects is not None:
                with suppress_proxy_sync_handlers():
                    for other in objects:
                        if other is None:
                            continue
                        try:
                            other.select_set(False)
                        except Exception:
                            continue
                    for primitive_id in ids:
                        obj = proxy_objects.get(primitive_id)
                        if obj is None:
                            continue
                        obj.select_set(True)
                        if primitive_id == active_id or active_object is None:
                            active_object = obj
                    if active_object is not None:
                        objects.active = active_object
        sync_native_transform_gizmo(ctx)

    scene.mathops_v2_active_primitive_id = active_id
    return True


def _duplicate_source_node(tree, source_node):
    primitive_type = sdf_nodes.primitive_type_from_node(source_node) or "sphere"
    node = sdf_nodes.create_primitive_node(tree, primitive_type)
    node.proxy_managed = True
    node.sdf_location = tuple(float(v) for v in source_node.sdf_location)
    node.sdf_rotation = tuple(float(v) for v in source_node.sdf_rotation)
    node.sdf_scale = tuple(float(v) for v in source_node.sdf_scale)
    node.color = tuple(float(v) for v in source_node.color)
    if primitive_type == "box":
        node.size = tuple(float(v) for v in source_node.size)
        node.bevel = float(source_node.bevel)
    else:
        node.radius = float(source_node.radius)
        if primitive_type in {"cylinder", "cone"}:
            node.height = float(source_node.height)
    sdf_nodes.insert_primitive_above_output(tree, node)
    return node


def duplicate_primitives(scene, primitive_ids, context=None):
    source_ids = decode_primitive_ids(encode_primitive_ids(primitive_ids))
    if not source_ids:
        return []

    try:
        tree = _ensure_scene_tree(scene, ensure=True)
    except Exception:
        return []

    source_nodes = []
    for primitive_id in source_ids:
        node = sdf_nodes.find_primitive_node(tree, primitive_id)
        if node is not None:
            source_nodes.append(node)
    if not source_nodes:
        return []

    active_source_id = active_primitive_id(scene, context)
    new_ids = []
    active_new_id = ""
    new_nodes = []
    with (
        suppress_graph_to_proxy_sync(),
        sdf_nodes.deferred_graph_updates(bpy.context, flush=False),
    ):
        for tree_node in tree.nodes:
            tree_node.select = False
        for source_node in source_nodes:
            node = _duplicate_source_node(tree, source_node)
            new_id = sdf_nodes.primitive_node_token(node)
            new_ids.append(new_id)
            new_nodes.append(node)
            node.select = True
            if (
                not active_new_id
                or sdf_nodes.primitive_node_token(source_node) == active_source_id
            ):
                active_new_id = new_id
                tree.nodes.active = node

    for node in new_nodes:
        _mirror_node_to_scene(scene, node, context)
    sdf_nodes.mark_tree_dirty(tree)
    _suppress_proxy_sync_for(0.75)
    select_primitives(scene, new_ids, context, active_new_id)
    return new_ids


def remove_primitives(scene, primitive_ids, context=None):
    target_ids = decode_primitive_ids(encode_primitive_ids(primitive_ids))
    if not target_ids:
        return False

    try:
        tree = _ensure_scene_tree(scene, ensure=False)
    except Exception:
        tree = None
    if tree is None:
        removed = False
        for primitive_id in target_ids:
            removed = _remove_record(scene, primitive_id) or removed
        if removed:
            _sync_store_scene(scene)
        return removed

    removed = False
    removed_ids = []
    with (
        suppress_graph_to_proxy_sync(),
        sdf_nodes.deferred_graph_updates(bpy.context, flush=False),
    ):
        for primitive_id in target_ids:
            node = sdf_nodes.find_primitive_node(tree, primitive_id)
            if node is None:
                continue
            removed = sdf_nodes.remove_primitive_node(tree, node) or removed
            if removed:
                removed_ids.append(primitive_id)
    if not removed:
        return False

    for primitive_id in removed_ids:
        _remove_primitive_mirror(scene, primitive_id)
    sdf_nodes.mark_tree_dirty(tree)
    _suppress_proxy_sync_for(0.5)
    if context is not None and getattr(context, "scene", None) == scene:
        sync_native_transform_gizmo(context)
    return True


def graph_to_proxy_sync_enabled():
    return _graph_to_proxy_sync_suppressed <= 0


def sync_primitive_node_update(scene, node, context=None):
    start = time.perf_counter()
    if scene is None or node is None or not sdf_nodes.is_primitive_node(node):
        return False
    node_id = sdf_nodes.primitive_node_token(node)
    record, _index = _find_record(scene, node_id)
    obj = find_proxy_object(scene, node_id)
    if record is None or obj is None:
        return False
    _record_from_node(record, node)

    with suppress_proxy_sync_handlers():
        _apply_node_to_proxy(node, obj)
    _store_dual_sync_state(scene, obj, node, _proxy_state(obj), _node_state(node))
    _sync_active_node(scene, getattr(node, "id_data", None), obj, node)

    tree = getattr(node, "id_data", None)
    if (
        context is not None
        and getattr(context, "scene", None) == scene
        and getattr(getattr(tree, "nodes", None), "active", None) == node
    ):
        _set_active_record(scene, node_id)
    runtime.debug_slow(
        f"Primitive node sync {node_id[:8]}",
        (time.perf_counter() - start) * 1000.0,
    )
    return True


@contextmanager
def suppress_graph_to_proxy_sync():
    global _graph_to_proxy_sync_suppressed
    _graph_to_proxy_sync_suppressed += 1
    try:
        yield
    finally:
        _graph_to_proxy_sync_suppressed -= 1


@contextmanager
def suppress_proxy_sync_handlers():
    global _proxy_sync_handlers_suppressed
    _proxy_sync_handlers_suppressed += 1
    try:
        yield
    finally:
        _proxy_sync_handlers_suppressed -= 1


def _sanitize_primitive_type(value):
    text = str(value or "sphere").strip().lower()
    return text if text in _PRIMITIVE_LABELS else "sphere"


def _coerce_float(value, default, minimum=0.0):
    try:
        return max(minimum, float(value))
    except Exception:
        return float(default)


def _coerce_vector(value, default, size=3, minimum=None, maximum=None):
    values = list(default)
    try:
        seq = list(value)
    except Exception:
        seq = []
    for index in range(min(size, len(seq))):
        try:
            item = float(seq[index])
        except Exception:
            item = values[index]
        if minimum is not None:
            item = max(minimum, item)
        if maximum is not None:
            item = min(maximum, item)
        values[index] = item
    return tuple(values[:size])


def _set_custom_property(obj, key, value, description):
    if isinstance(value, tuple):
        value = list(value)
    current = obj.get(key)
    if hasattr(current, "__iter__") and not isinstance(current, (str, bytes)):
        try:
            current = list(current)
        except Exception:
            pass
    if current != value:
        obj[key] = value
    try:
        obj.id_properties_ui(key).update(description=description)
    except Exception:
        pass


def _proxy_defaults(primitive_type):
    primitive_type = _sanitize_primitive_type(primitive_type)
    values = dict(_DEFAULTS[primitive_type])
    values[_PROP_TYPE_KEY] = primitive_type
    values[_PROP_COLOR_KEY] = _DEFAULT_COLOR
    return values


def _read_proxy_properties(obj):
    primitive_type = _sanitize_primitive_type(obj.get(_PROP_TYPE_KEY, "sphere"))
    defaults = _proxy_defaults(primitive_type)
    return {
        _PROP_TYPE_KEY: primitive_type,
        _PROP_COLOR_KEY: _coerce_vector(
            obj.get(_PROP_COLOR_KEY, defaults[_PROP_COLOR_KEY]),
            defaults[_PROP_COLOR_KEY],
            minimum=0.0,
            maximum=1.0,
        ),
        _PROP_SIZE_KEY: _coerce_vector(
            obj.get(_PROP_SIZE_KEY, defaults[_PROP_SIZE_KEY]),
            defaults[_PROP_SIZE_KEY],
            minimum=0.001,
        ),
        _PROP_RADIUS_KEY: _coerce_float(
            obj.get(_PROP_RADIUS_KEY, defaults[_PROP_RADIUS_KEY]),
            defaults[_PROP_RADIUS_KEY],
        ),
        _PROP_HEIGHT_KEY: _coerce_float(
            obj.get(_PROP_HEIGHT_KEY, defaults[_PROP_HEIGHT_KEY]),
            defaults[_PROP_HEIGHT_KEY],
        ),
        _PROP_BEVEL_KEY: _coerce_float(
            obj.get(_PROP_BEVEL_KEY, defaults[_PROP_BEVEL_KEY]),
            defaults[_PROP_BEVEL_KEY],
        ),
    }


def _write_proxy_properties(obj, props):
    _set_custom_property(
        obj, _PROP_TYPE_KEY, props[_PROP_TYPE_KEY], "MathOPS-v2 primitive type"
    )
    _set_custom_property(
        obj, _PROP_COLOR_KEY, props[_PROP_COLOR_KEY], "MathOPS-v2 primitive color"
    )
    _set_custom_property(
        obj, _PROP_SIZE_KEY, props[_PROP_SIZE_KEY], "MathOPS-v2 box size"
    )
    _set_custom_property(
        obj, _PROP_RADIUS_KEY, props[_PROP_RADIUS_KEY], "MathOPS-v2 primitive radius"
    )
    _set_custom_property(
        obj, _PROP_HEIGHT_KEY, props[_PROP_HEIGHT_KEY], "MathOPS-v2 primitive height"
    )
    _set_custom_property(
        obj, _PROP_BEVEL_KEY, props[_PROP_BEVEL_KEY], "MathOPS-v2 box bevel"
    )


def _proxy_state(obj, props=None):
    if props is None:
        props = _read_proxy_properties(obj)
    return {
        "type": props[_PROP_TYPE_KEY],
        "matrix": _matrix_signature(obj.matrix_world),
        "color": _signature_vector(props[_PROP_COLOR_KEY]),
        "size": _signature_vector(props[_PROP_SIZE_KEY]),
        "radius": round(float(props[_PROP_RADIUS_KEY]), 6),
        "height": round(float(props[_PROP_HEIGHT_KEY]), 6),
        "bevel": round(float(props[_PROP_BEVEL_KEY]), 6),
    }


def _node_properties(node):
    primitive_type = sdf_nodes.primitive_type_from_node(node) or "sphere"
    values = {
        _PROP_TYPE_KEY: primitive_type,
        _PROP_COLOR_KEY: tuple(
            float(component) for component in sdf_nodes.node_effective_color(node)
        ),
        _PROP_SIZE_KEY: _DEFAULT_SIZE,
        _PROP_RADIUS_KEY: _DEFAULTS[primitive_type][_PROP_RADIUS_KEY],
        _PROP_HEIGHT_KEY: _DEFAULTS[primitive_type][_PROP_HEIGHT_KEY],
        _PROP_BEVEL_KEY: _DEFAULTS[primitive_type][_PROP_BEVEL_KEY],
    }
    if primitive_type == "box":
        values[_PROP_SIZE_KEY] = tuple(
            float(component) for component in sdf_nodes.node_effective_size(node)
        )
        values[_PROP_BEVEL_KEY] = float(sdf_nodes.node_effective_bevel(node))
    else:
        values[_PROP_RADIUS_KEY] = float(sdf_nodes.node_effective_radius(node))
        if primitive_type in {"cylinder", "cone"}:
            values[_PROP_HEIGHT_KEY] = float(sdf_nodes.node_effective_height(node))
    return values


def _node_state(node):
    props = _node_properties(node)
    return {
        "type": props[_PROP_TYPE_KEY],
        "matrix": _matrix_signature(_node_to_blender_matrix(node)),
        "color": _signature_vector(props[_PROP_COLOR_KEY]),
        "size": _signature_vector(props[_PROP_SIZE_KEY]),
        "radius": round(float(props[_PROP_RADIUS_KEY]), 6),
        "height": round(float(props[_PROP_HEIGHT_KEY]), 6),
        "bevel": round(float(props[_PROP_BEVEL_KEY]), 6),
    }


def _signature_vector(values):
    return tuple(round(float(value), 6) for value in values)


def _matrix_signature(matrix):
    return tuple(
        round(float(matrix[row][column]), 6) for row in range(4) for column in range(4)
    )


def _state_matches_except_matrix(current, baseline):
    if current is None or baseline is None:
        return False
    return all(
        current.get(key) == baseline.get(key)
        for key in ("type", "color", "size", "radius", "height", "bevel")
    )


def _node_matrix(node):
    transform = sdf_nodes.node_effective_transform(node)
    return Matrix.LocRotScale(
        Vector(transform["location"]),
        Euler(transform["rotation"], "XYZ"),
        Vector(transform["scale"]),
    )


def _blender_to_renderer_matrix(matrix):
    return _BLENDER_TO_RENDERER_BASIS @ matrix @ _RENDERER_TO_BLENDER_BASIS


def _renderer_to_blender_matrix(matrix):
    return _RENDERER_TO_BLENDER_BASIS @ matrix @ _BLENDER_TO_RENDERER_BASIS


def _node_to_blender_matrix(node):
    return _renderer_to_blender_matrix(_node_matrix(node))


def _apply_proxy_to_node(obj, node):
    props = _read_proxy_properties(obj)
    renderer_matrix = _blender_to_renderer_matrix(obj.matrix_world)
    loc, rot, scale = renderer_matrix.decompose()
    node.sdf_location = tuple(float(component) for component in loc)
    node.sdf_rotation = tuple(float(component) for component in rot.to_euler("XYZ"))
    node.sdf_scale = tuple(max(abs(float(component)), 0.001) for component in scale)
    node.color = props[_PROP_COLOR_KEY]
    if props[_PROP_TYPE_KEY] == "box":
        node.size = props[_PROP_SIZE_KEY]
        node.bevel = props[_PROP_BEVEL_KEY]
    else:
        node.radius = props[_PROP_RADIUS_KEY]
        if props[_PROP_TYPE_KEY] in {"cylinder", "cone"}:
            node.height = props[_PROP_HEIGHT_KEY]


def _apply_proxy_transform_to_node(obj, node):
    renderer_matrix = _blender_to_renderer_matrix(obj.matrix_world)
    loc, rot, scale = renderer_matrix.decompose()
    node.sdf_location = tuple(float(component) for component in loc)
    node.sdf_rotation = tuple(float(component) for component in rot.to_euler("XYZ"))
    node.sdf_scale = tuple(max(abs(float(component)), 0.001) for component in scale)


def _apply_node_to_proxy(node, obj):
    obj.rotation_mode = "XYZ"
    obj.matrix_world = _node_to_blender_matrix(node)
    _write_proxy_properties(obj, _node_properties(node))
    obj.empty_display_type = "PLAIN_AXES"
    obj.empty_display_size = 0.01


def sync_tree_to_scene_records(scene, tree, context=None):
    if getattr(tree, "bl_idname", "") != sdf_nodes.TREE_IDNAME:
        return False

    primitive_nodes = list(sdf_nodes.iter_primitive_nodes(tree))
    node_ids = []
    changed = False
    existing = {
        str(getattr(record, "primitive_id", "") or ""): record
        for record in _scene_records(scene)
    }

    for node in primitive_nodes:
        node_id = sdf_nodes.primitive_node_token(node)
        node_ids.append(node_id)
        record = existing.get(node_id)
        if record is None:
            record = scene.mathops_v2_primitives.add()
            record.primitive_id = node_id
            changed = True
        before = _record_state(record)
        _record_from_node(record, node)
        if _record_state(record) != before:
            changed = True

    node_id_set = set(node_ids)
    for index in range(len(scene.mathops_v2_primitives) - 1, -1, -1):
        record = scene.mathops_v2_primitives[index]
        if str(getattr(record, "primitive_id", "") or "") in node_id_set:
            continue
        scene.mathops_v2_primitives.remove(index)
        changed = True

    active_node = getattr(getattr(tree, "nodes", None), "active", None)
    if sdf_nodes.is_primitive_node(active_node):
        active_id = sdf_nodes.primitive_node_token(active_node)
    else:
        current_id = str(getattr(scene, "mathops_v2_active_primitive_id", "") or "")
        active_id = (
            current_id
            if current_id in node_id_set
            else (node_ids[0] if node_ids else "")
        )
    if getattr(scene, "mathops_v2_active_primitive_id", "") != active_id:
        scene.mathops_v2_active_primitive_id = active_id
        changed = True

    _remove_handle(scene)
    if context is not None and getattr(context, "scene", None) == scene:
        sync_native_transform_gizmo(context)
    return changed


def _ensure_proxy_identity(obj):
    if not obj.mathops_v2_sdf_proxy:
        obj.mathops_v2_sdf_proxy = True
    if not obj.mathops_v2_sdf_proxy_id:
        obj.mathops_v2_sdf_proxy_id = uuid.uuid4().hex


def _active_collection(context):
    collection = getattr(context, "collection", None)
    if collection is not None:
        return collection
    return context.scene.collection


def _select_only(context, obj):
    view_layer = getattr(context, "view_layer", None)
    objects = None if view_layer is None else getattr(view_layer, "objects", None)
    if objects is None or obj is None:
        return
    with suppress_proxy_sync_handlers():
        for other in objects:
            if other is None:
                continue
            try:
                other.select_set(False)
            except Exception:
                continue
        obj.select_set(True)
    objects.active = obj


def _create_proxy_object(context, primitive_type):
    obj = bpy.data.objects.new(_PRIMITIVE_LABELS[primitive_type], None)
    _active_collection(context).objects.link(obj)
    obj.empty_display_type = "PLAIN_AXES"
    obj.empty_display_size = 0.01
    obj.rotation_mode = "XYZ"
    cursor = context.scene.cursor
    obj.location = cursor.location.copy()
    obj.rotation_euler = cursor.rotation_euler
    obj.scale = (1.0, 1.0, 1.0)
    _ensure_proxy_identity(obj)
    _write_proxy_properties(obj, _proxy_defaults(primitive_type))
    _select_only(context, obj)
    return obj


def _proxy_record_id(obj):
    return str(
        getattr(obj, "mathops_v2_sdf_node_id", "")
        or getattr(obj, "mathops_v2_sdf_proxy_id", "")
        or ""
    )


def _proxy_target_collection(scene, context=None):
    if context is not None and getattr(context, "scene", None) == scene:
        return _active_collection(context)
    return scene.collection


def _configure_record_proxy(obj, record_id):
    obj.mathops_v2_sdf_proxy = True
    obj.mathops_v2_sdf_proxy_id = record_id
    obj.mathops_v2_sdf_node_id = record_id
    obj.hide_render = True


def _apply_record_to_proxy(record, obj):
    record_id = str(getattr(record, "primitive_id", "") or "")
    _configure_record_proxy(obj, record_id)
    obj.rotation_mode = "XYZ"
    obj.matrix_world = _record_matrix(record)
    _write_proxy_properties(obj, _record_props(record))
    obj.empty_display_type = "PLAIN_AXES"
    obj.empty_display_size = 0.01


def _create_record_proxy(scene, record, context=None):
    primitive_type = _sanitize_primitive_type(
        getattr(record, "primitive_type", "sphere")
    )
    obj = bpy.data.objects.new(_PRIMITIVE_LABELS[primitive_type], None)
    _proxy_target_collection(scene, context).objects.link(obj)
    _apply_record_to_proxy(record, obj)
    return obj


def _mirror_node_to_scene(scene, node, context=None):
    if node is None or not sdf_nodes.is_primitive_node(node):
        return None
    node_id = sdf_nodes.primitive_node_token(node)
    record = _ensure_record(scene, node_id, sdf_nodes.primitive_type_from_node(node))
    _record_from_node(record, node)
    obj = find_proxy_object(scene, node_id)
    with suppress_proxy_sync_handlers():
        if obj is None:
            obj = _create_record_proxy(scene, record, context)
        else:
            _apply_record_to_proxy(record, obj)
    _store_dual_sync_state(scene, obj, node, _proxy_state(obj), _node_state(node))
    return obj


def _remove_primitive_mirror(scene, primitive_id):
    obj = find_proxy_object(scene, primitive_id)
    if obj is not None:
        key = _sync_key(scene, obj)
        with suppress_proxy_sync_handlers():
            bpy.data.objects.remove(obj, do_unlink=True)
        _sync_state.pop(key, None)
    else:
        scene_name = _scene_key(scene)
        for key, state in list(_sync_state.items()):
            if key[0] != scene_name:
                continue
            if str(state.get("node_id", "") or key[1] or "") == str(primitive_id):
                del _sync_state[key]
    return _remove_record(scene, primitive_id)


def _queue_pending_record_node_sync(scene, primitive_id):
    if scene is None or not primitive_id:
        return
    _pending_record_node_sync.setdefault(_scene_key(scene), set()).add(
        str(primitive_id)
    )


def _apply_record_transform_to_node(record, node):
    _node_from_blender_matrix(node, _record_matrix(record))


def _flush_pending_record_node_sync(scene):
    pending_ids = _pending_record_node_sync.pop(_scene_key(scene), set())
    if not pending_ids:
        return False
    try:
        tree = _ensure_scene_tree(scene, ensure=False)
    except Exception:
        return False
    if tree is None:
        return False

    changed = False
    with (
        suppress_graph_to_proxy_sync(),
        sdf_nodes.deferred_graph_updates(bpy.context, flush=False),
    ):
        for primitive_id in pending_ids:
            record, _index = _find_record(scene, primitive_id)
            if record is None:
                continue
            node = sdf_nodes.find_primitive_node(tree, primitive_id)
            if node is None:
                continue
            _apply_record_transform_to_node(record, node)
            changed = True
    if changed:
        sdf_nodes.mark_tree_dirty(tree)
    return changed


def _record_snapshot_from_entry(entry):
    if entry is None:
        return None
    return entry.get("record_snapshot", entry.get("node_snapshot"))


def _store_record_sync_state(scene, obj, record_snapshot, object_snapshot):
    proxy_id = _proxy_record_id(obj)
    _sync_state[_sync_key(scene, obj)] = {
        "node_id": proxy_id,
        "object_name": obj.name,
        "snapshot": object_snapshot,
        "object_snapshot": object_snapshot,
        "record_snapshot": record_snapshot,
        "node_snapshot": record_snapshot,
    }


def _prune_store_sync_state(scene, current_keys):
    scene_name = _scene_key(scene)
    for key in list(_sync_state):
        if key[0] == scene_name and key not in current_keys:
            del _sync_state[key]


def _ensure_store_proxy(scene, record, proxy_lookup, context=None):
    record_id = str(getattr(record, "primitive_id", "") or "")
    obj = proxy_lookup.get(record_id)
    if obj is None:
        obj = _create_record_proxy(scene, record, context)
        proxy_lookup[record_id] = obj
    else:
        _configure_record_proxy(obj, record_id)

    return obj


def _sync_store_record_to_proxy(scene, record, proxy_lookup, context=None):
    obj = _ensure_store_proxy(scene, record, proxy_lookup, context)

    record_state = _record_state(record)
    object_state = _proxy_state(obj)
    if object_state != record_state:
        _apply_record_to_proxy(record, obj)
        object_state = _proxy_state(obj)
    _store_record_sync_state(scene, obj, record_state, object_state)
    return obj


def _sync_store_proxy_full(scene, obj, record, node_lookup=None):
    key = _sync_key(scene, obj)
    previous = _sync_state.get(key)
    record_state = _record_state(record)
    object_state = _proxy_state(obj)

    if previous is None:
        if object_state != record_state:
            _record_from_object(record, obj)
            if node_lookup is not None:
                node = node_lookup.get(str(getattr(record, "primitive_id", "") or ""))
                if node is not None:
                    _apply_proxy_to_node(obj, node)
            record_state = _record_state(record)
            object_state = _proxy_state(obj)
        _store_record_sync_state(scene, obj, record_state, object_state)
        return record_state

    baseline_object = previous.get("object_snapshot")
    baseline_record = _record_snapshot_from_entry(previous)
    object_changed = object_state != baseline_object
    record_changed = record_state != baseline_record

    if object_changed and not record_changed:
        _record_from_object(record, obj)
        if node_lookup is not None:
            node = node_lookup.get(str(getattr(record, "primitive_id", "") or ""))
            if node is not None:
                _apply_proxy_to_node(obj, node)
        record_state = _record_state(record)
        object_state = _proxy_state(obj)
    elif record_changed and not object_changed:
        _apply_record_to_proxy(record, obj)
        object_state = _proxy_state(obj)
        record_state = _record_state(record)
    elif object_changed and record_changed and object_state != record_state:
        _record_from_object(record, obj)
        if node_lookup is not None:
            node = node_lookup.get(str(getattr(record, "primitive_id", "") or ""))
            if node is not None:
                _apply_proxy_to_node(obj, node)
        record_state = _record_state(record)
        object_state = _proxy_state(obj)

    _store_record_sync_state(scene, obj, record_state, object_state)
    return record_state


def _sync_store_proxy_to_record(scene, obj, record, node_lookup=None):
    before = _record_state(record)
    props = _read_proxy_properties(obj)
    object_state = _proxy_state(obj, props)
    previous = _sync_state.get(_sync_key(scene, obj))
    baseline_object = None if previous is None else previous.get("object_snapshot")
    changed = previous is None or object_state != baseline_object
    if changed:
        _record_from_object(record, obj)
        if node_lookup is not None:
            node = node_lookup.get(str(getattr(record, "primitive_id", "") or ""))
            if node is not None:
                _apply_proxy_to_node(obj, node)
    record_state = _record_state(record)
    _store_record_sync_state(scene, obj, record_state, object_state)
    return before != record_state


def _store_scene_active(scene, context=None):
    if _has_scene_store(scene):
        return True
    return _migrate_legacy_proxies(scene, context)


def _sync_store_scene(scene, updated_objects=None):
    if not _store_scene_active(scene):
        return False

    _remove_handle(scene)

    all_proxies = [obj for obj in scene.objects if _is_proxy_object(obj)]
    current_keys = {_sync_key(scene, obj) for obj in all_proxies}

    tree = None
    try:
        tree = _ensure_scene_tree(scene, ensure=False)
    except Exception:
        tree = None

    deleted_targets_removed = _remove_deleted_proxies(scene, tree, current_keys)

    records = list(_scene_records(scene))
    record_lookup = {
        str(getattr(record, "primitive_id", "") or ""): record for record in records
    }
    record_ids = set(record_lookup)

    duplicates_resolved = False
    if updated_objects is None or len(all_proxies) != len(record_ids):
        duplicates_resolved = _resolve_duplicate_proxies(scene, None)
        if duplicates_resolved:
            all_proxies = [obj for obj in scene.objects if _is_proxy_object(obj)]
            current_keys = {_sync_key(scene, obj) for obj in all_proxies}

    proxy_lookup = {}
    stale_proxies = []
    for obj in all_proxies:
        proxy_id = _proxy_record_id(obj)
        if proxy_id in record_lookup and proxy_id not in proxy_lookup:
            proxy_lookup[proxy_id] = obj
            continue
        stale_proxies.append(obj)

    context = bpy.context if getattr(bpy.context, "scene", None) == scene else None
    records_changed = deleted_targets_removed
    needs_full_sync = (
        updated_objects is None
        or duplicates_resolved
        or len(proxy_lookup) != len(record_ids)
    )
    node_lookup = None
    if tree is not None and sdf_nodes.proxy_tree_materialized(tree):
        node_lookup = _primitive_node_lookup(tree)

    current_keys = set()

    if not needs_full_sync and updated_objects is not None:
        for candidate in updated_objects:
            if candidate is None or not _is_proxy_object(candidate):
                continue
            obj = scene.objects.get(candidate.name)
            if obj is None or not _is_proxy_object(obj):
                continue
            proxy_id = _proxy_record_id(obj)
            record = record_lookup.get(proxy_id)
            if record is None:
                needs_full_sync = True
                break
            if getattr(bpy.context, "active_object", None) == obj:
                _set_active_record(scene, proxy_id)
            if _sync_store_proxy_to_record(scene, obj, record, node_lookup):
                records_changed = True
            current_keys.add(_sync_key(scene, obj))

    if needs_full_sync:
        synced_objects = []
        for record in records:
            obj = _ensure_store_proxy(scene, record, proxy_lookup, context)
            before = _record_state(record)
            after = _sync_store_proxy_full(scene, obj, record, node_lookup)
            if after != before:
                records_changed = True
            synced_objects.append(obj)
            current_keys.add(_sync_key(scene, obj))
        valid_names = {obj.name for obj in synced_objects}
        stale_proxies = [
            obj
            for obj in scene.objects
            if _is_proxy_object(obj) and obj.name not in valid_names
        ]

    for obj in stale_proxies:
        bpy.data.objects.remove(obj, do_unlink=True)

    _prune_store_sync_state(scene, current_keys)

    if records_changed:
        _mark_selection_dirty(scene)

    if context is not None:
        sync_native_transform_gizmo(context)
    return True


def _ensure_scene_tree(scene, ensure=False):
    settings = getattr(scene, "mathops_v2_settings", None)
    if settings is None:
        raise RuntimeError("MathOPS-v2 scene settings are unavailable")
    if not settings.use_sdf_nodes:
        settings.use_sdf_nodes = True
    if ensure:
        return sdf_nodes.get_selected_tree(settings, create=True, ensure=True)
    try:
        return sdf_nodes.get_selected_tree(settings, create=False, ensure=False)
    except Exception:
        return sdf_nodes.ensure_scene_tree(scene, settings)


def _create_proxy_node(scene, tree, obj):
    primitive_type = _read_proxy_properties(obj)[_PROP_TYPE_KEY]
    del scene
    with (
        suppress_graph_to_proxy_sync(),
        sdf_nodes.deferred_graph_updates(bpy.context, flush=False),
    ):
        node = sdf_nodes.create_primitive_node(tree, primitive_type)
        node.proxy_managed = True
        obj.mathops_v2_sdf_node_id = sdf_nodes.primitive_node_token(node)
        sdf_nodes.insert_primitive_above_output(tree, node)
        _apply_proxy_to_node(obj, node)
    return node


def _canonical_duplicate(objects, state):
    remembered_name = ""
    if state is not None:
        remembered_name = state.get("object_name", "")
    for obj in objects:
        if obj.name == remembered_name:
            return obj
    return objects[0]


def _resolve_duplicate_proxies(scene, tree):
    del tree
    changed = False
    proxies_by_id = {}
    for obj in scene.objects:
        if not _is_proxy_object(obj):
            continue
        _ensure_proxy_identity(obj)
        proxies_by_id.setdefault(obj.mathops_v2_sdf_proxy_id, []).append(obj)

    for proxy_id, objects in proxies_by_id.items():
        if len(objects) < 2:
            continue
        keep = _canonical_duplicate(
            objects, _sync_state.get((_scene_key(scene), proxy_id))
        )
        for obj in objects:
            if obj == keep:
                continue
            obj.mathops_v2_sdf_proxy_id = uuid.uuid4().hex
            obj.mathops_v2_sdf_node_id = ""
            _sync_state.pop(_sync_key(scene, obj), None)
            changed = True
    return changed


def _tracked_proxy_count(scene):
    scene_name = _scene_key(scene)
    return sum(1 for key in _sync_state if key[0] == scene_name)


def _remove_deleted_proxies(scene, tree, current_keys):
    scene_name = _scene_key(scene)
    removed = False
    for key, state in list(_sync_state.items()):
        if key[0] != scene_name or key in current_keys:
            continue
        primitive_id = str(state.get("node_id", "") or key[1] or "")
        if tree is not None:
            node = sdf_nodes.find_primitive_node(tree, primitive_id)
            if node is not None:
                sdf_nodes.remove_primitive_node(tree, node)
                removed = True
        if primitive_id and _remove_record(scene, primitive_id):
            removed = True
        del _sync_state[key]
    return removed


def _remove_missing_proxy_targets(scene, tree):
    current_proxy_ids = {
        _proxy_record_id(obj)
        for obj in scene.objects
        if _is_proxy_object(obj) and _proxy_record_id(obj)
    }
    removed_ids = set()

    if tree is not None:
        for node in list(sdf_nodes.iter_primitive_nodes(tree)):
            if not getattr(node, "proxy_managed", False):
                continue
            node_id = sdf_nodes.primitive_node_token(node)
            if node_id in current_proxy_ids:
                continue
            removed_ids.add(node_id)

    scene_name = _scene_key(scene)
    for key, state in list(_sync_state.items()):
        if key[0] != scene_name:
            continue
        primitive_id = str(state.get("node_id", "") or key[1] or "")
        if not primitive_id or primitive_id in current_proxy_ids:
            continue
        removed_ids.add(primitive_id)

    if not removed_ids:
        return False

    with (
        suppress_graph_to_proxy_sync(),
        sdf_nodes.deferred_graph_updates(bpy.context, flush=False),
    ):
        if tree is not None:
            for primitive_id in removed_ids:
                node = sdf_nodes.find_primitive_node(tree, primitive_id)
                if node is not None and getattr(node, "proxy_managed", False):
                    sdf_nodes.remove_primitive_node(tree, node)

    for primitive_id in removed_ids:
        _remove_record(scene, primitive_id)

    for key, state in list(_sync_state.items()):
        if key[0] != scene_name:
            continue
        primitive_id = str(state.get("node_id", "") or key[1] or "")
        if primitive_id in removed_ids:
            del _sync_state[key]

    _mark_selection_dirty(scene)
    return True


def _primitive_node_lookup(tree):
    lookup = {}
    for node in sdf_nodes.iter_primitive_nodes(tree):
        lookup[sdf_nodes.primitive_node_token(node)] = node
    return lookup


def _sync_active_node(scene, tree, obj, node):
    if node is None:
        return
    context = bpy.context
    if getattr(context, "scene", None) != scene:
        return
    if getattr(context, "active_object", None) != obj:
        return
    if getattr(tree.nodes, "active", None) != node:
        try:
            tree.nodes.active = node
        except Exception:
            pass


def _sync_proxy_node_link(scene, tree, obj, props, node_lookup):
    _ensure_proxy_identity(obj)
    node_id = str(obj.mathops_v2_sdf_node_id or obj.mathops_v2_sdf_proxy_id or "")
    if not obj.mathops_v2_sdf_node_id and node_id:
        obj.mathops_v2_sdf_node_id = node_id

    node = node_lookup.get(node_id)
    if not sdf_nodes.proxy_tree_materialized(tree):
        return node
    if node is not None and props[_PROP_TYPE_KEY] != (
        sdf_nodes.primitive_type_from_node(node) or ""
    ):
        sdf_nodes.remove_primitive_node(tree, node)
        node_lookup.pop(node_id, None)
        node = None

    if node is None:
        node = _create_proxy_node(scene, tree, obj)
        node_lookup[obj.mathops_v2_sdf_node_id] = node
    else:
        node.proxy_managed = True
    return node


def _store_sync_state(scene, obj, node, snapshot):
    key = _sync_key(scene, obj)
    previous = _sync_state.get(key, {})
    entry = {
        "node_id": sdf_nodes.primitive_node_token(node),
        "object_name": obj.name,
        "snapshot": snapshot,
        "object_snapshot": previous.get("object_snapshot"),
        "node_snapshot": previous.get("node_snapshot"),
    }
    if snapshot is not None:
        entry["object_snapshot"] = snapshot
        entry["node_snapshot"] = snapshot
    _sync_state[key] = entry


def _store_dual_sync_state(scene, obj, node, object_snapshot, node_snapshot):
    node_id = obj.mathops_v2_sdf_node_id or obj.mathops_v2_sdf_proxy_id
    if node is not None:
        node_id = sdf_nodes.primitive_node_token(node)
    _sync_state[_sync_key(scene, obj)] = {
        "node_id": node_id,
        "object_name": obj.name,
        "snapshot": object_snapshot,
        "object_snapshot": object_snapshot,
        "node_snapshot": node_snapshot,
    }


def _sync_record_from_proxy(scene, obj, node=None):
    record_id = str(obj.mathops_v2_sdf_node_id or obj.mathops_v2_sdf_proxy_id or "")
    if not record_id:
        return False
    record, _index = _find_record(scene, record_id)
    if record is None:
        return False
    before = _record_state(record)
    if node is not None:
        _record_from_node(record, node)
    else:
        _record_from_object(record, obj)
    return _record_state(record) != before


def _sync_proxy_object_from_object(scene, tree, obj, node_lookup):
    props = _read_proxy_properties(obj)
    node = _sync_proxy_node_link(scene, tree, obj, props, node_lookup)
    object_state = _proxy_state(obj, props)
    previous = _sync_state.get(_sync_key(scene, obj))
    baseline = None if previous is None else previous.get("object_snapshot")

    if node is None:
        if previous is None or object_state != baseline:
            _sync_record_from_proxy(scene, obj, None)
            sdf_nodes.mark_tree_dirty(tree)
        previous_node = (
            object_state if previous is None else previous.get("node_snapshot")
        )
        _store_dual_sync_state(scene, obj, None, object_state, previous_node)
        return

    if previous is None or object_state != baseline:
        transform_only = _state_matches_except_matrix(object_state, baseline)
        if (
            transform_only
            and _proxy_transform_active(scene)
            and _scene_supports_record_render(scene)
        ):
            _sync_record_from_proxy(scene, obj, None)
            _queue_pending_record_node_sync(scene, _proxy_record_id(obj))
            _mark_selection_dirty(scene)
            sdf_nodes.mark_tree_transform_dirty(tree)
            node_state = (
                previous.get("node_snapshot")
                if previous is not None
                else _node_state(node)
            )
            _store_dual_sync_state(scene, obj, node, object_state, node_state)
            return
        if transform_only:
            _apply_proxy_transform_to_node(obj, node)
        else:
            _apply_proxy_to_node(obj, node)
        _sync_record_from_proxy(scene, obj, node)
        sdf_nodes.mark_tree_dirty(tree)
        node_state = dict(object_state) if transform_only else _node_state(node)
        _store_dual_sync_state(scene, obj, node, object_state, node_state)
    else:
        node_snapshot = previous.get("node_snapshot")
        _store_dual_sync_state(scene, obj, node, baseline, node_snapshot)
    _sync_active_node(scene, tree, obj, node)


def _sync_proxy_object_full(scene, tree, obj, node_lookup):
    props = _read_proxy_properties(obj)
    node = _sync_proxy_node_link(scene, tree, obj, props, node_lookup)

    key = _sync_key(scene, obj)
    previous = _sync_state.get(key)
    object_state = _proxy_state(obj, props)
    if node is None:
        baseline_object = None if previous is None else previous.get("object_snapshot")
        if previous is None or object_state != baseline_object:
            sdf_nodes.mark_tree_dirty(tree)
        node_snapshot = (
            object_state if previous is None else previous.get("node_snapshot")
        )
        _store_dual_sync_state(scene, obj, None, object_state, node_snapshot)
        return

    node_state = _node_state(node)

    if previous is None:
        if object_state != node_state:
            _apply_proxy_to_node(obj, node)
            node_state = _node_state(node)
            object_state = _proxy_state(obj, props)
        _store_dual_sync_state(scene, obj, node, object_state, node_state)
        _sync_active_node(scene, tree, obj, node)
        return

    baseline_object = previous.get("object_snapshot")
    baseline_node = previous.get("node_snapshot")
    object_changed = object_state != baseline_object
    node_changed = node_state != baseline_node

    if object_changed and not node_changed:
        _apply_proxy_to_node(obj, node)
        node_state = _node_state(node)
        object_state = _proxy_state(obj, props)
    elif node_changed and not object_changed:
        _apply_node_to_proxy(node, obj)
        object_state = _proxy_state(obj)
        node_state = _node_state(node)
    elif object_changed and node_changed and object_state != node_state:
        _apply_proxy_to_node(obj, node)
        node_state = _node_state(node)
        object_state = _proxy_state(obj, props)

    _store_dual_sync_state(scene, obj, node, object_state, node_state)
    _sync_active_node(scene, tree, obj, node)


def sync_proxy_nodes_to_tree(scene, tree):
    all_proxies = [obj for obj in scene.objects if _is_proxy_object(obj)]
    if not all_proxies:
        return

    sdf_nodes.materialize_proxy_tree(scene, tree)
    if not sdf_nodes.proxy_tree_materialized(tree):
        return

    node_lookup = _primitive_node_lookup(tree)
    with (
        suppress_graph_to_proxy_sync(),
        sdf_nodes.deferred_graph_updates(bpy.context, flush=False),
    ):
        for obj in all_proxies:
            props = _read_proxy_properties(obj)
            node = _sync_proxy_node_link(scene, tree, obj, props, node_lookup)
            _apply_proxy_to_node(obj, node)
            object_state = _proxy_state(obj, props)
            node_state = _node_state(node)
            _store_dual_sync_state(scene, obj, node, object_state, node_state)


def _sync_proxy_scene(scene, candidate_objects=None):
    scene_name = _scene_key(scene)

    try:
        tree = _ensure_scene_tree(scene, ensure=False)
    except Exception:
        return

    if candidate_objects is not None:
        sync_objects = []
        seen_names = set()
        for candidate in candidate_objects:
            if candidate is None:
                continue
            obj = scene.objects.get(candidate.name)
            if obj is None or not _is_proxy_object(obj) or obj.name in seen_names:
                continue
            seen_names.add(obj.name)
            sync_objects.append(obj)
        node_lookup = _primitive_node_lookup(tree)
        for obj in sync_objects:
            _sync_proxy_object_from_object(scene, tree, obj, node_lookup)
        return

    all_proxies = [obj for obj in scene.objects if _is_proxy_object(obj)]
    has_state = any(key[0] == scene_name for key in _sync_state)
    if not all_proxies and not has_state:
        return

    duplicates_resolved = False
    if all_proxies:
        if len(all_proxies) != _tracked_proxy_count(scene):
            duplicates_resolved = _resolve_duplicate_proxies(scene, tree)
        all_proxies = [obj for obj in scene.objects if _is_proxy_object(obj)]

    node_lookup = _primitive_node_lookup(tree)
    sync_objects = all_proxies
    for obj in sync_objects:
        _sync_proxy_object_full(scene, tree, obj, node_lookup)

    current_keys = {_sync_key(scene, obj) for obj in all_proxies}
    _remove_deleted_proxies(scene, tree, current_keys)
    if duplicates_resolved:
        _pending_cleanup_scenes.add(_scene_key(scene))
        if not bpy.app.timers.is_registered(_deferred_proxy_cleanup):
            bpy.app.timers.register(_deferred_proxy_cleanup, first_interval=0.0)


def _sync_proxy_objects(scenes=None, candidate_objects_by_scene=None):
    global _sync_in_progress
    if _sync_in_progress:
        return

    start = time.perf_counter()
    _sync_in_progress = True
    try:
        with (
            suppress_graph_to_proxy_sync(),
            sdf_nodes.deferred_graph_updates(bpy.context, flush=False),
        ):
            scene_iterable = bpy.data.scenes if scenes is None else scenes
            for scene in scene_iterable:
                scene_candidates = None
                if candidate_objects_by_scene is not None:
                    scene_candidates = candidate_objects_by_scene.get(
                        _scene_key(scene), []
                    )
                settings = getattr(scene, "mathops_v2_settings", None)
                use_node_scene = bool(
                    settings is not None and getattr(settings, "use_sdf_nodes", False)
                )
                if use_node_scene and scene_candidates is not None:
                    _sync_proxy_scene(scene, scene_candidates)
                    continue
                if _sync_store_scene(scene, scene_candidates):
                    continue
                _sync_proxy_scene(scene, scene_candidates)
    finally:
        _sync_in_progress = False
        total_ms = (time.perf_counter() - start) * 1000.0
        scene_count = len(tuple(bpy.data.scenes if scenes is None else scenes))
        candidate_count = 0
        if candidate_objects_by_scene is not None:
            candidate_count = sum(
                len(items) for items in candidate_objects_by_scene.values()
            )
        runtime.debug_slow(
            f"Proxy sync scenes={scene_count} candidates={candidate_count}",
            total_ms,
        )


def sync_from_graph(context=None):
    if not graph_to_proxy_sync_enabled():
        return
    start = time.perf_counter()
    scene = None if context is None else getattr(context, "scene", None)
    if scene is not None:
        try:
            tree = _ensure_scene_tree(scene, ensure=False)
        except Exception:
            return
        records_start = time.perf_counter()
        sync_tree_to_scene_records(scene, tree, context)
        records_ms = (time.perf_counter() - records_start) * 1000.0
        proxies_start = time.perf_counter()
        _sync_proxy_objects((scene,))
        proxies_ms = (time.perf_counter() - proxies_start) * 1000.0
        total_ms = (time.perf_counter() - start) * 1000.0
        runtime.debug_slow(
            f"Graph sync scene={scene.name} records={records_ms:.2f} proxies={proxies_ms:.2f}",
            total_ms,
        )
        return
    for scene in bpy.data.scenes:
        try:
            tree = _ensure_scene_tree(scene, ensure=False)
        except Exception:
            continue
        sync_tree_to_scene_records(scene, tree)
    _sync_proxy_objects()
    runtime.debug_slow("Graph sync all scenes", (time.perf_counter() - start) * 1000.0)


@persistent
def _proxy_sync_load_post(_dummy):
    if _proxy_sync_handlers_suppressed > 0:
        return
    for scene in bpy.data.scenes:
        settings = getattr(scene, "mathops_v2_settings", None)
        if settings is None or not getattr(settings, "use_sdf_nodes", False):
            continue
        if _has_scene_store(scene):
            continue
        try:
            tree = _ensure_scene_tree(scene, ensure=False)
        except Exception:
            continue
        sync_tree_to_scene_records(scene, tree)
        _sync_proxy_objects((scene,))


@persistent
def _proxy_sync_depsgraph_post(scene, depsgraph):
    if _proxy_sync_handlers_suppressed > 0:
        return
    if time.perf_counter() < float(
        getattr(runtime, "proxy_sync_suppressed_until", 0.0)
    ):
        return

    if _proxy_transform_active(scene):
        active = getattr(bpy.context, "active_object", None)
        if active is not None and _is_proxy_object(active) and _has_scene_store(scene):
            record_id = str(
                active.mathops_v2_sdf_node_id or active.mathops_v2_sdf_proxy_id or ""
            )
            if record_id:
                record, _idx = _find_record(scene, record_id)
                if record is not None:
                    before = _record_state(record)
                    _record_from_object(record, active)
                    if _record_state(record) != before:
                        sdf_nodes.mark_tree_transform_dirty(
                            _ensure_scene_tree(scene, ensure=False)
                        )
        return

    if _native_proxy_duplicate_transform_active(scene):
        _schedule_proxy_sync(0.05)
        return
    updated_objects = []
    deleted_proxy = False
    for update in depsgraph.updates:
        update_id = getattr(update, "id", None)
        if isinstance(update_id, bpy.types.Object):
            scene_object = scene.objects.get(update_id.name)
            if scene_object is not None and (
                _is_proxy_object(scene_object) or _is_handle_object(scene_object)
            ):
                updated_objects.append(scene_object)
                continue
            if _is_proxy_object(update_id):
                deleted_proxy = True
    settings = getattr(scene, "mathops_v2_settings", None)
    use_node_scene = bool(
        settings is not None and getattr(settings, "use_sdf_nodes", False)
    )
    tree = None
    missing_proxy_targets = False
    if use_node_scene:
        current_proxy_count = sum(1 for obj in scene.objects if _is_proxy_object(obj))
        tracked_proxy_count = _tracked_proxy_count(scene)
        missing_proxy_targets = tracked_proxy_count > current_proxy_count
    if not updated_objects and not deleted_proxy and not missing_proxy_targets:
        return
    if use_node_scene and (deleted_proxy or missing_proxy_targets):
        if tree is None:
            try:
                tree = _ensure_scene_tree(scene, ensure=False)
            except Exception:
                tree = None
        _remove_missing_proxy_targets(scene, tree)
        if not updated_objects:
            return
    _sync_proxy_objects(
        (scene,),
        {
            _scene_key(scene): updated_objects,
        },
    )


def _ensure_handler(handler_list, callback):
    if callback not in handler_list:
        handler_list.append(callback)


def _remove_handler(handler_list, callback):
    if callback in handler_list:
        handler_list.remove(callback)


def _schedule_proxy_sync(first_interval=0.0):
    if not bpy.app.timers.is_registered(_deferred_proxy_sync):
        bpy.app.timers.register(_deferred_proxy_sync, first_interval=first_interval)


def _suppress_proxy_sync_for(seconds):
    runtime.proxy_sync_suppressed_until = max(
        float(getattr(runtime, "proxy_sync_suppressed_until", 0.0)),
        time.perf_counter() + max(0.0, float(seconds)),
    )


def _scene_has_duplicate_proxy_ids(scene):
    proxy_ids = set()
    for obj in scene.objects:
        if not _is_proxy_object(obj):
            continue
        proxy_id = str(getattr(obj, "mathops_v2_sdf_proxy_id", "") or "")
        if not proxy_id:
            continue
        if proxy_id in proxy_ids:
            return True
        proxy_ids.add(proxy_id)
    return False


def _active_operator_idnames():
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return set()
    idnames = set()
    for operator in getattr(wm, "operators", ()):
        idname = str(getattr(operator, "bl_idname", "") or "")
        if not idname:
            idname = str(
                getattr(getattr(operator, "bl_rna", None), "identifier", "") or ""
            )
        if idname:
            idnames.add(idname)
    return idnames


def _native_proxy_duplicate_transform_active(scene):
    if scene is None:
        return False
    idnames = _active_operator_idnames()
    if not idnames:
        return False
    duplicate_active = any(
        idname.startswith("OBJECT_OT_duplicate") for idname in idnames
    )
    translate_active = "TRANSFORM_OT_translate" in idnames
    if not duplicate_active and not translate_active:
        return False
    return _scene_has_duplicate_proxy_ids(scene)


def _proxy_transform_active(scene):
    if scene is None or getattr(bpy.context, "scene", None) != scene:
        return False
    active_object = getattr(bpy.context, "active_object", None)
    if not _is_proxy_object(active_object):
        return False
    idnames = _active_operator_idnames()
    return any(
        idname
        in {"TRANSFORM_OT_translate", "TRANSFORM_OT_rotate", "TRANSFORM_OT_resize"}
        for idname in idnames
    )


def _scene_supports_record_render(scene):
    try:
        tree = _ensure_scene_tree(scene, ensure=False)
    except Exception:
        return False
    if tree is None:
        return False
    return sdf_nodes._is_pure_proxy_union_tree(
        tree
    ) or sdf_nodes._is_pure_proxy_csg_union_tree(tree)


def _deferred_proxy_sync():
    if time.perf_counter() < float(
        getattr(runtime, "proxy_sync_suppressed_until", 0.0)
    ):
        return 0.05
    for scene in bpy.data.scenes:
        if _native_proxy_duplicate_transform_active(scene):
            return 0.05
        if _proxy_transform_active(scene):
            return 0.05
        _flush_pending_record_node_sync(scene)
    _sync_proxy_objects()
    sync_from_graph()
    return None


def _deferred_proxy_cleanup():
    if _sync_in_progress:
        return 0.0

    pending_scene_names = list(_pending_cleanup_scenes)
    _pending_cleanup_scenes.clear()
    if not pending_scene_names:
        return None

    with (
        suppress_graph_to_proxy_sync(),
        sdf_nodes.deferred_graph_updates(bpy.context, flush=False),
    ):
        for scene_name in pending_scene_names:
            scene = bpy.data.scenes.get(scene_name)
            if scene is None:
                continue
            try:
                tree = _ensure_scene_tree(scene, ensure=False)
            except Exception:
                continue
            referenced_node_ids = {
                obj.mathops_v2_sdf_node_id
                for obj in scene.objects
                if _is_proxy_object(obj) and obj.mathops_v2_sdf_node_id
            }
            orphan_nodes = []
            for node in sdf_nodes.iter_primitive_nodes(tree):
                if not getattr(node, "proxy_managed", False):
                    continue
                if sdf_nodes.primitive_node_token(node) not in referenced_node_ids:
                    orphan_nodes.append(node)
            for node in orphan_nodes:
                _remove_record(scene, sdf_nodes.primitive_node_token(node))
                sdf_nodes.remove_primitive_node(tree, node)
    if _pending_cleanup_scenes:
        return 0.0
    return None


def start_proxy_sync():
    _ensure_handler(bpy.app.handlers.load_post, _proxy_sync_load_post)
    _ensure_handler(bpy.app.handlers.depsgraph_update_post, _proxy_sync_depsgraph_post)
    if not bpy.app.timers.is_registered(_deferred_proxy_cleanup):
        bpy.app.timers.register(_deferred_proxy_cleanup, first_interval=0.0)


def stop_proxy_sync():
    _remove_handler(bpy.app.handlers.depsgraph_update_post, _proxy_sync_depsgraph_post)
    _remove_handler(bpy.app.handlers.load_post, _proxy_sync_load_post)
    if bpy.app.timers.is_registered(_deferred_proxy_sync):
        bpy.app.timers.unregister(_deferred_proxy_sync)
    if bpy.app.timers.is_registered(_deferred_proxy_cleanup):
        bpy.app.timers.unregister(_deferred_proxy_cleanup)
    _pending_cleanup_scenes.clear()
    _sync_state.clear()
    for scene in bpy.data.scenes:
        _remove_handle(scene)


class MATHOPS_V2_OT_add_sdf_proxy(Operator):
    bl_idname = "mathops_v2.add_sdf_proxy"
    bl_label = "Add SDF Primitive"
    bl_description = (
        "Add an SDF primitive proxy empty and union it into the scene graph"
    )
    bl_options = {"REGISTER", "UNDO"}

    primitive_type: bpy.props.EnumProperty(
        name="Primitive",
        items=tuple(
            (
                identifier,
                _PRIMITIVE_LABELS[identifier].replace("SDF ", ""),
                _PRIMITIVE_LABELS[identifier],
                _PRIMITIVE_ICONS[identifier],
                index,
            )
            for index, identifier in enumerate(_PRIMITIVE_ORDER)
        ),
        default="sphere",
    )

    def invoke(self, context, event):
        del event
        return self.execute(context)

    def execute(self, context):
        primitive_type = _sanitize_primitive_type(self.primitive_type)
        node = None
        with (
            suppress_proxy_sync_handlers(),
            suppress_graph_to_proxy_sync(),
            sdf_nodes.deferred_graph_updates(context, flush=False),
        ):
            tree = _ensure_scene_tree(context.scene, ensure=True)
            node = sdf_nodes.create_primitive_node(tree, primitive_type)
            node.proxy_managed = True
            for tree_node in tree.nodes:
                tree_node.select = False
            tree.nodes.active = node
            node.select = True
            cursor = context.scene.cursor
            _node_from_blender_matrix(
                node,
                Matrix.LocRotScale(
                    cursor.location.copy(),
                    cursor.rotation_euler.copy(),
                    Vector((1.0, 1.0, 1.0)),
                ),
            )
            defaults = _proxy_defaults(primitive_type)
            node.color = defaults[_PROP_COLOR_KEY]
            if primitive_type == "box":
                node.size = defaults[_PROP_SIZE_KEY]
                node.bevel = defaults[_PROP_BEVEL_KEY]
            else:
                node.radius = defaults[_PROP_RADIUS_KEY]
                if primitive_type in {"cylinder", "cone"}:
                    node.height = defaults[_PROP_HEIGHT_KEY]
            sdf_nodes.insert_primitive_above_output(tree, node)
            sdf_nodes.mark_tree_dirty(tree)
            try:
                from .render import bridge

                bridge.force_redraw_viewports(context)
            except Exception:
                pass
        if node is not None:
            _mirror_node_to_scene(context.scene, node, context)
        active_id = str(getattr(context.scene, "mathops_v2_active_primitive_id", ""))
        _suppress_proxy_sync_for(0.25)
        set_active_primitive(context.scene, active_id, context)
        return {"FINISHED"}


class VIEW3D_MT_mathops_v2_sdf_add(Menu):
    bl_idname = "VIEW3D_MT_mathops_v2_sdf_add"
    bl_label = "SDF Mesh"

    def draw(self, context):
        del context
        layout = self.layout
        layout.operator_context = "EXEC_REGION_WIN"
        for primitive_type in _PRIMITIVE_ORDER:
            operator = layout.operator(
                MATHOPS_V2_OT_add_sdf_proxy.bl_idname,
                text=_PRIMITIVE_LABELS[primitive_type],
                icon=_PRIMITIVE_ICONS[primitive_type],
            )
            operator.primitive_type = primitive_type


def menu_func_sdf_add(self, context):
    del context
    self.layout.menu(VIEW3D_MT_mathops_v2_sdf_add.bl_idname, icon="MESH_CUBE")
    self.layout.separator()


classes = (
    MATHOPS_V2_OT_add_sdf_proxy,
    VIEW3D_MT_mathops_v2_sdf_add,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_MT_add.prepend(menu_func_sdf_add)
    start_proxy_sync()


def unregister():
    stop_proxy_sync()
    try:
        bpy.types.VIEW3D_MT_add.remove(menu_func_sdf_add)
    except Exception:
        pass
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
