import uuid
from contextlib import contextmanager

import bpy
from bpy.app.handlers import persistent
from bpy.types import Menu, Operator
from mathutils import Euler, Matrix, Vector

from . import sdf_nodes


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


def _record_from_object(record, obj):
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


def ensure_active_handle(scene, context=None):
    del context
    _remove_handle(scene)
    return None


def set_active_primitive(scene, primitive_id, context=None):
    _set_active_record(scene, primitive_id)
    del context
    return None


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
    source_record = _active_record(scene)
    source_handle = None
    if source_object is not None and _is_handle_object(source_object):
        source_handle = source_object
        source_record, _index = _find_record(
            scene, source_object.mathops_v2_sdf_handle_id
        )
    if source_record is None:
        return None
    record = _new_record(scene, source_record.primitive_type, source=source_record)
    if source_handle is not None:
        _record_from_object(record, source_handle)
    _set_active_record(scene, record.primitive_id)
    del context
    _remove_handle(scene)
    return record, None


def _scene_key(scene):
    return getattr(scene, "name_full", scene.name)


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


def graph_to_proxy_sync_enabled():
    return _graph_to_proxy_sync_suppressed <= 0


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
        _PROP_COLOR_KEY: tuple(float(component) for component in node.color),
        _PROP_SIZE_KEY: _DEFAULT_SIZE,
        _PROP_RADIUS_KEY: _DEFAULTS[primitive_type][_PROP_RADIUS_KEY],
        _PROP_HEIGHT_KEY: _DEFAULTS[primitive_type][_PROP_HEIGHT_KEY],
        _PROP_BEVEL_KEY: _DEFAULTS[primitive_type][_PROP_BEVEL_KEY],
    }
    if primitive_type == "box":
        values[_PROP_SIZE_KEY] = tuple(float(component) for component in node.size)
        values[_PROP_BEVEL_KEY] = float(node.bevel)
    else:
        values[_PROP_RADIUS_KEY] = float(node.radius)
        if primitive_type in {"cylinder", "cone"}:
            values[_PROP_HEIGHT_KEY] = float(node.height)
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


def _node_matrix(node):
    return Matrix.LocRotScale(
        Vector(node.sdf_location),
        Euler(node.sdf_rotation, "XYZ"),
        Vector(node.sdf_scale),
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


def _apply_node_to_proxy(node, obj):
    obj.rotation_mode = "XYZ"
    obj.matrix_world = _node_to_blender_matrix(node)
    _write_proxy_properties(obj, _node_properties(node))
    obj.empty_display_type = "PLAIN_AXES"
    obj.empty_display_size = 0.01


def _ensure_proxy_identity(obj):
    obj.mathops_v2_sdf_proxy = True
    if not obj.mathops_v2_sdf_proxy_id:
        obj.mathops_v2_sdf_proxy_id = uuid.uuid4().hex


def _active_collection(context):
    collection = getattr(context, "collection", None)
    if collection is not None:
        return collection
    return context.scene.collection


def _select_only(context, obj):
    for other in context.view_layer.objects:
        other.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj


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


def _store_scene_active(scene, context=None):
    if _has_scene_store(scene):
        return True
    return _migrate_legacy_proxies(scene, context)


def _sync_store_scene(scene, updated_objects=None):
    if not _store_scene_active(scene):
        return False

    for obj in _handle_objects(scene):
        bpy.data.objects.remove(obj, do_unlink=True)
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
    with suppress_graph_to_proxy_sync(), sdf_nodes.deferred_graph_updates(bpy.context):
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
    for key, state in list(_sync_state.items()):
        if key[0] != scene_name or key in current_keys:
            continue
        node = sdf_nodes.find_primitive_node(tree, state.get("node_id", ""))
        if node is not None:
            sdf_nodes.remove_primitive_node(tree, node)
        del _sync_state[key]


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
        tree.nodes.active = node


def _sync_proxy_node_link(scene, tree, obj, props, node_lookup):
    _ensure_proxy_identity(obj)
    if not sdf_nodes.proxy_tree_materialized(tree):
        if not obj.mathops_v2_sdf_node_id:
            obj.mathops_v2_sdf_node_id = obj.mathops_v2_sdf_proxy_id
        return None
    node = node_lookup.get(obj.mathops_v2_sdf_node_id)
    if node is not None and props[_PROP_TYPE_KEY] != (
        sdf_nodes.primitive_type_from_node(node) or ""
    ):
        sdf_nodes.remove_primitive_node(tree, node)
        node_lookup.pop(obj.mathops_v2_sdf_node_id, None)
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


def _sync_proxy_object_from_object(scene, tree, obj, node_lookup):
    props = _read_proxy_properties(obj)
    node = _sync_proxy_node_link(scene, tree, obj, props, node_lookup)
    object_state = _proxy_state(obj, props)
    previous = _sync_state.get(_sync_key(scene, obj))
    baseline = None if previous is None else previous.get("object_snapshot")

    if node is None:
        if previous is None or object_state != baseline:
            sdf_nodes.mark_tree_dirty(tree)
        previous_node = (
            object_state if previous is None else previous.get("node_snapshot")
        )
        _store_dual_sync_state(scene, obj, None, object_state, previous_node)
        return

    if previous is None or object_state != baseline:
        sdf_nodes.mark_tree_dirty(tree)
        node_state = _node_state(node)
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
    with suppress_graph_to_proxy_sync(), sdf_nodes.deferred_graph_updates(bpy.context):
        for obj in all_proxies:
            props = _read_proxy_properties(obj)
            node = _sync_proxy_node_link(scene, tree, obj, props, node_lookup)
            _apply_proxy_to_node(obj, node)
            object_state = _proxy_state(obj, props)
            node_state = _node_state(node)
            _store_dual_sync_state(scene, obj, node, object_state, node_state)


def _sync_proxy_scene(scene, candidate_objects=None):
    scene_name = _scene_key(scene)
    all_proxies = [obj for obj in scene.objects if _is_proxy_object(obj)]
    has_state = any(key[0] == scene_name for key in _sync_state)
    if not all_proxies and not has_state:
        return

    try:
        tree = _ensure_scene_tree(scene, ensure=False)
    except Exception:
        return

    if not sdf_nodes.proxy_tree_materialized(
        tree
    ) and sdf_nodes._tree_open_in_node_editor(tree):
        sdf_nodes.materialize_proxy_tree(scene, tree)

    duplicates_resolved = False
    if all_proxies:
        if candidate_objects is None or len(all_proxies) != _tracked_proxy_count(scene):
            duplicates_resolved = _resolve_duplicate_proxies(scene, tree)
        all_proxies = [obj for obj in scene.objects if _is_proxy_object(obj)]

    node_lookup = _primitive_node_lookup(tree)

    if candidate_objects is None:
        sync_objects = all_proxies
    else:
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

    sync_function = (
        _sync_proxy_object_full
        if candidate_objects is None
        else _sync_proxy_object_from_object
    )
    for obj in sync_objects:
        sync_function(scene, tree, obj, node_lookup)

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

    _sync_in_progress = True
    try:
        with (
            suppress_graph_to_proxy_sync(),
            sdf_nodes.deferred_graph_updates(bpy.context),
        ):
            scene_iterable = bpy.data.scenes if scenes is None else scenes
            for scene in scene_iterable:
                scene_candidates = None
                if candidate_objects_by_scene is not None:
                    scene_candidates = candidate_objects_by_scene.get(
                        _scene_key(scene), []
                    )
                if _sync_store_scene(scene, scene_candidates):
                    continue
                _sync_proxy_scene(scene, scene_candidates)
    finally:
        _sync_in_progress = False


def sync_from_graph(context=None):
    if not graph_to_proxy_sync_enabled():
        return
    scene = None if context is None else getattr(context, "scene", None)
    if scene is not None:
        _sync_proxy_objects((scene,))
        return
    _sync_proxy_objects()


@persistent
def _proxy_sync_load_post(_dummy):
    if _proxy_sync_handlers_suppressed > 0:
        return
    _sync_proxy_objects()


@persistent
def _proxy_sync_depsgraph_post(scene, depsgraph):
    if _proxy_sync_handlers_suppressed > 0:
        return
    updated_objects = []
    for update in depsgraph.updates:
        update_id = getattr(update, "id", None)
        if isinstance(update_id, bpy.types.Object):
            scene_object = scene.objects.get(update_id.name)
            if scene_object is not None and (
                _is_proxy_object(scene_object) or _is_handle_object(scene_object)
            ):
                updated_objects.append(scene_object)
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


def _deferred_proxy_sync():
    _sync_proxy_objects()
    return None


def _deferred_proxy_cleanup():
    if _sync_in_progress:
        return 0.0

    pending_scene_names = list(_pending_cleanup_scenes)
    _pending_cleanup_scenes.clear()
    if not pending_scene_names:
        return None

    with suppress_graph_to_proxy_sync(), sdf_nodes.deferred_graph_updates(bpy.context):
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
                sdf_nodes.remove_primitive_node(tree, node)
    if _pending_cleanup_scenes:
        return 0.0
    return None


def start_proxy_sync():
    _ensure_handler(bpy.app.handlers.load_post, _proxy_sync_load_post)
    _ensure_handler(bpy.app.handlers.depsgraph_update_post, _proxy_sync_depsgraph_post)
    if not bpy.app.timers.is_registered(_deferred_proxy_sync):
        bpy.app.timers.register(_deferred_proxy_sync, first_interval=0.0)
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

    def execute(self, context):
        primitive_type = _sanitize_primitive_type(self.primitive_type)
        with suppress_proxy_sync_handlers():
            tree = _ensure_scene_tree(context.scene, ensure=False)
            _store_scene_active(context.scene, context)
            record = _new_record(context.scene, primitive_type)
            cursor = context.scene.cursor
            record.location = tuple(float(v) for v in cursor.location)
            record.rotation = tuple(float(v) for v in cursor.rotation_euler)
            record.scale = (1.0, 1.0, 1.0)
            _set_active_record(context.scene, record.primitive_id)
            _remove_handle(context.scene)
            sdf_nodes.mark_tree_dirty(tree)
            try:
                from .render import bridge

                bridge.force_redraw_viewports(context)
            except Exception:
                pass
        return {"FINISHED"}


class VIEW3D_MT_mathops_v2_sdf_add(Menu):
    bl_idname = "VIEW3D_MT_mathops_v2_sdf_add"
    bl_label = "SDF Mesh"

    def draw(self, context):
        del context
        layout = self.layout
        layout.operator_context = "INVOKE_REGION_WIN"
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
