import bpy
from math import degrees, radians
from bpy.types import GizmoGroup, Operator
from bpy_extras import view3d_utils
from mathutils import Euler, Matrix, Vector

from . import runtime, sdf_nodes, sdf_proxies
from .render import bridge


_PICK_MAX_STEPS = 256
_PICK_HIT_EPSILON = 5.0e-4
_PICK_MIN_STEP = 1.0e-4
_PICK_AABB_PAD = 5.0e-2
_SCALE_GIZMO_BASE = 0.35
_TRANSLATE_PIXEL_SCALE = 0.01
_ROTATE_PIXEL_SCALE = 0.01
_SCALE_PIXEL_SCALE = 0.01

_AXIS_COLORS = (
    (0.95, 0.35, 0.35),
    (0.45, 0.85, 0.35),
    (0.35, 0.55, 0.95),
)

_addon_keymaps = []


def active_primitive_node(scene):
    return sdf_nodes.active_primitive_node(scene, create=False)


def active_primitive_record(scene):
    return sdf_proxies.active_record(scene)


def _vector2(value):
    if hasattr(value, "x") and hasattr(value, "y"):
        return Vector((float(value.x), float(value.y)))
    return Vector((float(value[0]), float(value[1])))


def _vector3(value):
    if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
        return Vector((float(value.x), float(value.y), float(value.z)))
    return Vector((float(value[0]), float(value[1]), float(value[2])))


def _record_lookup(scene):
    return {
        str(getattr(record, "primitive_id", "") or ""): record
        for record in getattr(scene, "mathops_v2_primitives", ())
    }


def _target_from_node(scene, node, record_lookup=None):
    del scene, record_lookup
    if node is None:
        return None, None
    return "node", node


def _target_key(kind, target):
    if kind == "record":
        return kind, str(getattr(target, "primitive_id", "") or "")
    if kind == "node":
        return kind, sdf_nodes.primitive_node_token(target)
    return kind, ""


def _active_target(scene):
    node = active_primitive_node(scene)
    if node is not None:
        return _target_from_node(scene, node)
    record = active_primitive_record(scene)
    if record is not None:
        return "record", record
    return None, None


def _location_from_target(kind, target):
    if kind == "record":
        return Vector(target.location)
    if kind == "node":
        return Vector(sdf_proxies._node_to_blender_matrix(target).translation)
    return Vector((0.0, 0.0, 0.0))


def _set_location_on_target(kind, target, value):
    if kind == "record":
        target.location = tuple(float(v) for v in value)
    elif kind == "node":
        matrix = sdf_proxies._node_to_blender_matrix(target).copy()
        matrix.translation = _vector3(value)
        sdf_proxies._node_from_blender_matrix(target, matrix)


def _rotation_from_target(kind, target):
    if kind == "record":
        return Vector(target.rotation)
    if kind == "node":
        _location, rotation, _scale = sdf_proxies._node_to_blender_matrix(
            target
        ).decompose()
        return Vector(rotation.to_euler("XYZ"))
    return Vector((0.0, 0.0, 0.0))


def _set_rotation_on_target(kind, target, value):
    if kind == "record":
        target.rotation = tuple(float(v) for v in value)
    elif kind == "node":
        location, _rotation, scale = sdf_proxies._node_to_blender_matrix(
            target
        ).decompose()
        matrix = Matrix.LocRotScale(
            location,
            Euler(tuple(float(v) for v in value), "XYZ"),
            scale,
        )
        sdf_proxies._node_from_blender_matrix(target, matrix)


def _scale_from_target(kind, target):
    if kind == "record":
        return Vector(target.scale)
    if kind == "node":
        _location, _rotation, scale = sdf_proxies._node_to_blender_matrix(
            target
        ).decompose()
        return Vector(scale)
    return Vector((1.0, 1.0, 1.0))


def _set_scale_on_target(kind, target, value):
    clamped = [max(0.001, float(v)) for v in value]
    if kind == "record":
        target.scale = tuple(clamped)
    elif kind == "node":
        location, rotation, _scale = sdf_proxies._node_to_blender_matrix(
            target
        ).decompose()
        matrix = Matrix.LocRotScale(location, rotation, Vector(clamped))
        sdf_proxies._node_from_blender_matrix(target, matrix)


def _selected_targets(scene):
    tree = _selected_tree(scene)
    if tree is not None:
        record_lookup = _record_lookup(scene)
        targets = []
        seen = set()
        for node in tree.nodes:
            if not getattr(node, "select", False) or not sdf_nodes.is_primitive_node(
                node
            ):
                continue
            kind, target = _target_from_node(scene, node, record_lookup)
            key = _target_key(kind, target)
            if key in seen:
                continue
            seen.add(key)
            targets.append((kind, target))
        if targets:
            return targets
    kind, target = _active_target(scene)
    if kind is None:
        return []
    return [(kind, target)]


def _selection_center(scene, targets=None):
    if targets is None:
        targets = _selected_targets(scene)
    if not targets:
        return Vector((0.0, 0.0, 0.0))
    center = Vector((0.0, 0.0, 0.0))
    for kind, target in targets:
        center += _location_from_target(kind, target)
    return center / len(targets)


def _apply_target_locations(scene, updates):
    changed = False
    targets = []
    for kind, target, value in updates:
        _set_location_on_target(kind, target, value)
        targets.append((kind, target))
        changed = True
    if changed:
        _mark_scene_dirty(scene, targets)


def _mark_scene_dirty(scene, targets=None):
    settings = getattr(scene, "mathops_v2_settings", None)
    if settings is None:
        return
    try:
        tree = sdf_nodes.get_selected_tree(settings, create=False, ensure=False)
    except Exception:
        tree = None

    target_list = [] if targets is None else list(targets)
    if not target_list:
        kind, target = _active_target(scene)
        if kind is not None:
            target_list.append((kind, target))

    node_targets = []
    needs_full_sync = False
    seen = set()
    for kind, target in target_list:
        if kind != "node":
            needs_full_sync = True
            continue
        node_id = sdf_nodes.primitive_node_token(target)
        if node_id in seen:
            continue
        seen.add(node_id)
        node_targets.append(target)

    if not needs_full_sync and node_targets:
        for node in node_targets:
            try:
                sdf_proxies.sync_primitive_node_update(scene, node, bpy.context)
            except Exception:
                needs_full_sync = True
                break

    if tree is not None:
        sdf_nodes.mark_tree_dirty(tree)
    if needs_full_sync or not node_targets:
        try:
            sdf_proxies.sync_from_graph(bpy.context)
        except Exception:
            pass
    bridge.force_redraw_viewports()


def _target_location(scene):
    kind, target = _active_target(scene)
    return _location_from_target(kind, target)


def _set_target_location(scene, value):
    kind, target = _active_target(scene)
    if kind is None:
        return
    _apply_target_locations(scene, ((kind, target, value),))


def _target_rotation(scene):
    kind, target = _active_target(scene)
    return _rotation_from_target(kind, target)


def _set_target_rotation(scene, value):
    kind, target = _active_target(scene)
    if kind is None:
        return
    _set_rotation_on_target(kind, target, value)
    _mark_scene_dirty(scene, ((kind, target),))


def _target_scale(scene):
    kind, target = _active_target(scene)
    return _scale_from_target(kind, target)


def _set_target_scale(scene, value):
    kind, target = _active_target(scene)
    if kind is None:
        return
    _set_scale_on_target(kind, target, value)
    _mark_scene_dirty(scene, ((kind, target),))


def _record_axes(record):
    rotation_matrix = Euler(record.rotation, "XYZ").to_matrix()
    origin = Vector(record.location)
    axes = [Vector(rotation_matrix.col[index]) for index in range(3)]
    up_hints = (axes[1], axes[2], axes[1])
    return rotation_matrix, origin, axes, up_hints


def _transform_mode(scene) -> str:
    settings = getattr(scene, "mathops_v2_settings", None)
    if settings is None:
        return "TRANSLATE"
    return getattr(settings, "viewport_transform_mode", "TRANSLATE")


def _selected_tree(scene):
    settings = getattr(scene, "mathops_v2_settings", None)
    if settings is None or not getattr(settings, "use_sdf_nodes", False):
        return None
    try:
        tree = sdf_nodes.get_selected_tree(settings, create=False, ensure=False)
    except Exception:
        return None
    return tree


def _tag_redraws(tree):
    try:
        window_manager = getattr(bpy.context, "window_manager", None)
        if window_manager is None:
            return
        for window in window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
                    continue
                if area.type != "NODE_EDITOR":
                    continue
                space = area.spaces.active
                if getattr(space, "node_tree", None) == tree:
                    area.tag_redraw()
    except Exception:
        pass


def _set_active_node(scene, node, extend=False, frame=False):
    tree = _selected_tree(scene)
    if tree is None or node is None:
        return False
    sdf_nodes._select_tree_node(tree, node, extend=extend)
    sdf_proxies.set_active_primitive(
        scene, sdf_nodes.primitive_node_token(node), bpy.context, extend=extend
    )
    if frame:
        try:
            sdf_nodes.focus_scene_node(
                bpy.context,
                sdf_nodes.primitive_node_token(node),
                create=False,
                extend=extend,
                frame=True,
            )
        except Exception:
            pass
    _tag_redraws(tree)
    bridge.force_redraw_viewports()
    return True


def _viewport_allows_sdf_pick(context):
    area = getattr(context, "area", None)
    space = getattr(context, "space_data", None)
    scene = getattr(context, "scene", None)
    settings = getattr(scene, "mathops_v2_settings", None)
    shading = getattr(space, "shading", None)
    return (
        area is not None
        and area.type == "VIEW_3D"
        and space is not None
        and space.type == "VIEW_3D"
        and scene is not None
        and settings is not None
        and getattr(settings, "use_sdf_nodes", False)
        and getattr(settings, "viewport_preview", False)
        and getattr(shading, "type", "") == "RENDERED"
        and getattr(context, "mode", "OBJECT") == "OBJECT"
        and getattr(getattr(scene, "render", None), "engine", runtime.ENGINE_ID)
        == runtime.ENGINE_ID
    )


def _viewport_allows_sdf_edit(context):
    area = getattr(context, "area", None)
    space = getattr(context, "space_data", None)
    scene = getattr(context, "scene", None)
    settings = getattr(scene, "mathops_v2_settings", None)
    kind, _target = _active_target(scene)
    return (
        area is not None
        and area.type == "VIEW_3D"
        and space is not None
        and space.type == "VIEW_3D"
        and scene is not None
        and settings is not None
        and getattr(settings, "use_sdf_nodes", False)
        and getattr(context, "mode", "OBJECT") == "OBJECT"
        and kind is not None
    )


def _kernel(x, k):
    if k <= 0.0:
        return 0.0
    m = max(0.0, k - x)
    return (m * m) * 0.25 / k


def _sd_round_box_2d(point, half_size, corner_rounding):
    if point.x > 0.0:
        radius = corner_rounding[0] if point.y > 0.0 else corner_rounding[1]
    else:
        radius = corner_rounding[2] if point.y > 0.0 else corner_rounding[3]
    q = Vector((abs(point.x), abs(point.y))) - half_size + Vector((radius, radius))
    outside = Vector((max(q.x, 0.0), max(q.y, 0.0))).length
    return min(max(q.x, q.y), 0.0) + outside - radius


def _sd_extrude(distance_2d, z, half_height, rounding):
    q = Vector((distance_2d + rounding, abs(z) - half_height))
    outside = Vector((max(q.x, 0.0), max(q.y, 0.0))).length
    return min(max(q.x, q.y), 0.0) + outside - rounding


def _sd_cone(position, radius, half_height):
    p = Vector(
        (Vector((position.x, position.z)).length - radius, position.y + half_height)
    )
    edge = Vector((-radius, 2.0 * half_height))
    denom = max(edge.dot(edge), 1.0e-12)
    q = p - edge * max(0.0, min(1.0, p.dot(edge) / denom))
    dist = q.length
    if max(q.x, q.y) > 0.0:
        return dist
    return -min(dist, p.y)


def _eval_primitive(node_payload, point):
    matrix = sdf_nodes._json_matrix_to_world_to_local(node_payload["matrix"])
    local = matrix @ Vector((point.x, point.y, point.z, 1.0))
    local = Vector(local[:3])
    primitive_type = node_payload.get("primitiveType")
    if primitive_type == "sphere":
        return local.length - float(node_payload.get("radius", 0.0))
    if primitive_type == "box":
        sides = node_payload.get("sides", (0.0, 0.0, 0.0))
        half_sides = Vector((float(sides[0]), float(sides[1]), float(sides[2]))) * 0.5
        bevel = node_payload.get("bevel", (0.0, 0.0, 0.0, 0.0))
        corner_rounding = [float(value) * 0.5 for value in bevel]
        distance_2d = _sd_round_box_2d(
            Vector((local.x, local.z)),
            Vector((half_sides.x, half_sides.z)),
            corner_rounding,
        )
        top_rounding = float(node_payload.get("round_x", 0.0))
        bottom_rounding = float(node_payload.get("round_y", 0.0))
        extrude_rounding = top_rounding if local.y > 0.0 else bottom_rounding
        return _sd_extrude(
            distance_2d, local.y, half_sides.y - extrude_rounding, extrude_rounding
        )
    if primitive_type == "cylinder":
        radius = float(node_payload.get("radius", 0.0))
        half_height = float(node_payload.get("height", 0.0)) * 0.5
        delta = Vector((Vector((local.x, local.z)).length, local.y))
        delta = Vector((abs(delta.x), abs(delta.y))) - Vector((radius, half_height))
        return (
            min(max(delta.x, delta.y), 0.0)
            + Vector((max(delta.x, 0.0), max(delta.y, 0.0))).length
        )
    if primitive_type == "cone":
        return _sd_cone(
            local,
            float(node_payload.get("radius", 0.0)),
            float(node_payload.get("height", 0.0)) * 0.5,
        )
    return float("inf")


def _eval_payload(node_payload, point):
    node_type = node_payload.get("nodeType")
    if node_type == "primitive":
        return _eval_primitive(node_payload, point)

    left = _eval_payload(node_payload["leftChild"], point)
    right = _eval_payload(node_payload["rightChild"], point)
    blend_radius = max(0.0, float(node_payload.get("blendRadius", 0.0)))
    blend_mode = node_payload.get("blendMode")
    if blend_mode == "union":
        return min(left, right) - _kernel(abs(left - right), blend_radius)
    if blend_mode == "sub":
        return max(left, -right) - _kernel(abs(left + right), blend_radius)
    if blend_mode == "inter":
        return max(left, right) - _kernel(abs(left - right), blend_radius)
    return float("inf")


def _pick_payload_primitive(node_payload, point):
    node_type = node_payload.get("nodeType")
    if node_type == "primitive":
        return _eval_primitive(node_payload, point), node_payload

    left_distance, left_payload = _pick_payload_primitive(
        node_payload["leftChild"], point
    )
    right_distance, right_payload = _pick_payload_primitive(
        node_payload["rightChild"], point
    )
    blend_mode = node_payload.get("blendMode")
    if blend_mode == "union":
        return (
            (left_distance, left_payload)
            if left_distance <= right_distance
            else (right_distance, right_payload)
        )
    if blend_mode == "sub":
        effective_right = -right_distance
        return (
            (left_distance, left_payload)
            if left_distance >= effective_right
            else (effective_right, right_payload)
        )
    return (
        (left_distance, left_payload)
        if left_distance >= right_distance
        else (right_distance, right_payload)
    )


def _ray_aabb_intersection(ray_origin, ray_direction, aabb_min, aabb_max):
    t_min = -float("inf")
    t_max = float("inf")
    for axis in range(3):
        origin_value = float(ray_origin[axis])
        direction_value = float(ray_direction[axis])
        min_value = float(aabb_min[axis]) - _PICK_AABB_PAD
        max_value = float(aabb_max[axis]) + _PICK_AABB_PAD
        if abs(direction_value) < 1.0e-12:
            if origin_value < min_value or origin_value > max_value:
                return None
            continue
        inv = 1.0 / direction_value
        t0 = (min_value - origin_value) * inv
        t1 = (max_value - origin_value) * inv
        if t0 > t1:
            t0, t1 = t1, t0
        t_min = max(t_min, t0)
        t_max = min(t_max, t1)
        if t_max < max(t_min, 0.0):
            return None
    return max(t_min, 0.0), t_max


def _raymarch_pick(tree, payload, ray_origin, ray_direction):
    hit_range = _ray_aabb_intersection(
        ray_origin,
        ray_direction,
        payload.get("aabb_min", (-1.0, -1.0, -1.0)),
        payload.get("aabb_max", (1.0, 1.0, 1.0)),
    )
    if hit_range is None:
        return None

    t_value, t_limit = hit_range
    for _step in range(_PICK_MAX_STEPS):
        if t_value > t_limit:
            return None
        point = ray_origin + ray_direction * t_value
        distance = _eval_payload(payload, point)
        if abs(distance) <= _PICK_HIT_EPSILON:
            _distance, primitive_payload = _pick_payload_primitive(payload, point)
            return sdf_nodes.find_primitive_node(
                tree, primitive_payload.get("nodeId", "")
            )
        t_value += max(abs(distance), _PICK_MIN_STEP)
    return None


def _raymarch_pick_id(payload, ray_origin, ray_direction):
    hit_range = _ray_aabb_intersection(
        ray_origin,
        ray_direction,
        payload.get("aabb_min", (-1.0, -1.0, -1.0)),
        payload.get("aabb_max", (1.0, 1.0, 1.0)),
    )
    if hit_range is None:
        return None

    t_value, t_limit = hit_range
    for _step in range(_PICK_MAX_STEPS):
        if t_value > t_limit:
            return None
        point = ray_origin + ray_direction * t_value
        distance = _eval_payload(payload, point)
        if abs(distance) <= _PICK_HIT_EPSILON:
            _distance, primitive_payload = _pick_payload_primitive(payload, point)
            return primitive_payload.get("nodeId", "")
        t_value += max(abs(distance), _PICK_MIN_STEP)
    return None


def _viewport_region_coords(context, event):
    region = next(
        (region for region in context.area.regions if region.type == "WINDOW"), None
    )
    if region is None:
        return None, None
    coord = (int(event.mouse_region_x), int(event.mouse_region_y))
    if (
        coord[0] < 0
        or coord[1] < 0
        or coord[0] >= region.width
        or coord[1] >= region.height
    ):
        return region, None
    return region, coord


def _pick_node_in_region(context, region, coord):
    scene = context.scene
    tree = _selected_tree(scene)
    settings = getattr(scene, "mathops_v2_settings", None)
    if tree is None or settings is None:
        return None

    payload = None
    primitive_id = None
    scene_cache = bridge.graph_scene_cache(settings, create=True)
    if scene_cache is not None and not sdf_nodes.proxy_tree_materialized(tree):
        payload = scene_cache.get("payload")
    if payload is None:
        try:
            payload = sdf_nodes.compile_tree_payload(tree)
        except Exception:
            return None

    region3d = getattr(context.space_data, "region_3d", None)
    if region3d is None:
        return None
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, region3d, coord)
    ray_direction = view3d_utils.region_2d_to_vector_3d(region, region3d, coord)
    if ray_direction.length_squared > 0.0:
        ray_direction.normalize()
    ray_origin = Vector(bridge.blender_to_renderer_vec(ray_origin))
    ray_direction = Vector(bridge.blender_to_renderer_vec(ray_direction))
    if ray_direction.length_squared > 0.0:
        ray_direction.normalize()

    if scene_cache is not None and not sdf_nodes.proxy_tree_materialized(tree):
        primitive_id = _raymarch_pick_id(payload, ray_origin, ray_direction)
        if primitive_id:
            try:
                from . import sdf_proxies

                sdf_proxies.set_active_primitive(scene, primitive_id, bpy.context)
            except Exception:
                pass
            return primitive_id
        return None

    return _raymarch_pick(tree, payload, ray_origin, ray_direction)


def _node_rotation_matrix(node):
    location, rotation, _scale = sdf_proxies._node_to_blender_matrix(node).decompose()
    del location
    return rotation.to_matrix()


def _basis_from_axis(origin, axis, up_hint):
    axis = _vector3(axis)
    if axis.length_squared <= 1.0e-12:
        axis = Vector((0.0, 0.0, 1.0))
    else:
        axis.normalize()

    up_hint_vec = _vector3(up_hint)
    up = up_hint_vec - up_hint_vec.project(axis)
    if up.length_squared <= 1.0e-12:
        up = axis.orthogonal()
    up.normalize()
    side = Vector(
        (
            axis.y * up.z - axis.z * up.y,
            axis.z * up.x - axis.x * up.z,
            axis.x * up.y - axis.y * up.x,
        )
    )
    if side.length_squared <= 1.0e-12:
        side = axis.orthogonal()
    side.normalize()

    matrix = Matrix.Identity(4)
    matrix.col[0].xyz = side
    matrix.col[1].xyz = up
    matrix.col[2].xyz = axis
    matrix.col[3].xyz = origin
    return matrix


def _node_axes(node):
    rotation_matrix = _node_rotation_matrix(node)
    origin = Vector(sdf_proxies._node_to_blender_matrix(node).translation)
    axes = [Vector(rotation_matrix.col[index]) for index in range(3)]
    up_hints = (axes[1], axes[2], axes[1])
    return rotation_matrix, origin, axes, up_hints


def _target_axes(scene):
    kind, target = _active_target(scene)
    if kind == "record":
        return _record_axes(target)
    if kind == "node":
        return _node_axes(target)
    return (
        Matrix.Identity(3),
        Vector((0.0, 0.0, 0.0)),
        [
            Vector((1.0, 0.0, 0.0)),
            Vector((0.0, 1.0, 0.0)),
            Vector((0.0, 0.0, 1.0)),
        ],
        [Vector((0.0, 1.0, 0.0))] * 3,
    )


def _rotation_get(axis_index):
    return float(_target_rotation(bpy.context.scene)[axis_index])


def _rotation_set(axis_index, value):
    current = _target_rotation(bpy.context.scene)
    rotation = [float(current.x), float(current.y), float(current.z)]
    rotation[axis_index] = float(value)
    _set_target_rotation(bpy.context.scene, Vector(rotation))


def _scale_get(axis_index):
    return _SCALE_GIZMO_BASE + float(_target_scale(bpy.context.scene)[axis_index]) - 1.0


def _scale_set(axis_index, value):
    current = _target_scale(bpy.context.scene)
    scale = [float(current.x), float(current.y), float(current.z)]
    scale[axis_index] = max(0.001, 1.0 + float(value) - _SCALE_GIZMO_BASE)
    _set_target_scale(bpy.context.scene, Vector(scale))


def _mouse_ray(context, coord):
    region = getattr(context, "region", None)
    region3d = getattr(context.space_data, "region_3d", None)
    if region is None or region3d is None:
        return None, None
    origin = view3d_utils.region_2d_to_origin_3d(region, region3d, coord)
    direction = view3d_utils.region_2d_to_vector_3d(region, region3d, coord)
    if direction.length_squared > 0.0:
        direction.normalize()
    return _vector3(origin), _vector3(direction)


def _intersect_plane(ray_origin, ray_direction, plane_origin, plane_normal):
    denom = plane_normal.dot(ray_direction)
    if abs(denom) <= 1.0e-8:
        return Vector(plane_origin)
    t_value = plane_normal.dot(plane_origin - ray_origin) / denom
    return ray_origin + ray_direction * t_value


def _axis_parameter(axis_origin, axis_direction, ray_origin, ray_direction):
    axis_direction = _vector3(axis_direction)
    ray_direction = _vector3(ray_direction)
    w0 = axis_origin - ray_origin
    a_value = axis_direction.dot(axis_direction)
    b_value = axis_direction.dot(ray_direction)
    c_value = ray_direction.dot(ray_direction)
    d_value = axis_direction.dot(w0)
    e_value = ray_direction.dot(w0)
    denom = a_value * c_value - b_value * b_value
    if abs(denom) <= 1.0e-8:
        return 0.0
    return (b_value * e_value - c_value * d_value) / denom


def _constraint_axis_vector(axis_name):
    return {
        "X": Vector((1.0, 0.0, 0.0)),
        "Y": Vector((0.0, 1.0, 0.0)),
        "Z": Vector((0.0, 0.0, 1.0)),
    }.get(axis_name)


class MATHOPS_V2_OT_transform_sdf(Operator):
    bl_idname = "mathops_v2.transform_sdf"
    bl_label = "Transform SDF"
    bl_options = {"REGISTER", "UNDO", "BLOCKING"}

    mode: bpy.props.EnumProperty(
        items=(
            ("TRANSLATE", "Move", "Move primitive"),
            ("ROTATE", "Rotate", "Rotate primitive"),
            ("SCALE", "Scale", "Scale primitive"),
        ),
        default="TRANSLATE",
    )
    constraint_axis_init: bpy.props.StringProperty(default="", options={"HIDDEN"})
    gizmo_axis_vector: bpy.props.FloatVectorProperty(
        size=3,
        default=(0.0, 0.0, 0.0),
        options={"HIDDEN"},
    )
    delete_on_cancel_ids: bpy.props.StringProperty(default="", options={"HIDDEN"})
    restore_selection_ids: bpy.props.StringProperty(default="", options={"HIDDEN"})
    restore_active_id: bpy.props.StringProperty(default="", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return _viewport_allows_sdf_edit(context)

    def invoke(self, context, event):
        scene = context.scene
        self.start_mouse = _vector2((event.mouse_region_x, event.mouse_region_y))
        self.translate_targets = (
            _selected_targets(scene) if self.mode == "TRANSLATE" else []
        )
        self.translate_start_locations = [
            _location_from_target(kind, target).copy()
            for kind, target in self.translate_targets
        ]
        self.initial_location = (
            _selection_center(scene, self.translate_targets)
            if self.translate_targets
            else _target_location(scene)
        ).copy()
        self.initial_rotation = _target_rotation(scene).copy()
        self.initial_scale = _target_scale(scene).copy()
        self.constraint_axis = str(getattr(self, "constraint_axis_init", "") or "")
        self.numeric_text = ""
        self.current_value = 0.0

        origin, direction = _mouse_ray(context, self.start_mouse)
        self.view_vector = Vector((0.0, 0.0, -1.0)) if direction is None else direction
        self.view_right = Vector((1.0, 0.0, 0.0))
        space = getattr(context, "space_data", None)
        region3d = None if space is None else getattr(space, "region_3d", None)
        if region3d is not None:
            view_inv = region3d.view_matrix.inverted()
            self.view_right = _vector3(view_inv.col[0].xyz)
        self.start_plane_point = self.initial_location.copy()
        if origin is not None and direction is not None:
            self.start_plane_point = _intersect_plane(
                origin, direction, self.initial_location, self.view_vector
            )

        window_manager = getattr(context, "window_manager", None)
        if window_manager is not None:
            window_manager.modal_handler_add(self)
        self._update_status(context)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE"}:
            self._cancel(context)
            self._finish(context)
            return {"CANCELLED"}
        if (
            event.type in {"RET", "NUMPAD_ENTER", "LEFTMOUSE"}
            and event.value == "PRESS"
        ):
            self._finish(context)
            return {"FINISHED"}
        if event.value == "PRESS" and event.type in {"X", "Y", "Z"}:
            self.constraint_axis = event.type
            self._apply(context, event)
            self._update_status(context)
            return {"RUNNING_MODAL"}
        if event.value == "PRESS" and event.type == "BACK_SPACE":
            self.numeric_text = self.numeric_text[:-1]
            self._apply(context, event)
            self._update_status(context)
            return {"RUNNING_MODAL"}
        if (
            event.value == "PRESS"
            and len(getattr(event, "ascii", "")) == 1
            and event.ascii in "0123456789.-"
        ):
            self.numeric_text += event.ascii
            self._apply(context, event)
            self._update_status(context)
            return {"RUNNING_MODAL"}
        if event.type == "MOUSEMOVE":
            self._apply(context, event)
            self._update_status(context)
        return {"RUNNING_MODAL"}

    def _apply(self, context, event):
        scene = context.scene
        axis = _constraint_axis_vector(self.constraint_axis)
        if self.mode == "TRANSLATE":
            gizmo_axis = _vector3(self.gizmo_axis_vector)
            if gizmo_axis.length_squared > 1.0e-12:
                axis = gizmo_axis
            if axis is not None:
                numeric_value = (
                    None
                    if self.numeric_text in {"", "-", ".", "-."}
                    else float(self.numeric_text)
                )
                if numeric_value is None:
                    ray_origin, ray_direction = _mouse_ray(
                        context, Vector((event.mouse_region_x, event.mouse_region_y))
                    )
                    if ray_origin is None or ray_direction is None:
                        return
                    value = _axis_parameter(
                        self.initial_location, axis, ray_origin, ray_direction
                    )
                else:
                    value = numeric_value
                self.current_value = value
                self._apply_translate_delta(scene, axis * value)
                return
            ray_origin, ray_direction = _mouse_ray(
                context, Vector((event.mouse_region_x, event.mouse_region_y))
            )
            if ray_origin is None or ray_direction is None:
                return
            point = _intersect_plane(
                ray_origin, ray_direction, self.initial_location, self.view_vector
            )
            delta = point - self.start_plane_point
            if self.numeric_text not in {"", "-", ".", "-."}:
                delta = self.view_right.normalized() * float(self.numeric_text)
            self.current_value = delta.length
            self._apply_translate_delta(scene, delta)
            return

        if self.mode == "ROTATE":
            axis_vec = axis if axis is not None else self.view_vector
            angle = (event.mouse_region_x - self.start_mouse.x) * _ROTATE_PIXEL_SCALE
            if self.numeric_text not in {"", "-", ".", "-."}:
                angle = radians(float(self.numeric_text))
            matrix = (
                Matrix.Rotation(angle, 4, axis_vec)
                @ Euler(
                    (
                        float(self.initial_rotation.x),
                        float(self.initial_rotation.y),
                        float(self.initial_rotation.z),
                    ),
                    "XYZ",
                )
                .to_matrix()
                .to_4x4()
            )
            self.current_value = degrees(angle)
            _set_target_rotation(scene, matrix.to_euler("XYZ"))
            return

        factor = 1.0 + (event.mouse_region_x - self.start_mouse.x) * _SCALE_PIXEL_SCALE
        if self.numeric_text not in {"", "-", ".", "-."}:
            factor = float(self.numeric_text)
        factor = max(0.001, factor)
        self.current_value = factor
        if axis is not None:
            scale = self.initial_scale.copy()
            index = {"X": 0, "Y": 1, "Z": 2}[self.constraint_axis]
            scale[index] = max(0.001, self.initial_scale[index] * factor)
            _set_target_scale(scene, scale)
            return
        _set_target_scale(scene, self.initial_scale * factor)

    def _restore(self, context):
        scene = context.scene
        if self.translate_targets:
            _apply_target_locations(
                scene,
                [
                    (kind, target, value)
                    for (kind, target), value in zip(
                        self.translate_targets, self.translate_start_locations
                    )
                ],
            )
        else:
            _set_target_location(scene, self.initial_location)
        _set_target_rotation(scene, self.initial_rotation)
        _set_target_scale(scene, self.initial_scale)

    def _cancel(self, context):
        delete_ids = sdf_proxies.decode_primitive_ids(self.delete_on_cancel_ids)
        if not delete_ids:
            self._restore(context)
            return
        scene = context.scene
        sdf_proxies.remove_primitives(scene, delete_ids, context)
        restore_ids = sdf_proxies.decode_primitive_ids(self.restore_selection_ids)
        if restore_ids:
            sdf_proxies.select_primitives(
                scene,
                restore_ids,
                context,
                self.restore_active_id or restore_ids[-1],
            )

    def _apply_translate_delta(self, scene, delta):
        if not self.translate_targets:
            _set_target_location(scene, self.initial_location + delta)
            return
        _apply_target_locations(
            scene,
            [
                (kind, target, value + delta)
                for (kind, target), value in zip(
                    self.translate_targets, self.translate_start_locations
                )
            ],
        )

    def _update_status(self, context):
        label = {"TRANSLATE": "G", "ROTATE": "R", "SCALE": "S"}[self.mode]
        axis = self.constraint_axis or "View"
        exact = self.numeric_text or f"{self.current_value:.3f}"
        context.workspace.status_text_set(f"{label} | Axis: {axis} | Value: {exact}")

    def _finish(self, context):
        context.workspace.status_text_set(None)


class MATHOPS_V2_OT_duplicate_sdf(Operator):
    bl_idname = "mathops_v2.duplicate_sdf"
    bl_label = "Duplicate SDF"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _viewport_allows_sdf_edit(context)

    def execute(self, context):
        scene = context.scene
        source_ids = sdf_proxies.selected_primitive_ids(scene, context)
        if not source_ids:
            return {"CANCELLED"}
        restore_active_id = sdf_proxies.active_primitive_id(scene, context)
        new_ids = sdf_proxies.duplicate_primitives(scene, source_ids, context)
        if not new_ids:
            return {"CANCELLED"}
        bpy.ops.mathops_v2.transform_sdf(
            "INVOKE_DEFAULT",
            mode="TRANSLATE",
            delete_on_cancel_ids=sdf_proxies.encode_primitive_ids(new_ids),
            restore_selection_ids=sdf_proxies.encode_primitive_ids(source_ids),
            restore_active_id=str(restore_active_id or ""),
        )
        return {"FINISHED"}


class MATHOPS_V2_OT_pick_sdf(Operator):
    bl_idname = "mathops_v2.pick_sdf"
    bl_label = "Pick SDF"
    bl_description = (
        "Click a visible SDF surface in the viewport to select its primitive node"
    )
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return _viewport_allows_sdf_pick(context)

    def invoke(self, context, event):
        if getattr(getattr(context, "space_data", None), "region_3d", None) is None:
            self.report({"ERROR"}, "Open a 3D View to pick an SDF")
            return {"CANCELLED"}
        window = getattr(context, "window", None)
        if window is not None:
            window.cursor_modal_set("EYEDROPPER")
        workspace = getattr(context, "workspace", None)
        if workspace is not None:
            workspace.status_text_set("Click an SDF surface to select its node")
        window_manager = getattr(context, "window_manager", None)
        if window_manager is not None:
            window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE"}:
            self._finish(context)
            return {"CANCELLED"}
        if event.type != "LEFTMOUSE" or event.value != "PRESS":
            return {"RUNNING_MODAL"}

        region, coord = _viewport_region_coords(context, event)
        if region is None or coord is None:
            self.report({"INFO"}, "Click inside the 3D viewport window")
            return {"RUNNING_MODAL"}

        scene = context.scene
        hit = _pick_node_in_region(context, region, coord)

        self._finish(context)
        if hit is None:
            self.report({"INFO"}, "No visible SDF primitive hit")
            return {"FINISHED"}

        if isinstance(hit, str):
            self.report({"INFO"}, "Selected SDF primitive")
            return {"FINISHED"}

        _set_active_node(scene, hit, frame=True)
        self.report({"INFO"}, f"Selected '{hit.name}'")
        return {"FINISHED"}

    def _finish(self, context):
        context.window.cursor_modal_restore()
        context.workspace.status_text_set(None)


class MATHOPS_V2_OT_select_sdf_click(Operator):
    bl_idname = "mathops_v2.select_sdf_click"
    bl_label = "Select SDF Click"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return _viewport_allows_sdf_pick(context)

    def invoke(self, context, event):
        if (
            getattr(event, "alt", False)
            or getattr(event, "ctrl", False)
            or getattr(event, "oskey", False)
        ):
            return {"PASS_THROUGH"}

        region, coord = _viewport_region_coords(context, event)
        if region is None or coord is None:
            return {"PASS_THROUGH"}

        hit = _pick_node_in_region(context, region, coord)
        if hit is None:
            return {"PASS_THROUGH"}

        if isinstance(hit, str):
            return {"FINISHED"}

        _set_active_node(
            context.scene,
            hit,
            extend=bool(getattr(event, "shift", False)),
            frame=True,
        )
        return {"FINISHED"}


class _MathOPSV2GizmoBase:
    transform_mode = "TRANSLATE"

    @classmethod
    def poll(cls, context):
        del cls, context
        return False


class MATHOPS_V2_GGT_sdf_translate(_MathOPSV2GizmoBase, GizmoGroup):
    bl_idname = "VIEW3D_GGT_mathops_v2_sdf_translate"
    bl_label = "MathOPS-v2 SDF Move"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options = {"3D", "PERSISTENT"}
    transform_mode = "TRANSLATE"

    @classmethod
    def poll(cls, context):
        del context
        return False

    def setup(self, context):
        del context
        self.move_gizmos = []
        self.move_ops = []
        for axis_index, color in enumerate(_AXIS_COLORS):
            arrow = self.gizmos.new("GIZMO_GT_arrow_3d")
            op = arrow.target_set_operator(MATHOPS_V2_OT_transform_sdf.bl_idname)
            op.mode = "TRANSLATE"
            op.constraint_axis_init = "XYZ"[axis_index]
            op.gizmo_axis_vector = (0.0, 0.0, 0.0)
            arrow.scale_basis = 0.32
            arrow.line_width = 2.4
            arrow.color = color[0], color[1], color[2]
            arrow.alpha = 0.7
            arrow.color_highlight = color[0], color[1], color[2]
            arrow.alpha_highlight = 1.0
            arrow.use_draw_modal = True
            self.move_gizmos.append(arrow)
            self.move_ops.append(op)

    def refresh(self, context):
        self._update_gizmos(context)

    def draw_prepare(self, context):
        self._update_gizmos(context)

    def _update_gizmos(self, context):
        kind, _target = _active_target(context.scene)
        if kind is None:
            return
        _rotation_matrix, _origin, axes, up_hints = _target_axes(context.scene)
        origin = _selection_center(context.scene)
        for axis_index, gizmo in enumerate(self.move_gizmos):
            axis = axes[axis_index]
            gizmo.matrix_basis = _basis_from_axis(origin, axis, up_hints[axis_index])
            self.move_ops[axis_index].gizmo_axis_vector = (
                float(axis.x),
                float(axis.y),
                float(axis.z),
            )


class MATHOPS_V2_GGT_sdf_rotate(_MathOPSV2GizmoBase, GizmoGroup):
    bl_idname = "VIEW3D_GGT_mathops_v2_sdf_rotate"
    bl_label = "MathOPS-v2 SDF Rotate"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options = {"3D", "PERSISTENT"}
    transform_mode = "ROTATE"

    def setup(self, context):
        del context
        self.rotate_gizmos = []
        for axis_index, color in enumerate(_AXIS_COLORS):
            dial = self.gizmos.new("GIZMO_GT_dial_3d")
            dial.target_set_handler(
                "offset",
                get=lambda axis_index=axis_index: _rotation_get(axis_index),
                set=lambda value, axis_index=axis_index: _rotation_set(
                    axis_index, value
                ),
            )
            dial.draw_options = {"ANGLE_START_Y"}
            dial.scale_basis = 0.7
            dial.line_width = 2.5
            dial.color = color[0], color[1], color[2]
            dial.alpha = 0.55
            dial.color_highlight = color[0], color[1], color[2]
            dial.alpha_highlight = 0.95
            dial.use_draw_modal = True
            self.rotate_gizmos.append(dial)

    def refresh(self, context):
        self._update_matrices(context)

    def draw_prepare(self, context):
        self._update_matrices(context)

    def _update_matrices(self, context):
        kind, _target = _active_target(context.scene)
        if kind is None:
            return
        _rotation_matrix, origin, axes, up_hints = _target_axes(context.scene)
        for axis_index, gizmo in enumerate(self.rotate_gizmos):
            gizmo.matrix_basis = _basis_from_axis(
                origin, axes[axis_index], up_hints[axis_index]
            )


class MATHOPS_V2_GGT_sdf_scale(_MathOPSV2GizmoBase, GizmoGroup):
    bl_idname = "VIEW3D_GGT_mathops_v2_sdf_scale"
    bl_label = "MathOPS-v2 SDF Scale"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options = {"3D", "PERSISTENT"}
    transform_mode = "SCALE"

    def setup(self, context):
        del context
        self.scale_gizmos = []
        for axis_index, color in enumerate(_AXIS_COLORS):
            arrow = self.gizmos.new("GIZMO_GT_arrow_3d")
            arrow.target_set_handler(
                "offset",
                get=lambda axis_index=axis_index: _scale_get(axis_index),
                set=lambda value, axis_index=axis_index: _scale_set(axis_index, value),
            )
            arrow.draw_style = "BOX"
            arrow.scale_basis = 0.28
            arrow.line_width = 2.0
            arrow.color = color[0], color[1], color[2]
            arrow.alpha = 0.65
            arrow.color_highlight = color[0], color[1], color[2]
            arrow.alpha_highlight = 0.95
            arrow.use_draw_modal = True
            self.scale_gizmos.append(arrow)

    def refresh(self, context):
        self._update_matrices(context)

    def draw_prepare(self, context):
        self._update_matrices(context)

    def _update_matrices(self, context):
        kind, _target = _active_target(context.scene)
        if kind is None:
            return
        _rotation_matrix, origin, axes, up_hints = _target_axes(context.scene)
        for axis_index, gizmo in enumerate(self.scale_gizmos):
            gizmo.matrix_basis = _basis_from_axis(
                origin, axes[axis_index], up_hints[axis_index]
            )


classes = (
    MATHOPS_V2_OT_pick_sdf,
    MATHOPS_V2_OT_select_sdf_click,
    MATHOPS_V2_OT_transform_sdf,
    MATHOPS_V2_OT_duplicate_sdf,
    MATHOPS_V2_GGT_sdf_translate,
    MATHOPS_V2_GGT_sdf_rotate,
    MATHOPS_V2_GGT_sdf_scale,
)


def _register_keymaps():
    wm = getattr(bpy.context, "window_manager", None)
    keyconfig = None if wm is None else getattr(wm.keyconfigs, "addon", None)
    if keyconfig is None:
        return
    km = keyconfig.keymaps.new(name="3D View", space_type="VIEW_3D")
    kmi = km.keymap_items.new(
        MATHOPS_V2_OT_select_sdf_click.bl_idname,
        "LEFTMOUSE",
        "CLICK",
    )
    _addon_keymaps.append((km, kmi))
    km = keyconfig.keymaps.new(name="Object Mode", space_type="EMPTY")
    kmi = km.keymap_items.new(
        MATHOPS_V2_OT_duplicate_sdf.bl_idname,
        "D",
        "PRESS",
        shift=True,
    )
    _addon_keymaps.append((km, kmi))


def _unregister_keymaps():
    while _addon_keymaps:
        km, kmi = _addon_keymaps.pop()
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    _register_keymaps()


def unregister():
    _unregister_keymaps()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
