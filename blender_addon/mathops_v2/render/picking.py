import math

from bpy_extras import view3d_utils
from mathutils import Vector

from .. import runtime
from ..nodes import sdf_tree


_MISS_DISTANCE = 1.0e20


def _clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def _dot4(row, point):
    return (
        float(row[0]) * float(point[0])
        + float(row[1]) * float(point[1])
        + float(row[2]) * float(point[2])
        + float(row[3]) * float(point[3])
    )


def _world_to_local(rows, world_point):
    world4 = (float(world_point[0]), float(world_point[1]), float(world_point[2]), 1.0)
    row0, row1, row2 = rows
    return Vector((_dot4(row0, world4), _dot4(row1, world4), _dot4(row2, world4)))


def _local_to_world(rows, local_point):
    row0, row1, row2 = rows
    basis_x = Vector((float(row0[0]), float(row1[0]), float(row2[0])))
    basis_y = Vector((float(row0[1]), float(row1[1]), float(row2[1])))
    basis_z = Vector((float(row0[2]), float(row1[2]), float(row2[2])))
    origin = -(basis_x * float(row0[3]) + basis_y * float(row1[3]) + basis_z * float(row2[3]))
    return origin + basis_x * float(local_point[0]) + basis_y * float(local_point[1]) + basis_z * float(local_point[2])


def _sabs(value, blend):
    if blend <= 1.0e-4:
        return abs(value)
    axis_value = abs(float(value))
    if axis_value >= blend:
        return axis_value
    t = axis_value / blend
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2
    return blend * (0.25 + 1.5 * t2 - t3 + 0.25 * t4)


def _mirror_origin(origin_object):
    try:
        if origin_object is None:
            return Vector((0.0, 0.0, 0.0))
        translation = origin_object.matrix_world.translation
        return Vector((float(translation[0]), float(translation[1]), float(translation[2])))
    except ReferenceError:
        return Vector((0.0, 0.0, 0.0))


def _array_origin(origin_object, primitive_center):
    try:
        if origin_object is None:
            return Vector((float(primitive_center[0]), float(primitive_center[1]), float(primitive_center[2])))
        translation = origin_object.matrix_world.translation
        return Vector((float(translation[0]), float(translation[1]), float(translation[2])))
    except ReferenceError:
        return Vector((float(primitive_center[0]), float(primitive_center[1]), float(primitive_center[2])))


def _fold_finite_grid_axis(axis_value, origin, primitive_center, spacing, count, blend):
    if count <= 1 or spacing <= 1.0e-6:
        return float(axis_value)
    base = (float(primitive_center) - float(origin)) / float(spacing)
    center_offset = 0.5 * float(count - 1)
    shifted = (float(axis_value) - float(origin)) / float(spacing) - base + center_offset
    index = _clamp(round(shifted), 0.0, float(count - 1))
    local = shifted - index
    pull_right = (0.5 - local) * float(spacing)
    pull_right = pull_right - _sabs(pull_right, blend)
    pull_left = (0.5 + local) * float(spacing)
    pull_left = pull_left - _sabs(pull_left, blend)
    if index < 0.5:
        pull_left = 0.0
    if index > float(count) - 1.5:
        pull_right = 0.0
    local += (pull_right - pull_left) / float(spacing)
    return float(origin) + (local + base) * float(spacing)


def _wrap_pi(value):
    two_pi = math.tau
    wrapped = math.fmod(float(value) + math.pi, two_pi)
    if wrapped < 0.0:
        wrapped += two_pi
    return wrapped - math.pi


def _apply_radial_array(world_point, origin, primitive_center, radius, count, blend):
    if count <= 1 or radius <= 1.0e-6:
        return Vector((float(world_point[0]), float(world_point[1]), float(world_point[2])))
    point = Vector((float(world_point[0]), float(world_point[1]), float(world_point[2])))
    origin = Vector((float(origin[0]), float(origin[1]), float(origin[2])))
    primitive_center = Vector((float(primitive_center[0]), float(primitive_center[1]), float(primitive_center[2])))
    q = point - origin
    base_offset = primitive_center - origin + Vector((float(radius), 0.0, 0.0))
    base_angle = math.atan2(float(base_offset[1]), float(base_offset[0]))
    base_radius = max(math.hypot(float(base_offset[0]), float(base_offset[1])), 1.0e-4)
    sector = math.tau / float(count)
    angle = math.atan2(float(q[1]), float(q[0]))
    angle_rel = _wrap_pi(angle - base_angle)
    norm_a = angle_rel / sector
    index = round(norm_a)
    local = norm_a - index
    mirrored = (abs(index) % 2) == 1
    at_defect = (count % 2) != 0 and abs(angle_rel) > math.pi - sector * 0.5
    if at_defect:
        local = abs(local)
    elif mirrored:
        local = -local
    arc = sector * base_radius
    pull_right = (0.5 - local) * arc
    pull_right = pull_right - _sabs(pull_right, blend)
    pull_left = (0.5 + local) * arc
    pull_left = pull_left - _sabs(pull_left, blend)
    local += (pull_right - pull_left) / arc
    fold_angle = local * sector + base_angle
    radius_xy = math.hypot(float(q[0]), float(q[1]))
    q_fold = Vector((radius_xy * math.cos(fold_angle), radius_xy * math.sin(fold_angle), float(q[2])))
    return primitive_center + (q_fold - base_offset)


def _apply_warps(world_point, primitive_center, rows, warps):
    warped_point = Vector((float(world_point[0]), float(world_point[1]), float(world_point[2])))
    primitive_center = Vector((float(primitive_center[0]), float(primitive_center[1]), float(primitive_center[2])))
    for warp in tuple(warps or ()):
        kind = str(warp[0] or "")
        if kind == "mirror":
            _kind, flags, origin_object, blend = warp
            blend = max(float(blend), 0.0)
            origin = _mirror_origin(origin_object)
            if int(flags) & 1:
                side = 1.0 if primitive_center.x >= origin.x else -1.0
                warped_point.x = origin.x + side * _sabs(side * (warped_point.x - origin.x), blend)
            if int(flags) & 2:
                side = 1.0 if primitive_center.y >= origin.y else -1.0
                warped_point.y = origin.y + side * _sabs(side * (warped_point.y - origin.y), blend)
            if int(flags) & 4:
                side = 1.0 if primitive_center.z >= origin.z else -1.0
                warped_point.z = origin.z + side * _sabs(side * (warped_point.z - origin.z), blend)
            continue
        if kind == "grid":
            _kind, count_x, count_y, count_z, spacing, origin_object, blend = warp
            blend = max(float(blend), 0.0)
            local_point = _world_to_local(rows, warped_point)
            local_origin = _world_to_local(rows, _array_origin(origin_object, primitive_center))
            local_center = _world_to_local(rows, primitive_center)
            local_point.x = _fold_finite_grid_axis(local_point.x, local_origin.x, local_center.x, abs(float(spacing[0])), int(count_x), blend)
            local_point.y = _fold_finite_grid_axis(local_point.y, local_origin.y, local_center.y, abs(float(spacing[1])), int(count_y), blend)
            local_point.z = _fold_finite_grid_axis(local_point.z, local_origin.z, local_center.z, abs(float(spacing[2])), int(count_z), blend)
            warped_point = _local_to_world(rows, local_point)
            continue
        if kind == "radial":
            _kind, count, radius, origin_object, blend = warp
            local_point = _world_to_local(rows, warped_point)
            local_origin = _world_to_local(rows, _array_origin(origin_object, primitive_center))
            local_center = _world_to_local(rows, primitive_center)
            local_point = _apply_radial_array(local_point, local_origin, local_center, float(radius), int(count), max(float(blend), 0.0))
            warped_point = _local_to_world(rows, local_point)
    return warped_point


def _primitive_local_point(spec, entry, world_point):
    row0, row1, row2 = spec["world_to_local"]
    rows = (row0, row1, row2)
    warped_point = _apply_warps(world_point, spec.get("center", (0.0, 0.0, 0.0)), rows, entry.get("warps", ()))
    return _world_to_local(rows, warped_point)


def _safe_component(value):
    return max(float(value), 1.0e-6)


def _safe_vec(values):
    return Vector((_safe_component(values[0]), _safe_component(values[1]), _safe_component(values[2])))


def _sd_ellipsoid(point, radius):
    radius = _safe_vec(radius)
    radius_sq = Vector((radius[0] * radius[0], radius[1] * radius[1], radius[2] * radius[2]))
    k0 = Vector((point[0] / radius[0], point[1] / radius[1], point[2] / radius[2])).length
    k1 = Vector((point[0] / radius_sq[0], point[1] / radius_sq[1], point[2] / radius_sq[2])).length
    if k1 <= 1.0e-12:
        return -min(float(radius[0]), float(radius[1]), float(radius[2]))
    return k0 * (k0 - 1.0) / k1


def _sd_box(point, half_size):
    q = Vector((abs(point[0]) - half_size[0], abs(point[1]) - half_size[1], abs(point[2]) - half_size[2]))
    outside = Vector((max(q[0], 0.0), max(q[1], 0.0), max(q[2], 0.0))).length
    inside = min(max(q[0], max(q[1], q[2])), 0.0)
    return outside + inside


def _sd_cylinder(point, radius, half_height):
    radial = math.hypot(float(point[0]), float(point[1]))
    dx = abs(radial) - float(radius)
    dy = abs(float(point[2])) - float(half_height)
    outside = math.hypot(max(dx, 0.0), max(dy, 0.0))
    inside = min(max(dx, dy), 0.0)
    return outside + inside


def _sd_torus(point, radii):
    qx = math.hypot(float(point[0]), float(point[2])) - float(radii[0])
    qy = float(point[1])
    return math.hypot(qx, qy) - float(radii[1])


def eval_primitive(compiled, primitive_index, world_point):
    primitive_specs = compiled.get("primitive_specs", ())
    primitive_entries = compiled.get("primitive_entries", ())
    if primitive_index < 0 or primitive_index >= len(primitive_specs) or primitive_index >= len(primitive_entries):
        return _MISS_DISTANCE

    spec = primitive_specs[primitive_index]
    entry = primitive_entries[primitive_index]
    local_point = _primitive_local_point(spec, entry, world_point)
    primitive_type = str(spec.get("primitive_type", "sphere") or "sphere")
    meta = tuple(float(value) for value in spec.get("meta", (0.0, 0.5, 0.0, 0.0)))
    scale = _safe_vec(spec.get("scale", (1.0, 1.0, 1.0)))
    min_scale = max(min(float(scale[0]), float(scale[1]), float(scale[2])), 1.0e-6)
    if primitive_type == "sphere":
        return _sd_ellipsoid(local_point, Vector((meta[1] * scale[0], meta[1] * scale[1], meta[1] * scale[2])))
    if primitive_type == "box":
        return _sd_box(local_point, Vector((meta[1] * scale[0], meta[2] * scale[1], meta[3] * scale[2])))
    if primitive_type == "cylinder":
        scaled_point = Vector((float(local_point[0]) / float(scale[0]), float(local_point[1]) / float(scale[1]), float(local_point[2]) / float(scale[2])))
        return _sd_cylinder(scaled_point, meta[1], meta[2]) * min_scale
    if primitive_type == "torus":
        scaled_point = Vector((float(local_point[0]) / float(scale[0]), float(local_point[1]) / float(scale[1]), float(local_point[2]) / float(scale[2])))
        return _sd_torus(scaled_point, (meta[1], meta[2])) * min_scale
    return _MISS_DISTANCE


def _op_smooth_union(lhs, rhs, blend):
    h = _clamp(0.5 + 0.5 * (rhs - lhs) / blend, 0.0, 1.0)
    return (rhs * (1.0 - h)) + (lhs * h) - blend * h * (1.0 - h)


def _op_smooth_subtract(lhs, rhs, blend):
    h = _clamp(0.5 - 0.5 * (rhs + lhs) / blend, 0.0, 1.0)
    return (lhs * (1.0 - h)) + (-rhs * h) + blend * h * (1.0 - h)


def _op_smooth_intersect(lhs, rhs, blend):
    h = _clamp(0.5 - 0.5 * (rhs - lhs) / blend, 0.0, 1.0)
    return (rhs * (1.0 - h)) + (lhs * h) + blend * h * (1.0 - h)


def _apply_op(kind, lhs, rhs, blend):
    if kind == 1:
        if blend <= 1.0e-6:
            return min(lhs, rhs)
        return _op_smooth_union(lhs, rhs, blend)
    if kind == 2:
        if blend <= 1.0e-6:
            return max(lhs, -rhs)
        return _op_smooth_subtract(lhs, rhs, blend)
    if kind == 3:
        if blend <= 1.0e-6:
            return max(lhs, rhs)
        return _op_smooth_intersect(lhs, rhs, blend)
    return rhs


def eval_scene(compiled, world_point):
    stack = []
    for instruction in compiled.get("instruction_rows", ()):
        kind = int(float(instruction[0]) + 0.5)
        if kind == 0:
            distance_value = eval_primitive(compiled, int(float(instruction[1]) + 0.5), world_point)
        else:
            if len(stack) < 2:
                return _MISS_DISTANCE
            rhs = stack.pop()
            lhs = stack.pop()
            distance_value = _apply_op(kind, lhs, rhs, max(float(instruction[3]), 0.0))
        stack.append(distance_value)
    return stack[-1] if stack else _MISS_DISTANCE


def _bbox_intersect(bounds_min, bounds_max, ray_origin, ray_direction):
    t0 = 0.0
    t1 = 1.0e30
    for axis in range(3):
        origin = float(ray_origin[axis])
        direction = float(ray_direction[axis])
        axis_min = float(bounds_min[axis])
        axis_max = float(bounds_max[axis])
        if abs(direction) < 1.0e-8:
            if origin < axis_min or origin > axis_max:
                return None
            continue
        inv_direction = 1.0 / direction
        axis_t0 = (axis_min - origin) * inv_direction
        axis_t1 = (axis_max - origin) * inv_direction
        t0 = max(t0, min(axis_t0, axis_t1))
        t1 = min(t1, max(axis_t0, axis_t1))
        if t1 <= t0:
            return None
    return t0


def _view_ray(region, region_data, coord):
    ray_direction = view3d_utils.region_2d_to_vector_3d(region, region_data, coord)
    if bool(getattr(region_data, "is_perspective", True)):
        ray_origin = region_data.view_matrix.inverted().translation
    else:
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, region_data, coord)
    return Vector(ray_origin), Vector(ray_direction).normalized()


def _pick_primitive_entry(compiled, world_point):
    return _pick_primitive_entry_filtered(compiled, world_point)


def _entry_object(entry):
    hit_object = entry.get("object")
    if hit_object is not None:
        return hit_object
    hit_node = entry.get("node")
    if hit_node is None:
        return None
    return getattr(hit_node, "target", None)


def _entry_visible(entry, visible_proxy_pointers=None):
    if visible_proxy_pointers is None:
        return True
    object_pointer = runtime.safe_pointer(_entry_object(entry))
    return object_pointer != 0 and object_pointer in visible_proxy_pointers


def _combine_surface_hit(kind, lhs_hit, rhs_hit, blend):
    lhs_distance, lhs_entry = lhs_hit
    rhs_distance, rhs_entry = rhs_hit

    if kind == 1:
        if blend <= 1.0e-6:
            return (lhs_distance, lhs_entry) if lhs_distance <= rhs_distance else (rhs_distance, rhs_entry)
        h = _clamp(0.5 + 0.5 * (rhs_distance - lhs_distance) / blend, 0.0, 1.0)
        distance_value = (rhs_distance * (1.0 - h)) + (lhs_distance * h) - blend * h * (1.0 - h)
        return distance_value, (lhs_entry if h > 0.5 else rhs_entry)

    if kind == 2:
        if blend <= 1.0e-6:
            return (lhs_distance, lhs_entry) if lhs_distance >= -rhs_distance else (max(lhs_distance, -rhs_distance), rhs_entry)
        h = _clamp(0.5 - 0.5 * (rhs_distance + lhs_distance) / blend, 0.0, 1.0)
        distance_value = (lhs_distance * (1.0 - h)) + (-rhs_distance * h) + blend * h * (1.0 - h)
        return distance_value, (rhs_entry if h > 0.5 else lhs_entry)

    if kind == 3:
        if blend <= 1.0e-6:
            return (lhs_distance, lhs_entry) if lhs_distance >= rhs_distance else (rhs_distance, rhs_entry)
        h = _clamp(0.5 - 0.5 * (rhs_distance - lhs_distance) / blend, 0.0, 1.0)
        distance_value = (rhs_distance * (1.0 - h)) + (lhs_distance * h) + blend * h * (1.0 - h)
        return distance_value, (lhs_entry if h > 0.5 else rhs_entry)

    return rhs_distance, rhs_entry


def _surface_entry_from_node(compiled, world_point, node, visible_proxy_pointers=None):
    if node is None:
        return _MISS_DISTANCE, None

    kind = str(node.get("kind", "") or "")
    if kind == "primitive":
        primitive_index = int(node.get("primitive_index", -1))
        primitive_entries = compiled.get("primitive_entries", ())
        if primitive_index < 0 or primitive_index >= len(primitive_entries):
            return _MISS_DISTANCE, None
        entry = primitive_entries[primitive_index]
        if not _entry_visible(entry, visible_proxy_pointers):
            return _MISS_DISTANCE, None
        return eval_primitive(compiled, primitive_index, world_point), entry

    if kind != "op":
        return _MISS_DISTANCE, None

    lhs_hit = _surface_entry_from_node(compiled, world_point, node.get("left"), visible_proxy_pointers)
    rhs_hit = _surface_entry_from_node(compiled, world_point, node.get("right"), visible_proxy_pointers)
    return _combine_surface_hit(
        int(node.get("op", 0)),
        lhs_hit,
        rhs_hit,
        max(float(node.get("blend", 0.0)), 0.0),
    )


def _pick_primitive_entry_filtered(compiled, world_point, visible_proxy_pointers=None):
    _distance_value, entry = _surface_entry_from_node(
        compiled,
        world_point,
        compiled.get("root_node"),
        visible_proxy_pointers,
    )
    return entry


def pick_viewport_sdf(context, coord):
    scene = getattr(context, "scene", None)
    region = getattr(context, "region", None)
    region_data = getattr(context, "region_data", None)
    if scene is None or region is None or region_data is None:
        return None

    visible_proxy_pointers = None
    visible_objects = getattr(context, "visible_objects", None)
    if visible_objects is not None:
        visible_proxy_pointers = {
            runtime.safe_pointer(obj)
            for obj in visible_objects
            if runtime.is_sdf_proxy(obj)
        }
        if not visible_proxy_pointers:
            return None

    compiled = sdf_tree.compile_scene(scene)
    if int(compiled.get("primitive_count", 0)) <= 0 or int(compiled.get("instruction_count", 0)) <= 0:
        return None

    ray_origin, ray_direction = _view_ray(region, region_data, coord)
    bounds_min, bounds_max = compiled.get("scene_bounds", ((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0)))
    travel = _bbox_intersect(bounds_min, bounds_max, ray_origin, ray_direction)
    if travel is None:
        return None

    settings = runtime.scene_settings(scene)
    surface_epsilon = 0.0015 if settings is None else float(getattr(settings, "surface_epsilon", 0.0015))
    max_distance = 200.0 if settings is None else float(getattr(settings, "max_distance", 200.0))
    max_steps = 96 if settings is None else int(getattr(settings, "max_steps", 96))

    travel += 1.0e-4
    march_distance = 0.0
    hit_point = None
    for _step in range(max(0, min(max_steps, 4096))):
        world_point = ray_origin + ray_direction * travel
        if (
            world_point[0] < bounds_min[0]
            or world_point[1] < bounds_min[1]
            or world_point[2] < bounds_min[2]
            or world_point[0] >= bounds_max[0]
            or world_point[1] >= bounds_max[1]
            or world_point[2] >= bounds_max[2]
        ):
            break
        distance_value = eval_scene(compiled, world_point)
        if distance_value < surface_epsilon:
            hit_point = world_point
            break
        step_distance = abs(distance_value)
        travel += step_distance
        march_distance += step_distance
        if march_distance > max_distance:
            break

    if hit_point is None:
        return None

    hit_entry = _pick_primitive_entry_filtered(compiled, hit_point, visible_proxy_pointers=visible_proxy_pointers)
    if hit_entry is None:
        return None
    hit_node = hit_entry.get("node")
    hit_object = _entry_object(hit_entry)
    return {
        "object": hit_object,
        "node": hit_node,
        "point": hit_point,
        "compiled": compiled,
    }
