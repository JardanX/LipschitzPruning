import math

from bpy_extras import view3d_utils
from mathutils import Euler, Matrix, Vector

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


def _array_field_origin(frame_node=None, branch_center=(0.0, 0.0, 0.0)):
    if frame_node is not None and bool(getattr(frame_node, "use_array_transform", False)):
        location = getattr(frame_node, "array_location", (0.0, 0.0, 0.0))
        return Vector((float(location[0]), float(location[1]), float(location[2])))
    return Vector((float(branch_center[0]), float(branch_center[1]), float(branch_center[2])))


def _array_scale(frame_node=None):
    if frame_node is not None and bool(getattr(frame_node, "use_array_transform", False)):
        scale = getattr(frame_node, "array_scale", (1.0, 1.0, 1.0))
        return Vector((max(abs(float(scale[0])), 1.0e-6), max(abs(float(scale[1])), 1.0e-6), max(abs(float(scale[2])), 1.0e-6)))
    return Vector((1.0, 1.0, 1.0))


def _array_rotation(_origin_object=None, frame_node=None):
    if frame_node is not None and bool(getattr(frame_node, "use_array_transform", False)):
        rotation = getattr(frame_node, "array_rotation", (0.0, 0.0, 0.0))
        return Matrix.LocRotScale(Vector((0.0, 0.0, 0.0)), Euler((float(rotation[0]), float(rotation[1]), float(rotation[2])), "XYZ"), None).to_quaternion()
    return Matrix.Identity(3).to_quaternion()


def _array_repeat_origin(origin_object, primitive_center, frame_node=None):
    try:
        origin_object = runtime.object_identity(origin_object)
        if origin_object is None:
            return Vector((float(primitive_center[0]), float(primitive_center[1]), float(primitive_center[2])))
        translation = origin_object.matrix_world.translation
        field_origin = _array_field_origin(frame_node, primitive_center)
        rotation = _array_rotation(None, frame_node)
        scale = _array_scale(frame_node)
        local = rotation.inverted() @ (Vector((float(translation[0]), float(translation[1]), float(translation[2]))) - field_origin)
        return Vector((float(primitive_center[0]) + (local[0] / scale[0]), float(primitive_center[1]) + (local[1] / scale[1]), float(primitive_center[2]) + (local[2] / scale[2])))
    except ReferenceError:
        return Vector((float(primitive_center[0]), float(primitive_center[1]), float(primitive_center[2])))


def _world_to_array_local(origin, rotation, scale, branch_center, world_point):
    point = Vector((float(world_point[0]), float(world_point[1]), float(world_point[2])))
    local = rotation.inverted() @ (point - origin)
    return Vector((float(branch_center[0]) + (local[0] / scale[0]), float(branch_center[1]) + (local[1] / scale[1]), float(branch_center[2]) + (local[2] / scale[2])))


def _apply_instance_offset_inverse(point, branch_center, offset_location, offset_rotation, offset_scale):
    centered = Vector((
        float(point[0]) - float(branch_center[0]) - float(offset_location[0]),
        float(point[1]) - float(branch_center[1]) - float(offset_location[1]),
        float(point[2]) - float(branch_center[2]) - float(offset_location[2]),
    ))
    local = offset_rotation.inverted() @ centered
    return Vector((
        float(branch_center[0]) + (local[0] / float(offset_scale[0])),
        float(branch_center[1]) + (local[1] / float(offset_scale[1])),
        float(branch_center[2]) + (local[2] / float(offset_scale[2])),
    ))


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
    distance_scale = 1.0
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
            _kind, count_x, count_y, count_z, spacing, _origin_object, frame_node, _source_node, branch_center, offset_location, offset_rotation, offset_scale, blend = warp
            blend = max(float(blend), 0.0)
            field_origin = _array_field_origin(frame_node, branch_center)
            rotation = _array_rotation(None, frame_node)
            scale = _array_scale(frame_node)
            distance_scale *= max(min(float(scale[0]), float(scale[1]), float(scale[2])), 1.0e-6)
            distance_scale *= max(min(float(offset_scale[0]), float(offset_scale[1]), float(offset_scale[2])), 1.0e-6)
            array_point = _world_to_array_local(field_origin, rotation, scale, branch_center, warped_point)
            offset_center = sdf_tree._offset_center(branch_center, offset_location)
            array_point.x = _fold_finite_grid_axis(array_point.x, float(offset_center[0]), float(offset_center[0]), abs(float(spacing[0])), int(count_x), blend)
            array_point.y = _fold_finite_grid_axis(array_point.y, float(offset_center[1]), float(offset_center[1]), abs(float(spacing[1])), int(count_y), blend)
            array_point.z = _fold_finite_grid_axis(array_point.z, float(offset_center[2]), float(offset_center[2]), abs(float(spacing[2])), int(count_z), blend)
            offset_quaternion = Matrix.LocRotScale(Vector((0.0, 0.0, 0.0)), Euler((float(offset_rotation[0]), float(offset_rotation[1]), float(offset_rotation[2])), "XYZ"), None).to_quaternion()
            warped_point = _apply_instance_offset_inverse(array_point, branch_center, offset_location, offset_quaternion, offset_scale)
            continue
        if kind == "radial":
            _kind, count, radius, origin_object, frame_node, _source_node, branch_center, offset_location, offset_rotation, offset_scale, blend = warp
            field_origin = _array_field_origin(frame_node, branch_center)
            rotation = _array_rotation(None, frame_node)
            scale = _array_scale(frame_node)
            distance_scale *= max(min(float(scale[0]), float(scale[1]), float(scale[2])), 1.0e-6)
            distance_scale *= max(min(float(offset_scale[0]), float(offset_scale[1]), float(offset_scale[2])), 1.0e-6)
            repeat_origin = _array_repeat_origin(origin_object, branch_center, frame_node)
            array_point = _world_to_array_local(field_origin, rotation, scale, branch_center, warped_point)
            array_point = _apply_radial_array(array_point, repeat_origin, Vector(sdf_tree._offset_center(branch_center, offset_location)), float(radius), int(count), max(float(blend), 0.0))
            offset_quaternion = Matrix.LocRotScale(Vector((0.0, 0.0, 0.0)), Euler((float(offset_rotation[0]), float(offset_rotation[1]), float(offset_rotation[2])), "XYZ"), None).to_quaternion()
            warped_point = _apply_instance_offset_inverse(array_point, branch_center, offset_location, offset_quaternion, offset_scale)
    return warped_point, distance_scale


def _primitive_local_point(spec, entry, world_point):
    row0, row1, row2 = spec["world_to_local"]
    rows = (row0, row1, row2)
    warped_point, distance_scale = _apply_warps(world_point, spec.get("center", (0.0, 0.0, 0.0)), rows, entry.get("warps", ()))
    return _world_to_local(rows, warped_point), distance_scale


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


def _sd_round_box_2d(point, bounds, radii):
    rx = float(radii[0]) if float(point[0]) > 0.0 else float(radii[2])
    ry = float(radii[1]) if float(point[0]) > 0.0 else float(radii[3])
    rc = rx if float(point[1]) > 0.0 else ry
    q = Vector((abs(float(point[0])) - float(bounds[0]) + rc, abs(float(point[1])) - float(bounds[1]) + rc))
    outside = Vector((max(q[0], 0.0), max(q[1], 0.0))).length
    inside = min(max(q[0], q[1]), 0.0)
    return inside + outside - rc


def _sd_chamfer_box_2d(point, bounds, radii):
    rx = float(radii[0]) if float(point[0]) > 0.0 else float(radii[2])
    ry = float(radii[1]) if float(point[0]) > 0.0 else float(radii[3])
    rc = rx if float(point[1]) > 0.0 else ry
    q = Vector((abs(float(point[0])) - float(bounds[0]), abs(float(point[1])) - float(bounds[1])))
    if rc < 0.001:
        outside = Vector((max(q[0], 0.0), max(q[1], 0.0))).length
        inside = min(max(q[0], q[1]), 0.0)
        return outside + inside
    d_cham = (q[0] + q[1] + rc) * 0.70710678
    d_box = max(q[0], q[1])
    d_value = max(d_box, d_cham)
    if d_value <= 0.0:
        return d_value
    if d_box <= 0.0:
        return d_cham
    if q[1] <= -rc or q[0] <= -rc:
        return Vector((max(q[0], 0.0), max(q[1], 0.0))).length
    t = (-q[0] + q[1] + rc) / (2.0 * rc)
    if t <= 0.0:
        return (q - Vector((0.0, -rc))).length
    if t >= 1.0:
        return (q - Vector((-rc, 0.0))).length
    return d_cham


def _sd_advanced_box(point, half_size, corners, edge_top, edge_bottom, taper, corner_mode, edge_mode):
    taper_z = max(float(half_size[2]), 0.001)
    zn = _clamp(float(point[2]) / taper_z, -1.0, 1.0)
    mix_value = (zn + 1.0) * 0.5
    tap_top = max(float(taper), 0.0)
    tap_bottom = max(-float(taper), 0.0)
    tap_factor = max(1.0 - tap_top * mix_value - tap_bottom * (1.0 - mix_value), 0.001)
    size_xy = Vector((float(half_size[0]) * tap_factor, float(half_size[1]) * tap_factor))
    slope = math.sqrt((float(half_size[0]) * float(half_size[0])) + (float(half_size[1]) * float(half_size[1]))) * (tap_top + tap_bottom) / (2.0 * taper_z)
    lipschitz = math.sqrt(1.0 + slope * slope)
    max_radius = min(float(size_xy[0]), float(size_xy[1]))
    scaled_corners = tuple(float(value) * max_radius for value in corners)
    point_xy = Vector((float(point[0]), float(point[1])))
    d2d = _sd_round_box_2d(point_xy, size_xy, scaled_corners) if corner_mode == "SMOOTH" else _sd_chamfer_box_2d(point_xy, size_xy, scaled_corners)
    max_face_radius = min(float(size_xy[0]), float(size_xy[1]))
    dz = abs(float(point[2])) - float(half_size[2])
    edge_radius = (float(edge_top) if float(point[2]) > 0.0 else float(edge_bottom)) * min(max_face_radius, float(half_size[2]))
    if edge_radius > 0.001:
        if edge_mode == "SMOOTH":
            dd0 = d2d + edge_radius
            dd1 = dz + edge_radius
            return (min(max(dd0, dd1), 0.0) + math.hypot(max(dd0, 0.0), max(dd1, 0.0)) - edge_radius) / lipschitz
        base = max(d2d, dz)
        cham = (d2d + dz + edge_radius) * 0.70710678
        d_value = max(base, cham)
        if d_value <= 0.0:
            return d_value / lipschitz
        if d2d <= 0.0 and dz <= 0.0:
            return cham / lipschitz
        if dz <= -edge_radius:
            return d2d / lipschitz
        if d2d <= -edge_radius:
            return dz / lipschitz
        tc2 = (-d2d + dz + edge_radius) / (2.0 * edge_radius)
        if tc2 <= 0.0:
            return math.hypot(d2d, dz + edge_radius) / lipschitz
        if tc2 >= 1.0:
            return math.hypot(d2d + edge_radius, dz) / lipschitz
        return cham / lipschitz
    return _sd_prism(d2d, float(point[2]), float(half_size[2])) / lipschitz


def _sd_cylinder(point, radius, half_height):
    radial = math.hypot(float(point[0]), float(point[1]))
    dx = abs(radial) - float(radius)
    dy = abs(float(point[2])) - float(half_height)
    outside = math.hypot(max(dx, 0.0), max(dy, 0.0))
    inside = min(max(dx, dy), 0.0)
    return outside + inside


def _sd_advanced_cylinder(point, radius, half_height, bevel_top, bevel_bottom, taper, bevel_mode):
    taper_h = max(float(half_height), 0.001)
    zn = _clamp(float(point[2]) / taper_h, -1.0, 1.0)
    mix_value = (zn + 1.0) * 0.5
    tap_top = max(float(taper), 0.0)
    tap_bottom = max(-float(taper), 0.0)
    tap_factor = max(1.0 - tap_top * mix_value - tap_bottom * (1.0 - mix_value), 0.001)
    scaled_radius = float(radius) * tap_factor
    slope = float(radius) * (tap_top + tap_bottom) / (2.0 * taper_h)
    lipschitz = math.sqrt(1.0 + slope * slope)
    d2d = math.hypot(float(point[0]), float(point[1])) - scaled_radius
    dz = abs(float(point[2])) - float(half_height)
    edge_bevel = float(bevel_top) if float(point[2]) > 0.0 else float(bevel_bottom)
    bevel_radius = min(max(edge_bevel, 0.0), max(min(scaled_radius, float(half_height)) - 0.001, 0.0))
    if bevel_radius > 0.001:
        if bevel_mode == "SMOOTH":
            dd0 = d2d + bevel_radius
            dd1 = dz + bevel_radius
            return (min(max(dd0, dd1), 0.0) + math.hypot(max(dd0, 0.0), max(dd1, 0.0)) - bevel_radius) / lipschitz
        base = max(d2d, dz)
        cham = (d2d + dz + bevel_radius) * 0.70710678
        d_value = max(base, cham)
        if d_value <= 0.0:
            return d_value / lipschitz
        if d2d <= 0.0 and dz <= 0.0:
            return cham / lipschitz
        if dz <= -bevel_radius:
            return d2d / lipschitz
        if d2d <= -bevel_radius:
            return dz / lipschitz
        tc2 = (-d2d + dz + bevel_radius) / (2.0 * bevel_radius)
        if tc2 <= 0.0:
            return math.hypot(d2d, dz + bevel_radius) / lipschitz
        if tc2 >= 1.0:
            return math.hypot(d2d + bevel_radius, dz) / lipschitz
        return cham / lipschitz
    return _sd_prism(d2d, float(point[2]), float(half_height)) / lipschitz


def _sd_torus(point, radii):
    qx = math.hypot(float(point[0]), float(point[1])) - float(radii[0])
    qy = float(point[2])
    return math.hypot(qx, qy) - float(radii[1])


def _sd_capped_torus(point, half_angle, major_radius, minor_radius):
    px = abs(float(point[0]))
    py = float(point[1])
    pz = float(point[2])
    sc = (math.sin(float(half_angle)), math.cos(float(half_angle)))
    if (sc[1] * px) > (sc[0] * py):
        k = (px * sc[0]) + (py * sc[1])
    else:
        k = math.hypot(px, py)
    return math.sqrt((px * px) + (py * py) + (pz * pz) + (float(major_radius) * float(major_radius)) - (2.0 * float(major_radius) * k)) - float(minor_radius)


def _sd_cone(point, bottom_radius, top_radius, half_height):
    radial = math.hypot(float(point[0]), float(point[1]))
    qx = radial
    qy = float(point[2])
    k1x = float(top_radius)
    k1y = float(half_height)
    k2x = float(top_radius) - float(bottom_radius)
    k2y = 2.0 * float(half_height)
    cax = qx - min(qx, float(bottom_radius) if qy < 0.0 else float(top_radius))
    cay = abs(qy) - float(half_height)
    denom = max((k2x * k2x) + (k2y * k2y), 1.0e-12)
    t = _clamp((((k1x - qx) * k2x) + ((k1y - qy) * k2y)) / denom, 0.0, 1.0)
    cbx = qx - k1x + (k2x * t)
    cby = qy - k1y + (k2y * t)
    sign = -1.0 if (cbx < 0.0 and cay < 0.0) else 1.0
    return sign * math.sqrt(min((cax * cax) + (cay * cay), (cbx * cbx) + (cby * cby)))


def _sd_advanced_cone(point, bottom_radius, top_radius, half_height, bevel, bevel_mode):
    radial = math.hypot(float(point[0]), float(point[1]))
    edge_x = float(top_radius) - float(bottom_radius)
    edge_y = 2.0 * float(half_height)
    edge_len = max(math.hypot(edge_x, edge_y), 1.0e-12)
    dside = ((edge_y * (radial - float(bottom_radius))) - (edge_x * (float(point[2]) + float(half_height)))) / edge_len
    dz = abs(float(point[2])) - float(half_height)
    edge_radius = min(max(float(bevel), 0.0), max(float(half_height) - 0.001, 0.0))
    if edge_radius > 0.001:
        if bevel_mode == "SMOOTH":
            dd0 = dside + edge_radius
            dd1 = dz + edge_radius
            return min(max(dd0, dd1), 0.0) + math.hypot(max(dd0, 0.0), max(dd1, 0.0)) - edge_radius
        base = max(dside, dz)
        cham = (dside + dz + edge_radius) * 0.70710678
        d_value = max(base, cham)
        if d_value <= 0.0:
            return d_value
        if dside <= 0.0 and dz <= 0.0:
            return cham
        if dz <= -edge_radius:
            return dside
        if dside <= -edge_radius:
            return dz
        tc2 = (-dside + dz + edge_radius) / (2.0 * edge_radius)
        if tc2 <= 0.0:
            return math.hypot(dside, dz + edge_radius)
        if tc2 >= 1.0:
            return math.hypot(dside + edge_radius, dz)
        return cham
    return _sd_cone(point, bottom_radius, top_radius, half_height)


def _sd_round_cone(point, bottom_radius, top_radius, half_height):
    radial = math.hypot(float(point[0]), float(point[1]))
    total_height = max(2.0 * float(half_height), 1.0e-6)
    qx = radial
    qy = float(point[2]) + float(half_height)
    if abs(float(bottom_radius) - float(top_radius)) >= total_height:
        return (math.hypot(qx, qy) - float(bottom_radius)) if float(bottom_radius) >= float(top_radius) else (math.hypot(qx, qy - total_height) - float(top_radius))
    slope = (float(bottom_radius) - float(top_radius)) / total_height
    axis = math.sqrt(max(1.0 - slope * slope, 1.0e-12))
    k = (-slope * qx) + (axis * qy)
    if k <= 0.0:
        return math.hypot(qx, qy) - float(bottom_radius)
    if k >= axis * total_height:
        return math.hypot(qx, qy - total_height) - float(top_radius)
    return (axis * qx) + (slope * qy) - float(bottom_radius)


def _capsule_end_radii(radius, taper):
    tap_top = max(float(taper), 0.0)
    tap_bottom = max(-float(taper), 0.0)
    return max(float(radius) * max(1.0 - tap_bottom, 0.0), 0.0), max(float(radius) * max(1.0 - tap_top, 0.0), 0.0)


def _sd_capsule(point, radius, half_height, taper=0.0):
    bottom_radius, top_radius = _capsule_end_radii(radius, taper)
    return _sd_round_cone(point, bottom_radius, top_radius, half_height)


def _sd_regular_polygon_2d(point, radius, sides):
    sides = max(3, int(sides))
    angle_step = math.pi / float(sides)
    inner_radius = float(radius) * math.cos(angle_step)
    half_edge = float(radius) * math.sin(angle_step)
    x = -float(point[1])
    y = float(point[0])
    sector = angle_step * math.floor((math.atan2(y, x) + angle_step) / angle_step / 2.0) * 2.0
    cos_sector = math.cos(sector)
    sin_sector = math.sin(sector)
    px = (cos_sector * x) + (sin_sector * y)
    py = (-sin_sector * x) + (cos_sector * y)
    return math.hypot(px - inner_radius, py - _clamp(py, -half_edge, half_edge)) * (1.0 if (px - inner_radius) >= 0.0 else -1.0)


def _sd_star_polygon_2d(point, radius, sides, star):
    if float(star) < 0.001:
        return _sd_regular_polygon_2d(point, radius, sides)
    sides = max(3, int(sides))
    angle_step = math.pi / float(sides)
    inner_radius = float(radius) * math.cos(angle_step) * max(1.0 - float(star), 0.01)
    angle = math.atan2(float(point[1]), float(point[0]))
    sector = math.floor((angle + angle_step) / (2.0 * angle_step)) * 2.0 * angle_step
    cos_sector = math.cos(sector)
    sin_sector = math.sin(sector)
    qx = (cos_sector * float(point[0])) + (sin_sector * float(point[1]))
    qy = abs((-sin_sector * float(point[0])) + (cos_sector * float(point[1])))
    ax = float(radius)
    ay = 0.0
    bx = inner_radius * math.cos(angle_step)
    by = inner_radius * math.sin(angle_step)
    abx = bx - ax
    aby = by - ay
    denom = max((abx * abx) + (aby * aby), 1.0e-12)
    t = _clamp((((qx - ax) * abx) + ((qy - ay) * aby)) / denom, 0.0, 1.0)
    projx = ax + (abx * t)
    projy = ay + (aby * t)
    dist = math.hypot(qx - projx, qy - projy)
    cross_value = abx * (qy - ay) - aby * (qx - ax)
    return -dist if cross_value > 0.0 else dist


def _sd_segment_2d(point, start, end):
    edge = end - start
    edge_len_sq = edge.dot(edge)
    if edge_len_sq <= 1.0e-12:
        return (point - start).length
    t = _clamp((point - start).dot(edge) / edge_len_sq, 0.0, 1.0)
    return (point - (start + edge * t)).length


def _sd_polygon_2d(point, points):
    if len(points) < 3:
        return _MISS_DISTANCE
    distance_value = _MISS_DISTANCE
    winding = 0
    point_2d = Vector((float(point[0]), float(point[1])))
    vertices = [Vector((float(vertex[0]), float(vertex[1]))) for vertex in points]
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        distance_value = min(distance_value, _sd_segment_2d(point_2d, start, end))
        edge = end - start
        rel = point_2d - start
        cross_value = (edge[0] * rel[1]) - (edge[1] * rel[0])
        if start[1] <= point_2d[1] and end[1] > point_2d[1] and cross_value > 0.0:
            winding += 1
        if start[1] > point_2d[1] and end[1] <= point_2d[1] and cross_value < 0.0:
            winding -= 1
    return -distance_value if winding != 0 else distance_value


def _cubic_bezier_point_2d(a, b, c, d, t):
    inv_t = 1.0 - t
    return (
        a * (inv_t * inv_t * inv_t)
        + b * (3.0 * inv_t * inv_t * t)
        + c * (3.0 * inv_t * t * t)
        + d * (t * t * t)
    )


def _sd_bezier_polygon_2d(point, control_points):
    if len(control_points) < 3:
        return _MISS_DISTANCE
    distance_value = _MISS_DISTANCE
    winding = 0
    point_2d = Vector((float(point[0]), float(point[1])))
    steps = 12
    for index, start in enumerate(control_points):
        end = control_points[(index + 1) % len(control_points)]
        p0 = Vector((float(start[0]), float(start[1])))
        p1 = Vector((float(start[4]), float(start[5])))
        p2 = Vector((float(end[2]), float(end[3])))
        p3 = Vector((float(end[0]), float(end[1])))
        a = p0
        for step in range(1, steps + 1):
            t = float(step) / float(steps)
            b = _cubic_bezier_point_2d(p0, p1, p2, p3, t)
            distance_value = min(distance_value, _sd_segment_2d(point_2d, a, b))
            edge = b - a
            rel = point_2d - a
            cross_value = (edge[0] * rel[1]) - (edge[1] * rel[0])
            if a[1] <= point_2d[1] and b[1] > point_2d[1] and cross_value > 0.0:
                winding += 1
            if a[1] > point_2d[1] and b[1] <= point_2d[1] and cross_value < 0.0:
                winding -= 1
            a = b
    return -distance_value if winding != 0 else distance_value


def _sd_prism(distance_2d, z_value, half_height):
    dz = abs(float(z_value)) - float(half_height)
    outside = math.hypot(max(float(distance_2d), 0.0), max(dz, 0.0))
    inside = min(max(float(distance_2d), dz), 0.0)
    return outside + inside


def _sd_advanced_ngon(point, radius, half_height, sides, corner, edge_top, edge_bottom, taper, edge_mode, star):
    taper_h = max(float(half_height), 0.001)
    zn = _clamp(float(point[2]) / taper_h, -1.0, 1.0)
    mix_value = (zn + 1.0) * 0.5
    tap_top = max(float(taper), 0.0)
    tap_bottom = max(-float(taper), 0.0)
    tap_factor = max(1.0 - tap_top * mix_value - tap_bottom * (1.0 - mix_value), 0.001)
    scaled_radius = float(radius) * tap_factor
    slope = float(radius) * (tap_top + tap_bottom) / (2.0 * taper_h)
    lipschitz = math.sqrt(1.0 + slope * slope)
    angle_step = math.pi / float(max(3, int(sides)))
    apothem = scaled_radius * math.cos(angle_step)
    bevel_radius = float(corner) * apothem
    inner_radius = scaled_radius - (bevel_radius / max(math.cos(angle_step), 0.001))
    point_xy = Vector((float(point[0]), float(point[1])))
    d2d = (_sd_star_polygon_2d(point_xy, inner_radius, sides, star) if float(star) > 0.001 else _sd_regular_polygon_2d(point_xy, inner_radius, sides)) - bevel_radius
    dz = abs(float(point[2])) - float(half_height)
    edge_radius = (float(edge_top) if float(point[2]) > 0.0 else float(edge_bottom)) * min(apothem, float(half_height))
    if edge_radius > 0.001:
        if edge_mode == "SMOOTH":
            dd0 = d2d + edge_radius
            dd1 = dz + edge_radius
            return (min(max(dd0, dd1), 0.0) + math.hypot(max(dd0, 0.0), max(dd1, 0.0)) - edge_radius) / lipschitz
        base = max(d2d, dz)
        cham = (d2d + dz + edge_radius) * 0.70710678
        d_value = max(base, cham)
        if d_value <= 0.0:
            return d_value / lipschitz
        if d2d <= 0.0 and dz <= 0.0:
            return cham / lipschitz
        if dz <= -edge_radius:
            return d2d / lipschitz
        if d2d <= -edge_radius:
            return dz / lipschitz
        tc2 = (-d2d + dz + edge_radius) / (2.0 * edge_radius)
        if tc2 <= 0.0:
            return math.hypot(d2d, dz + edge_radius) / lipschitz
        if tc2 >= 1.0:
            return math.hypot(d2d + edge_radius, dz) / lipschitz
        return cham / lipschitz
    return _sd_prism(d2d, float(point[2]), float(half_height)) / lipschitz


def _sd_advanced_polygon(point, half_height, points, edge_top, edge_bottom, taper, edge_mode):
    taper_h = max(float(half_height), 0.001)
    zn = _clamp(float(point[2]) / taper_h, -1.0, 1.0)
    mix_value = (zn + 1.0) * 0.5
    tap_top = max(float(taper), 0.0)
    tap_bottom = max(-float(taper), 0.0)
    tap_factor = max(1.0 - tap_top * mix_value - tap_bottom * (1.0 - mix_value), 0.001)
    slope = (tap_top + tap_bottom) / (2.0 * taper_h)
    lipschitz = math.sqrt(1.0 + slope * slope)
    d2d = _sd_polygon_2d(Vector((float(point[0]) / tap_factor, float(point[1]) / tap_factor)), points) * tap_factor
    dz = abs(float(point[2])) - float(half_height)
    edge_radius = (float(edge_top) if float(point[2]) > 0.0 else float(edge_bottom)) * float(half_height)
    if edge_radius > 0.001:
        if edge_mode == "SMOOTH":
            dd0 = d2d + edge_radius
            dd1 = dz + edge_radius
            return (min(max(dd0, dd1), 0.0) + math.hypot(max(dd0, 0.0), max(dd1, 0.0)) - edge_radius) / lipschitz
        base = max(d2d, dz)
        cham = (d2d + dz + edge_radius) * 0.70710678
        d_value = max(base, cham)
        if d_value <= 0.0:
            return d_value / lipschitz
        if d2d <= 0.0 and dz <= 0.0:
            return cham / lipschitz
        if dz <= -edge_radius:
            return d2d / lipschitz
        if d2d <= -edge_radius:
            return dz / lipschitz
        tc2 = (-d2d + dz + edge_radius) / (2.0 * edge_radius)
        if tc2 <= 0.0:
            return math.hypot(d2d, dz + edge_radius) / lipschitz
        if tc2 >= 1.0:
            return math.hypot(d2d + edge_radius, dz) / lipschitz
        return cham / lipschitz
    return _sd_prism(d2d, float(point[2]), float(half_height)) / lipschitz


def _sd_advanced_bezier_polygon(point, half_height, control_points, edge_top, edge_bottom, taper, edge_mode):
    taper_h = max(float(half_height), 0.001)
    zn = _clamp(float(point[2]) / taper_h, -1.0, 1.0)
    mix_value = (zn + 1.0) * 0.5
    tap_top = max(float(taper), 0.0)
    tap_bottom = max(-float(taper), 0.0)
    tap_factor = max(1.0 - tap_top * mix_value - tap_bottom * (1.0 - mix_value), 0.001)
    slope = (tap_top + tap_bottom) / (2.0 * taper_h)
    lipschitz = math.sqrt(1.0 + slope * slope)
    d2d = _sd_bezier_polygon_2d(Vector((float(point[0]) / tap_factor, float(point[1]) / tap_factor)), control_points) * tap_factor
    dz = abs(float(point[2])) - float(half_height)
    edge_radius = (float(edge_top) if float(point[2]) > 0.0 else float(edge_bottom)) * float(half_height)
    if edge_radius > 0.001:
        if edge_mode == "SMOOTH":
            dd0 = d2d + edge_radius
            dd1 = dz + edge_radius
            return (min(max(dd0, dd1), 0.0) + math.hypot(max(dd0, 0.0), max(dd1, 0.0)) - edge_radius) / lipschitz
        base = max(d2d, dz)
        cham = (d2d + dz + edge_radius) * 0.70710678
        d_value = max(base, cham)
        if d_value <= 0.0:
            return d_value / lipschitz
        if d2d <= 0.0 and dz <= 0.0:
            return cham / lipschitz
        if dz <= -edge_radius:
            return d2d / lipschitz
        if d2d <= -edge_radius:
            return dz / lipschitz
        tc2 = (-d2d + dz + edge_radius) / (2.0 * edge_radius)
        if tc2 <= 0.0:
            return math.hypot(d2d, dz + edge_radius) / lipschitz
        if tc2 >= 1.0:
            return math.hypot(d2d + edge_radius, dz) / lipschitz
        return cham / lipschitz
    return _sd_prism(d2d, float(point[2]), float(half_height)) / lipschitz


def eval_primitive(compiled, primitive_index, world_point):
    primitive_specs = compiled.get("primitive_specs", ())
    primitive_entries = compiled.get("primitive_entries", ())
    if primitive_index < 0 or primitive_index >= len(primitive_specs) or primitive_index >= len(primitive_entries):
        return _MISS_DISTANCE

    spec = primitive_specs[primitive_index]
    entry = primitive_entries[primitive_index]
    local_point, distance_scale = _primitive_local_point(spec, entry, world_point)
    primitive_type = str(spec.get("primitive_type", "sphere") or "sphere")
    meta = tuple(float(value) for value in spec.get("meta", (0.0, 0.5, 0.0, 0.0)))
    extra = dict(spec.get("extra", {}))
    scale = _safe_vec(spec.get("scale", (1.0, 1.0, 1.0)))
    min_scale = max(min(float(scale[0]), float(scale[1]), float(scale[2])), 1.0e-6)
    raw_bevel = max(float(extra.get("bevel", 0.0)), 0.0)
    bevel = max(float(extra.get("effective_bevel", raw_bevel)), 0.0)
    if primitive_type == "sphere":
        return (_sd_ellipsoid(local_point, Vector((meta[1] * scale[0], meta[1] * scale[1], meta[1] * scale[2]))) - bevel) * distance_scale
    if primitive_type == "box":
        distance_value = _sd_advanced_box(
            local_point,
            Vector((meta[1] * scale[0], meta[2] * scale[1], meta[3] * scale[2])),
            tuple(float(value) for value in extra.get("box_corners", (0.0, 0.0, 0.0, 0.0))),
            float(extra.get("box_edge_top", 0.0)),
            float(extra.get("box_edge_bottom", 0.0)),
            float(extra.get("box_taper", 0.0)),
            str(extra.get("box_corner_mode", "SMOOTH") or "SMOOTH"),
            str(extra.get("box_edge_mode", "SMOOTH") or "SMOOTH"),
        ) if any(abs(float(extra.get(name, 0.0))) > 1.0e-6 for name in ("box_edge_top", "box_edge_bottom", "box_taper")) or any(abs(float(value)) > 1.0e-6 for value in extra.get("box_corners", (0.0, 0.0, 0.0, 0.0))) else _sd_box(local_point, Vector((meta[1] * scale[0], meta[2] * scale[1], meta[3] * scale[2])))
        return (distance_value - bevel) * distance_scale
    if primitive_type == "cylinder":
        scaled_point = Vector((float(local_point[0]) / float(scale[0]), float(local_point[1]) / float(scale[1]), float(local_point[2]) / float(scale[2])))
        bevel_top = max(float(extra.get("cylinder_bevel_top", raw_bevel)), 0.0)
        bevel_bottom = max(float(extra.get("cylinder_bevel_bottom", raw_bevel)), 0.0)
        advanced = abs(float(extra.get("cylinder_taper", 0.0))) > 1.0e-6 or bevel_top > 1.0e-6 or bevel_bottom > 1.0e-6
        distance_value = _sd_advanced_cylinder(scaled_point, meta[1], meta[2], bevel_top, bevel_bottom, float(extra.get("cylinder_taper", 0.0)), str(extra.get("cylinder_bevel_mode", "SMOOTH") or "SMOOTH")) if advanced else _sd_cylinder(scaled_point, meta[1], meta[2])
        return distance_value * min_scale * distance_scale
    if primitive_type == "torus":
        scaled_point = Vector((float(local_point[0]) / float(scale[0]), float(local_point[1]) / float(scale[1]), float(local_point[2]) / float(scale[2])))
        angle = max(0.0, min(float(extra.get("torus_angle", 6.283185307179586)), 6.283185307179586))
        distance_value = _sd_capped_torus(scaled_point, angle * 0.5, meta[1], meta[2]) if (6.283185307179586 - angle) > 1.0e-6 else _sd_torus(scaled_point, (meta[1], meta[2]))
        return (distance_value - bevel) * min_scale * distance_scale
    if primitive_type == "cone":
        scaled_point = Vector((float(local_point[0]) / float(scale[0]), float(local_point[1]) / float(scale[1]), float(local_point[2]) / float(scale[2])))
        distance_value = _sd_advanced_cone(scaled_point, meta[1], meta[2], meta[3], raw_bevel, str(extra.get("cone_bevel_mode", "SMOOTH") or "SMOOTH")) if raw_bevel > 1.0e-6 else _sd_cone(scaled_point, meta[1], meta[2], meta[3])
        return distance_value * min_scale * distance_scale
    if primitive_type == "capsule":
        scaled_point = Vector((float(local_point[0]) / float(scale[0]), float(local_point[1]) / float(scale[1]), float(local_point[2]) / float(scale[2])))
        return _sd_capsule(scaled_point, meta[1], meta[2], float(extra.get("capsule_taper", 0.0))) * min_scale * distance_scale
    if primitive_type == "ngon":
        scaled_point = Vector((float(local_point[0]) / float(scale[0]), float(local_point[1]) / float(scale[1]), float(local_point[2]) / float(scale[2])))
        advanced = any(abs(float(extra.get(name, 0.0))) > 1.0e-6 for name in ("ngon_corner", "ngon_edge_top", "ngon_edge_bottom", "ngon_taper", "ngon_star"))
        distance_value = _sd_advanced_ngon(scaled_point, meta[1], meta[2], int(round(meta[3])), float(extra.get("ngon_corner", 0.0)), float(extra.get("ngon_edge_top", 0.0)), float(extra.get("ngon_edge_bottom", 0.0)), float(extra.get("ngon_taper", 0.0)), str(extra.get("ngon_edge_mode", "SMOOTH") or "SMOOTH"), float(extra.get("ngon_star", 0.0))) if advanced else _sd_prism(_sd_regular_polygon_2d(scaled_point.xy, meta[1], int(round(meta[3]))), scaled_point[2], meta[2])
        return (distance_value - bevel) * min_scale * distance_scale
    if primitive_type == "polygon":
        scaled_point = Vector((float(local_point[0]) / float(scale[0]), float(local_point[1]) / float(scale[1]), float(local_point[2]) / float(scale[2])))
        polygon_points = spec.get("polygon_points", ())
        advanced = any(abs(float(extra.get(name, 0.0))) > 1.0e-6 for name in ("polygon_edge_top", "polygon_edge_bottom", "polygon_taper"))
        distance_value = _sd_advanced_polygon(scaled_point, meta[3], polygon_points, float(extra.get("polygon_edge_top", 0.0)), float(extra.get("polygon_edge_bottom", 0.0)), float(extra.get("polygon_taper", 0.0)), str(extra.get("polygon_edge_mode", "SMOOTH") or "SMOOTH")) if advanced else _sd_prism(_sd_polygon_2d(scaled_point.xy, polygon_points), scaled_point[2], meta[3])
        return (distance_value - bevel) * min_scale * distance_scale
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
    hit_owner = sdf_tree._proxy_transform_owner(hit_node) if hit_node is not None else None
    return {
        "object": hit_object,
        "node": hit_owner or hit_node,
        "point": hit_point,
        "compiled": compiled,
    }
