from contextlib import contextmanager
import uuid

import bpy
import nodeitems_utils
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Node, NodeSocket, NodeTree
from mathutils import Euler, Matrix, Vector
from nodeitems_utils import NodeCategory, NodeItem

from .. import runtime


NODE_CATEGORY_ID = "MATHOPS_V2_SDF_NODES"
TRANSFORM_SOCKET_IDNAME = "MathOPSV2TransformSocket"
TRANSFORM_NODE_IDNAME = "MathOPSV2TransformNode"
BREAK_TRANSFORM_NODE_IDNAME = "MathOPSV2BreakTransformNode"
MIRROR_NODE_IDNAME = "MathOPSV2MirrorNode"
ARRAY_NODE_IDNAME = "MathOPSV2ArrayNode"

_PRIMITIVE_TYPE_TO_ID = {
    "sphere": 0.0,
    "box": 1.0,
    "cylinder": 2.0,
    "torus": 3.0,
    "cone": 4.0,
    "capsule": 5.0,
    "ngon": 6.0,
    "polygon": 7.0,
}

_CSG_OPERATION_TO_ID = {
    "UNION": 1.0,
    "SUBTRACT": 2.0,
    "INTERSECT": 3.0,
}

_NODE_PRIMITIVE_ITEMS = (
    ("sphere", "Sphere", "Sphere SDF proxy"),
    ("box", "Box", "Box SDF proxy"),
    ("cylinder", "Cylinder", "Cylinder SDF proxy"),
    ("torus", "Torus", "Torus SDF proxy"),
    ("cone", "Cone", "Cone SDF proxy"),
    ("capsule", "Capsule", "Capsule SDF proxy"),
    ("ngon", "N-Gon", "Regular polygon prism SDF proxy"),
    ("polygon", "Polygon", "Polygon prism SDF proxy"),
)

_NODE_PRIMITIVE_LABELS = {
    identifier: label for identifier, label, _description in _NODE_PRIMITIVE_ITEMS
}

_IDENTITY_LOCATION = (0.0, 0.0, 0.0)
_IDENTITY_ROTATION = (0.0, 0.0, 0.0)
_IDENTITY_SCALE = (1.0, 1.0, 1.0)
_BEZIER_CIRCLE_HANDLE = 0.5522847498307936
_DEFAULT_POLYGON_POINTS = (
    (0.0, 1.0),
    (1.0, 0.0),
    (0.0, -1.0),
    (-1.0, 0.0),
)

BLEND_MODE_ITEMS = (
    ("SMOOTH", "Smooth", "Smooth rounding"),
    ("CHAMFER", "Chamfer", "Chamfer 45-degree cut"),
)

POLYGON_INTERPOLATION_ITEMS = (
    ("VECTOR", "Vector", "Straight segments between control points"),
    ("BEZIER", "Bezier", "Bezier interpolation with editable handles"),
)

_DIRECT_FLOAT_ATTRS = (
    "bevel",
    "cylinder_taper",
    "capsule_taper",
    "torus_angle",
    "box_edge_top",
    "box_edge_bottom",
    "box_taper",
    "ngon_corner",
    "ngon_edge_top",
    "ngon_edge_bottom",
    "ngon_taper",
    "ngon_star",
    "polygon_edge_top",
    "polygon_edge_bottom",
    "polygon_taper",
    "polygon_line_thickness",
)

_DIRECT_VECTOR_ATTRS = (
    "box_corners",
)

_DIRECT_INT_ATTRS = (
    "ngon_sides",
)

_DIRECT_BOOL_ATTRS = (
    "polygon_is_line",
)

_DIRECT_ENUM_ATTRS = (
    "cylinder_bevel_mode",
    "cone_bevel_mode",
    "box_corner_mode",
    "box_edge_mode",
    "ngon_edge_mode",
    "polygon_interpolation",
    "polygon_edge_mode",
)

_MIRROR_AXIS_X = 1
_MIRROR_AXIS_Y = 2
_MIRROR_AXIS_Z = 4
_MIRROR_SIDE_X = 8
_MIRROR_SIDE_Y = 16
_MIRROR_SIDE_Z = 32
_MIRROR_WARP_PACK_SCALE = 256
_WARP_ROWS_PER_ENTRY = 9

_WARP_KIND_MIRROR = 1
_WARP_KIND_GRID = 2
_WARP_KIND_RADIAL = 3

_ARRAY_MODE_GRID = "GRID"
_ARRAY_MODE_RADIAL = "RADIAL"

_SOCKET_TRANSFORM = "Transform"
_SOCKET_TRANSFORM_OFFSET = "Transform Offset"
_SOCKET_LOCATION = "Location"
_SOCKET_ROTATION = "Rotation"
_SOCKET_SCALE = "Scale"
_SOCKET_RADIUS = "Radius"
_SOCKET_TOP_RADIUS = "Top Radius"
_SOCKET_BOTTOM_RADIUS = "Bottom Radius"
_SOCKET_HALF_SIZE = "Half Size"
_SOCKET_HEIGHT = "Height"
_SOCKET_MAJOR_RADIUS = "Major Radius"
_SOCKET_MINOR_RADIUS = "Minor Radius"
_SOCKET_BLEND = "Blend"

_object_node_updates_suppressed = 0
_polygon_point_item_updates_suppressed = 0
_branch_frame_sync_suppressed = 0
_object_node_free_cleanup_suppressed = 0
_BRANCH_FRAME_TAG = "_mathops_v2_branch_frame"
_polygon_sample_cache = {}


def _tag_redraw(_self=None, context=None):
    runtime.clear_error()
    runtime.note_interaction()
    runtime.tag_redraw(context)


def default_polygon_points():
    return tuple((float(point[0]), float(point[1]), 0.0) for point in _DEFAULT_POLYGON_POINTS)


def default_polygon_control_points():
    handle = float(_BEZIER_CIRCLE_HANDLE)
    return (
        (0.0, 1.0, 0.0, -handle, 1.0, handle, 1.0),
        (1.0, 0.0, 0.0, 1.0, handle, 1.0, -handle),
        (0.0, -1.0, 0.0, handle, -1.0, -handle, -1.0),
        (-1.0, 0.0, 0.0, -1.0, -handle, -1.0, handle),
    )


def _polygon_interpolation(owner):
    return str(getattr(owner, "polygon_interpolation", "VECTOR") or "VECTOR").strip().upper()


def _polygon_min_point_count(owner):
    return 2 if bool(getattr(owner, "polygon_is_line", False)) else 3


def polygon_control_point_data(owner, ensure_default=False):
    primitive_type = str(getattr(owner, "primitive_type", "") or "").strip().lower()
    raw_points = getattr(owner, "polygon_points", ()) or ()
    points = []
    for point in raw_points:
        co = getattr(point, "co", point)
        if hasattr(point, "corner"):
            corner = getattr(point, "corner", 0.0)
        elif len(point) > 2:
            corner = point[2]
        else:
            corner = 0.0
        if hasattr(point, "handle_left"):
            handle_left = getattr(point, "handle_left", co)
        elif len(point) > 4:
            handle_left = (point[3], point[4])
        else:
            handle_left = co
        if hasattr(point, "handle_right"):
            handle_right = getattr(point, "handle_right", co)
        elif len(point) > 6:
            handle_right = (point[5], point[6])
        else:
            handle_right = co
        try:
            points.append(
                (
                    float(co[0]),
                    float(co[1]),
                    float(corner),
                    float(handle_left[0]),
                    float(handle_left[1]),
                    float(handle_right[0]),
                    float(handle_right[1]),
                )
            )
        except Exception:
            continue
    if ensure_default and primitive_type == "polygon" and len(points) < _polygon_min_point_count(owner):
        return default_polygon_control_points()
    return tuple(points)


def polygon_point_data(owner, ensure_default=False):
    return tuple((point[0], point[1], point[2]) for point in polygon_control_point_data(owner, ensure_default=ensure_default))


def polygon_point_coords(owner, ensure_default=False):
    return tuple((point[0], point[1]) for point in polygon_point_data(owner, ensure_default=ensure_default))


def set_polygon_points(owner, coords):
    points = getattr(owner, "polygon_points", None)
    if points is None or not hasattr(points, "add"):
        return False

    current = polygon_control_point_data(owner)
    target = []
    for index, point in enumerate(coords):
        x = float(point[0])
        y = float(point[1])
        corner = float(point[2]) if len(point) > 2 else 0.0
        if len(point) > 6:
            handle_left = (float(point[3]), float(point[4]))
            handle_right = (float(point[5]), float(point[6]))
        elif index < len(current):
            dx = x - float(current[index][0])
            dy = y - float(current[index][1])
            handle_left = (float(current[index][3]) + dx, float(current[index][4]) + dy)
            handle_right = (float(current[index][5]) + dx, float(current[index][6]) + dy)
        else:
            handle_left = (x, y)
            handle_right = (x, y)
        target.append((x, y, corner, float(handle_left[0]), float(handle_left[1]), float(handle_right[0]), float(handle_right[1])))
    target = tuple(target)
    if current == target:
        return False

    with suppress_polygon_point_item_updates():
        while len(points) > 0:
            points.remove(len(points) - 1)
        for x, y, corner, handle_left_x, handle_left_y, handle_right_x, handle_right_y in target:
            item = points.add()
            item.co = (float(x), float(y))
            if hasattr(item, "corner"):
                item.corner = float(corner)
            if hasattr(item, "handle_left"):
                item.handle_left = (float(handle_left_x), float(handle_left_y))
            if hasattr(item, "handle_right"):
                item.handle_right = (float(handle_right_x), float(handle_right_y))
    active_index = min(max(int(getattr(owner, "polygon_active_index", 0)), 0), max(len(points) - 1, 0))
    if hasattr(owner, "polygon_active_index") and int(getattr(owner, "polygon_active_index", 0)) != active_index:
        owner.polygon_active_index = active_index
    return True


def _expanded_polygon_line_points(points, half_thickness):
    count = len(points)
    if count < 2 or half_thickness <= 0.0:
        return points

    vertices = [Vector((point[0], point[1])) for point in points]
    left_side = [Vector((0.0, 0.0)) for _index in range(count)]
    right_side = [Vector((0.0, 0.0)) for _index in range(count)]

    for index in range(count):
        normal_prev = Vector((0.0, 0.0))
        normal_next = Vector((0.0, 0.0))
        if index > 0:
            delta = (vertices[index] - vertices[index - 1]).normalized()
            normal_prev = Vector((-delta.y, delta.x))
        if index < count - 1:
            delta = (vertices[index + 1] - vertices[index]).normalized()
            normal_next = Vector((-delta.y, delta.x))

        if index == 0:
            left_side[index] = vertices[index] + normal_next * half_thickness
            right_side[index] = vertices[index] - normal_next * half_thickness
        elif index == count - 1:
            left_side[index] = vertices[index] + normal_prev * half_thickness
            right_side[index] = vertices[index] - normal_prev * half_thickness
        else:
            miter = (normal_prev + normal_next)
            if miter.length < 1.0e-6:
                miter = normal_next.copy()
            else:
                miter.normalize()
            dot_value = miter.dot(normal_prev)
            if abs(dot_value) < 0.1:
                dot_value = 0.1 if dot_value >= 0.0 else -0.1
            miter_length = max(min(half_thickness / dot_value, half_thickness * 3.0), -half_thickness * 3.0)
            left_side[index] = vertices[index] + miter * miter_length
            right_side[index] = vertices[index] - miter * miter_length

    expanded = []
    for point in left_side:
        expanded.append((float(point.x), float(point.y), 0.0))
    for point in reversed(right_side):
        expanded.append((float(point.x), float(point.y), 0.0))
    return tuple(expanded)


def _bezier_point(p0, p1, p2, p3, t):
    inv_t = 1.0 - t
    return (
        (inv_t * inv_t * inv_t * p0)
        + (3.0 * inv_t * inv_t * t * p1)
        + (3.0 * inv_t * t * t * p2)
        + (t * t * t * p3)
    )


def _sample_polygon_control_points(control_points, interpolation, is_line=False, resolution=1):
    if not control_points:
        return ()

    if interpolation != "BEZIER" or len(control_points) < 2:
        return tuple((float(point[0]), float(point[1]), float(point[2])) for point in control_points)

    steps = max(int(resolution), 1)
    count = len(control_points)
    segment_count = max(count - 1, 0) if is_line else count
    sampled = [(float(control_points[0][0]), float(control_points[0][1]), float(control_points[0][2]))]
    for index in range(segment_count):
        start = control_points[index]
        end = control_points[index + 1] if is_line else control_points[(index + 1) % count]
        p0 = float(start[0])
        p1 = float(start[5])
        p2 = float(end[3])
        p3 = float(end[0])
        q0 = float(start[1])
        q1 = float(start[6])
        q2 = float(end[4])
        q3 = float(end[1])
        for step in range(1, steps + 1):
            if not is_line and index == segment_count - 1 and step == steps:
                continue
            t = float(step) / float(steps)
            sampled.append(
                (
                    float(_bezier_point(p0, p1, p2, p3, t)),
                    float(_bezier_point(q0, q1, q2, q3, t)),
                    0.0,
                )
            )
    return tuple(sampled)


def _polygon_curve_resolution(obj, interpolation="VECTOR"):
    try:
        obj = runtime.object_identity(obj)
        curve = None if obj is None else getattr(obj, "data", None)
        if curve is not None and hasattr(curve, "resolution_u"):
            return min(max(int(getattr(curve, "resolution_u", 1)), 1), 12)
    except ReferenceError:
        pass
    return 12 if str(interpolation or "VECTOR").strip().upper() == "BEZIER" else 1


def _polygon_sample_cache_key(owner, target):
    target_key = runtime.object_key(target)
    if target_key:
        return ("curve", target_key)
    return ("owner", runtime.safe_pointer(owner))


def effective_polygon_control_point_data(owner, ensure_default=False):
    target = runtime.object_identity(getattr(owner, "target", None))
    if runtime.is_polygon_curve_proxy(target):
        return polygon_curve_control_point_data(target, ensure_default=ensure_default)
    return polygon_control_point_data(owner, ensure_default=ensure_default)


def effective_polygon_point_data(owner, ensure_default=False):
    control_points = effective_polygon_control_point_data(owner, ensure_default=ensure_default)
    target = runtime.object_identity(getattr(owner, "target", None))
    if runtime.is_polygon_curve_proxy(target):
        interpolation = polygon_curve_interpolation(target)
        resolution = _polygon_curve_resolution(target, interpolation)
    else:
        interpolation = _polygon_interpolation(owner)
        resolution = 16 if interpolation == "BEZIER" else 1
    is_line = bool(getattr(owner, "polygon_is_line", False))
    line_thickness = max(float(getattr(owner, "polygon_line_thickness", 0.1)) * 0.5, 0.0)
    cache_key = _polygon_sample_cache_key(owner, target)
    signature = (control_points, interpolation, is_line, line_thickness, int(resolution))
    cached = _polygon_sample_cache.get(cache_key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    points = _sample_polygon_control_points(control_points, interpolation, is_line=is_line, resolution=resolution)
    if not is_line:
        result = points
    else:
        result = _expanded_polygon_line_points(points, line_thickness)
    if len(_polygon_sample_cache) > 256:
        _polygon_sample_cache.clear()
    _polygon_sample_cache[cache_key] = (signature, result)
    if not is_line:
        return points
    return result


def ensure_polygon_defaults(owner):
    primitive_type = str(getattr(owner, "primitive_type", "") or "").strip().lower()
    if primitive_type != "polygon":
        return False
    points = polygon_point_data(owner)
    if len(points) >= _polygon_min_point_count(owner):
        return False
    return set_polygon_points(owner, default_polygon_control_points())


def polygon_curve_control_point_data(obj, ensure_default=False):
    try:
        obj = runtime.object_identity(obj)
        if obj is None or getattr(obj, "type", "") != "CURVE":
            return default_polygon_control_points() if ensure_default else ()
        if hasattr(obj, "update_from_editmode") and str(getattr(obj, "mode", "") or "") == "EDIT":
            try:
                obj.update_from_editmode()
            except Exception:
                pass
        curve = getattr(obj, "data", None)
        splines = () if curve is None else getattr(curve, "splines", ())
        spline = None if len(splines) == 0 else splines[0]
        if spline is None:
            return default_polygon_control_points() if ensure_default else ()

        points = []
        if getattr(spline, "type", "") == "BEZIER":
            for point in getattr(spline, "bezier_points", ()):
                points.append(
                    (
                        float(point.co.x),
                        float(point.co.y),
                        0.0,
                        float(point.handle_left.x),
                        float(point.handle_left.y),
                        float(point.handle_right.x),
                        float(point.handle_right.y),
                    )
                )
        else:
            for point in getattr(spline, "points", ()):
                co = point.co
                weight = float(co[3]) if len(co) > 3 else 1.0
                if abs(weight) <= 1.0e-6:
                    weight = 1.0
                x = float(co[0] / weight)
                y = float(co[1] / weight)
                points.append((x, y, 0.0, x, y, x, y))
        if ensure_default and len(points) < 2:
            return default_polygon_control_points()
        return tuple(points)
    except ReferenceError:
        return default_polygon_control_points() if ensure_default else ()


def polygon_curve_point_data(obj, ensure_default=False):
    return tuple((point[0], point[1], point[2]) for point in polygon_curve_control_point_data(obj, ensure_default=ensure_default))


def polygon_curve_interpolation(obj):
    try:
        obj = runtime.object_identity(obj)
        if obj is None or getattr(obj, "type", "") != "CURVE":
            return "VECTOR"
        curve = getattr(obj, "data", None)
        splines = () if curve is None else getattr(curve, "splines", ())
        spline = None if len(splines) == 0 else splines[0]
        if spline is None:
            return "VECTOR"
        if getattr(spline, "type", "") != "BEZIER":
            return "VECTOR"
        for point in getattr(spline, "bezier_points", ()):
            if str(getattr(point, "handle_left_type", "VECTOR") or "VECTOR") != "VECTOR":
                return "BEZIER"
            if str(getattr(point, "handle_right_type", "VECTOR") or "VECTOR") != "VECTOR":
                return "BEZIER"
        return "VECTOR"
    except ReferenceError:
        return "VECTOR"


def polygon_curve_is_line(obj):
    try:
        obj = runtime.object_identity(obj)
        if obj is None or getattr(obj, "type", "") != "CURVE":
            return False
        curve = getattr(obj, "data", None)
        splines = () if curve is None else getattr(curve, "splines", ())
        spline = None if len(splines) == 0 else splines[0]
        if spline is None:
            return False
        return not bool(getattr(spline, "use_cyclic_u", True))
    except ReferenceError:
        return False


def _recenter_polygon_curve_proxy(obj, control_points, interpolation, is_line=False, resolution=1):
    sampled = _sample_polygon_control_points(control_points, interpolation, is_line=is_line, resolution=resolution)
    if not sampled:
        return False, control_points
    min_x = min(float(point[0]) for point in sampled)
    max_x = max(float(point[0]) for point in sampled)
    min_y = min(float(point[1]) for point in sampled)
    max_y = max(float(point[1]) for point in sampled)
    center = Vector(((min_x + max_x) * 0.5, (min_y + max_y) * 0.5, 0.0))
    if center.length <= 1.0e-6:
        return False, control_points
    shifted = []
    for point in control_points:
        shifted.append(
            (
                float(point[0]) - center.x,
                float(point[1]) - center.y,
                float(point[2]),
                float(point[3]) - center.x,
                float(point[4]) - center.y,
                float(point[5]) - center.x,
                float(point[6]) - center.y,
            )
        )
    world_offset = obj.matrix_world.to_3x3() @ center
    _apply_matrix_to_object(obj, Matrix.Translation(world_offset))
    return True, tuple(shifted)


def sync_polygon_curve_proxy(obj, coords=(), is_line=False, interpolation="VECTOR"):
    try:
        obj = runtime.object_identity(obj)
        if obj is None or getattr(obj, "type", "") != "CURVE":
            return False

        changed = False
        if obj.rotation_mode != "XYZ":
            obj.rotation_mode = "XYZ"
            changed = True
        if obj.show_name:
            obj.show_name = False
            changed = True
        if obj.show_in_front:
            obj.show_in_front = False
            changed = True
        if not obj.hide_render:
            obj.hide_render = True
            changed = True

        curve = getattr(obj, "data", None)
        if curve is None:
            return changed
        if getattr(curve, "dimensions", "3D") != "2D":
            curve.dimensions = "2D"
            changed = True
        if hasattr(curve, "fill_mode") and getattr(curve, "fill_mode", "FULL") != "NONE":
            curve.fill_mode = "NONE"
            changed = True
        interpolation = str(interpolation or "VECTOR").strip().upper()
        target_resolution = 1 if interpolation == "VECTOR" else 32
        if hasattr(curve, "resolution_u") and int(getattr(curve, "resolution_u", target_resolution)) != target_resolution:
            curve.resolution_u = target_resolution
            changed = True
        if hasattr(curve, "render_resolution_u") and int(getattr(curve, "render_resolution_u", target_resolution)) != target_resolution:
            curve.render_resolution_u = target_resolution
            changed = True

        target_points = tuple(
            (
                float(point[0]),
                float(point[1]),
                float(point[2]) if len(point) > 2 else 0.0,
                float(point[3]) if len(point) > 6 else float(point[0]),
                float(point[4]) if len(point) > 6 else float(point[1]),
                float(point[5]) if len(point) > 6 else float(point[0]),
                float(point[6]) if len(point) > 6 else float(point[1]),
            )
            for point in (coords or default_polygon_control_points())
        )
        if len(target_points) < 2:
            target_points = default_polygon_control_points()

        recentered, target_points = _recenter_polygon_curve_proxy(
            obj,
            target_points,
            interpolation,
            is_line=bool(is_line),
            resolution=target_resolution,
        )
        changed = recentered or changed

        splines = getattr(curve, "splines", None)
        if splines is None:
            return changed

        spline = None if len(splines) == 0 else splines[0]
        needs_rebuild = len(splines) != 1 or spline is None
        if not needs_rebuild:
            spline_type = str(getattr(spline, "type", "") or "")
            if spline_type == "BEZIER":
                needs_rebuild = len(getattr(spline, "bezier_points", ())) != len(target_points)
            elif spline_type == "POLY":
                needs_rebuild = len(getattr(spline, "points", ())) != len(target_points)
            else:
                needs_rebuild = True

        if needs_rebuild:
            for existing in tuple(splines):
                splines.remove(existing)
            spline = splines.new("BEZIER")
            if len(target_points) > 1:
                spline.bezier_points.add(len(target_points) - 1)
            changed = True

        if getattr(spline, "type", "") == "BEZIER":
            for point, target in zip(spline.bezier_points, target_points):
                target_co = (float(target[0]), float(target[1]), 0.0)
                if _float_seq_changed(tuple(point.co), target_co):
                    point.co = target_co
                    changed = True
                target_left = (float(target[3]), float(target[4]), 0.0)
                if _float_seq_changed(tuple(point.handle_left), target_left):
                    point.handle_left = target_left
                    changed = True
                target_right = (float(target[5]), float(target[6]), 0.0)
                if _float_seq_changed(tuple(point.handle_right), target_right):
                    point.handle_right = target_right
                    changed = True
                if interpolation == "VECTOR":
                    if str(getattr(point, "handle_left_type", "VECTOR") or "VECTOR") != "VECTOR":
                        point.handle_left_type = "VECTOR"
                        changed = True
                    if str(getattr(point, "handle_right_type", "VECTOR") or "VECTOR") != "VECTOR":
                        point.handle_right_type = "VECTOR"
                        changed = True
                else:
                    if str(getattr(point, "handle_left_type", "FREE") or "FREE") == "VECTOR":
                        point.handle_left_type = "FREE"
                        changed = True
                    if str(getattr(point, "handle_right_type", "FREE") or "FREE") == "VECTOR":
                        point.handle_right_type = "FREE"
                        changed = True
        else:
            for point, target in zip(spline.points, target_points):
                target_co = (float(target[0]), float(target[1]), 0.0, 1.0)
                if _float_seq_changed(tuple(point.co), target_co):
                    point.co = target_co
                    changed = True

        cyclic = not bool(is_line)
        if bool(getattr(spline, "use_cyclic_u", True)) != cyclic:
            spline.use_cyclic_u = cyclic
            changed = True
        return changed
    except ReferenceError:
        return False


@contextmanager
def suppress_object_node_updates():
    global _object_node_updates_suppressed
    _object_node_updates_suppressed += 1
    try:
        yield
    finally:
        _object_node_updates_suppressed -= 1


@contextmanager
def suppress_polygon_point_item_updates():
    global _polygon_point_item_updates_suppressed
    _polygon_point_item_updates_suppressed += 1
    try:
        yield
    finally:
        _polygon_point_item_updates_suppressed -= 1


@contextmanager
def suppress_branch_frame_sync():
    global _branch_frame_sync_suppressed
    _branch_frame_sync_suppressed += 1
    try:
        yield
    finally:
        _branch_frame_sync_suppressed -= 1


@contextmanager
def suppress_object_node_free_cleanup():
    global _object_node_free_cleanup_suppressed
    _object_node_free_cleanup_suppressed += 1
    try:
        yield
    finally:
        _object_node_free_cleanup_suppressed -= 1


def _detach_proxy_binding(node):
    with suppress_object_node_updates():
        node.target = None
        node.use_proxy = False


def _poll_sdf_proxy(_self, obj):
    return runtime.is_sdf_proxy(obj)


def _scene_for_tree(tree, context=None):
    if context is not None:
        scene = getattr(context, "scene", None)
        settings = None if scene is None else runtime.scene_settings(scene)
        if settings is not None and getattr(settings, "node_tree", None) == tree:
            return scene

    scenes = getattr(bpy.data, "scenes", ())
    for scene in scenes:
        settings = runtime.scene_settings(scene)
        if settings is not None and getattr(settings, "node_tree", None) == tree:
            return scene
    return None


def _auto_arrange_tree(tree, context=None):
    if tree is None or getattr(tree, "bl_idname", "") != runtime.TREE_IDNAME:
        return False
    from ..operators import node_arrange

    return bool(node_arrange.arrange_tree(tree, context=context))


def _node_absolute_location(node):
    location = getattr(node, "location", None)
    x = 0.0 if location is None else float(location.x)
    y = 0.0 if location is None else float(location.y)
    parent = getattr(node, "parent", None)
    while parent is not None:
        parent_location = getattr(parent, "location", None)
        if parent_location is not None:
            x += float(parent_location.x)
            y += float(parent_location.y)
        parent = getattr(parent, "parent", None)
    return x, y


def _set_node_parent_keep_absolute(node, parent):
    absolute_x, absolute_y = _node_absolute_location(node)
    parent_x, parent_y = (0.0, 0.0) if parent is None else _node_absolute_location(parent)
    node.parent = parent
    node.location = (absolute_x - parent_x, absolute_y - parent_y)


def _auto_branch_frames(tree):
    if tree is None:
        return []
    return [node for node in tree.nodes if getattr(node, "bl_idname", "") == "NodeFrame" and bool(node.get(_BRANCH_FRAME_TAG, False))]


def _safe_sync_branch_frames(tree):
    sync_branch_frames = globals().get("_sync_branch_frames")
    if sync_branch_frames is None:
        return False
    return bool(sync_branch_frames(tree))


def _mark_tree_dirty(tree, static=False, context=None):
    scene = _scene_for_tree(tree, context=context)
    if scene is None:
        return
    if static:
        runtime.mark_scene_static_dirty(scene)
    else:
        runtime.mark_scene_transform_dirty(scene)


def _update_object_node_transform(self, context):
    if _object_node_updates_suppressed > 0:
        return
    _ensure_object_node_sockets(self)
    ensure_object_node_id(self)
    transform_source = transform_source_node(self)
    _write_transform_to_node_source(
        self,
        tuple(float(component) for component in self.sdf_location),
        tuple(float(component) for component in self.sdf_rotation),
        tuple(float(component) for component in self.sdf_scale),
    )
    if transform_source is not None:
        _sync_tree_proxy_transforms(getattr(self, "id_data", None), source_node=transform_source)
    else:
        sync_node_to_proxy(self)
    _mark_tree_dirty(getattr(self, "id_data", None), static=False, context=context)
    _tag_redraw(self, context)


def _update_object_node_payload(self, context):
    if _object_node_updates_suppressed > 0:
        return
    ensure_polygon_defaults(self)
    _ensure_object_node_sockets(self)
    with suppress_object_node_updates():
        _set_socket_float_default(_input_socket(self, _SOCKET_RADIUS, MathOPSV2FloatSocket.bl_idname), getattr(self, "radius", 0.5))
        _set_socket_float_default(_input_socket(self, _SOCKET_TOP_RADIUS, MathOPSV2FloatSocket.bl_idname), _cone_top_radius_value(self))
        _set_socket_float_default(_input_socket(self, _SOCKET_BOTTOM_RADIUS, MathOPSV2FloatSocket.bl_idname), _cone_bottom_radius_value(self))
        _set_socket_vector_default(_input_socket(self, _SOCKET_HALF_SIZE, MathOPSV2VectorSocket.bl_idname), getattr(self, "size", (0.5, 0.5, 0.5)))
        _set_socket_float_default(_input_socket(self, _SOCKET_HEIGHT, MathOPSV2FloatSocket.bl_idname), getattr(self, "height", 1.0))
        _set_socket_float_default(_input_socket(self, _SOCKET_MAJOR_RADIUS, MathOPSV2FloatSocket.bl_idname), getattr(self, "major_radius", 0.75))
        _set_socket_float_default(_input_socket(self, _SOCKET_MINOR_RADIUS, MathOPSV2FloatSocket.bl_idname), getattr(self, "minor_radius", 0.25))
    ensure_object_node_id(self)
    sync_node_to_proxy(self)
    _mark_tree_dirty(getattr(self, "id_data", None), static=True, context=context)
    _tag_redraw(self, context)


def _update_object_node_target(self, context):
    if _object_node_updates_suppressed > 0:
        return
    self.use_proxy = self.target is not None
    _ensure_object_node_sockets(self)
    ensure_object_node_id(self)
    sync_node_to_proxy(self)
    _mark_tree_dirty(getattr(self, "id_data", None), static=True, context=context)
    _tag_redraw(self, context)


def _update_csg_node_data(self, context):
    _ensure_csg_node_sockets(self)
    _mark_tree_dirty(getattr(self, "id_data", None), static=True, context=context)
    _tag_redraw(self, context)


def _update_mirror_node_data(self, context):
    _ensure_mirror_node_sockets(self)
    _mark_tree_dirty(getattr(self, "id_data", None), static=True, context=context)
    _tag_redraw(self, context)


def _update_array_node_data(self, context):
    _ensure_array_node_sockets(self)
    _mark_tree_dirty(getattr(self, "id_data", None), static=True, context=context)
    _tag_redraw(self, context)


def _update_array_node_transform(self, context):
    if _object_node_updates_suppressed > 0:
        return
    _ensure_array_node_sockets(self)
    _mark_tree_dirty(getattr(self, "id_data", None), static=False, context=context)
    _tag_redraw(self, context)


def _update_transform_node_data(self, context):
    if _object_node_updates_suppressed > 0:
        return
    _ensure_transform_node_sockets(self)
    tree = getattr(self, "id_data", None)
    _sync_tree_proxy_transforms(tree, source_node=self)
    _mark_tree_dirty(tree, static=False, context=context)
    _tag_redraw(self, context)


def _property_is_explicit(owner, attribute):
    checker = getattr(owner, "is_property_set", None)
    if checker is None:
        return False
    try:
        return bool(checker(attribute))
    except Exception:
        return False


def _cone_top_radius_value(owner):
    return max(0.0, float(getattr(owner, "cone_top_radius", 0.0))) if _property_is_explicit(owner, "cone_top_radius") else 0.0


def _cone_bottom_radius_value(owner):
    if _property_is_explicit(owner, "cone_bottom_radius"):
        return max(0.0, float(getattr(owner, "cone_bottom_radius", 0.5)))
    return max(0.0, float(getattr(owner, "radius", 0.5)))


def _cylinder_bevel_top_value(owner):
    if _property_is_explicit(owner, "cylinder_bevel_top"):
        return max(0.0, float(getattr(owner, "cylinder_bevel_top", 0.0)))
    return max(0.0, float(getattr(owner, "bevel", 0.0)))


def _cylinder_bevel_bottom_value(owner):
    if _property_is_explicit(owner, "cylinder_bevel_bottom"):
        return max(0.0, float(getattr(owner, "cylinder_bevel_bottom", 0.0)))
    return max(0.0, float(getattr(owner, "bevel", 0.0)))


def _capsule_end_radii(radius, taper):
    tap_top, tap_bottom = _split_taper(taper)
    return max(float(radius) * max(1.0 - tap_bottom, 0.0), 0.0), max(float(radius) * max(1.0 - tap_top, 0.0), 0.0)


def _migrate_cylinder_bevel_fields(owner):
    if str(getattr(owner, "primitive_type", "") or "") != "cylinder":
        return
    if _property_is_explicit(owner, "cylinder_bevel_top") or _property_is_explicit(owner, "cylinder_bevel_bottom"):
        return
    legacy_bevel = max(0.0, float(getattr(owner, "bevel", 0.0)))
    if legacy_bevel <= 0.0:
        return
    setattr(owner, "cylinder_bevel_top", legacy_bevel)
    setattr(owner, "cylinder_bevel_bottom", legacy_bevel)


def _sync_object_node_parameter_attrs_from_sockets(node):
    with suppress_object_node_updates():
        node.radius = _float_socket_value(_input_socket(node, _SOCKET_RADIUS, MathOPSV2FloatSocket.bl_idname), getattr(node, "radius", 0.5), min_value=0.001)
        node.cone_top_radius = _float_socket_value(_input_socket(node, _SOCKET_TOP_RADIUS, MathOPSV2FloatSocket.bl_idname), _cone_top_radius_value(node), min_value=0.0)
        node.cone_bottom_radius = _float_socket_value(_input_socket(node, _SOCKET_BOTTOM_RADIUS, MathOPSV2FloatSocket.bl_idname), _cone_bottom_radius_value(node), min_value=0.0)
        node.size = _vector_socket_value(_input_socket(node, _SOCKET_HALF_SIZE, MathOPSV2VectorSocket.bl_idname), getattr(node, "size", (0.5, 0.5, 0.5)), _sanitized_scale)
        node.height = _float_socket_value(_input_socket(node, _SOCKET_HEIGHT, MathOPSV2FloatSocket.bl_idname), getattr(node, "height", 1.0), min_value=0.001)
        node.major_radius = _float_socket_value(_input_socket(node, _SOCKET_MAJOR_RADIUS, MathOPSV2FloatSocket.bl_idname), getattr(node, "major_radius", 0.75), min_value=0.001)
        node.minor_radius = _float_socket_value(_input_socket(node, _SOCKET_MINOR_RADIUS, MathOPSV2FloatSocket.bl_idname), getattr(node, "minor_radius", 0.25), min_value=0.001)


def _update_socket_value(self, context):
    if _object_node_updates_suppressed > 0:
        return
    node = getattr(self, "node", None)
    if node is None:
        return
    node_idname = getattr(node, "bl_idname", "")
    if node_idname == runtime.CSG_NODE_IDNAME and getattr(self, "name", "") == _SOCKET_BLEND:
        if float(getattr(self, "default_value", 0.0)) < 0.0:
            with suppress_object_node_updates():
                self.default_value = 0.0
    if node_idname == runtime.OBJECT_NODE_IDNAME:
        _sync_object_node_parameter_attrs_from_sockets(node)
        ensure_polygon_defaults(node)
        _ensure_object_node_sockets(node)
        ensure_object_node_id(node)
        sync_node_to_proxy(node)
        _mark_tree_dirty(getattr(node, "id_data", None), static=True, context=context)
        _tag_redraw(node, context)
        return
    if node_idname == TRANSFORM_NODE_IDNAME:
        _update_transform_node_data(node, context)
        return
    if node_idname == runtime.CSG_NODE_IDNAME:
        _update_csg_node_data(node, context)
        return
    if node_idname == MIRROR_NODE_IDNAME:
        _update_mirror_node_data(node, context)
        return
    if node_idname == ARRAY_NODE_IDNAME:
        _update_array_node_data(node, context)


class MathOPSV2SDFSocket(NodeSocket):
    bl_idname = "MathOPSV2SDFSocket"
    bl_label = "SDF"

    def draw(self, _context, layout, _node, text):
        layout.label(text=text or self.name)

    def draw_color(self, _context, _node):
        return (0.45, 0.65, 1.0, 1.0)


class MathOPSV2TransformSocket(NodeSocket):
    bl_idname = TRANSFORM_SOCKET_IDNAME
    bl_label = "Transform"

    def draw(self, _context, layout, _node, text):
        layout.label(text=text or self.name)

    def draw_color(self, _context, _node):
        return (0.66, 0.46, 0.92, 1.0)


class MathOPSV2FloatSocket(NodeSocket):
    bl_idname = "MathOPSV2FloatSocket"
    bl_label = "Float"

    default_value: FloatProperty(default=0.0, update=_update_socket_value)

    def draw(self, _context, layout, _node, text):
        if self.is_output or self.is_linked:
            layout.label(text=text or self.name)
            return
        layout.prop(self, "default_value", text=text or self.name)

    def draw_color(self, _context, _node):
        return (0.84, 0.72, 0.32, 1.0)


class MathOPSV2VectorSocket(NodeSocket):
    bl_idname = "MathOPSV2VectorSocket"
    bl_label = "Vector"

    default_value: FloatVectorProperty(
        size=3,
        default=_IDENTITY_LOCATION,
        subtype="XYZ",
        update=_update_socket_value,
    )

    def draw(self, _context, layout, _node, text):
        if self.is_output or self.is_linked:
            layout.label(text=text or self.name)
            return
        layout.prop(self, "default_value", text=text or self.name)

    def draw_color(self, _context, _node):
        return (0.36, 0.74, 0.74, 1.0)


class MathOPSV2EulerSocket(NodeSocket):
    bl_idname = "MathOPSV2EulerSocket"
    bl_label = "Rotation"

    default_value: FloatVectorProperty(
        size=3,
        default=_IDENTITY_ROTATION,
        subtype="EULER",
        update=_update_socket_value,
    )

    def draw(self, _context, layout, _node, text):
        if self.is_output or self.is_linked:
            layout.label(text=text or self.name)
            return
        layout.prop(self, "default_value", text=text or self.name)

    def draw_color(self, _context, _node):
        return (0.54, 0.64, 0.92, 1.0)


def _polygon_point_owner(point_item):
    id_data = getattr(point_item, "id_data", None)
    if id_data is None:
        return None
    try:
        path = point_item.path_from_id()
    except Exception:
        return None
    marker = ".polygon_points["
    if marker not in path:
        return None
    owner_path = path.split(marker, 1)[0]
    try:
        return id_data.path_resolve(owner_path)
    except Exception:
        return None


def _update_polygon_point_item(point_item, context):
    if _polygon_point_item_updates_suppressed > 0:
        return
    owner = _polygon_point_owner(point_item)
    if owner is None:
        return
    scene = getattr(context, "scene", None) or getattr(bpy.context, "scene", None)
    if getattr(owner, "bl_idname", "") == runtime.OBJECT_NODE_IDNAME:
        sync_node_to_proxy(owner, include_transform=False)
    else:
        source_tree_name = str(getattr(owner, "source_tree_name", "") or "")
        source_node_name = str(getattr(owner, "source_node_name", "") or "")
        if source_tree_name and source_node_name:
            tree = bpy.data.node_groups.get(source_tree_name)
            if tree is not None:
                node = tree.nodes.get(source_node_name)
                if node is not None:
                    sync_proxy_to_node(node)
    if scene is not None:
        runtime.mark_scene_static_dirty(scene)
    runtime.clear_error()
    runtime.note_interaction()
    runtime.tag_redraw(context)


class MathOPSV2PolygonPoint(bpy.types.PropertyGroup):
    co: FloatVectorProperty(size=2, default=(0.0, 0.0), update=_update_polygon_point_item)
    corner: FloatProperty(default=0.0, min=0.0, update=_update_polygon_point_item)
    handle_left: FloatVectorProperty(size=2, default=(0.0, 0.0), update=_update_polygon_point_item)
    handle_right: FloatVectorProperty(size=2, default=(0.0, 0.0), update=_update_polygon_point_item)


class MathOPSV2NodeTree(NodeTree):
    bl_idname = runtime.TREE_IDNAME
    bl_label = "MathOPS SDF"
    bl_icon = "NODETREE"


class _MathOPSV2NodeBase(Node):
    @classmethod
    def poll(cls, node_tree):
        return getattr(node_tree, "bl_idname", "") == runtime.TREE_IDNAME


class MathOPSV2OutputNode(_MathOPSV2NodeBase):
    bl_idname = runtime.OUTPUT_NODE_IDNAME
    bl_label = "Scene Output"
    bl_width_default = 180
    bl_options = {"INTERNAL"}

    def init(self, _context):
        self.inputs.new(MathOPSV2SDFSocket.bl_idname, "SDF")
        self.is_active_output = True

    def free(self):
        tree = getattr(self, "id_data", None)
        if tree is None or getattr(tree, "bl_idname", "") != runtime.TREE_IDNAME:
            return

        def _restore_output():
            try:
                ensure_graph_output(tree)
            except Exception:
                return None
            runtime.tag_redraw()
            return None

        bpy.app.timers.register(_restore_output, first_interval=0.0)


class MathOPSV2ObjectNode(_MathOPSV2NodeBase):
    bl_idname = runtime.OBJECT_NODE_IDNAME
    bl_label = "SDF Initializer"
    bl_width_default = 260

    target: PointerProperty(type=bpy.types.Object, poll=_poll_sdf_proxy, update=_update_object_node_target)
    use_proxy: BoolProperty(default=False, options={"HIDDEN"})
    proxy_id: StringProperty(default="", options={"HIDDEN"})
    primitive_type: EnumProperty(items=_NODE_PRIMITIVE_ITEMS, default="sphere", update=_update_object_node_payload)
    sdf_location: FloatVectorProperty(
        name="Location",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="XYZ",
        update=_update_object_node_transform,
    )
    sdf_rotation: FloatVectorProperty(
        name="Rotation",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="EULER",
        update=_update_object_node_transform,
    )
    sdf_scale: FloatVectorProperty(
        name="Scale",
        size=3,
        default=(1.0, 1.0, 1.0),
        min=0.001,
        subtype="XYZ",
        update=_update_object_node_transform,
    )
    radius: FloatProperty(name="Radius", default=0.5, min=0.001, update=_update_object_node_payload)
    cone_top_radius: FloatProperty(name="Top Radius", default=0.0, min=0.0, update=_update_object_node_payload)
    cone_bottom_radius: FloatProperty(name="Bottom Radius", default=0.5, min=0.0, update=_update_object_node_payload)
    size: FloatVectorProperty(
        name="Half Size",
        size=3,
        default=(0.5, 0.5, 0.5),
        min=0.001,
        subtype="XYZ",
        update=_update_object_node_payload,
    )
    height: FloatProperty(name="Height", default=1.0, min=0.001, update=_update_object_node_payload)
    major_radius: FloatProperty(name="Major Radius", default=0.75, min=0.001, update=_update_object_node_payload)
    minor_radius: FloatProperty(name="Minor Radius", default=0.25, min=0.001, update=_update_object_node_payload)
    bevel: FloatProperty(name="Bevel", default=0.0, min=0.0, update=_update_object_node_payload)
    cylinder_bevel_top: FloatProperty(name="Bevel Top", default=0.0, min=0.0, update=_update_object_node_payload)
    cylinder_bevel_bottom: FloatProperty(name="Bevel Bottom", default=0.0, min=0.0, update=_update_object_node_payload)
    cylinder_taper: FloatProperty(name="Taper", default=0.0, min=-1.0, max=1.0, update=_update_object_node_payload)
    cylinder_bevel_mode: EnumProperty(name="Bevel Mode", items=BLEND_MODE_ITEMS, default="SMOOTH", update=_update_object_node_payload)
    cone_bevel_mode: EnumProperty(name="Bevel Mode", items=BLEND_MODE_ITEMS, default="SMOOTH", update=_update_object_node_payload)
    capsule_taper: FloatProperty(name="Taper", default=0.0, min=-1.0, max=1.0, update=_update_object_node_payload)
    torus_angle: FloatProperty(name="Angle", default=6.283185307179586, min=0.0, max=6.283185307179586, subtype="ANGLE", update=_update_object_node_payload)
    box_corners: FloatVectorProperty(name="Corner Bevels", size=4, default=(0.0, 0.0, 0.0, 0.0), min=0.0, max=1.0, update=_update_object_node_payload)
    box_edge_top: FloatProperty(name="Edge Top", default=0.0, min=0.0, max=1.0, update=_update_object_node_payload)
    box_edge_bottom: FloatProperty(name="Edge Bottom", default=0.0, min=0.0, max=1.0, update=_update_object_node_payload)
    box_taper: FloatProperty(name="Taper", default=0.0, min=-1.0, max=1.0, update=_update_object_node_payload)
    box_corner_mode: EnumProperty(name="Corners", items=BLEND_MODE_ITEMS, default="SMOOTH", update=_update_object_node_payload)
    box_edge_mode: EnumProperty(name="Edges", items=BLEND_MODE_ITEMS, default="SMOOTH", update=_update_object_node_payload)
    ngon_sides: IntProperty(name="Sides", default=6, min=3, soft_max=32, update=_update_object_node_payload)
    ngon_corner: FloatProperty(name="Bevel", default=0.0, min=0.0, max=1.0, update=_update_object_node_payload)
    ngon_edge_top: FloatProperty(name="Edge Top", default=0.0, min=0.0, max=1.0, update=_update_object_node_payload)
    ngon_edge_bottom: FloatProperty(name="Edge Bottom", default=0.0, min=0.0, max=1.0, update=_update_object_node_payload)
    ngon_taper: FloatProperty(name="Taper", default=0.0, min=-1.0, max=1.0, update=_update_object_node_payload)
    ngon_edge_mode: EnumProperty(name="Edges", items=BLEND_MODE_ITEMS, default="SMOOTH", update=_update_object_node_payload)
    ngon_star: FloatProperty(name="Star", default=0.0, min=0.0, max=1.0, update=_update_object_node_payload)
    polygon_points: CollectionProperty(type=MathOPSV2PolygonPoint)
    polygon_active_index: IntProperty(default=0, options={"HIDDEN"})
    polygon_interpolation: EnumProperty(name="Interpolation", items=POLYGON_INTERPOLATION_ITEMS, default="VECTOR", update=_update_object_node_payload)
    polygon_edge_top: FloatProperty(name="Edge Top", default=0.0, min=0.0, max=1.0, update=_update_object_node_payload)
    polygon_edge_bottom: FloatProperty(name="Edge Bottom", default=0.0, min=0.0, max=1.0, update=_update_object_node_payload)
    polygon_taper: FloatProperty(name="Taper", default=0.0, min=-1.0, max=1.0, update=_update_object_node_payload)
    polygon_edge_mode: EnumProperty(name="Edges", items=BLEND_MODE_ITEMS, default="SMOOTH", update=_update_object_node_payload)
    polygon_is_line: BoolProperty(name="Line", default=False, update=_update_object_node_payload)
    polygon_line_thickness: FloatProperty(name="Thickness", default=0.1, min=0.001, update=_update_object_node_payload)

    def init(self, _context):
        ensure_polygon_defaults(self)
        _ensure_object_node_sockets(self)
        ensure_object_node_id(self)

    def copy(self, _node):
        with suppress_object_node_updates():
            self.proxy_id = uuid.uuid4().hex
            self.target = None
            self.use_proxy = False

    def free(self):
        tree = getattr(self, "id_data", None)
        target = getattr(self, "target", None)
        proxy_id = str(getattr(self, "proxy_id", "") or "")
        settings = None if target is None else runtime.object_settings(target)
        remove_target = bool(
            getattr(self, "use_proxy", False)
            and target is not None
            and settings is not None
            and (not proxy_id or str(getattr(settings, "proxy_id", "") or "") == proxy_id)
            and not _other_initializer_uses_proxy(tree, self, target, proxy_id)
        )
        if remove_target and bool(getattr(settings, "enabled", False)):
            settings.enabled = False
        if _object_node_free_cleanup_suppressed > 0:
            return

        def _cleanup_after_delete():
            if remove_target:
                try:
                    if runtime.safe_pointer(target):
                        bpy.data.objects.remove(target, do_unlink=True)
                except Exception:
                    pass
            try:
                cleanup_graph_structure(tree)
            except Exception:
                return None
            return None

        bpy.app.timers.register(_cleanup_after_delete, first_interval=0.0)

    def draw_buttons(self, _context, layout):
        _ensure_object_node_sockets(self)
        layout.prop(self, "primitive_type", text="Type")
        primitive_type = str(self.primitive_type or "sphere")
        if primitive_type == "box":
            layout.prop(self, "bevel")
        elif primitive_type == "cylinder":
            row = layout.row(align=True)
            row.prop(self, "cylinder_bevel_top")
            row.prop(self, "cylinder_bevel_bottom")
            row = layout.row(align=True)
            row.prop(self, "cylinder_taper")
            row.prop(self, "cylinder_bevel_mode", text="")
        elif primitive_type == "cone":
            row = layout.row(align=True)
            row.prop(self, "bevel")
            row.prop(self, "cone_bevel_mode", text="")
        if primitive_type == "torus":
            layout.prop(self, "torus_angle", text="Angle")
        elif primitive_type == "capsule":
            layout.prop(self, "capsule_taper")
        elif primitive_type == "box":
            layout.prop(self, "box_corners")
            row = layout.row(align=True)
            row.prop(self, "box_edge_top")
            row.prop(self, "box_edge_bottom")
            layout.prop(self, "box_taper")
            row = layout.row(align=True)
            row.prop(self, "box_corner_mode")
            row.prop(self, "box_edge_mode")
        elif primitive_type == "ngon":
            row = layout.row(align=True)
            row.prop(self, "ngon_sides", text="Sides")
            row.prop(self, "ngon_star")
            layout.prop(self, "ngon_corner")
            row = layout.row(align=True)
            row.prop(self, "ngon_edge_top")
            row.prop(self, "ngon_edge_bottom")
            row = layout.row(align=True)
            row.prop(self, "ngon_taper")
            row.prop(self, "ngon_edge_mode")
        elif primitive_type == "polygon":
            layout.label(text=f"Points: {len(self.polygon_points)}")
            layout.prop(self, "polygon_interpolation")
            row = layout.row(align=True)
            row.prop(self, "polygon_edge_top")
            row.prop(self, "polygon_edge_bottom")
            layout.prop(self, "polygon_taper")
            layout.prop(self, "polygon_edge_mode")
            layout.prop(self, "polygon_is_line")
            if self.polygon_is_line:
                layout.prop(self, "polygon_line_thickness")
            target = runtime.object_identity(getattr(self, "target", None))
            if runtime.is_polygon_curve_proxy(target):
                layout.label(text="Edit points and handles on the curve", icon="INFO")
            else:
                layout.label(text="Edit points from the inspector", icon="INFO")
        if self.target is None:
            row = layout.row(align=True)
            row.label(text="No viewport handle")
            operator = row.operator("mathops_v2.create_node_proxy", text="", icon="ADD")
            operator.tree_name = self.id_data.name
            operator.node_name = self.name
            operator.primitive_type = self.primitive_type
        else:
            row = layout.row(align=True)
            row.label(text=f"Handle: {self.target.name}")
        transform_source = transform_source_node(self)
        if transform_source is not None:
            layout.label(text=f"Transform: {transform_source.name}", icon="CONSTRAINT")

    def draw_label(self):
        primitive_name = _NODE_PRIMITIVE_LABELS.get(self.primitive_type, "SDF")
        return f"{primitive_name} Initializer"


class MathOPSV2TransformNode(_MathOPSV2NodeBase):
    bl_idname = TRANSFORM_NODE_IDNAME
    bl_label = "Make Transform"
    bl_width_default = 220

    transform_location: FloatVectorProperty(
        name="Location",
        size=3,
        default=_IDENTITY_LOCATION,
        subtype="XYZ",
        update=_update_transform_node_data,
    )
    transform_rotation: FloatVectorProperty(
        name="Rotation",
        size=3,
        default=_IDENTITY_ROTATION,
        subtype="EULER",
        update=_update_transform_node_data,
    )
    transform_scale: FloatVectorProperty(
        name="Scale",
        size=3,
        default=_IDENTITY_SCALE,
        min=0.001,
        subtype="XYZ",
        update=_update_transform_node_data,
    )

    def init(self, _context):
        _ensure_transform_node_sockets(self)

    def draw_buttons(self, _context, layout):
        pass


class MathOPSV2BreakTransformNode(_MathOPSV2NodeBase):
    bl_idname = BREAK_TRANSFORM_NODE_IDNAME
    bl_label = "Break Transform"
    bl_width_default = 220

    override_location: BoolProperty(name="Position", default=False, update=_update_transform_node_data)
    override_rotation: BoolProperty(name="Rotation", default=False, update=_update_transform_node_data)
    override_scale: BoolProperty(name="Scale", default=False, update=_update_transform_node_data)
    break_location: FloatVectorProperty(
        name="Location",
        size=3,
        default=_IDENTITY_LOCATION,
        subtype="XYZ",
        update=_update_transform_node_data,
    )
    break_rotation: FloatVectorProperty(
        name="Rotation",
        size=3,
        default=_IDENTITY_ROTATION,
        subtype="EULER",
        update=_update_transform_node_data,
    )
    break_scale: FloatVectorProperty(
        name="Scale",
        size=3,
        default=_IDENTITY_SCALE,
        min=0.001,
        subtype="XYZ",
        update=_update_transform_node_data,
    )

    def init(self, _context):
        _ensure_break_transform_node_sockets(self)

    def draw_buttons(self, _context, layout):
        layout.label(text="Split transform into vectors")


class MathOPSV2CSGNode(_MathOPSV2NodeBase):
    bl_idname = runtime.CSG_NODE_IDNAME
    bl_label = "CSG"
    bl_width_default = 190

    operation: EnumProperty(
        name="Operation",
        items=(
            ("UNION", "Union", "Combine both SDF branches"),
            ("SUBTRACT", "Subtract", "Subtract B from A"),
            ("INTERSECT", "Intersect", "Keep the overlap of A and B"),
        ),
        default="UNION",
        update=_update_csg_node_data,
    )
    blend: FloatProperty(
        name="Blend",
        default=0.0,
        min=0.0,
        max=2.0,
        description="Smooth blend radius for the CSG operation",
        update=_update_csg_node_data,
    )

    def init(self, _context):
        _ensure_csg_node_sockets(self)

    def draw_buttons(self, _context, layout):
        layout.prop(self, "operation", text="")


class MathOPSV2MirrorNode(_MathOPSV2NodeBase):
    bl_idname = MIRROR_NODE_IDNAME
    bl_label = "Mirror"
    bl_width_default = 220

    mirror_x: BoolProperty(name="X", default=True, update=_update_mirror_node_data)
    mirror_y: BoolProperty(name="Y", default=False, update=_update_mirror_node_data)
    mirror_z: BoolProperty(name="Z", default=False, update=_update_mirror_node_data)
    blend: FloatProperty(
        name="Blend",
        default=0.0,
        min=0.0,
        max=100.0,
        description="Blend radius for smooth mirror",
        update=_update_mirror_node_data,
    )
    origin_object: PointerProperty(type=bpy.types.Object, update=_update_mirror_node_data)

    def init(self, _context):
        _ensure_mirror_node_sockets(self)

    def draw_buttons(self, _context, layout):
        row = layout.row(align=True)
        row.prop(self, "mirror_x", toggle=True)
        row.prop(self, "mirror_y", toggle=True)
        row.prop(self, "mirror_z", toggle=True)
        layout.prop(self, "blend", text="Blend")
        layout.prop(self, "origin_object", text="Origin")


class MathOPSV2ArrayNode(_MathOPSV2NodeBase):
    bl_idname = ARRAY_NODE_IDNAME
    bl_label = "Array"
    bl_width_default = 240

    array_mode: EnumProperty(
        name="Mode",
        items=(
            (_ARRAY_MODE_GRID, "Grid", "Repeat in a finite 3D grid"),
            (_ARRAY_MODE_RADIAL, "Radial", "Repeat around the Z axis"),
        ),
        default=_ARRAY_MODE_GRID,
        update=_update_array_node_data,
    )
    count_x: IntProperty(name="Count X", default=1, min=1, max=128, update=_update_array_node_data)
    count_y: IntProperty(name="Count Y", default=1, min=1, max=128, update=_update_array_node_data)
    count_z: IntProperty(name="Count Z", default=1, min=1, max=128, update=_update_array_node_data)
    spacing: FloatVectorProperty(
        name="Spacing",
        size=3,
        default=(1.0, 1.0, 1.0),
        subtype="XYZ",
        update=_update_array_node_data,
    )
    radial_count: IntProperty(name="Count", default=6, min=1, max=512, update=_update_array_node_data)
    radius: FloatProperty(name="Radius", default=2.0, min=0.0, max=100.0, update=_update_array_node_data)
    blend: FloatProperty(
        name="Blend",
        default=0.0,
        min=0.0,
        max=100.0,
        description="Blend radius for smooth array repetition",
        update=_update_array_node_data,
    )
    origin_object: PointerProperty(type=bpy.types.Object, update=_update_array_node_data)
    use_array_transform: BoolProperty(default=False, options={"HIDDEN"})
    array_location: FloatVectorProperty(
        name="Array Location",
        size=3,
        default=_IDENTITY_LOCATION,
        subtype="XYZ",
        options={"HIDDEN"},
        update=_update_array_node_transform,
    )
    array_rotation: FloatVectorProperty(
        name="Array Rotation",
        size=3,
        default=_IDENTITY_ROTATION,
        subtype="EULER",
        options={"HIDDEN"},
        update=_update_array_node_transform,
    )
    array_scale: FloatVectorProperty(
        name="Array Scale",
        size=3,
        default=_IDENTITY_SCALE,
        min=0.001,
        subtype="XYZ",
        options={"HIDDEN"},
        update=_update_array_node_transform,
    )

    def init(self, _context):
        _ensure_array_node_sockets(self)

    def draw_buttons(self, _context, layout):
        layout.prop(self, "array_mode", text="")
        if self.array_mode == _ARRAY_MODE_GRID:
            row = layout.row(align=True)
            row.prop(self, "count_x")
            row.prop(self, "count_y")
            row.prop(self, "count_z")
            layout.prop(self, "spacing")
        else:
            layout.prop(self, "radial_count")
            layout.prop(self, "radius")
            layout.prop(self, "origin_object", text="Origin")
        layout.prop(self, "blend", text="Blend")


def _find_output_node(tree):
    for node in tree.nodes:
        if getattr(node, "bl_idname", "") == runtime.OUTPUT_NODE_IDNAME:
            return node
    return None


def _candidate_root_nodes(tree):
    candidates = []
    for node in tree.nodes:
        node_idname = getattr(node, "bl_idname", "")
        if node_idname == runtime.OUTPUT_NODE_IDNAME:
            continue
        if node_idname in {MIRROR_NODE_IDNAME, ARRAY_NODE_IDNAME}:
            source_socket = _input_socket(node, "SDF", MathOPSV2SDFSocket.bl_idname)
            if source_socket is None or not source_socket.is_linked:
                continue
        if node_idname == runtime.CSG_NODE_IDNAME:
            left = _input_socket(node, "A", MathOPSV2SDFSocket.bl_idname)
            right = _input_socket(node, "B", MathOPSV2SDFSocket.bl_idname)
            if left is None or right is None or not left.is_linked or not right.is_linked:
                continue
        socket = _surface_output_socket(node)
        if socket is None:
            continue
        if not socket.is_linked:
            candidates.append(node)
    candidates.sort(key=lambda node: (float(node.location.x), float(node.location.y), node.name))
    return candidates


def _ensure_output_node(tree):
    output = _find_output_node(tree)
    if output is not None:
        if not output.inputs[0].is_linked:
            candidates = _candidate_root_nodes(tree)
            if candidates:
                root = candidates[-1]
                tree.links.new(_surface_output_socket(root), output.inputs[0])
                output.location = (root.location.x + 280.0, root.location.y)
        return output
    output = tree.nodes.new(runtime.OUTPUT_NODE_IDNAME)
    candidates = _candidate_root_nodes(tree)
    if candidates:
        root = candidates[-1]
        tree.links.new(_surface_output_socket(root), output.inputs[0])
        output.location = (root.location.x + 280.0, root.location.y)
    else:
        output.location = (400.0, 0.0)
    return output


def ensure_graph_output(tree):
    return _ensure_output_node(tree)


def cleanup_graph_structure(tree):
    if tree is None or getattr(tree, "bl_idname", "") != runtime.TREE_IDNAME:
        return False

    changed = False
    while True:
        collapsed = False
        for node in list(tree.nodes):
            if getattr(node, "bl_idname", "") != runtime.CSG_NODE_IDNAME:
                continue
            _ensure_csg_node_sockets(node)
            left_source = _linked_source_socket(_input_socket(node, "A", MathOPSV2SDFSocket.bl_idname))
            right_source = _linked_source_socket(_input_socket(node, "B", MathOPSV2SDFSocket.bl_idname))
            linked_sources = [socket for socket in (left_source, right_source) if socket is not None]
            if len(linked_sources) >= 2:
                continue

            output_socket = _surface_output_socket(node)
            output_links = [] if output_socket is None else list(output_socket.links)
            if len(linked_sources) == 1:
                source_socket = linked_sources[0]
                for link in output_links:
                    to_socket = getattr(link, "to_socket", None)
                    if to_socket is not None:
                        tree.links.new(source_socket, to_socket)
            tree.nodes.remove(node)
            changed = True
            collapsed = True
            break
        if not collapsed:
            break

    ensure_graph_output(tree)
    frame_changed = _safe_sync_branch_frames(tree)
    arranged = _auto_arrange_tree(tree)
    if changed:
        scene = _scene_for_tree(tree)
        if scene is not None:
            runtime.mark_scene_static_dirty(scene)
    if changed or frame_changed or arranged:
        runtime.note_interaction()
        runtime.tag_redraw()
    return changed


def _linked_source_node(socket):
    if not socket.is_linked or not socket.links:
        return None
    return socket.links[0].from_node


def _linked_source_socket(socket):
    if socket is None or not socket.is_linked or not socket.links:
        return None
    return socket.links[0].from_socket


def _surface_output_socket(node):
    for socket in getattr(node, "outputs", ()):
        if getattr(socket, "bl_idname", "") == MathOPSV2SDFSocket.bl_idname:
            return socket
    return None


def _socket_by_name(sockets, name, bl_idname=""):
    for socket in sockets:
        if socket.name != name:
            continue
        if bl_idname and getattr(socket, "bl_idname", "") != bl_idname:
            continue
        return socket
    return None


def _transform_input_socket(node):
    return _socket_by_name(getattr(node, "inputs", ()), _SOCKET_TRANSFORM, TRANSFORM_SOCKET_IDNAME)


def _array_offset_input_socket(node):
    return _socket_by_name(getattr(node, "inputs", ()), _SOCKET_TRANSFORM_OFFSET, TRANSFORM_SOCKET_IDNAME)


def _input_socket(node, name, bl_idname=""):
    return _socket_by_name(getattr(node, "inputs", ()), name, bl_idname)


def _output_socket(node, name, bl_idname=""):
    return _socket_by_name(getattr(node, "outputs", ()), name, bl_idname)


def _ensure_named_input_socket(node, bl_idname, name):
    socket = _socket_by_name(getattr(node, "inputs", ()), name, bl_idname)
    created = False
    if socket is None:
        socket = node.inputs.new(bl_idname, name)
        created = True
    return socket, created


def _ensure_named_output_socket(node, bl_idname, name):
    socket = _socket_by_name(getattr(node, "outputs", ()), name, bl_idname)
    created = False
    if socket is None:
        socket = node.outputs.new(bl_idname, name)
        created = True
    return socket, created


def _set_socket_float_default(socket, value, epsilon=1.0e-6):
    if socket is None:
        return False
    current = float(getattr(socket, "default_value", 0.0))
    target = float(value)
    if abs(current - target) <= epsilon:
        return False
    socket.default_value = target
    return True


def _set_socket_vector_default(socket, value, epsilon=1.0e-6):
    if socket is None:
        return False
    current = tuple(float(component) for component in getattr(socket, "default_value", (0.0, 0.0, 0.0)))
    target = tuple(float(component) for component in value)
    if not _float_seq_changed(current, target, epsilon):
        return False
    socket.default_value = target
    return True


def _sync_object_node_socket_visibility(node):
    primitive_type = str(getattr(node, "primitive_type", "sphere") or "sphere").strip().lower()
    visibility = {
        _SOCKET_RADIUS: primitive_type in {"sphere", "cylinder", "capsule", "ngon"},
        _SOCKET_TOP_RADIUS: primitive_type == "cone",
        _SOCKET_BOTTOM_RADIUS: primitive_type == "cone",
        _SOCKET_HALF_SIZE: primitive_type == "box",
        _SOCKET_HEIGHT: primitive_type in {"cylinder", "cone", "capsule", "ngon", "polygon"},
        _SOCKET_MAJOR_RADIUS: primitive_type == "torus",
        _SOCKET_MINOR_RADIUS: primitive_type == "torus",
    }
    for name, shown in visibility.items():
        socket = _socket_by_name(getattr(node, "inputs", ()), name)
        if socket is None:
            continue
        if bool(getattr(socket, "hide", False)) != (not shown):
            socket.hide = not shown
    return node


def _ensure_transform_node_sockets(node):
    if getattr(node, "bl_idname", "") != TRANSFORM_NODE_IDNAME:
        return node
    location_socket, location_created = _ensure_named_input_socket(node, MathOPSV2VectorSocket.bl_idname, _SOCKET_LOCATION)
    rotation_socket, rotation_created = _ensure_named_input_socket(node, MathOPSV2EulerSocket.bl_idname, _SOCKET_ROTATION)
    scale_socket, scale_created = _ensure_named_input_socket(node, MathOPSV2VectorSocket.bl_idname, _SOCKET_SCALE)
    _ensure_named_output_socket(node, MathOPSV2TransformSocket.bl_idname, _SOCKET_TRANSFORM)
    with suppress_object_node_updates():
        if location_created:
            _set_socket_vector_default(location_socket, getattr(node, "transform_location", _IDENTITY_LOCATION))
        if rotation_created:
            _set_socket_vector_default(rotation_socket, getattr(node, "transform_rotation", _IDENTITY_ROTATION))
        if scale_created:
            _set_socket_vector_default(scale_socket, getattr(node, "transform_scale", _IDENTITY_SCALE))
    return node


def _ensure_break_transform_node_sockets(node):
    if getattr(node, "bl_idname", "") != BREAK_TRANSFORM_NODE_IDNAME:
        return node
    _ensure_named_input_socket(node, MathOPSV2TransformSocket.bl_idname, _SOCKET_TRANSFORM)
    _ensure_named_output_socket(node, MathOPSV2VectorSocket.bl_idname, _SOCKET_LOCATION)
    _ensure_named_output_socket(node, MathOPSV2EulerSocket.bl_idname, _SOCKET_ROTATION)
    _ensure_named_output_socket(node, MathOPSV2VectorSocket.bl_idname, _SOCKET_SCALE)
    return node


def _ensure_csg_node_sockets(node):
    if getattr(node, "bl_idname", "") != runtime.CSG_NODE_IDNAME:
        return node
    _ensure_named_input_socket(node, MathOPSV2SDFSocket.bl_idname, "A")
    _ensure_named_input_socket(node, MathOPSV2SDFSocket.bl_idname, "B")
    blend_socket, created = _ensure_named_input_socket(node, MathOPSV2FloatSocket.bl_idname, _SOCKET_BLEND)
    _ensure_named_output_socket(node, MathOPSV2SDFSocket.bl_idname, "SDF")
    if created:
        with suppress_object_node_updates():
            _set_socket_float_default(blend_socket, getattr(node, "blend", 0.0))
    return node


def _ensure_mirror_node_sockets(node):
    if getattr(node, "bl_idname", "") != MIRROR_NODE_IDNAME:
        return node
    _ensure_named_input_socket(node, MathOPSV2SDFSocket.bl_idname, "SDF")
    _ensure_named_output_socket(node, MathOPSV2SDFSocket.bl_idname, "SDF")
    _ensure_named_output_socket(node, MathOPSV2TransformSocket.bl_idname, _SOCKET_TRANSFORM)
    return node


def _ensure_array_node_sockets(node):
    if getattr(node, "bl_idname", "") != ARRAY_NODE_IDNAME:
        return node
    _ensure_named_input_socket(node, MathOPSV2SDFSocket.bl_idname, "SDF")
    _ensure_named_input_socket(node, MathOPSV2TransformSocket.bl_idname, _SOCKET_TRANSFORM_OFFSET)
    _ensure_named_output_socket(node, MathOPSV2SDFSocket.bl_idname, "SDF")
    _ensure_named_output_socket(node, MathOPSV2TransformSocket.bl_idname, _SOCKET_TRANSFORM)
    return node


def _ensure_object_node_sockets(node):
    if getattr(node, "bl_idname", "") != runtime.OBJECT_NODE_IDNAME:
        return node
    _ensure_named_input_socket(node, MathOPSV2TransformSocket.bl_idname, _SOCKET_TRANSFORM)
    radius_socket, radius_created = _ensure_named_input_socket(node, MathOPSV2FloatSocket.bl_idname, _SOCKET_RADIUS)
    bottom_radius_socket, bottom_radius_created = _ensure_named_input_socket(node, MathOPSV2FloatSocket.bl_idname, _SOCKET_BOTTOM_RADIUS)
    top_radius_socket, top_radius_created = _ensure_named_input_socket(node, MathOPSV2FloatSocket.bl_idname, _SOCKET_TOP_RADIUS)
    size_socket, size_created = _ensure_named_input_socket(node, MathOPSV2VectorSocket.bl_idname, _SOCKET_HALF_SIZE)
    height_socket, height_created = _ensure_named_input_socket(node, MathOPSV2FloatSocket.bl_idname, _SOCKET_HEIGHT)
    major_socket, major_created = _ensure_named_input_socket(node, MathOPSV2FloatSocket.bl_idname, _SOCKET_MAJOR_RADIUS)
    minor_socket, minor_created = _ensure_named_input_socket(node, MathOPSV2FloatSocket.bl_idname, _SOCKET_MINOR_RADIUS)
    _ensure_named_output_socket(node, MathOPSV2SDFSocket.bl_idname, "SDF")
    with suppress_object_node_updates():
        _migrate_cylinder_bevel_fields(node)
        if radius_created:
            _set_socket_float_default(radius_socket, getattr(node, "radius", 0.5))
        if top_radius_created:
            _set_socket_float_default(top_radius_socket, _cone_top_radius_value(node))
        if bottom_radius_created:
            _set_socket_float_default(bottom_radius_socket, _cone_bottom_radius_value(node))
        if size_created:
            _set_socket_vector_default(size_socket, getattr(node, "size", (0.5, 0.5, 0.5)))
        if height_created:
            _set_socket_float_default(height_socket, getattr(node, "height", 1.0))
        if major_created:
            _set_socket_float_default(major_socket, getattr(node, "major_radius", 0.75))
        if minor_created:
            _set_socket_float_default(minor_socket, getattr(node, "minor_radius", 0.25))
    _sync_object_node_socket_visibility(node)
    return node


def _ensure_node_sockets(node):
    node_idname = getattr(node, "bl_idname", "")
    if node_idname == runtime.OBJECT_NODE_IDNAME:
        return _ensure_object_node_sockets(node)
    if node_idname == TRANSFORM_NODE_IDNAME:
        return _ensure_transform_node_sockets(node)
    if node_idname == BREAK_TRANSFORM_NODE_IDNAME:
        return _ensure_break_transform_node_sockets(node)
    if node_idname == runtime.CSG_NODE_IDNAME:
        return _ensure_csg_node_sockets(node)
    if node_idname == MIRROR_NODE_IDNAME:
        return _ensure_mirror_node_sockets(node)
    if node_idname == ARRAY_NODE_IDNAME:
        return _ensure_array_node_sockets(node)
    return node


def transform_source_node(node):
    socket = _transform_input_socket(node)
    if socket is None:
        return None
    return _linked_source_node(socket)


def _sanitized_location(values):
    return tuple(float(component) for component in values)


def _sanitized_rotation(values):
    return tuple(float(component) for component in values)


def _sanitized_scale(values):
    return tuple(max(0.001, abs(float(component))) for component in values)


def _identity_transform_values():
    return (_IDENTITY_LOCATION, _IDENTITY_ROTATION, _IDENTITY_SCALE)


def _resolve_vector_input_socket(socket, fallback, sanitizer, visiting=None):
    if socket is not None:
        source_socket = _linked_source_socket(socket)
        if source_socket is not None:
            resolved = _resolve_vector_output_socket(source_socket, visiting=visiting)
            if resolved is not None:
                return sanitizer(resolved)
    return sanitizer(getattr(socket, "default_value", fallback) if socket is not None else fallback)


def _break_transform_components(node, visiting=None):
    source = transform_source_node(node)
    if source is None:
        return _identity_transform_values()
    return _resolved_transform_values(source, visiting=visiting)


def break_transform_values(node):
    return _break_transform_components(node)


def _resolve_vector_output_socket(socket, visiting=None):
    node = getattr(socket, "node", None)
    node_idname = getattr(node, "bl_idname", "")
    if node_idname != BREAK_TRANSFORM_NODE_IDNAME:
        return None
    location, rotation, scale = _break_transform_components(node, visiting=visiting)
    if socket.name == _SOCKET_LOCATION:
        return location
    if socket.name == _SOCKET_ROTATION:
        return rotation
    if socket.name == _SOCKET_SCALE:
        return scale
    return None


def _transform_node_values(node, visiting=None):
    _ensure_transform_node_sockets(node)
    return (
        _resolve_vector_input_socket(_input_socket(node, _SOCKET_LOCATION), getattr(node, "transform_location", _IDENTITY_LOCATION), _sanitized_location, visiting=visiting),
        _resolve_vector_input_socket(_input_socket(node, _SOCKET_ROTATION), getattr(node, "transform_rotation", _IDENTITY_ROTATION), _sanitized_rotation, visiting=visiting),
        _resolve_vector_input_socket(_input_socket(node, _SOCKET_SCALE), getattr(node, "transform_scale", _IDENTITY_SCALE), _sanitized_scale, visiting=visiting),
    )


def _mirror_transform_values(node):
    transform = _object_transform_values(getattr(node, "origin_object", None))
    return _identity_transform_values() if transform is None else transform


def _array_node_transform_values(node):
    return (
        _sanitized_location(getattr(node, "array_location", _IDENTITY_LOCATION)),
        _sanitized_rotation(getattr(node, "array_rotation", _IDENTITY_ROTATION)),
        _sanitized_scale(getattr(node, "array_scale", _IDENTITY_SCALE)),
    )


def _resolve_transform_socket_values(socket, visiting=None, prefer_proxy=False):
    source = _linked_source_node(socket)
    if source is None:
        return _identity_transform_values()
    return _resolved_transform_values(source, visiting=visiting, prefer_proxy=prefer_proxy)


def _array_transform_offset_values(node, visiting=None):
    _ensure_array_node_sockets(node)
    return _resolve_transform_socket_values(_array_offset_input_socket(node), visiting=visiting, prefer_proxy=False)


def _array_source_node(node):
    _ensure_array_node_sockets(node)
    return _linked_source_node(_input_socket(node, "SDF", MathOPSV2SDFSocket.bl_idname))


def _array_default_center(node):
    source = _array_source_node(node)
    if source is None:
        return _IDENTITY_LOCATION
    try:
        return _bounds_center(_subtree_bounds(source))
    except Exception:
        return _IDENTITY_LOCATION


def _array_transform_values(node, visiting=None, prefer_proxy=True):
    _ensure_array_node_sockets(node)
    if bool(getattr(node, "use_array_transform", False)):
        return _array_node_transform_values(node)
    return (_array_default_center(node), _IDENTITY_ROTATION, _IDENTITY_SCALE)


def _write_vector_input_socket(socket, values, sanitizer, visiting=None):
    if socket is None:
        return False
    source_socket = _linked_source_socket(socket)
    if source_socket is not None:
        return _write_vector_output_socket(source_socket, values, sanitizer, visiting=visiting)
    target = sanitizer(values)
    current = sanitizer(getattr(socket, "default_value", target))
    if not _float_seq_changed(current, target):
        return False
    socket.default_value = target
    return True


def _write_vector_output_socket(socket, values, sanitizer, visiting=None):
    node = getattr(socket, "node", None)
    if getattr(node, "bl_idname", "") != BREAK_TRANSFORM_NODE_IDNAME:
        return False
    source = transform_source_node(node)
    if source is None:
        return False
    target = sanitizer(values)
    if socket.name == _SOCKET_LOCATION:
        return _write_transform_values(source, location=target, visiting=visiting)
    if socket.name == _SOCKET_ROTATION:
        return _write_transform_values(source, rotation=target, visiting=visiting)
    if socket.name == _SOCKET_SCALE:
        return _write_transform_values(source, scale=target, visiting=visiting)
    return False


def _resolved_transform_values(node, visiting=None, prefer_proxy=True):
    if node is None:
        return _identity_transform_values()
    if visiting is None:
        visiting = set()
    node_key = int(node.as_pointer())
    if node_key in visiting:
        raise RuntimeError(f"Cycle detected at node '{node.name}'")

    visiting.add(node_key)
    try:
        node_idname = getattr(node, "bl_idname", "")
        if node_idname == runtime.OBJECT_NODE_IDNAME:
            _ensure_object_node_sockets(node)
            source = transform_source_node(node)
            if source is not None:
                return _resolved_transform_values(source, visiting=visiting, prefer_proxy=prefer_proxy)
            target = getattr(node, "target", None)
            if prefer_proxy and runtime.is_sdf_proxy(target) and runtime.safe_pointer(_proxy_transform_owner(node)) == runtime.safe_pointer(node):
                return (
                    _sanitized_location(target.location),
                    _sanitized_rotation(target.rotation_euler),
                    _sanitized_scale(target.scale),
                )
            return (
                _sanitized_location(node.sdf_location),
                _sanitized_rotation(node.sdf_rotation),
                _sanitized_scale(node.sdf_scale),
            )
        if node_idname == TRANSFORM_NODE_IDNAME:
            return _transform_node_values(node, visiting=visiting)
        if node_idname == MIRROR_NODE_IDNAME:
            return _mirror_transform_values(node)
        if node_idname == ARRAY_NODE_IDNAME:
            return _array_transform_values(node, visiting=visiting, prefer_proxy=prefer_proxy)
    finally:
        visiting.remove(node_key)

    return _identity_transform_values()


def _write_transform_values(node, location=None, rotation=None, scale=None, visiting=None):
    if node is None:
        return False
    if visiting is None:
        visiting = set()
    node_key = int(node.as_pointer())
    if node_key in visiting:
        return False

    visiting.add(node_key)
    try:
        node_idname = getattr(node, "bl_idname", "")
        changed = False
        if node_idname == TRANSFORM_NODE_IDNAME:
            _ensure_transform_node_sockets(node)
            if location is not None:
                changed = _write_vector_input_socket(_input_socket(node, _SOCKET_LOCATION), location, _sanitized_location, visiting=visiting) or changed
            if rotation is not None:
                changed = _write_vector_input_socket(_input_socket(node, _SOCKET_ROTATION), rotation, _sanitized_rotation, visiting=visiting) or changed
            if scale is not None:
                changed = _write_vector_input_socket(_input_socket(node, _SOCKET_SCALE), scale, _sanitized_scale, visiting=visiting) or changed
            return changed

        if node_idname == ARRAY_NODE_IDNAME:
            _ensure_array_node_sockets(node)
            current_location, current_rotation, current_scale = _array_transform_values(node, visiting=visiting, prefer_proxy=False)
            target_location = current_location if location is None else _sanitized_location(location)
            target_rotation = current_rotation if rotation is None else _sanitized_rotation(rotation)
            target_scale = current_scale if scale is None else _sanitized_scale(scale)
            origin_object = runtime.object_identity(getattr(node, "origin_object", None))
            branch_center = _array_default_center(node)
            current_matrix = _array_field_matrix(current_location, current_rotation, current_scale, branch_center)
            target_matrix = _array_field_matrix(target_location, target_rotation, target_scale, branch_center)
            changed = False
            with suppress_object_node_updates():
                if not bool(getattr(node, "use_array_transform", False)):
                    node.use_array_transform = True
                    changed = True
                if _float_seq_changed(tuple(getattr(node, "array_location", _IDENTITY_LOCATION)), target_location):
                    node.array_location = target_location
                    changed = True
                if _float_seq_changed(tuple(getattr(node, "array_rotation", _IDENTITY_ROTATION)), target_rotation):
                    node.array_rotation = target_rotation
                    changed = True
                if _float_seq_changed(tuple(getattr(node, "array_scale", _IDENTITY_SCALE)), target_scale):
                    node.array_scale = target_scale
                    changed = True
            if origin_object is not None and _apply_matrix_to_object(origin_object, target_matrix @ current_matrix.inverted_safe()):
                changed = True
            if changed:
                _update_array_node_transform(node, None)
            return changed

        if node_idname == runtime.OBJECT_NODE_IDNAME:
            _ensure_object_node_sockets(node)
            if location is not None and _float_seq_changed(tuple(node.sdf_location), location):
                node.sdf_location = location
                changed = True
            if rotation is not None and _float_seq_changed(tuple(node.sdf_rotation), rotation):
                node.sdf_rotation = rotation
                changed = True
            if scale is not None and _float_seq_changed(tuple(node.sdf_scale), scale):
                node.sdf_scale = scale
                changed = True
            return changed
        return False
    finally:
        visiting.remove(node_key)


def _write_transform_to_node_source(node, location, rotation, scale):
    source = transform_source_node(node)
    if source is None:
        return False
    with suppress_object_node_updates():
        return _write_transform_values(
            source,
            location=_sanitized_location(location),
            rotation=_sanitized_rotation(rotation),
            scale=_sanitized_scale(scale),
        )


def _sync_resolved_transform_to_node(node):
    location, rotation, scale = _resolved_transform_values(node, prefer_proxy=False)
    changed = False
    with suppress_object_node_updates():
        if _float_seq_changed(tuple(node.sdf_location), location):
            node.sdf_location = location
            changed = True
        if _float_seq_changed(tuple(node.sdf_rotation), rotation):
            node.sdf_rotation = rotation
            changed = True
        if _float_seq_changed(tuple(node.sdf_scale), scale):
            node.sdf_scale = scale
            changed = True
    return changed


def _transform_chain_contains(node, source_key, visiting=None):
    if node is None:
        return False
    if visiting is None:
        visiting = set()
    node_key = int(node.as_pointer())
    if node_key in visiting:
        return False
    if node_key == source_key:
        return True
    visiting.add(node_key)
    try:
        source = transform_source_node(node)
        if source is None:
            return False
        return _transform_chain_contains(source, source_key, visiting=visiting)
    finally:
        visiting.remove(node_key)


def _sync_tree_proxy_transforms(tree, source_node=None):
    if tree is None:
        return False
    changed = False
    source_key = 0 if source_node is None else int(source_node.as_pointer())
    for node in initializer_nodes(tree):
        _ensure_object_node_sockets(node)
        if source_key and not _transform_chain_contains(node, source_key):
            continue
        changed = _sync_resolved_transform_to_node(node) or changed
        changed = bool(sync_node_to_proxy(node) or changed)
    return changed


def get_scene_tree(scene, create=False):
    settings = runtime.scene_settings(scene)
    if settings is None:
        return None
    tree = getattr(settings, "node_tree", None)
    if tree is not None and getattr(tree, "bl_idname", "") == runtime.TREE_IDNAME:
        return tree
    tree = _recover_scene_tree(scene, settings)
    if tree is not None:
        return tree
    if not create:
        return None
    return new_scene_tree(scene)


def new_scene_tree(scene):
    tree = bpy.data.node_groups.new(name=f"{scene.name} SDF", type=runtime.TREE_IDNAME)
    _ensure_output_node(tree)
    settings = runtime.scene_settings(scene)
    if settings is not None:
        settings.node_tree = tree
    runtime.mark_scene_static_dirty(scene)
    runtime.tag_redraw()
    return tree


def ensure_scene_tree(scene):
    tree = get_scene_tree(scene, create=True)
    ensure_graph_output(tree)
    return tree


def _scene_proxy_tree_names(scene):
    tree_names = set()
    for obj in scene.objects:
        settings = runtime.object_settings(obj)
        if settings is None or not bool(getattr(settings, "enabled", False)):
            continue
        tree_name = str(getattr(settings, "source_tree_name", "") or "")
        if tree_name:
            tree_names.add(tree_name)
    return tree_names


def _available_scene_trees():
    node_groups = getattr(bpy.data, "node_groups", None)
    if node_groups is None:
        return []
    return [tree for tree in node_groups if getattr(tree, "bl_idname", "") == runtime.TREE_IDNAME]


def _recover_scene_tree(scene, settings):
    trees = _available_scene_trees()
    if not trees:
        return None
    if len(trees) == 1:
        settings.node_tree = trees[0]
        return trees[0]

    preferred_names = _scene_proxy_tree_names(scene)
    scene_tree_name = f"{scene.name} SDF"
    scene_object_pointers = {runtime.safe_pointer(obj) for obj in scene.objects}
    best_tree = None
    best_score = None

    for tree in trees:
        linked_targets = 0
        for node in initializer_nodes(tree):
            target_pointer = runtime.safe_pointer(getattr(node, "target", None))
            if target_pointer and target_pointer in scene_object_pointers:
                linked_targets += 1

        score = (
            1 if tree.name == scene_tree_name else 0,
            1 if tree.name in preferred_names else 0,
            linked_targets,
            len(initializer_nodes(tree)),
        )
        if best_score is None or score > best_score:
            best_tree = tree
            best_score = score

    if best_tree is None or best_score == (0, 0, 0, 0):
        return None
    settings.node_tree = best_tree
    return best_tree


def _node_editor_space(area):
    for space in area.spaces:
        if space.type == "NODE_EDITOR":
            return space
    return None


def _node_editor_window_region(area):
    for region in area.regions:
        if region.type == "WINDOW":
            return region
    return None


def _node_editor_context(context, tree=None):
    window = getattr(context, "window", None)
    screen = getattr(context, "screen", None)
    if window is None or screen is None:
        return None, None, None, None

    preferred = None
    fallback = None
    for area in screen.areas:
        if area.type != "NODE_EDITOR":
            continue
        space = _node_editor_space(area)
        region = _node_editor_window_region(area)
        if space is None or region is None:
            continue
        if tree is not None and getattr(space, "tree_type", "") == runtime.TREE_IDNAME and getattr(space, "node_tree", None) == tree:
            return window, area, space, region
        if preferred is None and getattr(space, "tree_type", "") == runtime.TREE_IDNAME:
            preferred = (window, area, space, region)
        if fallback is None:
            fallback = (window, area, space, region)
    return preferred or fallback or (None, None, None, None)


def focus_scene_tree(context, create=True):
    tree = get_scene_tree(context.scene, create=create)
    if tree is None:
        return False
    _window, area, space, _region = _node_editor_context(context, tree=tree)
    if area is None or space is None:
        return False
    if getattr(space, "tree_type", "") != runtime.TREE_IDNAME:
        space.tree_type = runtime.TREE_IDNAME
    if getattr(space, "node_tree", None) != tree:
        space.node_tree = tree
    area.tag_redraw()
    return True


def select_nodes_in_editor(context, nodes, active_node=None, reveal=True):
    selected_nodes = []
    seen = set()
    tree = None
    for node in tuple(nodes or ()):
        if node is None:
            continue
        node_tree = getattr(node, "id_data", None)
        if node_tree is None or getattr(node_tree, "bl_idname", "") != runtime.TREE_IDNAME:
            continue
        if tree is None:
            tree = node_tree
        if node_tree != tree:
            continue
        node_key = runtime.safe_pointer(node)
        if node_key == 0 or node_key in seen:
            continue
        seen.add(node_key)
        selected_nodes.append(node)
    if tree is None or not selected_nodes:
        return False

    if active_node not in selected_nodes:
        active_node = selected_nodes[-1]

    selected_keys = {runtime.safe_pointer(node) for node in selected_nodes}
    for tree_node in tree.nodes:
        tree_node.select = runtime.safe_pointer(tree_node) in selected_keys
    tree.nodes.active = active_node

    window, area, space, region = _node_editor_context(context, tree=tree)
    if area is None or space is None:
        return False

    if getattr(space, "tree_type", "") != runtime.TREE_IDNAME:
        space.tree_type = runtime.TREE_IDNAME
    if getattr(space, "node_tree", None) != tree:
        space.node_tree = tree
    area.tag_redraw()
    runtime.tag_redraw(context)

    if not reveal or region is None:
        return True

    try:
        with context.temp_override(window=window, area=area, region=region, space_data=space):
            bpy.ops.node.view_selected("INVOKE_DEFAULT")
    except Exception:
        try:
            with context.temp_override(window=window, area=area, region=region, space_data=space):
                bpy.ops.node.view_selected()
        except Exception:
            return True
    return True


def select_node_in_editor(context, node, reveal=True):
    tree = getattr(node, "id_data", None)
    if tree is None or getattr(tree, "bl_idname", "") != runtime.TREE_IDNAME:
        return False
    return select_nodes_in_editor(context, (node,), active_node=node, reveal=reveal)


def list_proxy_objects(scene):
    proxies = [obj for obj in scene.objects if runtime.is_sdf_proxy(obj)]
    proxies.sort(key=lambda obj: obj.name_full)
    return proxies


def _other_initializer_uses_proxy(tree, current_node, target, proxy_id):
    current_key = runtime.safe_pointer(current_node)
    target_key = runtime.object_key(target)
    proxy_id = str(proxy_id or "")
    for node in initializer_nodes(tree):
        if runtime.safe_pointer(node) == current_key:
            continue
        if not bool(getattr(node, "use_proxy", False)):
            continue
        other_target = getattr(node, "target", None)
        other_target_key = runtime.object_key(other_target)
        if target_key and other_target_key == target_key and runtime.is_sdf_proxy(other_target):
            return True
        if proxy_id and str(getattr(node, "proxy_id", "") or "") == proxy_id:
            return True
    return False


def _initializer_proxy_binding(node):
    if getattr(node, "bl_idname", "") != runtime.OBJECT_NODE_IDNAME:
        return None, 0
    if not bool(getattr(node, "use_proxy", False)):
        return None, 0

    proxy_id = str(getattr(node, "proxy_id", "") or "")
    target = getattr(node, "target", None)
    target_key = runtime.object_key(target)
    if target_key and runtime.is_sdf_proxy(target):
        settings = runtime.object_settings(target)
        target_proxy_id = "" if settings is None else str(getattr(settings, "proxy_id", "") or "")
        binding_id = target_proxy_id or proxy_id
        if binding_id:
            return ("proxy", binding_id), 2
        return ("target", target_key), 2
    if proxy_id:
        return ("proxy", proxy_id), 1
    return None, 0


def deduplicate_initializer_nodes(tree):
    if tree is None or getattr(tree, "bl_idname", "") != runtime.TREE_IDNAME:
        return False

    kept = {}
    duplicates = []
    for node in initializer_nodes(tree):
        key, score = _initializer_proxy_binding(node)
        if key is None:
            continue
        current = kept.get(key)
        if current is None:
            kept[key] = (node, score)
            continue
        kept_node, kept_score = current
        if kept_score >= score:
            duplicates.append(node)
            continue
        duplicates.append(kept_node)
        kept[key] = (node, score)

    if not duplicates:
        return False

    with suppress_object_node_free_cleanup():
        for node in duplicates:
            if runtime.safe_pointer(node) == 0:
                continue
            _detach_proxy_binding(node)
            tree.nodes.remove(node)

    cleanup_graph_structure(tree)
    scene = _scene_for_tree(tree)
    if scene is not None:
        runtime.mark_scene_static_dirty(scene)
    runtime.note_interaction()
    runtime.tag_redraw()
    return True


def scene_summary(scene):
    tree = get_scene_tree(scene, create=False)
    return {
        "tree_name": "None" if tree is None else tree.name,
        "proxy_count": len(list_proxy_objects(scene)),
    }


def initializer_nodes(tree):
    if tree is None:
        return []
    return [node for node in tree.nodes if getattr(node, "bl_idname", "") == runtime.OBJECT_NODE_IDNAME]


def warp_origin_referenced(tree, obj):
    object_pointer = runtime.object_key(obj)
    if tree is None or object_pointer == 0:
        return False
    for node in tree.nodes:
        if getattr(node, "bl_idname", "") not in {MIRROR_NODE_IDNAME, ARRAY_NODE_IDNAME}:
            continue
        if runtime.object_key(getattr(node, "origin_object", None)) == object_pointer:
            return True
    return False


def ensure_object_node_id(node):
    proxy_id = str(getattr(node, "proxy_id", "") or "")
    if proxy_id:
        return proxy_id
    proxy_id = uuid.uuid4().hex
    with suppress_object_node_updates():
        node.proxy_id = proxy_id
    return proxy_id


def find_initializer_node(tree, obj=None, proxy_id=""):
    object_pointer = runtime.object_key(obj)
    proxy_id = str(proxy_id or "")
    for node in initializer_nodes(tree):
        if object_pointer and runtime.object_key(getattr(node, "target", None)) == object_pointer:
            return node
        if proxy_id and str(getattr(node, "proxy_id", "") or "") == proxy_id:
            return node
    return None


def node_input_socket(node, name):
    return _input_socket(node, name)


def node_output_socket(node, name):
    return _output_socket(node, name)


def _append_unique_node(nodes, seen, node):
    node_key = runtime.safe_pointer(node)
    if node_key == 0 or node_key in seen:
        return False
    seen.add(node_key)
    nodes.append(node)
    return True


def _collect_transform_related_nodes(node, nodes, seen):
    if node is None:
        return
    _append_unique_node(nodes, seen, node)
    node_idname = getattr(node, "bl_idname", "")
    if node_idname == TRANSFORM_NODE_IDNAME:
        for socket_name in (_SOCKET_LOCATION, _SOCKET_ROTATION, _SOCKET_SCALE):
            source_socket = _linked_source_socket(_input_socket(node, socket_name))
            source_node = None if source_socket is None else getattr(source_socket, "node", None)
            if source_node is not None:
                _collect_transform_related_nodes(source_node, nodes, seen)
        return
    if node_idname == BREAK_TRANSFORM_NODE_IDNAME:
        _collect_transform_related_nodes(transform_source_node(node), nodes, seen)


def _collect_downstream_nodes(node, nodes, seen):
    output_socket = _surface_output_socket(node)
    if output_socket is None:
        return
    for link in list(output_socket.links):
        downstream = getattr(link, "to_node", None)
        if downstream is None or getattr(downstream, "bl_idname", "") == runtime.OUTPUT_NODE_IDNAME:
            continue
        _append_unique_node(nodes, seen, downstream)
        _collect_downstream_nodes(downstream, nodes, seen)


def _surface_downstream_nodes(node):
    output_socket = _surface_output_socket(node)
    if output_socket is None:
        return []
    downstream = []
    for link in list(output_socket.links):
        target = getattr(link, "to_node", None)
        if target is None or getattr(target, "bl_idname", "") == runtime.OUTPUT_NODE_IDNAME:
            continue
        downstream.append(target)
    return downstream


def _proxy_transform_owner(node):
    owner = node
    current = node
    seen = set()
    while current is not None:
        current_key = runtime.safe_pointer(current)
        if current_key == 0 or current_key in seen:
            break
        seen.add(current_key)
        downstream = _surface_downstream_nodes(current)
        if len(downstream) != 1:
            break
        next_node = downstream[0]
        if getattr(next_node, "bl_idname", "") != ARRAY_NODE_IDNAME:
            break
        owner = next_node
        current = next_node
    return owner


def inspector_related_nodes(node):
    if node is None:
        return []
    nodes = []
    seen = set()
    _append_unique_node(nodes, seen, node)
    _collect_transform_related_nodes(transform_source_node(node), nodes, seen)
    _collect_downstream_nodes(node, nodes, seen)
    return nodes


def _collect_branch_source_nodes(node, nodes, seen):
    if not _append_unique_node(nodes, seen, node):
        return
    for socket in getattr(node, "inputs", ()):
        source = _linked_source_node(socket)
        if source is not None:
            _collect_branch_source_nodes(source, nodes, seen)


def _branch_contains_node(branch_root, node):
    if branch_root is None or node is None:
        return False
    target_key = runtime.safe_pointer(node)
    if target_key == 0:
        return False
    seen = set()
    _collect_branch_source_nodes(branch_root, [], seen)
    return target_key in seen


def _branch_order_entries_from_root(node):
    if node is None:
        return []
    if getattr(node, "bl_idname", "") == runtime.CSG_NODE_IDNAME:
        _ensure_csg_node_sockets(node)
        left = _linked_source_node(_input_socket(node, "A", MathOPSV2SDFSocket.bl_idname))
        right = _linked_source_node(_input_socket(node, "B", MathOPSV2SDFSocket.bl_idname))
        if left is not None and right is not None:
            entries = _branch_order_entries_from_root(left)
            entries.append({
                "branch_root": right,
                "csg_node": node,
            })
            return entries
    return [{"branch_root": node, "csg_node": None}]


def branch_order_entries(tree):
    if tree is None or getattr(tree, "bl_idname", "") != runtime.TREE_IDNAME:
        return []
    output = _find_output_node(tree)
    if output is None or not getattr(output, "inputs", None):
        return []
    root = _linked_source_node(output.inputs[0])
    if root is None:
        return []

    entries = []
    for index, entry in enumerate(_branch_order_entries_from_root(root)):
        branch_root = entry["branch_root"]
        csg_node = entry["csg_node"]
        entries.append(
            {
                "index": index,
                "branch_root": branch_root,
                "csg_node": csg_node,
                "operation": None if csg_node is None else str(getattr(csg_node, "operation", "UNION") or "UNION"),
            }
        )
    return entries


def branch_entry_index(tree, node):
    if tree is None or node is None:
        return -1
    node_key = runtime.safe_pointer(node)
    if node_key == 0:
        return -1
    entries = branch_order_entries(tree)
    for entry in entries:
        csg_node = entry["csg_node"]
        if csg_node is not None and runtime.safe_pointer(csg_node) == node_key:
            return int(entry["index"])
    for entry in entries:
        if _branch_contains_node(entry["branch_root"], node):
            return int(entry["index"])
    return -1


def branch_root_name_for_node(tree, node):
    if tree is None or node is None:
        return ""
    node_key = runtime.safe_pointer(node)
    if node_key == 0:
        return ""

    entries = branch_order_entries(tree)
    node_idname = getattr(node, "bl_idname", "")
    if node_idname == "NodeFrame" and bool(node.get(_BRANCH_FRAME_TAG, False)):
        label = str(getattr(node, "label", "") or "")
        for entry in entries:
            branch_root = entry["branch_root"]
            if branch_root is not None and _branch_frame_label(branch_root) == label:
                return str(getattr(branch_root, "name", "") or "")

    for entry in entries:
        branch_root = entry["branch_root"]
        csg_node = entry["csg_node"]
        if branch_root is not None and runtime.safe_pointer(branch_root) == node_key:
            return str(getattr(branch_root, "name", "") or "")
        if csg_node is not None and runtime.safe_pointer(csg_node) == node_key:
            return "" if branch_root is None else str(getattr(branch_root, "name", "") or "")
        if _branch_contains_node(branch_root, node):
            return "" if branch_root is None else str(getattr(branch_root, "name", "") or "")
    return ""


def _collect_branch_graph_nodes(node, nodes, seen):
    if not _append_unique_node(nodes, seen, node):
        return
    node_idname = getattr(node, "bl_idname", "")
    if node_idname == runtime.CSG_NODE_IDNAME:
        _ensure_csg_node_sockets(node)
        for socket_name in ("A", "B"):
            source = _linked_source_node(_input_socket(node, socket_name, MathOPSV2SDFSocket.bl_idname))
            if source is not None:
                _collect_branch_graph_nodes(source, nodes, seen)
        return
    if node_idname in {MIRROR_NODE_IDNAME, ARRAY_NODE_IDNAME}:
        source = _linked_source_node(_input_socket(node, "SDF", MathOPSV2SDFSocket.bl_idname))
        if source is not None:
            _collect_branch_graph_nodes(source, nodes, seen)


def _branch_graph_nodes(branch_root):
    if branch_root is None:
        return []
    nodes = []
    seen = set()
    _collect_branch_graph_nodes(branch_root, nodes, seen)
    return nodes


def _branch_frame_label(branch_root):
    for node in _branch_graph_nodes(branch_root):
        if getattr(node, "bl_idname", "") == runtime.OBJECT_NODE_IDNAME:
            return f"{str(getattr(node, 'name', '') or 'SDF')} Branch"
    return f"{str(getattr(branch_root, 'name', '') or 'Branch')} Branch"


def _expected_branch_frames(tree):
    expected = []
    for entry in branch_order_entries(tree):
        branch_root = entry["branch_root"]
        branch_nodes = [node for node in _branch_graph_nodes(branch_root) if runtime.safe_pointer(node) != 0]
        if not branch_nodes:
            continue
        expected.append(
            {
                "label": _branch_frame_label(branch_root),
                "branch_nodes": branch_nodes,
                "node_keys": frozenset(runtime.safe_pointer(node) for node in branch_nodes if runtime.safe_pointer(node) != 0),
            }
        )
    return expected


def _sync_branch_frames(tree):
    if tree is None or getattr(tree, "bl_idname", "") != runtime.TREE_IDNAME:
        return False
    if _branch_frame_sync_suppressed > 0:
        return False

    changed = False
    with suppress_branch_frame_sync():
        expected = _expected_branch_frames(tree)
        auto_frames = _auto_branch_frames(tree)
        existing = []
        auto_frame_keys = {runtime.safe_pointer(frame) for frame in auto_frames}
        for frame in auto_frames:
            frame_key = runtime.safe_pointer(frame)
            existing.append(
                {
                    "frame": frame,
                    "label": str(getattr(frame, "label", "") or ""),
                    "node_keys": frozenset(
                        runtime.safe_pointer(node)
                        for node in tree.nodes
                        if runtime.safe_pointer(getattr(node, "parent", None)) == frame_key and runtime.safe_pointer(node) != 0
                    ),
                }
            )

        if len(existing) == len(expected):
            unmatched = list(existing)
            matches = True
            for wanted in expected:
                match = next(
                    (
                        item
                        for item in unmatched
                        if item["label"] == wanted["label"] and item["node_keys"] == wanted["node_keys"]
                    ),
                    None,
                )
                if match is None:
                    matches = False
                    break
                unmatched.remove(match)
            if matches and not unmatched:
                return False

        for node in list(tree.nodes):
            if getattr(node, "bl_idname", "") == "NodeFrame":
                continue
            if runtime.safe_pointer(getattr(node, "parent", None)) not in auto_frame_keys:
                continue
            _set_node_parent_keep_absolute(node, None)
            changed = True

        for frame in auto_frames:
            if runtime.safe_pointer(frame) == 0:
                continue
            tree.nodes.remove(frame)
            changed = True

        for wanted in expected:
            branch_nodes = list(wanted["branch_nodes"])
            frame = tree.nodes.new("NodeFrame")
            frame[_BRANCH_FRAME_TAG] = True
            frame.label = wanted["label"]
            frame.shrink = True
            top_left_x = min(_node_absolute_location(node)[0] for node in branch_nodes) - 40.0
            top_left_y = max(_node_absolute_location(node)[1] for node in branch_nodes) + 30.0
            frame.location = (top_left_x, top_left_y)
            for node in branch_nodes:
                _set_node_parent_keep_absolute(node, frame)
            changed = True
    return changed


def _clear_socket_links(tree, socket):
    if tree is None or socket is None:
        return
    for link in list(getattr(socket, "links", ())):
        tree.links.remove(link)


def move_branch_entry(tree, branch_root_name, direction):
    if tree is None or getattr(tree, "bl_idname", "") != runtime.TREE_IDNAME:
        return None
    branch_root_name = str(branch_root_name or "")
    direction = str(direction or "").strip().upper()
    if direction not in {"UP", "DOWN"}:
        return None

    output = _find_output_node(tree)
    entries = branch_order_entries(tree)
    if output is None or len(entries) < 3:
        return None

    entry_index = -1
    for index, entry in enumerate(entries):
        branch_root = entry["branch_root"]
        if branch_root is not None and str(getattr(branch_root, "name", "") or "") == branch_root_name:
            entry_index = index
            break
    if entry_index <= 0:
        return None

    swap_index = entry_index - 1 if direction == "UP" else entry_index + 1
    if swap_index <= 0 or swap_index >= len(entries):
        return None

    entries[entry_index], entries[swap_index] = entries[swap_index], entries[entry_index]

    _clear_socket_links(tree, output.inputs[0])
    for entry in entries[1:]:
        csg_node = entry["csg_node"]
        _ensure_csg_node_sockets(csg_node)
        _clear_socket_links(tree, _input_socket(csg_node, "A", MathOPSV2SDFSocket.bl_idname))
        _clear_socket_links(tree, _input_socket(csg_node, "B", MathOPSV2SDFSocket.bl_idname))

    current_root = entries[0]["branch_root"]
    _ensure_node_sockets(current_root)
    for entry in entries[1:]:
        csg_node = entry["csg_node"]
        branch_root = entry["branch_root"]
        _ensure_csg_node_sockets(csg_node)
        _ensure_node_sockets(branch_root)
        current_socket = _surface_output_socket(current_root)
        branch_socket = _surface_output_socket(branch_root)
        left_input = _input_socket(csg_node, "A", MathOPSV2SDFSocket.bl_idname)
        right_input = _input_socket(csg_node, "B", MathOPSV2SDFSocket.bl_idname)
        if current_socket is None or branch_socket is None or left_input is None or right_input is None:
            return None
        tree.links.new(current_socket, left_input)
        tree.links.new(branch_socket, right_input)
        current_root = csg_node

    output_socket = _surface_output_socket(current_root)
    if output_socket is None:
        return None
    tree.links.new(output_socket, output.inputs[0])
    _safe_sync_branch_frames(tree)
    _auto_arrange_tree(tree)
    _mark_tree_dirty(tree, static=True)
    runtime.note_interaction()
    runtime.tag_redraw()
    return entries[swap_index]["branch_root"]


def _float_seq_changed(lhs, rhs, epsilon=1.0e-6):
    if len(lhs) != len(rhs):
        return True
    for left, right in zip(lhs, rhs):
        if abs(float(left) - float(right)) > epsilon:
            return True
    return False


def _object_transform_values(obj):
    try:
        obj = runtime.object_identity(obj)
        if obj is None:
            return None
        return (
            _sanitized_location(obj.location),
            _sanitized_rotation(obj.rotation_euler),
            _sanitized_scale(obj.scale),
        )
    except ReferenceError:
        return None


def _write_transform_to_object(obj, location=None, rotation=None, scale=None):
    try:
        obj = runtime.object_identity(obj)
        if obj is None:
            return False

        changed = False
        if location is not None:
            target_location = _sanitized_location(location)
            if _float_seq_changed(tuple(obj.location), target_location):
                obj.location = target_location
                changed = True
        if rotation is not None:
            target_rotation = _sanitized_rotation(rotation)
            if obj.rotation_mode != "XYZ":
                obj.rotation_mode = "XYZ"
                changed = True
            if _float_seq_changed(tuple(obj.rotation_euler), target_rotation):
                obj.rotation_euler = target_rotation
                changed = True
        if scale is not None:
            target_scale = _sanitized_scale(scale)
            if _float_seq_changed(tuple(obj.scale), target_scale):
                obj.scale = target_scale
                changed = True
        return changed
    except ReferenceError:
        return False


def _apply_matrix_to_object(obj, matrix):
    try:
        obj = runtime.object_identity(obj)
        if obj is None:
            return False
        current = obj.matrix_world.copy()
        target = matrix @ current
        if current == target:
            return False
        obj.matrix_world = target
        return True
    except ReferenceError:
        return False


def _set_float_vector_attr(owner, attribute, value, epsilon=1.0e-6):
    current = tuple(float(component) for component in getattr(owner, attribute))
    target = tuple(float(component) for component in value)
    if _float_seq_changed(current, target, epsilon):
        setattr(owner, attribute, target)


def _set_float_attr(owner, attribute, value, epsilon=1.0e-6):
    current = float(getattr(owner, attribute))
    target = float(value)
    if abs(current - target) > epsilon:
        setattr(owner, attribute, target)


def _set_int_attr(owner, attribute, value):
    current = int(getattr(owner, attribute))
    target = int(value)
    if current != target:
        setattr(owner, attribute, target)


def _set_bool_attr(owner, attribute, value):
    current = bool(getattr(owner, attribute))
    target = bool(value)
    if current != target:
        setattr(owner, attribute, target)


def _set_string_attr(owner, attribute, value):
    current = str(getattr(owner, attribute, "") or "")
    target = str(value or "")
    if current != target:
        setattr(owner, attribute, target)


def _copy_direct_attrs(source, target):
    for attribute in _DIRECT_FLOAT_ATTRS:
        _set_float_attr(target, attribute, float(getattr(source, attribute, 0.0)))
    for attribute in _DIRECT_VECTOR_ATTRS:
        _set_float_vector_attr(target, attribute, tuple(float(component) for component in getattr(source, attribute, (0.0, 0.0, 0.0, 0.0))))
    for attribute in _DIRECT_INT_ATTRS:
        _set_int_attr(target, attribute, int(getattr(source, attribute, 0)))
    for attribute in _DIRECT_BOOL_ATTRS:
        _set_bool_attr(target, attribute, bool(getattr(source, attribute, False)))
    for attribute in _DIRECT_ENUM_ATTRS:
        _set_string_attr(target, attribute, str(getattr(source, attribute, "") or ""))


def _float_socket_value(socket, fallback, min_value=0.0):
    value = fallback if socket is None else getattr(socket, "default_value", fallback)
    return max(float(min_value), float(value))


def _vector_socket_value(socket, fallback, sanitizer, visiting=None):
    return _resolve_vector_input_socket(socket, fallback, sanitizer, visiting=visiting)


def _node_socket_float(node, socket_name, attribute, fallback, min_value=0.0):
    socket = _input_socket(node, socket_name, MathOPSV2FloatSocket.bl_idname) if getattr(node, "bl_idname", "") else None
    if socket is not None:
        return _float_socket_value(socket, getattr(node, attribute, fallback), min_value=min_value)
    return max(float(min_value), float(getattr(node, attribute, fallback)))


def _node_socket_vector(node, socket_name, attribute, fallback, sanitizer, visiting=None):
    socket = _input_socket(node, socket_name) if getattr(node, "bl_idname", "") else None
    if socket is not None:
        return _vector_socket_value(socket, getattr(node, attribute, fallback), sanitizer, visiting=visiting)
    return sanitizer(getattr(node, attribute, fallback))


def primitive_supports_bevel(primitive_type):
    primitive_type = str(primitive_type or "sphere").strip().lower()
    return primitive_type in {"box", "cylinder", "cone"}


def object_parameter_socket_names(node):
    primitive_type = str(getattr(node, "primitive_type", "sphere") or "sphere").strip().lower()
    if primitive_type == "sphere":
        return (_SOCKET_RADIUS,)
    if primitive_type == "box":
        return (_SOCKET_HALF_SIZE,)
    if primitive_type in {"cylinder", "capsule", "ngon"}:
        return (_SOCKET_RADIUS, _SOCKET_HEIGHT)
    if primitive_type == "cone":
        return (_SOCKET_BOTTOM_RADIUS, _SOCKET_TOP_RADIUS, _SOCKET_HEIGHT)
    if primitive_type == "torus":
        return (_SOCKET_MAJOR_RADIUS, _SOCKET_MINOR_RADIUS)
    if primitive_type == "polygon":
        return (_SOCKET_HEIGHT,)
    return ()


def _node_transform_values(node):
    _ensure_object_node_sockets(node)
    return _resolved_transform_values(node)


def _configure_proxy_display(obj):
    if obj.rotation_mode != "XYZ":
        obj.rotation_mode = "XYZ"
    if getattr(obj, "type", "") == "EMPTY":
        if obj.empty_display_type != "PLAIN_AXES":
            obj.empty_display_type = "PLAIN_AXES"
        if abs(float(obj.empty_display_size) - 0.01) > 1.0e-6:
            obj.empty_display_size = 0.01
    elif getattr(obj, "type", "") == "CURVE":
        curve = getattr(obj, "data", None)
        if curve is not None:
            if getattr(curve, "dimensions", "3D") != "2D":
                curve.dimensions = "2D"
            if hasattr(curve, "fill_mode") and getattr(curve, "fill_mode", "FULL") != "NONE":
                curve.fill_mode = "NONE"
    if obj.show_name:
        obj.show_name = False
    if obj.show_in_front:
        obj.show_in_front = False
    if not obj.hide_render:
        obj.hide_render = True


def _mirror_node_to_proxy_settings(node, obj):
    settings = runtime.object_settings(obj)
    if settings is None:
        return None
    ensure_polygon_defaults(node)
    proxy_id = ensure_object_node_id(node)
    if not settings.enabled:
        settings.enabled = True
    if str(settings.proxy_id or "") != proxy_id:
        settings.proxy_id = proxy_id
    source_tree_name = getattr(node.id_data, "name", "")
    if str(settings.source_tree_name or "") != source_tree_name:
        settings.source_tree_name = source_tree_name
    source_node_name = getattr(node, "name", "")
    if str(settings.source_node_name or "") != source_node_name:
        settings.source_node_name = source_node_name
    primitive_type = str(node.primitive_type or "sphere")
    if str(settings.primitive_type or "") != primitive_type:
        settings.primitive_type = primitive_type
    _set_float_attr(settings, "radius", _node_socket_float(node, _SOCKET_RADIUS, "radius", 0.5, min_value=0.001))
    _set_float_attr(settings, "cone_top_radius", _node_socket_float(node, _SOCKET_TOP_RADIUS, "cone_top_radius", _cone_top_radius_value(node), min_value=0.0))
    _set_float_attr(settings, "cone_bottom_radius", _node_socket_float(node, _SOCKET_BOTTOM_RADIUS, "cone_bottom_radius", _cone_bottom_radius_value(node), min_value=0.0))
    _set_float_attr(settings, "cylinder_bevel_top", _cylinder_bevel_top_value(node))
    _set_float_attr(settings, "cylinder_bevel_bottom", _cylinder_bevel_bottom_value(node))
    _set_float_vector_attr(settings, "size", _node_socket_vector(node, _SOCKET_HALF_SIZE, "size", (0.5, 0.5, 0.5), _sanitized_scale))
    _set_float_attr(settings, "height", _node_socket_float(node, _SOCKET_HEIGHT, "height", 1.0, min_value=0.001))
    _set_float_attr(settings, "major_radius", _node_socket_float(node, _SOCKET_MAJOR_RADIUS, "major_radius", 0.75, min_value=0.001))
    _set_float_attr(settings, "minor_radius", _node_socket_float(node, _SOCKET_MINOR_RADIUS, "minor_radius", 0.25, min_value=0.001))
    _copy_direct_attrs(node, settings)
    _set_int_attr(settings, "ngon_sides", max(3, int(getattr(node, "ngon_sides", 6))))
    set_polygon_points(settings, polygon_control_point_data(node, ensure_default=True))
    _set_int_attr(settings, "polygon_active_index", int(getattr(node, "polygon_active_index", 0)))
    return settings


def sync_node_to_proxy(node, include_transform=True):
    target = runtime.object_identity(getattr(node, "target", None))
    if target is None or not runtime.is_sdf_proxy(target):
        return None

    if runtime.safe_pointer(getattr(node, "target", None)) != runtime.safe_pointer(target):
        with suppress_object_node_updates():
            node.target = target

    _ensure_object_node_sockets(node)
    ensure_polygon_defaults(node)
    ensure_object_node_id(node)
    _configure_proxy_display(target)
    _mirror_node_to_proxy_settings(node, target)
    if str(getattr(node, "primitive_type", "") or "") == "polygon" and getattr(target, "type", "") == "CURVE":
        sync_polygon_curve_proxy(
            target,
            polygon_control_point_data(node, ensure_default=True),
            is_line=bool(getattr(node, "polygon_is_line", False)),
            interpolation=_polygon_interpolation(node),
        )
    if include_transform:
        transform_owner = _proxy_transform_owner(node)
        location, rotation, scale = _resolved_transform_values(transform_owner, prefer_proxy=False)
        _set_float_vector_attr(target, "location", location)
        _set_float_vector_attr(target, "rotation_euler", rotation)
        _set_float_vector_attr(target, "scale", scale)
    return target


def sync_proxy_to_node(node):
    target = runtime.object_identity(getattr(node, "target", None))
    if target is None or not runtime.is_sdf_proxy(target):
        return False

    if runtime.safe_pointer(getattr(node, "target", None)) != runtime.safe_pointer(target):
        with suppress_object_node_updates():
            node.target = target

    _ensure_object_node_sockets(node)
    changed = False
    settings = runtime.object_settings(target)
    if settings is not None and str(getattr(settings, "primitive_type", "sphere") or "sphere") == "polygon" and getattr(target, "type", "") == "CURVE":
        curve_interpolation = polygon_curve_interpolation(target)
        curve_is_line = polygon_curve_is_line(target)
        curve_points = polygon_curve_control_point_data(target, ensure_default=True)
        recentered, centered_points = _recenter_polygon_curve_proxy(
            target,
            curve_points,
            curve_interpolation,
            is_line=curve_is_line,
            resolution=_polygon_curve_resolution(target, curve_interpolation),
        )
        if recentered:
            sync_polygon_curve_proxy(target, centered_points, is_line=curve_is_line, interpolation=curve_interpolation)
            curve_points = centered_points
            changed = True
        if polygon_control_point_data(settings, ensure_default=True) != curve_points:
            set_polygon_points(settings, curve_points)
            changed = True
        if bool(getattr(settings, "polygon_is_line", False)) != curve_is_line:
            settings.polygon_is_line = curve_is_line
            changed = True
        if _polygon_interpolation(settings) != curve_interpolation:
            settings.polygon_interpolation = curve_interpolation
            changed = True
    transform_owner = _proxy_transform_owner(node)
    transform_source = transform_source_node(node)
    target_location = tuple(float(component) for component in target.location)
    target_rotation = tuple(float(component) for component in target.rotation_euler)
    target_scale = tuple(abs(float(component)) for component in target.scale)
    if runtime.safe_pointer(transform_owner) == runtime.safe_pointer(node):
        changed = _write_transform_to_node_source(node, target_location, target_rotation, target_scale) or changed
    else:
        changed = _write_transform_values(transform_owner, location=target_location, rotation=target_rotation, scale=target_scale) or changed
    with suppress_object_node_updates():
        if settings is not None:
            primitive_type = str(getattr(settings, "primitive_type", "sphere") or "sphere")
            if str(getattr(node, "primitive_type", "sphere") or "sphere") != primitive_type:
                node.primitive_type = primitive_type
                changed = True
            ensure_polygon_defaults(node)
            current_points = polygon_control_point_data(node)
            target_points = polygon_control_point_data(settings, ensure_default=(primitive_type == "polygon"))
            if current_points != target_points:
                set_polygon_points(node, target_points)
                changed = True
            target_active_index = max(0, int(getattr(settings, "polygon_active_index", 0)))
            if int(getattr(node, "polygon_active_index", 0)) != target_active_index:
                node.polygon_active_index = target_active_index
                changed = True
            if int(getattr(node, "ngon_sides", 6)) != max(3, int(getattr(settings, "ngon_sides", 6))):
                node.ngon_sides = max(3, int(getattr(settings, "ngon_sides", 6)))
                changed = True
            if abs(float(getattr(node, "radius", 0.5)) - float(getattr(settings, "radius", 0.5))) > 1.0e-6:
                node.radius = float(getattr(settings, "radius", 0.5))
                changed = True
            target_cone_top_radius = _cone_top_radius_value(settings)
            if abs(float(getattr(node, "cone_top_radius", 0.0)) - target_cone_top_radius) > 1.0e-6:
                node.cone_top_radius = target_cone_top_radius
                changed = True
            target_cone_bottom_radius = _cone_bottom_radius_value(settings)
            if abs(float(getattr(node, "cone_bottom_radius", 0.5)) - target_cone_bottom_radius) > 1.0e-6:
                node.cone_bottom_radius = target_cone_bottom_radius
                changed = True
            target_cylinder_bevel_top = _cylinder_bevel_top_value(settings)
            if abs(float(getattr(node, "cylinder_bevel_top", 0.0)) - target_cylinder_bevel_top) > 1.0e-6:
                node.cylinder_bevel_top = target_cylinder_bevel_top
                changed = True
            target_cylinder_bevel_bottom = _cylinder_bevel_bottom_value(settings)
            if abs(float(getattr(node, "cylinder_bevel_bottom", 0.0)) - target_cylinder_bevel_bottom) > 1.0e-6:
                node.cylinder_bevel_bottom = target_cylinder_bevel_bottom
                changed = True
            if _float_seq_changed(tuple(getattr(node, "size", (0.5, 0.5, 0.5))), tuple(float(component) for component in getattr(settings, "size", (0.5, 0.5, 0.5)))):
                node.size = tuple(float(component) for component in getattr(settings, "size", (0.5, 0.5, 0.5)))
                changed = True
            if abs(float(getattr(node, "height", 1.0)) - float(getattr(settings, "height", 1.0))) > 1.0e-6:
                node.height = float(getattr(settings, "height", 1.0))
                changed = True
            if abs(float(getattr(node, "major_radius", 0.75)) - float(getattr(settings, "major_radius", 0.75))) > 1.0e-6:
                node.major_radius = float(getattr(settings, "major_radius", 0.75))
                changed = True
            if abs(float(getattr(node, "minor_radius", 0.25)) - float(getattr(settings, "minor_radius", 0.25))) > 1.0e-6:
                node.minor_radius = float(getattr(settings, "minor_radius", 0.25))
                changed = True
            for attribute in _DIRECT_FLOAT_ATTRS:
                target_value = float(getattr(settings, attribute, 0.0))
                if abs(float(getattr(node, attribute, 0.0)) - target_value) > 1.0e-6:
                    setattr(node, attribute, target_value)
                    changed = True
            for attribute in _DIRECT_VECTOR_ATTRS:
                target_value = tuple(float(component) for component in getattr(settings, attribute, (0.0, 0.0, 0.0, 0.0)))
                if _float_seq_changed(tuple(float(component) for component in getattr(node, attribute, target_value)), target_value):
                    setattr(node, attribute, target_value)
                    changed = True
            for attribute in _DIRECT_BOOL_ATTRS:
                target_value = bool(getattr(settings, attribute, False))
                if bool(getattr(node, attribute, False)) != target_value:
                    setattr(node, attribute, target_value)
                    changed = True
            for attribute in _DIRECT_ENUM_ATTRS:
                target_value = str(getattr(settings, attribute, "") or "")
                if str(getattr(node, attribute, "") or "") != target_value:
                    setattr(node, attribute, target_value)
                    changed = True
            _set_socket_float_default(_input_socket(node, _SOCKET_RADIUS, MathOPSV2FloatSocket.bl_idname), getattr(node, "radius", 0.5))
            _set_socket_float_default(_input_socket(node, _SOCKET_TOP_RADIUS, MathOPSV2FloatSocket.bl_idname), _cone_top_radius_value(node))
            _set_socket_float_default(_input_socket(node, _SOCKET_BOTTOM_RADIUS, MathOPSV2FloatSocket.bl_idname), _cone_bottom_radius_value(node))
            _set_socket_vector_default(_input_socket(node, _SOCKET_HALF_SIZE, MathOPSV2VectorSocket.bl_idname), getattr(node, "size", (0.5, 0.5, 0.5)))
            _set_socket_float_default(_input_socket(node, _SOCKET_HEIGHT, MathOPSV2FloatSocket.bl_idname), getattr(node, "height", 1.0))
            _set_socket_float_default(_input_socket(node, _SOCKET_MAJOR_RADIUS, MathOPSV2FloatSocket.bl_idname), getattr(node, "major_radius", 0.75))
            _set_socket_float_default(_input_socket(node, _SOCKET_MINOR_RADIUS, MathOPSV2FloatSocket.bl_idname), getattr(node, "minor_radius", 0.25))
        if not bool(getattr(node, "use_proxy", False)):
            node.use_proxy = True
            changed = True
        proxy_id = str(getattr(runtime.object_settings(target), "proxy_id", "") or "")
        if proxy_id and str(getattr(node, "proxy_id", "") or "") != proxy_id:
            node.proxy_id = proxy_id
            changed = True
        if runtime.safe_pointer(transform_owner) == runtime.safe_pointer(node):
            if _float_seq_changed(tuple(node.sdf_location), target_location):
                node.sdf_location = target_location
                changed = True
            if _float_seq_changed(tuple(node.sdf_rotation), target_rotation):
                node.sdf_rotation = target_rotation
                changed = True
            if _float_seq_changed(tuple(node.sdf_scale), target_scale):
                node.sdf_scale = target_scale
                changed = True
    if changed:
        _sync_object_node_socket_visibility(node)
        if runtime.safe_pointer(transform_owner) == runtime.safe_pointer(node) and transform_source is not None:
            _sync_tree_proxy_transforms(getattr(node, "id_data", None), source_node=transform_source)
        elif runtime.safe_pointer(transform_owner) != runtime.safe_pointer(node):
            _sync_tree_proxy_transforms(getattr(node, "id_data", None))
    return changed


def _adopt_proxy_data(node, obj):
    obj = runtime.object_identity(obj)
    settings = runtime.object_settings(obj)
    if settings is not None and str(getattr(settings, "primitive_type", "") or "") == "polygon" and getattr(obj, "type", "") == "CURVE":
        set_polygon_points(settings, polygon_curve_control_point_data(obj, ensure_default=True))
        settings.polygon_is_line = polygon_curve_is_line(obj)
        settings.polygon_interpolation = polygon_curve_interpolation(obj)
    with suppress_object_node_updates():
        if settings is not None:
            proxy_id = str(settings.proxy_id or "")
            node.proxy_id = proxy_id or uuid.uuid4().hex
            node.primitive_type = str(settings.primitive_type or "sphere")
            node.radius = float(settings.radius)
            node.cone_top_radius = _cone_top_radius_value(settings)
            node.cone_bottom_radius = _cone_bottom_radius_value(settings)
            node.cylinder_bevel_top = _cylinder_bevel_top_value(settings)
            node.cylinder_bevel_bottom = _cylinder_bevel_bottom_value(settings)
            node.size = tuple(float(component) for component in settings.size)
            node.height = float(settings.height)
            node.major_radius = float(settings.major_radius)
            node.minor_radius = float(settings.minor_radius)
            _copy_direct_attrs(settings, node)
            node.ngon_sides = max(3, int(getattr(settings, "ngon_sides", 6)))
            set_polygon_points(node, polygon_control_point_data(settings, ensure_default=(str(getattr(settings, "primitive_type", "") or "") == "polygon")))
            node.polygon_active_index = max(0, int(getattr(settings, "polygon_active_index", 0)))
            _set_socket_float_default(_input_socket(node, _SOCKET_RADIUS, MathOPSV2FloatSocket.bl_idname), settings.radius)
            _set_socket_float_default(_input_socket(node, _SOCKET_TOP_RADIUS, MathOPSV2FloatSocket.bl_idname), _cone_top_radius_value(settings))
            _set_socket_float_default(_input_socket(node, _SOCKET_BOTTOM_RADIUS, MathOPSV2FloatSocket.bl_idname), _cone_bottom_radius_value(settings))
            _set_socket_vector_default(_input_socket(node, _SOCKET_HALF_SIZE, MathOPSV2VectorSocket.bl_idname), settings.size)
            _set_socket_float_default(_input_socket(node, _SOCKET_HEIGHT, MathOPSV2FloatSocket.bl_idname), settings.height)
            _set_socket_float_default(_input_socket(node, _SOCKET_MAJOR_RADIUS, MathOPSV2FloatSocket.bl_idname), settings.major_radius)
            _set_socket_float_default(_input_socket(node, _SOCKET_MINOR_RADIUS, MathOPSV2FloatSocket.bl_idname), settings.minor_radius)
        else:
            ensure_object_node_id(node)
            ensure_polygon_defaults(node)
        node.sdf_location = tuple(float(component) for component in obj.location)
        node.sdf_rotation = tuple(float(component) for component in obj.rotation_euler)
        node.sdf_scale = tuple(float(component) for component in obj.scale)
        node.use_proxy = True
        node.target = obj
    _sync_object_node_socket_visibility(node)


def attach_proxy_to_node(node, obj, adopt_proxy=False, tag_redraw=True):
    obj = runtime.object_identity(obj)
    _ensure_object_node_sockets(node)
    if adopt_proxy:
        _adopt_proxy_data(node, obj)
        sync_node_to_proxy(node, include_transform=True)
    else:
        with suppress_object_node_updates():
            node.use_proxy = True
            node.target = obj
        sync_node_to_proxy(node, include_transform=True)
    if tag_redraw:
        runtime.tag_redraw()
    return obj


def add_proxy_to_tree(scene, obj):
    tree, nodes = add_proxies_to_tree(scene, (obj,))
    return tree, (None if not nodes else nodes[0])


def add_proxies_to_tree(scene, objects):
    scene = runtime.scene_identity(scene)
    tree = ensure_scene_tree(scene)
    output = _ensure_output_node(tree)
    existing_root = _linked_source_node(output.inputs[0])
    linked_nodes = []
    new_nodes = []
    seen_targets = set()

    for raw_obj in objects:
        obj = runtime.object_identity(raw_obj)
        obj_key = runtime.object_key(obj)
        if not obj_key or obj_key in seen_targets:
            continue
        seen_targets.add(obj_key)
        settings = runtime.object_settings(obj)
        proxy_id = "" if settings is None else str(getattr(settings, "proxy_id", "") or "")
        existing_node = find_initializer_node(tree, obj=obj, proxy_id=proxy_id)
        if existing_node is not None:
            if runtime.safe_pointer(getattr(existing_node, "target", None)) != runtime.safe_pointer(obj):
                attach_proxy_to_node(existing_node, obj, adopt_proxy=True, tag_redraw=False)
            linked_nodes.append(existing_node)
            continue
        object_node = tree.nodes.new(runtime.OBJECT_NODE_IDNAME)
        attach_proxy_to_node(object_node, obj, adopt_proxy=True, tag_redraw=False)
        new_nodes.append(object_node)
        linked_nodes.append(object_node)

    if not new_nodes:
        return tree, linked_nodes

    runtime.mark_scene_static_dirty(scene)
    if existing_root is not None and output.inputs[0].is_linked and output.inputs[0].links:
        tree.links.remove(output.inputs[0].links[0])

    current_root = existing_root
    next_y = 0.0 if existing_root is None else existing_root.location.y - 220.0
    for index, object_node in enumerate(new_nodes):
        if current_root is None:
            object_node.location = (120.0, 0.0)
            current_root = object_node
            next_y = -220.0
            continue

        object_node.location = (current_root.location.x, next_y)
        csg_node = tree.nodes.new(runtime.CSG_NODE_IDNAME)
        csg_node.operation = "UNION"
        csg_node.location = (current_root.location.x + 260.0, current_root.location.y)
        tree.links.new(_surface_output_socket(current_root), csg_node.inputs[0])
        tree.links.new(object_node.outputs[0], csg_node.inputs[1])
        current_root = csg_node
        next_y -= 220.0

    if current_root is not None:
        output.location = (current_root.location.x + 280.0, current_root.location.y)
        tree.links.new(_surface_output_socket(current_root), output.inputs[0])
    _safe_sync_branch_frames(tree)
    _auto_arrange_tree(tree)
    runtime.tag_redraw()
    return tree, linked_nodes


def _node_transform_matrix(node):
    location_values, rotation_values, _scale_values = _node_transform_values(node)
    location = Vector(location_values)
    rotation = Euler(rotation_values, "XYZ")
    return Matrix.LocRotScale(location, rotation, None)


def _append_polygon_rows(node, polygon_rows):
    use_bezier = (
        not bool(getattr(node, "polygon_is_line", False))
        and _polygon_interpolation(node) == "BEZIER"
    )
    point_offset = len(polygon_rows)
    if use_bezier:
        control_points = effective_polygon_control_point_data(node, ensure_default=True)
        for x, y, _corner, handle_left_x, handle_left_y, handle_right_x, handle_right_y in control_points:
            polygon_rows.append((float(x), float(y), float(handle_right_x), float(handle_right_y)))
            polygon_rows.append((float(handle_left_x), float(handle_left_y), 0.0, 0.0))
        return point_offset, len(control_points), control_points
    points = effective_polygon_point_data(node, ensure_default=True)
    for x, y, _corner in points:
        polygon_rows.append((float(x), float(y), 0.0, 0.0))
    return point_offset, len(points), points


def _mode_to_id(mode):
    return 1.0 if str(mode or "SMOOTH") == "CHAMFER" else 0.0


def _split_taper(value):
    taper = float(value)
    return max(taper, 0.0), max(-taper, 0.0)


def _safe_cone_radii(bottom_radius, top_radius):
    bottom = max(0.0, float(bottom_radius))
    top = max(0.0, float(top_radius))
    if max(bottom, top) <= 1.0e-6:
        bottom = 0.001
    return bottom, top


def _primitive_extra_data(node, primitive_type):
    extra = {"bevel": max(0.0, float(getattr(node, "bevel", 0.0))) if primitive_supports_bevel(primitive_type) else 0.0}
    if primitive_type == "cylinder":
        extra.update(
            {
                "cylinder_bevel_top": _cylinder_bevel_top_value(node),
                "cylinder_bevel_bottom": _cylinder_bevel_bottom_value(node),
                "cylinder_taper": float(getattr(node, "cylinder_taper", 0.0)),
                "cylinder_bevel_mode": str(getattr(node, "cylinder_bevel_mode", "SMOOTH") or "SMOOTH"),
            }
        )
    elif primitive_type == "cone":
        extra["cone_bevel_mode"] = str(getattr(node, "cone_bevel_mode", "SMOOTH") or "SMOOTH")
    elif primitive_type == "capsule":
        extra["capsule_taper"] = float(getattr(node, "capsule_taper", 0.0))
    elif primitive_type == "torus":
        extra["torus_angle"] = max(0.0, min(6.283185307179586, float(getattr(node, "torus_angle", 6.283185307179586))))
    elif primitive_type == "box":
        extra.update(
            {
                "box_corners": tuple(float(component) for component in getattr(node, "box_corners", (0.0, 0.0, 0.0, 0.0))),
                "box_edge_top": float(getattr(node, "box_edge_top", 0.0)),
                "box_edge_bottom": float(getattr(node, "box_edge_bottom", 0.0)),
                "box_taper": float(getattr(node, "box_taper", 0.0)),
                "box_corner_mode": str(getattr(node, "box_corner_mode", "SMOOTH") or "SMOOTH"),
                "box_edge_mode": str(getattr(node, "box_edge_mode", "SMOOTH") or "SMOOTH"),
            }
        )
    elif primitive_type == "ngon":
        extra.update(
            {
                "ngon_corner": float(getattr(node, "ngon_corner", 0.0)),
                "ngon_edge_top": float(getattr(node, "ngon_edge_top", 0.0)),
                "ngon_edge_bottom": float(getattr(node, "ngon_edge_bottom", 0.0)),
                "ngon_taper": float(getattr(node, "ngon_taper", 0.0)),
                "ngon_edge_mode": str(getattr(node, "ngon_edge_mode", "SMOOTH") or "SMOOTH"),
                "ngon_star": float(getattr(node, "ngon_star", 0.0)),
            }
        )
    elif primitive_type == "polygon":
        extra.update(
            {
                "polygon_edge_top": float(getattr(node, "polygon_edge_top", 0.0)),
                "polygon_edge_bottom": float(getattr(node, "polygon_edge_bottom", 0.0)),
                "polygon_taper": float(getattr(node, "polygon_taper", 0.0)),
                "polygon_interpolation": "BEZIER" if (_polygon_interpolation(node) == "BEZIER" and not bool(getattr(node, "polygon_is_line", False))) else "VECTOR",
                "polygon_edge_mode": str(getattr(node, "polygon_edge_mode", "SMOOTH") or "SMOOTH"),
                "polygon_is_line": bool(getattr(node, "polygon_is_line", False)),
                "polygon_line_thickness": float(getattr(node, "polygon_line_thickness", 0.1)),
            }
        )
    return extra


def _primitive_extra_rows(primitive_type, extra):
    row5 = [float(extra.get("effective_bevel", extra.get("bevel", 0.0))), 0.0, 0.0, 0.0]
    row6 = [0.0, 0.0, 0.0, 0.0]
    row7 = [0.0, 0.0, 0.0, 0.0]
    if primitive_type == "cylinder":
        row5[0] = float(extra.get("cylinder_bevel_top", 0.0))
        row5[1] = float(extra.get("cylinder_bevel_bottom", 0.0))
        row5[2] = float(extra.get("cylinder_taper", 0.0))
        row5[3] = _mode_to_id(extra.get("cylinder_bevel_mode", "SMOOTH"))
    elif primitive_type == "cone":
        row5[0] = float(extra.get("bevel", 0.0))
        row5[1] = _mode_to_id(extra.get("cone_bevel_mode", "SMOOTH"))
    elif primitive_type == "capsule":
        row5[1] = float(extra.get("capsule_taper", 0.0))
    elif primitive_type == "torus":
        row5[1] = float(extra.get("torus_angle", 6.283185307179586))
    elif primitive_type == "box":
        corners = tuple(float(component) for component in extra.get("box_corners", (0.0, 0.0, 0.0, 0.0)))
        row5[1:] = corners[:3]
        row6[0] = corners[3] if len(corners) > 3 else 0.0
        row6[1] = float(extra.get("box_edge_top", 0.0))
        row6[2] = float(extra.get("box_edge_bottom", 0.0))
        row6[3] = float(extra.get("box_taper", 0.0))
        row7[0] = _mode_to_id(extra.get("box_corner_mode", "SMOOTH"))
        row7[1] = _mode_to_id(extra.get("box_edge_mode", "SMOOTH"))
    elif primitive_type == "ngon":
        row5[1] = float(extra.get("ngon_corner", 0.0))
        row5[2] = float(extra.get("ngon_edge_top", 0.0))
        row5[3] = float(extra.get("ngon_edge_bottom", 0.0))
        row6[0] = float(extra.get("ngon_taper", 0.0))
        row6[1] = _mode_to_id(extra.get("ngon_edge_mode", "SMOOTH"))
        row6[2] = float(extra.get("ngon_star", 0.0))
    elif primitive_type == "polygon":
        row5[1] = float(extra.get("polygon_edge_top", 0.0))
        row5[2] = float(extra.get("polygon_edge_bottom", 0.0))
        row5[3] = float(extra.get("polygon_taper", 0.0))
        row6[0] = _mode_to_id(extra.get("polygon_edge_mode", "SMOOTH"))
        row6[1] = 1.0 if bool(extra.get("polygon_is_line", False)) else 0.0
        row6[2] = float(extra.get("polygon_line_thickness", 0.1))
        row6[3] = 1.0 if str(extra.get("polygon_interpolation", "VECTOR") or "VECTOR") == "BEZIER" else 0.0
    return tuple(row5), tuple(row6), tuple(row7)


def _effective_bevel(bevel, *dimensions):
    bevel_value = max(0.0, float(bevel))
    if bevel_value <= 0.0:
        return 0.0
    positive = [max(0.0, float(value)) for value in dimensions if float(value) > 0.0]
    if not positive:
        return 0.0
    return min(bevel_value, max(min(positive) - 0.001, 0.0))


def _render_meta_with_bevel(primitive_type, raw_meta, bevel):
    render_meta = list(raw_meta)
    effective_bevel = 0.0
    if primitive_type == "box":
        effective_bevel = _effective_bevel(bevel, raw_meta[1], raw_meta[2], raw_meta[3])
        render_meta[1] = max(raw_meta[1] - effective_bevel, 0.001)
        render_meta[2] = max(raw_meta[2] - effective_bevel, 0.001)
        render_meta[3] = max(raw_meta[3] - effective_bevel, 0.001)
    return tuple(render_meta), effective_bevel


def _primitive_parameters_from_node(node, polygon_rows=None):
    primitive_type = str(node.primitive_type or "sphere").strip().lower()
    raw_meta = [0.0, 0.0, 0.0, 0.0]
    raw_meta[0] = _PRIMITIVE_TYPE_TO_ID.get(primitive_type, 0.0)
    if primitive_type == "sphere":
        raw_meta[1] = _node_socket_float(node, _SOCKET_RADIUS, "radius", 0.5, min_value=0.001)
    elif primitive_type == "box":
        size = _node_socket_vector(node, _SOCKET_HALF_SIZE, "size", (0.5, 0.5, 0.5), _sanitized_scale)
        raw_meta[1] = float(size[0])
        raw_meta[2] = float(size[1])
        raw_meta[3] = float(size[2])
    elif primitive_type == "cylinder":
        raw_meta[1] = _node_socket_float(node, _SOCKET_RADIUS, "radius", 0.5, min_value=0.001)
        raw_meta[2] = _node_socket_float(node, _SOCKET_HEIGHT, "height", 1.0, min_value=0.001) * 0.5
    elif primitive_type == "torus":
        raw_meta[1] = _node_socket_float(node, _SOCKET_MAJOR_RADIUS, "major_radius", 0.75, min_value=0.001)
        raw_meta[2] = _node_socket_float(node, _SOCKET_MINOR_RADIUS, "minor_radius", 0.25, min_value=0.001)
    elif primitive_type == "cone":
        legacy_bottom = _node_socket_float(node, _SOCKET_RADIUS, "radius", 0.5, min_value=0.0)
        bottom_socket = _input_socket(node, _SOCKET_BOTTOM_RADIUS, MathOPSV2FloatSocket.bl_idname)
        bottom_radius = _node_socket_float(node, _SOCKET_BOTTOM_RADIUS, "cone_bottom_radius", _cone_bottom_radius_value(node), min_value=0.0)
        top_radius = _node_socket_float(node, _SOCKET_TOP_RADIUS, "cone_top_radius", _cone_top_radius_value(node), min_value=0.0)
        if not _property_is_explicit(node, "cone_bottom_radius") and not bool(getattr(bottom_socket, "is_linked", False)):
            bottom_radius = legacy_bottom
        bottom_radius, top_radius = _safe_cone_radii(bottom_radius, top_radius)
        raw_meta[1] = bottom_radius
        raw_meta[2] = top_radius
        raw_meta[3] = _node_socket_float(node, _SOCKET_HEIGHT, "height", 1.0, min_value=0.001) * 0.5
    elif primitive_type == "capsule":
        raw_meta[1] = _node_socket_float(node, _SOCKET_RADIUS, "radius", 0.5, min_value=0.001)
        raw_meta[2] = _node_socket_float(node, _SOCKET_HEIGHT, "height", 1.0, min_value=0.001) * 0.5
    elif primitive_type == "ngon":
        raw_meta[1] = _node_socket_float(node, _SOCKET_RADIUS, "radius", 0.5, min_value=0.001)
        raw_meta[2] = _node_socket_float(node, _SOCKET_HEIGHT, "height", 1.0, min_value=0.001) * 0.5
        raw_meta[3] = float(max(3, int(getattr(node, "ngon_sides", 6))))
    elif primitive_type == "polygon":
        half_height = _node_socket_float(node, _SOCKET_HEIGHT, "height", 1.0, min_value=0.001) * 0.5
        point_offset, point_count, _points = _append_polygon_rows(node, polygon_rows if polygon_rows is not None else [])
        raw_meta[1] = float(point_offset)
        raw_meta[2] = float(point_count)
        raw_meta[3] = half_height
    else:
        raise RuntimeError(f"Unsupported primitive type '{primitive_type}'")
    _location, _rotation, scale = _node_transform_values(node)
    extra = _primitive_extra_data(node, primitive_type)
    render_meta, effective_bevel = _render_meta_with_bevel(primitive_type, raw_meta, extra.get("bevel", 0.0))
    extra["effective_bevel"] = effective_bevel
    extra["raw_meta"] = tuple(raw_meta)
    return primitive_type, tuple(raw_meta), tuple(render_meta), scale, extra


def _primitive_local_extents(primitive_type, meta, scale, node=None):
    if primitive_type == "sphere":
        return Vector((meta[1] * scale[0], meta[1] * scale[1], meta[1] * scale[2]))
    if primitive_type == "box":
        return Vector((meta[1] * scale[0], meta[2] * scale[1], meta[3] * scale[2]))
    if primitive_type == "cylinder":
        return Vector((meta[1] * scale[0], meta[1] * scale[1], meta[2] * scale[2]))
    if primitive_type == "torus":
        return Vector(
            (
                (meta[1] + meta[2]) * scale[0],
                (meta[1] + meta[2]) * scale[1],
                meta[2] * scale[2],
            )
        )
    if primitive_type == "cone":
        max_radius = max(float(meta[1]), float(meta[2]), 0.001)
        return Vector((max_radius * scale[0], max_radius * scale[1], meta[3] * scale[2]))
    if primitive_type == "capsule":
        bottom_radius, top_radius = _capsule_end_radii(meta[1], getattr(node, "capsule_taper", 0.0) if node is not None else 0.0)
        max_radius = max(bottom_radius, top_radius, 0.001)
        return Vector((max_radius * scale[0], max_radius * scale[1], (meta[2] + max_radius) * scale[2]))
    if primitive_type == "ngon":
        return Vector((meta[1] * scale[0], meta[1] * scale[1], meta[2] * scale[2]))
    if primitive_type == "polygon":
        points = polygon_point_coords(node, ensure_default=True) if node is not None else default_polygon_points()
        if not points:
            return Vector((scale[0], scale[1], meta[3] * scale[2]))
        max_x = max(abs(float(point[0])) for point in points)
        max_y = max(abs(float(point[1])) for point in points)
        return Vector((max_x * scale[0], max_y * scale[1], meta[3] * scale[2]))
    return Vector((1.0, 1.0, 1.0))


def _primitive_lipschitz(primitive_type, scale):
    if primitive_type == "sphere":
        min_scale = max(min(scale[0], scale[1], scale[2]), 1.0e-6)
        return max(scale[0], scale[1], scale[2]) / min_scale
    return 1.0


def _mirror_axis_flags(node):
    flags = 0
    if bool(getattr(node, "mirror_x", False)):
        flags |= _MIRROR_AXIS_X
    if bool(getattr(node, "mirror_y", False)):
        flags |= _MIRROR_AXIS_Y
    if bool(getattr(node, "mirror_z", False)):
        flags |= _MIRROR_AXIS_Z
    return flags


def _mirror_origin_world(origin_object):
    try:
        origin_object = runtime.object_identity(origin_object)
        if origin_object is None:
            return _IDENTITY_LOCATION
        return _sanitized_location(origin_object.matrix_world.translation)
    except ReferenceError:
        return _IDENTITY_LOCATION


def _array_frame_transform_values(frame_node):
    try:
        if frame_node is None or not bool(getattr(frame_node, "use_array_transform", False)):
            return None
        return _array_node_transform_values(frame_node)
    except ReferenceError:
        return None


def _packed_quaternion_xyz(matrix):
    rotation = matrix.to_3x3().normalized().to_quaternion()
    sign = -1.0 if float(rotation.w) < 0.0 else 1.0
    return (
        float(rotation.x) * sign,
        float(rotation.y) * sign,
        float(rotation.z) * sign,
    )


def _packed_euler_xyz(rotation):
    return _packed_quaternion_xyz(Matrix.LocRotScale(Vector((0.0, 0.0, 0.0)), Euler(rotation, "XYZ"), None))


def _array_transform_matrix(location, rotation, scale):
    return Matrix.LocRotScale(Vector(location), Euler(rotation, "XYZ"), Vector(scale))


def _array_offset_matrix(location, rotation, scale, branch_center):
    return (
        Matrix.Translation(Vector(branch_center) + Vector(location))
        @ _array_transform_matrix(_IDENTITY_LOCATION, rotation, scale)
        @ Matrix.Translation(-Vector(branch_center))
    )


def _array_field_matrix(location, rotation, scale, branch_center):
    return (
        Matrix.Translation(Vector(location))
        @ _array_transform_matrix(_IDENTITY_LOCATION, rotation, scale)
        @ Matrix.Translation(-Vector(branch_center))
    )


def _array_frame_matrix(frame_node, branch_center):
    frame_transform = _array_frame_transform_values(frame_node)
    if frame_transform is None:
        return Matrix.Identity(4)
    location, rotation, scale = frame_transform
    return _array_field_matrix(location, rotation, scale, branch_center)


def _transform_point(matrix, point):
    result = matrix @ Vector((float(point[0]), float(point[1]), float(point[2]), 1.0))
    return (float(result[0]), float(result[1]), float(result[2]))


def _array_frame_origin_world(frame_node, branch_center):
    frame_transform = _array_frame_transform_values(frame_node)
    if frame_transform is not None:
        return frame_transform[0]
    return tuple(float(component) for component in branch_center)


def _array_repeat_origin_local(origin_object, primitive_center, frame_node=None):
    try:
        origin_object = runtime.object_identity(origin_object)
        if origin_object is None:
            return tuple(float(component) for component in primitive_center)
        world_origin = _sanitized_location(origin_object.matrix_world.translation)
    except ReferenceError:
        return tuple(float(component) for component in primitive_center)
    inverse = _array_frame_matrix(frame_node, primitive_center).inverted_safe()
    return _transform_point(inverse, world_origin)


def _array_rotation_world(frame_node=None):
    frame_transform = _array_frame_transform_values(frame_node)
    if frame_transform is not None:
        _location, rotation, _scale = frame_transform
        return _packed_euler_xyz(rotation)
    return (0.0, 0.0, 0.0)


def _array_scale_world(frame_node=None):
    frame_transform = _array_frame_transform_values(frame_node)
    if frame_transform is not None:
        _location, _rotation, scale = frame_transform
        return _sanitized_scale(scale)
    return _IDENTITY_SCALE


def _transform_bounds(bounds_min, bounds_max, matrix):
    corners = []
    for x in (float(bounds_min[0]), float(bounds_max[0])):
        for y in (float(bounds_min[1]), float(bounds_max[1])):
            for z in (float(bounds_min[2]), float(bounds_max[2])):
                corners.append(_transform_point(matrix, (x, y, z)))
    mins = [min(corner[axis] for corner in corners) for axis in range(3)]
    maxs = [max(corner[axis] for corner in corners) for axis in range(3)]
    return tuple(mins), tuple(maxs)


def _bounds_center(bounds):
    if bounds is None:
        return _IDENTITY_LOCATION
    return tuple(0.5 * (float(bounds[0][axis]) + float(bounds[1][axis])) for axis in range(3))


def _offset_center(branch_center, offset_location):
    return tuple(float(branch_center[axis]) + float(offset_location[axis]) for axis in range(3))


def _mirror_warp_descriptor(node):
    flags = _mirror_axis_flags(node)
    if flags == 0:
        return None
    blend = max(0.0, float(getattr(node, "blend", 0.0)))
    return ("mirror", int(flags), runtime.object_identity(getattr(node, "origin_object", None)), blend)


def _array_warp_descriptor(node, source_node=None, branch_center=None):
    mode = str(getattr(node, "array_mode", _ARRAY_MODE_GRID) or _ARRAY_MODE_GRID)
    blend = max(0.0, float(getattr(node, "blend", 0.0)))
    origin_object = runtime.object_identity(getattr(node, "origin_object", None))
    frame_node = node
    if mode == _ARRAY_MODE_RADIAL:
        count = max(1, int(getattr(node, "radial_count", 1)))
        radius = max(0.0, float(getattr(node, "radius", 0.0)))
        if count <= 1 or radius <= 1.0e-6:
            return None
        return ("radial", count, radius, origin_object, frame_node, source_node, branch_center, blend)

    count_x = max(1, int(getattr(node, "count_x", 1)))
    count_y = max(1, int(getattr(node, "count_y", 1)))
    count_z = max(1, int(getattr(node, "count_z", 1)))
    spacing_values = getattr(node, "spacing", (1.0, 1.0, 1.0))
    spacing = tuple(abs(float(component)) for component in spacing_values)
    has_spread = (
        (count_x > 1 and spacing[0] > 1.0e-6)
        or (count_y > 1 and spacing[1] > 1.0e-6)
        or (count_z > 1 and spacing[2] > 1.0e-6)
    )
    if not has_spread:
        return None
    return ("grid", count_x, count_y, count_z, spacing, None, frame_node, source_node, branch_center, blend)


def _subtree_bounds(node, warps=(), visiting=None):
    if node is None:
        return None
    if visiting is None:
        visiting = set()
    node_key = int(node.as_pointer())
    if node_key in visiting:
        raise RuntimeError(f"Cycle detected at node '{node.name}'")

    visiting.add(node_key)
    try:
        node_idname = getattr(node, "bl_idname", "")
        if node_idname == runtime.OBJECT_NODE_IDNAME:
            spec = _primitive_spec_from_node(node, 0, warps=warps)
            return tuple(spec["bounds_min"]), tuple(spec["bounds_max"])

        if node_idname == MIRROR_NODE_IDNAME:
            _ensure_mirror_node_sockets(node)
            source = _linked_source_node(_input_socket(node, "SDF", MathOPSV2SDFSocket.bl_idname))
            if source is None:
                return None
            mirror_warp = _mirror_warp_descriptor(node)
            next_warps = tuple(warps) if mirror_warp is None else tuple(warps) + (mirror_warp,)
            return _subtree_bounds(source, warps=next_warps, visiting=visiting)

        if node_idname == ARRAY_NODE_IDNAME:
            _ensure_array_node_sockets(node)
            source = _linked_source_node(_input_socket(node, "SDF", MathOPSV2SDFSocket.bl_idname))
            if source is None:
                return None
            array_warp = _array_warp_descriptor(node, source_node=source)
            next_warps = tuple(warps) if array_warp is None else _resolve_warp_stack(tuple(warps) + (array_warp,))
            return _subtree_bounds(source, warps=next_warps, visiting=visiting)

        if node_idname == runtime.CSG_NODE_IDNAME:
            _ensure_csg_node_sockets(node)
            left = _linked_source_node(_input_socket(node, "A", MathOPSV2SDFSocket.bl_idname))
            right = _linked_source_node(_input_socket(node, "B", MathOPSV2SDFSocket.bl_idname))
            if left is None or right is None:
                return None
            left_bounds = _subtree_bounds(left, warps=warps, visiting=visiting)
            right_bounds = _subtree_bounds(right, warps=warps, visiting=visiting)
            operation = _CSG_OPERATION_TO_ID.get(getattr(node, "operation", "UNION"), 1.0)
            if operation == int(_CSG_OPERATION_TO_ID["UNION"]):
                bounds = _union_bounds(left_bounds, right_bounds)
                return _expand_bounds(bounds, _blend_bounds_padding(getattr(node, "blend", 0.0)))
            if operation == int(_CSG_OPERATION_TO_ID["SUBTRACT"]):
                return left_bounds
            if operation == int(_CSG_OPERATION_TO_ID["INTERSECT"]):
                return _intersect_bounds(left_bounds, right_bounds)
            return _union_bounds(left_bounds, right_bounds)

        return None
    finally:
        visiting.remove(node_key)


def _resolve_array_warp(warp, prefix_warps=()):
    kind = warp[0]
    if kind == "grid":
        if len(warp) >= 13:
            _kind, count_x, count_y, count_z, spacing, origin_object, frame_node, source_node, _branch_center, _offset_location, _offset_rotation, _offset_scale, blend = warp
        else:
            _kind, count_x, count_y, count_z, spacing, origin_object, frame_node, source_node, _branch_center, blend = warp
        branch_center = _bounds_center(_subtree_bounds(source_node, warps=tuple(prefix_warps)))
        offset_location, offset_rotation, offset_scale = _array_transform_offset_values(frame_node)
        return (
            "grid",
            count_x,
            count_y,
            count_z,
            spacing,
            origin_object,
            frame_node,
            source_node,
            branch_center,
            offset_location,
            offset_rotation,
            offset_scale,
            blend,
        )
    if kind == "radial":
        if len(warp) >= 11:
            _kind, count, radius, origin_object, frame_node, source_node, _branch_center, _offset_location, _offset_rotation, _offset_scale, blend = warp
        else:
            _kind, count, radius, origin_object, frame_node, source_node, _branch_center, blend = warp
        branch_center = _bounds_center(_subtree_bounds(source_node, warps=tuple(prefix_warps)))
        offset_location, offset_rotation, offset_scale = _array_transform_offset_values(frame_node)
        return (
            "radial",
            count,
            radius,
            origin_object,
            frame_node,
            source_node,
            branch_center,
            offset_location,
            offset_rotation,
            offset_scale,
            blend,
        )
    return warp


def _resolve_warp_stack(warps):
    resolved = []
    for warp in tuple(warps or ()):
        kind = warp[0]
        if kind in {"grid", "radial"}:
            resolved.append(_resolve_array_warp(warp, tuple(resolved)))
        else:
            resolved.append(warp)
    return tuple(resolved)


def _warp_signature(warps):
    signature = []
    for warp in warps:
        kind = warp[0]
        if kind == "mirror":
            _kind, flags, origin_object, blend = warp
            signature.append((kind, int(flags), runtime.object_key(origin_object), round(float(blend), 6)))
            continue
        if kind == "grid":
            _kind, count_x, count_y, count_z, spacing, origin_object, frame_node, source_node, _branch_center, offset_location, offset_rotation, offset_scale, blend = warp
            signature.append(
                (
                    kind,
                    int(count_x),
                    int(count_y),
                    int(count_z),
                    tuple(round(float(component), 6) for component in spacing),
                    runtime.object_key(origin_object),
                    runtime.safe_pointer(frame_node),
                    runtime.safe_pointer(source_node),
                    tuple(round(float(component), 6) for component in offset_location),
                    tuple(round(float(component), 6) for component in offset_rotation),
                    tuple(round(float(component), 6) for component in offset_scale),
                    round(float(blend), 6),
                )
            )
            continue
        if kind == "radial":
            _kind, count, radius, origin_object, frame_node, source_node, _branch_center, offset_location, offset_rotation, offset_scale, blend = warp
            signature.append(
                (
                    kind,
                    int(count),
                    round(float(radius), 6),
                    runtime.object_key(origin_object),
                    runtime.safe_pointer(frame_node),
                    runtime.safe_pointer(source_node),
                    tuple(round(float(component), 6) for component in offset_location),
                    tuple(round(float(component), 6) for component in offset_rotation),
                    tuple(round(float(component), 6) for component in offset_scale),
                    round(float(blend), 6),
                )
            )
    return tuple(signature)


def _blend_bounds_padding(blend):
    return max(0.0, float(blend)) * 0.25


def _apply_mirror_warp_to_bounds(bounds_min, bounds_max, flags, origin, blend=0.0):
    mins = [float(value) for value in bounds_min]
    maxs = [float(value) for value in bounds_max]
    padding = _blend_bounds_padding(blend)
    for axis, mask in enumerate((_MIRROR_AXIS_X, _MIRROR_AXIS_Y, _MIRROR_AXIS_Z)):
        if (flags & mask) == 0:
            continue
        center = float(origin[axis])
        extent = max(abs(mins[axis] - center), abs(maxs[axis] - center))
        mins[axis] = center - extent - padding
        maxs[axis] = center + extent + padding
    return tuple(mins), tuple(maxs)


def _apply_grid_warp_to_bounds(bounds_min, bounds_max, count_x, count_y, count_z, spacing, blend=0.0):
    mins = [float(value) for value in bounds_min]
    maxs = [float(value) for value in bounds_max]
    counts = (int(count_x), int(count_y), int(count_z))
    padding = _blend_bounds_padding(blend)
    span = Vector((
        abs(float(spacing[0])) * 0.5 * max(counts[0] - 1, 0),
        abs(float(spacing[1])) * 0.5 * max(counts[1] - 1, 0),
        abs(float(spacing[2])) * 0.5 * max(counts[2] - 1, 0),
    ))
    span_radius = float(span.length) + padding
    for axis in range(3):
        mins[axis] -= span_radius
        maxs[axis] += span_radius
    return tuple(mins), tuple(maxs)


def _apply_radial_warp_to_bounds(bounds_min, bounds_max, origin, radius, blend=0.0):
    mins = [float(value) for value in bounds_min]
    maxs = [float(value) for value in bounds_max]
    center_x = 0.5 * (mins[0] + maxs[0])
    center_y = 0.5 * (mins[1] + maxs[1])
    center_z = 0.5 * (mins[2] + maxs[2])
    extent_x = 0.5 * max(maxs[0] - mins[0], 0.0)
    extent_y = 0.5 * max(maxs[1] - mins[1], 0.0)
    extent_z = 0.5 * max(maxs[2] - mins[2], 0.0)
    radial_extent = (extent_x * extent_x + extent_y * extent_y + extent_z * extent_z) ** 0.5
    base_dx = center_x - float(origin[0]) + float(radius)
    base_dy = center_y - float(origin[1])
    base_dz = center_z - float(origin[2])
    ring_radius = (base_dx * base_dx + base_dy * base_dy + base_dz * base_dz) ** 0.5
    total_radius = ring_radius + radial_extent + _blend_bounds_padding(blend)
    for axis in range(3):
        mins[axis] = float(origin[axis]) - total_radius
        maxs[axis] = float(origin[axis]) + total_radius
    return tuple(mins), tuple(maxs)


def _apply_warp_stack_to_bounds(bounds_min, bounds_max, warps):
    current_min = tuple(bounds_min)
    current_max = tuple(bounds_max)
    for warp in warps:
        kind = warp[0]
        if kind == "mirror":
            _kind, flags, origin_object, blend = warp
            current_min, current_max = _apply_mirror_warp_to_bounds(
                current_min,
                current_max,
                int(flags),
                _mirror_origin_world(origin_object),
                blend,
            )
            continue
        current_center = tuple(0.5 * (current_min[axis] + current_max[axis]) for axis in range(3))
        if kind == "grid":
            _kind, count_x, count_y, count_z, spacing, _origin_object, frame_node, _source_node, branch_center, offset_location, offset_rotation, offset_scale, blend = warp
            current_min, current_max = _transform_bounds(
                current_min,
                current_max,
                _array_offset_matrix(offset_location, offset_rotation, offset_scale, branch_center),
            )
            current_min, current_max = _apply_grid_warp_to_bounds(
                current_min,
                current_max,
                count_x,
                count_y,
                count_z,
                spacing,
                blend,
            )
            current_min, current_max = _transform_bounds(current_min, current_max, _array_frame_matrix(frame_node, branch_center))
            continue
        if kind == "radial":
            _kind, _count, radius, origin_object, frame_node, _source_node, branch_center, offset_location, offset_rotation, offset_scale, blend = warp
            current_min, current_max = _transform_bounds(
                current_min,
                current_max,
                _array_offset_matrix(offset_location, offset_rotation, offset_scale, branch_center),
            )
            current_min, current_max = _apply_radial_warp_to_bounds(
                current_min,
                current_max,
                _array_repeat_origin_local(origin_object, branch_center, frame_node),
                radius,
                blend,
            )
            current_min, current_max = _transform_bounds(current_min, current_max, _array_frame_matrix(frame_node, branch_center))
    return current_min, current_max


def _append_warp_rows(warps, warp_rows, primitive_center=_IDENTITY_LOCATION):
    if warp_rows is None or not warps:
        return 0, 0
    warp_offset = len(warp_rows)
    center = tuple(float(component) for component in primitive_center)
    for warp in warps:
        kind = warp[0]
        if kind == "mirror":
            _kind, flags, origin_object, blend = warp
            origin = _mirror_origin_world(origin_object)
            packed_flags = int(flags)
            if (packed_flags & _MIRROR_AXIS_X) != 0 and float(center[0]) >= float(origin[0]):
                packed_flags |= _MIRROR_SIDE_X
            if (packed_flags & _MIRROR_AXIS_Y) != 0 and float(center[1]) >= float(origin[1]):
                packed_flags |= _MIRROR_SIDE_Y
            if (packed_flags & _MIRROR_AXIS_Z) != 0 and float(center[2]) >= float(origin[2]):
                packed_flags |= _MIRROR_SIDE_Z
            warp_rows.append((float(_WARP_KIND_MIRROR), float(packed_flags), float(blend), 0.0))
            warp_rows.append((float(origin[0]), float(origin[1]), float(origin[2]), 0.0))
            warp_rows.append((0.0, 0.0, 0.0, 0.0))
            warp_rows.append((0.0, 0.0, 0.0, 0.0))
            warp_rows.append((0.0, 0.0, 0.0, 0.0))
            warp_rows.append((0.0, 0.0, 0.0, 0.0))
            warp_rows.append((0.0, 0.0, 0.0, 0.0))
            warp_rows.append((0.0, 0.0, 0.0, 0.0))
            warp_rows.append((0.0, 0.0, 0.0, 0.0))
            continue
        if kind == "grid":
            _kind, count_x, count_y, count_z, spacing, _origin_object, frame_node, _source_node, branch_center, offset_location, offset_rotation, offset_scale, blend = warp
            origin = _array_frame_origin_world(frame_node, branch_center)
            rotation = _array_rotation_world(frame_node)
            scale = _array_scale_world(frame_node)
            warp_rows.append((float(_WARP_KIND_GRID), float(count_x), float(count_y), float(count_z)))
            warp_rows.append((float(spacing[0]), float(spacing[1]), float(spacing[2]), float(blend)))
            warp_rows.append((float(origin[0]), float(origin[1]), float(origin[2]), 0.0))
            warp_rows.append((float(rotation[0]), float(rotation[1]), float(rotation[2]), 0.0))
            warp_rows.append((float(scale[0]), float(scale[1]), float(scale[2]), 0.0))
            warp_rows.append((float(branch_center[0]), float(branch_center[1]), float(branch_center[2]), 0.0))
            warp_rows.append((float(offset_location[0]), float(offset_location[1]), float(offset_location[2]), 0.0))
            warp_rows.append((float(_packed_euler_xyz(offset_rotation)[0]), float(_packed_euler_xyz(offset_rotation)[1]), float(_packed_euler_xyz(offset_rotation)[2]), 0.0))
            warp_rows.append((float(offset_scale[0]), float(offset_scale[1]), float(offset_scale[2]), 0.0))
            continue
        if kind == "radial":
            _kind, count, radius, origin_object, frame_node, _source_node, branch_center, offset_location, offset_rotation, offset_scale, blend = warp
            repeat_origin = _array_repeat_origin_local(origin_object, branch_center, frame_node)
            field_origin = _array_frame_origin_world(frame_node, branch_center)
            rotation = _array_rotation_world(frame_node)
            scale = _array_scale_world(frame_node)
            warp_rows.append((float(_WARP_KIND_RADIAL), float(count), float(blend), float(radius)))
            warp_rows.append((float(repeat_origin[0]), float(repeat_origin[1]), float(repeat_origin[2]), 0.0))
            warp_rows.append((float(field_origin[0]), float(field_origin[1]), float(field_origin[2]), 0.0))
            warp_rows.append((float(rotation[0]), float(rotation[1]), float(rotation[2]), 0.0))
            warp_rows.append((float(scale[0]), float(scale[1]), float(scale[2]), 0.0))
            warp_rows.append((float(branch_center[0]), float(branch_center[1]), float(branch_center[2]), 0.0))
            warp_rows.append((float(offset_location[0]), float(offset_location[1]), float(offset_location[2]), 0.0))
            warp_rows.append((float(_packed_euler_xyz(offset_rotation)[0]), float(_packed_euler_xyz(offset_rotation)[1]), float(_packed_euler_xyz(offset_rotation)[2]), 0.0))
            warp_rows.append((float(offset_scale[0]), float(offset_scale[1]), float(offset_scale[2]), 0.0))
    return warp_offset, len(warps)


def _pack_warp_info(warp_offset, warp_count):
    offset = max(0, int(warp_offset))
    count = max(0, int(warp_count))
    if count >= _MIRROR_WARP_PACK_SCALE:
        raise RuntimeError("Warp stack exceeds shader packing limit")
    return float(offset * _MIRROR_WARP_PACK_SCALE + count)


def _primitive_spec_from_node(node, primitive_index, warps=()):
    location, _rotation, _scale = _node_transform_values(node)
    primitive_type, raw_meta, render_meta, scale, extra = _primitive_parameters_from_node(node)
    transform = _node_transform_matrix(node)
    world_to_local = transform.inverted_safe()
    rows = [tuple(float(value) for value in world_to_local[index]) for index in range(3)]
    center = location
    rotation = transform.to_3x3()
    polygon_points = ()
    if primitive_type == "polygon":
        polygon_points = effective_polygon_point_data(node, ensure_default=True)
        half_height = float(raw_meta[3])
        local_min_x = min(float(point[0]) for point in polygon_points) * float(scale[0])
        local_max_x = max(float(point[0]) for point in polygon_points) * float(scale[0])
        local_min_y = min(float(point[1]) for point in polygon_points) * float(scale[1])
        local_max_y = max(float(point[1]) for point in polygon_points) * float(scale[1])
        local_corners = (
            Vector((local_min_x, local_min_y, -half_height * float(scale[2]))),
            Vector((local_min_x, local_min_y, half_height * float(scale[2]))),
            Vector((local_min_x, local_max_y, -half_height * float(scale[2]))),
            Vector((local_min_x, local_max_y, half_height * float(scale[2]))),
            Vector((local_max_x, local_min_y, -half_height * float(scale[2]))),
            Vector((local_max_x, local_min_y, half_height * float(scale[2]))),
            Vector((local_max_x, local_max_y, -half_height * float(scale[2]))),
            Vector((local_max_x, local_max_y, half_height * float(scale[2]))),
        )
        world_corners = [Vector(center) + (rotation @ corner) for corner in local_corners]
        bounds_min = tuple(min(float(corner[axis]) for corner in world_corners) for axis in range(3))
        bounds_max = tuple(max(float(corner[axis]) for corner in world_corners) for axis in range(3))
    else:
        local_extents = _primitive_local_extents(primitive_type, raw_meta, scale, node=node)
        world_extents = Vector((
            abs(rotation[0][0]) * local_extents[0] + abs(rotation[0][1]) * local_extents[1] + abs(rotation[0][2]) * local_extents[2],
            abs(rotation[1][0]) * local_extents[0] + abs(rotation[1][1]) * local_extents[1] + abs(rotation[1][2]) * local_extents[2],
            abs(rotation[2][0]) * local_extents[0] + abs(rotation[2][1]) * local_extents[1] + abs(rotation[2][2]) * local_extents[2],
        ))
        bounds_min = tuple(center[axis] - float(world_extents[axis]) for axis in range(3))
        bounds_max = tuple(center[axis] + float(world_extents[axis]) for axis in range(3))
    bounds_min, bounds_max = _apply_warp_stack_to_bounds(bounds_min, bounds_max, warps)
    spec = {
        "primitive_index": int(primitive_index),
        "primitive_type": primitive_type,
        "meta": render_meta,
        "extra": extra,
        "world_to_local": tuple(rows),
        "scale": tuple(scale),
        "center": center,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "lipschitz": float(_primitive_lipschitz(primitive_type, scale)),
    }
    if primitive_type == "polygon":
        spec["polygon_points"] = tuple((float(point[0]), float(point[1])) for point in polygon_points)
        spec["polygon_control_points"] = tuple(
            (
                float(point[0]),
                float(point[1]),
                float(point[3]),
                float(point[4]),
                float(point[5]),
                float(point[6]),
            )
            for point in effective_polygon_control_point_data(node, ensure_default=True)
        )
    return spec


def _union_bounds(bounds_a, bounds_b):
    if bounds_a is None:
        return bounds_b
    if bounds_b is None:
        return bounds_a
    return (
        tuple(min(float(bounds_a[0][axis]), float(bounds_b[0][axis])) for axis in range(3)),
        tuple(max(float(bounds_a[1][axis]), float(bounds_b[1][axis])) for axis in range(3)),
    )


def _intersect_bounds(bounds_a, bounds_b):
    if bounds_a is None:
        return bounds_b
    if bounds_b is None:
        return bounds_a

    mins = [max(float(bounds_a[0][axis]), float(bounds_b[0][axis])) for axis in range(3)]
    maxs = [min(float(bounds_a[1][axis]), float(bounds_b[1][axis])) for axis in range(3)]
    for axis in range(3):
        if maxs[axis] < mins[axis]:
            center = 0.5 * (mins[axis] + maxs[axis])
            mins[axis] = center
            maxs[axis] = center
    return tuple(mins), tuple(maxs)


def _expand_bounds(bounds, padding):
    if bounds is None:
        return None
    expand = max(0.0, float(padding))
    if expand <= 1.0e-6:
        return bounds
    return (
        tuple(float(bounds[0][axis]) - expand for axis in range(3)),
        tuple(float(bounds[1][axis]) + expand for axis in range(3)),
    )


def _root_node_bounds(root_node, primitive_specs):
    if root_node is None:
        return None

    kind = root_node.get("kind")
    if kind == "primitive":
        primitive_index = int(root_node.get("primitive_index", -1))
        if primitive_index < 0 or primitive_index >= len(primitive_specs):
            return None
        spec = primitive_specs[primitive_index]
        return tuple(spec["bounds_min"]), tuple(spec["bounds_max"])

    if kind != "op":
        return None

    op = int(root_node.get("op", 0))
    left_bounds = _root_node_bounds(root_node.get("left"), primitive_specs)
    right_bounds = _root_node_bounds(root_node.get("right"), primitive_specs)
    if op == int(_CSG_OPERATION_TO_ID["UNION"]):
        bounds = _union_bounds(left_bounds, right_bounds)
        return _expand_bounds(bounds, _blend_bounds_padding(root_node.get("blend", 0.0)))
    if op == int(_CSG_OPERATION_TO_ID["SUBTRACT"]):
        return left_bounds
    if op == int(_CSG_OPERATION_TO_ID["INTERSECT"]):
        return _intersect_bounds(left_bounds, right_bounds)
    return _union_bounds(left_bounds, right_bounds)


def _scene_bounds(root_node, primitive_specs):
    tight_bounds = _tight_scene_bounds(root_node, primitive_specs)
    mins = [float(tight_bounds[0][axis]) for axis in range(3)]
    maxs = [float(tight_bounds[1][axis]) for axis in range(3)]

    extents = [maxs[axis] - mins[axis] for axis in range(3)]
    max_extent = max(extents) if extents else 1.0
    padding = max(0.5, max_extent * 0.1)
    return (
        tuple(mins[axis] - padding for axis in range(3)),
        tuple(maxs[axis] + padding for axis in range(3)),
    )


def _tight_scene_bounds(root_node, primitive_specs):
    if not primitive_specs:
        return ((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))

    bounds = _root_node_bounds(root_node, primitive_specs)
    if bounds is None:
        mins = [float("inf"), float("inf"), float("inf")]
        maxs = [float("-inf"), float("-inf"), float("-inf")]
        for spec in primitive_specs:
            for axis in range(3):
                mins[axis] = min(mins[axis], float(spec["bounds_min"][axis]))
                maxs[axis] = max(maxs[axis], float(spec["bounds_max"][axis]))
    else:
        mins = [float(bounds[0][axis]) for axis in range(3)]
        maxs = [float(bounds[1][axis]) for axis in range(3)]

    for axis in range(3):
        if (maxs[axis] - mins[axis]) < 1.0e-4:
            mins[axis] -= 5.0e-5
            maxs[axis] += 5.0e-5
    return (
        tuple(mins[axis] for axis in range(3)),
        tuple(maxs[axis] for axis in range(3)),
    )


def _primitive_rows_from_node(node, warps=(), warp_rows=None, polygon_rows=None):
    primitive_type, _raw_meta, render_meta, scale, extra = _primitive_parameters_from_node(node, polygon_rows=polygon_rows)
    location, _rotation, _node_scale = _node_transform_values(node)
    transform = _node_transform_matrix(node).inverted_safe()
    rows = [tuple(float(value) for value in transform[index]) for index in range(3)]
    warp_offset, warp_count = _append_warp_rows(warps, warp_rows, primitive_center=location)
    extra_rows = _primitive_extra_rows(primitive_type, extra)
    return [
        tuple(render_meta),
        rows[0],
        rows[1],
        rows[2],
        (scale[0], scale[1], scale[2], _pack_warp_info(warp_offset, warp_count)),
        extra_rows[0],
        extra_rows[1],
        extra_rows[2],
    ]


def _node_key(node):
    proxy_id = str(getattr(node, "proxy_id", "") or "")
    if proxy_id:
        return proxy_id
    return str(node.as_pointer())


def _append_instruction(instructions, row):
    instructions.append(tuple(float(value) for value in row))


def _stack_usage(instructions):
    depth = 0
    max_depth = 0
    for instruction in instructions:
        kind = int(float(instruction[0]))
        if kind == 0:
            depth += 1
            if depth > max_depth:
                max_depth = depth
            continue
        depth -= 1
        if depth <= 0:
            raise RuntimeError("Graph emitted an invalid instruction sequence")
    if depth != 1 and instructions:
        raise RuntimeError("Graph emitted an incomplete instruction sequence")
    return max_depth


def _emit_node(node, primitive_rows, polygon_rows, warp_rows, instructions, primitive_map, primitive_specs, primitive_entries, visiting, warps=()):
    node_key = node.as_pointer()
    if node_key in visiting:
        raise RuntimeError(f"Cycle detected at node '{node.name}'")

    visiting.add(node_key)
    try:
        node_idname = getattr(node, "bl_idname", "")
        if node_idname == runtime.OBJECT_NODE_IDNAME:
            ensure_object_node_id(node)
            proxy_key = (_node_key(node), _warp_signature(warps))
            primitive_index = primitive_map.get(proxy_key)
            if primitive_index is None:
                primitive_index = len(primitive_entries)
                primitive_map[proxy_key] = primitive_index
                primitive_rows.extend(_primitive_rows_from_node(node, warps=warps, warp_rows=warp_rows, polygon_rows=polygon_rows))
                primitive_specs.append(_primitive_spec_from_node(node, primitive_index, warps=warps))
                primitive_entries.append({"node": node, "warps": tuple(warps), "object": getattr(node, "target", None)})
            instruction_index = len(instructions)
            _append_instruction(instructions, (0.0, float(primitive_index), 0.0, 0.0))
            return {
                "kind": "primitive",
                "primitive_index": int(primitive_index),
                "instruction_index": int(instruction_index),
            }

        if node_idname == MIRROR_NODE_IDNAME:
            _ensure_mirror_node_sockets(node)
            source = _linked_source_node(_input_socket(node, "SDF", MathOPSV2SDFSocket.bl_idname))
            if source is None:
                raise RuntimeError(f"Node '{node.name}' needs its SDF input connected")
            mirror_warp = _mirror_warp_descriptor(node)
            next_warps = warps if mirror_warp is None else (tuple(warps) + (mirror_warp,))
            return _emit_node(
                source,
                primitive_rows,
                polygon_rows,
                warp_rows,
                instructions,
                primitive_map,
                primitive_specs,
                primitive_entries,
                visiting,
                warps=next_warps,
            )

        if node_idname == ARRAY_NODE_IDNAME:
            _ensure_array_node_sockets(node)
            source = _linked_source_node(_input_socket(node, "SDF", MathOPSV2SDFSocket.bl_idname))
            if source is None:
                raise RuntimeError(f"Node '{node.name}' needs its SDF input connected")
            array_warp = _array_warp_descriptor(node, source_node=source)
            next_warps = tuple(warps) if array_warp is None else _resolve_warp_stack(tuple(warps) + (array_warp,))
            return _emit_node(
                source,
                primitive_rows,
                polygon_rows,
                warp_rows,
                instructions,
                primitive_map,
                primitive_specs,
                primitive_entries,
                visiting,
                warps=next_warps,
            )

        if node_idname == runtime.CSG_NODE_IDNAME:
            _ensure_csg_node_sockets(node)
            left = _linked_source_node(_input_socket(node, "A", MathOPSV2SDFSocket.bl_idname))
            right = _linked_source_node(_input_socket(node, "B", MathOPSV2SDFSocket.bl_idname))
            if left is None or right is None:
                raise RuntimeError(f"Node '{node.name}' needs both inputs connected")
            left_compiled = _emit_node(
                left,
                primitive_rows,
                polygon_rows,
                warp_rows,
                instructions,
                primitive_map,
                primitive_specs,
                primitive_entries,
                visiting,
                warps=warps,
            )
            right_compiled = _emit_node(
                right,
                primitive_rows,
                polygon_rows,
                warp_rows,
                instructions,
                primitive_map,
                primitive_specs,
                primitive_entries,
                visiting,
                warps=warps,
            )
            operation = _CSG_OPERATION_TO_ID.get(getattr(node, "operation", "UNION"), 1.0)
            blend = _node_socket_float(node, _SOCKET_BLEND, "blend", 0.0, min_value=0.0)
            instruction_index = len(instructions)
            _append_instruction(
                instructions,
                (
                    operation,
                    0.0,
                    0.0,
                    blend,
                ),
            )
            return {
                "kind": "op",
                "op": int(operation),
                "blend": blend,
                "instruction_index": int(instruction_index),
                "left": left_compiled,
                "right": right_compiled,
            }

        raise RuntimeError(f"Unsupported node type '{getattr(node, 'bl_label', node.name)}'")
    finally:
        visiting.remove(node_key)


def _compile_tree(tree):
    output = _find_output_node(tree)
    if output is None:
        raise RuntimeError(f"Tree '{tree.name}' is missing a Scene Output node")
    root = _linked_source_node(output.inputs[0])
    if root is None:
        raise RuntimeError(f"Tree '{tree.name}' needs the Scene Output connected")

    primitive_rows = []
    polygon_rows = []
    warp_rows = []
    instructions = []
    primitive_map = {}
    primitive_specs = []
    primitive_entries = []
    root_node = _emit_node(root, primitive_rows, polygon_rows, warp_rows, instructions, primitive_map, primitive_specs, primitive_entries, set())
    return primitive_rows, polygon_rows, warp_rows, instructions, primitive_specs, primitive_entries, root_node


def _compile_scene_union(scene):
    primitive_rows = []
    polygon_rows = []
    warp_rows = []
    instructions = []
    primitive_specs = []
    primitive_entries = []
    root_node = None
    proxies = list_proxy_objects(scene)
    tree = get_scene_tree(scene, create=False)
    for primitive_index, obj in enumerate(proxies):
        node = find_initializer_node(tree, obj=obj)
        if node is not None:
            primitive_rows.extend(_primitive_rows_from_node(node, warp_rows=warp_rows, polygon_rows=polygon_rows))
            primitive_specs.append(_primitive_spec_from_node(node, primitive_index))
            primitive_entries.append({"node": node, "warps": (), "object": getattr(node, "target", None)})
        else:
            temp_node = type("_TempNode", (), {})()
            settings = runtime.object_settings(obj)
            temp_node.primitive_type = getattr(settings, "primitive_type", "sphere") if settings is not None else "sphere"
            temp_node.radius = getattr(settings, "radius", 0.5) if settings is not None else 0.5
            temp_node.cone_top_radius = _cone_top_radius_value(settings) if settings is not None else 0.0
            temp_node.cone_bottom_radius = _cone_bottom_radius_value(settings) if settings is not None else 0.5
            temp_node.cylinder_bevel_top = _cylinder_bevel_top_value(settings) if settings is not None else 0.0
            temp_node.cylinder_bevel_bottom = _cylinder_bevel_bottom_value(settings) if settings is not None else 0.0
            temp_node.size = getattr(settings, "size", (0.5, 0.5, 0.5)) if settings is not None else (0.5, 0.5, 0.5)
            temp_node.height = getattr(settings, "height", 1.0) if settings is not None else 1.0
            temp_node.major_radius = getattr(settings, "major_radius", 0.75) if settings is not None else 0.75
            temp_node.minor_radius = getattr(settings, "minor_radius", 0.25) if settings is not None else 0.25
            for attribute in _DIRECT_FLOAT_ATTRS:
                setattr(temp_node, attribute, getattr(settings, attribute, 0.0) if settings is not None else 0.0)
            for attribute in _DIRECT_VECTOR_ATTRS:
                setattr(temp_node, attribute, getattr(settings, attribute, (0.0, 0.0, 0.0, 0.0)) if settings is not None else (0.0, 0.0, 0.0, 0.0))
            for attribute in _DIRECT_BOOL_ATTRS:
                setattr(temp_node, attribute, getattr(settings, attribute, False) if settings is not None else False)
            for attribute in _DIRECT_ENUM_ATTRS:
                setattr(temp_node, attribute, getattr(settings, attribute, "SMOOTH") if settings is not None else "SMOOTH")
            temp_node.ngon_sides = getattr(settings, "ngon_sides", 6) if settings is not None else 6
            temp_node.polygon_points = polygon_control_point_data(settings, ensure_default=True) if settings is not None else default_polygon_control_points()
            temp_node.sdf_location = tuple(float(component) for component in obj.location)
            temp_node.sdf_rotation = tuple(float(component) for component in obj.rotation_euler)
            temp_node.sdf_scale = tuple(float(component) for component in obj.scale)
            primitive_rows.extend(_primitive_rows_from_node(temp_node, warp_rows=warp_rows, polygon_rows=polygon_rows))
            primitive_specs.append(_primitive_spec_from_node(temp_node, primitive_index))
            primitive_entries.append({"node": temp_node, "warps": (), "object": obj})
        primitive_instruction_index = len(instructions)
        _append_instruction(instructions, (0.0, float(primitive_index), 0.0, 0.0))
        primitive_node = {
            "kind": "primitive",
            "primitive_index": int(primitive_index),
            "instruction_index": int(primitive_instruction_index),
        }
        if root_node is None:
            root_node = primitive_node
        else:
            op_instruction_index = len(instructions)
            root_node = {
                "kind": "op",
                "op": int(_CSG_OPERATION_TO_ID["UNION"]),
                "blend": 0.0,
                "instruction_index": int(op_instruction_index),
                "left": root_node,
                "right": primitive_node,
            }
        if primitive_index > 0:
            _append_instruction(instructions, (1.0, 0.0, 0.0, 0.0))
    return primitive_rows, polygon_rows, warp_rows, instructions, primitive_specs, primitive_entries, root_node


def refresh_compiled_scene_dynamic(compiled):
    scene = runtime.scene_identity(compiled.get("scene"))
    primitive_entries = list(compiled.get("primitive_entries", ()))
    if not primitive_entries:
        primitive_entries = [{"node": node, "warps": (), "object": getattr(node, "target", None)} for node in compiled.get("primitive_nodes", ())]
    if not primitive_entries:
        return compiled

    primitive_rows = []
    polygon_rows = []
    warp_rows = []
    primitive_specs = []
    primitive_nodes = []
    for primitive_index, entry in enumerate(primitive_entries):
        node = entry["node"]
        if runtime.safe_pointer(node) == 0:
            return compile_scene(scene) if scene is not None else compiled
        warps = _resolve_warp_stack(tuple(entry.get("warps", ())))
        try:
            primitive_rows.extend(_primitive_rows_from_node(node, warps=warps, warp_rows=warp_rows, polygon_rows=polygon_rows))
            primitive_specs.append(_primitive_spec_from_node(node, primitive_index, warps=warps))
        except ReferenceError:
            return compile_scene(scene) if scene is not None else compiled
        entry["warps"] = warps
        primitive_nodes.append(node)

    render_bounds = _tight_scene_bounds(compiled.get("root_node"), primitive_specs)
    scene_bounds = _scene_bounds(compiled.get("root_node"), primitive_specs)
    instruction_rows = list(compiled.get("instruction_rows", ()))
    rows = primitive_rows + warp_rows + instruction_rows
    refreshed = dict(compiled)
    refreshed.update(
        {
            "primitive_rows": primitive_rows,
            "polygon_rows": polygon_rows,
            "warp_rows": warp_rows,
            "primitive_specs": primitive_specs,
            "primitive_entries": primitive_entries,
            "primitive_nodes": primitive_nodes,
            "primitive_count": len(primitive_entries),
            "warp_row_count": len(warp_rows),
            "instruction_base": len(primitive_rows) + len(warp_rows),
            "render_bounds": render_bounds,
            "scene_bounds": scene_bounds,
            "rows": rows,
            "hash": runtime.hash_compiled_rows(primitive_rows, polygon_rows, warp_rows, instruction_rows),
        }
    )
    return refreshed


def compiled_dynamic_signature(compiled):
    primitive_entries = list(compiled.get("primitive_entries", ()))
    if not primitive_entries:
        primitive_entries = [{"node": node, "warps": (), "object": getattr(node, "target", None)} for node in compiled.get("primitive_nodes", ())]

    signature = []
    for entry in primitive_entries:
        node = entry.get("node")
        node_key = runtime.safe_pointer(node)
        if node_key == 0:
            return None
        try:
            location, rotation, scale = _node_transform_values(node)
        except ReferenceError:
            return None

        item = [
            node_key,
            tuple(round(float(component), 6) for component in location),
            tuple(round(float(component), 6) for component in rotation),
            tuple(round(float(component), 6) for component in scale),
        ]

        for warp in _resolve_warp_stack(tuple(entry.get("warps", ()))):
            kind = warp[0]
            if kind == "mirror":
                _kind, _flags, origin_object, _blend = warp
                item.append((kind, tuple(round(float(component), 6) for component in _mirror_origin_world(origin_object))))
                continue
            if kind == "grid":
                _kind, _count_x, _count_y, _count_z, _spacing, _origin_object, frame_node, _source_node, branch_center, offset_location, offset_rotation, offset_scale, _blend = warp
                if _array_frame_transform_values(frame_node) is not None or any(abs(float(component)) > 1.0e-6 for component in offset_location + offset_rotation) or any(abs(float(component) - 1.0) > 1.0e-6 for component in offset_scale):
                    item.append(
                        (
                            kind,
                            tuple(round(float(component), 6) for component in _array_frame_origin_world(frame_node, branch_center)),
                            tuple(round(float(component), 6) for component in _array_rotation_world(frame_node)),
                            tuple(round(float(component), 6) for component in _array_scale_world(frame_node)),
                            tuple(round(float(component), 6) for component in branch_center),
                            tuple(round(float(component), 6) for component in offset_location),
                            tuple(round(float(component), 6) for component in offset_rotation),
                            tuple(round(float(component), 6) for component in offset_scale),
                        )
                    )
                continue
            if kind == "radial":
                _kind, _count, _radius, origin_object, frame_node, _source_node, branch_center, offset_location, offset_rotation, offset_scale, _blend = warp
                if origin_object is not None or _array_frame_transform_values(frame_node) is not None or any(abs(float(component)) > 1.0e-6 for component in offset_location + offset_rotation) or any(abs(float(component) - 1.0) > 1.0e-6 for component in offset_scale):
                    item.append(
                        (
                            kind,
                            tuple(round(float(component), 6) for component in _array_repeat_origin_local(origin_object, branch_center, frame_node)),
                            tuple(round(float(component), 6) for component in _array_frame_origin_world(frame_node, branch_center)),
                            tuple(round(float(component), 6) for component in _array_rotation_world(frame_node)),
                            tuple(round(float(component), 6) for component in _array_scale_world(frame_node)),
                            tuple(round(float(component), 6) for component in branch_center),
                            tuple(round(float(component), 6) for component in offset_location),
                            tuple(round(float(component), 6) for component in offset_rotation),
                            tuple(round(float(component), 6) for component in offset_scale),
                        )
                    )

        signature.append(tuple(item))

    return tuple(signature)


def scene_structure_signature(scene):
    scene = runtime.scene_identity(scene)
    settings = runtime.scene_settings(scene)
    tree = None if settings is None else getattr(settings, "node_tree", None)
    if tree is not None and getattr(tree, "bl_idname", "") == runtime.TREE_IDNAME:
        node_items = []
        for node in tree.nodes:
            node_items.append(
                (
                    str(getattr(node, "name", "") or ""),
                    str(getattr(node, "bl_idname", "") or ""),
                )
            )

        link_items = []
        for link in tree.links:
            from_node = getattr(link, "from_node", None)
            to_node = getattr(link, "to_node", None)
            from_socket = getattr(link, "from_socket", None)
            to_socket = getattr(link, "to_socket", None)
            link_items.append(
                (
                    str(getattr(from_node, "name", "") or ""),
                    str(getattr(from_socket, "name", "") or ""),
                    str(getattr(to_node, "name", "") or ""),
                    str(getattr(to_socket, "name", "") or ""),
                )
            )

        return (
            "tree",
            runtime.safe_pointer(tree),
            tuple(node_items),
            tuple(link_items),
        )

    proxies = []
    for obj in list_proxy_objects(scene):
        settings = runtime.object_settings(obj)
        proxies.append(
            (
                runtime.object_key(obj),
                "" if settings is None else str(getattr(settings, "proxy_id", "") or ""),
            )
        )
    return ("union", tuple(proxies))


def ensure_unique_initializer_ids(tree):
    seen = set()
    changed = False
    for node in initializer_nodes(tree):
        proxy_id = ensure_object_node_id(node)
        if proxy_id in seen:
            with suppress_object_node_updates():
                node.proxy_id = uuid.uuid4().hex
            changed = True
        seen.add(str(node.proxy_id))
    return changed


def prune_tree(tree, valid_target_pointers=None):
    if tree is None:
        return False

    valid_targets = None if valid_target_pointers is None else set(int(pointer) for pointer in valid_target_pointers if pointer)
    changed = False
    pruned = False

    with suppress_object_node_free_cleanup():
        for node in list(tree.nodes):
            if getattr(node, "bl_idname", "") != runtime.OBJECT_NODE_IDNAME:
                continue
            target = getattr(node, "target", None)
            target_pointer = runtime.object_key(target)
            if target_pointer and runtime.is_sdf_proxy(target) and (valid_targets is None or target_pointer in valid_targets):
                continue
            if not bool(getattr(node, "use_proxy", False)):
                continue
            _detach_proxy_binding(node)
            tree.nodes.remove(node)
            changed = True
            pruned = True

    ensure_graph_output(tree)
    cleanup_changed = cleanup_graph_structure(tree)
    changed = cleanup_changed or changed
    if pruned and not cleanup_changed:
        scene = _scene_for_tree(tree)
        if scene is not None:
            runtime.mark_scene_static_dirty(scene)
    return changed


def compile_scene(scene):
    primitive_rows = []
    polygon_rows = []
    warp_rows = []
    instructions = []
    primitive_specs = []
    primitive_entries = []
    root_node = None
    message = ""
    settings = runtime.scene_settings(scene)
    tree = None if settings is None else getattr(settings, "node_tree", None)
    if tree is not None and getattr(tree, "bl_idname", "") == runtime.TREE_IDNAME:
        try:
            primitive_rows, polygon_rows, warp_rows, instructions, primitive_specs, primitive_entries, root_node = _compile_tree(tree)
        except RuntimeError as exc:
            message = f"Graph fallback: {exc}"
            primitive_entries = []

    if not instructions:
        primitive_rows, polygon_rows, warp_rows, instructions, primitive_specs, primitive_entries, root_node = _compile_scene_union(scene)

    stack_depth = _stack_usage(instructions)
    if stack_depth > runtime.MAX_STACK:
        raise RuntimeError(
            f"Graph stack depth {stack_depth} exceeds shader stack limit {runtime.MAX_STACK}"
        )

    render_bounds = _tight_scene_bounds(root_node, primitive_specs)
    scene_bounds = _scene_bounds(root_node, primitive_specs)
    scene_hash = runtime.hash_compiled_rows(primitive_rows, polygon_rows, warp_rows, instructions)
    topology_hash = runtime.hash_instruction_rows(instructions)
    primitive_nodes = [entry["node"] for entry in primitive_entries]
    return {
        "scene": runtime.scene_identity(scene),
        "scene_structure_signature": scene_structure_signature(scene),
        "primitive_rows": primitive_rows,
        "polygon_rows": polygon_rows,
        "warp_rows": warp_rows,
        "instruction_rows": instructions,
        "primitive_specs": primitive_specs,
        "primitive_entries": primitive_entries,
        "primitive_nodes": primitive_nodes,
        "root_node": root_node,
        "primitive_count": len(primitive_entries),
        "warp_row_count": len(warp_rows),
        "instruction_count": len(instructions),
        "instruction_base": len(primitive_rows) + len(warp_rows),
        "stack_depth": stack_depth,
        "render_bounds": render_bounds,
        "scene_bounds": scene_bounds,
        "rows": primitive_rows + warp_rows + instructions,
        "hash": scene_hash,
        "topology_hash": topology_hash,
        "message": message,
    }


class _MathOPSV2NodeCategory(NodeCategory):
    @classmethod
    def poll(cls, context):
        space = getattr(context, "space_data", None)
        return space is not None and getattr(space, "tree_type", "") == runtime.TREE_IDNAME


node_categories = [
    _MathOPSV2NodeCategory(
        "MATHOPS_V2_OBJECTS",
        "Objects",
        items=[NodeItem(runtime.OBJECT_NODE_IDNAME)],
    ),
    _MathOPSV2NodeCategory(
        "MATHOPS_V2_CSG",
        "CSG",
        items=[NodeItem(runtime.CSG_NODE_IDNAME)],
    ),
    _MathOPSV2NodeCategory(
        "MATHOPS_V2_MODIFIERS",
        "Modifiers",
        items=[NodeItem(MIRROR_NODE_IDNAME), NodeItem(ARRAY_NODE_IDNAME)],
    ),
    _MathOPSV2NodeCategory(
        "MATHOPS_V2_UTILS",
        "Utilities",
        items=[NodeItem(TRANSFORM_NODE_IDNAME), NodeItem(BREAK_TRANSFORM_NODE_IDNAME)],
    ),
]


classes = (
    MathOPSV2PolygonPoint,
    MathOPSV2SDFSocket,
    MathOPSV2TransformSocket,
    MathOPSV2FloatSocket,
    MathOPSV2VectorSocket,
    MathOPSV2EulerSocket,
    MathOPSV2OutputNode,
    MathOPSV2ObjectNode,
    MathOPSV2TransformNode,
    MathOPSV2BreakTransformNode,
    MathOPSV2CSGNode,
    MathOPSV2MirrorNode,
    MathOPSV2ArrayNode,
    MathOPSV2NodeTree,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    nodeitems_utils.register_node_categories(NODE_CATEGORY_ID, node_categories)


def unregister():
    nodeitems_utils.unregister_node_categories(NODE_CATEGORY_ID)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
