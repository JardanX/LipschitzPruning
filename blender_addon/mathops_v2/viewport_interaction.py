import bpy
from bpy.types import GizmoGroup, Operator
from bpy_extras import view3d_utils
from mathutils import Euler, Matrix, Vector

from . import runtime, sdf_nodes
from .render import bridge


_PICK_MAX_STEPS = 256
_PICK_HIT_EPSILON = 5.0e-4
_PICK_MIN_STEP = 1.0e-4
_PICK_AABB_PAD = 5.0e-2
_SCALE_GIZMO_BASE = 0.35

_AXIS_COLORS = (
    (0.95, 0.35, 0.35),
    (0.45, 0.85, 0.35),
    (0.35, 0.55, 0.95),
)

_addon_keymaps = []


def active_primitive_node(scene):
    return sdf_nodes.active_primitive_node(scene, create=False)


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
        for window in bpy.context.window_manager.windows:
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


def _set_active_node(scene, node, extend=False):
    tree = _selected_tree(scene)
    if tree is None or node is None:
        return False
    if not extend:
        for tree_node in tree.nodes:
            tree_node.select = False
    node.select = True
    tree.nodes.active = node
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
    if tree is None:
        return None

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
    return _raymarch_pick(tree, payload, ray_origin, ray_direction)


def _node_rotation_matrix(node):
    return Euler(node.sdf_rotation, "XYZ").to_matrix()


def _basis_from_axis(origin, axis, up_hint):
    axis = Vector(axis)
    if axis.length_squared <= 1.0e-12:
        axis = Vector((0.0, 0.0, 1.0))
    else:
        axis.normalize()

    up = Vector(up_hint) - Vector(up_hint).project(axis)
    if up.length_squared <= 1.0e-12:
        up = axis.orthogonal()
    up.normalize()
    side = axis.cross(up)
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
    origin = Vector(node.sdf_location)
    axes = [Vector(rotation_matrix.col[index]) for index in range(3)]
    up_hints = (axes[1], axes[2], axes[1])
    return rotation_matrix, origin, axes, up_hints


def _move_get():
    node = active_primitive_node(bpy.context.scene)
    return tuple(node.sdf_location) if node is not None else (0.0, 0.0, 0.0)


def _move_set(value):
    node = active_primitive_node(bpy.context.scene)
    if node is None:
        return
    node.sdf_location = tuple(float(component) for component in value)


def _rotation_get(axis_index):
    node = active_primitive_node(bpy.context.scene)
    return float(node.sdf_rotation[axis_index]) if node is not None else 0.0


def _rotation_set(axis_index, value):
    node = active_primitive_node(bpy.context.scene)
    if node is None:
        return
    rotation = list(node.sdf_rotation)
    rotation[axis_index] = float(value)
    node.sdf_rotation = rotation


def _scale_get(axis_index):
    node = active_primitive_node(bpy.context.scene)
    if node is None:
        return _SCALE_GIZMO_BASE
    return _SCALE_GIZMO_BASE + float(node.sdf_scale[axis_index]) - 1.0


def _scale_set(axis_index, value):
    node = active_primitive_node(bpy.context.scene)
    if node is None:
        return
    scale = list(node.sdf_scale)
    scale[axis_index] = max(0.001, 1.0 + float(value) - _SCALE_GIZMO_BASE)
    node.sdf_scale = scale


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
        context.window.cursor_modal_set("EYEDROPPER")
        context.workspace.status_text_set("Click an SDF surface to select its node")
        context.window_manager.modal_handler_add(self)
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
        node = _pick_node_in_region(context, region, coord)

        self._finish(context)
        if node is None:
            self.report({"INFO"}, "No visible SDF primitive hit")
            return {"FINISHED"}

        _set_active_node(scene, node)
        self.report({"INFO"}, f"Selected '{node.name}'")
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

        node = _pick_node_in_region(context, region, coord)
        if node is None:
            return {"PASS_THROUGH"}

        _set_active_node(
            context.scene, node, extend=bool(getattr(event, "shift", False))
        )
        return {"FINISHED"}


class _MathOPSV2GizmoBase:
    transform_mode = "TRANSLATE"

    @classmethod
    def poll(cls, context):
        space = getattr(context, "space_data", None)
        scene = getattr(context, "scene", None)
        return (
            space is not None
            and space.type == "VIEW_3D"
            and scene is not None
            and _transform_mode(scene) == cls.transform_mode
            and active_primitive_node(scene) is not None
        )


class MATHOPS_V2_GGT_sdf_translate(_MathOPSV2GizmoBase, GizmoGroup):
    bl_idname = "VIEW3D_GGT_mathops_v2_sdf_translate"
    bl_label = "MathOPS-v2 SDF Move"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options = {"3D", "PERSISTENT"}
    transform_mode = "TRANSLATE"

    def setup(self, context):
        del context
        self.move_gizmo = self.gizmos.new("GIZMO_GT_move_3d")
        self.move_gizmo.target_set_handler("offset", get=_move_get, set=_move_set)
        self.move_gizmo.scale_basis = 0.16
        self.move_gizmo.line_width = 2.0
        self.move_gizmo.color = 0.9, 0.9, 0.9
        self.move_gizmo.alpha = 0.55
        self.move_gizmo.color_highlight = 1.0, 1.0, 1.0
        self.move_gizmo.alpha_highlight = 1.0
        self.move_gizmo.use_draw_modal = True

    def refresh(self, context):
        self._update_matrix(context)

    def draw_prepare(self, context):
        self._update_matrix(context)

    def _update_matrix(self, context):
        node = active_primitive_node(context.scene)
        if node is None:
            return
        rotation_matrix, _origin, _axes, _up_hints = _node_axes(node)
        move_basis = rotation_matrix.to_4x4()
        move_basis.translation = Vector((0.0, 0.0, 0.0))
        self.move_gizmo.matrix_basis = move_basis


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
        node = active_primitive_node(context.scene)
        if node is None:
            return
        _rotation_matrix, origin, axes, up_hints = _node_axes(node)
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
        node = active_primitive_node(context.scene)
        if node is None:
            return
        _rotation_matrix, origin, axes, up_hints = _node_axes(node)
        for axis_index, gizmo in enumerate(self.scale_gizmos):
            gizmo.matrix_basis = _basis_from_axis(
                origin, axes[axis_index], up_hints[axis_index]
            )


classes = (
    MATHOPS_V2_OT_pick_sdf,
    MATHOPS_V2_OT_select_sdf_click,
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
