import bpy
import gpu
from bpy.props import IntProperty, StringProperty
from bpy.types import Gizmo, GizmoGroup, Menu, Operator
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector

from .. import runtime
from ..nodes import sdf_tree


_CORNER_HANDLE_GAP = 0.15
_context_point_index = -1


def _local_to_world_offset(obj, local_x, local_y):
    world = obj.matrix_world @ Vector((local_x, local_y, 0.0))
    origin = obj.matrix_world.translation
    return (world.x - origin.x, world.y - origin.y, world.z - origin.z)


def _world_offset_to_local(obj, wx, wy, wz):
    origin = obj.matrix_world.translation
    world = Vector((wx + origin.x, wy + origin.y, wz + origin.z))
    local = obj.matrix_world.inverted() @ world
    return float(local.x), float(local.y)


def _resolve_polygon_binding(context, tree_name="", node_name=""):
    scene = getattr(context, "scene", None)
    node = None
    obj = None
    if tree_name and node_name:
        tree = bpy.data.node_groups.get(tree_name)
        if tree is not None:
            node = tree.nodes.get(node_name)
        obj = runtime.object_identity(getattr(node, "target", None)) if node is not None else None
    else:
        obj = runtime.object_identity(getattr(context, "object", None))
        if scene is not None:
            tree = sdf_tree.get_scene_tree(scene, create=False)
            if tree is not None:
                settings = runtime.object_settings(obj)
                proxy_id = "" if settings is None else str(getattr(settings, "proxy_id", "") or "")
                node = sdf_tree.find_initializer_node(tree, obj=obj, proxy_id=proxy_id)
    settings = runtime.object_settings(obj)
    if not runtime.is_sdf_proxy(obj) or settings is None:
        obj = None
        settings = None
    primitive_type = str(getattr(node, "primitive_type", "") or "") if node is not None else str(getattr(settings, "primitive_type", "") or "")
    if primitive_type != "polygon":
        return None, None, None
    if node is not None:
        sdf_tree.ensure_polygon_defaults(node)
    if settings is not None:
        sdf_tree.ensure_polygon_defaults(settings)
    return obj, settings, node


def _polygon_owner(context, tree_name="", node_name=""):
    obj, settings, node = _resolve_polygon_binding(context, tree_name=tree_name, node_name=node_name)
    return obj, (node if node is not None else settings), node


def _apply_polygon_points(context, points, active_index, tree_name="", node_name=""):
    obj, owner, node = _polygon_owner(context, tree_name=tree_name, node_name=node_name)
    if obj is None or owner is None:
        return False

    changed = sdf_tree.set_polygon_points(owner, points)
    active_index = min(max(int(active_index), 0), max(len(points) - 1, 0))
    if int(getattr(owner, "polygon_active_index", 0)) != active_index:
        owner.polygon_active_index = active_index
        changed = True
    if node is not None:
        sdf_tree.sync_node_to_proxy(node, include_transform=False)
    if not changed:
        return False
    scene = getattr(context, "scene", None) or getattr(bpy.context, "scene", None)
    if scene is not None:
        runtime.mark_scene_static_dirty(scene)
    runtime.clear_error()
    runtime.note_interaction()
    runtime.tag_redraw(context)
    return True


def _polygon_points(owner):
    return sdf_tree.polygon_point_data(owner, ensure_default=True)


def _bisector_for_point(owner, index):
    points = _polygon_points(owner)
    count = len(points)
    if count < 3:
        return Vector((0.0, 1.0))
    previous_index = (index - 1) % count
    next_index = (index + 1) % count
    point = Vector((points[index][0], points[index][1]))
    previous_point = Vector((points[previous_index][0], points[previous_index][1]))
    next_point = Vector((points[next_index][0], points[next_index][1]))
    to_previous = (previous_point - point).normalized()
    to_next = (next_point - point).normalized()
    bisector = to_previous + to_next
    if bisector.length < 1.0e-6:
        bisector = Vector((-to_previous.y, to_previous.x))
    else:
        bisector.normalize()
    cross = (to_previous.x * to_next.y) - (to_previous.y * to_next.x)
    if cross > 0.0:
        bisector = -bisector
    return bisector


def _make_point_getter(point_index):
    def getter():
        context = bpy.context
        obj, owner, _node = _polygon_owner(context)
        if obj is None or owner is None:
            return (0.0, 0.0, 0.0)
        points = _polygon_points(owner)
        if point_index >= len(points):
            return (0.0, 0.0, 0.0)
        point = points[point_index]
        return _local_to_world_offset(obj, point[0], point[1])

    return getter


def _make_point_setter(point_index):
    def setter(value):
        context = bpy.context
        obj, owner, _node = _polygon_owner(context)
        if obj is None or owner is None:
            return
        points = list(_polygon_points(owner))
        if point_index >= len(points):
            return
        local_x, local_y = _world_offset_to_local(obj, value[0], value[1], value[2])
        points[point_index] = (local_x, local_y, points[point_index][2])
        _apply_polygon_points(context, points, point_index)

    return setter


def _make_corner_getter(point_index):
    def getter():
        context = bpy.context
        obj, owner, _node = _polygon_owner(context)
        if obj is None or owner is None:
            return (0.0, 0.0, 0.0)
        points = _polygon_points(owner)
        if point_index >= len(points):
            return (0.0, 0.0, 0.0)
        point = points[point_index]
        bisector = _bisector_for_point(owner, point_index)
        length = point[2] + _CORNER_HANDLE_GAP
        return _local_to_world_offset(obj, point[0] + bisector.x * length, point[1] + bisector.y * length)

    return getter


def _make_corner_setter(point_index):
    def setter(value):
        context = bpy.context
        obj, owner, _node = _polygon_owner(context)
        if obj is None or owner is None:
            return
        points = list(_polygon_points(owner))
        if point_index >= len(points):
            return
        local_x, local_y = _world_offset_to_local(obj, value[0], value[1], value[2])
        base = Vector((points[point_index][0], points[point_index][1]))
        new_pos = Vector((local_x, local_y))
        bisector = _bisector_for_point(owner, point_index)
        corner = max((new_pos - base).dot(bisector) - _CORNER_HANDLE_GAP, 0.0)
        points[point_index] = (points[point_index][0], points[point_index][1], corner)
        _apply_polygon_points(context, points, point_index)

    return setter


class MATHOPS_V2_GT_corner_line(Gizmo):
    bl_idname = "MATHOPS_V2_GT_corner_line"

    def setup(self):
        self._point_index = 0
        self._color = (0.25, 0.45, 0.7, 0.4)

    def draw(self, context):
        obj, owner, _node = _polygon_owner(context)
        if obj is None or owner is None:
            return
        points = _polygon_points(owner)
        index = self._point_index
        if index >= len(points):
            return
        point = points[index]
        bisector = _bisector_for_point(owner, index)
        offset = point[2] + _CORNER_HANDLE_GAP
        start = _local_to_world_offset(obj, point[0], point[1])
        end = _local_to_world_offset(obj, point[0] + bisector.x * offset, point[1] + bisector.y * offset)
        gpu.matrix.push()
        gpu.matrix.multiply_matrix(self.matrix_basis)
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        batch = batch_for_shader(shader, "LINES", {"pos": [start, end]})
        shader.bind()
        shader.uniform_float("color", self._color)
        gpu.state.blend_set("ALPHA")
        batch.draw(shader)
        gpu.state.blend_set("NONE")
        gpu.matrix.pop()

    def draw_select(self, _context, _select_id):
        pass

    def test_select(self, _context, _location):
        return -1


class MATHOPS_V2_OT_polygon_point_add(Operator):
    bl_idname = "mathops_v2.polygon_point_add"
    bl_label = "Add Polygon Point"
    bl_options = {"REGISTER", "UNDO"}

    tree_name: StringProperty(default="")
    node_name: StringProperty(default="")

    def execute(self, context):
        _obj, owner, _node = _polygon_owner(context, tree_name=self.tree_name, node_name=self.node_name)
        if owner is None:
            return {"CANCELLED"}
        points = list(_polygon_points(owner))
        points.append((0.0, 0.0, 0.0))
        _apply_polygon_points(context, points, len(points) - 1, tree_name=self.tree_name, node_name=self.node_name)
        return {"FINISHED"}


class MATHOPS_V2_OT_polygon_point_remove(Operator):
    bl_idname = "mathops_v2.polygon_point_remove"
    bl_label = "Remove Polygon Point"
    bl_options = {"REGISTER", "UNDO"}

    tree_name: StringProperty(default="")
    node_name: StringProperty(default="")

    def execute(self, context):
        _obj, owner, _node = _polygon_owner(context, tree_name=self.tree_name, node_name=self.node_name)
        if owner is None:
            return {"CANCELLED"}
        points = list(_polygon_points(owner))
        min_points = 2 if bool(getattr(owner, "polygon_is_line", False)) else 3
        if len(points) <= min_points:
            return {"CANCELLED"}
        points.pop(-1)
        _apply_polygon_points(context, points, len(points) - 1, tree_name=self.tree_name, node_name=self.node_name)
        return {"FINISHED"}


class MATHOPS_V2_OT_polygon_point_remove_index(Operator):
    bl_idname = "mathops_v2.polygon_point_remove_index"
    bl_label = "Delete Point"
    bl_options = {"REGISTER", "UNDO"}

    index: IntProperty(default=-1)

    @classmethod
    def poll(cls, context):
        _obj, owner, _node = _polygon_owner(context)
        if owner is None:
            return False
        min_points = 2 if bool(getattr(owner, "polygon_is_line", False)) else 3
        return len(_polygon_points(owner)) > min_points

    def execute(self, context):
        _obj, owner, _node = _polygon_owner(context)
        if owner is None:
            return {"CANCELLED"}
        points = list(_polygon_points(owner))
        min_points = 2 if bool(getattr(owner, "polygon_is_line", False)) else 3
        if len(points) <= min_points or self.index < 0 or self.index >= len(points):
            return {"CANCELLED"}
        points.pop(self.index)
        _apply_polygon_points(context, points, min(self.index, len(points) - 1))
        return {"FINISHED"}


class MATHOPS_V2_MT_polygon_point_context(Menu):
    bl_idname = "MATHOPS_V2_MT_polygon_point_context"
    bl_label = "Polygon Point"

    def draw(self, _context):
        layout = self.layout
        if _context_point_index >= 0:
            operator = layout.operator("mathops_v2.polygon_point_remove_index", icon="X")
            operator.index = _context_point_index


def _mouse_to_local_xy(context, event, obj):
    from bpy_extras import view3d_utils

    region = context.region
    rv3d = context.region_data
    coord = (event.mouse_region_x, event.mouse_region_y)
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    matrix_inverse = obj.matrix_world.inverted()
    local_origin = matrix_inverse @ ray_origin
    local_direction = (matrix_inverse @ (ray_origin + view_vector) - local_origin).normalized()
    if abs(local_direction.z) < 1.0e-6:
        return None
    t = -local_origin.z / local_direction.z
    local_pos = local_origin + local_direction * t
    return Vector((local_pos.x, local_pos.y))


def _nearest_edge_index(owner, local_xy):
    points = _polygon_points(owner)
    best_distance = 1.0e20
    best_index = 0
    for index in range(len(points)):
        a = Vector((points[index][0], points[index][1]))
        b = Vector((points[(index + 1) % len(points)][0], points[(index + 1) % len(points)][1]))
        edge = b - a
        edge_sq = edge.dot(edge)
        if edge_sq < 1.0e-12:
            distance = (local_xy - a).length
        else:
            t = max(0.0, min(1.0, (local_xy - a).dot(edge) / edge_sq))
            closest = a + edge * t
            distance = (local_xy - closest).length
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


class MATHOPS_V2_OT_polygon_point_add_click(Operator):
    bl_idname = "mathops_v2.polygon_point_add_click"
    bl_label = "Subdivide Polygon Edge"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj, owner, _node = _polygon_owner(context)
        return obj is not None and owner is not None

    def invoke(self, context, event):
        obj, owner, _node = _polygon_owner(context)
        if obj is None or owner is None:
            return {"CANCELLED"}
        local_xy = _mouse_to_local_xy(context, event, obj)
        if local_xy is None:
            return {"CANCELLED"}
        points = list(_polygon_points(owner))
        edge_index = _nearest_edge_index(owner, local_xy)
        a = Vector((points[edge_index][0], points[edge_index][1]))
        b = Vector((points[(edge_index + 1) % len(points)][0], points[(edge_index + 1) % len(points)][1]))
        midpoint = (a + b) * 0.5
        points.insert(edge_index + 1, (float(midpoint.x), float(midpoint.y), 0.0))
        _apply_polygon_points(context, points, edge_index + 1)
        return {"FINISHED"}


class MATHOPS_V2_OT_polygon_point_context_menu(Operator):
    bl_idname = "mathops_v2.polygon_point_context_menu"
    bl_label = "Polygon Point Context Menu"

    @classmethod
    def poll(cls, context):
        obj, owner, _node = _polygon_owner(context)
        return obj is not None and owner is not None

    def invoke(self, context, event):
        global _context_point_index
        from bpy_extras import view3d_utils

        obj, owner, _node = _polygon_owner(context)
        if obj is None or owner is None:
            return {"CANCELLED"}
        region = context.region
        rv3d = context.region_data
        coord = Vector((event.mouse_region_x, event.mouse_region_y))
        best_distance = 30.0
        best_index = -1
        for index, point in enumerate(_polygon_points(owner)):
            world = obj.matrix_world @ Vector((point[0], point[1], 0.0))
            screen = view3d_utils.location_3d_to_region_2d(region, rv3d, world)
            if screen is None:
                continue
            distance = (coord - screen).length
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index < 0:
            bpy.ops.wm.call_menu(name="VIEW3D_MT_object_context_menu")
            return {"FINISHED"}
        _context_point_index = best_index
        bpy.ops.wm.call_menu(name="MATHOPS_V2_MT_polygon_point_context")
        return {"FINISHED"}


class VIEW3D_GGT_mathops_polygon(GizmoGroup):
    bl_idname = "VIEW3D_GGT_mathops_polygon"
    bl_label = "MathOPS Polygon Points"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options = {"3D", "PERSISTENT", "SHOW_MODAL_ALL"}

    @classmethod
    def poll(cls, context):
        obj, owner, _node = _polygon_owner(context)
        if obj is None or owner is None:
            return False
        if getattr(obj, "type", "") != "EMPTY":
            return False
        if not obj.select_get():
            return False
        overlay = getattr(getattr(context, "space_data", None), "overlay", None)
        return overlay is None or bool(getattr(overlay, "show_overlays", True))

    def setup(self, context):
        self.point_gizmos = []
        self.corner_gizmos = []
        self.line_gizmos = []
        self._last_count = -1
        self._was_modal = False
        self._rebuild(context)

    def _rebuild(self, context):
        for gizmo in self.point_gizmos:
            self.gizmos.remove(gizmo)
        for gizmo in self.corner_gizmos:
            self.gizmos.remove(gizmo)
        for gizmo in self.line_gizmos:
            self.gizmos.remove(gizmo)
        self.point_gizmos.clear()
        self.corner_gizmos.clear()
        self.line_gizmos.clear()

        obj, owner, _node = _polygon_owner(context)
        if obj is None or owner is None:
            self._last_count = 0
            return
        matrix = Matrix.Translation(obj.matrix_world.translation)
        for index, _point in enumerate(_polygon_points(owner)):
            point_gizmo = self.gizmos.new("GIZMO_GT_move_3d")
            point_gizmo.draw_style = "CROSS_2D"
            point_gizmo.draw_options = {"ALIGN_VIEW"}
            point_gizmo.use_draw_modal = True
            point_gizmo.use_draw_value = True
            point_gizmo.scale_basis = 0.08
            point_gizmo.color = (0.4, 0.7, 1.0)
            point_gizmo.alpha = 0.9
            point_gizmo.color_highlight = (1.0, 1.0, 1.0)
            point_gizmo.alpha_highlight = 1.0
            point_gizmo.matrix_basis = matrix
            point_gizmo.target_set_handler("offset", get=_make_point_getter(index), set=_make_point_setter(index))
            self.point_gizmos.append(point_gizmo)

            corner_gizmo = self.gizmos.new("GIZMO_GT_move_3d")
            corner_gizmo.draw_style = "RING_2D"
            corner_gizmo.draw_options = {"ALIGN_VIEW"}
            corner_gizmo.use_draw_modal = True
            corner_gizmo.use_draw_value = True
            corner_gizmo.scale_basis = 0.06
            corner_gizmo.color = (0.25, 0.45, 0.7)
            corner_gizmo.alpha = 0.5
            corner_gizmo.color_highlight = (0.6, 0.9, 1.0)
            corner_gizmo.alpha_highlight = 1.0
            corner_gizmo.matrix_basis = matrix
            corner_gizmo.target_set_handler("offset", get=_make_corner_getter(index), set=_make_corner_setter(index))
            self.corner_gizmos.append(corner_gizmo)

            line_gizmo = self.gizmos.new("MATHOPS_V2_GT_corner_line")
            line_gizmo._point_index = index
            line_gizmo.matrix_basis = matrix
            self.line_gizmos.append(line_gizmo)

        self._last_count = len(_polygon_points(owner))

    def draw_prepare(self, context):
        obj, owner, _node = _polygon_owner(context)
        if obj is None or owner is None:
            return
        points = _polygon_points(owner)
        if len(points) != self._last_count:
            self._rebuild(context)
            return
        any_modal = any(gizmo.is_modal for gizmo in self.point_gizmos) or any(gizmo.is_modal for gizmo in self.corner_gizmos)
        if self._was_modal and not any_modal:
            bpy.app.timers.register(lambda: (bpy.ops.ed.undo_push(message="Edit MathOPS Polygon"), None)[1], first_interval=0.0)
        self._was_modal = any_modal
        matrix = Matrix.Translation(obj.matrix_world.translation)
        for index in range(len(points)):
            point_gizmo = self.point_gizmos[index]
            corner_gizmo = self.corner_gizmos[index]
            line_gizmo = self.line_gizmos[index]
            point_gizmo.matrix_basis = matrix
            corner_gizmo.matrix_basis = matrix
            line_gizmo.matrix_basis = matrix
            active = point_gizmo.is_highlight or point_gizmo.is_modal or corner_gizmo.is_highlight or corner_gizmo.is_modal
            if active:
                point_gizmo.color = (1.0, 1.0, 1.0)
                point_gizmo.alpha = 1.0
                corner_gizmo.color = (0.4, 0.7, 1.0)
                corner_gizmo.alpha = 0.9
                line_gizmo._color = (0.4, 0.7, 1.0, 0.9)
            else:
                point_gizmo.color = (0.4, 0.7, 1.0)
                point_gizmo.alpha = 0.9
                corner_gizmo.color = (0.25, 0.45, 0.7)
                corner_gizmo.alpha = 0.5
                line_gizmo._color = (0.25, 0.45, 0.7, 0.4)


classes = (
    MATHOPS_V2_GT_corner_line,
    MATHOPS_V2_OT_polygon_point_add,
    MATHOPS_V2_OT_polygon_point_remove,
    MATHOPS_V2_OT_polygon_point_remove_index,
    MATHOPS_V2_OT_polygon_point_add_click,
    MATHOPS_V2_OT_polygon_point_context_menu,
    MATHOPS_V2_MT_polygon_point_context,
    VIEW3D_GGT_mathops_polygon,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
