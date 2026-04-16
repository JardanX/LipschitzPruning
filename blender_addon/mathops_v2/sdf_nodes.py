import hashlib
import json
import re
from pathlib import Path

import bpy
import nodeitems_utils
from bpy.props import FloatProperty, FloatVectorProperty
from bpy.types import Node, NodeSocket, NodeTree, Operator, Panel
from mathutils import Euler, Matrix, Vector
from nodeitems_utils import NodeCategory, NodeItem

from . import runtime


TREE_IDNAME = "MATHOPS_V2_SDF_TREE"
SOCKET_IDNAME = "MATHOPS_V2_SDF_SOCKET"
OUTPUT_NODE_IDNAME = "MATHOPS_V2_SCENE_OUTPUT"
SPHERE_NODE_IDNAME = "MATHOPS_V2_SDF_SPHERE"
BOX_NODE_IDNAME = "MATHOPS_V2_SDF_BOX"
CYLINDER_NODE_IDNAME = "MATHOPS_V2_SDF_CYLINDER"
CONE_NODE_IDNAME = "MATHOPS_V2_SDF_CONE"
UNION_NODE_IDNAME = "MATHOPS_V2_CSG_UNION"
SUBTRACT_NODE_IDNAME = "MATHOPS_V2_CSG_SUBTRACT"
INTERSECT_NODE_IDNAME = "MATHOPS_V2_CSG_INTERSECT"
NODE_CATEGORY_ID = "MATHOPS_V2_SDF_NODES"
_INVALID_SCENE_NAME = "_invalid_.json"
_OUTPUT_ENFORCE_LOCKS = set()
_IDENTITY_MATRIX_3X4 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def _graph_updated(_owner=None, context=None):
    runtime.dynamic_aabb_state.clear()
    runtime.current_effective_aabb = None
    runtime.last_error_message = ""
    try:
        from .render import bridge

        bridge.force_redraw_viewports(context)
    except Exception:
        pass


def _owner_scene(settings):
    scene = getattr(settings, "id_data", None)
    return scene if getattr(scene, "bl_rna", None) == bpy.types.Scene.bl_rna else None


def _default_tree_name(scene) -> str:
    return f"{scene.name} MathOPS SDF"


def _surface_output_socket(node):
    if getattr(node, "bl_idname", "") == OUTPUT_NODE_IDNAME:
        return None
    for socket in getattr(node, "outputs", []):
        if getattr(socket, "bl_idname", "") == SOCKET_IDNAME:
            return socket
    return None


def _surface_root_candidate(tree):
    candidates = []
    for node in tree.nodes:
        socket = _surface_output_socket(node)
        if socket is None:
            continue
        if any(link.to_node.bl_idname != OUTPUT_NODE_IDNAME for link in socket.links):
            continue
        candidates.append(node)
    if not candidates:
        return None
    candidates.sort(key=lambda node: (float(node.location.x), float(-node.location.y)))
    return candidates[-1]


def _position_output_node(tree, output_node, source_node=None):
    if source_node is not None:
        output_node.location = (source_node.location.x + 280.0, source_node.location.y)
        return
    others = [node for node in tree.nodes if node != output_node]
    if not others:
        output_node.location = (300.0, 0.0)
        return
    output_node.location = (
        max(float(node.location.x) for node in others) + 280.0,
        float(others[0].location.y),
    )


def _restore_output_later(tree_name):
    tree = bpy.data.node_groups.get(tree_name)
    if tree is not None and getattr(tree, "bl_idname", "") == TREE_IDNAME:
        _build_default_tree(tree)
    return None


def _find_output_node(tree):
    output_nodes = [node for node in tree.nodes if node.bl_idname == OUTPUT_NODE_IDNAME]
    if not output_nodes:
        return None

    linked_outputs = [
        node for node in output_nodes if node.inputs and node.inputs[0].is_linked
    ]
    return linked_outputs[0] if linked_outputs else output_nodes[0]


def _ensure_output_node(tree):
    tree_key = tree.as_pointer()
    if tree_key in _OUTPUT_ENFORCE_LOCKS:
        return _find_output_node(tree)

    _OUTPUT_ENFORCE_LOCKS.add(tree_key)
    try:
        output_nodes = [
            node for node in tree.nodes if node.bl_idname == OUTPUT_NODE_IDNAME
        ]
        output_node = _find_output_node(tree)

        if output_node is None:
            output_node = tree.nodes.new(OUTPUT_NODE_IDNAME)

        for node in list(output_nodes):
            if node != output_node:
                tree.nodes.remove(node)

        source_node = _surface_root_candidate(tree)
        _position_output_node(tree, output_node, source_node)

        if (
            source_node is not None
            and output_node.inputs
            and not output_node.inputs[0].is_linked
        ):
            source_socket = _surface_output_socket(source_node)
            if source_socket is not None:
                tree.links.new(source_socket, output_node.inputs[0])

        output_node.select = False
        return output_node
    finally:
        _OUTPUT_ENFORCE_LOCKS.discard(tree_key)


def _build_default_tree(tree):
    output = _ensure_output_node(tree)
    if any(node.bl_idname != OUTPUT_NODE_IDNAME for node in tree.nodes):
        return tree

    sphere = tree.nodes.new(SPHERE_NODE_IDNAME)
    sphere.location = (0.0, 0.0)
    if output is not None and output.inputs and not output.inputs[0].is_linked:
        tree.links.new(sphere.outputs[0], output.inputs[0])
    _position_output_node(tree, output, sphere)
    return tree


def ensure_scene_tree(scene, settings=None):
    if settings is None:
        settings = scene.mathops_v2_settings

    tree = getattr(settings, "sdf_node_tree", None)
    if tree is not None and tree.bl_idname == TREE_IDNAME:
        _build_default_tree(tree)
        return tree

    tree_name = _default_tree_name(scene)
    existing = bpy.data.node_groups.get(tree_name)
    if existing is not None and existing.bl_idname == TREE_IDNAME:
        tree = existing
    else:
        tree = bpy.data.node_groups.new(tree_name, TREE_IDNAME)
    _build_default_tree(tree)
    settings.sdf_node_tree = tree
    runtime.debug_log(f"Using scene SDF tree '{tree.name}'")
    return tree


def new_scene_tree(scene, settings=None):
    if settings is None:
        settings = scene.mathops_v2_settings

    tree = bpy.data.node_groups.new(_default_tree_name(scene), TREE_IDNAME)
    _build_default_tree(tree)
    settings.sdf_node_tree = tree
    runtime.debug_log(f"Created scene SDF tree '{tree.name}'")
    return tree


def get_selected_tree(settings, create=False, ensure=False):
    tree = getattr(settings, "sdf_node_tree", None)
    if tree is not None and tree.bl_idname == TREE_IDNAME:
        if ensure:
            _build_default_tree(tree)
        return tree

    if not create:
        raise RuntimeError("Enable Use Nodes to create a scene SDF graph")

    scene = _owner_scene(settings)
    if scene is None:
        raise RuntimeError("Scene-owned SDF tree is unavailable")
    if ensure:
        return ensure_scene_tree(scene, settings)

    tree_name = _default_tree_name(scene)
    tree = bpy.data.node_groups.get(tree_name)
    if tree is not None and tree.bl_idname == TREE_IDNAME:
        return tree
    raise RuntimeError("Scene SDF graph is unavailable")


def _focus_tree_in_editor(context, tree):
    focused = False
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "NODE_EDITOR":
                continue
            space = area.spaces.active
            if getattr(space, "tree_type", None) != TREE_IDNAME:
                continue
            space.node_tree = tree
            area.tag_redraw()
            focused = True
    return focused


def focus_scene_tree(context, create=False):
    settings = context.scene.mathops_v2_settings
    tree = get_selected_tree(settings, create=create, ensure=create)
    _focus_tree_in_editor(context, tree)
    return tree


def _sync_node_editors():
    try:
        initialize_scene_trees()
        for window in bpy.context.window_manager.windows:
            scene = getattr(window, "scene", None)
            if scene is None:
                continue
            settings = scene.mathops_v2_settings
            if not getattr(settings, "use_sdf_nodes", False):
                continue
            tree = ensure_scene_tree(scene, settings)
            for area in window.screen.areas:
                if area.type != "NODE_EDITOR":
                    continue
                space = area.spaces.active
                if getattr(space, "tree_type", None) != TREE_IDNAME:
                    continue
                if getattr(space, "node_tree", None) == tree:
                    continue
                try:
                    space.node_tree = tree
                    area.tag_redraw()
                except Exception:
                    pass
    except Exception:
        pass
    return 0.1


def start_editor_sync():
    if not bpy.app.timers.is_registered(_sync_node_editors):
        bpy.app.timers.register(_sync_node_editors, first_interval=0.0)


def stop_editor_sync():
    if bpy.app.timers.is_registered(_sync_node_editors):
        bpy.app.timers.unregister(_sync_node_editors)


def initialize_scene_trees():
    for scene in bpy.data.scenes:
        settings = scene.mathops_v2_settings
        if getattr(settings, "use_sdf_nodes", False):
            ensure_scene_tree(scene, settings)


def post_register():
    initialize_scene_trees()
    start_editor_sync()


def pre_unregister():
    stop_editor_sync()


def _active_editor_tree(context, create=False):
    space = getattr(context, "space_data", None)
    if space is None or space.type != "NODE_EDITOR":
        return None
    if getattr(space, "tree_type", None) != TREE_IDNAME:
        return None

    settings = context.scene.mathops_v2_settings
    if not getattr(settings, "use_sdf_nodes", False):
        return None

    try:
        tree = get_selected_tree(settings, create=create, ensure=False)
    except Exception:
        return None
    if getattr(space, "node_tree", None) != tree:
        try:
            space.node_tree = tree
            context.area.tag_redraw()
        except Exception:
            pass
    return tree


def _transform_annotations():
    return {
        "sdf_location": FloatVectorProperty(
            name="Location",
            size=3,
            subtype="TRANSLATION",
            default=(0.0, 0.0, 0.0),
            update=_graph_updated,
        ),
        "sdf_rotation": FloatVectorProperty(
            name="Rotation",
            size=3,
            subtype="EULER",
            default=(0.0, 0.0, 0.0),
            update=_graph_updated,
        ),
        "sdf_scale": FloatVectorProperty(
            name="Scale",
            size=3,
            subtype="XYZ",
            default=(1.0, 1.0, 1.0),
            min=0.001,
            soft_min=0.001,
            update=_graph_updated,
        ),
    }


class _MathOPSV2NodeBase:
    bl_width_default = 220

    @classmethod
    def poll(cls, node_tree):
        return node_tree.bl_idname == TREE_IDNAME

    def _draw_transform(self, layout):
        box = layout.box()
        box.label(text="Transform")
        col = box.column(align=True)
        col.prop(self, "sdf_location")
        col.prop(self, "sdf_rotation")
        col.prop(self, "sdf_scale")

    def _world_to_local_matrix(self):
        local_to_world = Matrix.LocRotScale(
            Vector(self.sdf_location),
            Euler(self.sdf_rotation, "XYZ"),
            Vector(self.sdf_scale),
        )
        return local_to_world.inverted_safe()


class _MathOPSV2PrimitiveNodeBase(_MathOPSV2NodeBase):
    __annotations__ = dict(_transform_annotations())
    __annotations__.update(
        {
            "color": FloatVectorProperty(
                name="Color",
                size=3,
                subtype="COLOR",
                min=0.0,
                max=1.0,
                default=(0.8, 0.8, 0.8),
                update=_graph_updated,
            )
        }
    )

    def init(self, context):
        del context
        self.outputs.new(SOCKET_IDNAME, "SDF")

    def _draw_primitive_footer(self, layout):
        layout.prop(self, "color")
        self._draw_transform(layout)

    def draw_buttons_ext(self, context, layout):
        self.draw_buttons(context, layout)


class _MathOPSV2CSGNodeBase(_MathOPSV2NodeBase):
    blend_radius: FloatProperty(
        name="Blend Radius",
        default=0.0,
        min=0.0,
        soft_min=0.0,
        update=_graph_updated,
    )

    blend_mode = "union"

    def init(self, context):
        del context
        self.inputs.new(SOCKET_IDNAME, "A")
        self.inputs.new(SOCKET_IDNAME, "B")
        self.outputs.new(SOCKET_IDNAME, "SDF")

    def draw_buttons(self, context, layout):
        del context
        layout.prop(self, "blend_radius")

    def draw_buttons_ext(self, context, layout):
        self.draw_buttons(context, layout)


class MathOPSV2SDFTree(NodeTree):
    bl_idname = TREE_IDNAME
    bl_label = "MathOPS-v2 SDF"
    bl_icon = "NODETREE"

    def update(self):
        _graph_updated(self, bpy.context)


class MathOPSV2SDFSocket(NodeSocket):
    bl_idname = SOCKET_IDNAME
    bl_label = "SDF"

    def draw(self, context, layout, node, text):
        del context, node
        layout.label(text=text or self.bl_label)

    def draw_color(self, context, node):
        del context, node
        return 1.0, 0.45, 0.1, 1.0


class MathOPSV2SceneOutputNode(_MathOPSV2NodeBase, Node):
    bl_idname = OUTPUT_NODE_IDNAME
    bl_label = "Scene Output"

    def init(self, context):
        del context
        self.inputs.new(SOCKET_IDNAME, "Surface")

    def draw_buttons(self, context, layout):
        del context, layout
        pass

    def free(self):
        tree_name = getattr(getattr(self, "id_data", None), "name", "")
        if tree_name:
            bpy.app.timers.register(
                lambda tree_name=tree_name: _restore_output_later(tree_name),
                first_interval=0.0,
            )


class MathOPSV2SDFSphereNode(_MathOPSV2PrimitiveNodeBase, Node):
    bl_idname = SPHERE_NODE_IDNAME
    bl_label = "SDF Sphere"

    radius: FloatProperty(
        name="Radius",
        default=0.5,
        min=0.0,
        soft_min=0.0,
        update=_graph_updated,
    )

    def draw_buttons(self, context, layout):
        del context
        layout.prop(self, "radius")
        self._draw_primitive_footer(layout)


class MathOPSV2SDFBoxNode(_MathOPSV2PrimitiveNodeBase, Node):
    bl_idname = BOX_NODE_IDNAME
    bl_label = "SDF Box"

    size: FloatVectorProperty(
        name="Size",
        size=3,
        subtype="XYZ",
        default=(1.0, 1.0, 1.0),
        min=0.001,
        soft_min=0.001,
        update=_graph_updated,
    )
    bevel: FloatProperty(
        name="Bevel",
        default=0.0,
        min=0.0,
        soft_min=0.0,
        update=_graph_updated,
    )

    def draw_buttons(self, context, layout):
        del context
        layout.prop(self, "size")
        layout.prop(self, "bevel")
        self._draw_primitive_footer(layout)


class MathOPSV2SDFCylinderNode(_MathOPSV2PrimitiveNodeBase, Node):
    bl_idname = CYLINDER_NODE_IDNAME
    bl_label = "SDF Cylinder"

    radius: FloatProperty(
        name="Radius",
        default=0.35,
        min=0.0,
        soft_min=0.0,
        update=_graph_updated,
    )
    height: FloatProperty(
        name="Height",
        default=1.0,
        min=0.0,
        soft_min=0.0,
        update=_graph_updated,
    )

    def draw_buttons(self, context, layout):
        del context
        layout.prop(self, "radius")
        layout.prop(self, "height")
        self._draw_primitive_footer(layout)


class MathOPSV2SDFConeNode(_MathOPSV2PrimitiveNodeBase, Node):
    bl_idname = CONE_NODE_IDNAME
    bl_label = "SDF Cone"

    radius: FloatProperty(
        name="Radius",
        default=0.35,
        min=0.0,
        soft_min=0.0,
        update=_graph_updated,
    )
    height: FloatProperty(
        name="Height",
        default=1.0,
        min=0.0,
        soft_min=0.0,
        update=_graph_updated,
    )

    def draw_buttons(self, context, layout):
        del context
        layout.prop(self, "radius")
        layout.prop(self, "height")
        self._draw_primitive_footer(layout)


class MathOPSV2CSGUnionNode(_MathOPSV2CSGNodeBase, Node):
    bl_idname = UNION_NODE_IDNAME
    bl_label = "CSG Union"
    blend_mode = "union"


class MathOPSV2CSGSubtractNode(_MathOPSV2CSGNodeBase, Node):
    bl_idname = SUBTRACT_NODE_IDNAME
    bl_label = "CSG Subtract"
    blend_mode = "sub"


class MathOPSV2CSGIntersectNode(_MathOPSV2CSGNodeBase, Node):
    bl_idname = INTERSECT_NODE_IDNAME
    bl_label = "CSG Intersect"
    blend_mode = "inter"


def _matrix_to_json_values(matrix: Matrix):
    return [float(matrix[row][col]) for row in range(3) for col in range(4)]


def _json_matrix_to_world_to_local(matrix_values):
    values = [float(v) for v in matrix_values]
    return Matrix(
        (
            (values[0], values[1], values[2], values[3]),
            (values[4], values[5], values[6], values[7]),
            (values[8], values[9], values[10], values[11]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def _primitive_local_corners(node_payload):
    primitive_type = node_payload.get("primitiveType")
    if primitive_type == "sphere":
        radius = float(node_payload.get("radius", 0.0))
        half = Vector((radius, radius, radius))
    elif primitive_type == "box":
        sides = node_payload.get("sides", (0.0, 0.0, 0.0))
        half = Vector((float(sides[0]), float(sides[1]), float(sides[2]))) * 0.5
    elif primitive_type == "cylinder":
        radius = float(node_payload.get("radius", 0.0))
        half = Vector((radius, float(node_payload.get("height", 0.0)) * 0.5, radius))
    elif primitive_type == "cone":
        radius = float(node_payload.get("radius", 0.0))
        half = Vector((radius, float(node_payload.get("height", 0.0)) * 0.5, radius))
    else:
        half = Vector((0.0, 0.0, 0.0))

    corners = []
    for x in (-half.x, half.x):
        for y in (-half.y, half.y):
            for z in (-half.z, half.z):
                corners.append(Vector((x, y, z, 1.0)))
    return corners


def _union_bounds(bounds_a, bounds_b):
    if bounds_a is None:
        return bounds_b
    if bounds_b is None:
        return bounds_a
    min_a, max_a = bounds_a
    min_b, max_b = bounds_b
    return (
        tuple(min(a, b) for a, b in zip(min_a, min_b)),
        tuple(max(a, b) for a, b in zip(max_a, max_b)),
    )


def _inflate_bounds(bounds, padding):
    if bounds is None or padding <= 0.0:
        return bounds
    aabb_min, aabb_max = bounds
    return (
        tuple(float(v) - padding for v in aabb_min),
        tuple(float(v) + padding for v in aabb_max),
    )


def _payload_bounds(payload):
    stack = [(payload, Matrix.Identity(4), False)]
    bounds_by_id = {}
    pad_by_id = {}

    while stack:
        node_payload, parent_world_to_local, visited = stack.pop()
        if not isinstance(node_payload, dict):
            continue

        node_matrix = _json_matrix_to_world_to_local(
            node_payload.get(
                "matrix",
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            )
        )
        world_to_local = parent_world_to_local @ node_matrix
        node_id = id(node_payload)
        node_type = node_payload.get("nodeType")

        if visited:
            if node_type == "primitive":
                try:
                    local_to_world = world_to_local.inverted()
                except Exception:
                    bounds_by_id[node_id] = None
                    pad_by_id[node_id] = 0.0
                    continue
                world_points = [
                    local_to_world @ corner
                    for corner in _primitive_local_corners(node_payload)
                ]
                bounds_by_id[node_id] = (
                    tuple(min(point[i] for point in world_points) for i in range(3)),
                    tuple(max(point[i] for point in world_points) for i in range(3)),
                )
                pad_by_id[node_id] = 0.0
            elif node_type == "binaryOperator":
                left = node_payload.get("leftChild")
                right = node_payload.get("rightChild")
                left_bounds = bounds_by_id.get(id(left))
                right_bounds = bounds_by_id.get(id(right))
                bounds_by_id[node_id] = _union_bounds(left_bounds, right_bounds)
                local_pad = 0.0
                if node_payload.get("blendMode") == "union":
                    local_pad = (
                        max(0.0, float(node_payload.get("blendRadius", 0.0))) * 0.25
                    )
                pad_by_id[node_id] = max(
                    local_pad,
                    pad_by_id.get(id(left), 0.0),
                    pad_by_id.get(id(right), 0.0),
                )
            else:
                bounds_by_id[node_id] = None
                pad_by_id[node_id] = 0.0
            continue

        stack.append((node_payload, parent_world_to_local, True))
        if node_type == "binaryOperator":
            stack.append((node_payload.get("rightChild", {}), world_to_local, False))
            stack.append((node_payload.get("leftChild", {}), world_to_local, False))

    return _inflate_bounds(
        bounds_by_id.get(id(payload)), pad_by_id.get(id(payload), 0.0)
    )


def _count_payload_nodes(payload):
    if not isinstance(payload, dict):
        return 0
    count = 0
    stack = [payload]
    while stack:
        current = stack.pop()
        if not isinstance(current, dict):
            continue
        count += 1
        if current.get("nodeType") == "binaryOperator":
            left = current.get("leftChild")
            right = current.get("rightChild")
            if isinstance(left, dict):
                stack.append(left)
            if isinstance(right, dict):
                stack.append(right)
    return count


def _socket_source_node(socket, label, node):
    if not socket.is_linked or not socket.links:
        raise RuntimeError(f"Node '{node.name}' is missing its {label} input")
    return socket.links[0].from_node


def _serialize_primitive(node, primitive_type):
    payload = {
        "nodeType": "primitive",
        "primitiveType": primitive_type,
        "color": [float(v) for v in node.color],
        "round_x": 0.0,
        "round_y": 0.0,
        "matrix": _matrix_to_json_values(node._world_to_local_matrix()),
    }
    if primitive_type == "sphere":
        payload["radius"] = float(node.radius)
    elif primitive_type == "box":
        bevel = max(0.0, float(node.bevel))
        payload["sides"] = [float(v) for v in node.size]
        payload["bevel"] = [bevel, bevel, bevel, bevel]
    else:
        payload["radius"] = float(node.radius)
        payload["height"] = float(node.height)
    return payload


def _serialize_node(node, active_stack):
    node_key = node.as_pointer()
    if node_key in active_stack:
        raise RuntimeError(f"Cycle detected at node '{node.name}'")

    active_stack.add(node_key)
    try:
        if node.bl_idname == SPHERE_NODE_IDNAME:
            return _serialize_primitive(node, "sphere")
        if node.bl_idname == BOX_NODE_IDNAME:
            return _serialize_primitive(node, "box")
        if node.bl_idname == CYLINDER_NODE_IDNAME:
            return _serialize_primitive(node, "cylinder")
        if node.bl_idname == CONE_NODE_IDNAME:
            return _serialize_primitive(node, "cone")
        if node.bl_idname in {
            UNION_NODE_IDNAME,
            SUBTRACT_NODE_IDNAME,
            INTERSECT_NODE_IDNAME,
        }:
            left_child = _serialize_node(
                _socket_source_node(node.inputs[0], "A", node), active_stack
            )
            right_child = _serialize_node(
                _socket_source_node(node.inputs[1], "B", node), active_stack
            )
            return {
                "nodeType": "binaryOperator",
                "leftChild": left_child,
                "rightChild": right_child,
                "blendMode": node.blend_mode,
                "blendRadius": float(node.blend_radius),
                "matrix": list(_IDENTITY_MATRIX_3X4),
            }

        raise RuntimeError(f"Unsupported node type '{node.bl_label}'")
    finally:
        active_stack.remove(node_key)


def _root_output_node(tree):
    output_node = _find_output_node(tree)
    if (
        output_node is None
        or not output_node.inputs
        or not output_node.inputs[0].is_linked
    ):
        raise RuntimeError(f"Tree '{tree.name}' needs a connected Scene Output node")
    return output_node


def compile_tree_payload(tree):
    output_node = _root_output_node(tree)
    root_node = _socket_source_node(output_node.inputs[0], "Surface", output_node)
    payload = _serialize_node(root_node, set())
    bounds = _payload_bounds(payload)
    if bounds is None:
        bounds = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    compiled = dict(payload)
    compiled["aabb_min"] = [float(v) for v in bounds[0]]
    compiled["aabb_max"] = [float(v) for v in bounds[1]]
    return compiled


def compile_tree_json(tree):
    return json.dumps(
        compile_tree_payload(tree), ensure_ascii=True, separators=(",", ":")
    )


def generated_scene_dir() -> Path:
    path = Path(__file__).resolve().parent / ".generated_scenes"
    path.mkdir(parents=True, exist_ok=True)
    return path


def invalid_scene_path() -> Path:
    return generated_scene_dir() / _INVALID_SCENE_NAME


def _scene_file_name(tree_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", tree_name).strip("._")
    return f"{safe_name or 'scene'}.json"


def ensure_generated_scene(settings, create=False):
    tree = get_selected_tree(settings, create=create, ensure=False)
    scene_json = compile_tree_json(tree)
    scene_hash = hashlib.sha1(scene_json.encode("utf-8")).hexdigest()
    scene_path = generated_scene_dir() / _scene_file_name(tree.name)
    cache_key = tree.name_full
    cached = runtime.generated_scene_cache.get(cache_key)
    if cached and cached["hash"] == scene_hash and scene_path.is_file():
        return scene_path

    scene_path.write_text(scene_json, encoding="utf-8")
    runtime.generated_scene_cache[cache_key] = {
        "hash": scene_hash,
        "path": str(scene_path),
    }
    runtime.debug_log(f"Compiled SDF node tree '{tree.name}' to {scene_path.name}")
    return scene_path


class MATHOPS_V2_OT_edit_scene_sdf_tree(Operator):
    bl_idname = "mathops_v2.edit_scene_sdf_tree"
    bl_label = "Edit SDF Graph"
    bl_description = "Focus the scene SDF graph in the node editor"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        settings = getattr(getattr(context, "scene", None), "mathops_v2_settings", None)
        return settings is not None and getattr(settings, "use_sdf_nodes", False)

    def execute(self, context):
        tree = focus_scene_tree(context, create=True)
        if not _focus_tree_in_editor(context, tree):
            self.report({"INFO"}, "Open a Node Editor and switch it to MathOPS-v2 SDF")
        return {"FINISHED"}


class MATHOPS_V2_OT_new_scene_sdf_tree(Operator):
    bl_idname = "mathops_v2.new_scene_sdf_tree"
    bl_label = "New SDF Graph"
    bl_description = "Create another MathOPS-v2 SDF graph for this scene"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        settings = getattr(getattr(context, "scene", None), "mathops_v2_settings", None)
        return settings is not None and getattr(settings, "use_sdf_nodes", False)

    def execute(self, context):
        settings = context.scene.mathops_v2_settings
        tree = new_scene_tree(context.scene, settings)
        _focus_tree_in_editor(context, tree)
        _graph_updated(tree, context)
        return {"FINISHED"}


class MATHOPS_V2_PT_sdf_graph(Panel):
    bl_label = "MathOPS-v2"
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "UI"
    bl_category = "MathOPS-v2"

    @classmethod
    def poll(cls, context):
        space = getattr(context, "space_data", None)
        return (
            space is not None
            and space.type == "NODE_EDITOR"
            and space.tree_type == TREE_IDNAME
        )

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mathops_v2_settings

        layout.prop(settings, "use_sdf_nodes")
        if not settings.use_sdf_nodes:
            layout.label(text="Enable Use Nodes in Render settings", icon="INFO")
            return

        tree = _active_editor_tree(context, create=False)
        if tree is None:
            layout.label(text="Scene SDF tree unavailable", icon="ERROR")
            return

        row = layout.row(align=True)
        row.prop_search(
            settings, "sdf_node_tree", bpy.data, "node_groups", text="Graph"
        )
        row.operator(MATHOPS_V2_OT_new_scene_sdf_tree.bl_idname, text="", icon="ADD")
        layout.operator(MATHOPS_V2_OT_edit_scene_sdf_tree.bl_idname, icon="NODETREE")
        layout.separator()
        try:
            payload = compile_tree_payload(tree)
            layout.label(text=f"Nodes: {_count_payload_nodes(payload)}")
            layout.label(
                text=(
                    f"Bounds: ({payload['aabb_min'][0]:.2f}, {payload['aabb_min'][1]:.2f}, {payload['aabb_min'][2]:.2f}) "
                    f"to ({payload['aabb_max'][0]:.2f}, {payload['aabb_max'][1]:.2f}, {payload['aabb_max'][2]:.2f})"
                )
            )
        except Exception as exc:
            layout.label(text=str(exc), icon="ERROR")


class _MathOPSV2NodeCategory(NodeCategory):
    @classmethod
    def poll(cls, context):
        space = getattr(context, "space_data", None)
        return space is not None and getattr(space, "tree_type", None) == TREE_IDNAME


node_categories = [
    _MathOPSV2NodeCategory(
        "MATHOPS_V2_SDF_PRIMS",
        "Primitives",
        items=[
            NodeItem(SPHERE_NODE_IDNAME),
            NodeItem(BOX_NODE_IDNAME),
            NodeItem(CYLINDER_NODE_IDNAME),
            NodeItem(CONE_NODE_IDNAME),
        ],
    ),
    _MathOPSV2NodeCategory(
        "MATHOPS_V2_SDF_CSG",
        "CSG",
        items=[
            NodeItem(UNION_NODE_IDNAME),
            NodeItem(SUBTRACT_NODE_IDNAME),
            NodeItem(INTERSECT_NODE_IDNAME),
        ],
    ),
]


classes = (
    MathOPSV2SDFTree,
    MathOPSV2SDFSocket,
    MathOPSV2SceneOutputNode,
    MathOPSV2SDFSphereNode,
    MathOPSV2SDFBoxNode,
    MathOPSV2SDFCylinderNode,
    MathOPSV2SDFConeNode,
    MathOPSV2CSGUnionNode,
    MathOPSV2CSGSubtractNode,
    MathOPSV2CSGIntersectNode,
    MATHOPS_V2_OT_edit_scene_sdf_tree,
    MATHOPS_V2_OT_new_scene_sdf_tree,
    MATHOPS_V2_PT_sdf_graph,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    nodeitems_utils.register_node_categories(NODE_CATEGORY_ID, node_categories)


def unregister():
    nodeitems_utils.unregister_node_categories(NODE_CATEGORY_ID)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
