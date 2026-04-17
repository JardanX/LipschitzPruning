from contextlib import contextmanager
import uuid

import bpy
import nodeitems_utils
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, PointerProperty, StringProperty
from bpy.types import Node, NodeSocket, NodeTree
from mathutils import Euler, Matrix, Vector
from nodeitems_utils import NodeCategory, NodeItem

from .. import runtime


NODE_CATEGORY_ID = "MATHOPS_V2_SDF_NODES"

_PRIMITIVE_TYPE_TO_ID = {
    "sphere": 0.0,
    "box": 1.0,
    "cylinder": 2.0,
    "torus": 3.0,
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
)

_NODE_PRIMITIVE_LABELS = {
    identifier: label for identifier, label, _description in _NODE_PRIMITIVE_ITEMS
}

_object_node_updates_suppressed = 0


def _tag_redraw(_self=None, context=None):
    runtime.clear_error()
    runtime.note_interaction()
    runtime.tag_redraw(context)


@contextmanager
def suppress_object_node_updates():
    global _object_node_updates_suppressed
    _object_node_updates_suppressed += 1
    try:
        yield
    finally:
        _object_node_updates_suppressed -= 1


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
    ensure_object_node_id(self)
    sync_node_to_proxy(self)
    _mark_tree_dirty(getattr(self, "id_data", None), static=False, context=context)
    _tag_redraw(self, context)


def _update_object_node_payload(self, context):
    if _object_node_updates_suppressed > 0:
        return
    ensure_object_node_id(self)
    sync_node_to_proxy(self)
    _mark_tree_dirty(getattr(self, "id_data", None), static=True, context=context)
    _tag_redraw(self, context)


def _update_object_node_target(self, context):
    if _object_node_updates_suppressed > 0:
        return
    self.use_proxy = self.target is not None
    ensure_object_node_id(self)
    sync_node_to_proxy(self)
    _mark_tree_dirty(getattr(self, "id_data", None), static=True, context=context)
    _tag_redraw(self, context)


def _update_csg_node_data(self, context):
    _mark_tree_dirty(getattr(self, "id_data", None), static=True, context=context)
    _tag_redraw(self, context)


class MathOPSV2SDFSocket(NodeSocket):
    bl_idname = "MathOPSV2SDFSocket"
    bl_label = "SDF"

    def draw(self, _context, layout, _node, text):
        layout.label(text=text or self.name)

    def draw_color(self, _context, _node):
        return (0.45, 0.65, 1.0, 1.0)


class MathOPSV2NodeTree(NodeTree):
    bl_idname = runtime.TREE_IDNAME
    bl_label = "MathOPS SDF"
    bl_icon = "NODETREE"

    def update(self):
        if Node.bl_rna_get_subclass_py(runtime.OUTPUT_NODE_IDNAME, None) is None:
            return
        ensure_graph_output(self)
        _mark_tree_dirty(self, static=True)
        runtime.note_interaction()
        runtime.tag_redraw()


class _MathOPSV2NodeBase(Node):
    @classmethod
    def poll(cls, node_tree):
        return getattr(node_tree, "bl_idname", "") == runtime.TREE_IDNAME

    def update(self):
        runtime.note_interaction()
        runtime.tag_redraw()


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

    def init(self, _context):
        self.outputs.new(MathOPSV2SDFSocket.bl_idname, "SDF")
        ensure_object_node_id(self)

    def copy(self, _node):
        with suppress_object_node_updates():
            self.proxy_id = uuid.uuid4().hex
            self.target = None
            self.use_proxy = False

    def draw_buttons(self, _context, layout):
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

        transform_box = layout.box()
        transform_box.label(text="Transform")
        transform_box.prop(self, "sdf_location")
        transform_box.prop(self, "sdf_rotation", text="Rotation")
        transform_box.prop(self, "sdf_scale")

        primitive_box = layout.box()
        primitive_box.label(text="Primitive")
        primitive_box.prop(self, "primitive_type", text="Type")
        primitive_type = str(self.primitive_type or "sphere")
        if primitive_type == "sphere":
            primitive_box.prop(self, "radius")
        elif primitive_type == "box":
            primitive_box.prop(self, "size")
        elif primitive_type == "cylinder":
            primitive_box.prop(self, "radius")
            primitive_box.prop(self, "height")
        elif primitive_type == "torus":
            primitive_box.prop(self, "major_radius")
            primitive_box.prop(self, "minor_radius")

        meta_box = layout.box()
        meta_box.label(text="Viewport Handle")
        if self.target is None:
            meta_box.label(text="No proxy handle assigned")
        else:
            meta_box.prop(self.target, "name", text="Name")
        meta_box.prop(self, "proxy_id", text="ID")

    def draw_label(self):
        primitive_name = _NODE_PRIMITIVE_LABELS.get(self.primitive_type, "SDF")
        return f"{primitive_name} Initializer"


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
        self.inputs.new(MathOPSV2SDFSocket.bl_idname, "A")
        self.inputs.new(MathOPSV2SDFSocket.bl_idname, "B")
        self.outputs.new(MathOPSV2SDFSocket.bl_idname, "SDF")

    def draw_buttons(self, _context, layout):
        layout.prop(self, "operation", text="")
        layout.prop(self, "blend")


def _find_output_node(tree):
    for node in tree.nodes:
        if getattr(node, "bl_idname", "") == runtime.OUTPUT_NODE_IDNAME:
            return node
    return None


def _candidate_root_nodes(tree):
    candidates = []
    for node in tree.nodes:
        if getattr(node, "bl_idname", "") == runtime.OUTPUT_NODE_IDNAME:
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


def _linked_source_node(socket):
    if not socket.is_linked or not socket.links:
        return None
    return socket.links[0].from_node


def _surface_output_socket(node):
    if not getattr(node, "outputs", None):
        return None
    return node.outputs[0]


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


def focus_scene_tree(context, create=True):
    tree = get_scene_tree(context.scene, create=create)
    if tree is None:
        return False
    screen = getattr(context, "screen", None)
    if screen is None:
        return False
    for area in screen.areas:
        if area.type != "NODE_EDITOR":
            continue
        for space in area.spaces:
            if space.type != "NODE_EDITOR":
                continue
            space.tree_type = runtime.TREE_IDNAME
            space.node_tree = tree
            area.tag_redraw()
            return True
    return False


def list_proxy_objects(scene):
    proxies = [obj for obj in scene.objects if runtime.is_sdf_proxy(obj)]
    proxies.sort(key=lambda obj: obj.name_full)
    return proxies


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


def ensure_object_node_id(node):
    proxy_id = str(getattr(node, "proxy_id", "") or "")
    if proxy_id:
        return proxy_id
    proxy_id = uuid.uuid4().hex
    with suppress_object_node_updates():
        node.proxy_id = proxy_id
    return proxy_id


def find_initializer_node(tree, obj=None, proxy_id=""):
    object_pointer = runtime.safe_pointer(obj)
    proxy_id = str(proxy_id or "")
    for node in initializer_nodes(tree):
        if object_pointer and runtime.safe_pointer(getattr(node, "target", None)) == object_pointer:
            return node
        if proxy_id and str(getattr(node, "proxy_id", "") or "") == proxy_id:
            return node
    return None


def _float_seq_changed(lhs, rhs, epsilon=1.0e-6):
    if len(lhs) != len(rhs):
        return True
    for left, right in zip(lhs, rhs):
        if abs(float(left) - float(right)) > epsilon:
            return True
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


def _node_transform_values(node):
    target = getattr(node, "target", None)
    if runtime.is_sdf_proxy(target):
        return (
            tuple(float(component) for component in target.location),
            tuple(float(component) for component in target.rotation_euler),
            tuple(abs(float(component)) for component in target.scale),
        )
    return (
        tuple(float(component) for component in node.sdf_location),
        tuple(float(component) for component in node.sdf_rotation),
        tuple(abs(float(component)) for component in node.sdf_scale),
    )


def _configure_proxy_display(obj):
    if obj.rotation_mode != "XYZ":
        obj.rotation_mode = "XYZ"
    if obj.empty_display_type != "PLAIN_AXES":
        obj.empty_display_type = "PLAIN_AXES"
    if abs(float(obj.empty_display_size) - 0.01) > 1.0e-6:
        obj.empty_display_size = 0.01
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
    _set_float_attr(settings, "radius", node.radius)
    _set_float_vector_attr(settings, "size", node.size)
    _set_float_attr(settings, "height", node.height)
    _set_float_attr(settings, "major_radius", node.major_radius)
    _set_float_attr(settings, "minor_radius", node.minor_radius)
    return settings


def sync_node_to_proxy(node, include_transform=True):
    target = getattr(node, "target", None)
    if target is None or not runtime.is_sdf_proxy(target):
        return None

    ensure_object_node_id(node)
    _configure_proxy_display(target)
    _mirror_node_to_proxy_settings(node, target)
    if include_transform:
        _set_float_vector_attr(target, "location", node.sdf_location)
        _set_float_vector_attr(target, "rotation_euler", node.sdf_rotation)
        _set_float_vector_attr(target, "scale", node.sdf_scale)
    return target


def sync_proxy_to_node(node):
    target = getattr(node, "target", None)
    if target is None or not runtime.is_sdf_proxy(target):
        return False

    changed = False
    with suppress_object_node_updates():
        if not bool(getattr(node, "use_proxy", False)):
            node.use_proxy = True
            changed = True
        proxy_id = str(getattr(runtime.object_settings(target), "proxy_id", "") or "")
        if proxy_id and str(getattr(node, "proxy_id", "") or "") != proxy_id:
            node.proxy_id = proxy_id
            changed = True
        target_location = tuple(float(component) for component in target.location)
        target_rotation = tuple(float(component) for component in target.rotation_euler)
        target_scale = tuple(abs(float(component)) for component in target.scale)
        if _float_seq_changed(tuple(node.sdf_location), target_location):
            node.sdf_location = target_location
            changed = True
        if _float_seq_changed(tuple(node.sdf_rotation), target_rotation):
            node.sdf_rotation = target_rotation
            changed = True
        if _float_seq_changed(tuple(node.sdf_scale), target_scale):
            node.sdf_scale = target_scale
            changed = True
    return changed


def _adopt_proxy_data(node, obj):
    settings = runtime.object_settings(obj)
    with suppress_object_node_updates():
        if settings is not None:
            proxy_id = str(settings.proxy_id or "")
            node.proxy_id = proxy_id or uuid.uuid4().hex
            node.primitive_type = str(settings.primitive_type or "sphere")
            node.radius = float(settings.radius)
            node.size = tuple(float(component) for component in settings.size)
            node.height = float(settings.height)
            node.major_radius = float(settings.major_radius)
            node.minor_radius = float(settings.minor_radius)
        else:
            ensure_object_node_id(node)
        node.sdf_location = tuple(float(component) for component in obj.location)
        node.sdf_rotation = tuple(float(component) for component in obj.rotation_euler)
        node.sdf_scale = tuple(float(component) for component in obj.scale)
        node.use_proxy = True
        node.target = obj


def attach_proxy_to_node(node, obj, adopt_proxy=False):
    if adopt_proxy:
        _adopt_proxy_data(node, obj)
        sync_node_to_proxy(node, include_transform=True)
    else:
        with suppress_object_node_updates():
            node.use_proxy = True
            node.target = obj
        sync_node_to_proxy(node, include_transform=True)
    runtime.tag_redraw()
    return obj


def add_proxy_to_tree(scene, obj):
    tree = ensure_scene_tree(scene)
    output = _ensure_output_node(tree)
    existing_root = _linked_source_node(output.inputs[0])
    for node in tree.nodes:
        if getattr(node, "bl_idname", "") == runtime.OBJECT_NODE_IDNAME and getattr(node, "target", None) == obj:
            return tree, node

    object_node = tree.nodes.new(runtime.OBJECT_NODE_IDNAME)
    attach_proxy_to_node(object_node, obj, adopt_proxy=True)
    runtime.mark_scene_static_dirty(scene)
    if existing_root is None:
        object_node.location = (120.0, 0.0)
        output.location = (400.0, 0.0)
        tree.links.new(object_node.outputs[0], output.inputs[0])
        runtime.tag_redraw()
        return tree, object_node

    old_link = output.inputs[0].links[0]
    tree.links.remove(old_link)
    csg_node = tree.nodes.new(runtime.CSG_NODE_IDNAME)
    csg_node.operation = "UNION"
    csg_node.location = (existing_root.location.x + 260.0, existing_root.location.y)
    object_node.location = (existing_root.location.x, existing_root.location.y - 220.0)
    output.location = (csg_node.location.x + 280.0, csg_node.location.y)
    tree.links.new(_surface_output_socket(existing_root), csg_node.inputs[0])
    tree.links.new(object_node.outputs[0], csg_node.inputs[1])
    tree.links.new(csg_node.outputs[0], output.inputs[0])
    runtime.tag_redraw()
    return tree, object_node


def _node_transform_matrix(node):
    location_values, rotation_values, _scale_values = _node_transform_values(node)
    location = Vector(location_values)
    rotation = Euler(rotation_values, "XYZ")
    return Matrix.LocRotScale(location, rotation, None)


def _primitive_parameters_from_node(node):
    primitive_type = str(node.primitive_type or "sphere").strip().lower()
    meta = [0.0, 0.0, 0.0, 0.0]
    meta[0] = _PRIMITIVE_TYPE_TO_ID.get(primitive_type, 0.0)
    if primitive_type == "sphere":
        meta[1] = float(node.radius)
    elif primitive_type == "box":
        meta[1] = float(node.size[0])
        meta[2] = float(node.size[1])
        meta[3] = float(node.size[2])
    elif primitive_type == "cylinder":
        meta[1] = float(node.radius)
        meta[2] = float(node.height) * 0.5
    elif primitive_type == "torus":
        meta[1] = float(node.major_radius)
        meta[2] = float(node.minor_radius)
    else:
        raise RuntimeError(f"Unsupported primitive type '{primitive_type}'")
    _location, _rotation, scale = _node_transform_values(node)
    return primitive_type, tuple(meta), scale


def _primitive_local_extents(primitive_type, meta, scale):
    if primitive_type == "sphere":
        return Vector((meta[1] * scale[0], meta[1] * scale[1], meta[1] * scale[2]))
    if primitive_type == "box":
        return Vector((meta[1] * scale[0], meta[2] * scale[1], meta[3] * scale[2]))
    if primitive_type == "cylinder":
        radial_scale = max(0.5 * (scale[0] + scale[2]), 1.0e-6)
        return Vector((meta[1] * radial_scale, meta[2] * scale[1], meta[1] * radial_scale))
    if primitive_type == "torus":
        radial_scale = max(0.5 * (scale[0] + scale[2]), 1.0e-6)
        minor_scale = max(min(radial_scale, scale[1]), 1.0e-6)
        outer = (meta[1] * radial_scale) + (meta[2] * minor_scale)
        vertical = meta[2] * minor_scale
        return Vector((outer, vertical, outer))
    return Vector((1.0, 1.0, 1.0))


def _primitive_lipschitz(primitive_type, scale):
    if primitive_type == "sphere":
        min_scale = max(min(scale[0], scale[1], scale[2]), 1.0e-6)
        return max(scale[0], scale[1], scale[2]) / min_scale
    return 1.0


def _primitive_spec_from_node(node, primitive_index):
    location, _rotation, _scale = _node_transform_values(node)
    primitive_type, meta, scale = _primitive_parameters_from_node(node)
    transform = _node_transform_matrix(node)
    world_to_local = transform.inverted_safe()
    rows = [tuple(float(value) for value in world_to_local[index]) for index in range(3)]
    center = location
    rotation = transform.to_3x3()
    local_extents = _primitive_local_extents(primitive_type, meta, scale)
    world_extents = Vector((
        abs(rotation[0][0]) * local_extents[0] + abs(rotation[0][1]) * local_extents[1] + abs(rotation[0][2]) * local_extents[2],
        abs(rotation[1][0]) * local_extents[0] + abs(rotation[1][1]) * local_extents[1] + abs(rotation[1][2]) * local_extents[2],
        abs(rotation[2][0]) * local_extents[0] + abs(rotation[2][1]) * local_extents[1] + abs(rotation[2][2]) * local_extents[2],
    ))
    bounds_min = tuple(center[axis] - float(world_extents[axis]) for axis in range(3))
    bounds_max = tuple(center[axis] + float(world_extents[axis]) for axis in range(3))
    return {
        "primitive_index": int(primitive_index),
        "primitive_type": primitive_type,
        "meta": meta,
        "world_to_local": tuple(rows),
        "scale": tuple(scale),
        "center": center,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "lipschitz": float(_primitive_lipschitz(primitive_type, scale)),
    }


def _scene_bounds_from_specs(primitive_specs):
    if not primitive_specs:
        return ((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))

    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    for spec in primitive_specs:
        for axis in range(3):
            mins[axis] = min(mins[axis], float(spec["bounds_min"][axis]))
            maxs[axis] = max(maxs[axis], float(spec["bounds_max"][axis]))

    extents = [maxs[axis] - mins[axis] for axis in range(3)]
    max_extent = max(extents) if extents else 1.0
    padding = max(0.5, max_extent * 0.1)
    return (
        tuple(mins[axis] - padding for axis in range(3)),
        tuple(maxs[axis] + padding for axis in range(3)),
    )


def _primitive_rows_from_node(node):
    _primitive_type, meta, scale = _primitive_parameters_from_node(node)
    transform = _node_transform_matrix(node).inverted_safe()
    rows = [tuple(float(value) for value in transform[index]) for index in range(3)]
    return [tuple(meta), rows[0], rows[1], rows[2], (scale[0], scale[1], scale[2], 0.0)]


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


def _emit_node(node, primitive_rows, instructions, primitive_map, primitive_specs, primitive_nodes, visiting):
    node_key = node.as_pointer()
    if node_key in visiting:
        raise RuntimeError(f"Cycle detected at node '{node.name}'")

    visiting.add(node_key)
    try:
        node_idname = getattr(node, "bl_idname", "")
        if node_idname == runtime.OBJECT_NODE_IDNAME:
            ensure_object_node_id(node)
            proxy_key = _node_key(node)
            primitive_index = primitive_map.get(proxy_key)
            if primitive_index is None:
                primitive_index = len(primitive_map)
                primitive_map[proxy_key] = primitive_index
                primitive_rows.extend(_primitive_rows_from_node(node))
                primitive_specs.append(_primitive_spec_from_node(node, primitive_index))
                primitive_nodes.append(node)
            instruction_index = len(instructions)
            _append_instruction(instructions, (0.0, float(primitive_index), 0.0, 0.0))
            return {
                "kind": "primitive",
                "primitive_index": int(primitive_index),
                "instruction_index": int(instruction_index),
            }

        if node_idname == runtime.CSG_NODE_IDNAME:
            left = _linked_source_node(node.inputs[0])
            right = _linked_source_node(node.inputs[1])
            if left is None or right is None:
                raise RuntimeError(f"Node '{node.name}' needs both inputs connected")
            left_compiled = _emit_node(left, primitive_rows, instructions, primitive_map, primitive_specs, primitive_nodes, visiting)
            right_compiled = _emit_node(right, primitive_rows, instructions, primitive_map, primitive_specs, primitive_nodes, visiting)
            operation = _CSG_OPERATION_TO_ID.get(getattr(node, "operation", "UNION"), 1.0)
            blend = float(getattr(node, "blend", 0.0))
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
    instructions = []
    primitive_map = {}
    primitive_specs = []
    primitive_nodes = []
    root_node = _emit_node(root, primitive_rows, instructions, primitive_map, primitive_specs, primitive_nodes, set())
    return primitive_rows, instructions, primitive_specs, primitive_nodes, root_node


def _compile_scene_union(scene):
    primitive_rows = []
    instructions = []
    primitive_specs = []
    primitive_nodes = []
    root_node = None
    proxies = list_proxy_objects(scene)
    tree = get_scene_tree(scene, create=False)
    for primitive_index, obj in enumerate(proxies):
        node = find_initializer_node(tree, obj=obj)
        if node is not None:
            primitive_rows.extend(_primitive_rows_from_node(node))
            primitive_specs.append(_primitive_spec_from_node(node, primitive_index))
            primitive_nodes.append(node)
        else:
            temp_node = type("_TempNode", (), {})()
            settings = runtime.object_settings(obj)
            temp_node.primitive_type = getattr(settings, "primitive_type", "sphere") if settings is not None else "sphere"
            temp_node.radius = getattr(settings, "radius", 0.5) if settings is not None else 0.5
            temp_node.size = getattr(settings, "size", (0.5, 0.5, 0.5)) if settings is not None else (0.5, 0.5, 0.5)
            temp_node.height = getattr(settings, "height", 1.0) if settings is not None else 1.0
            temp_node.major_radius = getattr(settings, "major_radius", 0.75) if settings is not None else 0.75
            temp_node.minor_radius = getattr(settings, "minor_radius", 0.25) if settings is not None else 0.25
            temp_node.sdf_location = tuple(float(component) for component in obj.location)
            temp_node.sdf_rotation = tuple(float(component) for component in obj.rotation_euler)
            temp_node.sdf_scale = tuple(float(component) for component in obj.scale)
            primitive_rows.extend(_primitive_rows_from_node(temp_node))
            primitive_specs.append(_primitive_spec_from_node(temp_node, primitive_index))
            primitive_nodes.append(temp_node)
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
    return primitive_rows, instructions, primitive_specs, primitive_nodes, root_node


def refresh_compiled_scene_dynamic(compiled):
    primitive_nodes = list(compiled.get("primitive_nodes", ()))
    if not primitive_nodes:
        return compiled

    primitive_rows = []
    primitive_specs = []
    for primitive_index, node in enumerate(primitive_nodes):
        primitive_rows.extend(_primitive_rows_from_node(node))
        primitive_specs.append(_primitive_spec_from_node(node, primitive_index))

    scene_bounds = _scene_bounds_from_specs(primitive_specs)
    instruction_rows = list(compiled.get("instruction_rows", ()))
    rows = primitive_rows + instruction_rows
    refreshed = dict(compiled)
    refreshed.update(
        {
            "primitive_rows": primitive_rows,
            "primitive_specs": primitive_specs,
            "primitive_count": len(primitive_rows) // runtime.PRIMITIVE_TEXELS,
            "scene_bounds": scene_bounds,
            "rows": rows,
            "hash": runtime.hash_compiled_rows(primitive_rows, instruction_rows),
        }
    )
    return refreshed


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

    for node in list(tree.nodes):
        if getattr(node, "bl_idname", "") != runtime.OBJECT_NODE_IDNAME:
            continue
        target = getattr(node, "target", None)
        target_pointer = runtime.safe_pointer(target)
        if target_pointer and runtime.is_sdf_proxy(target) and (valid_targets is None or target_pointer in valid_targets):
            continue
        if not bool(getattr(node, "use_proxy", False)):
            continue
        tree.nodes.remove(node)
        changed = True

    simplified = True
    while simplified:
        simplified = False
        for node in list(tree.nodes):
            if getattr(node, "bl_idname", "") != runtime.CSG_NODE_IDNAME:
                continue
            left = _linked_source_node(node.inputs[0])
            right = _linked_source_node(node.inputs[1])
            replacement_socket = None
            if left is None and right is None:
                replacement_socket = None
            elif left is None:
                replacement_socket = _surface_output_socket(right)
            elif right is None:
                replacement_socket = _surface_output_socket(left)
            else:
                continue

            output_socket = _surface_output_socket(node)
            consumers = [link.to_socket for link in list(output_socket.links)]
            for link in list(output_socket.links):
                tree.links.remove(link)
            if replacement_socket is not None:
                for to_socket in consumers:
                    for link in list(to_socket.links):
                        tree.links.remove(link)
                    tree.links.new(replacement_socket, to_socket)
            tree.nodes.remove(node)
            changed = True
            simplified = True
            break

    ensure_graph_output(tree)
    if changed:
        scene = _scene_for_tree(tree)
        if scene is not None:
            runtime.mark_scene_static_dirty(scene)
    return changed


def compile_scene(scene):
    primitive_rows = []
    instructions = []
    primitive_specs = []
    root_node = None
    message = ""
    settings = runtime.scene_settings(scene)
    tree = None if settings is None else getattr(settings, "node_tree", None)
    if tree is not None and getattr(tree, "bl_idname", "") == runtime.TREE_IDNAME:
        try:
            primitive_rows, instructions, primitive_specs, primitive_nodes, root_node = _compile_tree(tree)
        except RuntimeError as exc:
            message = f"Graph fallback: {exc}"
            primitive_nodes = []
    else:
        primitive_nodes = []

    if not instructions:
        primitive_rows, instructions, primitive_specs, primitive_nodes, root_node = _compile_scene_union(scene)

    stack_depth = _stack_usage(instructions)
    if stack_depth > runtime.MAX_STACK:
        raise RuntimeError(
            f"Graph stack depth {stack_depth} exceeds shader stack limit {runtime.MAX_STACK}"
        )

    scene_bounds = _scene_bounds_from_specs(primitive_specs)
    scene_hash = runtime.hash_compiled_rows(primitive_rows, instructions)
    topology_hash = runtime.hash_instruction_rows(instructions)
    return {
        "primitive_rows": primitive_rows,
        "instruction_rows": instructions,
        "primitive_specs": primitive_specs,
        "primitive_nodes": primitive_nodes,
        "root_node": root_node,
        "primitive_count": len(primitive_rows) // runtime.PRIMITIVE_TEXELS,
        "instruction_count": len(instructions),
        "stack_depth": stack_depth,
        "scene_bounds": scene_bounds,
        "rows": primitive_rows + instructions,
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
]


classes = (
    MathOPSV2SDFSocket,
    MathOPSV2OutputNode,
    MathOPSV2ObjectNode,
    MathOPSV2CSGNode,
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
