import uuid

import bpy
from bpy.props import EnumProperty, StringProperty
from bpy.types import Menu, Operator

from .. import meshing, properties, runtime
from ..nodes import sdf_tree
from ..render import picking


_addon_keymaps = []


_PRIMITIVE_ICONS = {
    "sphere": "MESH_UVSPHERE",
    "box": "MESH_CUBE",
    "cylinder": "MESH_CYLINDER",
    "torus": "MESH_TORUS",
    "cone": "MESH_CONE",
    "capsule": "MESH_UVSPHERE",
    "ngon": "MESH_CIRCLE",
    "polygon": "CURVE_DATA",
}

_EMPTY_DISPLAY_TYPES = {
    "sphere": "PLAIN_AXES",
    "box": "PLAIN_AXES",
    "cylinder": "PLAIN_AXES",
    "torus": "PLAIN_AXES",
    "cone": "PLAIN_AXES",
    "capsule": "PLAIN_AXES",
    "ngon": "PLAIN_AXES",
    "polygon": "PLAIN_AXES",
}


def _configure_proxy_defaults(obj, primitive_type):
    settings = obj.mathops_v2_sdf
    settings.enabled = True
    settings.proxy_id = uuid.uuid4().hex
    settings.source_tree_name = ""
    settings.source_node_name = ""
    settings.primitive_type = primitive_type
    settings.radius = 0.5
    settings.cone_top_radius = 0.0
    settings.cone_bottom_radius = 0.5
    settings.size = (0.5, 0.5, 0.5)
    settings.height = 1.0
    settings.major_radius = 0.75
    settings.minor_radius = 0.25
    settings.bevel = 0.0
    settings.cylinder_bevel_top = 0.0
    settings.cylinder_bevel_bottom = 0.0
    settings.cylinder_taper = 0.0
    settings.cylinder_bevel_mode = "SMOOTH"
    settings.cone_bevel_mode = "SMOOTH"
    settings.capsule_taper = 0.0
    settings.torus_angle = 6.283185307179586
    settings.box_corners = (0.0, 0.0, 0.0, 0.0)
    settings.box_edge_top = 0.0
    settings.box_edge_bottom = 0.0
    settings.box_taper = 0.0
    settings.box_corner_mode = "SMOOTH"
    settings.box_edge_mode = "SMOOTH"
    settings.ngon_sides = 6
    settings.ngon_corner = 0.0
    settings.ngon_edge_top = 0.0
    settings.ngon_edge_bottom = 0.0
    settings.ngon_taper = 0.0
    settings.ngon_edge_mode = "SMOOTH"
    settings.ngon_star = 0.0
    sdf_tree.set_polygon_points(settings, sdf_tree.default_polygon_control_points())
    settings.polygon_interpolation = "VECTOR"
    settings.polygon_edge_top = 0.0
    settings.polygon_edge_bottom = 0.0
    settings.polygon_taper = 0.0
    settings.polygon_edge_mode = "SMOOTH"
    settings.polygon_is_line = False
    settings.polygon_line_thickness = 0.1

    obj.rotation_mode = "XYZ"
    if getattr(obj, "type", "") == "EMPTY":
        obj.empty_display_type = _EMPTY_DISPLAY_TYPES.get(primitive_type, "PLAIN_AXES")
        obj.empty_display_size = 0.01
    obj.show_name = False
    obj.show_in_front = False
    obj.hide_render = True
    if primitive_type == "polygon" and getattr(obj, "type", "") == "CURVE":
        sdf_tree.sync_polygon_curve_proxy(obj, sdf_tree.default_polygon_control_points(), is_line=False, interpolation="VECTOR")


def _link_proxy_object(context, primitive_type):
    label = dict((identifier, name) for identifier, name, _description in properties.PRIMITIVE_ITEMS)[primitive_type]
    if primitive_type == "polygon":
        data = bpy.data.curves.new(f"SDF {label}", type="CURVE")
        obj = bpy.data.objects.new(f"SDF {label}", data)
    else:
        obj = bpy.data.objects.new(f"SDF {label}", None)
    _configure_proxy_defaults(obj, primitive_type)
    obj.location = context.scene.cursor.location.copy()
    collection = getattr(context, "collection", None)
    if collection is None:
        collection = context.scene.collection
    collection.objects.link(obj)
    for selected in context.selected_objects:
        selected.select_set(False)
    _select_proxy_object(context, obj)
    return obj


def _viewport_pick_enabled(context):
    area = getattr(context, "area", None)
    if area is None or area.type != "VIEW_3D":
        return False
    if getattr(getattr(context, "scene", None), "render", None) is None:
        return False
    if getattr(context.scene.render, "engine", "") != runtime.ENGINE_ID:
        return False
    if getattr(context, "region_data", None) is None:
        return False
    if getattr(context, "mode", "OBJECT") != "OBJECT":
        return False
    settings = runtime.scene_settings(context.scene)
    if settings is None or not bool(getattr(settings, "viewport_preview", False)):
        return False
    shading = getattr(getattr(context, "space_data", None), "shading", None)
    return shading is not None and getattr(shading, "type", "") == "RENDERED"


def _object_in_view_layer(context, obj):
    if obj is None:
        return False
    view_layer = getattr(context, "view_layer", None)
    if view_layer is None:
        return False
    try:
        layer_object = view_layer.objects.get(obj.name)
    except Exception:
        layer_object = None
    return runtime.safe_pointer(layer_object) == runtime.safe_pointer(obj)


def _select_proxy_object(context, obj, extend=False):
    if not _object_in_view_layer(context, obj):
        return False
    if not extend:
        for selected in context.selected_objects:
            if selected != obj:
                selected.select_set(False)
    try:
        obj.select_set(True)
        context.view_layer.objects.active = obj
    except RuntimeError:
        return False
    return True


def _generated_mesh_name(scene):
    return f"{scene.name} MathOPS Mesh"


def _ensure_generated_mesh_object(context):
    name = _generated_mesh_name(context.scene)
    obj = bpy.data.objects.get(name)
    if obj is not None and obj.type == "MESH":
        return obj

    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    obj["mathops_v2_generated_mesh"] = True
    collection = getattr(context, "collection", None)
    if collection is None:
        collection = context.scene.collection
    if collection.objects.get(obj.name) is None:
        collection.objects.link(obj)
    return obj


class MATHOPS_V2_OT_add_sdf_proxy(Operator):
    bl_idname = "mathops_v2.add_sdf_proxy"
    bl_label = "Add SDF Proxy"
    bl_description = "Add an SDF proxy object and insert it into the scene graph"
    bl_options = {"REGISTER", "UNDO"}

    primitive_type: EnumProperty(name="Primitive", items=properties.PRIMITIVE_ITEMS, default="sphere")

    def execute(self, context):
        obj = _link_proxy_object(context, self.primitive_type)
        sdf_tree.add_proxy_to_tree(context.scene, obj)
        runtime.clear_error()
        runtime.tag_redraw(context)
        return {"FINISHED"}


class MATHOPS_V2_OT_new_sdf_graph(Operator):
    bl_idname = "mathops_v2.new_sdf_graph"
    bl_label = "New SDF Graph"
    bl_description = "Create a fresh MathOPS SDF node graph for the scene"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        sdf_tree.new_scene_tree(context.scene)
        sdf_tree.focus_scene_tree(context, create=False)
        runtime.clear_error()
        runtime.tag_redraw(context)
        return {"FINISHED"}


class MATHOPS_V2_OT_edit_sdf_graph(Operator):
    bl_idname = "mathops_v2.edit_sdf_graph"
    bl_label = "Edit SDF Graph"
    bl_description = "Focus the scene SDF graph in a Node Editor"
    bl_options = {"REGISTER"}

    def execute(self, context):
        if not sdf_tree.focus_scene_tree(context, create=True):
            self.report({"INFO"}, "Open a Node Editor and switch it to MathOPS SDF")
        return {"FINISHED"}


class MATHOPS_V2_OT_extract_dual_contour_mesh(Operator):
    bl_idname = "mathops_v2.extract_dual_contour_mesh"
    bl_label = "Extract CPU Mesh"
    bl_description = "Build a Blender mesh from the current MathOPS SDF graph using CPU dual contouring"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = runtime.scene_settings(context.scene)
        if settings is None:
            self.report({"ERROR"}, "MathOPS scene settings are unavailable")
            return {"CANCELLED"}

        try:
            compiled = sdf_tree.compile_scene(context.scene)
            algorithm = str(getattr(settings, "mesh_algorithm", "DUAL_CONTOURING") or "DUAL_CONTOURING")
            resolution = getattr(settings, "mesh_resolution", 48)
            if algorithm == "ISO_SIMPLEX":
                result = meshing.extract_iso_simplex_mesh(compiled, resolution)
            else:
                result = meshing.extract_dual_contour_mesh(compiled, resolution)
        except Exception as exc:
            runtime.set_error(str(exc))
            self.report({"ERROR"}, str(exc))
            runtime.tag_redraw(context)
            return {"CANCELLED"}

        vertices = list(result.get("vertices", ()))
        triangles = list(result.get("triangles", ()))
        if not vertices or not triangles:
            message = "No surface generated for the current SDF graph"
            runtime.set_error(message)
            self.report({"WARNING"}, message)
            runtime.tag_redraw(context)
            return {"CANCELLED"}

        obj = _ensure_generated_mesh_object(context)
        old_mesh = getattr(obj, "data", None)
        mesh_name = getattr(old_mesh, "name", _generated_mesh_name(context.scene))
        smooth_shading = bool(getattr(settings, "mesh_smooth_shading", True))
        mesh = bpy.data.meshes.new(mesh_name)
        mesh.from_pydata(vertices, [], triangles)
        mesh.update()
        for polygon in mesh.polygons:
            polygon.use_smooth = smooth_shading
        obj.data = mesh
        if old_mesh is not None and old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)

        for selected in context.selected_objects:
            if selected != obj:
                selected.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj

        runtime.clear_error()
        runtime.note_interaction()
        runtime.tag_redraw(context)
        self.report(
            {"INFO"},
            f"{algorithm.replace('_', ' ').title()}: {len(vertices)} verts, {len(triangles)} tris | grid {tuple(result.get('grid_dimensions', (0, 0, 0)))} | active {int(result.get('active_cells', 0))}",
        )
        return {"FINISHED"}


class MATHOPS_V2_OT_move_sdf_branch(Operator):
    bl_idname = "mathops_v2.move_sdf_branch"
    bl_label = "Move SDF Branch"
    bl_description = "Move this CSG branch up or down in the main scene chain"
    bl_options = {"REGISTER", "UNDO"}

    tree_name: StringProperty(default="")
    branch_root_name: StringProperty(default="")
    direction: EnumProperty(
        name="Direction",
        items=(
            ("UP", "Up", "Move this branch earlier in the chain"),
            ("DOWN", "Down", "Move this branch later in the chain"),
        ),
        default="UP",
    )

    @classmethod
    def poll(cls, context):
        space = getattr(context, "space_data", None)
        if space is None or getattr(space, "type", "") != "NODE_EDITOR":
            return False
        tree = getattr(space, "edit_tree", None)
        return tree is not None and getattr(tree, "bl_idname", "") == runtime.TREE_IDNAME

    def execute(self, context):
        space = getattr(context, "space_data", None)
        tree = None if space is None else getattr(space, "edit_tree", None)
        if tree is None or getattr(tree, "bl_idname", "") != runtime.TREE_IDNAME:
            tree = bpy.data.node_groups.get(self.tree_name)
        if tree is None or getattr(tree, "bl_idname", "") != runtime.TREE_IDNAME:
            self.report({"ERROR"}, "SDF node tree not found")
            return {"CANCELLED"}

        branch_root_name = str(self.branch_root_name or "")
        if not branch_root_name:
            branch_root_name = sdf_tree.branch_root_name_for_node(tree, getattr(context, "active_node", None))
        if not branch_root_name:
            self.report({"INFO"}, "Select a branch node or branch frame")
            return {"CANCELLED"}

        moved_root = sdf_tree.move_branch_entry(tree, branch_root_name, self.direction)
        if moved_root is None:
            self.report({"INFO"}, "Branch can't move further in that direction")
            return {"CANCELLED"}

        sdf_tree.select_node_in_editor(context, moved_root, reveal=False)
        runtime.clear_error()
        runtime.tag_redraw(context)
        return {"FINISHED"}


class MATHOPS_V2_OT_create_node_proxy(Operator):
    bl_idname = "mathops_v2.create_node_proxy"
    bl_label = "Create Node Proxy"
    bl_description = "Create an SDF proxy object for this node"
    bl_options = {"REGISTER", "UNDO"}

    tree_name: StringProperty(default="")
    node_name: StringProperty(default="")
    primitive_type: EnumProperty(name="Primitive", items=properties.PRIMITIVE_ITEMS, default="sphere")

    def execute(self, context):
        tree = bpy.data.node_groups.get(self.tree_name)
        if tree is None or getattr(tree, "bl_idname", "") != runtime.TREE_IDNAME:
            self.report({"ERROR"}, "SDF node tree not found")
            return {"CANCELLED"}

        node = tree.nodes.get(self.node_name)
        if node is None or getattr(node, "bl_idname", "") != runtime.OBJECT_NODE_IDNAME:
            self.report({"ERROR"}, "SDF initializer node not found")
            return {"CANCELLED"}

        obj = _link_proxy_object(context, str(getattr(node, "primitive_type", self.primitive_type) or self.primitive_type))
        sdf_tree.attach_proxy_to_node(node, obj, adopt_proxy=False)
        runtime.clear_error()
        runtime.tag_redraw(context)
        return {"FINISHED"}


class MATHOPS_V2_OT_pick_sdf(Operator):
    bl_idname = "mathops_v2.pick_sdf"
    bl_label = "Pick SDF"
    bl_description = "Select the visible MathOPS SDF under the mouse"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return _viewport_pick_enabled(context)

    def invoke(self, context, event):
        if bool(getattr(event, "alt", False)) or bool(getattr(event, "ctrl", False)) or bool(getattr(event, "oskey", False)):
            return {"PASS_THROUGH"}
        try:
            hit = picking.pick_viewport_sdf(context, (float(event.mouse_region_x), float(event.mouse_region_y)))
        except Exception as exc:
            runtime.set_error(str(exc))
            runtime.tag_redraw(context)
            return {"PASS_THROUGH"}
        if hit is None:
            return {"PASS_THROUGH"}
        obj = hit.get("object")
        node = hit.get("node")
        if not runtime.is_sdf_proxy(obj) and node is None:
            return {"PASS_THROUGH"}
        if runtime.is_sdf_proxy(obj):
            _select_proxy_object(context, obj, extend=bool(getattr(event, "shift", False)))
        if node is not None:
            sdf_tree.select_node_in_editor(context, node, reveal=True)
        runtime.clear_error()
        runtime.note_interaction()
        runtime.tag_redraw(context)
        return {"FINISHED"}


class VIEW3D_MT_mathops_v2_add(Menu):
    bl_idname = "VIEW3D_MT_mathops_v2_add"
    bl_label = "MathOPS SDF"

    def draw(self, _context):
        layout = self.layout
        for primitive_type, label, _description in properties.PRIMITIVE_ITEMS:
            operator = layout.operator(
                MATHOPS_V2_OT_add_sdf_proxy.bl_idname,
                text=label,
                icon=_PRIMITIVE_ICONS.get(primitive_type, "EMPTY_AXIS"),
            )
            operator.primitive_type = primitive_type


def _draw_add_menu(self, _context):
    self.layout.separator()
    self.layout.menu(VIEW3D_MT_mathops_v2_add.bl_idname, icon="META_BALL")


classes = (
    MATHOPS_V2_OT_add_sdf_proxy,
    MATHOPS_V2_OT_new_sdf_graph,
    MATHOPS_V2_OT_edit_sdf_graph,
    MATHOPS_V2_OT_extract_dual_contour_mesh,
    MATHOPS_V2_OT_move_sdf_branch,
    MATHOPS_V2_OT_create_node_proxy,
    MATHOPS_V2_OT_pick_sdf,
    VIEW3D_MT_mathops_v2_add,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_MT_add.append(_draw_add_menu)

    window_manager = getattr(bpy.context, "window_manager", None)
    keyconfigs = None if window_manager is None else getattr(window_manager, "keyconfigs", None)
    keyconfig = None if keyconfigs is None else getattr(keyconfigs, "addon", None)
    if keyconfig is None:
        return

    keymap = keyconfig.keymaps.new(name="3D View", space_type="VIEW_3D")
    keymap_item = keymap.keymap_items.new(MATHOPS_V2_OT_pick_sdf.bl_idname, type="LEFTMOUSE", value="CLICK", head=True)
    _addon_keymaps.append((keymap, keymap_item))

    keymap = keyconfig.keymaps.new(name="Node Editor", space_type="NODE_EDITOR")
    keymap_item = keymap.keymap_items.new(
        MATHOPS_V2_OT_move_sdf_branch.bl_idname,
        type="A",
        value="PRESS",
        ctrl=True,
        head=True,
    )
    keymap_item.properties.direction = "UP"
    _addon_keymaps.append((keymap, keymap_item))

    keymap_item = keymap.keymap_items.new(
        MATHOPS_V2_OT_move_sdf_branch.bl_idname,
        type="D",
        value="PRESS",
        ctrl=True,
        head=True,
    )
    keymap_item.properties.direction = "DOWN"
    _addon_keymaps.append((keymap, keymap_item))


def unregister():
    for keymap, keymap_item in _addon_keymaps:
        try:
            keymap.keymap_items.remove(keymap_item)
        except Exception:
            pass
    _addon_keymaps.clear()

    try:
        bpy.types.VIEW3D_MT_add.remove(_draw_add_menu)
    except Exception:
        pass
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
