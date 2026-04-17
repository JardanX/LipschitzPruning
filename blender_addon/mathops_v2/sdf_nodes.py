import hashlib
import json
import re
import struct
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import bpy
import nodeitems_utils
from bpy.app.handlers import persistent
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    StringProperty,
)
from bpy.types import Node, NodeSocket, NodeTree, Operator, Panel
from mathutils import Euler, Matrix, Vector
from nodeitems_utils import NodeCategory, NodeItem

from . import runtime


TREE_IDNAME = "MATHOPS_V2_SDF_TREE"
SOCKET_IDNAME = "MATHOPS_V2_SDF_SOCKET"
FLOAT_SOCKET_IDNAME = "MATHOPS_V2_FLOAT_SOCKET"
VECTOR_SOCKET_IDNAME = "MATHOPS_V2_VECTOR_SOCKET"
COLOR_SOCKET_IDNAME = "MATHOPS_V2_COLOR_SOCKET"
TRANSFORM_SOCKET_IDNAME = "MATHOPS_V2_TRANSFORM_SOCKET"
OUTPUT_NODE_IDNAME = "MATHOPS_V2_SCENE_OUTPUT"
SPHERE_NODE_IDNAME = "MATHOPS_V2_SDF_SPHERE"
BOX_NODE_IDNAME = "MATHOPS_V2_SDF_BOX"
CYLINDER_NODE_IDNAME = "MATHOPS_V2_SDF_CYLINDER"
CONE_NODE_IDNAME = "MATHOPS_V2_SDF_CONE"
CSG_NODE_IDNAME = "MATHOPS_V2_CSG"
UNION_NODE_IDNAME = "MATHOPS_V2_CSG_UNION"
SUBTRACT_NODE_IDNAME = "MATHOPS_V2_CSG_SUBTRACT"
INTERSECT_NODE_IDNAME = "MATHOPS_V2_CSG_INTERSECT"
VALUE_NODE_IDNAME = "MATHOPS_V2_VALUE"
VECTOR_NODE_IDNAME = "MATHOPS_V2_VECTOR"
COLOR_NODE_IDNAME = "MATHOPS_V2_COLOR"
TRANSFORM_NODE_IDNAME = "MATHOPS_V2_TRANSFORM"
BREAK_TRANSFORM_NODE_IDNAME = "MATHOPS_V2_BREAK_TRANSFORM"
PRIMITIVE_NODE_IDNAMES = {
    SPHERE_NODE_IDNAME,
    BOX_NODE_IDNAME,
    CYLINDER_NODE_IDNAME,
    CONE_NODE_IDNAME,
}
NODE_CATEGORY_ID = "MATHOPS_V2_SDF_NODES"
_INVALID_SCENE_NAME = "_invalid_.json"
_OUTPUT_ENFORCE_LOCKS = set()
_GRAPH_UPDATE_SUPPRESS = 0
_GRAPH_UPDATE_PENDING = False
_AUTO_INSERT_SUPPRESS = 0
_EDITOR_WINDOW_SCENES = {}
_EDITOR_SPACE_STATE = {}
_SYNC_MSGBUS_OWNER = object()
_PROXY_TREE_MATERIALIZED_KEY = "mathops_v2_proxy_tree_materialized"
_IDENTITY_MATRIX_3X4 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
_LIVE_GRAPH_COMPILE_GRACE = 0.2
_LIVE_GRAPH_COMPILE_INTERVAL = 1.0 / 30.0
_BLENDER_TO_RENDERER_BASIS = Matrix(
    (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
)
_RENDERER_TO_BLENDER_BASIS = _BLENDER_TO_RENDERER_BASIS.inverted()
_CSG_BLEND_MODE_ITEMS = (
    ("union", "Union", "Combine both SDF shapes"),
    ("sub", "Subtract", "Subtract B from A"),
    ("inter", "Intersect", "Keep only the overlap"),
)


def _graph_updated(_owner=None, context=None):
    global _GRAPH_UPDATE_PENDING
    if _GRAPH_UPDATE_SUPPRESS > 0:
        _GRAPH_UPDATE_PENDING = True
        return

    start = time.perf_counter()
    tree = (
        _owner
        if getattr(_owner, "bl_idname", "") == TREE_IDNAME
        else getattr(_owner, "id_data", None)
    )
    _activate_tree_for_context(tree, context)

    runtime.last_error_message = ""
    runtime.graph_interaction_time = time.perf_counter()

    try:
        from .render import bridge

        bridge.force_redraw_viewports(context)
    except Exception:
        pass
    owner_scene = _tree_owner_scene(tree)
    if owner_scene is not None and time.perf_counter() < getattr(
        runtime, "graph_sync_suppressed_until", 0.0
    ):
        _bootstrap_proxy_union_tree(tree)
        if getattr(tree, "bl_idname", "") == TREE_IDNAME:
            runtime.generated_scene_dirty.add(tree.name_full)
        runtime.debug_slow(
            f"Graph update suppressed tree={getattr(tree, 'name', 'Unknown')}",
            (time.perf_counter() - start) * 1000.0,
        )
        return
    try:
        from . import sdf_proxies

        if owner_scene is not None and is_primitive_node(_owner):
            if sdf_proxies.sync_primitive_node_update(owner_scene, _owner, context):
                mark_tree_transform_dirty(tree)
                runtime.debug_slow(
                    f"Graph update primitive-fast tree={getattr(tree, 'name', 'Unknown')}",
                    (time.perf_counter() - start) * 1000.0,
                )
                return
        sdf_proxies.sync_from_graph(context)
    except Exception:
        pass
    if getattr(tree, "bl_idname", "") == TREE_IDNAME:
        runtime.generated_scene_dirty.add(tree.name_full)
    runtime.debug_slow(
        f"Graph update full-sync tree={getattr(tree, 'name', 'Unknown')}",
        (time.perf_counter() - start) * 1000.0,
    )


def mark_tree_dirty(tree):
    if getattr(tree, "bl_idname", "") == TREE_IDNAME:
        runtime.generated_scene_dirty.add(tree.name_full)

    runtime.last_error_message = ""
    runtime.graph_interaction_time = time.perf_counter()


def mark_tree_transform_dirty(tree):
    if getattr(tree, "bl_idname", "") == TREE_IDNAME:
        runtime.scene_transform_dirty.add(tree.name_full)

    runtime.graph_interaction_time = time.perf_counter()


def suspend_graph_updates():
    global _GRAPH_UPDATE_SUPPRESS
    _GRAPH_UPDATE_SUPPRESS += 1


def resume_graph_updates(context=None, flush=True):
    global _GRAPH_UPDATE_SUPPRESS, _GRAPH_UPDATE_PENDING
    if _GRAPH_UPDATE_SUPPRESS <= 0:
        return
    _GRAPH_UPDATE_SUPPRESS -= 1
    if _GRAPH_UPDATE_SUPPRESS == 0 and _GRAPH_UPDATE_PENDING:
        _GRAPH_UPDATE_PENDING = False
        if flush:
            _graph_updated(context=context)


@contextmanager
def suppress_auto_insert():
    global _AUTO_INSERT_SUPPRESS
    _AUTO_INSERT_SUPPRESS += 1
    try:
        yield
    finally:
        _AUTO_INSERT_SUPPRESS -= 1


@contextmanager
def deferred_graph_updates(context=None, flush=True):
    suspend_graph_updates()
    try:
        yield
    finally:
        resume_graph_updates(context, flush=flush)


def _owner_scene(settings):
    scene = getattr(settings, "id_data", None)
    return scene if getattr(scene, "bl_rna", None) == bpy.types.Scene.bl_rna else None


def _default_tree_name(scene) -> str:
    return f"{scene.name} MathOPS SDF"


def _scene_identifier(scene) -> str:
    scene_id = str(getattr(scene, "mathops_v2_scene_id", "") or "")
    if not scene_id:
        scene_id = uuid.uuid4().hex
        try:
            scene.mathops_v2_scene_id = scene_id
        except Exception:
            pass
    return scene_id


def _tree_owner_scene_id(tree) -> str:
    return str(getattr(tree, "owner_scene_id", "") or "")


def _claim_tree_for_scene(tree, scene, force=False) -> bool:
    if getattr(tree, "bl_idname", "") != TREE_IDNAME or scene is None:
        return False

    scene_id = _scene_identifier(scene)
    owner_scene_id = _tree_owner_scene_id(tree)
    if owner_scene_id == scene_id:
        return True
    if owner_scene_id and not force:
        return False
    try:
        tree.owner_scene_id = scene_id
    except Exception:
        return False
    return True


def _tree_belongs_to_scene(tree, scene, claim_unowned=False) -> bool:
    if getattr(tree, "bl_idname", "") != TREE_IDNAME or scene is None:
        return False

    owner_scene_id = _tree_owner_scene_id(tree)
    if owner_scene_id == _scene_identifier(scene):
        return True
    if owner_scene_id:
        return False
    return _claim_tree_for_scene(tree, scene) if claim_unowned else False


def _tree_owner_scene(tree):
    owner_scene_id = _tree_owner_scene_id(tree)
    if not owner_scene_id:
        return None
    for scene in bpy.data.scenes:
        if _scene_identifier(scene) == owner_scene_id:
            return scene
    return None


def _set_selected_tree(settings, tree, scene=None, force=False):
    if getattr(tree, "bl_idname", "") != TREE_IDNAME:
        return False

    if scene is None:
        scene = _owner_scene(settings)
    if scene is not None and not _claim_tree_for_scene(tree, scene, force=force):
        return False
    if settings is not None and getattr(settings, "sdf_node_tree", None) != tree:
        settings.sdf_node_tree = tree
    return True


def _preferred_scene_tree(scene):
    preferred = bpy.data.node_groups.get(_default_tree_name(scene))
    if preferred is not None and _tree_belongs_to_scene(
        preferred, scene, claim_unowned=True
    ):
        return preferred

    owned_trees = sorted(
        (
            tree
            for tree in bpy.data.node_groups
            if getattr(tree, "bl_idname", "") == TREE_IDNAME
            and _tree_belongs_to_scene(tree, scene, claim_unowned=False)
        ),
        key=lambda tree: tree.name_full,
    )
    if owned_trees:
        return owned_trees[0]

    if preferred is not None and getattr(preferred, "bl_idname", "") == TREE_IDNAME:
        return preferred if _claim_tree_for_scene(preferred, scene) else None
    return None


def _activate_tree_for_context(tree, context=None):
    if getattr(tree, "bl_idname", "") != TREE_IDNAME:
        return

    scene = None if context is None else getattr(context, "scene", None)
    if scene is not None:
        settings = getattr(scene, "mathops_v2_settings", None)
        if settings is not None and getattr(settings, "use_sdf_nodes", False):
            _set_selected_tree(settings, tree, scene, force=True)
        return

    owner_scene = _tree_owner_scene(tree)
    if owner_scene is None:
        return
    settings = getattr(owner_scene, "mathops_v2_settings", None)
    if settings is None or not getattr(settings, "use_sdf_nodes", False):
        return
    _set_selected_tree(settings, tree, owner_scene)


def _surface_output_socket(node):
    if getattr(node, "bl_idname", "") == OUTPUT_NODE_IDNAME:
        return None
    for socket in getattr(node, "outputs", []):
        if getattr(socket, "bl_idname", "") == SOCKET_IDNAME:
            return socket
    return None


def is_primitive_node(node) -> bool:
    return getattr(node, "bl_idname", "") in PRIMITIVE_NODE_IDNAMES


def iter_primitive_nodes(tree):
    for node in getattr(tree, "nodes", []):
        if is_primitive_node(node):
            yield node


def primitive_node_token(node) -> str:
    token = str(getattr(node, "proxy_token", "") or "")
    if not token:
        token = uuid.uuid4().hex
        try:
            node.proxy_token = token
        except Exception:
            pass
    return token


def find_primitive_node(tree, node_id):
    if not node_id:
        return None
    for node in iter_primitive_nodes(tree):
        token = str(getattr(node, "proxy_token", "") or "")
        if not token:
            token = primitive_node_token(node)
        if token == node_id:
            return node
    return None


def active_primitive_node(scene, create=False):
    settings = getattr(scene, "mathops_v2_settings", None)
    if settings is None or not getattr(settings, "use_sdf_nodes", False):
        return None
    try:
        tree = get_selected_tree(settings, create=create, ensure=create)
    except Exception:
        return None
    active = getattr(getattr(tree, "nodes", None), "active", None)
    if is_primitive_node(active):
        return active

    active_object = getattr(bpy.context, "active_object", None)
    if getattr(active_object, "mathops_v2_sdf_proxy", False):
        return find_primitive_node(tree, active_object.mathops_v2_sdf_node_id)

    active_id = str(getattr(scene, "mathops_v2_active_primitive_id", "") or "")
    if active_id:
        return find_primitive_node(tree, active_id)
    return None


def _auto_insert_primitive_node(node):
    if _AUTO_INSERT_SUPPRESS > 0:
        return
    if getattr(node, "bl_idname", "") not in PRIMITIVE_NODE_IDNAMES:
        return

    tree = getattr(node, "id_data", None)
    if getattr(tree, "bl_idname", "") != TREE_IDNAME:
        return
    try:
        if getattr(node, "outputs", None) and any(node.outputs[0].links):
            return
    except Exception:
        pass
    try:
        insert_primitive_above_output(tree, node)
    except Exception:
        tree_name = getattr(tree, "name_full", "")
        node_name = getattr(node, "name", "")
        if tree_name and node_name:
            bpy.app.timers.register(
                lambda tree_name=tree_name,
                node_name=node_name: _retry_auto_insert_primitive_node(
                    tree_name, node_name
                ),
                first_interval=0.0,
            )
        return

    try:
        tree.nodes.active = node
        node.select = True
    except Exception:
        pass


def _retry_auto_insert_primitive_node(tree_name, node_name):
    tree = bpy.data.node_groups.get(tree_name)
    if getattr(tree, "bl_idname", "") != TREE_IDNAME:
        return None
    node = getattr(getattr(tree, "nodes", None), "get", lambda _name: None)(node_name)
    if node is not None:
        _auto_insert_primitive_node(node)
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


def proxy_tree_materialized(tree) -> bool:
    try:
        return bool(tree.get(_PROXY_TREE_MATERIALIZED_KEY, False))
    except Exception:
        return False


def set_proxy_tree_materialized(tree, value: bool):
    try:
        tree[_PROXY_TREE_MATERIALIZED_KEY] = bool(value)
    except Exception:
        pass


def _is_proxy_union_node(node) -> bool:
    node_idname = getattr(node, "bl_idname", "")
    if node_idname == UNION_NODE_IDNAME:
        return abs(float(getattr(node, "blend_radius", 0.0))) <= 1.0e-9
    if node_idname == CSG_NODE_IDNAME:
        return (
            getattr(node, "blend_mode", "") == "union"
            and abs(float(getattr(node, "blend_radius", 0.0))) <= 1.0e-9
        )
    return False


def _is_proxy_csg_union_node(node) -> bool:
    node_idname = getattr(node, "bl_idname", "")
    if node_idname == UNION_NODE_IDNAME:
        return True
    if node_idname == CSG_NODE_IDNAME:
        return getattr(node, "blend_mode", "") == "union"
    return False


def _bootstrap_proxy_union_tree(tree) -> bool:
    primitive_count = 0
    for node in tree.nodes:
        node_idname = getattr(node, "bl_idname", "")
        if node_idname == OUTPUT_NODE_IDNAME:
            continue
        if is_primitive_node(node):
            primitive_count += 1
            primitive_node_token(node)
            if not getattr(node, "proxy_managed", False):
                node.proxy_managed = True
            continue
        if _is_proxy_union_node(node):
            continue
        return False
    return primitive_count > 0


def _is_pure_proxy_union_tree(tree) -> bool:
    for node in tree.nodes:
        node_idname = getattr(node, "bl_idname", "")
        if node_idname == OUTPUT_NODE_IDNAME:
            continue
        if is_primitive_node(node):
            if not getattr(node, "proxy_managed", False):
                return False
            continue
        if _is_proxy_union_node(node):
            continue
        return False
    return True


def _is_pure_proxy_csg_union_tree(tree) -> bool:
    has_primitive = False
    for node in tree.nodes:
        node_idname = getattr(node, "bl_idname", "")
        if node_idname == OUTPUT_NODE_IDNAME:
            continue
        if is_primitive_node(node):
            has_primitive = True
            if not getattr(node, "proxy_managed", False):
                return False
            continue
        if _is_proxy_csg_union_node(node):
            continue
        return False
    return has_primitive


def _tree_open_in_node_editor(tree) -> bool:
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type != "NODE_EDITOR":
                    continue
                space = area.spaces.active
                if getattr(space, "tree_type", None) != TREE_IDNAME:
                    continue
                if getattr(space, "node_tree", None) == tree:
                    return True
    except Exception:
        pass
    return False


def dematerialize_proxy_tree(tree):
    if not proxy_tree_materialized(tree):
        return False
    if not _is_pure_proxy_union_tree(tree):
        return False
    if _tree_open_in_node_editor(tree):
        return False

    output = _find_output_node(tree)
    if output is None:
        output = _ensure_output_node(tree)
    removable_names = [node.name for node in tree.nodes if node != output]
    for name in removable_names:
        node = tree.nodes.get(name)
        if node is not None and node != output:
            tree.nodes.remove(node)
    for link in list(output.inputs[0].links):
        tree.links.remove(link)
    set_proxy_tree_materialized(tree, False)
    return True


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
        created_output = False

        if output_node is None:
            output_node = tree.nodes.new(OUTPUT_NODE_IDNAME)
            created_output = True

        for node in list(output_nodes):
            if node != output_node:
                tree.nodes.remove(node)

        source_node = _surface_root_candidate(tree)
        if created_output:
            _position_output_node(tree, output_node, source_node)

        if (
            source_node is not None
            and output_node.inputs
            and not output_node.inputs[0].is_linked
        ):
            source_socket = _surface_output_socket(source_node)
            if source_socket is not None:
                tree.links.new(source_socket, output_node.inputs[0])

        return output_node
    finally:
        _OUTPUT_ENFORCE_LOCKS.discard(tree_key)


def _build_default_tree(tree):
    _ensure_output_node(tree)
    return tree


def ensure_scene_tree(scene, settings=None):
    if settings is None:
        settings = scene.mathops_v2_settings

    tree = getattr(settings, "sdf_node_tree", None)
    if tree is not None and _tree_belongs_to_scene(tree, scene, claim_unowned=True):
        _build_default_tree(tree)
        _set_selected_tree(settings, tree, scene)
        return tree

    tree = _preferred_scene_tree(scene)
    if tree is None:
        tree_name = _default_tree_name(scene)
        existing = bpy.data.node_groups.get(tree_name)
        if existing is not None and existing.bl_idname == TREE_IDNAME:
            tree = existing
        else:
            tree = bpy.data.node_groups.new(tree_name, TREE_IDNAME)
        _claim_tree_for_scene(tree, scene, force=True)
    _build_default_tree(tree)
    _set_selected_tree(settings, tree, scene)
    runtime.debug_log(f"Using scene SDF tree '{tree.name}'")
    return tree


def new_scene_tree(scene, settings=None):
    if settings is None:
        settings = scene.mathops_v2_settings

    tree = bpy.data.node_groups.new(_default_tree_name(scene), TREE_IDNAME)
    _claim_tree_for_scene(tree, scene, force=True)
    _build_default_tree(tree)
    _set_selected_tree(settings, tree, scene)
    runtime.debug_log(f"Created scene SDF tree '{tree.name}'")
    return tree


def get_selected_tree(settings, create=False, ensure=False):
    scene = _owner_scene(settings)
    tree = getattr(settings, "sdf_node_tree", None)
    if tree is not None and getattr(tree, "bl_idname", "") == TREE_IDNAME:
        if scene is None or _tree_belongs_to_scene(tree, scene, claim_unowned=True):
            if ensure:
                _build_default_tree(tree)
            if scene is not None:
                _set_selected_tree(settings, tree, scene)
            return tree

    if scene is not None:
        tree = _preferred_scene_tree(scene)
        if tree is not None:
            _set_selected_tree(settings, tree, scene)
            if ensure:
                _build_default_tree(tree)
            return tree

    if not create:
        raise RuntimeError("Enable Use Nodes to create a scene SDF graph")

    if scene is None:
        raise RuntimeError("Scene-owned SDF tree is unavailable")
    return ensure_scene_tree(scene, settings)


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


def _frame_active_tree_node_in_editor(context, tree):
    framed = False
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "NODE_EDITOR":
                continue
            space = area.spaces.active
            if getattr(space, "tree_type", None) != TREE_IDNAME:
                continue
            if getattr(space, "node_tree", None) != tree:
                continue
            region = next(
                (item for item in area.regions if item.type == "WINDOW"), None
            )
            if region is None:
                continue
            try:
                with context.temp_override(
                    window=window,
                    area=area,
                    region=region,
                    space_data=space,
                ):
                    bpy.ops.node.view_selected()
                area.tag_redraw()
                framed = True
            except Exception:
                pass
    return framed


def _select_tree_node(tree, node, extend=False):
    if tree is None or node is None:
        return False
    if not extend:
        for tree_node in tree.nodes:
            tree_node.select = False
    node.select = True
    tree.nodes.active = node
    return True


def focus_scene_tree(context, create=False):
    settings = context.scene.mathops_v2_settings
    tree = get_selected_tree(settings, create=create, ensure=create)
    _focus_tree_in_editor(context, tree)
    return tree


def focus_scene_node(context, primitive_id, create=False, extend=False, frame=False):
    settings = context.scene.mathops_v2_settings
    tree = get_selected_tree(settings, create=create, ensure=create)
    node = find_primitive_node(tree, primitive_id)
    if node is None:
        _focus_tree_in_editor(context, tree)
        return tree, None, False

    _select_tree_node(tree, node, extend=extend)
    focused = _focus_tree_in_editor(context, tree)
    if frame:
        focused = _frame_active_tree_node_in_editor(context, tree) or focused
    return tree, node, focused


def _sync_node_editors():
    try:
        initialize_scene_trees()
        live_window_keys = set()
        live_space_keys = set()
        for window in bpy.context.window_manager.windows:
            scene = getattr(window, "scene", None)
            if scene is None:
                continue
            window_key = window.as_pointer()
            live_window_keys.add(window_key)
            settings = scene.mathops_v2_settings
            if not getattr(settings, "use_sdf_nodes", False):
                _EDITOR_WINDOW_SCENES[window_key] = _scene_identifier(scene)
                continue
            scene_key = _scene_identifier(scene)
            window_scene_changed = _EDITOR_WINDOW_SCENES.get(window_key) != scene_key
            _EDITOR_WINDOW_SCENES[window_key] = scene_key
            tree = ensure_scene_tree(scene, settings)
            for area in window.screen.areas:
                if area.type != "NODE_EDITOR":
                    continue
                space = area.spaces.active
                if getattr(space, "tree_type", None) != TREE_IDNAME:
                    continue
                space_key = space.as_pointer()
                live_space_keys.add(space_key)
                editor_tree = getattr(space, "node_tree", None)
                if getattr(editor_tree, "bl_idname", "") != TREE_IDNAME:
                    editor_tree = None
                previous_scene_key, previous_tree_key = _EDITOR_SPACE_STATE.get(
                    space_key, (None, None)
                )
                current_tree_key = (
                    0 if editor_tree is None else editor_tree.as_pointer()
                )
                editor_changed = current_tree_key != previous_tree_key

                if editor_tree is not None and editor_changed:
                    if editor_tree != tree:
                        _set_selected_tree(settings, editor_tree, scene, force=True)
                        tree = editor_tree
                elif editor_tree != tree:
                    try:
                        space.node_tree = tree
                        area.tag_redraw()
                    except Exception:
                        pass
                    editor_tree = tree
                    current_tree_key = tree.as_pointer()

                _EDITOR_SPACE_STATE[space_key] = (scene_key, current_tree_key)
        stale_window_keys = set(_EDITOR_WINDOW_SCENES) - live_window_keys
        for window_key in stale_window_keys:
            _EDITOR_WINDOW_SCENES.pop(window_key, None)
        stale_space_keys = set(_EDITOR_SPACE_STATE) - live_space_keys
        for space_key in stale_space_keys:
            _EDITOR_SPACE_STATE.pop(space_key, None)
    except Exception:
        pass
    return 0.1


def start_editor_sync():
    if bpy.app.timers.is_registered(_sync_node_editors):
        bpy.app.timers.unregister(_sync_node_editors)
    _sync_node_editors()


def stop_editor_sync():
    if bpy.app.timers.is_registered(_sync_node_editors):
        bpy.app.timers.unregister(_sync_node_editors)
    _EDITOR_WINDOW_SCENES.clear()
    _EDITOR_SPACE_STATE.clear()


def _ensure_handler(handler_list, callback):
    if callback not in handler_list:
        handler_list.append(callback)


def _remove_handler(handler_list, callback):
    if callback in handler_list:
        handler_list.remove(callback)


def _notify_editor_sync():
    _sync_node_editors()


def _subscribe_editor_sync_bus():
    try:
        bpy.msgbus.subscribe_rna(
            key=(bpy.types.Window, "scene"),
            owner=_SYNC_MSGBUS_OWNER,
            args=(),
            notify=_notify_editor_sync,
        )
        bpy.msgbus.subscribe_rna(
            key=(bpy.types.SpaceNodeEditor, "node_tree"),
            owner=_SYNC_MSGBUS_OWNER,
            args=(),
            notify=_notify_editor_sync,
        )
        bpy.msgbus.subscribe_rna(
            key=(bpy.types.SpaceNodeEditor, "tree_type"),
            owner=_SYNC_MSGBUS_OWNER,
            args=(),
            notify=_notify_editor_sync,
        )
    except Exception:
        pass


def _unsubscribe_editor_sync_bus():
    try:
        bpy.msgbus.clear_by_owner(_SYNC_MSGBUS_OWNER)
    except Exception:
        pass


def initialize_scene_trees():
    scenes = getattr(bpy.data, "scenes", None)
    if scenes is None:
        return False

    for scene in scenes:
        settings = scene.mathops_v2_settings
        if getattr(settings, "use_sdf_nodes", False):
            ensure_scene_tree(scene, settings)
    return True


def _deferred_post_register():
    if not initialize_scene_trees():
        return 0.1
    runtime.graph_sync_suppressed_until = time.perf_counter() + 0.5
    return None


@persistent
def _scene_tree_load_post(_dummy):
    runtime.graph_sync_suppressed_until = time.perf_counter() + 0.5
    initialize_scene_trees()
    start_editor_sync()


def post_register():
    _ensure_handler(bpy.app.handlers.load_post, _scene_tree_load_post)
    if not bpy.app.timers.is_registered(_deferred_post_register):
        bpy.app.timers.register(_deferred_post_register, first_interval=0.0)
    _subscribe_editor_sync_bus()
    start_editor_sync()


def pre_unregister():
    _remove_handler(bpy.app.handlers.load_post, _scene_tree_load_post)
    if bpy.app.timers.is_registered(_deferred_post_register):
        bpy.app.timers.unregister(_deferred_post_register)
    _unsubscribe_editor_sync_bus()
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

    scene = context.scene
    editor_tree = getattr(space, "node_tree", None)
    if getattr(editor_tree, "bl_idname", "") == TREE_IDNAME:
        _set_selected_tree(settings, editor_tree, scene, force=True)
        return editor_tree

    try:
        tree = get_selected_tree(settings, create=create, ensure=create)
    except Exception:
        return None
    if getattr(space, "node_tree", None) != tree:
        try:
            space.node_tree = tree
            context.area.tag_redraw()
        except Exception:
            pass
    return tree


def materialize_proxy_tree(scene, tree):
    if proxy_tree_materialized(tree):
        return tree

    for node in tree.nodes:
        node_idname = getattr(node, "bl_idname", "")
        if node_idname == OUTPUT_NODE_IDNAME:
            continue
        if is_primitive_node(node):
            if not getattr(node, "proxy_managed", False):
                return tree
            continue
        if node_idname == CSG_NODE_IDNAME:
            if getattr(node, "blend_mode", "") != "union":
                return tree
            if abs(float(getattr(node, "blend_radius", 0.0))) > 1.0e-9:
                return tree
            continue
        return tree

    try:
        from . import sdf_proxies
    except Exception:
        return tree

    cached = runtime.generated_scene_cache.get(tree.name_full, {})
    proxy_order = list(cached.get("proxy_order") or [])
    records = list(getattr(scene, "mathops_v2_primitives", ()))
    ordered_entries = []
    if records:
        record_lookup = {str(record.primitive_id): record for record in records}
        ordered_entries = [
            (proxy_id, record_lookup[proxy_id])
            for proxy_id in proxy_order
            if proxy_id in record_lookup
        ]
        seen_ids = {proxy_id for proxy_id, _record in ordered_entries}
        ordered_entries.extend(
            (str(record.primitive_id), record)
            for record in records
            if str(record.primitive_id) not in seen_ids
        )
    else:
        proxies = [
            obj for obj in scene.objects if getattr(obj, "mathops_v2_sdf_proxy", False)
        ]
        if not proxies:
            set_proxy_tree_materialized(tree, False)
            return tree
        proxy_lookup = {}
        for obj in proxies:
            proxy_id = str(
                getattr(obj, "mathops_v2_sdf_node_id", "")
                or getattr(obj, "mathops_v2_sdf_proxy_id", "")
            )
            if not proxy_id:
                continue
            proxy_lookup[proxy_id] = obj
        ordered_entries = [
            (proxy_id, proxy_lookup[proxy_id])
            for proxy_id in proxy_order
            if proxy_id in proxy_lookup
        ]
        seen_names = {obj.name for _proxy_id, obj in ordered_entries}
        ordered_entries.extend(
            (
                str(
                    getattr(obj, "mathops_v2_sdf_node_id", "")
                    or getattr(obj, "mathops_v2_sdf_proxy_id", "")
                ),
                obj,
            )
            for obj in proxies
            if obj.name not in seen_names
        )

    with (
        sdf_proxies.suppress_proxy_sync_handlers(),
        sdf_proxies.suppress_graph_to_proxy_sync(),
        deferred_graph_updates(bpy.context),
    ):
        output = _ensure_output_node(tree)
        for node in list(tree.nodes):
            if node != output:
                tree.nodes.remove(node)
        for primitive_id, source in ordered_entries:
            if hasattr(source, "primitive_type"):
                primitive_type = str(source.primitive_type or "sphere").strip().lower()
            else:
                primitive_type = (
                    str(source.get("sdf_type", "sphere") or "sphere").strip().lower()
                )
            node = create_primitive_node(tree, primitive_type)
            node.proxy_managed = True
            node.proxy_token = primitive_id
            insert_primitive_above_output(tree, node)
            if hasattr(source, "primitive_type"):
                node.sdf_location = tuple(float(v) for v in source.location)
                node.sdf_rotation = tuple(float(v) for v in source.rotation)
                node.sdf_scale = tuple(float(v) for v in source.scale)
                node.color = tuple(float(v) for v in source.color)
                if primitive_type == "box":
                    node.size = tuple(float(v) for v in source.size)
                    node.bevel = float(source.bevel)
                else:
                    node.radius = float(source.radius)
                    if primitive_type in {"cylinder", "cone"}:
                        node.height = float(source.height)
            else:
                source.mathops_v2_sdf_node_id = node.proxy_token
                sdf_proxies._apply_proxy_to_node(source, node)

    set_proxy_tree_materialized(tree, True)
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

    def _socket_linked(self, socket_name: str) -> bool:
        socket = getattr(getattr(self, "inputs", None), "get", lambda _name: None)(
            socket_name
        )
        return bool(socket is not None and socket.is_linked)

    def _draw_fallback_prop(
        self, layout, prop_name: str, socket_name: str | None = None
    ):
        row = layout.row()
        if socket_name is not None:
            row.enabled = not self._socket_linked(socket_name)
        row.prop(self, prop_name)

    def _draw_transform(self, layout):
        box = layout.box()
        box.label(text="Transform")
        col = box.column(align=True)
        col.enabled = not self._socket_linked("Transform")
        col.prop(self, "sdf_location")
        col.prop(self, "sdf_rotation")
        col.prop(self, "sdf_scale")

    def _world_to_local_matrix(self):
        return node_effective_world_to_local_matrix(self)


def _get_node_input(node, socket_name: str):
    return getattr(getattr(node, "inputs", None), "get", lambda _name: None)(
        socket_name
    )


def node_input_source_socket(node, socket_name: str):
    socket = _get_node_input(node, socket_name)
    if socket is None or not socket.is_linked or not socket.links:
        return None
    return socket.links[0].from_socket


def node_input_source_node(node, socket_name: str):
    socket = node_input_source_socket(node, socket_name)
    return None if socket is None else getattr(socket, "node", None)


def _transform_value(location, rotation, scale):
    return {
        "location": tuple(float(v) for v in location),
        "rotation": tuple(float(v) for v in rotation),
        "scale": tuple(float(v) for v in scale),
    }


def _evaluate_output_socket(socket, active_stack):
    node = getattr(socket, "node", None)
    if node is None:
        raise RuntimeError("Linked data socket has no source node")

    node_key = ("data", node.as_pointer(), str(getattr(socket, "name", "")))
    if node_key in active_stack:
        raise RuntimeError(f"Cycle detected at node '{node.name}'")

    active_stack.add(node_key)
    try:
        node_idname = getattr(node, "bl_idname", "")
        if node_idname == VALUE_NODE_IDNAME:
            return "float", float(getattr(node, "value", 0.0))
        if node_idname == VECTOR_NODE_IDNAME:
            return "vector", tuple(
                float(v) for v in getattr(node, "value", (0.0, 0.0, 0.0))
            )
        if node_idname == COLOR_NODE_IDNAME:
            return "color", tuple(
                float(v) for v in getattr(node, "value", (0.8, 0.8, 0.8))
            )
        if node_idname == TRANSFORM_NODE_IDNAME:
            location = _socket_data_value(
                _get_node_input(node, "Location"),
                "vector",
                active_stack,
                tuple(
                    float(v) for v in getattr(node, "location_value", (0.0, 0.0, 0.0))
                ),
            )
            rotation = _socket_data_value(
                _get_node_input(node, "Rotation"),
                "vector",
                active_stack,
                tuple(
                    float(v) for v in getattr(node, "rotation_value", (0.0, 0.0, 0.0))
                ),
            )
            scale = _socket_data_value(
                _get_node_input(node, "Scale"),
                "vector",
                active_stack,
                tuple(float(v) for v in getattr(node, "scale_value", (1.0, 1.0, 1.0))),
            )
            return (
                "transform",
                _transform_value(location, rotation, scale),
            )
        if node_idname == BREAK_TRANSFORM_NODE_IDNAME:
            transform = _socket_data_value(
                _get_node_input(node, "Transform"), "transform", active_stack, None
            )
            if transform is None:
                raise RuntimeError(f"Node '{node.name}' is missing its Transform input")
            output_name = str(getattr(socket, "name", ""))
            if output_name == "Location":
                return "vector", tuple(transform["location"])
            if output_name == "Rotation":
                return "vector", tuple(transform["rotation"])
            if output_name == "Scale":
                return "vector", tuple(transform["scale"])
        if is_primitive_node(node) and str(getattr(socket, "name", "")) == "Transform":
            return "transform", node_effective_transform(node, active_stack)

        raise RuntimeError(
            f"Node '{node.name}' does not provide compatible data for '{socket.name}'"
        )
    finally:
        active_stack.remove(node_key)


def _socket_data_value(socket, expected_kind: str, active_stack, fallback):
    if socket is None or not socket.is_linked or not socket.links:
        return fallback

    linked_kind, linked_value = _evaluate_output_socket(
        socket.links[0].from_socket, active_stack
    )
    if linked_kind == expected_kind:
        return linked_value
    if expected_kind == "vector" and linked_kind == "color":
        return tuple(float(v) for v in linked_value)
    if expected_kind == "color" and linked_kind == "vector":
        return tuple(float(v) for v in linked_value)
    raise RuntimeError(
        f"Socket '{socket.name}' expects {expected_kind} input, got {linked_kind}"
    )


def node_effective_transform(node, active_stack=None):
    stack = set() if active_stack is None else active_stack
    fallback = _transform_value(
        getattr(node, "sdf_location", (0.0, 0.0, 0.0)),
        getattr(node, "sdf_rotation", (0.0, 0.0, 0.0)),
        getattr(node, "sdf_scale", (1.0, 1.0, 1.0)),
    )
    return _socket_data_value(
        _get_node_input(node, "Transform"), "transform", stack, fallback
    )


def node_effective_color(node, active_stack=None):
    stack = set() if active_stack is None else active_stack
    fallback = tuple(float(v) for v in getattr(node, "color", (0.8, 0.8, 0.8)))
    return _socket_data_value(_get_node_input(node, "Color"), "color", stack, fallback)


def node_effective_size(node, active_stack=None):
    stack = set() if active_stack is None else active_stack
    fallback = tuple(float(v) for v in getattr(node, "size", (1.0, 1.0, 1.0)))
    return _socket_data_value(_get_node_input(node, "Size"), "vector", stack, fallback)


def node_effective_radius(node, active_stack=None):
    stack = set() if active_stack is None else active_stack
    fallback = float(getattr(node, "radius", 0.0))
    return _socket_data_value(_get_node_input(node, "Radius"), "float", stack, fallback)


def node_effective_height(node, active_stack=None):
    stack = set() if active_stack is None else active_stack
    fallback = float(getattr(node, "height", 0.0))
    return _socket_data_value(_get_node_input(node, "Height"), "float", stack, fallback)


def node_effective_bevel(node, active_stack=None):
    stack = set() if active_stack is None else active_stack
    fallback = float(getattr(node, "bevel", 0.0))
    return _socket_data_value(_get_node_input(node, "Bevel"), "float", stack, fallback)


def node_effective_blend_radius(node, active_stack=None):
    stack = set() if active_stack is None else active_stack
    fallback = float(getattr(node, "blend_radius", 0.0))
    return _socket_data_value(
        _get_node_input(node, "Blend Radius"), "float", stack, fallback
    )


def node_effective_world_to_local_matrix(node, active_stack=None):
    transform = node_effective_transform(node, active_stack)
    local_to_world = Matrix.LocRotScale(
        Vector(transform["location"]),
        Euler(transform["rotation"], "XYZ"),
        Vector(transform["scale"]),
    )
    return local_to_world.inverted_safe()


class _MathOPSV2PrimitiveNodeBase(_MathOPSV2NodeBase):
    __annotations__ = dict(_transform_annotations())
    __annotations__.update(
        {
            "proxy_token": StringProperty(default="", options={"HIDDEN"}),
            "proxy_managed": BoolProperty(default=False, options={"HIDDEN"}),
            "color": FloatVectorProperty(
                name="Color",
                size=3,
                subtype="COLOR",
                min=0.0,
                max=1.0,
                default=(0.8, 0.8, 0.8),
                update=_graph_updated,
            ),
        }
    )

    def init(self, context):
        del context
        self.inputs.new(TRANSFORM_SOCKET_IDNAME, "Transform")
        _set_socket_value_prop(self.inputs.new(COLOR_SOCKET_IDNAME, "Color"), "color")
        self.outputs.new(SOCKET_IDNAME, "SDF")
        self.outputs.new(TRANSFORM_SOCKET_IDNAME, "Transform")
        _auto_insert_primitive_node(self)

    def _draw_primitive_footer(self, layout):
        del layout

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
        _set_socket_value_prop(
            self.inputs.new(FLOAT_SOCKET_IDNAME, "Blend Radius"), "blend_radius"
        )
        self.outputs.new(SOCKET_IDNAME, "SDF")

    def _draw_csg_buttons(self, layout):
        del layout

    def draw_buttons_ext(self, context, layout):
        self.draw_buttons(context, layout)


class MathOPSV2SDFTree(NodeTree):
    bl_idname = TREE_IDNAME
    bl_label = "MathOPS-v2 SDF"
    bl_icon = "NODETREE"

    owner_scene_id: StringProperty(default="", options={"HIDDEN"})

    def update(self):
        _graph_updated(self, bpy.context)


class _MathOPSV2SocketBase(NodeSocket):
    socket_color = (0.8, 0.8, 0.8, 1.0)
    value_prop: StringProperty(default="", options={"HIDDEN"})

    def draw(self, context, layout, node, text):
        del context
        value_prop = str(getattr(self, "value_prop", "") or "")
        if (
            not self.is_output
            and not self.is_linked
            and value_prop
            and hasattr(node, value_prop)
        ):
            layout.prop(node, value_prop, text=text or self.bl_label)
            return
        layout.label(text=text or self.bl_label)

    def draw_color(self, context, node):
        del context, node
        return self.socket_color


class MathOPSV2SDFSocket(_MathOPSV2SocketBase):
    bl_idname = SOCKET_IDNAME
    bl_label = "SDF"
    socket_color = (1.0, 0.45, 0.1, 1.0)


class MathOPSV2FloatSocket(_MathOPSV2SocketBase):
    bl_idname = FLOAT_SOCKET_IDNAME
    bl_label = "Float"
    socket_color = (0.55, 0.75, 1.0, 1.0)


class MathOPSV2VectorSocket(_MathOPSV2SocketBase):
    bl_idname = VECTOR_SOCKET_IDNAME
    bl_label = "Vector"
    socket_color = (0.45, 0.85, 0.95, 1.0)


class MathOPSV2ColorSocket(_MathOPSV2SocketBase):
    bl_idname = COLOR_SOCKET_IDNAME
    bl_label = "Color"
    socket_color = (0.95, 0.55, 0.75, 1.0)


class MathOPSV2TransformSocket(_MathOPSV2SocketBase):
    bl_idname = TRANSFORM_SOCKET_IDNAME
    bl_label = "Transform"
    socket_color = (0.75, 0.65, 1.0, 1.0)


def _set_socket_value_prop(socket, prop_name: str):
    if socket is not None:
        socket.value_prop = prop_name


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


class MathOPSV2ValueNode(_MathOPSV2NodeBase, Node):
    bl_idname = VALUE_NODE_IDNAME
    bl_label = "Value"

    value: FloatProperty(name="Value", default=0.0, update=_graph_updated)

    def init(self, context):
        del context
        self.outputs.new(FLOAT_SOCKET_IDNAME, "Value")

    def draw_buttons(self, context, layout):
        del context
        layout.prop(self, "value")


class MathOPSV2VectorNode(_MathOPSV2NodeBase, Node):
    bl_idname = VECTOR_NODE_IDNAME
    bl_label = "Vector"

    value: FloatVectorProperty(
        name="Value",
        size=3,
        subtype="XYZ",
        default=(0.0, 0.0, 0.0),
        update=_graph_updated,
    )

    def init(self, context):
        del context
        self.outputs.new(VECTOR_SOCKET_IDNAME, "Vector")

    def draw_buttons(self, context, layout):
        del context
        layout.prop(self, "value")


class MathOPSV2ColorNode(_MathOPSV2NodeBase, Node):
    bl_idname = COLOR_NODE_IDNAME
    bl_label = "Color"

    value: FloatVectorProperty(
        name="Value",
        size=3,
        subtype="COLOR",
        min=0.0,
        max=1.0,
        default=(0.8, 0.8, 0.8),
        update=_graph_updated,
    )

    def init(self, context):
        del context
        self.outputs.new(COLOR_SOCKET_IDNAME, "Color")

    def draw_buttons(self, context, layout):
        del context
        layout.prop(self, "value")


class MathOPSV2TransformNode(_MathOPSV2NodeBase, Node):
    bl_idname = TRANSFORM_NODE_IDNAME
    bl_label = "Make Transform"

    location_value: FloatVectorProperty(
        name="Location",
        size=3,
        subtype="TRANSLATION",
        default=(0.0, 0.0, 0.0),
        update=_graph_updated,
    )
    rotation_value: FloatVectorProperty(
        name="Rotation",
        size=3,
        subtype="EULER",
        default=(0.0, 0.0, 0.0),
        update=_graph_updated,
    )
    scale_value: FloatVectorProperty(
        name="Scale",
        size=3,
        subtype="XYZ",
        default=(1.0, 1.0, 1.0),
        min=0.001,
        soft_min=0.001,
        update=_graph_updated,
    )

    def init(self, context):
        del context
        _set_socket_value_prop(
            self.inputs.new(VECTOR_SOCKET_IDNAME, "Location"), "location_value"
        )
        _set_socket_value_prop(
            self.inputs.new(VECTOR_SOCKET_IDNAME, "Rotation"), "rotation_value"
        )
        _set_socket_value_prop(
            self.inputs.new(VECTOR_SOCKET_IDNAME, "Scale"), "scale_value"
        )
        self.outputs.new(TRANSFORM_SOCKET_IDNAME, "Transform")

    def draw_buttons(self, context, layout):
        del context, layout
        pass


class MathOPSV2BreakTransformNode(_MathOPSV2NodeBase, Node):
    bl_idname = BREAK_TRANSFORM_NODE_IDNAME
    bl_label = "Break Transform"

    def init(self, context):
        del context
        self.inputs.new(TRANSFORM_SOCKET_IDNAME, "Transform")
        self.outputs.new(VECTOR_SOCKET_IDNAME, "Location")
        self.outputs.new(VECTOR_SOCKET_IDNAME, "Rotation")
        self.outputs.new(VECTOR_SOCKET_IDNAME, "Scale")

    def draw_buttons(self, context, layout):
        del context, layout
        pass


class MathOPSV2SDFSphereNode(_MathOPSV2PrimitiveNodeBase, Node):
    bl_idname = SPHERE_NODE_IDNAME
    bl_label = "SDF Sphere"

    def init(self, context):
        _MathOPSV2PrimitiveNodeBase.init(self, context)
        _set_socket_value_prop(self.inputs.new(FLOAT_SOCKET_IDNAME, "Radius"), "radius")

    radius: FloatProperty(
        name="Radius",
        default=0.5,
        min=0.0,
        soft_min=0.0,
        update=_graph_updated,
    )

    def draw_buttons(self, context, layout):
        del context
        self._draw_primitive_footer(layout)


class MathOPSV2SDFBoxNode(_MathOPSV2PrimitiveNodeBase, Node):
    bl_idname = BOX_NODE_IDNAME
    bl_label = "SDF Box"

    def init(self, context):
        _MathOPSV2PrimitiveNodeBase.init(self, context)
        _set_socket_value_prop(self.inputs.new(VECTOR_SOCKET_IDNAME, "Size"), "size")
        _set_socket_value_prop(self.inputs.new(FLOAT_SOCKET_IDNAME, "Bevel"), "bevel")

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
        self._draw_primitive_footer(layout)


class MathOPSV2SDFCylinderNode(_MathOPSV2PrimitiveNodeBase, Node):
    bl_idname = CYLINDER_NODE_IDNAME
    bl_label = "SDF Cylinder"

    def init(self, context):
        _MathOPSV2PrimitiveNodeBase.init(self, context)
        _set_socket_value_prop(self.inputs.new(FLOAT_SOCKET_IDNAME, "Radius"), "radius")
        _set_socket_value_prop(self.inputs.new(FLOAT_SOCKET_IDNAME, "Height"), "height")

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
        self._draw_primitive_footer(layout)


class MathOPSV2SDFConeNode(_MathOPSV2PrimitiveNodeBase, Node):
    bl_idname = CONE_NODE_IDNAME
    bl_label = "SDF Cone"

    def init(self, context):
        _MathOPSV2PrimitiveNodeBase.init(self, context)
        _set_socket_value_prop(self.inputs.new(FLOAT_SOCKET_IDNAME, "Radius"), "radius")
        _set_socket_value_prop(self.inputs.new(FLOAT_SOCKET_IDNAME, "Height"), "height")

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
        self._draw_primitive_footer(layout)


class MathOPSV2CSGUnionNode(_MathOPSV2CSGNodeBase, Node):
    bl_idname = UNION_NODE_IDNAME
    bl_label = "CSG Union"
    blend_mode = "union"

    def draw_buttons(self, context, layout):
        del context
        self._draw_csg_buttons(layout)


class MathOPSV2CSGSubtractNode(_MathOPSV2CSGNodeBase, Node):
    bl_idname = SUBTRACT_NODE_IDNAME
    bl_label = "CSG Subtract"
    blend_mode = "sub"

    def draw_buttons(self, context, layout):
        del context
        self._draw_csg_buttons(layout)


class MathOPSV2CSGIntersectNode(_MathOPSV2CSGNodeBase, Node):
    bl_idname = INTERSECT_NODE_IDNAME
    bl_label = "CSG Intersect"
    blend_mode = "inter"

    def draw_buttons(self, context, layout):
        del context
        self._draw_csg_buttons(layout)


class MathOPSV2CSGNode(_MathOPSV2CSGNodeBase, Node):
    bl_idname = CSG_NODE_IDNAME
    bl_label = "CSG"

    blend_mode: EnumProperty(
        name="Operation",
        items=_CSG_BLEND_MODE_ITEMS,
        default="union",
        update=_graph_updated,
    )

    def draw_buttons(self, context, layout):
        del context
        layout.prop(self, "blend_mode")
        self._draw_csg_buttons(layout)


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


def _primitive_payload_bounds(node_payload):
    node_matrix = _json_matrix_to_world_to_local(
        node_payload.get("matrix", list(_IDENTITY_MATRIX_3X4))
    )
    try:
        local_to_world = node_matrix.inverted()
    except Exception:
        return None
    world_points = [
        local_to_world @ corner for corner in _primitive_local_corners(node_payload)
    ]
    return (
        tuple(min(point[i] for point in world_points) for i in range(3)),
        tuple(max(point[i] for point in world_points) for i in range(3)),
    )


def _record_compile_state(record):
    return {
        "type": str(record.primitive_type or "sphere").strip().lower(),
        "location": tuple(round(float(v), 6) for v in record.location),
        "rotation": tuple(round(float(v), 6) for v in record.rotation),
        "scale": tuple(round(float(v), 6) for v in record.scale),
        "color": tuple(round(float(v), 6) for v in record.color),
        "size": tuple(round(float(v), 6) for v in record.size),
        "radius": round(float(record.radius), 6),
        "height": round(float(record.height), 6),
        "bevel": round(float(record.bevel), 6),
    }


def _is_exact_union_payload(node_payload):
    return (
        isinstance(node_payload, dict)
        and node_payload.get("nodeType") == "binaryOperator"
        and node_payload.get("blendMode") == "union"
        and abs(float(node_payload.get("blendRadius", 0.0))) <= 1.0e-9
    )


def _bounds_center(bounds):
    if bounds is None:
        return (0.0, 0.0, 0.0)
    return tuple((float(a) + float(b)) * 0.5 for a, b in zip(bounds[0], bounds[1]))


def _balanced_union_payload_sorted(items):
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        left_payload, left_bounds = items[0]
        right_payload, right_bounds = items[1]
        return (
            {
                "nodeType": "binaryOperator",
                "leftChild": left_payload,
                "rightChild": right_payload,
                "blendMode": "union",
                "blendRadius": 0.0,
                "matrix": list(_IDENTITY_MATRIX_3X4),
            },
            _union_bounds(left_bounds, right_bounds),
        )

    midpoint = len(items) // 2
    left_payload, left_bounds = _balanced_union_payload_sorted(items[:midpoint])
    right_payload, right_bounds = _balanced_union_payload_sorted(items[midpoint:])
    return (
        {
            "nodeType": "binaryOperator",
            "leftChild": left_payload,
            "rightChild": right_payload,
            "blendMode": "union",
            "blendRadius": 0.0,
            "matrix": list(_IDENTITY_MATRIX_3X4),
        },
        _union_bounds(left_bounds, right_bounds),
    )


def _balanced_union_payload(items):
    if len(items) == 1:
        return items[0]

    combined_bounds = None
    for _payload, bounds in items:
        combined_bounds = _union_bounds(combined_bounds, bounds)

    axis = 0
    if combined_bounds is not None:
        extents = [
            float(combined_bounds[1][index]) - float(combined_bounds[0][index])
            for index in range(3)
        ]
        axis = max(range(3), key=lambda index: extents[index])

    ordered = sorted(items, key=lambda item: _bounds_center(item[1])[axis])
    return _balanced_union_payload_sorted(ordered)


def _optimize_payload(node_payload):
    if not isinstance(node_payload, dict):
        return node_payload, None, []

    node_type = node_payload.get("nodeType")
    if node_type == "primitive":
        bounds = _primitive_payload_bounds(node_payload)
        return node_payload, bounds, [(node_payload, bounds)]

    if node_type != "binaryOperator":
        return node_payload, None, [(node_payload, None)]

    left_payload, left_bounds, left_terms = _optimize_payload(
        node_payload.get("leftChild")
    )
    right_payload, right_bounds, right_terms = _optimize_payload(
        node_payload.get("rightChild")
    )
    blend_mode = node_payload.get("blendMode")
    blend_radius = max(0.0, float(node_payload.get("blendRadius", 0.0)))

    if blend_mode == "union" and blend_radius <= 1.0e-9:
        union_terms = left_terms + right_terms
        optimized_payload, bounds = _balanced_union_payload(union_terms)
        return optimized_payload, bounds, union_terms

    optimized_payload = {
        "nodeType": "binaryOperator",
        "leftChild": left_payload,
        "rightChild": right_payload,
        "blendMode": blend_mode,
        "blendRadius": blend_radius,
        "matrix": list(_IDENTITY_MATRIX_3X4),
    }
    bounds = _union_bounds(left_bounds, right_bounds)
    if blend_mode == "union" and blend_radius > 0.0:
        bounds = _inflate_bounds(bounds, blend_radius * 0.25)
    return optimized_payload, bounds, [(optimized_payload, bounds)]


def _socket_source_node(socket, label, node):
    if not socket.is_linked or not socket.links:
        raise RuntimeError(f"Node '{node.name}' is missing its {label} input")
    return socket.links[0].from_node


def _serialize_primitive(node, primitive_type):
    active_stack = set()
    color = node_effective_color(node, active_stack)
    payload = {
        "nodeType": "primitive",
        "nodeId": primitive_node_token(node),
        "primitiveType": primitive_type,
        "color": [float(v) for v in color],
        "round_x": 0.0,
        "round_y": 0.0,
        "matrix": _matrix_to_json_values(
            node_effective_world_to_local_matrix(node, active_stack)
        ),
    }
    if primitive_type == "sphere":
        payload["radius"] = float(node_effective_radius(node, active_stack))
    elif primitive_type == "box":
        bevel = max(0.0, float(node_effective_bevel(node, active_stack)))
        payload["sides"] = [float(v) for v in node_effective_size(node, active_stack)]
        payload["bevel"] = [bevel, bevel, bevel, bevel]
    else:
        payload["radius"] = float(node_effective_radius(node, active_stack))
        payload["height"] = float(node_effective_height(node, active_stack))
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
            CSG_NODE_IDNAME,
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
                "blendRadius": float(node_effective_blend_radius(node, active_stack)),
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


def primitive_type_from_node(node) -> str | None:
    return {
        SPHERE_NODE_IDNAME: "sphere",
        BOX_NODE_IDNAME: "box",
        CYLINDER_NODE_IDNAME: "cylinder",
        CONE_NODE_IDNAME: "cone",
    }.get(getattr(node, "bl_idname", ""))


def create_primitive_node(tree, primitive_type):
    node_idname = {
        "sphere": SPHERE_NODE_IDNAME,
        "box": BOX_NODE_IDNAME,
        "cylinder": CYLINDER_NODE_IDNAME,
        "cone": CONE_NODE_IDNAME,
    }.get(primitive_type)
    if node_idname is None:
        raise RuntimeError(f"Unsupported primitive type '{primitive_type}'")
    with suppress_auto_insert():
        node = tree.nodes.new(node_idname)
    primitive_node_token(node)
    return node


def insert_primitive_above_output(tree, primitive_node):
    output_node = _ensure_output_node(tree)
    surface_socket = output_node.inputs[0]
    current_root = None
    if surface_socket.is_linked and surface_socket.links:
        current_root = surface_socket.links[0].from_node
        tree.links.remove(surface_socket.links[0])
        if current_root == primitive_node:
            current_root = None

    if current_root is None:
        primitive_node.location = (0.0, 0.0)
        tree.links.new(primitive_node.outputs[0], surface_socket)
        _position_output_node(tree, output_node, primitive_node)
        _graph_updated(tree, bpy.context)
        return primitive_node

    csg_node = tree.nodes.new(CSG_NODE_IDNAME)
    csg_node.blend_mode = "union"
    csg_node.location = (current_root.location.x + 260.0, current_root.location.y)
    primitive_node.location = (current_root.location.x, current_root.location.y - 220.0)
    tree.links.new(current_root.outputs[0], csg_node.inputs[0])
    tree.links.new(primitive_node.outputs[0], csg_node.inputs[1])
    tree.links.new(csg_node.outputs[0], surface_socket)
    _position_output_node(tree, output_node, csg_node)
    _graph_updated(tree, bpy.context)
    return csg_node


def remove_primitive_node(tree, primitive_node):
    if primitive_node is None or primitive_node.id_data != tree:
        return False

    output_node = _find_output_node(tree)
    if (
        output_node is not None
        and output_node.inputs
        and output_node.inputs[0].is_linked
    ):
        root_link = output_node.inputs[0].links[0]
        if root_link.from_node == primitive_node:
            tree.links.remove(root_link)
            tree.nodes.remove(primitive_node)
            _graph_updated(tree, bpy.context)
            return True

    parent_link = None
    for link in list(primitive_node.outputs[0].links):
        if getattr(link.to_node, "bl_idname", "") in {
            CSG_NODE_IDNAME,
        }:
            parent_link = link
            break

    if parent_link is None:
        tree.nodes.remove(primitive_node)
        _graph_updated(tree, bpy.context)
        return True

    parent_node = parent_link.to_node
    input_index = 0 if parent_link.to_socket == parent_node.inputs[0] else 1
    sibling_socket = parent_node.inputs[1 - input_index]
    sibling_node = None
    if sibling_socket.is_linked and sibling_socket.links:
        sibling_node = sibling_socket.links[0].from_node

    consumer_sockets = [link.to_socket for link in list(parent_node.outputs[0].links)]
    for link in list(parent_node.outputs[0].links):
        tree.links.remove(link)

    tree.nodes.remove(parent_node)
    tree.nodes.remove(primitive_node)

    if sibling_node is not None:
        sibling_output = _surface_output_socket(sibling_node)
        for socket in consumer_sockets:
            if sibling_output is not None:
                tree.links.new(sibling_output, socket)

    _graph_updated(tree, bpy.context)
    return True


def compile_tree_payload(tree):
    output_node = _root_output_node(tree)
    root_node = _socket_source_node(output_node.inputs[0], "Surface", output_node)
    payload = _serialize_node(root_node, set())
    payload, bounds, _union_terms = _optimize_payload(payload)
    if bounds is None:
        bounds = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    compiled = dict(payload)
    compiled["aabb_min"] = [float(v) for v in bounds[0]]
    compiled["aabb_max"] = [float(v) for v in bounds[1]]
    return compiled


def _strip_runtime_payload_metadata(node_payload):
    if not isinstance(node_payload, dict):
        return node_payload
    cleaned = {
        key: _strip_runtime_payload_metadata(value)
        for key, value in node_payload.items()
        if key != "nodeId"
    }
    return cleaned


def _update_payload_content_hash(digest, value):
    if isinstance(value, dict):
        digest.update(b"{")
        for key in sorted(value):
            if key == "nodeId":
                continue
            digest.update(key.encode("utf-8"))
            digest.update(b"=")
            _update_payload_content_hash(digest, value[key])
            digest.update(b";")
        digest.update(b"}")
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"[")
        for item in value:
            _update_payload_content_hash(digest, item)
            digest.update(b",")
        digest.update(b"]")
        return
    if isinstance(value, bool):
        digest.update(b"T" if value else b"F")
        return
    if isinstance(value, int):
        digest.update(b"i")
        digest.update(str(int(value)).encode("ascii"))
        return
    if isinstance(value, float):
        digest.update(b"f")
        digest.update(struct.pack("<d", float(value)))
        return
    if value is None:
        digest.update(b"N")
        return
    digest.update(b"s")
    digest.update(str(value).encode("utf-8"))


def _payload_content_hash(node_payload) -> str:
    digest = hashlib.sha1()
    _update_payload_content_hash(digest, node_payload)
    return digest.hexdigest()


def _update_payload_topology_hash(digest, node_payload):
    if not isinstance(node_payload, dict):
        digest.update(b"?")
        return

    node_type = str(node_payload.get("nodeType", ""))
    if node_type == "primitive":
        digest.update(b"P")
        return

    if node_type == "binaryOperator":
        digest.update(b"B")
        digest.update(str(node_payload.get("blendMode", "")).encode("utf-8"))
        digest.update(
            struct.pack("<d", round(float(node_payload.get("blendRadius", 0.0)), 6))
        )
        _update_payload_topology_hash(digest, node_payload.get("leftChild"))
        _update_payload_topology_hash(digest, node_payload.get("rightChild"))
        return

    digest.update(node_type.encode("utf-8"))


def _payload_topology_hash(node_payload) -> str:
    digest = hashlib.sha1()
    _update_payload_topology_hash(digest, node_payload)
    return digest.hexdigest()


def compile_tree_json(tree):
    return json.dumps(
        _strip_runtime_payload_metadata(compile_tree_payload(tree)),
        ensure_ascii=True,
        separators=(",", ":"),
    )


def generated_scene_dir() -> Path:
    path = Path(__file__).resolve().parent / ".generated_scenes"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_scene_cache_json(scene_cache) -> str:
    scene_hash = scene_cache.get("hash")
    scene_json = scene_cache.get("json")
    if scene_json is not None and scene_cache.get("json_hash") == scene_hash:
        return scene_json

    scene_json = json.dumps(
        _strip_runtime_payload_metadata(scene_cache["payload"]),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    scene_cache["json"] = scene_json
    scene_cache["json_hash"] = scene_hash
    return scene_json


def invalid_scene_path() -> Path:
    path = generated_scene_dir() / _INVALID_SCENE_NAME
    if not path.is_file():
        path.write_text(
            json.dumps(
                {
                    "nodeType": "primitive",
                    "primitiveType": "sphere",
                    "color": [0.0, 0.0, 0.0],
                    "round_x": 0.0,
                    "round_y": 0.0,
                    "matrix": [
                        1.0,
                        0.0,
                        0.0,
                        -1000000.0,
                        0.0,
                        1.0,
                        0.0,
                        -1000000.0,
                        0.0,
                        0.0,
                        1.0,
                        -1000000.0,
                    ],
                    "radius": 0.0,
                    "aabb_min": [-1.0, -1.0, -1.0],
                    "aabb_max": [1.0, 1.0, 1.0],
                },
                ensure_ascii=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
    return path


def _scene_file_name(tree_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", tree_name).strip("._")
    return f"{safe_name or 'scene'}.json"


def _blender_to_renderer_matrix(matrix):
    return _BLENDER_TO_RENDERER_BASIS @ matrix @ _RENDERER_TO_BLENDER_BASIS


def _scene_store_items(scene, cached_item_cache=None):
    records = getattr(scene, "mathops_v2_primitives", ())
    if len(records) == 0:
        return None, None
    items = []
    item_cache = {}
    for record in records:
        record_id = str(record.primitive_id)
        record_state = _record_compile_state(record)
        primitive_type = record_state["type"]
        if primitive_type not in {"sphere", "box", "cylinder", "cone"}:
            return None, None
        cached_entry = (
            None if cached_item_cache is None else cached_item_cache.get(record_id)
        )
        if cached_entry is not None and cached_entry.get("state") == record_state:
            payload = cached_entry["payload"]
            bounds = cached_entry["bounds"]
        else:
            blender_matrix = Matrix.LocRotScale(
                Vector(record.location),
                Euler(record.rotation, "XYZ"),
                Vector(record.scale),
            )
            renderer_matrix = _blender_to_renderer_matrix(blender_matrix)
            world_to_local = renderer_matrix.inverted_safe()
            payload = {
                "nodeType": "primitive",
                "nodeId": record_id,
                "primitiveType": primitive_type,
                "color": [float(v) for v in record.color],
                "round_x": 0.0,
                "round_y": 0.0,
                "matrix": _matrix_to_json_values(world_to_local),
            }
            if primitive_type == "sphere":
                payload["radius"] = float(record.radius)
            elif primitive_type == "box":
                bevel = max(0.0, float(record.bevel))
                payload["sides"] = [float(v) for v in record.size]
                payload["bevel"] = [bevel, bevel, bevel, bevel]
            else:
                payload["radius"] = float(record.radius)
                payload["height"] = float(record.height)
            bounds = _primitive_payload_bounds(payload)
        items.append((record_id, payload, bounds))
        item_cache[record_id] = {
            "state": record_state,
            "payload": payload,
            "bounds": bounds,
        }
    return items, item_cache


def _proxy_scene_fast_path_supported(scene, tree):
    records = getattr(scene, "mathops_v2_primitives", ())
    if len(records) == 0:
        return False
    if not proxy_tree_materialized(tree) and not _is_pure_proxy_union_tree(tree):
        return False

    record_ids = {str(getattr(record, "primitive_id", "") or "") for record in records}
    primitive_count = 0
    for node in tree.nodes:
        node_idname = getattr(node, "bl_idname", "")
        if node_idname == OUTPUT_NODE_IDNAME:
            continue
        if is_primitive_node(node):
            primitive_count += 1
            if not getattr(node, "proxy_managed", False):
                return False
            if primitive_node_token(node) not in record_ids:
                return False
            continue
        if _is_proxy_union_node(node):
            continue
        return False

    return primitive_count == len(record_ids)


def _proxy_scene_items(scene, tree, cached_item_cache=None):
    if _proxy_scene_fast_path_supported(scene, tree):
        store_items, item_cache = _scene_store_items(scene, cached_item_cache)
        if store_items is not None:
            return store_items, item_cache

    proxies = [
        obj for obj in scene.objects if getattr(obj, "mathops_v2_sdf_proxy", False)
    ]
    if not proxies:
        return None, None

    def build_items():
        items = []
        for obj in proxies:
            proxy_id = str(
                getattr(obj, "mathops_v2_sdf_node_id", "")
                or getattr(obj, "mathops_v2_sdf_proxy_id", "")
            )
            primitive_type = (
                str(obj.get("sdf_type", "sphere") or "sphere").strip().lower()
            )
            if primitive_type not in {"sphere", "box", "cylinder", "cone"}:
                return None
            color = obj.get("sdf_color", (0.8, 0.8, 0.8))
            renderer_matrix = _blender_to_renderer_matrix(obj.matrix_world)
            world_to_local = renderer_matrix.inverted_safe()
            payload = {
                "nodeType": "primitive",
                "nodeId": proxy_id,
                "primitiveType": primitive_type,
                "color": [float(color[0]), float(color[1]), float(color[2])],
                "round_x": 0.0,
                "round_y": 0.0,
                "matrix": _matrix_to_json_values(world_to_local),
            }
            if primitive_type == "sphere":
                payload["radius"] = float(obj.get("sdf_radius", 0.5))
            elif primitive_type == "box":
                bevel = max(0.0, float(obj.get("sdf_bevel", 0.0)))
                sides = obj.get("sdf_size", (1.0, 1.0, 1.0))
                payload["sides"] = [float(sides[0]), float(sides[1]), float(sides[2])]
                payload["bevel"] = [bevel, bevel, bevel, bevel]
            else:
                payload["radius"] = float(obj.get("sdf_radius", 0.35))
                payload["height"] = float(obj.get("sdf_height", 1.0))
            items.append((proxy_id, payload, _primitive_payload_bounds(payload)))
        return items

    if not proxy_tree_materialized(tree) and not _is_pure_proxy_union_tree(tree):
        return None, None

    proxy_node_ids = {
        str(getattr(obj, "mathops_v2_sdf_node_id", "") or "") for obj in proxies
    }
    primitive_count = 0
    for node in tree.nodes:
        node_idname = getattr(node, "bl_idname", "")
        if node_idname == OUTPUT_NODE_IDNAME:
            continue
        if is_primitive_node(node):
            primitive_count += 1
            if not getattr(node, "proxy_managed", False):
                return None, None
            if primitive_node_token(node) not in proxy_node_ids:
                return None, None
            continue
        if _is_proxy_union_node(node):
            continue
        return None, None

    if primitive_count != len(proxies):
        return None, None

    return build_items(), None


def _compile_proxy_csg_union_payload(scene, tree, cached=None):
    if not _is_pure_proxy_csg_union_tree(tree):
        return None

    cached_item_cache = None if cached is None else cached.get("item_cache")
    items, item_cache = _scene_store_items(scene, cached_item_cache)
    if not items:
        return None
    item_map = {proxy_id: (payload, bounds) for proxy_id, payload, bounds in items}

    output_node = _find_output_node(tree)
    if (
        output_node is None
        or not output_node.inputs
        or not output_node.inputs[0].is_linked
    ):
        return None

    def build_payload(node):
        if is_primitive_node(node):
            return item_map.get(primitive_node_token(node))
        if not _is_proxy_csg_union_node(node):
            return None
        left_node = _socket_source_node(node.inputs[0], "A", node)
        right_node = _socket_source_node(node.inputs[1], "B", node)
        left_item = build_payload(left_node)
        right_item = build_payload(right_node)
        if left_item is None or right_item is None:
            return None
        left_payload, left_bounds = left_item
        right_payload, right_bounds = right_item
        blend_radius = max(0.0, float(getattr(node, "blend_radius", 0.0)))
        payload = {
            "nodeType": "binaryOperator",
            "leftChild": left_payload,
            "rightChild": right_payload,
            "blendMode": "union",
            "blendRadius": blend_radius,
            "matrix": list(_IDENTITY_MATRIX_3X4),
        }
        bounds = _union_bounds(left_bounds, right_bounds)
        if blend_radius > 0.0:
            bounds = _inflate_bounds(bounds, blend_radius * 0.25)
        return payload, bounds

    root_node = _socket_source_node(output_node.inputs[0], "SDF", output_node)
    built = build_payload(root_node)
    if built is None:
        return None
    payload, bounds = built
    compiled = dict(payload)
    compiled["aabb_min"] = [float(v) for v in bounds[0]]
    compiled["aabb_max"] = [float(v) for v in bounds[1]]

    topology_tokens = []
    for node in tree.nodes:
        node_idname = getattr(node, "bl_idname", "")
        if node_idname == OUTPUT_NODE_IDNAME:
            continue
        if is_primitive_node(node):
            topology_tokens.append(f"P:{primitive_node_token(node)}")
        elif _is_proxy_csg_union_node(node):
            topology_tokens.append(
                f"U:{getattr(node, 'name', '')}:{round(float(getattr(node, 'blend_radius', 0.0)), 6)}"
            )
        else:
            return None
    topology_hash = hashlib.sha1("\n".join(topology_tokens).encode("utf-8")).hexdigest()
    return {
        "payload": compiled,
        "proxy_order": None,
        "topology_hash": topology_hash,
        "item_cache": item_cache,
        "node_count": _count_payload_nodes(compiled),
    }


def _compile_proxy_scene_payload(scene, tree, cached=None):
    cached_item_cache = None if cached is None else cached.get("item_cache")
    items, item_cache = _proxy_scene_items(scene, tree, cached_item_cache)
    if not items:
        return None
    item_map = {proxy_id: (payload, bounds) for proxy_id, payload, bounds in items}
    proxy_ids = set(item_map)

    ordered_ids = None
    cached_order = None if cached is None else cached.get("proxy_order")
    if isinstance(cached_order, list) and len(cached_order) == len(item_map):
        if set(cached_order) == proxy_ids:
            ordered_ids = list(cached_order)

    if ordered_ids is None:
        combined_bounds = None
        for _proxy_id, _payload, bounds in items:
            combined_bounds = _union_bounds(combined_bounds, bounds)
        axis = 0
        if combined_bounds is not None:
            extents = [
                float(combined_bounds[1][index]) - float(combined_bounds[0][index])
                for index in range(3)
            ]
            axis = max(range(3), key=lambda index: extents[index])
        ordered_ids = [
            proxy_id
            for proxy_id, _payload, _bounds in sorted(
                items, key=lambda item: _bounds_center(item[2])[axis]
            )
        ]

    ordered = [item_map[proxy_id] for proxy_id in ordered_ids]
    payload, bounds = _balanced_union_payload_sorted(ordered)
    compiled = dict(payload)
    compiled["aabb_min"] = [float(v) for v in bounds[0]]
    compiled["aabb_max"] = [float(v) for v in bounds[1]]
    topology_hash = hashlib.sha1("\n".join(ordered_ids).encode("utf-8")).hexdigest()
    return {
        "payload": compiled,
        "proxy_order": ordered_ids,
        "topology_hash": topology_hash,
        "item_cache": item_cache,
        "node_count": None
        if cached is None or ordered_ids != cached_order
        else cached.get("metadata", {}).get("node_count"),
    }


def ensure_compiled_scene_cache(settings, create=False):
    start = time.perf_counter()
    tree = get_selected_tree(settings, create=create, ensure=False)
    scene_path = generated_scene_dir() / _scene_file_name(tree.name)
    scene_path_key = str(scene_path.resolve())
    cache_key = tree.name_full
    cached = runtime.generated_scene_cache.get(cache_key)

    if cache_key in runtime.scene_transform_dirty:
        runtime.scene_transform_dirty.discard(cache_key)
        runtime.generated_scene_dirty.discard(cache_key)
        if cached is not None:
            scene = _owner_scene(settings)
            proxy_compiled = None
            if scene is not None:
                proxy_compiled = _compile_proxy_scene_payload(scene, tree, cached)
                if proxy_compiled is None:
                    proxy_compiled = _compile_proxy_csg_union_payload(
                        scene, tree, cached
                    )
            if proxy_compiled is not None:
                payload = proxy_compiled["payload"]
                topology_hash = proxy_compiled["topology_hash"]
                proxy_order = proxy_compiled["proxy_order"]
                static_hash = topology_hash or _payload_topology_hash(payload)
                metadata = {
                    "aabb_min": tuple(
                        float(v) for v in payload.get("aabb_min", (-1.0, -1.0, -1.0))
                    ),
                    "aabb_max": tuple(
                        float(v) for v in payload.get("aabb_max", (1.0, 1.0, 1.0))
                    ),
                    "node_count": (
                        proxy_compiled.get("node_count")
                        if proxy_compiled.get("node_count") is not None
                        else _count_payload_nodes(payload)
                    ),
                }
                scene_hash = cached.get("hash", "") + f"t{time.perf_counter()}"
                cached = {
                    "hash": scene_hash,
                    "path": str(scene_path),
                    "json": None,
                    "json_hash": None,
                    "payload": payload,
                    "metadata": metadata,
                    "tree_name": tree.name_full,
                    "topology_hash": static_hash,
                    "proxy_order": proxy_order,
                    "item_cache": proxy_compiled.get("item_cache"),
                }
                runtime.generated_scene_cache[cache_key] = cached
                runtime.generated_scene_last_compile[cache_key] = time.perf_counter()
                runtime.generated_scene_path_hashes[scene_path_key] = scene_hash
                runtime.debug_slow(
                    f"Transform-fast tree={tree.name_full}",
                    (time.perf_counter() - start) * 1000.0,
                )
                return cached
        else:
            runtime.generated_scene_dirty.add(cache_key)

    if cached and cache_key not in runtime.generated_scene_dirty:
        runtime.generated_scene_path_hashes[scene_path_key] = cached["hash"]
        return cached

    now = time.perf_counter()
    interaction_active = (
        now - runtime.graph_interaction_time
    ) <= _LIVE_GRAPH_COMPILE_GRACE
    if cached and cache_key in runtime.generated_scene_dirty and interaction_active:
        last_compile = float(runtime.generated_scene_last_compile.get(cache_key, 0.0))
        if (now - last_compile) < _LIVE_GRAPH_COMPILE_INTERVAL:
            runtime.generated_scene_path_hashes[scene_path_key] = cached["hash"]
            return cached

    scene = _owner_scene(settings)
    if scene is not None and len(getattr(scene, "mathops_v2_primitives", ())) == 0:
        _bootstrap_proxy_union_tree(tree)
        try:
            from . import sdf_proxies

            if _is_pure_proxy_union_tree(tree):
                sdf_proxies.sync_tree_to_scene_records(scene, tree)
            else:
                sdf_proxies._migrate_legacy_proxies(scene)
        except Exception:
            pass
    proxy_compiled = None
    compile_mode = "tree"
    if scene is not None:
        proxy_compiled = _compile_proxy_scene_payload(scene, tree, cached)
        if proxy_compiled is None:
            proxy_compiled = _compile_proxy_csg_union_payload(scene, tree, cached)
    if proxy_compiled is not None:
        payload = proxy_compiled["payload"]
        topology_hash = proxy_compiled["topology_hash"]
        proxy_order = proxy_compiled["proxy_order"]
        compile_mode = "proxy-fast"
    else:
        payload = compile_tree_payload(tree)
        topology_hash = None
        proxy_order = None
    hash_start = time.perf_counter()
    scene_hash = _payload_content_hash(payload)
    hash_ms = (time.perf_counter() - hash_start) * 1000.0
    static_hash = topology_hash or _payload_topology_hash(payload)
    metadata = {
        "aabb_min": tuple(
            float(v) for v in payload.get("aabb_min", (-1.0, -1.0, -1.0))
        ),
        "aabb_max": tuple(float(v) for v in payload.get("aabb_max", (1.0, 1.0, 1.0))),
        "node_count": (
            proxy_compiled.get("node_count")
            if proxy_compiled is not None
            and proxy_compiled.get("node_count") is not None
            else _count_payload_nodes(payload)
        ),
    }
    scene_json = None
    json_hash = None
    file_hash = None
    if cached is not None and cached.get("hash") == scene_hash:
        scene_json = cached.get("json")
        json_hash = cached.get("json_hash")
        file_hash = cached.get("file_hash")
    cached = {
        "hash": scene_hash,
        "path": str(scene_path),
        "json": scene_json,
        "json_hash": json_hash,
        "payload": payload,
        "metadata": metadata,
        "tree_name": tree.name_full,
        "topology_hash": static_hash,
        "proxy_order": proxy_order,
        "item_cache": None
        if proxy_compiled is None
        else proxy_compiled.get("item_cache"),
    }
    if file_hash == scene_hash:
        cached["file_hash"] = file_hash
    runtime.generated_scene_cache[cache_key] = cached
    runtime.generated_scene_last_compile[cache_key] = time.perf_counter()
    runtime.generated_scene_path_hashes[scene_path_key] = scene_hash
    runtime.generated_scene_dirty.discard(cache_key)
    runtime.debug_slow(
        (
            f"Scene cache compile tree={tree.name_full} mode={compile_mode} "
            f"nodes={metadata['node_count']} hash={hash_ms:.2f}"
        ),
        (time.perf_counter() - start) * 1000.0,
    )
    return cached


def ensure_generated_scene(settings, create=False):
    cached = ensure_compiled_scene_cache(settings, create=create)
    scene_path = Path(cached["path"])
    if cached.get("file_hash") != cached["hash"] or not scene_path.is_file():
        scene_path.write_text(ensure_scene_cache_json(cached), encoding="utf-8")
        cached["file_hash"] = cached["hash"]
        runtime.debug_log(
            f"Compiled SDF node tree '{cached['tree_name']}' to {scene_path.name}"
        )
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


class MATHOPS_V2_OT_view_sdf_node(Operator):
    bl_idname = "mathops_v2.view_sdf_node"
    bl_label = "View Linked Node"
    bl_description = "Focus and frame the linked SDF node in the graph"
    bl_options = {"REGISTER"}

    primitive_id: StringProperty(name="Primitive ID", default="", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        settings = getattr(getattr(context, "scene", None), "mathops_v2_settings", None)
        return settings is not None and getattr(settings, "use_sdf_nodes", False)

    def execute(self, context):
        tree, node, focused = focus_scene_node(
            context,
            str(self.primitive_id or ""),
            create=True,
            frame=True,
        )
        del tree
        if node is None:
            self.report({"WARNING"}, "Linked SDF node not found")
            return {"CANCELLED"}
        if not focused:
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

        tree = _active_editor_tree(context, create=True)
        if tree is None:
            layout.label(text="Scene SDF tree unavailable", icon="ERROR")
            return

        row = layout.row(align=True)
        row.label(text=tree.name, icon="NODETREE")
        row.operator(MATHOPS_V2_OT_new_scene_sdf_tree.bl_idname, text="", icon="ADD")
        layout.operator(MATHOPS_V2_OT_edit_scene_sdf_tree.bl_idname, icon="NODETREE")
        layout.separator()
        try:
            if proxy_tree_materialized(tree):
                payload = compile_tree_payload(tree)
                node_count = _count_payload_nodes(payload)
                bounds_min = payload["aabb_min"]
                bounds_max = payload["aabb_max"]
            else:
                cached = ensure_compiled_scene_cache(settings, create=False)
                node_count = int(cached["metadata"]["node_count"])
                bounds_min = cached["metadata"]["aabb_min"]
                bounds_max = cached["metadata"]["aabb_max"]
            layout.label(text=f"Nodes: {node_count}")
            layout.label(
                text=(
                    f"Bounds: ({bounds_min[0]:.2f}, {bounds_min[1]:.2f}, {bounds_min[2]:.2f}) "
                    f"to ({bounds_max[0]:.2f}, {bounds_max[1]:.2f}, {bounds_max[2]:.2f})"
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
            NodeItem(CSG_NODE_IDNAME),
        ],
    ),
    _MathOPSV2NodeCategory(
        "MATHOPS_V2_SDF_DATA",
        "Data",
        items=[
            NodeItem(VALUE_NODE_IDNAME),
            NodeItem(VECTOR_NODE_IDNAME),
            NodeItem(COLOR_NODE_IDNAME),
            NodeItem(TRANSFORM_NODE_IDNAME),
            NodeItem(BREAK_TRANSFORM_NODE_IDNAME),
        ],
    ),
]


classes = (
    MathOPSV2SDFTree,
    MathOPSV2SDFSocket,
    MathOPSV2FloatSocket,
    MathOPSV2VectorSocket,
    MathOPSV2ColorSocket,
    MathOPSV2TransformSocket,
    MathOPSV2SceneOutputNode,
    MathOPSV2ValueNode,
    MathOPSV2VectorNode,
    MathOPSV2ColorNode,
    MathOPSV2TransformNode,
    MathOPSV2BreakTransformNode,
    MathOPSV2SDFSphereNode,
    MathOPSV2SDFBoxNode,
    MathOPSV2SDFCylinderNode,
    MathOPSV2SDFConeNode,
    MathOPSV2CSGNode,
    MathOPSV2CSGUnionNode,
    MathOPSV2CSGSubtractNode,
    MathOPSV2CSGIntersectNode,
    MATHOPS_V2_OT_edit_scene_sdf_tree,
    MATHOPS_V2_OT_view_sdf_node,
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
