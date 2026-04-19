from collections import defaultdict

import bpy
from bpy.types import Operator

from .. import runtime


_COLUMN_GAP = 160.0
_ROW_GAP = 70.0
_MIN_NODE_WIDTH = 80.0
_MIN_NODE_HEIGHT = 36.0
_REROUTE_SIZE = (16.0, 16.0)
_ORDERING_SWEEPS = 12
_PLACEMENT_SWEEPS = 6

_addon_keymaps = []


def _edit_tree(context):
    space = getattr(context, "space_data", None)
    if space is None or getattr(space, "type", "") != "NODE_EDITOR":
        return None
    tree = getattr(space, "edit_tree", None)
    if tree is None or getattr(tree, "bl_idname", "") != runtime.TREE_IDNAME:
        return None
    return tree


def _ui_scale(context):
    preferences = getattr(context, "preferences", None)
    system = None if preferences is None else getattr(preferences, "system", None)
    return max(1.0, float(getattr(system, "ui_scale", 1.0)))


def _absolute_location(node):
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


def _top_group_root(node):
    root = node
    parent = getattr(root, "parent", None)
    while parent is not None:
        root = parent
        parent = getattr(root, "parent", None)
    return root


def _node_size(node, ui_scale):
    if getattr(node, "bl_idname", "") == "NodeReroute":
        return _REROUTE_SIZE

    width = float(getattr(node, "width", 0.0) or 0.0)
    height = 0.0
    dimensions = getattr(node, "dimensions", None)
    if dimensions is not None:
        width = max(width, float(dimensions.x) / ui_scale)
        height = max(height, float(dimensions.y) / ui_scale)

    width = max(width, _MIN_NODE_WIDTH)
    if height <= 1.0e-6:
        height = max(float(getattr(node, "bl_height_default", 0.0) or 0.0), _MIN_NODE_HEIGHT)
    return width, height


def _node_bounds(node, ui_scale):
    x, y = _absolute_location(node)
    width, height = _node_size(node, ui_scale)
    return {
        "left": x,
        "right": x + width,
        "top": y,
        "bottom": y - height,
    }


def _group_bounds(nodes, ui_scale):
    bounds = [_node_bounds(node, ui_scale) for node in nodes]
    left = min(bound["left"] for bound in bounds)
    right = max(bound["right"] for bound in bounds)
    top = max(bound["top"] for bound in bounds)
    bottom = min(bound["bottom"] for bound in bounds)
    return {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "width": right - left,
        "height": top - bottom,
        "center_x": (left + right) * 0.5,
        "center_y": (top + bottom) * 0.5,
    }


def _build_groups(tree, ui_scale):
    groups = {}
    group_root_by_node = {}
    group_nodes = defaultdict(list)
    edge_weights = defaultdict(int)

    for node in tree.nodes:
        group_root = _top_group_root(node)
        node_key = runtime.safe_pointer(node)
        group_key = runtime.safe_pointer(group_root)
        if node_key == 0 or group_key == 0:
            continue
        group_root_by_node[node_key] = group_key
        group_nodes[group_key].append(node)
        if group_key not in groups:
            groups[group_key] = {
                "root": group_root,
                "name": str(getattr(group_root, "name", "") or ""),
                "predecessors": defaultdict(int),
                "successors": defaultdict(int),
            }

    for group_key, nodes in group_nodes.items():
        groups[group_key]["nodes"] = nodes
        groups[group_key].update(_group_bounds(nodes, ui_scale))

    for link in tree.links:
        if not bool(getattr(link, "is_valid", True)):
            continue
        from_node = getattr(link, "from_node", None)
        to_node = getattr(link, "to_node", None)
        from_key = group_root_by_node.get(runtime.safe_pointer(from_node), 0)
        to_key = group_root_by_node.get(runtime.safe_pointer(to_node), 0)
        if from_key == 0 or to_key == 0 or from_key == to_key:
            continue
        edge_weights[(from_key, to_key)] += 1

    for (from_key, to_key), weight in edge_weights.items():
        groups[from_key]["successors"][to_key] += weight
        groups[to_key]["predecessors"][from_key] += weight

    return groups, edge_weights


def _compute_columns(groups):
    earliest_ranks = {}
    latest_ranks = {}
    visiting = set()

    def _earliest_rank(group_key):
        cached = earliest_ranks.get(group_key)
        if cached is not None:
            return cached
        if group_key in visiting:
            return 0
        visiting.add(group_key)
        rank = 0
        predecessors = groups[group_key]["predecessors"]
        if predecessors:
            rank = 1 + max(_earliest_rank(predecessor) for predecessor in predecessors)
        visiting.remove(group_key)
        earliest_ranks[group_key] = rank
        return rank

    def _latest_rank(group_key):
        cached = latest_ranks.get(group_key)
        if cached is not None:
            return cached
        successors = groups[group_key]["successors"]
        if not successors:
            rank = earliest_ranks[group_key]
        else:
            rank = min(_latest_rank(successor) - 1 for successor in successors)
            rank = max(int(rank), int(earliest_ranks[group_key]))
        latest_ranks[group_key] = rank
        return rank

    for group_key in groups:
        _earliest_rank(group_key)

    columns = defaultdict(list)
    for group_key, group in groups.items():
        group["earliest_column"] = int(earliest_ranks[group_key])
        group["latest_column"] = int(_latest_rank(group_key))
        column = int(group["latest_column"])
        group["column"] = column
        columns[column].append(group_key)
    return dict(sorted(columns.items()))


def _new_layout_item(kind, column, center_y, name, group_key=None):
    return {
        "kind": kind,
        "column": int(column),
        "center_y": float(center_y),
        "name": str(name or ""),
        "group_key": group_key,
        "predecessors": defaultdict(int),
        "successors": defaultdict(int),
    }


def _connect_layout_items(layout_items, from_item_key, to_item_key, weight):
    layout_items[from_item_key]["successors"][to_item_key] += int(weight)
    layout_items[to_item_key]["predecessors"][from_item_key] += int(weight)


def _build_layout(columns, groups, edge_weights):
    layout_items = {}
    layout_columns = defaultdict(list)

    for column, group_keys in columns.items():
        for group_key in group_keys:
            item_key = ("group", group_key)
            group = groups[group_key]
            layout_items[item_key] = _new_layout_item(
                "group",
                column,
                group["center_y"],
                group["name"],
                group_key=group_key,
            )
            layout_columns[column].append(item_key)

    sorted_edges = sorted(
        edge_weights.items(),
        key=lambda item: (
            groups[item[0][0]]["column"],
            groups[item[0][1]]["column"],
            -groups[item[0][0]]["center_y"],
            -groups[item[0][1]]["center_y"],
            groups[item[0][0]]["name"],
            groups[item[0][1]]["name"],
        ),
    )
    for edge_index, ((from_key, to_key), weight) in enumerate(sorted_edges):
        from_group = groups[from_key]
        to_group = groups[to_key]
        from_column = int(from_group["column"])
        to_column = int(to_group["column"])
        span = max(1, to_column - from_column)
        previous_item = ("group", from_key)
        if span > 1:
            for step in range(1, span):
                column = from_column + step
                factor = float(step) / float(span)
                center_y = (1.0 - factor) * float(from_group["center_y"]) + factor * float(to_group["center_y"])
                dummy_item = ("dummy", edge_index, step)
                layout_items[dummy_item] = _new_layout_item(
                    "dummy",
                    column,
                    center_y,
                    f"{from_group['name']}->{to_group['name']}",
                )
                layout_columns[column].append(dummy_item)
                _connect_layout_items(layout_items, previous_item, dummy_item, weight)
                previous_item = dummy_item
        _connect_layout_items(layout_items, previous_item, ("group", to_key), weight)

    for column, item_keys in layout_columns.items():
        item_keys.sort(
            key=lambda item_key: (
                -layout_items[item_key]["center_y"],
                0 if layout_items[item_key]["kind"] == "group" else 1,
                layout_items[item_key]["name"],
                str(item_key),
            )
        )
    return layout_items, dict(sorted(layout_columns.items()))


def _layout_positions(layout_columns):
    positions = {}
    for column, item_keys in layout_columns.items():
        for index, item_key in enumerate(item_keys):
            positions[item_key] = (column, index)
    return positions


def _weighted_layout_position(layout_item, positions, neighbor_key):
    neighbors = layout_item[neighbor_key]
    if not neighbors:
        return None
    total = 0.0
    weight_total = 0.0
    for item_key, weight in neighbors.items():
        position = positions.get(item_key)
        if position is None:
            continue
        total += float(position[1]) * float(weight)
        weight_total += float(weight)
    if weight_total <= 0.0:
        return None
    return total / weight_total


def _sort_layout_sweep(layout_items, layout_columns, column_keys, neighbor_key):
    positions = _layout_positions(layout_columns)
    for column in column_keys:
        scored_items = []
        for item_key in layout_columns[column]:
            layout_item = layout_items[item_key]
            barycenter = _weighted_layout_position(layout_item, positions, neighbor_key)
            previous_index = positions[item_key][1]
            scored_items.append(
                (
                    1 if barycenter is None else 0,
                    float(previous_index) if barycenter is None else float(barycenter),
                    0 if layout_item["kind"] == "group" else 1,
                    float(previous_index),
                    -float(layout_item["center_y"]),
                    layout_item["name"],
                    item_key,
                )
            )
        scored_items.sort()
        layout_columns[column] = [item_key for *_parts, item_key in scored_items]
        positions = _layout_positions(layout_columns)


def _reduce_crossings(layout_items, layout_columns):
    column_keys = sorted(layout_columns)
    if len(column_keys) < 2:
        return
    for _index in range(_ORDERING_SWEEPS):
        _sort_layout_sweep(layout_items, layout_columns, column_keys[1:], "predecessors")
        _sort_layout_sweep(layout_items, layout_columns, reversed(column_keys[:-1]), "successors")


def _real_columns(layout_items, layout_columns):
    columns = {}
    for column, item_keys in layout_columns.items():
        columns[column] = [
            layout_items[item_key]["group_key"]
            for item_key in item_keys
            if layout_items[item_key]["kind"] == "group"
        ]
    return columns


def _weighted_group_center(groups, weights, attribute):
    total = 0.0
    weight_total = 0.0
    for group_key, weight in weights.items():
        group = groups.get(group_key)
        if group is None:
            continue
        total += float(group[attribute]) * float(weight)
        weight_total += float(weight)
    if weight_total <= 0.0:
        return None, 0.0
    return total / weight_total, weight_total


def _column_ideals(groups, ordered_group_keys, mode):
    ideals = {}
    for group_key in ordered_group_keys:
        group = groups[group_key]
        current = float(group["target_center_y"])
        incoming, incoming_weight = _weighted_group_center(groups, group["predecessors"], "target_center_y")
        outgoing, outgoing_weight = _weighted_group_center(groups, group["successors"], "target_center_y")

        if mode == "predecessors":
            if incoming is None:
                ideals[group_key] = current
            else:
                ideals[group_key] = (incoming * 0.75) + (current * 0.25)
            continue

        if mode == "successors":
            if outgoing is None:
                ideals[group_key] = current
            else:
                ideals[group_key] = (outgoing * 0.75) + (current * 0.25)
            continue

        total = current
        weight_total = 1.0
        if incoming is not None:
            total += incoming * incoming_weight
            weight_total += incoming_weight
        if outgoing is not None:
            total += outgoing * outgoing_weight
            weight_total += outgoing_weight
        ideals[group_key] = total / weight_total
    return ideals


def _pack_column(groups, ordered_group_keys, ideals):
    if not ordered_group_keys:
        return

    centers = {}
    first_key = ordered_group_keys[0]
    centers[first_key] = float(ideals[first_key])
    for index in range(1, len(ordered_group_keys)):
        previous_key = ordered_group_keys[index - 1]
        group_key = ordered_group_keys[index]
        previous_group = groups[previous_key]
        group = groups[group_key]
        separation = (float(previous_group["height"]) * 0.5) + (float(group["height"]) * 0.5) + _ROW_GAP
        centers[group_key] = min(float(ideals[group_key]), centers[previous_key] - separation)

    for index in range(len(ordered_group_keys) - 2, -1, -1):
        group_key = ordered_group_keys[index]
        next_key = ordered_group_keys[index + 1]
        group = groups[group_key]
        next_group = groups[next_key]
        separation = (float(group["height"]) * 0.5) + (float(next_group["height"]) * 0.5) + _ROW_GAP
        centers[group_key] = max(centers[group_key], centers[next_key] + separation)

    ideal_average = sum(float(ideals[group_key]) for group_key in ordered_group_keys) / float(len(ordered_group_keys))
    center_average = sum(centers[group_key] for group_key in ordered_group_keys) / float(len(ordered_group_keys))
    shift = ideal_average - center_average
    for group_key in ordered_group_keys:
        groups[group_key]["target_center_y"] = centers[group_key] + shift


def _graph_center(groups, top_key, bottom_key):
    left = min(float(group["left"]) for group in groups.values())
    right = max(float(group["right"]) for group in groups.values())
    top = max(float(group[top_key]) for group in groups.values())
    bottom = min(float(group[bottom_key]) for group in groups.values())
    return (left + right) * 0.5, (top + bottom) * 0.5


def _assign_targets(groups, layout_items, layout_columns):
    center_x, center_y = _graph_center(groups, "top", "bottom")
    real_columns = _real_columns(layout_items, layout_columns)
    column_keys = sorted(real_columns)

    if not column_keys:
        return

    for group in groups.values():
        group["target_center_y"] = float(group["center_y"])

    for _index in range(_PLACEMENT_SWEEPS):
        for column in column_keys[1:]:
            ordered_group_keys = real_columns[column]
            _pack_column(groups, ordered_group_keys, _column_ideals(groups, ordered_group_keys, "predecessors"))
        for column in reversed(column_keys[:-1]):
            ordered_group_keys = real_columns[column]
            _pack_column(groups, ordered_group_keys, _column_ideals(groups, ordered_group_keys, "successors"))

    for column in column_keys:
        ordered_group_keys = real_columns[column]
        _pack_column(groups, ordered_group_keys, _column_ideals(groups, ordered_group_keys, "both"))

    target_tops = {}
    target_bottoms = {}
    for group_key, group in groups.items():
        height = float(group["height"])
        target_tops[group_key] = float(group["target_center_y"]) + (height * 0.5)
        target_bottoms[group_key] = float(group["target_center_y"]) - (height * 0.5)

    target_center_y = (max(target_tops.values()) + min(target_bottoms.values())) * 0.5
    delta_y = center_y - target_center_y
    for group_key, group in groups.items():
        group["target_center_y"] = float(group["target_center_y"]) + delta_y
        group["target_top"] = float(group["target_center_y"]) + (float(group["height"]) * 0.5)

    column_widths = []
    for column in column_keys:
        ordered_group_keys = real_columns[column]
        width = max(float(groups[group_key]["width"]) for group_key in ordered_group_keys) if ordered_group_keys else 0.0
        column_widths.append(width)

    total_width = sum(column_widths) + (_COLUMN_GAP * max(0, len(column_widths) - 1))
    left = center_x - (total_width * 0.5)
    for column_width, column in zip(column_widths, column_keys):
        for group_key in real_columns[column]:
            groups[group_key]["target_left"] = left
        left += column_width + _COLUMN_GAP


def _apply_targets(groups):
    moved = False
    for group in groups.values():
        target_left = group.get("target_left")
        target_top = group.get("target_top")
        if target_left is None or target_top is None:
            continue

        delta_x = float(target_left) - float(group["left"])
        delta_y = float(target_top) - float(group["top"])
        if abs(delta_x) <= 1.0e-4 and abs(delta_y) <= 1.0e-4:
            continue

        root = group["root"]
        root.location = (
            float(root.location.x) + delta_x,
            float(root.location.y) + delta_y,
        )
        moved = True
    return moved


class MATHOPS_V2_OT_arrange_graph(Operator):
    bl_idname = "mathops_v2.arrange_graph"
    bl_label = "Arrange Graph"
    bl_description = "Arrange the active MathOPS node graph while reducing link crossings"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _edit_tree(context) is not None

    def execute(self, context):
        tree = _edit_tree(context)
        if tree is None:
            self.report({"WARNING"}, "Open a MathOPS node tree in the Node Editor")
            return {"CANCELLED"}

        groups, edge_weights = _build_groups(tree, _ui_scale(context))
        if not groups:
            self.report({"WARNING"}, "No nodes to arrange")
            return {"CANCELLED"}

        columns = _compute_columns(groups)
        layout_items, layout_columns = _build_layout(columns, groups, edge_weights)
        _reduce_crossings(layout_items, layout_columns)
        _assign_targets(groups, layout_items, layout_columns)
        moved = _apply_targets(groups)
        runtime.note_interaction()
        runtime.tag_redraw(context)
        if not moved:
            self.report({"INFO"}, "Graph is already arranged")
        return {"FINISHED"}


classes = (MATHOPS_V2_OT_arrange_graph,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    window_manager = getattr(bpy.context, "window_manager", None)
    keyconfigs = None if window_manager is None else getattr(window_manager, "keyconfigs", None)
    keyconfig = None if keyconfigs is None else getattr(keyconfigs, "addon", None)
    if keyconfig is None:
        return

    keymap = keyconfig.keymaps.new(name="Node Editor", space_type="NODE_EDITOR")
    keymap_item = keymap.keymap_items.new(MATHOPS_V2_OT_arrange_graph.bl_idname, type="R", value="PRESS", shift=True)
    _addon_keymaps.append((keymap, keymap_item))


def unregister():
    for keymap, keymap_item in _addon_keymaps:
        try:
            keymap.keymap_items.remove(keymap_item)
        except Exception:
            pass
    _addon_keymaps.clear()

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
