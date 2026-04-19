MIN_PRIMITIVES = 1
GRID_LEVEL_DEFAULT = 4


def should_build(settings, compiled) -> bool:
    return bool(
        getattr(settings, "culling_enabled", True)
        and compiled.get("root_node") is not None
        and int(compiled.get("primitive_count", 0)) >= MIN_PRIMITIVES
        and int(compiled.get("instruction_count", 0)) > 1
    )


def grid_level(settings) -> int:
    value = int(getattr(settings, "pruning_grid_level", GRID_LEVEL_DEFAULT))
    value = max(2, min(10, value))
    if value % 2:
        value += 1
    return value


def final_grid_size(settings) -> int:
    return 1 << grid_level(settings)


def _round3(values):
    return tuple(round(float(value), 5) for value in values)


def _normalized_bounds(scene_bounds):
    bounds_min = [float(value) for value in scene_bounds[0]]
    bounds_max = [float(value) for value in scene_bounds[1]]
    for axis in range(3):
        if (bounds_max[axis] - bounds_min[axis]) < 1.0e-4:
            bounds_min[axis] -= 0.5
            bounds_max[axis] += 0.5
    return tuple(bounds_min), tuple(bounds_max)


def normalized_bounds(compiled):
    return _normalized_bounds(compiled["scene_bounds"])


def content_key(settings, compiled):
    bounds_min, bounds_max = _normalized_bounds(compiled["scene_bounds"])
    return (
        str(compiled["hash"]),
        grid_level(settings),
        _round3(bounds_min),
        _round3(bounds_max),
    )


def topology_key(settings, compiled):
    return (
        str(compiled.get("topology_hash", compiled["hash"])),
        int(compiled.get("instruction_count", 0)),
        final_grid_size(settings),
    )


def build_initial_topology(compiled):
    root_node = compiled.get("root_node")
    if root_node is None:
        return {"active_nodes": [], "parents": []}

    active_nodes = []
    parents = []

    def visit(node, local_sign=True):
        left_index = -1
        right_index = -1
        if node["kind"] == "op":
            left_index = visit(node["left"], True)
            right_index = visit(node["right"], int(node["op"]) != 2)

        current_index = len(active_nodes)
        active_nodes.append(
            {
                "instruction_index": int(node["instruction_index"]),
                "sign": bool(local_sign),
            }
        )
        parents.append(-1)
        if left_index >= 0:
            parents[left_index] = current_index
        if right_index >= 0:
            parents[right_index] = current_index
        return current_index

    visit(root_node, True)
    return {"active_nodes": active_nodes, "parents": parents}


def debug_mode_value(settings) -> int:
    mode = str(getattr(settings, "debug_shading", "SHADED") or "SHADED")
    if mode == "PRUNING_ACTIVE":
        return 1
    if mode == "PRUNING_FIELD":
        return 2
    if mode == "STEP_COUNT":
        return 3
    return 0
