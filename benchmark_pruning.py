import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import bpy
import gpu
from mathutils import Matrix


REPO_ROOT = Path(__file__).resolve().parent
ADDON_ROOT = REPO_ROOT / "blender_addon"
if str(ADDON_ROOT) not in sys.path:
    sys.path.insert(0, str(ADDON_ROOT))

import mathops_v2
from mathops_v2 import properties, runtime
from mathops_v2.nodes import sdf_tree
from mathops_v2.render.gpu_viewport import MathOPSV2GPUViewport


PRIMITIVE_TYPES = ("sphere", "box", "cylinder", "torus", "cone", "capsule", "ngon")
DEFAULT_COUNTS = (10, 50, 100, 150, 200)
DEFAULT_BLENDS = (0.0, 0.1, 0.25)
DEFAULT_OPERATIONS = ("UNION", "SUBTRACT", "INTERSECT")


class _FakeRegionData:
    def __init__(self, view_matrix, perspective_matrix, is_perspective):
        self.view_matrix = view_matrix
        self.perspective_matrix = perspective_matrix
        self.is_perspective = is_perspective


def _parse_csv_numbers(text, cast=float):
    values = []
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        values.append(cast(part))
    return tuple(values)


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Benchmark MathOPS pruning build and scene pass cost.")
    parser.add_argument("--counts", default=",".join(str(value) for value in DEFAULT_COUNTS))
    parser.add_argument("--blends", default=",".join(str(value) for value in DEFAULT_BLENDS))
    parser.add_argument("--operations", default=",".join(DEFAULT_OPERATIONS))
    parser.add_argument("--grid-level", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--draw-repeats", type=int, default=5)
    parser.add_argument("--output", default="")
    return parser.parse_args(argv)


def _ensure_addon_loaded(scene):
    if not hasattr(bpy.types.Scene, "mathops_v2"):
        mathops_v2.register()
    properties.ensure_scene_defaults(scene, context=None)


def _scene_and_tree():
    scene = bpy.context.scene
    _ensure_addon_loaded(scene)
    tree = sdf_tree.ensure_scene_tree(scene)
    return scene, runtime.scene_settings(scene), tree


def _reset_tree(tree):
    output = sdf_tree.ensure_graph_output(tree)
    while output.inputs[0].is_linked:
        tree.links.remove(output.inputs[0].links[0])
    for node in list(tree.nodes):
        if node == output:
            continue
        tree.nodes.remove(node)
    output.location = (1400.0, 0.0)
    return output


def _centered_grid_position(index, count, spacing, jitter=0.0):
    side = max(1, int(math.ceil(count ** (1.0 / 3.0))))
    x = index % side
    y = (index // side) % side
    z = index // (side * side)
    cx = (x - (side - 1) * 0.5) * spacing
    cy = (y - (side - 1) * 0.5) * spacing
    cz = (z - (side - 1) * 0.5) * spacing * 0.9
    if jitter > 0.0:
        cx += math.sin(index * 1.73) * jitter
        cy += math.cos(index * 2.11) * jitter
        cz += math.sin(index * 1.37) * jitter * 0.5
    return (cx, cy, cz)


def _cluster_position(index, radius_xy, radius_z):
    angle = index * 2.399963229728653
    ring = 0.35 + 0.65 * ((index % 11) / 10.0)
    x = math.cos(angle) * radius_xy * ring
    y = math.sin(angle) * radius_xy * ring
    z = (((index % 9) / 8.0) - 0.5) * radius_z
    return (x, y, z)


def _rotation_for_index(index):
    return (
        (index % 7) * 0.17,
        (index % 11) * 0.11,
        (index % 13) * 0.09,
    )


def _configure_primitive(node, primitive_type, scale_factor=1.0):
    node.primitive_type = primitive_type
    if primitive_type == "sphere":
        node.radius = 0.48 * scale_factor
        return
    if primitive_type == "box":
        node.size = (0.42 * scale_factor, 0.34 * scale_factor, 0.30 * scale_factor)
        return
    if primitive_type == "cylinder":
        node.radius = 0.32 * scale_factor
        node.height = 0.92 * scale_factor
        return
    if primitive_type == "torus":
        node.major_radius = 0.42 * scale_factor
        node.minor_radius = 0.14 * scale_factor
        return
    if primitive_type == "cone":
        node.cone_bottom_radius = 0.40 * scale_factor
        node.cone_top_radius = 0.16 * scale_factor
        node.height = 0.96 * scale_factor
        return
    if primitive_type == "capsule":
        node.radius = 0.23 * scale_factor
        node.height = 0.70 * scale_factor
        return
    if primitive_type == "ngon":
        node.radius = 0.43 * scale_factor
        node.height = 0.64 * scale_factor
        node.ngon_sides = 5 + (int(scale_factor * 10.0) % 4)


def _new_object_node(tree, index, primitive_type, location, rotation, scale_factor=1.0):
    node = tree.nodes.new(runtime.OBJECT_NODE_IDNAME)
    with sdf_tree.suppress_object_node_updates():
        _configure_primitive(node, primitive_type, scale_factor=scale_factor)
        node.sdf_location = location
        node.sdf_rotation = rotation
        node.sdf_scale = (1.0, 1.0, 1.0)
    return node


def _new_csg_node(tree, operation, blend):
    node = tree.nodes.new(runtime.CSG_NODE_IDNAME)
    node.operation = operation
    node.blend = blend
    return node


def _build_balanced_csg(tree, sources, operation, blend):
    current = list(sources)
    while len(current) > 1:
        next_level = []
        for index in range(0, len(current), 2):
            if index + 1 >= len(current):
                next_level.append(current[index])
                continue
            node = _new_csg_node(tree, operation, blend)
            tree.links.new(current[index].outputs[0], node.inputs[0])
            tree.links.new(current[index + 1].outputs[0], node.inputs[1])
            next_level.append(node)
        current = next_level
    return current[0]


def _build_case_graph(tree, output, operation, blend, count):
    if operation == "UNION":
        sources = []
        for index in range(count):
            primitive_type = PRIMITIVE_TYPES[index % len(PRIMITIVE_TYPES)]
            location = _centered_grid_position(index, count, spacing=1.85, jitter=0.16)
            rotation = _rotation_for_index(index)
            sources.append(_new_object_node(tree, index, primitive_type, location, rotation, scale_factor=1.0))
        root = _build_balanced_csg(tree, sources, "UNION", blend)
    elif operation == "INTERSECT":
        sources = []
        for index in range(count):
            primitive_type = PRIMITIVE_TYPES[index % len(PRIMITIVE_TYPES)]
            location = _cluster_position(index, radius_xy=0.30, radius_z=0.24)
            rotation = _rotation_for_index(index)
            sources.append(_new_object_node(tree, index, primitive_type, location, rotation, scale_factor=2.8))
        root = _build_balanced_csg(tree, sources, "INTERSECT", blend)
    else:
        base = _new_object_node(tree, 0, "sphere", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), scale_factor=5.0)
        cutters = []
        for index in range(1, count):
            primitive_type = PRIMITIVE_TYPES[index % len(PRIMITIVE_TYPES)]
            location = _centered_grid_position(index - 1, max(count - 1, 1), spacing=0.85, jitter=0.08)
            rotation = _rotation_for_index(index)
            cutters.append(_new_object_node(tree, index, primitive_type, location, rotation, scale_factor=1.05))
        if cutters:
            cutter_root = cutters[0] if len(cutters) == 1 else _build_balanced_csg(tree, cutters, "UNION", 0.0)
            root = _new_csg_node(tree, "SUBTRACT", blend)
            tree.links.new(base.outputs[0], root.inputs[0])
            tree.links.new(cutter_root.outputs[0], root.inputs[1])
        else:
            root = base
    tree.links.new(root.outputs[0], output.inputs[0])


def _compile_scene(scene):
    runtime.mark_scene_static_dirty(scene)
    start = time.perf_counter()
    compiled = sdf_tree.compile_scene(scene)
    return compiled, (time.perf_counter() - start) * 1000.0


def _ortho_matrix(left, right, bottom, top, near, far):
    return Matrix(
        (
            (2.0 / (right - left), 0.0, 0.0, -((right + left) / (right - left))),
            (0.0, 2.0 / (top - bottom), 0.0, -((top + bottom) / (top - bottom))),
            (0.0, 0.0, -2.0 / (far - near), -((far + near) / (far - near))),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def _camera_setup(compiled):
    bounds_min, bounds_max = compiled.get("render_bounds", compiled["scene_bounds"])
    min_x, min_y, min_z = (float(value) for value in bounds_min)
    max_x, max_y, max_z = (float(value) for value in bounds_max)
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    span_z = max(max_z - min_z, 1.0)
    half_extent = max(span_x, span_y) * 0.8 + 1.0
    max_extent = max(span_x, span_y, span_z)
    padding = max_extent * 0.6 + 1.0
    camera_z = max_z + max_extent * 2.5 + 2.0
    view_matrix = Matrix.Translation((-center_x, -center_y, -camera_z))
    projection_matrix = _ortho_matrix(-half_extent, half_extent, -half_extent, half_extent, max(0.1, camera_z - (max_z + padding)), camera_z - (min_z - padding))
    view_projection = projection_matrix @ view_matrix
    region_data = _FakeRegionData(view_matrix, view_projection, False)
    return region_data, (center_x, center_y, camera_z), view_projection.inverted(), view_projection


def _measure_pruning_build(renderer, settings, compiled, scene_texture, polygon_texture):
    renderer._content_key = None
    start = time.perf_counter()
    renderer._update_pruning(settings, compiled, scene_texture, polygon_texture)
    total_ms = (time.perf_counter() - start) * 1000.0
    return total_ms, float(renderer._pruning_stats.get("ms", 0.0))


def _draw_scene_pass_ms(renderer, settings, scene, scene_texture, polygon_texture, outline_texture, params_ubo, inv_view_projection, view_projection, repeats):
    samples = []
    background = runtime.scene_background_color(scene)
    for _ in range(max(1, repeats)):
        with renderer._offscreen.bind():
            gpu.state.depth_test_set("ALWAYS")
            gpu.state.depth_mask_set(True)
            gpu.state.blend_set("NONE")
            renderer._clear_texture(renderer._offscreen_color_texture, "FLOAT", background)
            renderer._clear_texture(renderer._offscreen_outline_texture, "FLOAT", (0.0,))
            renderer._clear_texture(renderer._offscreen_position_texture, "FLOAT", (0.0, 0.0, 0.0, 0.0))
            renderer._offscreen.clear(depth=1.0, stencil=0)
            gpu.state.depth_test_set("LESS_EQUAL")
            start = time.perf_counter()
            renderer._draw_scene_pass(None, settings, scene_texture, polygon_texture, outline_texture, params_ubo, inv_view_projection, view_projection)
            renderer._offscreen.read_color(0, 0, 1, 1, 4, 2, "FLOAT")
            samples.append((time.perf_counter() - start) * 1000.0)
    gpu.state.depth_test_set("NONE")
    gpu.state.depth_mask_set(False)
    gpu.state.blend_set("NONE")
    return statistics.median(samples)


def _measure_draw_pair(renderer, settings, scene, compiled, resolution, draw_repeats):
    if not renderer._ensure_offscreen(resolution, resolution):
        raise RuntimeError("GPU offscreen buffers are unavailable")
    scene_texture = renderer._ensure_scene_texture(compiled)
    polygon_texture = renderer._ensure_polygon_texture(compiled)
    outline_values = [0.0] * max(1, int(compiled.get("primitive_count", 0)))
    outline_texture = renderer._create_scalar_texture(len(outline_values), "R32F", outline_values)
    region_data, camera_position, inv_view_projection, view_projection = _camera_setup(compiled)
    params_no_prune = None
    params_prune = None

    renderer._disable_pruning()
    params_no_prune = renderer._ensure_params_ubo(settings, compiled, camera_position, region_data, False)
    draw_no_prune_ms = _draw_scene_pass_ms(
        renderer,
        settings,
        scene,
        scene_texture,
        polygon_texture,
        outline_texture,
        params_no_prune,
        inv_view_projection,
        view_projection,
        draw_repeats,
    )

    prune_total_ms, prune_dispatch_ms = _measure_pruning_build(renderer, settings, compiled, scene_texture, polygon_texture)
    params_prune = renderer._ensure_params_ubo(settings, compiled, camera_position, region_data, False)
    draw_prune_ms = _draw_scene_pass_ms(
        renderer,
        settings,
        scene,
        scene_texture,
        polygon_texture,
        outline_texture,
        params_prune,
        inv_view_projection,
        view_projection,
        draw_repeats,
    )

    benefit_ms = draw_no_prune_ms - draw_prune_ms
    break_even_frames = math.inf if benefit_ms <= 1.0e-6 else (prune_total_ms / benefit_ms)
    return {
        "draw_no_prune_ms": draw_no_prune_ms,
        "draw_prune_ms": draw_prune_ms,
        "draw_benefit_ms": benefit_ms,
        "prune_total_ms": prune_total_ms,
        "prune_dispatch_ms": prune_dispatch_ms,
        "break_even_frames": break_even_frames,
        "speedup": (draw_no_prune_ms / draw_prune_ms) if draw_prune_ms > 1.0e-6 else math.inf,
        "pruning_sequences": int(renderer._max_active_count),
    }


def _build_case(scene, tree, output, settings, operation, blend, count):
    _reset_tree(tree)
    _build_case_graph(tree, output, operation, blend, count)
    settings.culling_enabled = True
    compiled, compile_ms = _compile_scene(scene)
    return compiled, compile_ms


def _warmup(renderer, settings, scene, tree, output, resolution):
    settings.pruning_grid_level = max(2, min(8, int(settings.pruning_grid_level)))
    renderer._ensure_draw_shader()
    renderer._ensure_compute_shader()
    renderer._ensure_batch()
    if not renderer._ensure_offscreen(resolution, resolution):
        raise RuntimeError("GPU offscreen buffers are unavailable")
    compiled, _compile_ms = _build_case(scene, tree, output, settings, "UNION", 0.0, 10)
    scene_texture = renderer._ensure_scene_texture(compiled)
    polygon_texture = renderer._ensure_polygon_texture(compiled)
    renderer._update_pruning(settings, compiled, scene_texture, polygon_texture)
    _measure_draw_pair(renderer, settings, scene, compiled, resolution, 1)


def _results_summary(results):
    summary = {}
    for operation in sorted({entry["operation"] for entry in results}):
        op_entries = [entry for entry in results if entry["operation"] == operation]
        best = []
        for blend in sorted({entry["blend"] for entry in op_entries}):
            blend_entries = [entry for entry in op_entries if entry["blend"] == blend]
            viable = [entry for entry in blend_entries if entry["draw_benefit_ms"] > 0.0]
            if not viable:
                best.append({"blend": blend, "first_positive_count": None, "best_count": None, "best_break_even_frames": None})
                continue
            first_positive = min(viable, key=lambda entry: entry["count"])
            best_entry = min(viable, key=lambda entry: entry["break_even_frames"])
            best.append(
                {
                    "blend": blend,
                    "first_positive_count": int(first_positive["count"]),
                    "best_count": int(best_entry["count"]),
                    "best_break_even_frames": float(best_entry["break_even_frames"]),
                }
            )
        summary[operation] = best
    return summary


def main(argv=None):
    if bpy.app.background:
        raise RuntimeError("GPU pruning benchmark requires Blender foreground mode; run without --background.")
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    args = _parse_args(argv)
    counts = _parse_csv_numbers(args.counts, int)
    blends = _parse_csv_numbers(args.blends, float)
    operations = tuple(part.strip().upper() for part in str(args.operations).split(",") if part.strip())
    grid_level = max(2, min(8, int(args.grid_level)))
    resolution = max(64, int(args.resolution))

    scene, settings, tree = _scene_and_tree()
    output = sdf_tree.ensure_graph_output(tree)
    settings.pruning_grid_level = grid_level
    settings.culling_enabled = True
    settings.debug_shading = "SHADED"
    settings.disable_surface_shading = False

    renderer = MathOPSV2GPUViewport()
    _warmup(renderer, settings, scene, tree, output, resolution)

    results = []
    for operation in operations:
        for blend in blends:
            for count in counts:
                settings.pruning_grid_level = grid_level
                compiled, compile_ms = _build_case(scene, tree, output, settings, operation, blend, count)
                metrics = _measure_draw_pair(renderer, settings, scene, compiled, resolution, args.draw_repeats)
                result = {
                    "operation": operation,
                    "blend": float(blend),
                    "count": int(count),
                    "grid_level": grid_level,
                    "resolution": resolution,
                    "primitive_count": int(compiled.get("primitive_count", 0)),
                    "instruction_count": int(compiled.get("instruction_count", 0)),
                    "compile_ms": compile_ms,
                    **metrics,
                }
                results.append(result)
                break_even = result["break_even_frames"]
                break_even_text = "inf" if not math.isfinite(break_even) else f"{break_even:.2f}"
                print(
                    "CASE"
                    f" op={operation:<9}"
                    f" blend={blend:>4.2f}"
                    f" count={count:>3}"
                    f" compile={result['compile_ms']:.2f}ms"
                    f" prune={result['prune_total_ms']:.2f}ms"
                    f" no_prune={result['draw_no_prune_ms']:.2f}ms"
                    f" prune_draw={result['draw_prune_ms']:.2f}ms"
                    f" benefit={result['draw_benefit_ms']:.2f}ms"
                    f" break_even={break_even_text}"
                    f" active_seq={result['pruning_sequences']}"
                )

    payload = {
        "grid_level": grid_level,
        "resolution": resolution,
        "draw_repeats": int(args.draw_repeats),
        "results": results,
        "summary": _results_summary(results),
    }
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("RESULTS_JSON_BEGIN")
    print(json.dumps(payload, indent=2))
    print("RESULTS_JSON_END")
    try:
        bpy.ops.wm.quit_blender()
    except Exception:
        pass


if __name__ == "__main__":
    main()
