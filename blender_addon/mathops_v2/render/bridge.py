from pathlib import Path
import importlib.util
import json
import math
import shutil
import sys
import time
import gc

import bpy
import numpy as np
from mathutils import Matrix, Vector

from .. import runtime, sdf_nodes


_DYNAMIC_AABB_GROW_PAD = 0.08
_DYNAMIC_AABB_SHRINK_PAD = 0.03
_DYNAMIC_AABB_SHRINK_INTERVAL = 0.75
_DYNAMIC_AABB_SHRINK_THRESHOLD = 1.05


def addon_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def repo_dir() -> Path:
    return addon_dir().parent.parent


def scene_dir() -> Path:
    for candidate in (addon_dir() / "scenes", repo_dir() / "scenes"):
        if candidate.is_dir():
            return candidate
    return repo_dir() / "scenes"


def template_scene_path(template_id: str) -> Path:
    return scene_dir() / runtime.TEMPLATE_FILES[template_id]


def load_native_module():
    if runtime.native_module is None:
        runtime.debug_log("Loading native renderer module")
        abi_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        candidates = sorted(
            addon_dir().glob(f"lipschitz_pruning_native*{abi_tag}*.pyd")
        )
        if not candidates:
            candidates = sorted(
                (addon_dir() / "Release").glob(
                    f"lipschitz_pruning_native*{abi_tag}*.pyd"
                )
            )
        if not candidates:
            candidates = sorted(addon_dir().glob("lipschitz_pruning_native*.pyd"))
        if not candidates:
            candidates = sorted(
                (addon_dir() / "Release").glob("lipschitz_pruning_native*.pyd")
            )
        if not candidates:
            raise ImportError(f"No native module found in {addon_dir()}")

        module_path = candidates[0]
        module_mtime = module_path.stat().st_mtime_ns
        cache_dir = addon_dir() / ".native_cache"
        cache_dir.mkdir(exist_ok=True)
        cached_module_path = (
            cache_dir / f"{module_path.stem}.{module_mtime}{module_path.suffix}"
        )
        if not cached_module_path.exists():
            shutil.copy2(module_path, cached_module_path)

        module_name = f"{addon_dir().name}.lipschitz_pruning_native"
        sys.modules.pop(module_name, None)
        spec = importlib.util.spec_from_file_location(module_name, cached_module_path)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Unable to load native module spec from {cached_module_path}"
            )

        native_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = native_module
        spec.loader.exec_module(native_module)
        runtime.debug_log(f"Native module loaded from {cached_module_path.name}")
        runtime.native_module = native_module
    return runtime.native_module


def get_render_size(scene):
    scale = scene.render.resolution_percentage / 100.0
    width = max(1, int(scene.render.resolution_x * scale))
    height = max(1, int(scene.render.resolution_y * scale))
    return width, height


def get_viewport_render_size(
    context, settings, max_dim_override=None, region_size=None
):
    if region_size is None:
        width = max(1, int(context.region.width))
        height = max(1, int(context.region.height))
    else:
        width = max(1, int(region_size[0]))
        height = max(1, int(region_size[1]))
    if max_dim_override is None:
        max_dim = max(0, int(getattr(settings, "viewport_max_dim", 0)))
    else:
        max_dim = max(0, int(max_dim_override))
    if max_dim <= 0:
        return width, height

    current_max = max(width, height)
    if current_max <= max_dim:
        return width, height

    scale = float(max_dim) / float(current_max)
    return max(1, int(width * scale)), max(1, int(height * scale))


def blender_to_renderer_vec(value):
    return (float(value[0]), float(value[2]), float(-value[1]))


def renderer_to_blender_vec(value):
    return Vector((float(value[0]), float(-value[2]), float(value[1])))


def renderer_aabb_to_blender(aabb_min, aabb_max):
    corners = []
    for x in (aabb_min[0], aabb_max[0]):
        for y in (aabb_min[1], aabb_max[1]):
            for z in (aabb_min[2], aabb_max[2]):
                corners.append(renderer_to_blender_vec((x, y, z)))

    min_corner = Vector(
        (
            min(v.x for v in corners),
            min(v.y for v in corners),
            min(v.z for v in corners),
        )
    )
    max_corner = Vector(
        (
            max(v.x for v in corners),
            max(v.y for v in corners),
            max(v.z for v in corners),
        )
    )
    return min_corner, max_corner


def renderer_aabb_edge_points(aabb_min, aabb_max):
    corners = []
    for x in (aabb_min[0], aabb_max[0]):
        for y in (aabb_min[1], aabb_max[1]):
            for z in (aabb_min[2], aabb_max[2]):
                corners.append(renderer_to_blender_vec((x, y, z)))
    edges = (
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    )
    return [tuple(corners[i]) for edge in edges for i in edge]


def grid_level(settings) -> int:
    value = max(2, min(8, settings.final_grid_level))
    if value % 2 != 0:
        value -= 1
    return value


def shader_dir(settings) -> str:
    return (
        bpy.path.abspath(settings.shader_dir)
        if settings.shader_dir
        else str(addon_dir())
    )


def resolve_scene_path(settings, strict=False, create=False) -> Path:
    if not getattr(settings, "use_sdf_nodes", False):
        return template_scene_path(settings.template_scene)

    try:
        return sdf_nodes.ensure_generated_scene(settings, create=create)
    except Exception as exc:
        message = str(exc)
        if not strict and message in {
            "Enable Use Nodes to create a scene SDF graph",
            "Scene SDF graph is unavailable",
        }:
            return sdf_nodes.invalid_scene_path()
        set_last_error(message)
        if strict:
            raise
        return sdf_nodes.invalid_scene_path()


def camera_target(camera):
    origin = camera.matrix_world.translation
    forward = -(camera.matrix_world.to_3x3().col[2])
    target = origin + forward
    up = camera.matrix_world.to_3x3().col[1]
    return (
        blender_to_renderer_vec(origin),
        blender_to_renderer_vec(target),
        blender_to_renderer_vec(up),
    )


def view_camera(context, scene):
    region3d = getattr(context.space_data, "region_3d", None)
    if region3d is None:
        raise RuntimeError("Viewport region data is unavailable")

    view_inv = region3d.view_matrix.inverted()
    origin = view_inv.translation
    forward = -(view_inv.to_3x3().col[2])
    up = view_inv.to_3x3().col[1]

    if region3d.view_perspective == "CAMERA" and scene.camera is not None:
        camera = scene.camera.data
        if camera.type != "PERSP":
            raise RuntimeError("Orthographic camera preview is not supported yet")
        fov_y = float(camera.angle_y)
        target = origin + forward
    else:
        if not region3d.is_perspective:
            raise RuntimeError("Orthographic viewport preview is not supported yet")
        proj_y = abs(float(region3d.window_matrix[1][1]))
        fov_y = 2.0 * math.atan(1.0 / max(proj_y, 1e-6))
        target = origin + forward

    return (
        blender_to_renderer_vec(origin),
        blender_to_renderer_vec(target),
        blender_to_renderer_vec(up),
        float(fov_y),
    )


def camera_signature(camera_position, camera_target_value):
    values = tuple(camera_position) + tuple(camera_target_value)
    return tuple(round(float(value), 4) for value in values)


def force_redraw_viewports(context=None):
    if context is None:
        context = bpy.context
    try:
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
    except Exception:
        pass


def scene_content_token(scene_path: Path):
    scene_path = scene_path.resolve()
    cache_key = str(scene_path)
    generated_hash = runtime.generated_scene_path_hashes.get(cache_key)
    if generated_hash is not None:
        return ("hash", generated_hash)
    return ("mtime", scene_path.stat().st_mtime_ns)


def world_background_color(scene):
    world = getattr(scene, "world", None)
    if world is not None:
        try:
            return tuple(float(v) for v in world.color[:3])
        except Exception:
            pass
    return (0.05, 0.05, 0.05)


def count_nodes(node) -> int:
    if not isinstance(node, dict):
        return 0

    count = 0
    stack = [node]
    while stack:
        current = stack.pop()
        if not isinstance(current, dict):
            continue
        count += 1
        if current.get("nodeType") == "binaryOperator":
            stack.append(current.get("leftChild"))
            stack.append(current.get("rightChild"))
    return count


def load_scene_metadata(scene_path: Path):
    scene_path = scene_path.resolve()
    token = scene_content_token(scene_path)
    cache_key = str(scene_path)
    cached = runtime.scene_metadata_cache.get(cache_key)
    if cached and cached["token"] == token:
        return cached["data"]

    runtime.debug_log(f"Reading scene metadata from {scene_path}")
    payload = json.loads(scene_path.read_text(encoding="utf-8"))
    data = {
        "aabb_min": tuple(
            float(v) for v in payload.get("aabb_min", (-1.0, -1.0, -1.0))
        ),
        "aabb_max": tuple(float(v) for v in payload.get("aabb_max", (1.0, 1.0, 1.0))),
        "node_count": count_nodes(payload),
    }
    runtime.scene_metadata_cache[cache_key] = {"token": token, "data": data}
    return data


def load_scene_payload(scene_path: Path):
    scene_path = scene_path.resolve()
    token = scene_content_token(scene_path)
    cache_key = str(scene_path)
    cached = runtime.scene_payload_cache.get(cache_key)
    if cached and cached["token"] == token:
        return cached["data"]

    payload = json.loads(scene_path.read_text(encoding="utf-8"))
    runtime.scene_payload_cache[cache_key] = {
        "token": token,
        "data": payload,
    }
    return payload


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


def _primitive_local_corners(node):
    primitive_type = node.get("primitiveType")
    if primitive_type == "sphere":
        radius = float(node.get("radius", 0.0))
        half = Vector((radius, radius, radius))
    elif primitive_type == "box":
        sides = node.get("sides", (0.0, 0.0, 0.0))
        half = Vector((float(sides[0]), float(sides[1]), float(sides[2]))) * 0.5
    elif primitive_type == "cylinder":
        radius = float(node.get("radius", 0.0))
        half = Vector((radius, float(node.get("height", 0.0)) * 0.5, radius))
    elif primitive_type == "cone":
        radius = float(node.get("radius", 0.0))
        half = Vector((radius, float(node.get("height", 0.0)) * 0.5, radius))
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
        node, parent_world_to_local, visited = stack.pop()
        if not isinstance(node, dict):
            continue

        node_matrix = _json_matrix_to_world_to_local(
            node.get("matrix", _identity_matrix_3x4())
        )
        world_to_local = parent_world_to_local @ node_matrix
        node_id = id(node)
        node_type = node.get("nodeType")

        if visited:
            if node_type == "primitive":
                try:
                    local_to_world = world_to_local.inverted()
                except Exception:
                    bounds_by_id[node_id] = None
                    pad_by_id[node_id] = 0.0
                    continue
                world_points = [
                    local_to_world @ corner for corner in _primitive_local_corners(node)
                ]
                bounds_by_id[node_id] = (
                    tuple(min(point[i] for point in world_points) for i in range(3)),
                    tuple(max(point[i] for point in world_points) for i in range(3)),
                )
                pad_by_id[node_id] = 0.0
            elif node_type == "binaryOperator":
                left = node.get("leftChild")
                right = node.get("rightChild")
                left_bounds = bounds_by_id.get(id(left))
                right_bounds = bounds_by_id.get(id(right))
                bounds_by_id[node_id] = _union_bounds(left_bounds, right_bounds)
                local_pad = 0.0
                if node.get("blendMode") == "union":
                    local_pad = max(0.0, float(node.get("blendRadius", 0.0))) * 0.25
                pad_by_id[node_id] = max(
                    local_pad,
                    pad_by_id.get(id(left), 0.0),
                    pad_by_id.get(id(right), 0.0),
                )
            else:
                bounds_by_id[node_id] = None
                pad_by_id[node_id] = 0.0
            continue

        stack.append((node, parent_world_to_local, True))
        if node_type == "binaryOperator":
            stack.append((node.get("rightChild", {}), world_to_local, False))
            stack.append((node.get("leftChild", {}), world_to_local, False))

    root_id = id(payload)
    return _inflate_bounds(bounds_by_id.get(root_id), pad_by_id.get(root_id, 0.0))


def load_scene_tight_bounds(scene_path: Path):
    scene_path = scene_path.resolve()
    token = scene_content_token(scene_path)
    cache_key = str(scene_path)
    cached = runtime.scene_bounds_cache.get(cache_key)
    if cached and cached["token"] == token:
        return cached["data"]

    payload = load_scene_payload(scene_path)
    bounds = _payload_bounds(payload)
    if bounds is None:
        metadata = load_scene_metadata(scene_path)
        bounds = (metadata["aabb_min"], metadata["aabb_max"])
    runtime.scene_bounds_cache[cache_key] = {
        "token": token,
        "data": bounds,
    }
    return bounds


def safe_scene_metadata(scene_path: Path):
    if not scene_path.is_file():
        return None
    try:
        return load_scene_metadata(scene_path)
    except Exception:
        return None


def _expand_bounds(bounds, padding_ratio):
    aabb_min, aabb_max = bounds
    extents = [max(float(b) - float(a), 1e-4) for a, b in zip(aabb_min, aabb_max)]
    padding = [max(ext * padding_ratio, 1e-4) for ext in extents]
    return (
        tuple(float(a) - pad for a, pad in zip(aabb_min, padding)),
        tuple(float(b) + pad for b, pad in zip(aabb_max, padding)),
    )


def _contains_bounds(outer, inner):
    outer_min, outer_max = outer
    inner_min, inner_max = inner
    return all(
        o_min <= i_min and o_max >= i_max
        for o_min, o_max, i_min, i_max in zip(
            outer_min, outer_max, inner_min, inner_max
        )
    )


def _bounds_extent(bounds):
    aabb_min, aabb_max = bounds
    return tuple(max(float(b) - float(a), 0.0) for a, b in zip(aabb_min, aabb_max))


def _should_shrink_bounds(current_bounds, target_bounds):
    current_extent = _bounds_extent(current_bounds)
    target_extent = _bounds_extent(target_bounds)
    return any(
        curr > targ * _DYNAMIC_AABB_SHRINK_THRESHOLD
        for curr, targ in zip(current_extent, target_extent)
    )


def _effective_target_bounds(scene_path: Path, metadata, settings):
    authored_bounds = (tuple(metadata["aabb_min"]), tuple(metadata["aabb_max"]))
    tight_bounds = _union_bounds(load_scene_tight_bounds(scene_path), authored_bounds)
    if runtime.demo_anim_running:
        anim_time = demo_anim_time(settings)
        if anim_time is not None:
            center = _demo_anim_center(
                metadata["aabb_min"], metadata["aabb_max"], anim_time
            )
            radius = 0.2
            sphere_bounds = (
                (center.x - radius, center.y - radius, center.z - radius),
                (center.x + radius, center.y + radius, center.z + radius),
            )
            return _union_bounds(tight_bounds, sphere_bounds)
    return tight_bounds


def _dynamic_aabb(scene_path: Path, metadata, settings):
    state_key = (str(scene_path.resolve()), scene_content_token(scene_path))
    now = time.perf_counter()
    target_bounds = _effective_target_bounds(scene_path, metadata, settings)
    target_bounds = _expand_bounds(target_bounds, _DYNAMIC_AABB_SHRINK_PAD)
    state = runtime.dynamic_aabb_state.get("state")
    if state is None or state.get("key") != state_key:
        state = {
            "key": state_key,
            "bounds": target_bounds,
            "last_shrink_time": now,
        }
        runtime.dynamic_aabb_state["state"] = state
        runtime.debug_log(
            "Dynamic AABB reset: "
            f"min={tuple(round(v, 3) for v in target_bounds[0])}, "
            f"max={tuple(round(v, 3) for v in target_bounds[1])}"
        )
        return target_bounds

    current_bounds = state["bounds"]
    if not _contains_bounds(current_bounds, target_bounds):
        grown = _expand_bounds(
            _union_bounds(current_bounds, target_bounds), _DYNAMIC_AABB_GROW_PAD
        )
        state["bounds"] = grown
        state["last_shrink_time"] = now
        runtime.debug_log(
            "Dynamic AABB grew: "
            f"min={tuple(round(v, 3) for v in grown[0])}, "
            f"max={tuple(round(v, 3) for v in grown[1])}"
        )
        return grown

    if (now - state["last_shrink_time"]) >= _DYNAMIC_AABB_SHRINK_INTERVAL:
        if _should_shrink_bounds(current_bounds, target_bounds):
            state["bounds"] = target_bounds
            runtime.debug_log(
                "Dynamic AABB shrank: "
                f"min={tuple(round(v, 3) for v in target_bounds[0])}, "
                f"max={tuple(round(v, 3) for v in target_bounds[1])}"
            )
        state["last_shrink_time"] = now

    return state["bounds"]


def _identity_matrix_3x4():
    return [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def demo_anim_time(settings) -> float | None:
    if not runtime.demo_anim_running:
        return None
    elapsed = max(0.0, time.perf_counter() - runtime.demo_anim_start_time)
    return elapsed * float(getattr(settings, "demo_anim_speed", 4.0))


def demo_anim_frame_key(settings) -> int | None:
    anim_time = demo_anim_time(settings)
    if anim_time is None:
        return None
    return int(anim_time * 60.0)


def _demo_anim_center(aabb_min, aabb_max, anim_time: float):
    radius = 0.2
    center = Vector(
        (
            math.cos(anim_time),
            math.cos(anim_time * 0.3),
            math.sin(anim_time),
        )
    )
    center *= math.sin(anim_time * 0.56 + 123.4)
    scale = Vector(aabb_max) - Vector(aabb_min) - Vector((2.0 * radius,) * 3)
    return (
        Vector(aabb_min)
        + Vector((radius,) * 3)
        + (center * 0.5 + Vector((0.5,) * 3)) * scale
    )


def shading_mode_value(native_module, shading_mode):
    return {
        "SHADED": native_module.SHADING_MODE_SHADED,
        "HEATMAP": native_module.SHADING_MODE_HEATMAP,
        "NORMALS": native_module.SHADING_MODE_NORMALS,
        "AO": native_module.SHADING_MODE_AO,
    }[shading_mode]


def close_renderer():
    if runtime.renderer is not None:
        runtime.debug_log("Releasing native renderer")
        try:
            runtime.renderer.close()
        except Exception as exc:
            runtime.debug_log(f"Renderer close failed: {exc}")
    runtime.renderer = None
    runtime.renderer_key = None
    runtime.loaded_scene_key = None
    runtime.pruning_cache_key = None
    gc.collect()


def get_renderer(settings, width, height):
    native_module = load_native_module()
    key = (
        shader_dir(settings),
        width,
        height,
        grid_level(settings),
        settings.shading_mode,
    )

    if runtime.renderer is None or runtime.renderer_key != key:
        if runtime.renderer is not None and runtime.renderer_key != key:
            close_renderer()
        runtime.debug_log(
            f"Creating native renderer {width}x{height}, grid={key[3]}, shading={settings.shading_mode}"
        )
        runtime.renderer = native_module.Renderer(
            key[0],
            width,
            height,
            key[3],
            shading_mode_value(native_module, settings.shading_mode),
            settings.culling_enabled,
            settings.num_samples,
            settings.gamma,
        )
        runtime.renderer_key = key
        runtime.loaded_scene_key = None
        runtime.pruning_cache_key = None
    else:
        runtime.debug_log("Reusing existing native renderer")
    return runtime.renderer


def ensure_scene_camera(scene):
    if scene.camera is not None:
        return scene.camera

    camera_data = bpy.data.cameras.new("MathOPS-v2 Camera")
    camera_object = bpy.data.objects.new("MathOPS-v2 Camera", camera_data)
    scene.collection.objects.link(camera_object)
    scene.camera = camera_object
    return camera_object


def effective_aabb(settings, scene_path: Path):
    metadata = load_scene_metadata(scene_path)
    if settings.use_scene_bounds:
        if getattr(settings, "dynamic_aabb", True):
            aabb_min, aabb_max = _dynamic_aabb(scene_path, metadata, settings)
        else:
            aabb_min, aabb_max = _union_bounds(
                load_scene_tight_bounds(scene_path),
                (tuple(metadata["aabb_min"]), tuple(metadata["aabb_max"])),
            )
        runtime.current_effective_aabb = (aabb_min, aabb_max)
        return aabb_min, aabb_max, metadata
    aabb = (tuple(settings.aabb_min), tuple(settings.aabb_max))
    runtime.current_effective_aabb = aabb
    return aabb[0], aabb[1], metadata


def store_render_stats(settings, scene_path: Path, timings, scene_info):
    runtime.last_error_message = ""
    render_ms = float(timings.get("render_ms", 0.0))
    upload_ms = float(timings.get("upload_ms", 0.0))
    runtime.last_render_stats.update(
        {
            "scene_name": scene_path.stem,
            "scene_path": str(scene_path),
            "node_count": int(scene_info.get("node_count", 0)),
            "render_ms": render_ms,
            "shader_ms": float(
                timings.get("shader_ms", timings.get("tracing_ms", 0.0))
            ),
            "upload_ms": upload_ms,
            "frame_ms": float(timings.get("frame_ms", render_ms + upload_ms)),
            "tracing_ms": float(timings.get("tracing_ms", 0.0)),
            "culling_ms": float(timings.get("culling_ms", 0.0)),
            "eval_grid_ms": float(timings.get("eval_grid_ms", 0.0)),
            "pruning_mem_gb": float(timings.get("pruning_mem_gb", 0.0)),
            "tracing_mem_gb": float(timings.get("tracing_mem_gb", 0.0)),
            "active_ratio": float(timings.get("active_ratio", 0.0)),
            "tmp_ratio": float(timings.get("tmp_ratio", 0.0)),
            "background_color": runtime.last_render_stats.get(
                "background_color", (0.05, 0.05, 0.05)
            ),
            "background_alpha": runtime.last_render_stats.get("background_alpha", 1.0),
        }
    )
    runtime.debug_log(
        f"Render finished for {scene_path.name}: render={runtime.last_render_stats['render_ms']:.2f}ms, "
        f"tracing={runtime.last_render_stats['tracing_ms']:.2f}ms, culling={runtime.last_render_stats['culling_ms']:.2f}ms"
    )


def render_rgba(
    scene,
    width=None,
    height=None,
    camera_position=None,
    camera_target_value=None,
    camera_up=None,
    fov_y=None,
    background_alpha=1.0,
    background_color=None,
    interactive=False,
):
    settings = scene.mathops_v2_settings
    scene_path = resolve_scene_path(settings, strict=True, create=True)
    if not scene_path.is_file():
        if runtime.last_error_message:
            raise RuntimeError(runtime.last_error_message)
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    if (
        camera_position is None
        or camera_target_value is None
        or camera_up is None
        or fov_y is None
    ):
        camera = ensure_scene_camera(scene)
        camera_position, camera_target_value, camera_up = camera_target(camera)
        fov_y = (
            float(camera.data.angle_y) if camera.data.type == "PERSP" else 1.57079632679
        )

    if width is None or height is None:
        width, height = get_render_size(scene)

    if background_color is None:
        background_color = world_background_color(scene)

    runtime.last_render_stats["background_color"] = tuple(
        float(v) for v in background_color
    )
    runtime.last_render_stats["background_alpha"] = float(background_alpha)

    runtime.debug_log(
        f"Starting render: scene={scene_path.name}, size={width}x{height}, "
        f"camera={tuple(round(v, 3) for v in camera_position)}, "
        f"target={tuple(round(v, 3) for v in camera_target_value)}, "
        f"up={tuple(round(v, 3) for v in camera_up)}, fov_y={float(fov_y):.3f}"
    )
    renderer = get_renderer(settings, width, height)

    scene_key = (str(scene_path.resolve()), scene_content_token(scene_path))
    if runtime.loaded_scene_key != scene_key:
        runtime.debug_log(f"Uploading scene data from {scene_path.name}")
        renderer.load_scene_file(str(scene_path))
        runtime.loaded_scene_key = scene_key
        runtime.pruning_cache_key = None

    aabb_min, aabb_max, metadata = effective_aabb(settings, scene_path)
    renderer.set_aabb(list(aabb_min), list(aabb_max))

    pruning_cache_key = (
        runtime.renderer_key,
        runtime.loaded_scene_key,
        tuple(round(float(v), 6) for v in aabb_min),
        tuple(round(float(v), 6) for v in aabb_max),
        bool(settings.culling_enabled),
    )
    should_recompute_pruning = settings.culling_enabled and (
        runtime.pruning_cache_key != pruning_cache_key
    )
    if should_recompute_pruning:
        runtime.debug_log("Recomputing pruning hierarchy")
    elif settings.culling_enabled:
        runtime.debug_log("Reusing cached pruning hierarchy")

    renderer.configure(
        settings.culling_enabled,
        should_recompute_pruning,
        settings.num_samples,
        settings.gamma,
        settings.colormap_max,
        list(runtime.last_render_stats.get("background_color", (0.05, 0.05, 0.05))),
        float(runtime.last_render_stats.get("background_alpha", 1.0)),
    )

    render_start = time.perf_counter()
    rgba = renderer.render_rgba(
        list(camera_position),
        list(camera_target_value),
        list(camera_up),
        float(fov_y),
        interactive,
    )
    render_elapsed_ms = (time.perf_counter() - render_start) * 1000.0
    if settings.culling_enabled:
        runtime.pruning_cache_key = pruning_cache_key
    timings = dict(renderer.last_timings())
    if interactive:
        timings["render_ms"] = render_elapsed_ms
        timings["shader_ms"] = render_elapsed_ms
        timings["frame_ms"] = render_elapsed_ms
        timings["tracing_ms"] = render_elapsed_ms
        timings["culling_ms"] = 0.0
        timings["eval_grid_ms"] = 0.0
    else:
        timings["shader_ms"] = float(timings.get("tracing_ms", 0.0))
        timings.setdefault("frame_ms", render_elapsed_ms)
    scene_info = dict(renderer.scene_info())
    scene_info.setdefault("node_count", metadata["node_count"])
    store_render_stats(settings, scene_path, timings, scene_info)
    return rgba, width, height, scene_path


def rgba_to_pixels(rgba, width, height):
    pixels = np.frombuffer(rgba, dtype=np.uint8).astype(np.float32) / 255.0
    return pixels.reshape((height, width, 4))[::-1].copy()


def set_last_error(message: str):
    runtime.last_error_message = message
    runtime.debug_log(f"Error: {message}")


def register_compat_panels():
    runtime.compat_panels = []
    for panel_name in runtime.COMPAT_PANEL_NAMES:
        panel = getattr(bpy.types, panel_name, None)
        if panel is not None and hasattr(panel, "COMPAT_ENGINES"):
            panel.COMPAT_ENGINES.add(runtime.ENGINE_ID)
            runtime.compat_panels.append(panel)


def unregister_compat_panels():
    for panel in runtime.compat_panels:
        panel.COMPAT_ENGINES.discard(runtime.ENGINE_ID)
    runtime.compat_panels = []
