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
from mathutils import Vector

from .. import runtime


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


def resolve_scene_path(settings) -> Path:
    if settings.scene_source == "TEMPLATE":
        return template_scene_path(settings.template_scene)
    return Path(bpy.path.abspath(settings.scene_path))


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
    stat = scene_path.stat()
    cache_key = str(scene_path)
    cached = runtime.scene_metadata_cache.get(cache_key)
    if cached and cached["mtime"] == stat.st_mtime_ns:
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
    runtime.scene_metadata_cache[cache_key] = {"mtime": stat.st_mtime_ns, "data": data}
    return data


def safe_scene_metadata(scene_path: Path):
    if not scene_path.is_file():
        return None
    try:
        return load_scene_metadata(scene_path)
    except Exception:
        return None


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


def frame_camera_to_aabb(scene, aabb_min, aabb_max):
    camera = ensure_scene_camera(scene)
    aabb_min_blender, aabb_max_blender = renderer_aabb_to_blender(aabb_min, aabb_max)
    center = (aabb_min_blender + aabb_max_blender) * 0.5
    extent = aabb_max_blender - aabb_min_blender
    radius = max(extent.length * 0.5, max(extent.x, extent.y, extent.z) * 0.75, 1.0)
    direction = Vector((1.65, -1.2, 1.1)).normalized()
    camera.location = center + direction * max(radius * 2.75, 2.5)
    camera.rotation_euler = (
        (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    )
    return camera


def sync_scene_metadata(settings, scene_path: Path):
    metadata = load_scene_metadata(scene_path)
    settings.aabb_min = metadata["aabb_min"]
    settings.aabb_max = metadata["aabb_max"]
    settings.last_node_count = metadata["node_count"]
    settings.last_scene_name = scene_path.stem
    settings.last_scene_path = str(scene_path)
    settings.last_error = ""
    runtime.last_error_message = ""
    runtime.last_render_stats["scene_name"] = scene_path.stem
    runtime.last_render_stats["scene_path"] = str(scene_path)
    runtime.last_render_stats["node_count"] = metadata["node_count"]
    runtime.debug_log(
        f"Scene ready: {scene_path.name}, nodes={metadata['node_count']}, "
        f"aabb_min={tuple(round(v, 3) for v in metadata['aabb_min'])}, "
        f"aabb_max={tuple(round(v, 3) for v in metadata['aabb_max'])}"
    )
    return metadata


def effective_aabb(settings, scene_path: Path):
    metadata = load_scene_metadata(scene_path)
    if settings.use_scene_bounds:
        return metadata["aabb_min"], metadata["aabb_max"], metadata
    return tuple(settings.aabb_min), tuple(settings.aabb_max), metadata


def store_render_stats(settings, scene_path: Path, timings, scene_info):
    runtime.last_error_message = ""
    render_ms = float(timings.get("render_ms", 0.0))
    upload_ms = float(timings.get("upload_ms", 0.0))
    runtime.last_render_stats.update(
        {
            "scene_name": scene_path.stem,
            "scene_path": str(scene_path),
            "node_count": int(scene_info.get("node_count", settings.last_node_count)),
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
    collect_shader_stats=False,
):
    settings = scene.mathops_v2_settings
    scene_path = resolve_scene_path(settings)
    if not scene_path.is_file():
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

    scene_key = (str(scene_path.resolve()), scene_path.stat().st_mtime_ns)
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
        interactive and not collect_shader_stats,
    )
    render_elapsed_ms = (time.perf_counter() - render_start) * 1000.0
    if settings.culling_enabled:
        runtime.pruning_cache_key = pruning_cache_key
    timings = dict(renderer.last_timings())
    if interactive and not collect_shader_stats:
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


def set_last_error(settings, message: str):
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
