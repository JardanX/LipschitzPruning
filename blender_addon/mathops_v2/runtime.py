import time


ENGINE_ID = "MATHOPS_V2"

TEMPLATE_SCENES = (
    ("TREES", "Trees", "trees.json", "Bundled tree template"),
    ("MONUMENT", "Monument", "monument.json", "Bundled monument template"),
    ("MOLECULE", "Molecule", "molecule.json", "Bundled molecule template"),
)

TEMPLATE_FILES = {
    identifier: filename
    for identifier, _label, filename, _description in TEMPLATE_SCENES
}

COMPAT_PANEL_NAMES = (
    "RENDER_PT_context",
    "RENDER_PT_dimensions",
    "RENDER_PT_output",
    "RENDER_PT_color_management",
    "VIEWLAYER_PT_layer",
)

native_module = None
native_module_dll_dirs = []
renderer = None
renderer_key = None
loaded_scene_key = None
gpu_viewport_enabled = False
pruning_cache_key = None
scene_metadata_cache = {}
scene_payload_cache = {}
scene_bounds_cache = {}
generated_scene_cache = {}
generated_scene_path_hashes = {}
generated_scene_last_compile = {}
generated_scene_dirty = set()
scene_transform_dirty = set()
dynamic_aabb_state = {}
compat_panels = []
debug_log_buffer = []
last_error_message = ""
demo_anim_running = False
demo_anim_start_time = 0.0
demo_anim_timer_registered = False
current_effective_aabb = None
graph_interaction_time = 0.0
graph_sync_suppressed_until = 0.0
proxy_sync_suppressed_until = 0.0
last_render_stats = {
    "scene_name": "",
    "scene_path": "",
    "node_count": 0,
    "render_ms": 0.0,
    "shader_ms": 0.0,
    "upload_ms": 0.0,
    "frame_ms": 0.0,
    "tracing_ms": 0.0,
    "culling_ms": 0.0,
    "eval_grid_ms": 0.0,
    "pruning_mem_gb": 0.0,
    "tracing_mem_gb": 0.0,
    "active_ratio": 0.0,
    "tmp_ratio": 0.0,
    "background_color": (0.05, 0.05, 0.05),
    "background_alpha": 1.0,
}
SLOW_DEBUG_MS = 2.0


def debug_log(message: str):
    timestamp = time.strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(f"[MathOPS-v2] {line}")
    debug_log_buffer.append(line)
    if len(debug_log_buffer) > 80:
        del debug_log_buffer[:-80]


def debug_slow(message: str, duration_ms: float, threshold_ms: float = SLOW_DEBUG_MS):
    if float(duration_ms) < float(threshold_ms):
        return
    debug_log(f"{message}: {duration_ms:.2f}ms")


def clear_debug_log():
    debug_log_buffer.clear()


def reset_runtime():
    global \
        native_module, \
        native_module_dll_dirs, \
        renderer, \
        renderer_key, \
        loaded_scene_key, \
        gpu_viewport_enabled, \
        pruning_cache_key, \
        compat_panels, \
        last_error_message, \
        demo_anim_running, \
        demo_anim_start_time, \
        demo_anim_timer_registered, \
        current_effective_aabb, \
        graph_interaction_time, \
        graph_sync_suppressed_until, \
        proxy_sync_suppressed_until
    for handle in native_module_dll_dirs:
        try:
            handle.close()
        except Exception:
            pass
    native_module = None
    native_module_dll_dirs = []
    renderer = None
    renderer_key = None
    loaded_scene_key = None
    gpu_viewport_enabled = False
    pruning_cache_key = None
    compat_panels = []
    scene_metadata_cache.clear()
    scene_payload_cache.clear()
    scene_bounds_cache.clear()
    generated_scene_cache.clear()
    generated_scene_path_hashes.clear()
    generated_scene_last_compile.clear()
    generated_scene_dirty.clear()
    scene_transform_dirty.clear()
    dynamic_aabb_state.clear()
    last_error_message = ""
    demo_anim_running = False
    demo_anim_start_time = 0.0
    demo_anim_timer_registered = False
    current_effective_aabb = None
    graph_interaction_time = 0.0
    graph_sync_suppressed_until = 0.0
    proxy_sync_suppressed_until = 0.0
    last_render_stats.update(
        {
            "scene_name": "",
            "scene_path": "",
            "node_count": 0,
            "render_ms": 0.0,
            "shader_ms": 0.0,
            "upload_ms": 0.0,
            "frame_ms": 0.0,
            "tracing_ms": 0.0,
            "culling_ms": 0.0,
            "eval_grid_ms": 0.0,
            "pruning_mem_gb": 0.0,
            "tracing_mem_gb": 0.0,
            "active_ratio": 0.0,
            "tmp_ratio": 0.0,
            "background_color": (0.05, 0.05, 0.05),
            "background_alpha": 1.0,
        }
    )
    clear_debug_log()
