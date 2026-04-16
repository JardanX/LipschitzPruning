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
    "RENDER_PT_post_processing",
    "RENDER_PT_stamp",
    "RENDER_PT_stamp_note",
    "VIEWLAYER_PT_layer",
)

native_module = None
renderer = None
renderer_key = None
loaded_scene_key = None
gpu_viewport_enabled = False
pruning_cache_key = None
scene_metadata_cache = {}
compat_panels = []
debug_log_buffer = []
last_error_message = ""
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


def debug_log(message: str):
    timestamp = time.strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(f"[MathOPS-v2] {line}")
    debug_log_buffer.append(line)
    if len(debug_log_buffer) > 80:
        del debug_log_buffer[:-80]


def clear_debug_log():
    debug_log_buffer.clear()


def reset_runtime():
    global \
        native_module, \
        renderer, \
        renderer_key, \
        loaded_scene_key, \
        gpu_viewport_enabled, \
        pruning_cache_key, \
        compat_panels, \
        last_error_message
    native_module = None
    renderer = None
    renderer_key = None
    loaded_scene_key = None
    gpu_viewport_enabled = False
    pruning_cache_key = None
    compat_panels = []
    scene_metadata_cache.clear()
    last_error_message = ""
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
