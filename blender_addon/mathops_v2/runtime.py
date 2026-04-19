import hashlib
import struct
import time


ENGINE_ID = "MATHOPS_V2"
TREE_IDNAME = "MathOPSV2NodeTree"
OUTPUT_NODE_IDNAME = "MathOPSV2OutputNode"
OBJECT_NODE_IDNAME = "MathOPSV2ObjectNode"
CSG_NODE_IDNAME = "MathOPSV2CSGNode"

MAX_STACK = 256
PRIMITIVE_TEXELS = 5

BACKGROUND_COLOR = (0.035, 0.04, 0.05, 1.0)

last_error_message = ""
graph_interaction_time = 0.0
scene_revisions = {}


def set_error(message: str) -> None:
    global last_error_message
    last_error_message = str(message or "")


def clear_error() -> None:
    global last_error_message
    last_error_message = ""


def note_interaction() -> None:
    global graph_interaction_time
    graph_interaction_time = time.perf_counter()


def _scene_revision_entry(scene):
    key = scene_key(scene)
    if key == 0:
        return None
    entry = scene_revisions.get(key)
    if entry is None:
        entry = {"static": 0, "transform": 0}
        scene_revisions[key] = entry
    return entry


def scene_identity(scene):
    original = getattr(scene, "original", None)
    original_key = safe_pointer(original)
    if original_key:
        return original
    return scene


def scene_key(scene) -> int:
    return safe_pointer(scene_identity(scene))


def scene_revision_tuple(scene):
    entry = _scene_revision_entry(scene)
    if entry is None:
        return (0, 0)
    return (int(entry["static"]), int(entry["transform"]))


def mark_scene_static_dirty(scene) -> None:
    entry = _scene_revision_entry(scene)
    if entry is None:
        return
    entry["static"] += 1
    entry["transform"] += 1


def mark_scene_transform_dirty(scene) -> None:
    entry = _scene_revision_entry(scene)
    if entry is None:
        return
    entry["transform"] += 1


def interaction_active(grace_period: float) -> bool:
    if grace_period <= 0.0:
        return False
    return (time.perf_counter() - graph_interaction_time) < grace_period


def scene_settings(scene):
    return getattr(scene_identity(scene), "mathops_v2", None)


def scene_background_color(scene):
    scene = scene_identity(scene)
    color = BACKGROUND_COLOR
    try:
        world = getattr(scene, "world", None)
        world_color = getattr(world, "color", None)
        if world_color is not None and len(world_color) >= 3:
            color = (float(world_color[0]), float(world_color[1]), float(world_color[2]), 1.0)
    except Exception:
        pass
    return color


def object_settings(obj):
    obj = object_identity(obj)
    try:
        return getattr(obj, "mathops_v2_sdf", None)
    except ReferenceError:
        return None


def is_sdf_proxy(obj) -> bool:
    try:
        obj = object_identity(obj)
        settings = object_settings(obj)
        return bool(obj and obj.type == "EMPTY" and settings and settings.enabled)
    except ReferenceError:
        return False


def object_identity(obj):
    original = getattr(obj, "original", None)
    original_key = safe_pointer(original)
    if original_key:
        return original
    return obj


def object_key(obj) -> int:
    return safe_pointer(object_identity(obj))


def safe_pointer(rna) -> int:
    try:
        return 0 if rna is None else int(rna.as_pointer())
    except ReferenceError:
        return 0


def normalize3(values):
    x = float(values[0])
    y = float(values[1])
    z = float(values[2])
    length_sq = (x * x) + (y * y) + (z * z)
    if length_sq <= 1.0e-12:
        return (0.57735, 0.57735, 0.57735)
    inv_length = length_sq ** -0.5
    return (x * inv_length, y * inv_length, z * inv_length)


def tag_redraw(context=None) -> None:
    try:
        import bpy
    except Exception:
        return

    windows = []
    if context is not None:
        window = getattr(context, "window", None)
        if window is not None:
            windows.append(window)
    if not windows:
        windows.extend(getattr(bpy.context.window_manager, "windows", ()))

    for window in windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        for area in screen.areas:
            if area.type in {"VIEW_3D", "NODE_EDITOR", "PROPERTIES"}:
                area.tag_redraw()


def hash_compiled_rows(*row_blocks) -> str:
    digest = hashlib.sha1()
    digest.update(struct.pack("<I", len(row_blocks)))
    for rows in row_blocks:
        digest.update(struct.pack("<I", len(rows)))
        for row in rows:
            digest.update(struct.pack("<4f", *[float(value) for value in row]))
    return digest.hexdigest()


def hash_instruction_rows(instruction_rows) -> str:
    digest = hashlib.sha1()
    digest.update(struct.pack("<I", len(instruction_rows)))
    for row in instruction_rows:
        digest.update(struct.pack("<4f", *[float(value) for value in row]))
    return digest.hexdigest()
