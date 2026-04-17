import hashlib
import struct


ENGINE_ID = "MATHOPS_V2"
TREE_IDNAME = "MathOPSV2NodeTree"
OUTPUT_NODE_IDNAME = "MathOPSV2OutputNode"
OBJECT_NODE_IDNAME = "MathOPSV2ObjectNode"
CSG_NODE_IDNAME = "MathOPSV2CSGNode"

MAX_STACK = 256
PRIMITIVE_TEXELS = 5

BACKGROUND_COLOR = (0.035, 0.04, 0.05, 1.0)

last_error_message = ""


def set_error(message: str) -> None:
    global last_error_message
    last_error_message = str(message or "")


def clear_error() -> None:
    global last_error_message
    last_error_message = ""


def scene_settings(scene):
    return getattr(scene, "mathops_v2", None)


def object_settings(obj):
    try:
        return getattr(obj, "mathops_v2_sdf", None)
    except ReferenceError:
        return None


def is_sdf_proxy(obj) -> bool:
    try:
        settings = object_settings(obj)
        return bool(obj and obj.type == "EMPTY" and settings and settings.enabled)
    except ReferenceError:
        return False


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


def hash_compiled_rows(primitive_rows, instruction_rows) -> str:
    digest = hashlib.sha1()
    digest.update(struct.pack("<II", len(primitive_rows), len(instruction_rows)))
    for row in primitive_rows:
        digest.update(struct.pack("<4f", *[float(value) for value in row]))
    for row in instruction_rows:
        digest.update(struct.pack("<4f", *[float(value) for value in row]))
    return digest.hexdigest()
