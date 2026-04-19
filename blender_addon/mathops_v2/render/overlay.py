import bpy
from mathutils import Vector
from bpy.app.handlers import persistent

from .. import runtime


_OVERLAY_CACHE = {}
_MSGBUS_OWNER = object()
_TIMER_INTERVAL = 0.1


def use_native_ortho_grid(space) -> bool:
    region_3d = getattr(space, "region_3d", None)
    if region_3d is None or bool(getattr(region_3d, "is_perspective", True)):
        return False
    try:
        view_matrix_inv = region_3d.view_matrix.inverted()
    except Exception:
        return False
    forward = -Vector((view_matrix_inv[0][2], view_matrix_inv[1][2], view_matrix_inv[2][2]))
    if forward.length_squared == 0.0:
        return False
    forward.normalize()
    threshold = 0.99999999
    return (
        abs(forward.x) > threshold
        or abs(forward.y) > threshold
        or abs(forward.z) > threshold
    )


def _space_key(space):
    return runtime.safe_pointer(space) or id(space)


def _iter_viewports():
    window_manager = getattr(bpy.context, "window_manager", None)
    for window in getattr(window_manager, "windows", ()):
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        scene = getattr(window, "scene", None)
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    yield space, scene


def _cache_overlay_state(overlay):
    return {
        "show_floor": bool(getattr(overlay, "show_floor", False)),
        "show_ortho_grid": bool(getattr(overlay, "show_ortho_grid", False)),
        "show_axis_x": bool(getattr(overlay, "show_axis_x", False)),
        "show_axis_y": bool(getattr(overlay, "show_axis_y", False)),
        "show_axis_z": bool(getattr(overlay, "show_axis_z", False)),
        "grid_scale": float(getattr(overlay, "grid_scale", 1.0)),
        "grid_subdivisions": int(getattr(overlay, "grid_subdivisions", 1)),
    }


def _restore_overlay_state(space, state):
    overlay = getattr(space, "overlay", None)
    if overlay is None:
        return
    for key, value in state.items():
        try:
            setattr(overlay, key, value)
        except Exception:
            pass


def _overlay_is_suppressed(overlay):
    for prop_name in ("show_floor", "show_ortho_grid", "show_axis_x", "show_axis_y", "show_axis_z"):
        try:
            if bool(getattr(overlay, prop_name, False)):
                return False
        except Exception:
            return False
    return True


def _ensure_grid_defaults(scene, overlay):
    settings = runtime.scene_settings(scene)
    if settings is None or bool(getattr(settings, "grid_overlay_initialized", False)):
        return
    try:
        settings.show_floor = bool(getattr(overlay, "show_floor", True))
        settings.show_grid = bool(getattr(overlay, "show_ortho_grid", True))
        settings.show_axis_x = bool(getattr(overlay, "show_axis_x", True))
        settings.show_axis_y = bool(getattr(overlay, "show_axis_y", True))
        settings.show_axis_z = bool(getattr(overlay, "show_axis_z", False))
        settings.grid_scale_rectangular = float(getattr(overlay, "grid_scale", 1.0))
        settings.grid_subdivisions_rectangular = max(1, int(getattr(overlay, "grid_subdivisions", 1)))
        settings.grid_overlay_initialized = True
    except Exception:
        pass


def suppress_space_overlays(space, scene):
    if space is None or scene is None or getattr(scene.render, "engine", "") != runtime.ENGINE_ID:
        return
    if getattr(space, "type", "") != "VIEW_3D":
        return
    shading = getattr(space, "shading", None)
    overlay = getattr(space, "overlay", None)
    if shading is None or overlay is None or getattr(shading, "type", "") != "RENDERED":
        return

    key = _space_key(space)
    if key not in _OVERLAY_CACHE or not _overlay_is_suppressed(overlay):
        _OVERLAY_CACHE[key] = _cache_overlay_state(overlay)
    _ensure_grid_defaults(scene, overlay)

    settings = runtime.scene_settings(scene)
    if use_native_ortho_grid(space):
        if settings is not None:
            try:
                overlay.show_ortho_grid = bool(getattr(settings, "show_grid", True))
                overlay.show_axis_x = bool(getattr(settings, "show_axis_x", True))
                overlay.show_axis_y = bool(getattr(settings, "show_axis_y", True))
                overlay.show_axis_z = bool(getattr(settings, "show_axis_z", False))
            except Exception:
                pass
        return

    for prop_name in ("show_floor", "show_ortho_grid", "show_axis_x", "show_axis_y", "show_axis_z"):
        try:
            if getattr(overlay, prop_name, False):
                setattr(overlay, prop_name, False)
        except Exception:
            pass


def update_overlay_visibility():
    active_keys = set()
    for space, scene in _iter_viewports():
        key = _space_key(space)
        active_keys.add(key)
        if scene is not None and getattr(scene.render, "engine", "") == runtime.ENGINE_ID and getattr(getattr(space, "shading", None), "type", "") == "RENDERED":
            suppress_space_overlays(space, scene)
            continue
        cached = _OVERLAY_CACHE.pop(key, None)
        if cached is not None:
            _restore_overlay_state(space, cached)

    stale_keys = [key for key in _OVERLAY_CACHE.keys() if key not in active_keys]
    for key in stale_keys:
        _OVERLAY_CACHE.pop(key, None)


@persistent
def _on_depsgraph_update_post(_scene, _depsgraph):
    update_overlay_visibility()


def _notify_overlay_change():
    update_overlay_visibility()


def _overlay_poll_timer():
    try:
        update_overlay_visibility()
    except Exception:
        pass
    return _TIMER_INTERVAL


def register():
    if _on_depsgraph_update_post not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update_post)
    try:
        if not bpy.app.timers.is_registered(_overlay_poll_timer):
            bpy.app.timers.register(_overlay_poll_timer, first_interval=_TIMER_INTERVAL)
    except Exception:
        pass
    try:
        bpy.msgbus.subscribe_rna(
            key=(bpy.types.View3DShading, "type"),
            owner=_MSGBUS_OWNER,
            args=(),
            notify=_notify_overlay_change,
        )
        bpy.msgbus.subscribe_rna(
            key=(bpy.types.RenderSettings, "engine"),
            owner=_MSGBUS_OWNER,
            args=(),
            notify=_notify_overlay_change,
        )
    except Exception:
        pass
    update_overlay_visibility()


def unregister():
    try:
        bpy.msgbus.clear_by_owner(_MSGBUS_OWNER)
    except Exception:
        pass
    try:
        if bpy.app.timers.is_registered(_overlay_poll_timer):
            bpy.app.timers.unregister(_overlay_poll_timer)
    except Exception:
        pass
    if _on_depsgraph_update_post in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update_post)
    for space, _scene in _iter_viewports():
        cached = _OVERLAY_CACHE.pop(_space_key(space), None)
        if cached is not None:
            _restore_overlay_state(space, cached)
    _OVERLAY_CACHE.clear()
