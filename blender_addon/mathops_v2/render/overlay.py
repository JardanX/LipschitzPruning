import bpy
from bpy.app.handlers import persistent

from .. import runtime


_OVERLAY_CACHE = {}
_SUPPRESSED_KEYS = set()
_MSGBUS_OWNER = object()
_TIMER_INTERVAL = 0.1
_SAVE_RESTORE_ACTIVE = False


def use_native_ortho_grid(space) -> bool:
    region_3d = getattr(space, "region_3d", None)
    return region_3d is not None and not bool(getattr(region_3d, "is_perspective", True))


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


def _stored_overlay_state(scene):
    settings = runtime.scene_settings(scene)
    if settings is None:
        return None
    if bool(getattr(settings, "native_overlay_initialized", False)):
        return {
            "show_floor": bool(getattr(settings, "native_show_floor", True)),
            "show_ortho_grid": bool(getattr(settings, "native_show_ortho_grid", True)),
            "show_axis_x": bool(getattr(settings, "native_show_axis_x", True)),
            "show_axis_y": bool(getattr(settings, "native_show_axis_y", True)),
            "show_axis_z": bool(getattr(settings, "native_show_axis_z", False)),
            "grid_scale": float(getattr(settings, "native_grid_scale", 1.0)),
            "grid_subdivisions": max(1, int(getattr(settings, "native_grid_subdivisions", 1))),
        }
    return None


def _legacy_overlay_state(scene):
    settings = runtime.scene_settings(scene)
    if settings is None or not bool(getattr(settings, "grid_overlay_initialized", False)):
        return None
    return {
        "show_floor": bool(getattr(settings, "show_floor", True)),
        "show_ortho_grid": bool(getattr(settings, "show_grid", True)),
        "show_axis_x": bool(getattr(settings, "show_axis_x", True)),
        "show_axis_y": bool(getattr(settings, "show_axis_y", True)),
        "show_axis_z": bool(getattr(settings, "show_axis_z", False)),
        "grid_scale": float(getattr(settings, "grid_scale_rectangular", 1.0)),
        "grid_subdivisions": max(1, int(getattr(settings, "grid_subdivisions_rectangular", 1))),
    }


def _persist_overlay_state(scene, state):
    settings = runtime.scene_settings(scene)
    if settings is None or state is None:
        return
    try:
        settings.native_show_floor = bool(state.get("show_floor", True))
        settings.native_show_ortho_grid = bool(state.get("show_ortho_grid", True))
        settings.native_show_axis_x = bool(state.get("show_axis_x", True))
        settings.native_show_axis_y = bool(state.get("show_axis_y", True))
        settings.native_show_axis_z = bool(state.get("show_axis_z", False))
        settings.native_grid_scale = float(state.get("grid_scale", 1.0))
        settings.native_grid_subdivisions = max(1, int(state.get("grid_subdivisions", 1)))
        settings.native_overlay_initialized = True
    except Exception:
        pass


def _capture_current_overlay_state(scene, overlay):
    if overlay is None or _overlay_is_suppressed(overlay):
        return None
    state = _cache_overlay_state(overlay)
    _persist_overlay_state(scene, state)
    return state


def _restore_overlay_state(space, state):
    overlay = getattr(space, "overlay", None)
    if overlay is None:
        return False
    changed = False
    for key, value in state.items():
        try:
            if getattr(overlay, key, None) != value:
                setattr(overlay, key, value)
                changed = True
        except Exception:
            pass
    if changed:
        runtime.tag_redraw()
    return changed


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
    state = _stored_overlay_state(scene)
    if state is None:
        if overlay is None or _overlay_is_suppressed(overlay):
            return
        state = _cache_overlay_state(overlay)
    try:
        settings.show_floor = bool(state.get("show_floor", True))
        settings.show_grid = bool(state.get("show_ortho_grid", True))
        settings.show_axis_x = bool(state.get("show_axis_x", True))
        settings.show_axis_y = bool(state.get("show_axis_y", True))
        settings.show_axis_z = bool(state.get("show_axis_z", False))
        settings.grid_scale_rectangular = float(state.get("grid_scale", 1.0))
        settings.grid_subdivisions_rectangular = max(1, int(state.get("grid_subdivisions", 1)))
        settings.grid_overlay_initialized = True
    except Exception:
        pass


def _restore_space_from_state(space, scene):
    key = _space_key(space)
    state = _OVERLAY_CACHE.get(key) or _stored_overlay_state(scene)
    if state is None:
        overlay = getattr(space, "overlay", None)
        if overlay is not None and _overlay_is_suppressed(overlay):
            state = _legacy_overlay_state(scene)
    if state is None:
        return False
    _OVERLAY_CACHE[key] = dict(state)
    _restore_overlay_state(space, state)
    return True


def suppress_space_overlays(space, scene):
    if space is None or scene is None or getattr(scene.render, "engine", "") != runtime.ENGINE_ID:
        return
    if getattr(space, "type", "") != "VIEW_3D":
        return
    key = _space_key(space)
    if _SAVE_RESTORE_ACTIVE:
        _restore_space_from_state(space, scene)
        _SUPPRESSED_KEYS.discard(key)
        return
    shading = getattr(space, "shading", None)
    overlay = getattr(space, "overlay", None)
    if shading is None or overlay is None or getattr(shading, "type", "") != "RENDERED":
        return

    captured_state = _capture_current_overlay_state(scene, overlay)
    if captured_state is not None:
        _OVERLAY_CACHE[key] = captured_state
    elif key not in _OVERLAY_CACHE:
        stored_state = _stored_overlay_state(scene) or _legacy_overlay_state(scene)
        if stored_state is not None:
            _OVERLAY_CACHE[key] = stored_state
    _ensure_grid_defaults(scene, overlay)

    changed = False
    for prop_name in ("show_floor", "show_ortho_grid", "show_axis_x", "show_axis_y", "show_axis_z"):
        try:
            if getattr(overlay, prop_name, False):
                setattr(overlay, prop_name, False)
                changed = True
        except Exception:
            pass
    if changed:
        runtime.tag_redraw()
    _SUPPRESSED_KEYS.add(key)


def update_overlay_visibility():
    active_keys = set()
    for space, scene in _iter_viewports():
        key = _space_key(space)
        active_keys.add(key)
        overlay = getattr(space, "overlay", None)
        if _SAVE_RESTORE_ACTIVE:
            restored = _restore_space_from_state(space, scene)
            if restored:
                _SUPPRESSED_KEYS.discard(key)
            overlay = getattr(space, "overlay", None)
            if scene is not None and overlay is not None:
                captured_state = _capture_current_overlay_state(scene, overlay)
                if captured_state is not None:
                    _OVERLAY_CACHE[key] = captured_state
            continue
        if scene is not None and getattr(scene.render, "engine", "") == runtime.ENGINE_ID and getattr(getattr(space, "shading", None), "type", "") == "RENDERED":
            suppress_space_overlays(space, scene)
            continue
        if key in _SUPPRESSED_KEYS or (overlay is not None and _overlay_is_suppressed(overlay)):
            _restore_space_from_state(space, scene)
            overlay = getattr(space, "overlay", None)
        _SUPPRESSED_KEYS.discard(key)
        if scene is not None and overlay is not None:
            captured_state = _capture_current_overlay_state(scene, overlay)
            if captured_state is not None:
                _OVERLAY_CACHE[key] = captured_state

    stale_keys = [key for key in _OVERLAY_CACHE.keys() if key not in active_keys]
    for key in stale_keys:
        _OVERLAY_CACHE.pop(key, None)
        _SUPPRESSED_KEYS.discard(key)


@persistent
def _on_load_post(_dummy):
    _OVERLAY_CACHE.clear()
    _SUPPRESSED_KEYS.clear()
    _ensure_runtime_watchers()
    update_overlay_visibility()


@persistent
def _on_save_pre(_dummy):
    global _SAVE_RESTORE_ACTIVE
    _SAVE_RESTORE_ACTIVE = True
    update_overlay_visibility()


@persistent
def _on_save_post(_dummy):
    global _SAVE_RESTORE_ACTIVE
    _SAVE_RESTORE_ACTIVE = False
    update_overlay_visibility()


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


def _ensure_runtime_watchers():
    try:
        if not bpy.app.timers.is_registered(_overlay_poll_timer):
            bpy.app.timers.register(_overlay_poll_timer, first_interval=_TIMER_INTERVAL)
    except Exception:
        pass
    try:
        bpy.msgbus.clear_by_owner(_MSGBUS_OWNER)
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


def register():
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)
    if _on_save_pre not in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.append(_on_save_pre)
    if _on_save_post not in bpy.app.handlers.save_post:
        bpy.app.handlers.save_post.append(_on_save_post)
    if _on_depsgraph_update_post not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update_post)
    _ensure_runtime_watchers()
    update_overlay_visibility()


def unregister():
    global _SAVE_RESTORE_ACTIVE
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    if _on_save_pre in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.remove(_on_save_pre)
    if _on_save_post in bpy.app.handlers.save_post:
        bpy.app.handlers.save_post.remove(_on_save_post)
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
    for space, scene in _iter_viewports():
        state = _OVERLAY_CACHE.pop(_space_key(space), None) or _stored_overlay_state(scene)
        if state is not None:
            _restore_overlay_state(space, state)
    _OVERLAY_CACHE.clear()
    _SUPPRESSED_KEYS.clear()
    _SAVE_RESTORE_ACTIVE = False
