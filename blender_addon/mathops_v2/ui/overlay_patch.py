import bpy

from .. import properties, runtime


_ORIGINAL_GUIDES_DRAW = None


def _is_mathops_rendered(context):
    space = getattr(context, "space_data", None)
    shading = getattr(space, "shading", None)
    return (
        getattr(getattr(context, "scene", None), "render", None) is not None
        and getattr(context.scene.render, "engine", "") == runtime.ENGINE_ID
        and space is not None
        and getattr(space, "type", "") == "VIEW_3D"
        and shading is not None
        and getattr(shading, "type", "") == "RENDERED"
    )


def _draw_mathops_guides(self, context):
    properties.ensure_scene_defaults(context.scene, context)
    settings = runtime.scene_settings(context.scene)
    overlay = getattr(getattr(context, "space_data", None), "overlay", None)
    if settings is None or overlay is None:
        return

    layout = self.layout
    col = layout.column()
    col.active = bool(getattr(overlay, "show_overlays", True))

    row = col.row()
    row.prop(settings, "grid_type", expand=True)

    row = col.row(align=False)
    row.prop(settings, "show_grid", text="Grid")
    row.prop(settings, "show_floor", text="Floor")
    row.label(text="Axes")
    axes = row.row(align=True)
    axes.scale_x = 0.7
    axes.prop(settings, "show_axis_x", text="X", toggle=True)
    axes.prop(settings, "show_axis_y", text="Y", toggle=True)
    axes.prop(settings, "show_axis_z", text="Z", toggle=True)

    row = col.row(align=True)
    if getattr(settings, "grid_type", "RECTANGULAR") == "RADIAL":
        row.prop(settings, "grid_scale_radial", text="Scale")
        row.prop(settings, "grid_subdivisions_radial", text="Subdivisions")
    else:
        row.prop(settings, "grid_scale_rectangular", text="Scale")
        row.prop(settings, "grid_subdivisions_rectangular", text="Subdivisions")

    col.separator()
    col.label(text="MathOPS Outline")
    col.prop(settings, "outline_color", text="Color")
    col.prop(settings, "outline_opacity", text="Opacity", slider=True)
    col.label(text="Selected and active outlines use Blender theme colors", icon="INFO")


def _patched_guides_draw(self, context):
    if _is_mathops_rendered(context):
        _draw_mathops_guides(self, context)
        return
    if _ORIGINAL_GUIDES_DRAW is not None:
        _ORIGINAL_GUIDES_DRAW(self, context)


def register():
    global _ORIGINAL_GUIDES_DRAW
    try:
        from bl_ui.space_view3d import VIEW3D_PT_overlay_guides
    except Exception:
        return
    if _ORIGINAL_GUIDES_DRAW is None:
        _ORIGINAL_GUIDES_DRAW = VIEW3D_PT_overlay_guides.draw
        VIEW3D_PT_overlay_guides.draw = _patched_guides_draw


def unregister():
    global _ORIGINAL_GUIDES_DRAW
    if _ORIGINAL_GUIDES_DRAW is None:
        return
    try:
        from bl_ui.space_view3d import VIEW3D_PT_overlay_guides
        VIEW3D_PT_overlay_guides.draw = _ORIGINAL_GUIDES_DRAW
    except Exception:
        pass
    _ORIGINAL_GUIDES_DRAW = None
