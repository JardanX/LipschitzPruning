from bpy.types import Panel

from .. import runtime
from .. import operators
from ..render import bridge


class _MathOPSV2Panel:
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    @classmethod
    def poll(cls, context):
        return context.engine == runtime.ENGINE_ID


class MATHOPS_V2_PT_render_settings(_MathOPSV2Panel, Panel):
    bl_label = "MathOPS-v2"
    bl_idname = "MATHOPS_V2_PT_render_settings"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False


class MATHOPS_V2_PT_scene(_MathOPSV2Panel, Panel):
    bl_label = "Scene"
    bl_parent_id = "MATHOPS_V2_PT_render_settings"
    bl_idname = "MATHOPS_V2_PT_scene"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = context.scene.mathops_v2_settings

        layout.prop(settings, "scene_source", expand=True)

        if settings.scene_source == "TEMPLATE":
            row = layout.row(align=True)
            for template_id, label, _filename, _description in runtime.TEMPLATE_SCENES:
                op = row.operator(
                    operators.MATHOPS_V2_OT_load_template_scene.bl_idname,
                    text=label,
                    depress=settings.template_scene == template_id,
                )
                op.template_id = template_id
                op.frame_camera = True
        else:
            layout.prop(settings, "scene_path")

        scene_path = bridge.resolve_scene_path(settings)
        metadata = bridge.safe_scene_metadata(scene_path)

        row = layout.row(align=True)
        row.operator(operators.MATHOPS_V2_OT_refresh_scene.bl_idname, text="Refresh")
        frame = row.operator(
            operators.MATHOPS_V2_OT_refresh_scene.bl_idname, text="Frame Camera"
        )
        frame.frame_camera = True

        row = layout.row(align=True)
        row.operator("render.render", text="Render Still", icon="RENDER_STILL")
        row.operator(
            operators.MATHOPS_V2_OT_render_image.bl_idname,
            text="Render to Image",
            icon="IMAGE_DATA",
        )

        box = layout.box()
        box.label(text=f"Resolved Scene: {scene_path.name}")
        box.label(text="Template JSON does not create Blender objects", icon="INFO")
        box.label(
            text="Use Rendered viewport, Render Still, or Render to Image",
            icon="RENDER_STILL",
        )
        if metadata is None:
            box.label(text="Scene metadata unavailable", icon="ERROR")
        else:
            box.label(text=f"Nodes: {metadata['node_count']}")
            box.label(
                text=(
                    f"Bounds: ({metadata['aabb_min'][0]:.2f}, {metadata['aabb_min'][1]:.2f}, {metadata['aabb_min'][2]:.2f}) "
                    f"to ({metadata['aabb_max'][0]:.2f}, {metadata['aabb_max'][1]:.2f}, {metadata['aabb_max'][2]:.2f})"
                )
            )


class MATHOPS_V2_PT_quality(_MathOPSV2Panel, Panel):
    bl_label = "Quality"
    bl_parent_id = "MATHOPS_V2_PT_render_settings"
    bl_idname = "MATHOPS_V2_PT_quality"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        settings = context.scene.mathops_v2_settings
        col = layout.column()
        col.prop(settings, "final_grid_level")
        col.prop(settings, "num_samples")
        col.prop(settings, "gamma")


class MATHOPS_V2_PT_pruning(_MathOPSV2Panel, Panel):
    bl_label = "Pruning"
    bl_parent_id = "MATHOPS_V2_PT_render_settings"
    bl_idname = "MATHOPS_V2_PT_pruning"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        settings = context.scene.mathops_v2_settings
        col = layout.column()
        col.prop(settings, "culling_enabled")
        col.prop(settings, "recompute_pruning")


class MATHOPS_V2_PT_debug(_MathOPSV2Panel, Panel):
    bl_label = "Debug"
    bl_parent_id = "MATHOPS_V2_PT_render_settings"
    bl_idname = "MATHOPS_V2_PT_debug"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        settings = context.scene.mathops_v2_settings
        col = layout.column()
        col.prop(settings, "viewport_preview")
        col.prop(settings, "shading_mode")
        if settings.shading_mode == "HEATMAP":
            col.prop(settings, "colormap_max")
        col.prop(settings, "use_scene_bounds")
        if not settings.use_scene_bounds:
            col.prop(settings, "aabb_min")
            col.prop(settings, "aabb_max")
        col.prop(settings, "shader_dir")
        col.prop(settings, "image_name")


class MATHOPS_V2_PT_stats(_MathOPSV2Panel, Panel):
    bl_label = "Stats"
    bl_parent_id = "MATHOPS_V2_PT_render_settings"
    bl_idname = "MATHOPS_V2_PT_stats"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mathops_v2_settings
        stats = runtime.last_render_stats

        if runtime.last_error_message:
            box = layout.box()
            box.label(text=runtime.last_error_message, icon="ERROR")

        col = layout.column(align=True)
        col.label(
            text=f"Last Scene: {stats['scene_name'] or settings.last_scene_name or 'None'}"
        )
        col.label(text=f"Nodes: {stats['node_count'] or settings.last_node_count}")
        col.separator()
        col.label(text=f"Render: {stats['render_ms']:.2f} ms")
        col.label(text=f"Tracing: {stats['tracing_ms']:.2f} ms")
        col.label(text=f"Culling: {stats['culling_ms']:.2f} ms")
        col.label(text=f"Eval Grid: {stats['eval_grid_ms']:.2f} ms")
        col.separator()
        col.label(text=f"Pruning VRAM: {stats['pruning_mem_gb']:.3f} GB")
        col.label(text=f"Tracing VRAM: {stats['tracing_mem_gb']:.3f} GB")
        col.label(text=f"Active Ratio: {stats['active_ratio']:.2f}")
        col.label(text=f"Temp Ratio: {stats['tmp_ratio']:.2f}")


class MATHOPS_V2_PT_console(_MathOPSV2Panel, Panel):
    bl_label = "Console"
    bl_parent_id = "MATHOPS_V2_PT_render_settings"
    bl_idname = "MATHOPS_V2_PT_console"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mathops_v2_settings

        row = layout.row(align=True)
        row.prop(settings, "show_console", text="Show Log")
        row.operator(operators.MATHOPS_V2_OT_clear_console.bl_idname, text="Clear")

        if not settings.show_console:
            return

        box = layout.box()
        box.label(text=f"Engine: {context.scene.render.engine}")
        box.label(text=f"Scene Source: {settings.scene_source}")
        box.label(text=f"Resolved Path: {bridge.resolve_scene_path(settings)}")

        log_box = layout.box()
        if not runtime.debug_log_buffer:
            log_box.label(text="No log entries yet")
            return

        for line in runtime.debug_log_buffer[-12:]:
            log_box.label(text=line)


classes = (
    MATHOPS_V2_PT_render_settings,
    MATHOPS_V2_PT_scene,
    MATHOPS_V2_PT_quality,
    MATHOPS_V2_PT_pruning,
    MATHOPS_V2_PT_debug,
    MATHOPS_V2_PT_stats,
    MATHOPS_V2_PT_console,
)


def register():
    for cls in classes:
        import bpy

        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        import bpy

        bpy.utils.unregister_class(cls)
