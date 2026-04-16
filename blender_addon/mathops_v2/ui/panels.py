import bpy
from bpy.types import Panel

from .. import runtime, sdf_nodes
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
        scene_path = bridge.resolve_scene_path(settings)
        metadata = bridge.safe_scene_metadata(scene_path)

        layout.prop(settings, "use_sdf_nodes")
        if not settings.use_sdf_nodes:
            layout.prop(settings, "template_scene", text="Template")
        else:
            row = layout.row(align=True)
            row.prop_search(
                settings, "sdf_node_tree", bpy.data, "node_groups", text="Graph"
            )
            row.operator(
                sdf_nodes.MATHOPS_V2_OT_new_scene_sdf_tree.bl_idname,
                text="",
                icon="ADD",
            )
            layout.operator(
                sdf_nodes.MATHOPS_V2_OT_edit_scene_sdf_tree.bl_idname,
                text="Edit SDF Graph",
            )

        box = layout.box()
        box.label(text=f"Resolved Scene: {scene_path.name}")
        if not settings.use_sdf_nodes:
            box.label(
                text="Template scenes render through MathOPS-v2 only", icon="INFO"
            )
        elif settings.sdf_node_tree is None:
            box.label(text="Enable Use Nodes to create the scene graph", icon="INFO")
        else:
            box.label(
                text=f"Source Tree: {settings.sdf_node_tree.name}", icon="NODETREE"
            )
        if metadata is None:
            box.label(
                text=runtime.last_error_message or "Scene metadata unavailable",
                icon="ERROR",
            )
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
        col.prop(settings, "culling_enabled")
        col.prop(settings, "final_grid_level")
        col.prop(settings, "num_samples")
        col.prop(settings, "viewport_max_dim")
        col.prop(settings, "gamma")


class MATHOPS_V2_PT_viewport(_MathOPSV2Panel, Panel):
    bl_label = "Viewport"
    bl_parent_id = "MATHOPS_V2_PT_render_settings"
    bl_idname = "MATHOPS_V2_PT_viewport"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        settings = context.scene.mathops_v2_settings
        col = layout.column()
        col.prop(settings, "viewport_preview")
        col.prop(settings, "shading_mode")
        if settings.shading_mode == "HEATMAP":
            col.prop(settings, "colormap_max")


class MATHOPS_V2_PT_bounds(_MathOPSV2Panel, Panel):
    bl_label = "Bounds"
    bl_parent_id = "MATHOPS_V2_PT_render_settings"
    bl_idname = "MATHOPS_V2_PT_bounds"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        settings = context.scene.mathops_v2_settings
        col = layout.column()
        col.prop(settings, "use_scene_bounds")
        if settings.use_scene_bounds:
            col.prop(settings, "dynamic_aabb")
        col.prop(settings, "show_aabb_overlay")
        if not settings.use_scene_bounds:
            col.prop(settings, "aabb_min")
            col.prop(settings, "aabb_max")


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
        col.prop(settings, "demo_anim_speed")
        row = col.row(align=True)
        if runtime.demo_anim_running:
            row.operator(
                operators.MATHOPS_V2_OT_stop_demo_anim.bl_idname, text="Stop Demo Anim"
            )
        else:
            row.operator(
                operators.MATHOPS_V2_OT_start_demo_anim.bl_idname, text="Play Demo Anim"
            )
        col.label(text="Viewport-only debug sphere", icon="INFO")


classes = (
    MATHOPS_V2_PT_render_settings,
    MATHOPS_V2_PT_scene,
    MATHOPS_V2_PT_quality,
    MATHOPS_V2_PT_viewport,
    MATHOPS_V2_PT_bounds,
    MATHOPS_V2_PT_debug,
)


def register():
    for cls in classes:
        import bpy

        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        import bpy

        bpy.utils.unregister_class(cls)
