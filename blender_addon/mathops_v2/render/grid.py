import math
import struct

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from .. import runtime
from ..shaders import grid as grid_shader


SHOW_GRID = 1 << 0
SHOW_AXES = 1 << 1
AXIS_X = 1 << 2
AXIS_Y = 1 << 3
AXIS_Z = 1 << 4
PLANE_XY = 1 << 5
PLANE_XZ = 1 << 6
PLANE_YZ = 1 << 7

_THEME_CACHE = None
_THEME_CACHE_ID = None


def _flatten_matrix(mat):
    return [float(value) for column in mat.col for value in column]


def _color_shade(color, shade):
    shade_normalized = shade / 255.0
    return (
        max(0.0, min(1.0, color[0] + shade_normalized)),
        max(0.0, min(1.0, color[1] + shade_normalized)),
        max(0.0, min(1.0, color[2] + shade_normalized)),
        color[3] if len(color) > 3 else 1.0,
    )


def _color_blend_shade(color1, color2, blend_factor, shade):
    inv_blend = 1.0 - blend_factor
    blended = (
        color1[0] * inv_blend + color2[0] * blend_factor,
        color1[1] * inv_blend + color2[1] * blend_factor,
        color1[2] * inv_blend + color2[2] * blend_factor,
        color1[3] * inv_blend + color2[3] * blend_factor if len(color1) > 3 and len(color2) > 3 else 1.0,
    )
    return _color_shade(blended, shade)


def _view_type_from_rotation(view_matrix_inv):
    forward = -Vector((view_matrix_inv[0][2], view_matrix_inv[1][2], view_matrix_inv[2][2]))
    forward.normalize()
    threshold = 0.99999999
    if forward.z < -threshold:
        return ("TOP", True)
    if forward.z > threshold:
        return ("BOTTOM", True)
    if forward.y > threshold:
        return ("FRONT", True)
    if forward.y < -threshold:
        return ("BACK", True)
    if forward.x < -threshold:
        return ("RIGHT", True)
    if forward.x > threshold:
        return ("LEFT", True)
    return ("USER", False)


def _theme_colors():
    global _THEME_CACHE, _THEME_CACHE_ID
    theme = bpy.context.preferences.themes[0]
    theme_id = id(theme)
    if _THEME_CACHE is not None and _THEME_CACHE_ID == theme_id:
        return _THEME_CACHE

    _THEME_CACHE_ID = theme_id
    grid_base_raw = tuple(theme.view_3d.grid)
    grid_base = grid_base_raw if len(grid_base_raw) == 4 else grid_base_raw + (0.5,)
    try:
        background = tuple(theme.view_3d.space.gradients.high_gradient)
    except (AttributeError, TypeError):
        background = (0.2, 0.2, 0.2, 1.0)
    if len(background) == 3:
        background = background + (1.0,)
    axis_x_base = tuple(theme.user_interface.axis_x) + (1.0,) if len(theme.user_interface.axis_x) == 3 else tuple(theme.user_interface.axis_x)
    axis_y_base = tuple(theme.user_interface.axis_y) + (1.0,) if len(theme.user_interface.axis_y) == 3 else tuple(theme.user_interface.axis_y)
    axis_z_base = tuple(theme.user_interface.axis_z) + (1.0,) if len(theme.user_interface.axis_z) == 3 else tuple(theme.user_interface.axis_z)
    grid_color = _color_shade(grid_base, 10)
    background_sum = background[0] + background[1] + background[2]
    grid_sum = grid_color[0] + grid_color[1] + grid_color[2]
    grid_emphasis = _color_shade(grid_base, 30 if (grid_sum + 0.12) > background_sum else -10)
    x_axis_color = _color_blend_shade(grid_base, axis_x_base, 0.85, -20)
    y_axis_color = _color_blend_shade(grid_base, axis_y_base, 0.85, -20)
    z_axis_color = _color_blend_shade(grid_base, axis_z_base, 0.85, -20)
    min_grid_alpha = 0.4
    grid_color = (grid_color[0], grid_color[1], grid_color[2], max(grid_color[3], min_grid_alpha))
    grid_emphasis = (grid_emphasis[0], grid_emphasis[1], grid_emphasis[2], max(grid_emphasis[3], min_grid_alpha))
    _THEME_CACHE = (grid_color, grid_emphasis, x_axis_color, y_axis_color, z_axis_color)
    return _THEME_CACHE


class MathOPSV2GridRenderer:
    def __init__(self):
        self._shader = None
        self._batch = None
        self._ubo = None

    def free(self):
        self._shader = None
        self._batch = None
        self._ubo = None

    def _ensure_shader(self):
        if self._shader is not None:
            return self._shader
        interface = gpu.types.GPUStageInterfaceInfo("mathops_v2_grid_interface")
        interface.smooth("VEC2", "uvInterp")
        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.vertex_in(0, "VEC2", "position")
        shader_info.vertex_out(interface)
        shader_info.fragment_out(0, "VEC4", "FragColor")
        shader_info.typedef_source(grid_shader.UNIFORMS_SOURCE)
        shader_info.uniform_buf(0, "MathOPSV2GridParams", "params")
        shader_info.sampler(0, "FLOAT_2D", "sdfDepthTex")
        shader_info.depth_write("ANY")
        shader_info.vertex_source(grid_shader.VERTEX_SOURCE)
        shader_info.fragment_source(grid_shader.FRAGMENT_SOURCE)
        self._shader = gpu.shader.create_from_info(shader_info)
        return self._shader

    def _ensure_batch(self):
        if self._batch is not None:
            return self._batch
        shader = self._ensure_shader()
        self._batch = batch_for_shader(
            shader,
            "TRI_STRIP",
            {"position": ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))},
        )
        return self._batch

    def _plane_and_axes(self, view_matrix_inv, is_perspective, settings):
        show_floor = bool(getattr(settings, "show_floor", True))
        show_grid = bool(getattr(settings, "show_grid", True))
        show_axis_x = bool(getattr(settings, "show_axis_x", True))
        show_axis_y = bool(getattr(settings, "show_axis_y", True))
        show_axis_z = bool(getattr(settings, "show_axis_z", False))
        grid_flag = 0

        if is_perspective:
            if show_floor:
                grid_flag |= SHOW_GRID
            grid_flag |= PLANE_XY
            if show_axis_x or show_axis_y or show_axis_z:
                grid_flag |= SHOW_AXES
            if show_axis_x:
                grid_flag |= AXIS_X
            if show_axis_y:
                grid_flag |= AXIS_Y
            if show_axis_z:
                grid_flag |= AXIS_Z
            return (grid_flag, False)

        view_type, is_cardinal = _view_type_from_rotation(view_matrix_inv)
        if show_grid:
            grid_flag |= SHOW_GRID
        if show_axis_x or show_axis_y or show_axis_z:
            grid_flag |= SHOW_AXES
        if view_type in {"TOP", "BOTTOM"}:
            grid_flag |= PLANE_XY
            if show_axis_x:
                grid_flag |= AXIS_X
            if show_axis_y:
                grid_flag |= AXIS_Y
        elif view_type in {"FRONT", "BACK"}:
            grid_flag |= PLANE_XZ
            if show_axis_x:
                grid_flag |= AXIS_X
            if show_axis_z:
                grid_flag |= AXIS_Z
        elif view_type in {"LEFT", "RIGHT"}:
            grid_flag |= PLANE_YZ
            if show_axis_y:
                grid_flag |= AXIS_Y
            if show_axis_z:
                grid_flag |= AXIS_Z
        else:
            grid_flag |= PLANE_XY
            if show_axis_x:
                grid_flag |= AXIS_X
            if show_axis_y:
                grid_flag |= AXIS_Y
            if show_axis_z:
                grid_flag |= AXIS_Z
        return (grid_flag, is_cardinal)

    def draw(self, context, render_width, render_height, view_matrix, projection_matrix, depth_texture):
        if depth_texture is None:
            return
        settings = runtime.scene_settings(context.scene)
        if settings is None:
            return

        shader = self._ensure_shader()
        batch = self._ensure_batch()
        region3d = getattr(getattr(context, "space_data", None), "region_3d", None)
        if region3d is None:
            return

        view_matrix_inv = view_matrix.inverted()
        is_perspective = bool(region3d.is_perspective)
        grid_flag, is_cardinal = self._plane_and_axes(view_matrix_inv, is_perspective, settings)
        if (grid_flag & (SHOW_GRID | SHOW_AXES)) == 0:
            return

        view_inv_flat = _flatten_matrix(view_matrix_inv)
        proj_inv_flat = _flatten_matrix(projection_matrix.inverted())
        view_flat = _flatten_matrix(view_matrix)
        proj_flat = _flatten_matrix(projection_matrix)
        camera_position = view_matrix_inv.translation
        grid_color, grid_emphasis, x_axis_color, y_axis_color, z_axis_color = _theme_colors()
        grid_type = str(getattr(settings, "grid_type", "RECTANGULAR") or "RECTANGULAR")
        grid_level_frac = 0.0

        if grid_type == "RADIAL":
            base_grid_scale = float(getattr(settings, "grid_scale_radial", 1.0))
            subdivisions = max(1, int(getattr(settings, "grid_subdivisions_radial", 12)))
            if is_perspective:
                camera_height = max(abs(camera_position.z), 1e-6)
                ring_spacing = base_grid_scale * 8.0
                if ring_spacing > 0.0 and subdivisions > 1:
                    level = max(0.0, math.log(camera_height / ring_spacing) / math.log(float(subdivisions)))
                    level_int = int(level)
                    grid_level_frac = level - level_int
                    grid_scale = base_grid_scale * (subdivisions ** level_int)
                else:
                    grid_scale = base_grid_scale
            elif is_cardinal:
                view_distance = float(region3d.view_distance)
                if view_distance > 50.0:
                    scale_factor = min(1.0 + math.log(view_distance / 50.0) * 10.0, 100.0)
                else:
                    scale_factor = 1.0
                grid_scale = base_grid_scale * scale_factor
                if scale_factor > 1.0 and subdivisions > 1:
                    level = math.log(scale_factor) / math.log(float(subdivisions))
                    grid_level_frac = level - int(level)
            else:
                grid_scale = base_grid_scale
        else:
            base_grid_scale = float(getattr(settings, "grid_scale_rectangular", 1.0))
            subdivisions = max(1, int(getattr(settings, "grid_subdivisions_rectangular", 1)))
            if is_perspective:
                camera_height = max(abs(camera_position.z), 1e-6)
                step0 = base_grid_scale * subdivisions
                if step0 > 0.0 and subdivisions > 1:
                    level = max(0.0, math.log(camera_height / step0) / math.log(float(subdivisions)))
                    level_int = int(level)
                    grid_level_frac = level - level_int
                    grid_scale = step0 * (subdivisions ** level_int)
                else:
                    grid_scale = step0
            elif is_cardinal:
                view_distance = float(region3d.view_distance)
                step0 = base_grid_scale
                if view_distance > 0.0 and step0 > 0.0 and subdivisions > 1:
                    level = max(0.0, math.log(view_distance / step0) / math.log(float(subdivisions)))
                    level_int = int(level)
                    grid_level_frac = level - level_int
                    grid_scale = step0 * (subdivisions ** level_int)
                else:
                    grid_scale = step0
            else:
                view_distance = float(region3d.view_distance)
                step0 = base_grid_scale * subdivisions
                if view_distance > 0.0 and step0 > 0.0 and subdivisions > 1:
                    level = max(0.0, math.log(view_distance / step0) / math.log(float(subdivisions)))
                    level_int = int(level)
                    grid_level_frac = level - level_int
                    grid_scale = step0 * (subdivisions ** level_int)
                else:
                    grid_scale = step0

        clip_end = float(getattr(getattr(context, "space_data", None), "clip_end", 1000.0))
        if is_perspective:
            max_distance = clip_end
        else:
            view_distance = float(region3d.view_distance)
            max_distance = max(view_distance * 10.0, clip_end * 5.0)

        grid_data = struct.pack(
            "=16f16f16f16f4f4f4f4f4f4f4f4f4i4f",
            *view_inv_flat,
            *proj_inv_flat,
            *view_flat,
            *proj_flat,
            camera_position.x,
            camera_position.y,
            camera_position.z,
            1.0,
            float(render_width),
            float(render_height),
            0.0,
            0.0,
            grid_color[0],
            grid_color[1],
            grid_color[2],
            grid_color[3],
            grid_emphasis[0],
            grid_emphasis[1],
            grid_emphasis[2],
            grid_emphasis[3],
            x_axis_color[0],
            x_axis_color[1],
            x_axis_color[2],
            x_axis_color[3],
            y_axis_color[0],
            y_axis_color[1],
            y_axis_color[2],
            y_axis_color[3],
            z_axis_color[0],
            z_axis_color[1],
            z_axis_color[2],
            z_axis_color[3],
            float(max_distance),
            float(grid_scale),
            float(grid_level_frac),
            1.0 if (not is_perspective and is_cardinal) else 0.0,
            int(subdivisions),
            0 if is_perspective else 1,
            1 if grid_type == "RADIAL" else 0,
            int(grid_flag),
            0.0,
            0.0,
            0.0,
            0.0,
        )
        if self._ubo is None:
            self._ubo = gpu.types.GPUUniformBuf(grid_data)
        else:
            self._ubo.update(grid_data)

        shader.bind()
        shader.uniform_block("params", self._ubo)
        shader.uniform_sampler("sdfDepthTex", depth_texture)
        batch.draw(shader)
