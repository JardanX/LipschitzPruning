import math
import time

import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from .. import runtime
from ..nodes import sdf_tree
from ..shaders import pruning as pruning_shader
from ..shaders import raymarch
from . import grid, matcap, mesh_renderer, overlay, pruning


_COUNTERS_ACTIVE_BASE = 0
_COUNTERS_TMP_BASE = 10
_COUNTER_STATUS = 20
_COUNTERS_SIZE = 21
_FLOAT32_EXACT_UINT_LIMIT = 1 << 24
_TEXTURE_DATA_WIDTH = 4096
_MAX_ACTIVE_CAP = 100_000_000
_MAX_TMP_CAP = 400_000_000
_OUTLINE_ID_STRIDE = 1 << 20
_OUTLINE_COLOR_SELECTED = 1
_OUTLINE_COLOR_DEFAULT = 2
_OUTLINE_COLOR_ACTIVE = 3


class MathOPSV2GPUViewport:
    def __init__(self):
        self._draw_shader = None
        self._normal_shader = None
        self._blit_shader = None
        self._outline_shader = None
        self._compute_shader = None
        self._batch = None
        self._params_ubo = None
        self._scene_texture = None
        self._polygon_texture = None
        self._scene_hash = ""
        self._polygon_hash = ""
        self._outline_data_texture = None
        self._outline_signature = None
        self._offscreen = None
        self._normal_offscreen = None
        self._offscreen_color_texture = None
        self._offscreen_outline_texture = None
        self._offscreen_position_texture = None
        self._offscreen_normal_texture = None
        self._offscreen_depth_texture = None
        self._compiled_scene = None
        self._compiled_scene_scene_key = 0
        self._compiled_scene_static_revision = -1
        self._compiled_scene_transform_revision = -1
        self._compiled_scene_dynamic_signature = None
        self._dummy_scalar_texture = None
        self._grid_renderer = grid.MathOPSV2GridRenderer()
        self._mesh_renderer = mesh_renderer.MathOPSV2MeshRenderer()
        self._topology_key = None
        self._content_key = None
        self._resource_key = None
        self._init_active_nodes = None
        self._init_parents = None
        self._parents_tex = [None, None]
        self._active_nodes_tex = [None, None]
        self._cell_offsets_tex = [None, None]
        self._cell_counts_tex = [None, None]
        self._cell_errors_tex = [None, None]
        self._counters_tex = None
        self._old_to_new_tex = None
        self._tmp_tex = None
        self._final_output_idx = 0
        self._final_active_nodes = None
        self._final_cell_offsets = None
        self._final_cell_counts = None
        self._final_cell_errors = None
        self._active_capacity = 0
        self._tmp_capacity = 0
        self._max_active_count = 0
        self._pruning_stats = {"active": False, "cells": 0, "ms": 0.0}

    def free(self):
        self._draw_shader = None
        self._normal_shader = None
        self._blit_shader = None
        self._outline_shader = None
        self._compute_shader = None
        self._batch = None
        self._params_ubo = None
        self._scene_texture = None
        self._polygon_texture = None
        self._scene_hash = ""
        self._polygon_hash = ""
        self._outline_data_texture = None
        self._outline_signature = None
        self._offscreen = None
        self._normal_offscreen = None
        self._offscreen_color_texture = None
        self._offscreen_outline_texture = None
        self._offscreen_position_texture = None
        self._offscreen_normal_texture = None
        self._offscreen_depth_texture = None
        self._compiled_scene = None
        self._compiled_scene_scene_key = 0
        self._compiled_scene_static_revision = -1
        self._compiled_scene_transform_revision = -1
        self._compiled_scene_dynamic_signature = None
        self._dummy_scalar_texture = None
        if self._grid_renderer is not None:
            self._grid_renderer.free()
        if self._mesh_renderer is not None:
            self._mesh_renderer.free()
        self._topology_key = None
        self._content_key = None
        self._resource_key = None
        self._init_active_nodes = None
        self._init_parents = None
        self._parents_tex = [None, None]
        self._active_nodes_tex = [None, None]
        self._cell_offsets_tex = [None, None]
        self._cell_counts_tex = [None, None]
        self._cell_errors_tex = [None, None]
        self._counters_tex = None
        self._old_to_new_tex = None
        self._tmp_tex = None
        self._final_output_idx = 0
        self._final_active_nodes = None
        self._final_cell_offsets = None
        self._final_cell_counts = None
        self._final_cell_errors = None
        self._active_capacity = 0
        self._tmp_capacity = 0
        self._max_active_count = 0
        self._pruning_stats = {"active": False, "cells": 0, "ms": 0.0}

    def _packed_texture_size(self, count: int) -> tuple[int, int]:
        count = max(1, int(count))
        width = _TEXTURE_DATA_WIDTH
        height = (count + width - 1) // width
        return width, max(1, height)

    def _array_texture_capacity_limit(self) -> int:
        try:
            max_size = int(gpu.capabilities.max_texture_size_get())
        except Exception:
            max_size = _TEXTURE_DATA_WIDTH
        return _TEXTURE_DATA_WIDTH * max(1, max_size)

    def _texture_precision_active_limit(self) -> int:
        return min(self._array_texture_capacity_limit(), _FLOAT32_EXACT_UINT_LIMIT)

    def _create_scalar_texture(self, count: int, fmt: str, values=None, value_type: str = "FLOAT"):
        size = self._packed_texture_size(count)
        if values is None:
            texture = gpu.types.GPUTexture(size, format=fmt)
        else:
            needed = size[0] * size[1]
            if value_type == "FLOAT":
                payload = np.zeros(needed, dtype=np.float32)
                src = np.asarray(values, dtype=np.float32)
                payload[: min(len(src), needed)] = src[:needed]
                buffer = gpu.types.Buffer("FLOAT", len(payload), payload)
            else:
                payload = np.zeros(needed, dtype=np.uint32)
                src = np.asarray(values, dtype=np.uint32)
                payload[: min(len(src), needed)] = src[:needed]
                buffer = gpu.types.Buffer("UINT", len(payload), payload)
            texture = gpu.types.GPUTexture(size, format=fmt, data=buffer)
        try:
            texture.filter_mode(False)
        except Exception:
            pass
        return texture

    def _create_scene_texture(self, rows):
        texture_rows = rows if rows else [(0.0, 0.0, 0.0, 0.0)]
        flat = []
        for row in texture_rows:
            flat.extend(float(value) for value in row)
        buffer = gpu.types.Buffer("FLOAT", len(flat), flat)
        texture = gpu.types.GPUTexture((1, len(texture_rows)), format="RGBA32F", data=buffer)
        try:
            texture.filter_mode(False)
        except Exception:
            pass
        return texture

    def _clear_texture(self, texture, data_format: str, value):
        if texture is None:
            return
        texture.clear(format=data_format, value=value)

    def _ensure_dummy_scalar_texture(self):
        if self._dummy_scalar_texture is None:
            self._dummy_scalar_texture = self._create_scalar_texture(1, "R32F", [0.0])
        return self._dummy_scalar_texture

    def _ensure_scene_texture(self, compiled):
        scene_hash = str(compiled["hash"])
        if self._scene_texture is not None and self._scene_hash == scene_hash:
            return self._scene_texture
        self._scene_texture = self._create_scene_texture(compiled["rows"])
        self._scene_hash = scene_hash
        return self._scene_texture

    def _ensure_polygon_texture(self, compiled):
        scene_hash = str(compiled["hash"])
        if self._polygon_texture is not None and self._polygon_hash == scene_hash:
            return self._polygon_texture
        self._polygon_texture = self._create_scene_texture(compiled.get("polygon_rows", ()))
        self._polygon_hash = scene_hash
        return self._polygon_texture

    def _outline_theme_colors(self, context):
        settings = runtime.scene_settings(getattr(context, "scene", None))
        opacity = 1.0 if settings is None else max(0.0, min(1.0, float(getattr(settings, "outline_opacity", 1.0))))
        default_rgb = (0.0, 0.0, 0.0) if settings is None else tuple(float(component) for component in getattr(settings, "outline_color", (0.0, 0.0, 0.0)))
        default_color = (default_rgb[0], default_rgb[1], default_rgb[2], opacity)
        selected_color = (1.0, 0.55, 0.15, opacity)
        active_color = (1.0, 0.75, 0.3, opacity)
        try:
            theme = context.preferences.themes[0]
            selected_raw = tuple(theme.view_3d.object_selected)
            active_raw = tuple(theme.view_3d.object_active)
            if len(selected_raw) == 3:
                selected_color = selected_raw + (opacity,)
            if len(active_raw) == 3:
                active_color = active_raw + (opacity,)
        except Exception:
            pass
        return default_color, selected_color, active_color

    def _ensure_outline_data_texture(self, context, compiled):
        primitive_entries = tuple(compiled.get("primitive_entries", ()))
        active_object = runtime.object_identity(getattr(context, "active_object", None))
        active_key = runtime.object_key(active_object)
        resource_ids = {}
        packed_ids = []
        signature_items = [str(compiled.get("hash", "")), active_key]
        next_resource_id = 1

        for primitive_index, entry in enumerate(primitive_entries):
            obj = runtime.object_identity(entry.get("object", None))
            object_key = runtime.object_key(obj)
            resource_key = object_key if object_key else ("primitive", primitive_index)
            if resource_key not in resource_ids:
                resource_ids[resource_key] = next_resource_id
                next_resource_id += 1

            selected = False
            if obj is not None:
                try:
                    selected = bool(obj.select_get())
                except Exception:
                    selected = False

            color_id = _OUTLINE_COLOR_DEFAULT
            if selected and object_key == active_key:
                color_id = _OUTLINE_COLOR_ACTIVE
            elif selected:
                color_id = _OUTLINE_COLOR_SELECTED

            packed_id = float(color_id * _OUTLINE_ID_STRIDE + resource_ids[resource_key])
            packed_ids.append(packed_id)
            signature_items.append((resource_key, color_id, resource_ids[resource_key]))

        outline_signature = tuple(signature_items)
        if self._outline_signature == outline_signature and self._outline_data_texture is not None:
            return self._outline_data_texture

        values = packed_ids if packed_ids else [0.0]
        self._outline_data_texture = self._create_scalar_texture(max(1, len(values)), "R32F", values)
        self._outline_signature = outline_signature
        return self._outline_data_texture

    def _ensure_compiled_scene(self, scene):
        scene_key = runtime.scene_key(scene)
        static_revision, transform_revision = runtime.scene_revision_tuple(scene)
        structure_signature = sdf_tree.scene_structure_signature(scene)
        if (
            self._compiled_scene is None
            or self._compiled_scene_scene_key != scene_key
            or self._compiled_scene_static_revision != static_revision
            or self._compiled_scene.get("scene_structure_signature") != structure_signature
        ):
            self._compiled_scene = sdf_tree.compile_scene(scene)
            self._compiled_scene_scene_key = scene_key
            self._compiled_scene_static_revision = static_revision
            self._compiled_scene_transform_revision = transform_revision
            self._compiled_scene_dynamic_signature = sdf_tree.compiled_dynamic_signature(self._compiled_scene)
            return self._compiled_scene

        if self._compiled_scene_transform_revision != transform_revision:
            self._compiled_scene = sdf_tree.refresh_compiled_scene_dynamic(self._compiled_scene)
            self._compiled_scene_transform_revision = transform_revision
            self._compiled_scene_dynamic_signature = sdf_tree.compiled_dynamic_signature(self._compiled_scene)
            return self._compiled_scene

        if int(self._compiled_scene.get("warp_row_count", 0)) > 0:
            current_signature = sdf_tree.compiled_dynamic_signature(self._compiled_scene)
            if current_signature is None:
                self._compiled_scene = sdf_tree.compile_scene(scene)
                self._compiled_scene_static_revision = static_revision
                self._compiled_scene_transform_revision = transform_revision
            elif current_signature != self._compiled_scene_dynamic_signature:
                self._compiled_scene = sdf_tree.refresh_compiled_scene_dynamic(self._compiled_scene)
            self._compiled_scene_dynamic_signature = sdf_tree.compiled_dynamic_signature(self._compiled_scene)
        return self._compiled_scene

    def _ensure_draw_shader(self):
        if self._draw_shader is not None:
            return self._draw_shader

        interface = gpu.types.GPUStageInterfaceInfo("mathops_v2_interface")
        interface.smooth("VEC2", "uvInterp")

        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.push_constant("MAT4", "invViewProjectionMatrix")
        shader_info.push_constant("MAT4", "viewProjectionMatrix")
        shader_info.typedef_source(raymarch.UNIFORMS_SOURCE)
        shader_info.uniform_buf(0, "MathOPSV2ViewParams", "mathops")
        shader_info.sampler(0, "FLOAT_2D", "sceneData")
        shader_info.sampler(1, "FLOAT_2D", "pruneActiveNodes")
        shader_info.sampler(2, "FLOAT_2D", "pruneCellOffsets")
        shader_info.sampler(3, "FLOAT_2D", "pruneCellCounts")
        shader_info.sampler(4, "FLOAT_2D", "pruneCellErrors")
        shader_info.sampler(5, "FLOAT_2D", "matcapDiffuse")
        shader_info.sampler(6, "FLOAT_2D", "matcapSpecular")
        shader_info.sampler(7, "FLOAT_2D", "outlineData")
        shader_info.sampler(8, "FLOAT_2D", "polygonData")
        shader_info.vertex_in(0, "VEC2", "position")
        shader_info.vertex_out(interface)
        shader_info.fragment_out(0, "VEC4", "FragColor")
        shader_info.fragment_out(1, "FLOAT", "OutlineId")
        shader_info.fragment_out(2, "VEC4", "HitPosition")
        shader_info.depth_write("ANY")
        shader_info.vertex_source(raymarch.VERTEX_SOURCE)
        shader_info.fragment_source(raymarch.FRAGMENT_SOURCE)
        self._draw_shader = gpu.shader.create_from_info(shader_info)
        return self._draw_shader

    def _ensure_normal_shader(self):
        if self._normal_shader is not None:
            return self._normal_shader

        interface = gpu.types.GPUStageInterfaceInfo("mathops_v2_normal_interface")
        interface.smooth("VEC2", "uvInterp")

        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.vertex_in(0, "VEC2", "position")
        shader_info.vertex_out(interface)
        shader_info.fragment_out(0, "VEC4", "FragColor")
        shader_info.sampler(0, "FLOAT_2D", "posTex")
        shader_info.vertex_source(raymarch.VERTEX_SOURCE)
        shader_info.fragment_source(
            """
vec4 loadPos(ivec2 p)
{
  p = clamp(p, ivec2(0), textureSize(posTex, 0) - ivec2(1));
  return texelFetch(posTex, p, 0);
}

void main()
{
  ivec2 texSize = textureSize(posTex, 0);
  ivec2 pixel = clamp(ivec2(gl_FragCoord.xy), ivec2(0), texSize - ivec2(1));
  vec4 c = loadPos(pixel);
  if (c.w < 0.5) {
    FragColor = vec4(0.0);
    return;
  }

  vec3 pc = c.xyz;
  vec4 pl1 = loadPos(pixel + ivec2(-1, 0));
  vec4 pr1 = loadPos(pixel + ivec2( 1, 0));
  vec4 pd1 = loadPos(pixel + ivec2( 0,-1));
  vec4 pu1 = loadPos(pixel + ivec2( 0, 1));
  vec4 pl2 = loadPos(pixel + ivec2(-2, 0));
  vec4 pr2 = loadPos(pixel + ivec2( 2, 0));
  vec4 pd2 = loadPos(pixel + ivec2( 0,-2));
  vec4 pu2 = loadPos(pixel + ivec2( 0, 2));

  vec3 l = pc - pl1.xyz;
  vec3 r = pr1.xyz - pc;
  vec3 d = pc - pd1.xyz;
  vec3 u = pu1.xyz - pc;

  bool vl = pl1.w > 0.5 && pl2.w > 0.5;
  bool vr = pr1.w > 0.5 && pr2.w > 0.5;
  bool vd = pd1.w > 0.5 && pd2.w > 0.5;
  bool vu = pu1.w > 0.5 && pu2.w > 0.5;

  float he_l = vl ? length(2.0 * pl1.xyz - pl2.xyz - pc) : 1e10;
  float he_r = vr ? length(2.0 * pr1.xyz - pr2.xyz - pc) : 1e10;
  float ve_d = vd ? length(2.0 * pd1.xyz - pd2.xyz - pc) : 1e10;
  float ve_u = vu ? length(2.0 * pu1.xyz - pu2.xyz - pc) : 1e10;

  vec3 hDeriv;
  vec3 vDeriv;
  if (vl && vr) {
    float ratio = max(he_l, he_r) / max(min(he_l, he_r), 1e-20);
    hDeriv = (ratio > 4.0) ? ((he_l < he_r) ? l : r) : ((pr1.xyz - pl1.xyz) * 0.5);
  }
  else if (pr1.w > 0.5) {
    hDeriv = r;
  }
  else if (pl1.w > 0.5) {
    hDeriv = l;
  }
  else {
    hDeriv = vec3(1.0, 0.0, 0.0);
  }

  if (vd && vu) {
    float ratio = max(ve_d, ve_u) / max(min(ve_d, ve_u), 1e-20);
    vDeriv = (ratio > 4.0) ? ((ve_d < ve_u) ? d : u) : ((pu1.xyz - pd1.xyz) * 0.5);
  }
  else if (pu1.w > 0.5) {
    vDeriv = u;
  }
  else if (pd1.w > 0.5) {
    vDeriv = d;
  }
  else {
    vDeriv = vec3(0.0, 1.0, 0.0);
  }

  vec3 n = normalize(cross(vDeriv, hDeriv));
  if (any(isnan(n))) {
    n = vec3(0.0, 0.0, 1.0);
  }
  FragColor = vec4(n, 1.0);
}
"""
        )
        self._normal_shader = gpu.shader.create_from_info(shader_info)
        return self._normal_shader

    def _ensure_blit_shader(self):
        if self._blit_shader is not None:
            return self._blit_shader

        interface = gpu.types.GPUStageInterfaceInfo("mathops_v2_blit_interface")
        interface.smooth("VEC2", "uvInterp")

        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.typedef_source(raymarch.UNIFORMS_SOURCE)
        shader_info.uniform_buf(0, "MathOPSV2ViewParams", "mathops")
        shader_info.vertex_in(0, "VEC2", "position")
        shader_info.vertex_out(interface)
        shader_info.fragment_out(0, "VEC4", "FragColor")
        shader_info.sampler(0, "FLOAT_2D", "colorTex")
        shader_info.sampler(1, "FLOAT_2D", "depthTex")
        shader_info.sampler(2, "FLOAT_2D", "posTex")
        shader_info.sampler(3, "FLOAT_2D", "normalTex")
        shader_info.sampler(4, "FLOAT_2D", "matcapDiffuse")
        shader_info.sampler(5, "FLOAT_2D", "matcapSpecular")
        shader_info.depth_write("ANY")
        shader_info.vertex_source(raymarch.VERTEX_SOURCE)
        shader_info.fragment_source(
            """
int viewFlagsValue()
{
  return int(mathops.cameraPosition.w + 0.5);
}

bool orthographicViewValue()
{
  return (viewFlagsValue() & 1) != 0;
}

vec3 viewIncidentVector(vec3 viewPos)
{
  return orthographicViewValue() ? vec3(0.0, 0.0, 1.0) : normalize(-viewPos);
}

vec4 loadPos(ivec2 p)
{
  p = clamp(p, ivec2(0), textureSize(posTex, 0) - ivec2(1));
  return texelFetch(posTex, p, 0);
}

bool disableSurfaceShadingValue()
{
  return (viewFlagsValue() & 2) != 0;
}

int debugModeValue()
{
  return int(mathops.instructionDebugSpecular.y + 0.5);
}

bool showSpecularValue()
{
  return mathops.instructionDebugSpecular.w > 0.5;
}

float gammaValue()
{
  return max(0.001, mathops.viewRow3.w);
}

vec2 matcapUV(vec3 normalView, vec3 viewDirView)
{
  float a = 1.0 / max(1.0 + viewDirView.z, 0.001);
  float b = -viewDirView.x * viewDirView.y * a;
  vec3 basisX = vec3(
    1.0 - viewDirView.x * viewDirView.x * a,
    b,
    -viewDirView.x
  );
  vec3 basisY = vec3(
    b,
    1.0 - viewDirView.y * viewDirView.y * a,
    -viewDirView.y
  );
  vec2 uv = vec2(dot(basisX, normalView), dot(basisY, normalView));
  return clamp(uv * 0.496 + 0.5, vec2(0.0), vec2(1.0));
}

void main()
{
  vec4 color = texture(colorTex, uvInterp);
  FragColor = color;
  gl_FragDepth = texture(depthTex, uvInterp).r;
  if (debugModeValue() != 0 || disableSurfaceShadingValue()) {
    return;
  }

  vec4 normalTexel = texture(normalTex, uvInterp);
  if (normalTexel.w < 0.5) {
    return;
  }

  vec3 normal = normalize(normalTexel.xyz);
  ivec2 texSize = textureSize(posTex, 0);
  ivec2 pixel = clamp(ivec2(gl_FragCoord.xy), ivec2(0), texSize - ivec2(1));
  vec4 pos = loadPos(pixel);
  if (pos.w < 0.5) {
    return;
  }
  vec3 viewPos = vec3(
    dot(mathops.viewRow0, vec4(pos.xyz, 1.0)),
    dot(mathops.viewRow1, vec4(pos.xyz, 1.0)),
    dot(mathops.viewRow2, vec4(pos.xyz, 1.0))
  );
  vec3 normalView = normalize(vec3(
    dot(mathops.viewRow0.xyz, normal),
    dot(mathops.viewRow1.xyz, normal),
    dot(mathops.viewRow2.xyz, normal)
  ));
  if (dot(normalView, viewIncidentVector(viewPos)) < 0.0) {
    normal = -normal;
    normalView = -normalView;
  }
  vec3 viewDirView = viewIncidentVector(viewPos);
  vec2 uv = matcapUV(normalView, viewDirView);
  vec3 shaded = texture(matcapDiffuse, uv).rgb;
  if (showSpecularValue()) {
    shaded += texture(matcapSpecular, uv).rgb;
  }
  FragColor = vec4(pow(shaded, vec3(gammaValue())), color.a);
}
"""
        )
        self._blit_shader = gpu.shader.create_from_info(shader_info)
        return self._blit_shader

    def _ensure_outline_shader(self):
        if self._outline_shader is not None:
            return self._outline_shader

        interface = gpu.types.GPUStageInterfaceInfo("mathops_v2_outline_interface")
        interface.smooth("VEC2", "uvInterp")

        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.vertex_in(0, "VEC2", "position")
        shader_info.vertex_out(interface)
        shader_info.fragment_out(0, "VEC4", "FragColor")
        shader_info.sampler(0, "FLOAT_2D", "idTex")
        shader_info.push_constant("VEC4", "defaultOutlineColor")
        shader_info.push_constant("VEC4", "selectedOutlineColor")
        shader_info.push_constant("VEC4", "activeOutlineColor")
        shader_info.vertex_source(raymarch.VERTEX_SOURCE)
        shader_info.fragment_source(
            """
const int OUTLINE_ID_STRIDE = %d;

void main()
{
  ivec2 texSize = textureSize(idTex, 0);
  ivec2 coord = clamp(ivec2(gl_FragCoord.xy), ivec2(0), texSize - ivec2(1));
  int centerId = int(texelFetch(idTex, coord, 0).x + 0.5);
  if (centerId == 0) {
    discard;
    return;
  }

  bool edge = false;
  ivec2 offsets[4] = ivec2[4](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));
  for (int index = 0; index < 4; index++) {
    ivec2 neighborCoord = clamp(coord + offsets[index], ivec2(0), texSize - ivec2(1));
    int neighborId = int(texelFetch(idTex, neighborCoord, 0).x + 0.5);
    if (neighborId != centerId) {
      edge = true;
      break;
    }
  }

  if (!edge) {
    discard;
    return;
  }

  int colorId = centerId / OUTLINE_ID_STRIDE;
  vec4 lineColor = defaultOutlineColor;
  if (colorId == %d) {
    lineColor = selectedOutlineColor;
  }
  else if (colorId == %d) {
    lineColor = activeOutlineColor;
  }

  FragColor = lineColor;
}
"""
            % (_OUTLINE_ID_STRIDE, _OUTLINE_COLOR_SELECTED, _OUTLINE_COLOR_ACTIVE)
        )
        self._outline_shader = gpu.shader.create_from_info(shader_info)
        return self._outline_shader

    def _ensure_compute_shader(self):
        if self._compute_shader is not None:
            return self._compute_shader

        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.push_constant("VEC4", "aabbMin")
        shader_info.push_constant("VEC4", "aabbMax")
        shader_info.push_constant("INT", "totalNumNodes")
        shader_info.push_constant("INT", "primitiveCount")
        shader_info.push_constant("INT", "warpRowCount")
        shader_info.push_constant("INT", "currentGridSize")
        shader_info.push_constant("INT", "firstLevel")
        shader_info.push_constant("INT", "activeCounterIndex")
        shader_info.push_constant("INT", "tmpCounterIndex")
        shader_info.push_constant("INT", "statusCounterIndex")
        shader_info.push_constant("INT", "activeCapacity")
        shader_info.push_constant("INT", "tmpCapacity")
        shader_info.sampler(0, "FLOAT_2D", "sceneData")
        shader_info.sampler(1, "FLOAT_2D", "parentsInit")
        shader_info.sampler(2, "FLOAT_2D", "activeNodesInit")
        shader_info.sampler(3, "FLOAT_2D", "parentsIn")
        shader_info.sampler(4, "FLOAT_2D", "activeNodesIn")
        shader_info.sampler(5, "FLOAT_2D", "parentCellOffsetsTex")
        shader_info.sampler(6, "FLOAT_2D", "parentCellCountsTex")
        shader_info.sampler(7, "FLOAT_2D", "cellValueInTex")
        shader_info.sampler(8, "FLOAT_2D", "polygonData")
        shader_info.image(0, "R32F", "FLOAT_2D", "parentsOut", qualifiers={"WRITE"})
        shader_info.image(1, "R32F", "FLOAT_2D", "activeNodesOut", qualifiers={"WRITE"})
        shader_info.image(2, "R32F", "FLOAT_2D", "childCellOffsetsImg", qualifiers={"WRITE"})
        shader_info.image(3, "R32F", "FLOAT_2D", "cellCountsImg", qualifiers={"WRITE"})
        shader_info.image(4, "R32UI", "UINT_2D_ATOMIC", "countersImg", qualifiers={"READ", "WRITE"})
        shader_info.image(5, "R32F", "FLOAT_2D", "cellValueOutImg", qualifiers={"WRITE"})
        shader_info.image(6, "R32UI", "UINT_2D", "oldToNewImg", qualifiers={"READ", "WRITE"})
        shader_info.image(7, "R32UI", "UINT_2D", "tmpImg", qualifiers={"READ", "WRITE"})
        shader_info.local_group_size(*pruning_shader.LOCAL_GROUP_SIZE)
        shader_info.compute_source(pruning_shader.COMPUTE_SOURCE)
        self._compute_shader = gpu.shader.create_from_info(shader_info)
        return self._compute_shader

    def _ensure_batch(self):
        if self._batch is not None:
            return self._batch
        shader = self._ensure_draw_shader()
        self._batch = batch_for_shader(
            shader,
            "TRI_STRIP",
            {"position": ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))},
        )
        return self._batch

    def _ensure_offscreen(self, width: int, height: int) -> bool:
        width = max(1, int(width))
        height = max(1, int(height))
        if (
            self._offscreen is not None
            and self._normal_offscreen is not None
            and self._offscreen_color_texture is not None
            and self._offscreen_outline_texture is not None
            and self._offscreen_position_texture is not None
            and self._offscreen_normal_texture is not None
            and self._offscreen_depth_texture is not None
        ):
            if self._offscreen_color_texture.width == width and self._offscreen_color_texture.height == height:
                return True
            self._offscreen = None
            self._normal_offscreen = None
            self._offscreen_color_texture = None
            self._offscreen_outline_texture = None
            self._offscreen_position_texture = None
            self._offscreen_normal_texture = None
            self._offscreen_depth_texture = None

        try:
            self._offscreen_color_texture = gpu.types.GPUTexture((width, height), format="RGBA16F")
            self._offscreen_outline_texture = gpu.types.GPUTexture((width, height), format="R32F")
            self._offscreen_position_texture = gpu.types.GPUTexture((width, height), format="RGBA32F")
            self._offscreen_normal_texture = gpu.types.GPUTexture((width, height), format="RGBA16F")
            try:
                self._offscreen_outline_texture.filter_mode(False)
            except Exception:
                pass
            try:
                self._offscreen_position_texture.filter_mode(False)
            except Exception:
                pass
            try:
                self._offscreen_normal_texture.filter_mode(False)
            except Exception:
                pass
            self._offscreen_depth_texture = gpu.types.GPUTexture((width, height), format="DEPTH24_STENCIL8")
            self._offscreen = gpu.types.GPUFrameBuffer(
                color_slots=(self._offscreen_color_texture, self._offscreen_outline_texture, self._offscreen_position_texture),
                depth_slot=self._offscreen_depth_texture,
            )
            self._normal_offscreen = gpu.types.GPUFrameBuffer(color_slots=(self._offscreen_normal_texture,))
            return True
        except Exception:
            self._offscreen = None
            self._normal_offscreen = None
            self._offscreen_color_texture = None
            self._offscreen_outline_texture = None
            self._offscreen_position_texture = None
            self._offscreen_normal_texture = None
            self._offscreen_depth_texture = None
            return False

    def _should_draw_custom_grid(self, context, settings) -> bool:
        space = getattr(context, "space_data", None)
        shading = getattr(space, "shading", None)
        overlay_state = getattr(space, "overlay", None)
        if shading is None or overlay_state is None or getattr(shading, "type", "") != "RENDERED":
            return False
        if not bool(getattr(overlay_state, "show_overlays", True)):
            return False
        return any(
            bool(getattr(settings, name, False))
            for name in ("show_grid", "show_floor", "show_axis_x", "show_axis_y", "show_axis_z")
        )

    def _draw_scene_pass(self, context, settings, scene_texture, polygon_texture, outline_texture, params_ubo, inv_view_projection, view_projection):
        shader = self._ensure_draw_shader()
        batch = self._ensure_batch()
        shader.uniform_float("invViewProjectionMatrix", inv_view_projection)
        shader.uniform_float("viewProjectionMatrix", view_projection)
        shader.uniform_block("mathops", params_ubo)
        shader.uniform_sampler("sceneData", scene_texture)
        shader.uniform_sampler("pruneActiveNodes", self._final_active_nodes or self._ensure_dummy_scalar_texture())
        shader.uniform_sampler("pruneCellOffsets", self._final_cell_offsets or self._ensure_dummy_scalar_texture())
        shader.uniform_sampler("pruneCellCounts", self._final_cell_counts or self._ensure_dummy_scalar_texture())
        shader.uniform_sampler("pruneCellErrors", self._final_cell_errors or self._ensure_dummy_scalar_texture())
        diffuse_texture, specular_texture = matcap.get_matcap_textures(context, getattr(settings, "custom_matcap", ""))
        shader.uniform_sampler("matcapDiffuse", diffuse_texture)
        shader.uniform_sampler("matcapSpecular", specular_texture)
        shader.uniform_sampler("outlineData", outline_texture)
        shader.uniform_sampler("polygonData", polygon_texture)
        batch.draw(shader)

    def _draw_normal_pass(self):
        if self._normal_offscreen is None or self._offscreen_position_texture is None:
            return
        shader = self._ensure_normal_shader()
        batch = self._ensure_batch()
        with self._normal_offscreen.bind():
            gpu.state.depth_test_set("NONE")
            gpu.state.depth_mask_set(False)
            gpu.state.blend_set("NONE")
            self._clear_texture(self._offscreen_normal_texture, "FLOAT", (0.0, 0.0, 0.0, 0.0))
            shader.uniform_sampler("posTex", self._offscreen_position_texture)
            batch.draw(shader)

    def _blit_offscreen(self, context, settings, params_ubo):
        shader = self._ensure_blit_shader()
        batch = self._ensure_batch()
        diffuse_texture, specular_texture = matcap.get_matcap_textures(context, getattr(settings, "custom_matcap", ""))
        shader.uniform_block("mathops", params_ubo)
        shader.uniform_sampler("colorTex", self._offscreen_color_texture)
        shader.uniform_sampler("depthTex", self._offscreen_depth_texture)
        shader.uniform_sampler("posTex", self._offscreen_position_texture)
        shader.uniform_sampler("normalTex", self._offscreen_normal_texture)
        shader.uniform_sampler("matcapDiffuse", diffuse_texture)
        shader.uniform_sampler("matcapSpecular", specular_texture)
        batch.draw(shader)

    def _draw_outline_pass(self, context):
        if self._offscreen_outline_texture is None or self._offscreen_depth_texture is None:
            return
        settings = runtime.scene_settings(getattr(context, "scene", None))
        if settings is not None and float(getattr(settings, "outline_opacity", 1.0)) <= 0.0:
            return
        shader = self._ensure_outline_shader()
        batch = self._ensure_batch()
        default_color, selected_color, active_color = self._outline_theme_colors(context)
        shader.uniform_float("defaultOutlineColor", default_color)
        shader.uniform_float("selectedOutlineColor", selected_color)
        shader.uniform_float("activeOutlineColor", active_color)
        shader.uniform_sampler("idTex", self._offscreen_outline_texture)
        batch.draw(shader)

    def _ensure_topology_textures(self, settings, compiled):
        topology_key = pruning.topology_key(settings, compiled)
        if self._topology_key == topology_key and self._init_active_nodes is not None and self._init_parents is not None:
            return

        topology = pruning.build_initial_topology(compiled)
        active_values = [
            float(entry["instruction_index"] + 1) if entry["sign"] else -float(entry["instruction_index"] + 1)
            for entry in topology["active_nodes"]
        ]
        parent_values = [float(parent + 1) for parent in topology["parents"]]
        self._init_active_nodes = self._create_scalar_texture(max(1, len(active_values)), "R32F", active_values)
        self._init_parents = self._create_scalar_texture(max(1, len(parent_values)), "R32F", parent_values)
        self._topology_key = topology_key
        self._content_key = None

    def _capacity_for_scene(self, node_count: int) -> tuple[int, int]:
        texture_cap = min(self._array_texture_capacity_limit(), _MAX_TMP_CAP)
        active_cap_limit = min(self._texture_precision_active_limit(), _MAX_ACTIVE_CAP)
        active_default = min(max(int(node_count) * 4096, 8_000_000), active_cap_limit)
        tmp_default = min(max(max(active_default * 8, int(node_count) * 1024), 32_000_000), texture_cap)
        return active_default, tmp_default

    def _ensure_work_textures(self, settings, compiled, active_capacity=None, tmp_capacity=None):
        node_count = int(compiled["instruction_count"])
        final_grid_size = pruning.final_grid_size(settings)
        num_cells = final_grid_size * final_grid_size * final_grid_size
        texture_cap = self._array_texture_capacity_limit()
        if num_cells > texture_cap:
            raise RuntimeError("Pruning grid exceeds texture capacity")

        default_active, default_tmp = self._capacity_for_scene(node_count)
        if active_capacity is None:
            active_capacity = max(default_active, self._active_capacity)
        if tmp_capacity is None:
            tmp_capacity = max(default_tmp, self._tmp_capacity)

        active_capacity = min(int(active_capacity), self._texture_precision_active_limit())
        tmp_capacity = min(int(tmp_capacity), texture_cap)
        resource_key = (final_grid_size, active_capacity, tmp_capacity)
        if self._resource_key == resource_key and self._active_nodes_tex[0] is not None:
            return

        self._active_capacity = active_capacity
        self._tmp_capacity = tmp_capacity
        self._resource_key = resource_key

        for idx in range(2):
            self._parents_tex[idx] = self._create_scalar_texture(active_capacity, "R32F")
            self._active_nodes_tex[idx] = self._create_scalar_texture(active_capacity, "R32F")
            self._cell_offsets_tex[idx] = self._create_scalar_texture(num_cells, "R32F")
            self._cell_counts_tex[idx] = self._create_scalar_texture(num_cells, "R32F")
            self._cell_errors_tex[idx] = self._create_scalar_texture(num_cells, "R32F")

        self._counters_tex = self._create_scalar_texture(_COUNTERS_SIZE, "R32UI")
        self._old_to_new_tex = self._create_scalar_texture(tmp_capacity, "R32UI")
        self._tmp_tex = self._create_scalar_texture(tmp_capacity, "R32UI")
        self._content_key = None

    def _grow_work_textures(self, settings, compiled) -> bool:
        next_active = min(max(self._active_capacity * 2, 1), self._texture_precision_active_limit())
        next_tmp = min(max(self._tmp_capacity * 2, 1), self._array_texture_capacity_limit())
        if next_active == self._active_capacity and next_tmp == self._tmp_capacity:
            return False
        self._ensure_work_textures(settings, compiled, next_active, next_tmp)
        return True

    def _reset_pruning_state(self):
        self._clear_texture(self._counters_tex, "UINT", (0,))
        self._max_active_count = 0

    def _clear_dispatch_state(self, output_idx: int):
        del output_idx

    def _dispatch_pruning_level(
        self,
        scene_texture,
        polygon_texture,
        bounds_min,
        bounds_max,
        primitive_count,
        warp_row_count,
        total_nodes,
        current_grid_size,
        first_level,
        input_idx,
        output_idx,
    ):
        shader = self._ensure_compute_shader()
        dummy = self._ensure_dummy_scalar_texture()
        shader.uniform_float("aabbMin", (*bounds_min, 0.0))
        shader.uniform_float("aabbMax", (*bounds_max, 0.0))
        shader.uniform_int("totalNumNodes", [int(total_nodes)])
        shader.uniform_int("primitiveCount", [int(primitive_count)])
        shader.uniform_int("warpRowCount", [int(warp_row_count)])
        shader.uniform_int("currentGridSize", [int(current_grid_size)])
        shader.uniform_int("firstLevel", [1 if first_level else 0])
        shader.uniform_int("activeCounterIndex", [_COUNTERS_ACTIVE_BASE + int(round(math.log(current_grid_size, 2)))])
        shader.uniform_int("tmpCounterIndex", [_COUNTERS_TMP_BASE + int(round(math.log(current_grid_size, 2)))])
        shader.uniform_int("statusCounterIndex", [_COUNTER_STATUS])
        shader.uniform_int("activeCapacity", [int(self._active_capacity)])
        shader.uniform_int("tmpCapacity", [int(self._tmp_capacity)])
        shader.uniform_sampler("sceneData", scene_texture)
        shader.uniform_sampler("parentsInit", self._init_parents or dummy)
        shader.uniform_sampler("activeNodesInit", self._init_active_nodes or dummy)
        shader.uniform_sampler("parentsIn", dummy if first_level else self._parents_tex[input_idx])
        shader.uniform_sampler("activeNodesIn", dummy if first_level else self._active_nodes_tex[input_idx])
        shader.uniform_sampler("parentCellOffsetsTex", dummy if first_level else self._cell_offsets_tex[input_idx])
        shader.uniform_sampler("parentCellCountsTex", dummy if first_level else self._cell_counts_tex[input_idx])
        shader.uniform_sampler("cellValueInTex", dummy if first_level else self._cell_errors_tex[input_idx])
        shader.uniform_sampler("polygonData", polygon_texture)
        shader.image("parentsOut", self._parents_tex[output_idx])
        shader.image("activeNodesOut", self._active_nodes_tex[output_idx])
        shader.image("childCellOffsetsImg", self._cell_offsets_tex[output_idx])
        shader.image("cellCountsImg", self._cell_counts_tex[output_idx])
        shader.image("countersImg", self._counters_tex)
        shader.image("cellValueOutImg", self._cell_errors_tex[output_idx])
        shader.image("oldToNewImg", self._old_to_new_tex)
        shader.image("tmpImg", self._tmp_tex)

        group_size = pruning_shader.LOCAL_GROUP_SIZE
        dispatch_x = int(math.ceil(float(current_grid_size) / float(group_size[0])))
        dispatch_y = int(math.ceil(float(current_grid_size) / float(group_size[1])))
        dispatch_z = int(math.ceil(float(current_grid_size) / float(group_size[2])))
        gpu.compute.dispatch(shader, dispatch_x, dispatch_y, dispatch_z)

    def _disable_pruning(self):
        self._final_active_nodes = None
        self._final_cell_offsets = None
        self._final_cell_counts = None
        self._final_cell_errors = None
        self._content_key = None
        self._pruning_stats = {"active": False, "cells": 0, "ms": 0.0}
        self._max_active_count = 0

    def _update_pruning(self, settings, compiled, scene_texture, polygon_texture):
        self._pruning_stats = {"active": False, "cells": 0, "ms": 0.0}
        if not pruning.should_build(settings, compiled):
            self._disable_pruning()
            return

        self._ensure_topology_textures(settings, compiled)
        self._ensure_work_textures(settings, compiled)

        content_key = pruning.content_key(settings, compiled)
        if self._content_key == content_key and self._final_active_nodes is not None:
            self._pruning_stats = {
                "active": True,
                "cells": pruning.final_grid_size(settings) ** 3,
                "ms": 0.0,
            }
            return

        bounds_min, bounds_max = pruning.normalized_bounds(compiled)
        total_nodes = int(compiled["instruction_count"])
        primitive_count = int(compiled["primitive_count"])
        warp_row_count = int(compiled.get("warp_row_count", 0))
        grid_level = pruning.grid_level(settings)

        while True:
            self._reset_pruning_state()
            start = time.perf_counter()
            input_idx = 0
            output_idx = 1
            first_level = True
            overflow = False

            for level in range(2, grid_level + 1, 2):
                self._clear_dispatch_state(output_idx)
                current_grid_size = 1 << level
                self._dispatch_pruning_level(
                    scene_texture,
                    polygon_texture,
                    bounds_min,
                    bounds_max,
                    primitive_count,
                    warp_row_count,
                    total_nodes,
                    current_grid_size,
                    first_level,
                    input_idx,
                    output_idx,
                )
                first_level = False
                if level != grid_level:
                    input_idx, output_idx = output_idx, input_idx

            counters = np.frombuffer(self._counters_tex.read(), dtype=np.uint32)
            active_counts = counters[_COUNTERS_ACTIVE_BASE : _COUNTERS_ACTIVE_BASE + 10]
            overflow = int(counters[_COUNTER_STATUS]) != 0
            self._max_active_count = int(active_counts.max()) if active_counts.size else 0

            if overflow:
                if self._grow_work_textures(settings, compiled):
                    continue
                self._disable_pruning()
                return

            self._final_output_idx = output_idx
            self._final_active_nodes = self._active_nodes_tex[output_idx]
            self._final_cell_offsets = self._cell_offsets_tex[output_idx]
            self._final_cell_counts = self._cell_counts_tex[output_idx]
            self._final_cell_errors = self._cell_errors_tex[output_idx]
            self._content_key = content_key
            self._pruning_stats = {
                "active": True,
                "cells": pruning.final_grid_size(settings) ** 3,
                "ms": (time.perf_counter() - start) * 1000.0,
            }
            return

    def _ensure_params_ubo(self, settings, compiled, camera_position, region_data, show_specular):
        pruning_enabled = bool(self._final_active_nodes is not None)
        render_bounds_min, render_bounds_max = compiled.get("render_bounds", compiled["scene_bounds"])
        bounds_min, bounds_max = pruning.normalized_bounds(compiled)
        grid_size = float(pruning.final_grid_size(settings)) if pruning_enabled else 0.0
        view_matrix = region_data.view_matrix
        is_orthographic = not bool(region_data.is_perspective)
        view_flags = 0
        if is_orthographic:
            view_flags |= 1
        if bool(getattr(settings, "disable_surface_shading", False)):
            view_flags |= 2
        background_color = runtime.scene_background_color(getattr(settings, "id_data", None))

        viz_max = float(max(int(getattr(settings, "colormap_max", 25)), 1))
        data = [
            float(camera_position[0]),
            float(camera_position[1]),
            float(camera_position[2]),
            float(view_flags),
            float(settings.surface_epsilon),
            float(settings.max_distance),
            float(settings.max_steps),
            float(compiled["primitive_count"]),
            float(compiled["instruction_count"]),
            float(pruning.debug_mode_value(settings)),
            viz_max,
            1.0 if show_specular else 0.0,
            float(compiled.get("warp_row_count", 0)),
            float(background_color[0]),
            float(background_color[1]),
            float(background_color[2]),
            float(render_bounds_min[0]),
            float(render_bounds_min[1]),
            float(render_bounds_min[2]),
            0.0,
            float(render_bounds_max[0]),
            float(render_bounds_max[1]),
            float(render_bounds_max[2]),
            0.0,
            float(bounds_min[0]),
            float(bounds_min[1]),
            float(bounds_min[2]),
            float(grid_size),
            float(bounds_max[0]),
            float(bounds_max[1]),
            float(bounds_max[2]),
            1.0 if pruning_enabled else 0.0,
            float(view_matrix[0][0]),
            float(view_matrix[0][1]),
            float(view_matrix[0][2]),
            float(view_matrix[0][3]),
            float(view_matrix[1][0]),
            float(view_matrix[1][1]),
            float(view_matrix[1][2]),
            float(view_matrix[1][3]),
            float(view_matrix[2][0]),
            float(view_matrix[2][1]),
            float(view_matrix[2][2]),
            float(view_matrix[2][3]),
            float(view_matrix[3][0]),
            float(view_matrix[3][1]),
            float(view_matrix[3][2]),
            float(settings.gamma),
        ]
        buffer = gpu.types.Buffer("FLOAT", len(data), data)
        if self._params_ubo is None:
            self._params_ubo = gpu.types.GPUUniformBuf(buffer)
        else:
            self._params_ubo.update(buffer)
        return self._params_ubo

    def draw(self, context, depsgraph):
        scene = context.scene
        settings = runtime.scene_settings(scene)
        if settings is None:
            raise RuntimeError("MathOPS scene settings are unavailable")

        compiled = self._ensure_compiled_scene(scene)
        if compiled.get("message"):
            runtime.set_error(compiled["message"])
        else:
            runtime.clear_error()

        scene_texture = self._ensure_scene_texture(compiled)
        polygon_texture = self._ensure_polygon_texture(compiled)
        outline_texture = self._ensure_outline_data_texture(context, compiled)
        try:
            self._update_pruning(settings, compiled, scene_texture, polygon_texture)
        except Exception:
            self._disable_pruning()

        overlay.suppress_space_overlays(getattr(context, "space_data", None), scene)
        region_data = context.region_data
        inv_view_projection = region_data.perspective_matrix.inverted()
        view_projection = region_data.perspective_matrix
        camera_position = region_data.view_matrix.inverted().translation
        shading = getattr(getattr(context, "space_data", None), "shading", None)
        show_specular = bool(getattr(shading, "show_specular_highlight", True))
        region = getattr(context, "region", None)
        pixel_size = float(getattr(getattr(context.preferences, "system", None), "pixel_size", 1.0))
        render_width = max(1, int(getattr(region, "width", 1) * pixel_size))
        render_height = max(1, int(getattr(region, "height", 1) * pixel_size))
        params_ubo = self._ensure_params_ubo(
            settings,
            compiled,
            camera_position,
            region_data,
            show_specular,
        )
        if not self._ensure_offscreen(render_width, render_height):
            raise RuntimeError("MathOPS viewport outline buffers are unavailable")

        with self._offscreen.bind():
            gpu.state.depth_test_set("ALWAYS")
            gpu.state.depth_mask_set(True)
            gpu.state.blend_set("NONE")
            self._clear_texture(self._offscreen_color_texture, "FLOAT", runtime.scene_background_color(scene))
            self._clear_texture(self._offscreen_outline_texture, "FLOAT", (0.0,))
            self._clear_texture(self._offscreen_position_texture, "FLOAT", (0.0, 0.0, 0.0, 0.0))
            self._clear_texture(self._offscreen_normal_texture, "FLOAT", (0.0, 0.0, 0.0, 0.0))
            self._offscreen.clear(depth=1.0, stencil=0)
            self._draw_scene_pass(
                context,
                settings,
                scene_texture,
                polygon_texture,
                outline_texture,
                params_ubo,
                inv_view_projection,
                view_projection,
            )

            if bool(getattr(settings, "show_meshes", True)):
                self._mesh_renderer.draw(
                    context, depsgraph,
                    bool(getattr(settings, "mesh_backface_culling", True)),
                    view_projection,
                )

        self._draw_normal_pass()

        gpu.state.depth_test_set("ALWAYS")
        gpu.state.depth_mask_set(True)
        gpu.state.blend_set("NONE")
        self._blit_offscreen(context, settings, params_ubo)

        if self._should_draw_custom_grid(context, settings):
            gpu.state.depth_test_set("NONE")
            gpu.state.depth_mask_set(False)
            gpu.state.blend_set("ALPHA")
            self._grid_renderer.draw(
                context,
                render_width,
                render_height,
                region_data.view_matrix,
                region_data.window_matrix,
                self._offscreen_depth_texture,
            )

        gpu.state.depth_test_set("NONE")
        gpu.state.depth_mask_set(False)
        gpu.state.blend_set("ALPHA")
        self._draw_outline_pass(context)

        gpu.state.depth_test_set("NONE")
        gpu.state.depth_mask_set(False)
        gpu.state.blend_set("NONE")

        compiled["pruning_active"] = bool(self._pruning_stats["active"])
        compiled["pruning_cells"] = int(self._pruning_stats["cells"])
        compiled["pruning_sequences"] = int(self._max_active_count)
        compiled["pruning_ms"] = float(self._pruning_stats["ms"])
        compiled["pruning_pending"] = False
        return compiled
