import ctypes
import os
import re
import sys
import time
from pathlib import Path

import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from gpu_extras.presets import draw_texture_2d
from mathutils import Matrix

from .. import runtime
from . import bridge, matcap


_BINDING_PRIMS = 0
_BINDING_NODES = 1
_BINDING_BINARY_OPS = 2
_BINDING_PARENTS_IN = 3
_BINDING_PARENTS_OUT = 4
_BINDING_ACTIVE_NODES_IN = 5
_BINDING_ACTIVE_NODES_OUT = 6
_BINDING_PARENT_CELL_OFFSETS = 7
_BINDING_CELL_OFFSETS = 8
_BINDING_PARENT_CELL_COUNTS = 9
_BINDING_CELL_COUNTS = 10
_BINDING_COUNTERS = 11
_BINDING_CELL_ERROR_IN = 12
_BINDING_CELL_ERROR_OUT = 13
_BINDING_OLD_TO_NEW_SCRATCH = 14
_BINDING_TMP = 15

_COUNTERS_ACTIVE_BASE = 0
_COUNTERS_OLD_TO_NEW_BASE = 10
_COUNTERS_STATUS = 20
_COUNTERS_SIZE = 21

_GL_SHADER_STORAGE_BARRIER_BIT = 0x2000
_GL_BUFFER_UPDATE_BARRIER_BIT = 0x0200
_MAX_ACTIVE_CAP = 100_000_000
_MAX_TMP_CAP = 400_000_000
_FLOAT32_EXACT_UINT_LIMIT = 1 << 24
_TEXTURE_DATA_WIDTH = 4096
_PRIMITIVE_TEXELS = 7
_VIEWPORT_SHADER_FILES = (
    "culling.comp.glsl",
    "common.glsl",
    "common_culling.glsl",
    "eval.glsl",
    "extensions.glsl",
    "simple.frag.glsl",
)

_INCLUDE_RE = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.M)
_BUFFER_REFERENCE_RE = re.compile(
    r"layout\s*\(std430,\s*buffer_reference.*?\nvoid set_bit", re.S
)
_PUSH_CONSTANT_RE = re.compile(
    r"layout\s*\(push_constant\)\s*uniform\s+PushConstant\s*\{.*?\};\s*", re.S
)
_FRAGMENT_OUTPUT_RE = re.compile(
    r"^\s*layout\s*\(\s*location\s*=\s*0\s*\)\s*out\s+vec4\s+outColor\s*;\s*\n",
    re.M,
)
_FRAGMENT_SHADING_MODE_RE = re.compile(
    r"^\s*layout\s*\(\s*constant_id\s*=\s*0\s*\)\s*const\s+int\s+shading_mode\s*=\s*0\s*;\s*\n",
    re.M,
)
_UNUSED_UINT64_RE = re.compile(
    r"int\s+read_node_state\(uint64_t.*?void\s+write_node_state\(inout\s+uint64_t.*?\}\s*",
    re.S,
)
_PCG_BLOCK_RE = re.compile(
    r"#define\s+PCG\s+u64vec2.*?float\s+rand_float_0_1\(inout\s+PCG\s+pcg\)\s*\{.*?\}\s*",
    re.S,
)

_GLSL_INT16_HEADER = """
#extension GL_NV_gpu_shader5 : require
""".strip()

_COMMON_BUFFER_BLOCK_FRAGMENT_U32 = f"""
layout(std430, binding = {_BINDING_PRIMS}) buffer PrimitivesBuf {{ Primitive tab[]; }} prims;
layout(std430, binding = {_BINDING_NODES}) buffer NodesBuf {{ Node tab[]; }} nodes;
layout(std430, binding = {_BINDING_BINARY_OPS}) buffer BinaryOpsBuf {{ BinaryOp tab[]; }} binary_ops;
layout(std430, binding = {_BINDING_ACTIVE_NODES_OUT}) buffer ActiveNodesOutBuf {{ uint tab[]; }} active_nodes_out;
layout(std430, binding = {_BINDING_CELL_OFFSETS}) buffer CellOffsetsBuf {{ int tab[]; }} cells_offset;
layout(std430, binding = {_BINDING_CELL_COUNTS}) buffer CellCountsBuf {{ int tab[]; }} cells_num_active;
layout(std430, binding = {_BINDING_CELL_ERROR_OUT}) buffer CellErrorOutBuf {{ float tab[]; }} cell_error_out;

ActiveNode active_nodes_out_get(int idx) {{
    uint packed_word = active_nodes_out.tab[idx >> 1];
    uint val = ((idx & 1) == 0) ? (packed_word & 0xffffu) : (packed_word >> 16);
    return ActiveNode(val);
}}
""".strip()

_COMMON_BUFFER_BLOCK_FRAGMENT_U16 = f"""
layout(std430, binding = {_BINDING_PRIMS}) buffer PrimitivesBuf {{ Primitive tab[]; }} prims;
layout(std430, binding = {_BINDING_NODES}) buffer NodesBuf {{ Node tab[]; }} nodes;
layout(std430, binding = {_BINDING_BINARY_OPS}) buffer BinaryOpsBuf {{ BinaryOp tab[]; }} binary_ops;
layout(std430, binding = {_BINDING_ACTIVE_NODES_OUT}) buffer ActiveNodesOutBuf {{ ActiveNode tab[]; }} active_nodes_out;
layout(std430, binding = {_BINDING_CELL_OFFSETS}) buffer CellOffsetsBuf {{ int tab[]; }} cells_offset;
layout(std430, binding = {_BINDING_CELL_COUNTS}) buffer CellCountsBuf {{ int tab[]; }} cells_num_active;
layout(std430, binding = {_BINDING_CELL_ERROR_OUT}) buffer CellErrorOutBuf {{ float tab[]; }} cell_error_out;
""".strip()

_COMMON_BUFFER_BLOCK_COMPUTE_U32 = f"""
layout(std430, binding = {_BINDING_PRIMS}) buffer PrimitivesBuf {{ Primitive tab[]; }} prims;
layout(std430, binding = {_BINDING_NODES}) buffer NodesBuf {{ Node tab[]; }} nodes;
layout(std430, binding = {_BINDING_BINARY_OPS}) buffer BinaryOpsBuf {{ BinaryOp tab[]; }} binary_ops;
layout(std430, binding = {_BINDING_PARENTS_IN}) buffer ParentsInBuf {{ uint tab[]; }} parents_in;
layout(std430, binding = {_BINDING_PARENTS_OUT}) buffer ParentsOutBuf {{ uint tab[]; }} parents_out;
layout(std430, binding = {_BINDING_ACTIVE_NODES_IN}) buffer ActiveNodesInBuf {{ uint tab[]; }} active_nodes_in;
layout(std430, binding = {_BINDING_ACTIVE_NODES_OUT}) buffer ActiveNodesOutBuf {{ uint tab[]; }} active_nodes_out;
layout(std430, binding = {_BINDING_PARENT_CELL_OFFSETS}) buffer ParentCellOffsetsBuf {{ int tab[]; }} parent_cells_offset;
layout(std430, binding = {_BINDING_CELL_OFFSETS}) buffer ChildCellOffsetsBuf {{ int tab[]; }} child_cells_offset;
layout(std430, binding = {_BINDING_PARENT_CELL_COUNTS}) buffer ParentCellCountsBuf {{ int tab[]; }} parent_cells_num_active;
layout(std430, binding = {_BINDING_CELL_COUNTS}) buffer CellCountsBuf {{ int tab[]; }} num_active_out;
layout(std430, binding = {_BINDING_COUNTERS}) buffer CountersBuf {{ int tab[]; }} counters;
layout(std430, binding = {_BINDING_CELL_ERROR_IN}) buffer CellValueInBuf {{ float tab[]; }} cell_value_in;
layout(std430, binding = {_BINDING_CELL_ERROR_OUT}) buffer CellValueOutBuf {{ float tab[]; }} cell_value_out;
layout(std430, binding = {_BINDING_OLD_TO_NEW_SCRATCH}) buffer OldToNewScratchBuf {{ uint tab[]; }} old_to_new_scratch;
layout(std430, binding = {_BINDING_TMP}) buffer TmpBuf {{ Tmp tab[]; }} tmp;

ActiveNode active_nodes_in_get(int idx) {{
    uint packed_word = active_nodes_in.tab[idx >> 1];
    uint val = ((idx & 1) == 0) ? (packed_word & 0xffffu) : (packed_word >> 16);
    return ActiveNode(val);
}}

void active_nodes_out_set(int idx, ActiveNode value) {{
    int word_idx = idx >> 1;
    uint shift = uint((idx & 1) * 16);
    uint mask = 0xffffu << shift;
    uint word = active_nodes_out.tab[word_idx];
    active_nodes_out.tab[word_idx] = (word & ~mask) | ((value.idx_and_sign & 0xffffu) << shift);
}}
""".strip()

_COMMON_BUFFER_BLOCK_COMPUTE_U16 = f"""
layout(std430, binding = {_BINDING_PRIMS}) buffer PrimitivesBuf {{ Primitive tab[]; }} prims;
layout(std430, binding = {_BINDING_NODES}) buffer NodesBuf {{ Node tab[]; }} nodes;
layout(std430, binding = {_BINDING_BINARY_OPS}) buffer BinaryOpsBuf {{ BinaryOp tab[]; }} binary_ops;
layout(std430, binding = {_BINDING_PARENTS_IN}) buffer ParentsInBuf {{ uint16_t tab[]; }} parents_in;
layout(std430, binding = {_BINDING_PARENTS_OUT}) buffer ParentsOutBuf {{ uint16_t tab[]; }} parents_out;
layout(std430, binding = {_BINDING_ACTIVE_NODES_IN}) buffer ActiveNodesInBuf {{ ActiveNode tab[]; }} active_nodes_in;
layout(std430, binding = {_BINDING_ACTIVE_NODES_OUT}) buffer ActiveNodesOutBuf {{ ActiveNode tab[]; }} active_nodes_out;
layout(std430, binding = {_BINDING_PARENT_CELL_OFFSETS}) buffer ParentCellOffsetsBuf {{ int tab[]; }} parent_cells_offset;
layout(std430, binding = {_BINDING_CELL_OFFSETS}) buffer ChildCellOffsetsBuf {{ int tab[]; }} child_cells_offset;
layout(std430, binding = {_BINDING_PARENT_CELL_COUNTS}) buffer ParentCellCountsBuf {{ int tab[]; }} parent_cells_num_active;
layout(std430, binding = {_BINDING_CELL_COUNTS}) buffer CellCountsBuf {{ int tab[]; }} num_active_out;
layout(std430, binding = {_BINDING_COUNTERS}) buffer CountersBuf {{ int tab[]; }} counters;
layout(std430, binding = {_BINDING_CELL_ERROR_IN}) buffer CellValueInBuf {{ float tab[]; }} cell_value_in;
layout(std430, binding = {_BINDING_CELL_ERROR_OUT}) buffer CellValueOutBuf {{ float tab[]; }} cell_value_out;
layout(std430, binding = {_BINDING_OLD_TO_NEW_SCRATCH}) buffer OldToNewScratchBuf {{ uint tab[]; }} old_to_new_scratch;
layout(std430, binding = {_BINDING_TMP}) buffer TmpBuf {{ Tmp tab[]; }} tmp;
""".strip()

_COMMON_BUFFER_BLOCK_FRAGMENT_TEXTURE = f"""
const int MOPS_TEX_WIDTH = {_TEXTURE_DATA_WIDTH};

ivec2 mops_tex_coord(int texel_idx) {{
    return ivec2(texel_idx & (MOPS_TEX_WIDTH - 1), texel_idx >> 12);
}}

vec4 mops_fetch_texel(sampler2D tex, int texel_idx) {{
    return texelFetch(tex, mops_tex_coord(texel_idx), 0);
}}

float mops_fetch_word(sampler2D tex, int word_idx) {{
    vec4 texel = mops_fetch_texel(tex, word_idx >> 2);
    int lane = word_idx & 3;
    if (lane == 0) return texel.x;
    if (lane == 1) return texel.y;
    if (lane == 2) return texel.z;
    return texel.w;
}}

uint mops_fetch_bits(sampler2D tex, int idx) {{
    return uint(round(texelFetch(tex, mops_tex_coord(idx), 0).r));
}}

int mops_fetch_cell_offset(int idx) {{
    return int(mops_fetch_bits(mopsCellOffsetsTex, idx));
}}

int mops_fetch_cell_count(int idx) {{
    return int(mops_fetch_bits(mopsCellCountsTex, idx));
}}

float mops_fetch_cell_error(int idx) {{
    return texelFetch(mopsCellErrorsTex, mops_tex_coord(idx), 0).r;
}}

Primitive mops_fetch_primitive(int idx) {{
    int base = idx * {_PRIMITIVE_TEXELS};
    Primitive prim;
    vec4 t0 = mops_fetch_texel(mopsPrimsTex, base + 0);
    vec4 t1 = mops_fetch_texel(mopsPrimsTex, base + 1);
    vec4 t2 = mops_fetch_texel(mopsPrimsTex, base + 2);
    vec4 t3 = mops_fetch_texel(mopsPrimsTex, base + 3);
    vec4 t4 = mops_fetch_texel(mopsPrimsTex, base + 4);
    vec4 t5 = mops_fetch_texel(mopsPrimsTex, base + 5);
    vec4 t6 = mops_fetch_texel(mopsPrimsTex, base + 6);
    uint corner_data = uint(round(t5.z)) | (uint(round(t5.w)) << 8) | (uint(round(t6.x)) << 16) | (uint(round(t6.y)) << 24);
    prim.data = vec4(t0.xyz, uintBitsToFloat(corner_data));
    prim.m_row0 = t1;
    prim.m_row1 = t2;
    prim.m_row2 = t3;
    prim.extrude_rounding = t4.xy;
    prim.type = int(round(t0.w));
    prim.bevel = t4.z;
    prim.color = uint(round(t4.w)) | (uint(round(t5.x)) << 8) | (uint(round(t5.y)) << 16);
    prim.pad0 = t6.z;
    prim.pad1 = 0.0;
    prim.pad2 = 0.0;
    return prim;
}}

Node mops_fetch_node(int idx) {{
    vec4 t = mops_fetch_texel(mopsNodesTex, idx);
    Node node;
    node.type = int(round(t.x));
    node.idx_in_type = int(round(t.y));
    return node;
}}

BinaryOp mops_fetch_binary_op(int idx) {{
    vec4 t = mops_fetch_texel(mopsBinaryOpsTex, idx);
    BinaryOp op;
    uint sign_bit = uint(round(t.y)) & 1u;
    uint op_bits = uint(round(t.z)) & 3u;
    op.blend_factor_and_sign = (floatBitsToUint(t.x) & (~7u)) | sign_bit | (op_bits << 1);
    return op;
}}

ActiveNode mops_fetch_active_node(int idx) {{
    ActiveNode node;
    node.idx_and_sign = mops_fetch_bits(mopsActiveNodesTex, idx);
    return node;
}}
""".strip()

_COMMON_BUFFER_BLOCK_COMPUTE_TEXTURE = f"""
const int MOPS_TEX_WIDTH = {_TEXTURE_DATA_WIDTH};

ivec2 mops_tex_coord(int texel_idx) {{
    return ivec2(texel_idx & (MOPS_TEX_WIDTH - 1), texel_idx >> 12);
}}

ivec2 mops_tex_coord(uint texel_idx) {{
    return mops_tex_coord(int(texel_idx));
}}

vec4 mops_fetch_texel(sampler2D tex, int texel_idx) {{
    return texelFetch(tex, mops_tex_coord(texel_idx), 0);
}}

float mops_fetch_word(sampler2D tex, int word_idx) {{
    vec4 texel = mops_fetch_texel(tex, word_idx >> 2);
    int lane = word_idx & 3;
    if (lane == 0) return texel.x;
    if (lane == 1) return texel.y;
    if (lane == 2) return texel.z;
    return texel.w;
}}

uint mops_fetch_bits(sampler2D tex, int idx) {{
    return uint(round(texelFetch(tex, mops_tex_coord(idx), 0).r));
}}

int mops_counter_add(int idx, int value) {{
    return int(imageAtomicAdd(mopsCountersImg, mops_tex_coord(idx), uint(value)));
}}

void mops_counter_flag(int idx) {{
    imageAtomicAdd(mopsCountersImg, mops_tex_coord(idx), 1u);
}}

Primitive mops_fetch_primitive(int idx) {{
    int base = idx * {_PRIMITIVE_TEXELS};
    Primitive prim;
    vec4 t0 = mops_fetch_texel(mopsPrimsTex, base + 0);
    vec4 t1 = mops_fetch_texel(mopsPrimsTex, base + 1);
    vec4 t2 = mops_fetch_texel(mopsPrimsTex, base + 2);
    vec4 t3 = mops_fetch_texel(mopsPrimsTex, base + 3);
    vec4 t4 = mops_fetch_texel(mopsPrimsTex, base + 4);
    vec4 t5 = mops_fetch_texel(mopsPrimsTex, base + 5);
    vec4 t6 = mops_fetch_texel(mopsPrimsTex, base + 6);
    uint corner_data = uint(round(t5.z)) | (uint(round(t5.w)) << 8) | (uint(round(t6.x)) << 16) | (uint(round(t6.y)) << 24);
    prim.data = vec4(t0.xyz, uintBitsToFloat(corner_data));
    prim.m_row0 = t1;
    prim.m_row1 = t2;
    prim.m_row2 = t3;
    prim.extrude_rounding = t4.xy;
    prim.type = int(round(t0.w));
    prim.bevel = t4.z;
    prim.color = uint(round(t4.w)) | (uint(round(t5.x)) << 8) | (uint(round(t5.y)) << 16);
    prim.pad0 = t6.z;
    prim.pad1 = 0.0;
    prim.pad2 = 0.0;
    return prim;
}}

Node mops_fetch_node(int idx) {{
    vec4 t = mops_fetch_texel(mopsNodesTex, idx);
    Node node;
    node.type = int(round(t.x));
    node.idx_in_type = int(round(t.y));
    return node;
}}

BinaryOp mops_fetch_binary_op(int idx) {{
    vec4 t = mops_fetch_texel(mopsBinaryOpsTex, idx);
    BinaryOp op;
    uint sign_bit = uint(round(t.y)) & 1u;
    uint op_bits = uint(round(t.z)) & 3u;
    op.blend_factor_and_sign = (floatBitsToUint(t.x) & (~7u)) | sign_bit | (op_bits << 1);
    return op;
}}

ActiveNode active_nodes_in_get(int idx) {{
    ActiveNode node;
    node.idx_and_sign = mops_fetch_bits(mopsActiveNodesInTex, idx);
    return node;
}}

void active_nodes_out_set(int idx, ActiveNode value) {{
    imageStore(mopsActiveNodesOutImg, mops_tex_coord(idx), vec4(float(value.idx_and_sign), 0.0, 0.0, 0.0));
}}

void active_nodes_out_set(uint idx, ActiveNode value) {{
    active_nodes_out_set(int(idx), value);
}}

uint mops_parents_in_get(int idx) {{
    return mops_fetch_bits(mopsParentsInTex, idx);
}}

void mops_parents_out_set(int idx, uint value) {{
    imageStore(mopsParentsOutImg, mops_tex_coord(idx), vec4(float(value), 0.0, 0.0, 0.0));
}}

void mops_parents_out_set(uint idx, uint value) {{
    mops_parents_out_set(int(idx), value);
}}

int mops_parent_cells_offset_get(int idx) {{
    return int(mops_fetch_bits(mopsParentCellOffsetsTex, idx));
}}

void mops_child_cells_offset_set(int idx, int value) {{
    imageStore(mopsChildCellOffsetsImg, mops_tex_coord(idx), vec4(float(value), 0.0, 0.0, 0.0));
}}

void mops_child_cells_offset_set(uint idx, int value) {{
    mops_child_cells_offset_set(int(idx), value);
}}

int mops_parent_cells_count_get(int idx) {{
    return int(mops_fetch_bits(mopsParentCellCountsTex, idx));
}}

void mops_cell_counts_set(int idx, int value) {{
    imageStore(mopsCellCountsImg, mops_tex_coord(idx), vec4(float(value), 0.0, 0.0, 0.0));
}}

void mops_cell_counts_set(uint idx, int value) {{
    mops_cell_counts_set(int(idx), value);
}}

float mops_cell_value_in_get(int idx) {{
    return texelFetch(mopsCellValueInTex, mops_tex_coord(idx), 0).r;
}}

void mops_cell_value_out_set(int idx, float value) {{
    imageStore(mopsCellValueOutImg, mops_tex_coord(idx), vec4(value, 0.0, 0.0, 0.0));
}}

void mops_cell_value_out_set(uint idx, float value) {{
    mops_cell_value_out_set(int(idx), value);
}}

uint mops_old_to_new_get(int idx) {{
    return imageLoad(mopsOldToNewImg, mops_tex_coord(idx)).r;
}}

uint mops_old_to_new_get(uint idx) {{
    return mops_old_to_new_get(int(idx));
}}

void mops_old_to_new_set(int idx, uint value) {{
    imageStore(mopsOldToNewImg, mops_tex_coord(idx), uvec4(value, 0u, 0u, 0u));
}}

void mops_old_to_new_set(uint idx, uint value) {{
    mops_old_to_new_set(int(idx), value);
}}

Tmp mops_tmp_get(int idx) {{
    Tmp t;
    t.x = imageLoad(mopsTmpImg, mops_tex_coord(idx)).r;
    return t;
}}

Tmp mops_tmp_get(uint idx) {{
    return mops_tmp_get(int(idx));
}}

void mops_tmp_set(int idx, Tmp value) {{
    imageStore(mopsTmpImg, mops_tex_coord(idx), uvec4(value.x, 0u, 0u, 0u));
}}

void mops_tmp_set(uint idx, Tmp value) {{
    mops_tmp_set(int(idx), value);
}}

void mops_tmp_state_write(int idx, int state) {{
    Tmp t = mops_tmp_get(idx);
    t.x &= ~3u;
    t.x |= uint(state);
    mops_tmp_set(idx, t);
}}

void mops_tmp_state_write(uint idx, int state) {{
    mops_tmp_state_write(int(idx), state);
}}
""".strip()

_SHADER_CACHE = {}
_GL_FUNCTIONS_CACHE = None

_PCG_BLOCK_32 = """
#define PCG uvec2

uint rand_uint32(inout PCG pcg)
{
    pcg.x = 1664525u * pcg.x + 1013904223u + pcg.y;
    pcg.y ^= pcg.x * 747796405u + 2891336453u;
    uint x = pcg.x ^ (pcg.x >> 16);
    x *= 2246822519u;
    x ^= x >> 13;
    x *= 3266489917u;
    return x ^ (x >> 16);
}

void init_pcg(inout PCG pcg, uint seed)
{
    pcg.x = seed ^ 0xA511E9B3u;
    pcg.y = 0x9E3779B9u;
    rand_uint32(pcg);
}

float rand_float_0_1(inout PCG pcg) { return rand_uint32(pcg) * 2.32830616e-10f; }
""".strip()

_COMPUTE_PUSH_CONSTANTS = (
    ("VEC4", "aabb_min"),
    ("VEC4", "aabb_max"),
    ("INT", "total_num_nodes"),
    ("INT", "grid_size"),
    ("INT", "first_lvl"),
    ("INT", "active_capacity"),
    ("INT", "tmp_capacity"),
    ("INT", "active_counter_idx"),
    ("INT", "old_to_new_counter_idx"),
    ("INT", "status_counter_idx"),
)

_FRAGMENT_PARAMS_TYPEDEF = """
struct ViewportParams {
  vec4 aabb_min;
  vec4 aabb_max;
  ivec4 ints0;
  ivec4 ints1;
  vec4 floats0;
  mat4 u_mvp;
  mat4 u_view;
  mat4 u_view_inv;
  mat4 u_proj_inv;
  vec4 u_cam0;
  vec4 u_cam1;
  vec4 u_cam2;
  vec4 u_cam3;
};

#define mops_u_Resolution params.ints0.xy
#define mops_total_num_nodes params.ints0.z
#define mops_grid_size params.ints0.w
#define mops_aabb_min params.aabb_min
#define mops_aabb_max params.aabb_max
#define mops_culling_enabled params.ints1.y
#define mops_num_samples params.ints1.z
#define mops_show_specular params.ints1.w
#define mops_viz_max params.floats0.x
#define mops_alpha params.floats0.y
#define mops_gamma params.floats0.z
#define mops_u_mvp params.u_mvp
#define mops_u_view params.u_view
#define mops_u_view_inv params.u_view_inv
#define mops_u_proj_inv params.u_proj_inv
#define mops_u_cam0 params.u_cam0
#define mops_u_cam1 params.u_cam1
#define mops_u_cam2 params.u_cam2
#define mops_u_cam3 params.u_cam3
""".strip()


class _FragmentParamsUBO(ctypes.Structure):
    _fields_ = [
        ("aabb_min", ctypes.c_float * 4),
        ("aabb_max", ctypes.c_float * 4),
        ("ints0", ctypes.c_int * 4),
        ("ints1", ctypes.c_int * 4),
        ("floats0", ctypes.c_float * 4),
        ("u_mvp", ctypes.c_float * 16),
        ("u_view", ctypes.c_float * 16),
        ("u_view_inv", ctypes.c_float * 16),
        ("u_proj_inv", ctypes.c_float * 16),
        ("u_cam0", ctypes.c_float * 4),
        ("u_cam1", ctypes.c_float * 4),
        ("u_cam2", ctypes.c_float * 4),
        ("u_cam3", ctypes.c_float * 4),
    ]


def _flatten_matrix_column_major(matrix: Matrix) -> tuple[float, ...]:
    return tuple(float(matrix[row][col]) for col in range(4) for row in range(4))


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


_TEXTURE_COUNTER_READBACK = bool(_int_env("MATHOPS_V2_TEXTURE_COUNTER_READBACK", 1))
_VIEWPORT_FAST = bool(_int_env("MATHOPS_V2_VIEWPORT_FAST", 0))


def _shader_root() -> Path:
    for candidate in (bridge.repo_dir() / "shaders", bridge.addon_dir() / "shaders"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("MathOPS-v2 shader source directory not found")


def _expand_includes(path: Path, cache: dict[Path, str]) -> str:
    path = path.resolve()
    cached = cache.get(path)
    if cached is not None:
        return cached

    text = path.read_text(encoding="utf-8")

    def replace(match: re.Match[str]) -> str:
        include_path = (path.parent / match.group(1)).resolve()
        included = _expand_includes(include_path, cache)
        return included if included.endswith("\n") else included + "\n"

    expanded = _INCLUDE_RE.sub(replace, text)
    cache[path] = expanded
    return expanded


def _replace_common_buffer_refs(
    source: str, stage: str, use_int16_storage: bool
) -> str:
    if stage == "compute":
        block = (
            _COMMON_BUFFER_BLOCK_COMPUTE_U16
            if use_int16_storage
            else _COMMON_BUFFER_BLOCK_COMPUTE_U32
        )
    else:
        block = (
            _COMMON_BUFFER_BLOCK_FRAGMENT_U16
            if use_int16_storage
            else _COMMON_BUFFER_BLOCK_FRAGMENT_U32
        )
    replaced, count = _BUFFER_REFERENCE_RE.subn(
        block + "\n\nvoid set_bit", source, count=1
    )
    if count != 1:
        raise RuntimeError("Failed to adapt common.glsl buffer references")
    return replaced


def _replace_texture_fragment_refs(source: str) -> str:
    replaced, count = _BUFFER_REFERENCE_RE.subn(
        _COMMON_BUFFER_BLOCK_FRAGMENT_TEXTURE + "\n\nvoid set_bit", source, count=1
    )
    if count != 1:
        raise RuntimeError("Failed to adapt fragment shader texture references")

    replacements = (
        (r"\bprims\.tab\[([^\]]+)\]", r"mops_fetch_primitive(\1)"),
        (r"\bnodes\.tab\[([^\]]+)\]", r"mops_fetch_node(\1)"),
        (r"\bbinary_ops\.tab\[([^\]]+)\]", r"mops_fetch_binary_op(\1)"),
        (r"\bactive_nodes_out\.tab\[([^\]]+)\]", r"mops_fetch_active_node(\1)"),
        (r"\bcells_offset\.tab\[([^\]]+)\]", r"mops_fetch_cell_offset(\1)"),
        (r"\bcells_num_active\.tab\[([^\]]+)\]", r"mops_fetch_cell_count(\1)"),
        (r"\bcell_error_out\.tab\[([^\]]+)\]", r"mops_fetch_cell_error(\1)"),
    )
    for pattern, replacement in replacements:
        replaced = re.sub(pattern, replacement, replaced)
    return replaced


def _replace_texture_compute_refs(source: str) -> str:
    replaced = source.replace(
        _COMMON_BUFFER_BLOCK_COMPUTE_U32, _COMMON_BUFFER_BLOCK_COMPUTE_TEXTURE, 1
    )
    if replaced == source:
        raise RuntimeError("Failed to adapt compute shader texture references")

    assignment_replacements = (
        (
            r"(^\s*)active_nodes_out\.tab\[(.*?)\]\s*=\s*(.*?);$",
            r"\1active_nodes_out_set(\2, \3);",
        ),
        (
            r"(^\s*)parents_out\.tab\[(.*?)\]\s*=\s*(.*?);$",
            r"\1mops_parents_out_set(\2, \3);",
        ),
        (
            r"(^\s*)child_cells_offset\.tab\[(.*?)\]\s*=\s*(.*?);$",
            r"\1mops_child_cells_offset_set(\2, \3);",
        ),
        (
            r"(^\s*)num_active_out\.tab\[(.*?)\]\s*=\s*(.*?);$",
            r"\1mops_cell_counts_set(\2, \3);",
        ),
        (
            r"(^\s*)cell_value_out\.tab\[(.*?)\]\s*=\s*(.*?);$",
            r"\1mops_cell_value_out_set(\2, \3);",
        ),
        (
            r"(^\s*)old_to_new_scratch\.tab\[(.*?)\]\s*=\s*(.*?);$",
            r"\1mops_old_to_new_set(\2, \3);",
        ),
        (r"(^\s*)tmp\.tab\[(.*?)\]\s*=\s*(.*?);$", r"\1mops_tmp_set(\2, \3);"),
        (r"(^\s*)counters\.tab\[(.*?)\]\s*=\s*1\s*;$", r"\1mops_counter_flag(\2);"),
    )
    for pattern, replacement in assignment_replacements:
        replaced = re.sub(pattern, replacement, replaced, flags=re.M)

    direct_replacements = (
        (r"atomicAdd\(counters\.tab\[(.*?)\],\s*(.*?)\)", r"mops_counter_add(\1, \2)"),
        (r"\bprims\.tab\[([^\]]+)\]", r"mops_fetch_primitive(\1)"),
        (r"\bnodes\.tab\[([^\]]+)\]", r"mops_fetch_node(\1)"),
        (r"\bbinary_ops\.tab\[([^\]]+)\]", r"mops_fetch_binary_op(\1)"),
        (r"\bactive_nodes_in\.tab\[([^\]]+)\]", r"active_nodes_in_get(\1)"),
        (r"\bparents_in\.tab\[([^\]]+)\]", r"mops_parents_in_get(\1)"),
        (
            r"\bparent_cells_offset\.tab\[([^\]]+)\]",
            r"mops_parent_cells_offset_get(\1)",
        ),
        (
            r"\bparent_cells_num_active\.tab\[([^\]]+)\]",
            r"mops_parent_cells_count_get(\1)",
        ),
        (r"\bcell_value_in\.tab\[([^\]]+)\]", r"mops_cell_value_in_get(\1)"),
        (r"\bold_to_new_scratch\.tab\[([^\]]+)\]", r"mops_old_to_new_get(\1)"),
        (r"\btmp\.tab\[([^\]]+)\]", r"mops_tmp_get(\1)"),
    )
    for pattern, replacement in direct_replacements:
        replaced = re.sub(pattern, replacement, replaced)
    replaced = re.sub(
        r"Tmp_state_write\(mops_tmp_get\((.*?)\),\s*(.*?)\)",
        r"mops_tmp_state_write(\1, \2)",
        replaced,
    )
    return replaced


def _replace_push_constants(source: str, stage: str) -> str:
    replaced, count = _PUSH_CONSTANT_RE.subn("", source, count=1)
    if count != 1:
        raise RuntimeError(f"Failed to adapt {stage} shader uniforms")
    return replaced


def _transform_shader_source(
    source: str,
    stage: str,
    shading_mode_value: int | None = None,
    use_int16_storage: bool = False,
    texture_backend: bool = False,
) -> str:
    source = re.sub(r"^\s*#version[^\n]*\n", "", source, flags=re.M)
    source = re.sub(r"^\s*#extension[^\n]*\n", "", source, flags=re.M)
    source = source.replace(
        "layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;", ""
    )
    if not use_int16_storage:
        source = source.replace("uint16_t", "uint")
    if texture_backend and stage == "fragment":
        source = _replace_texture_fragment_refs(source)
    else:
        source = _replace_common_buffer_refs(source, stage, use_int16_storage)
    source = _replace_push_constants(source, stage)
    source = _UNUSED_UINT64_RE.sub("", source, count=1)
    source = _PCG_BLOCK_RE.sub(_PCG_BLOCK_32 + "\n", source, count=1)
    source = source.replace("cam.tab[0]", "u_cam0")
    source = source.replace("cam.tab[1]", "u_cam1")
    source = source.replace("cam.tab[2]", "u_cam2")
    source = source.replace("cam.tab[3]", "u_cam3")
    source = source.replace("mvp.m", "u_mvp")

    if stage == "compute":
        if not texture_backend:
            source = source.replace(
                "shared ActiveNode s_parent_active_nodes[64];\nshared uint s_parent_node_parents[64];\n\n#define INVALID_INDEX 0xffffu\n",
                "#define INVALID_INDEX 0xffffu\n",
            )
            source = re.sub(
                r"\n\s*if \(block\*64\+gl_LocalInvocationIndex < num_nodes\) \{\n\s*s_parent_active_nodes\[gl_LocalInvocationIndex\] = active_nodes_in\.tab\[parent_offset \+ block\*64 \+ gl_LocalInvocationIndex\];\n\s*\}\n\s*barrier\(\);\n",
                "\n",
                source,
            )
            source = re.sub(
                r"int tmp_offset = -1;\s*if \(subgroupElect\(\)\) \{\s*tmp_offset = atomicAdd\(old_to_new_count\.val, int\(gl_SubgroupSize\)\*num_nodes\);\s*\}\s*tmp_offset = subgroupBroadcastFirst\(tmp_offset\);",
                "int tmp_offset = atomicAdd(old_to_new_count.val, num_nodes);",
                source,
                flags=re.S,
            )
            source = re.sub(r"gl_SubgroupSize\s*\*\s*", "", source)
            source = re.sub(r"\s*\*\s*gl_SubgroupSize", "", source)
            source = re.sub(r"\s*\+\s*gl_SubgroupInvocationID", "", source)
        source = source.replace(
            "int new_parent_old_idx = Tmp_parent_get(tmp_i);",
            "int new_parent_old_idx = int(Tmp_parent_get(tmp_i));",
        )
        source = source.replace("active_count.val", "counters.tab[active_counter_idx]")
        source = source.replace(
            "old_to_new_count.val", "counters.tab[old_to_new_counter_idx]"
        )
        source = source.replace(
            "status_buf.overflow", "counters.tab[status_counter_idx]"
        )
        if use_int16_storage:
            source = source.replace(
                "return int(n.idx_and_sign & ~(1 << 15));",
                "return int(uint(n.idx_and_sign) & ~(1u << 15));",
            )
            source = source.replace(
                "return (n.idx_and_sign >> 15 == 0);",
                "return ((uint(n.idx_and_sign) >> 15) == 0u);",
            )
            source = source.replace(
                "    return ActiveNode(v);",
                "    return ActiveNode(uint16_t(v));",
            )
            source = source.replace(
                "tmp.tab[tmp_offset + parent_idx]",
                "tmp.tab[tmp_offset + int(parent_idx)]",
            )
            source = source.replace(
                "uint16_t new_parent_idx = new_parent_old_idx != INVALID_INDEX ? old_to_new_scratch.tab[tmp_offset + new_parent_old_idx] : uint16_t(INVALID_INDEX);",
                "uint16_t new_parent_idx = uint16_t(new_parent_old_idx != INVALID_INDEX ? old_to_new_scratch.tab[tmp_offset + new_parent_old_idx] : uint(INVALID_INDEX));",
            )
        if not use_int16_storage and not texture_backend:
            source = re.sub(
                r"(^\s*)active_nodes_out\.tab\[(.*?)\]\s*=\s*(.*?);$",
                r"\1active_nodes_out_set(\2, \3);",
                source,
                flags=re.M,
            )
            source = re.sub(
                r"active_nodes_in\.tab\[([^\]]+)\]",
                r"active_nodes_in_get(\1)",
                source,
            )
            source = source.replace(
                "    uint packed_word = active_nodes_in_get(idx >> 1);\n",
                "    uint packed_word = active_nodes_in.tab[idx >> 1];\n",
            )
            source = source.replace(
                "    active_nodes_out_set(word_idx, (word & ~mask) | ((value.idx_and_sign & 0xffffu) << shift));\n",
                "    active_nodes_out.tab[word_idx] = (word & ~mask) | ((value.idx_and_sign & 0xffffu) << shift);\n",
            )
            source = source.replace(
                "        int cell_offset = atomicAdd(counters.tab[active_counter_idx], 1);\n",
                "        int cell_offset = atomicAdd(counters.tab[active_counter_idx], 2);\n",
            )
        source = source.replace(
            "    int tmp_offset = atomicAdd(counters.tab[old_to_new_counter_idx], num_nodes);\n",
            "    int tmp_offset = atomicAdd(counters.tab[old_to_new_counter_idx], num_nodes);\n"
            "    if (tmp_offset + num_nodes > tmp_capacity) {\n"
            "        counters.tab[status_counter_idx] = 1;\n"
            "        num_active_out.tab[cell_idx] = 0;\n"
            "        child_cells_offset.tab[cell_idx] = 0;\n"
            "        cell_value_out.tab[cell_idx] = 0.0;\n"
            "        return;\n"
            "    }\n",
        )
        source = source.replace(
            "    int cell_offset = atomicAdd(counters.tab[active_counter_idx], cell_num_active);\n",
            (
                "    int cell_num_active_padded = cell_num_active;\n"
                if use_int16_storage or texture_backend
                else "    int cell_num_active_padded = (cell_num_active + 1) & ~1;\n"
            )
            + "    int cell_offset = atomicAdd(counters.tab[active_counter_idx], cell_num_active_padded);\n"
            "    if (cell_offset + cell_num_active_padded > active_capacity) {\n"
            "        counters.tab[status_counter_idx] = 1;\n"
            "        num_active_out.tab[cell_idx] = 0;\n"
            "        child_cells_offset.tab[cell_idx] = 0;\n"
            "        cell_value_out.tab[cell_idx] = 0.0;\n"
            "        return;\n"
            "    }\n",
        )
        if texture_backend:
            source = _replace_texture_compute_refs(source)
    else:
        source = _FRAGMENT_OUTPUT_RE.sub("", source, count=1)
        if shading_mode_value is None:
            raise RuntimeError("Fragment shader requires a shading mode value")
        source = _FRAGMENT_SHADING_MODE_RE.sub(
            f"const int shading_mode = {int(shading_mode_value)};\n",
            source,
            count=1,
        )
        fragment_aliases = (
            ("u_Resolution", "mops_u_Resolution"),
            ("total_num_nodes", "mops_total_num_nodes"),
            ("grid_size", "mops_grid_size"),
            ("aabb_min", "mops_aabb_min"),
            ("aabb_max", "mops_aabb_max"),
            ("culling_enabled", "mops_culling_enabled"),
            ("num_samples", "mops_num_samples"),
            ("viz_max", "mops_viz_max"),
            ("alpha", "mops_alpha"),
            ("gamma", "mops_gamma"),
            ("u_mvp", "mops_u_mvp"),
            ("u_cam0", "mops_u_cam0"),
            ("u_cam1", "mops_u_cam1"),
            ("u_cam2", "mops_u_cam2"),
            ("u_cam3", "mops_u_cam3"),
        )
        for original, replacement in fragment_aliases:
            source = re.sub(rf"\b{original}\b", replacement, source)
        source = source.replace(
            "uint get_cell_idx(ivec3 cell, int mops_grid_size) {",
            "uint get_cell_idx(ivec3 cell, int grid_size_local) {",
        )
        source = source.replace(
            "ivec3 get_cell(uint cell_idx, int mops_grid_size) {",
            "ivec3 get_cell(uint cell_idx, int grid_size_local) {",
        )
        source = source.replace(
            "    cell.z = int(cell_idx / (mops_grid_size*mops_grid_size));\n"
            "    cell.y = int((cell_idx % (mops_grid_size*mops_grid_size)) / mops_grid_size);\n"
            "    cell.x = int(cell_idx % mops_grid_size);\n",
            "    cell.z = int(cell_idx / (grid_size_local * grid_size_local));\n"
            "    cell.y = int((cell_idx % (grid_size_local * grid_size_local)) / grid_size_local);\n"
            "    cell.x = int(cell_idx % grid_size_local);\n",
        )
        source = source.replace(
            "uint get_parent_cell_idx(uint cell_idx, int mops_grid_size) {",
            "uint get_parent_cell_idx(uint cell_idx, int grid_size_local) {",
        )
        source = source.replace(
            "//ivec3 cell = get_cell(cell_idx, mops_grid_size);\n"
            "    //ivec3 parent_cell = cell / 2;\n"
            "    //return get_cell_idx(parent_cell, mops_grid_size/2);\n",
            "//ivec3 cell = get_cell(cell_idx, grid_size_local);\n"
            "    //ivec3 parent_cell = cell / 2;\n"
            "    //return get_cell_idx(parent_cell, grid_size_local/2);\n",
        )
        source = re.sub(
            r"init_pcg\(pcg,\s*uint64_t\(gl_FragCoord\.y\)\s*\*\s*uint64_t\([^\)]+\.x\)\s*\+\s*uint64_t\(gl_FragCoord\.x\)\s*\);",
            "init_pcg(pcg, uint(gl_FragCoord.y) * uint(mops_u_Resolution.x) + uint(gl_FragCoord.x));",
            source,
        )
        source = source.replace("        uv.y = 1 - uv.y;\n", "")
        source = source.replace(
            "    int num_active = cells_num_active.tab[cell_idx];\n",
            "    int num_active = bool(mops_culling_enabled) ? cells_num_active.tab[cell_idx] : mops_total_num_nodes;\n",
        )
        source = source.replace(
            "            gl_FragDepth = projected_depth;\n",
            "",
        )
        source = source.replace("        gl_FragDepth = 1;\n", "")
        source = source.replace(
            "    outColor /= mops_num_samples;\n",
            "    outColor /= float(max(mops_num_samples, 1));\n",
        )
        if not use_int16_storage:
            source = re.sub(
                r"active_nodes_out\.tab\[([^\]]+)\]",
                r"active_nodes_out_get(\1)",
                source,
            )
            source = source.replace(
                "    uint packed_word = active_nodes_out_get(idx >> 1);\n",
                "    uint packed_word = active_nodes_out.tab[idx >> 1];\n",
            )
        else:
            source = source.replace(
                "return int(n.idx_and_sign & ~(1 << 15));",
                "return int(uint(n.idx_and_sign) & ~(1u << 15));",
            )
            source = source.replace(
                "return (n.idx_and_sign >> 15 == 0);",
                "return ((uint(n.idx_and_sign) >> 15) == 0u);",
            )
            source = source.replace(
                "    return ActiveNode(v);",
                "    return ActiveNode(uint16_t(v));",
            )

    prefix_lines = []
    if texture_backend and stage == "compute":
        prefix_lines.extend(
            (
                "#extension GL_KHR_shader_subgroup_basic : require",
                "#extension GL_KHR_shader_subgroup_vote : require",
                "#extension GL_KHR_shader_subgroup_ballot : require",
            )
        )
    if use_int16_storage:
        prefix_lines.append(_GLSL_INT16_HEADER)
    if stage == "fragment":
        prefix_lines.append("#define MATHOPS_BLENDER_VIEWPORT 1")
    if stage != "compute" and _VIEWPORT_FAST:
        prefix_lines.append("#define MATHOPS_VIEWPORT_FAST 1")
    if prefix_lines:
        source = "\n".join(prefix_lines) + "\n" + source.lstrip()

    return source.strip() + "\n"


def _shader_source(
    filename: str,
    stage: str,
    shading_mode_value: int | None = None,
    use_int16_storage: bool = False,
    texture_backend: bool = False,
) -> str:
    source_token = _shader_source_token()
    key = (
        filename,
        stage,
        shading_mode_value,
        use_int16_storage,
        texture_backend,
        source_token,
    )
    cached = _SHADER_CACHE.get(key)
    if cached is not None:
        return cached

    source = _expand_includes(_shader_root() / filename, {})
    source = _transform_shader_source(
        source,
        stage,
        shading_mode_value,
        use_int16_storage=use_int16_storage,
        texture_backend=texture_backend,
    )
    _SHADER_CACHE[key] = source
    return source


def _shader_source_token() -> tuple[int, ...]:
    root = _shader_root()
    mtimes = []
    for filename in _VIEWPORT_SHADER_FILES:
        mtimes.append((root / filename).stat().st_mtime_ns)
    mtimes.append((bridge.repo_dir() / "include" / "constants.h").stat().st_mtime_ns)
    return tuple(mtimes)


def _declare_push_constants(shader_info, entries):
    for attr_type, name in entries:
        shader_info.push_constant(attr_type, name)


def _gl_functions():
    global _GL_FUNCTIONS_CACHE
    if _GL_FUNCTIONS_CACHE is not None:
        return _GL_FUNCTIONS_CACHE

    c_int = ctypes.c_int
    c_uint = ctypes.c_uint
    c_ptr_int = ctypes.POINTER(ctypes.c_int)
    c_ptr_uint = ctypes.POINTER(ctypes.c_uint)
    c_size = ctypes.c_ssize_t
    c_void_p = ctypes.c_void_p

    if sys.platform == "win32":
        gl_lib = ctypes.windll.opengl32
        wgl_get_proc_address = gl_lib.wglGetProcAddress
        wgl_get_proc_address.restype = ctypes.c_void_p
        wgl_get_proc_address.argtypes = [ctypes.c_char_p]

        def get_proc(name, signature):
            address = wgl_get_proc_address(name)
            if address in (None, 0, 1, 2, 3, ctypes.c_void_p(-1).value):
                fn = getattr(gl_lib, name.decode("ascii"), None)
                if fn is None:
                    raise RuntimeError(f"OpenGL function {name!r} is unavailable")
                fn.restype = signature._restype_
                fn.argtypes = signature._argtypes_
                return fn
            return ctypes.cast(address, signature)

        gl_get_integerv = gl_lib.glGetIntegerv
    else:
        gl_lib = ctypes.CDLL(
            "/System/Library/Frameworks/OpenGL.framework/OpenGL"
            if sys.platform == "darwin"
            else "libGL.so.1"
        )

        def get_proc(name, signature):
            fn = getattr(gl_lib, name.decode("ascii"))
            fn.restype = signature._restype_
            fn.argtypes = signature._argtypes_
            return fn

        gl_get_integerv = gl_lib.glGetIntegerv

    gl_get_integerv.argtypes = [c_uint, c_ptr_int]

    _GL_FUNCTIONS_CACHE = {
        "glGetIntegerv": gl_get_integerv,
        "glBindBufferBase": get_proc(
            b"glBindBufferBase", ctypes.CFUNCTYPE(None, c_uint, c_uint, c_uint)
        ),
        "glBindBufferRange": get_proc(
            b"glBindBufferRange",
            ctypes.CFUNCTYPE(None, c_uint, c_uint, c_uint, c_size, c_size),
        ),
        "glGenBuffers": get_proc(
            b"glGenBuffers", ctypes.CFUNCTYPE(None, c_int, c_ptr_uint)
        ),
        "glDeleteBuffers": get_proc(
            b"glDeleteBuffers", ctypes.CFUNCTYPE(None, c_int, c_ptr_uint)
        ),
        "glBindBuffer": get_proc(
            b"glBindBuffer", ctypes.CFUNCTYPE(None, c_uint, c_uint)
        ),
        "glBufferData": get_proc(
            b"glBufferData", ctypes.CFUNCTYPE(None, c_uint, c_size, c_void_p, c_uint)
        ),
        "glBufferSubData": get_proc(
            b"glBufferSubData",
            ctypes.CFUNCTYPE(None, c_uint, c_size, c_size, c_void_p),
        ),
        "glGetBufferSubData": get_proc(
            b"glGetBufferSubData",
            ctypes.CFUNCTYPE(None, c_uint, c_size, c_size, c_void_p),
        ),
        "glMemoryBarrier": get_proc(b"glMemoryBarrier", ctypes.CFUNCTYPE(None, c_uint)),
    }
    return _GL_FUNCTIONS_CACHE


class _SSBO:
    TARGET = 0x90D2
    DYNAMIC_DRAW = 0x88E8

    def __init__(self):
        self.buffer_id = 0
        self.size = 0

    def ensure_size(self, size: int):
        size = max(int(size), 1)
        if self.buffer_id == 0:
            buf = ctypes.c_uint(0)
            _gl_functions()["glGenBuffers"](1, ctypes.byref(buf))
            self.buffer_id = buf.value
        if size == self.size:
            return
        _gl_functions()["glBindBuffer"](self.TARGET, self.buffer_id)
        _gl_functions()["glBufferData"](
            self.TARGET,
            size,
            ctypes.c_void_p(0),
            self.DYNAMIC_DRAW,
        )
        _gl_functions()["glBindBuffer"](self.TARGET, 0)
        self.size = size

    def upload(self, data: bytes):
        self.ensure_size(len(data))
        self.update(data)

    def update(self, data: bytes, offset: int = 0):
        if self.buffer_id == 0:
            self.ensure_size(offset + len(data))
        if offset + len(data) > self.size:
            raise RuntimeError("Buffer update exceeds allocated size")
        payload = ctypes.create_string_buffer(
            data if data else b"\x00", max(len(data), 1)
        )
        _gl_functions()["glBindBuffer"](self.TARGET, self.buffer_id)
        _gl_functions()["glBufferSubData"](
            self.TARGET,
            offset,
            max(len(data), 1),
            ctypes.cast(payload, ctypes.c_void_p),
        )
        _gl_functions()["glBindBuffer"](self.TARGET, 0)

    def bind(self, binding: int):
        if self.buffer_id != 0:
            _gl_functions()["glBindBufferBase"](self.TARGET, binding, self.buffer_id)

    def bind_range(self, binding: int, offset: int, size: int):
        if self.buffer_id != 0:
            _gl_functions()["glBindBufferRange"](
                self.TARGET, binding, self.buffer_id, offset, size
            )

    def read(self, size: int, offset: int = 0) -> bytes:
        if self.buffer_id == 0 or size <= 0:
            return b""
        payload = ctypes.create_string_buffer(size)
        _gl_functions()["glBindBuffer"](self.TARGET, self.buffer_id)
        _gl_functions()["glGetBufferSubData"](
            self.TARGET,
            offset,
            size,
            ctypes.cast(payload, ctypes.c_void_p),
        )
        _gl_functions()["glBindBuffer"](self.TARGET, 0)
        return payload.raw[:size]

    def free(self):
        if self.buffer_id != 0:
            buf = ctypes.c_uint(self.buffer_id)
            _gl_functions()["glDeleteBuffers"](1, ctypes.byref(buf))
            self.buffer_id = 0
            self.size = 0


class MathOPSV2GPUViewport:
    def __init__(self):
        self.draw_shader = None
        self.draw_shader_mode = None
        self.draw_shader_backend = None
        self.compute_shader = None
        self.texture_scene_mode = False
        self.use_int16_storage = True
        self.int16_storage_supported = (
            True if _int_env("MATHOPS_V2_TRY_GL_INT16", 1) else False
        )
        self.batch = None
        self.offscreen = None
        self.offscreen_size = (0, 0)
        self.scene_key = None
        self.pruning_key = None
        self.scene_info = None
        self.logged_backend_warning = False
        self.logged_shader_failure = False
        self.runtime_failed = False
        self.last_overflow_message = ""
        self.frame_culling_ms = 0.0
        self.active_capacity = 0
        self.tmp_capacity = 0
        self.max_active_limit = _int_env(
            "MATHOPS_V2_VIEWPORT_MAX_ACTIVE_HARD", _MAX_ACTIVE_CAP
        )
        self.max_tmp_limit = _int_env("MATHOPS_V2_VIEWPORT_MAX_TMP_HARD", _MAX_TMP_CAP)
        self.num_cells = 0
        self.final_grid_level = 0
        self.final_input_idx = 0
        self.final_output_idx = 1
        self.max_active_count = 0
        self.max_tmp_count = 0
        self.culling_overflow = False
        self._buffer_config_key = None
        self._parents_init_raw = b""
        self._active_init_raw = b""
        self.fragment_params_data = _FragmentParamsUBO()
        self.params_ubo = None
        self.scene_texture_bytes = 0
        self.prims_tex = None
        self.nodes_tex = None
        self.binary_ops_tex = None
        self.parents_init_tex = None
        self.active_nodes_init_tex = None
        self.parents_tex = [None, None]
        self.active_nodes_tex = [None, None]
        self.cell_offsets_tex = [None, None]
        self.num_active_tex = [None, None]
        self.cell_errors_tex = [None, None]
        self.counters_tex = None
        self.old_to_new_scratch_tex = None
        self.tmp_tex = None
        self.scene_static_key = None
        self.shader_source_token = None

        self.prims_ssbo = _SSBO()
        self.nodes_ssbo = _SSBO()
        self.binary_ops_ssbo = _SSBO()
        self.parents_ssbo = [_SSBO(), _SSBO()]
        self.active_nodes_ssbo = [_SSBO(), _SSBO()]
        self.cell_offsets_ssbo = [_SSBO(), _SSBO()]
        self.num_active_ssbo = [_SSBO(), _SSBO()]
        self.cell_errors_ssbo = [_SSBO(), _SSBO()]
        self.counters_ssbo = _SSBO()
        self.old_to_new_scratch_ssbo = _SSBO()
        self.tmp_ssbo = _SSBO()

    def free(self):
        self.draw_shader = None
        self.draw_shader_mode = None
        self.draw_shader_backend = None
        self.compute_shader = None
        self.texture_scene_mode = False
        self.batch = None
        if self.offscreen is not None:
            try:
                self.offscreen.free()
            except Exception:
                pass
        self.offscreen = None
        self.offscreen_size = (0, 0)
        self.scene_key = None
        self.scene_static_key = None
        self.pruning_key = None
        self.scene_info = None
        self.runtime_failed = False
        self.shader_source_token = None
        self.int16_storage_supported = (
            True if _int_env("MATHOPS_V2_TRY_GL_INT16", 1) else False
        )
        self._buffer_config_key = None
        self._parents_init_raw = b""
        self._active_init_raw = b""
        self.params_ubo = None
        self.scene_texture_bytes = 0
        for texture_name in (
            "prims_tex",
            "nodes_tex",
            "binary_ops_tex",
            "parents_init_tex",
            "active_nodes_init_tex",
            "counters_tex",
            "old_to_new_scratch_tex",
            "tmp_tex",
        ):
            texture = getattr(self, texture_name, None)
            if texture is not None:
                try:
                    texture.free()
                except Exception:
                    pass
            setattr(self, texture_name, None)
        for texture_list_name in (
            "parents_tex",
            "active_nodes_tex",
            "cell_offsets_tex",
            "num_active_tex",
            "cell_errors_tex",
        ):
            texture_list = getattr(self, texture_list_name, None)
            if texture_list is not None:
                for texture in texture_list:
                    if texture is not None:
                        try:
                            texture.free()
                        except Exception:
                            pass
            setattr(self, texture_list_name, [None, None])
        for buffer in (
            self.prims_ssbo,
            self.nodes_ssbo,
            self.binary_ops_ssbo,
            self.parents_ssbo[0],
            self.parents_ssbo[1],
            self.active_nodes_ssbo[0],
            self.active_nodes_ssbo[1],
            self.cell_offsets_ssbo[0],
            self.cell_offsets_ssbo[1],
            self.num_active_ssbo[0],
            self.num_active_ssbo[1],
            self.cell_errors_ssbo[0],
            self.cell_errors_ssbo[1],
            self.counters_ssbo,
            self.old_to_new_scratch_ssbo,
            self.tmp_ssbo,
        ):
            buffer.free()

    def _backend_type(self) -> str:
        try:
            return gpu.platform.backend_type_get()
        except Exception:
            return "UNKNOWN"

    def supported(self) -> bool:
        return self._backend_type() != "UNKNOWN"

    def _unsupported_backend_message(self) -> str:
        return f"Viewport GPU path is unavailable on {self._backend_type()}"

    def _ensure_shaders(self, shading_mode: str) -> bool:
        if self.runtime_failed:
            return False
        backend = self._backend_type()
        source_token = _shader_source_token()
        if (
            self.draw_shader is not None
            and self.draw_shader_mode == shading_mode
            and self.draw_shader_backend == backend
            and self.compute_shader is not None
            and self.shader_source_token == source_token
        ):
            return True
        if not self.supported():
            if not self.logged_backend_warning:
                runtime.debug_log(self._unsupported_backend_message())
                self.logged_backend_warning = True
            runtime.last_error_message = self._unsupported_backend_message()
            return False

        shading_mode_value = {
            "SHADED": 0,
            "HEATMAP": 1,
            "NORMALS": 2,
            "AO": 3,
        }[shading_mode]

        self.draw_shader = None
        self.compute_shader = None
        self.batch = None
        self.texture_scene_mode = backend != "OPENGL"

        if self.texture_scene_mode:
            try:
                compute_shader_info = gpu.types.GPUShaderCreateInfo()
                compute_shader_info.local_group_size(4, 4, 4)
                _declare_push_constants(compute_shader_info, _COMPUTE_PUSH_CONSTANTS)
                compute_shader_info.sampler(0, "FLOAT_2D", "mopsPrimsTex")
                compute_shader_info.sampler(1, "FLOAT_2D", "mopsNodesTex")
                compute_shader_info.sampler(2, "FLOAT_2D", "mopsBinaryOpsTex")
                compute_shader_info.sampler(3, "FLOAT_2D", "mopsParentsInTex")
                compute_shader_info.sampler(4, "FLOAT_2D", "mopsActiveNodesInTex")
                compute_shader_info.sampler(5, "FLOAT_2D", "mopsParentCellOffsetsTex")
                compute_shader_info.sampler(6, "FLOAT_2D", "mopsParentCellCountsTex")
                compute_shader_info.sampler(7, "FLOAT_2D", "mopsCellValueInTex")
                compute_shader_info.image(
                    0, "R32F", "FLOAT_2D", "mopsParentsOutImg", qualifiers={"WRITE"}
                )
                compute_shader_info.image(
                    1, "R32F", "FLOAT_2D", "mopsActiveNodesOutImg", qualifiers={"WRITE"}
                )
                compute_shader_info.image(
                    2,
                    "R32F",
                    "FLOAT_2D",
                    "mopsChildCellOffsetsImg",
                    qualifiers={"WRITE"},
                )
                compute_shader_info.image(
                    3, "R32F", "FLOAT_2D", "mopsCellCountsImg", qualifiers={"WRITE"}
                )
                compute_shader_info.image(
                    4,
                    "R32UI",
                    "UINT_2D_ATOMIC",
                    "mopsCountersImg",
                    qualifiers={"READ", "WRITE"},
                )
                compute_shader_info.image(
                    5, "R32F", "FLOAT_2D", "mopsCellValueOutImg", qualifiers={"WRITE"}
                )
                compute_shader_info.image(
                    6,
                    "R32UI",
                    "UINT_2D",
                    "mopsOldToNewImg",
                    qualifiers={"READ", "WRITE"},
                )
                compute_shader_info.image(
                    7,
                    "R32UI",
                    "UINT_2D",
                    "mopsTmpImg",
                    qualifiers={"READ", "WRITE"},
                )
                compute_shader_info.compute_source(
                    _shader_source(
                        "culling.comp.glsl",
                        "compute",
                        use_int16_storage=False,
                        texture_backend=True,
                    )
                )
                self.compute_shader = gpu.shader.create_from_info(compute_shader_info)

                shader_info = gpu.types.GPUShaderCreateInfo()
                shader_info.vertex_in(0, "VEC2", "pos")
                shader_info.fragment_out(0, "VEC4", "outColor")
                shader_info.depth_write("ANY")
                shader_info.typedef_source(_FRAGMENT_PARAMS_TYPEDEF)
                shader_info.uniform_buf(0, "ViewportParams", "params")
                shader_info.sampler(0, "FLOAT_2D", "mopsPrimsTex")
                shader_info.sampler(1, "FLOAT_2D", "mopsNodesTex")
                shader_info.sampler(2, "FLOAT_2D", "mopsBinaryOpsTex")
                shader_info.sampler(3, "FLOAT_2D", "mopsActiveNodesTex")
                shader_info.sampler(4, "FLOAT_2D", "mopsCellOffsetsTex")
                shader_info.sampler(5, "FLOAT_2D", "mopsCellCountsTex")
                shader_info.sampler(6, "FLOAT_2D", "mopsCellErrorsTex")
                shader_info.sampler(7, "FLOAT_2D", "mopsMatcapTex")
                shader_info.sampler(8, "FLOAT_2D", "mopsMatcapSpecularTex")
                shader_info.vertex_source(
                    "void main(){  gl_Position = vec4(pos, 0.0, 1.0);}"
                )
                shader_info.fragment_source(
                    _shader_source(
                        "simple.frag.glsl",
                        "fragment",
                        shading_mode_value=shading_mode_value,
                        use_int16_storage=False,
                        texture_backend=True,
                    )
                )
                self.draw_shader = gpu.shader.create_from_info(shader_info)
                self.draw_shader_mode = shading_mode
                self.draw_shader_backend = backend
                self.shader_source_token = source_token
                self.use_int16_storage = False
                verts = ((-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0))
                self.batch = batch_for_shader(self.draw_shader, "TRIS", {"pos": verts})
                runtime.debug_log(
                    f"Viewport shader backend: {backend} texture-packed scene"
                )
                return True
            except Exception as exc:
                if not self.logged_shader_failure:
                    runtime.debug_log(
                        f"Failed to compile viewport texture shader: {exc}"
                    )
                    self.logged_shader_failure = True
                runtime.last_error_message = (
                    f"Viewport GPU shader compile failed: {exc}"
                )
                self.runtime_failed = True
                return False

        last_exc = None
        attempts = (False,) if self.int16_storage_supported is False else (True, False)
        for use_int16_storage in attempts:
            try:
                compute_shader_info = gpu.types.GPUShaderCreateInfo()
                compute_shader_info.local_group_size(4, 4, 4)
                _declare_push_constants(compute_shader_info, _COMPUTE_PUSH_CONSTANTS)
                compute_shader_info.compute_source(
                    _shader_source(
                        "culling.comp.glsl",
                        "compute",
                        use_int16_storage=use_int16_storage,
                    )
                )
                compute_shader = gpu.shader.create_from_info(compute_shader_info)

                shader_info = gpu.types.GPUShaderCreateInfo()
                shader_info.vertex_in(0, "VEC2", "pos")
                shader_info.fragment_out(0, "VEC4", "outColor")
                shader_info.depth_write("ANY")
                shader_info.typedef_source(_FRAGMENT_PARAMS_TYPEDEF)
                shader_info.uniform_buf(0, "ViewportParams", "params")
                shader_info.sampler(0, "FLOAT_2D", "mopsMatcapTex")
                shader_info.sampler(1, "FLOAT_2D", "mopsMatcapSpecularTex")
                shader_info.vertex_source(
                    "void main(){  gl_Position = vec4(pos, 0.0, 1.0);}"
                )
                shader_info.fragment_source(
                    _shader_source(
                        "simple.frag.glsl",
                        "fragment",
                        shading_mode_value=shading_mode_value,
                        use_int16_storage=use_int16_storage,
                    )
                )
                draw_shader = gpu.shader.create_from_info(shader_info)

                self.compute_shader = compute_shader
                self.draw_shader = draw_shader
                self.draw_shader_mode = shading_mode
                self.draw_shader_backend = backend
                self.shader_source_token = source_token
                if self.use_int16_storage != use_int16_storage:
                    self.pruning_key = None
                    self._buffer_config_key = None
                self.use_int16_storage = use_int16_storage
                if use_int16_storage:
                    self.int16_storage_supported = True

                if self.batch is None:
                    verts = ((-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0))
                    self.batch = batch_for_shader(
                        self.draw_shader, "TRIS", {"pos": verts}
                    )

                runtime.debug_log(
                    f"Viewport shader storage: {'int16' if use_int16_storage else 'packed32'}"
                )
                return True
            except Exception as exc:
                if use_int16_storage:
                    self.int16_storage_supported = False
                last_exc = exc

        if not self.logged_shader_failure:
            runtime.debug_log(
                f"Failed to compile exact viewport GPU shaders: {last_exc}"
            )
            self.logged_shader_failure = True
        runtime.last_error_message = f"Viewport GPU shader compile failed: {last_exc}"
        self.draw_shader = None
        self.draw_shader_mode = None
        self.compute_shader = None
        self.batch = None
        self.shader_source_token = None
        self.runtime_failed = True
        return False

    def _renderer_to_blender_matrix(self):
        return Matrix(
            (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, -1.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        )

    def _free_scene_textures(self):
        for texture_name in ("prims_tex", "nodes_tex", "binary_ops_tex"):
            texture = getattr(self, texture_name, None)
            if texture is not None:
                try:
                    texture.free()
                except Exception:
                    pass
                setattr(self, texture_name, None)

    def _texture_from_raw_bytes(self, data: bytes):
        words = np.frombuffer(data, dtype=np.uint32).view(np.float32)
        texel_count = max(1, (len(words) + 3) // 4)
        height = max(1, (texel_count + _TEXTURE_DATA_WIDTH - 1) // _TEXTURE_DATA_WIDTH)
        needed_words = _TEXTURE_DATA_WIDTH * height * 4
        payload = np.zeros(needed_words, dtype=np.float32)
        if len(words):
            payload[: len(words)] = words
        buffer = gpu.types.Buffer("FLOAT", len(payload), payload)
        texture = gpu.types.GPUTexture(
            (_TEXTURE_DATA_WIDTH, height), format="RGBA32F", data=buffer
        )
        return texture, int(texture.width) * int(texture.height) * 16

    def _texture_from_float_rgba(self, payload: np.ndarray):
        data = np.asarray(payload, dtype=np.float32)
        texel_count = max(1, data.shape[0])
        height = max(1, (texel_count + _TEXTURE_DATA_WIDTH - 1) // _TEXTURE_DATA_WIDTH)
        padded = np.zeros((_TEXTURE_DATA_WIDTH * height, 4), dtype=np.float32)
        padded[: data.shape[0], :] = data
        flat = np.ascontiguousarray(padded.reshape(-1), dtype=np.float32)
        buffer = gpu.types.Buffer("FLOAT", len(flat), flat)
        texture = gpu.types.GPUTexture(
            (_TEXTURE_DATA_WIDTH, height), format="RGBA32F", data=buffer
        )
        return texture, int(texture.width) * int(texture.height) * 16

    def _pack_nodes_texture(self, node_bytes: bytes):
        values = np.frombuffer(node_bytes, dtype=np.int32).reshape((-1, 2))
        payload = np.zeros((values.shape[0], 4), dtype=np.float32)
        payload[:, 0] = values[:, 0].astype(np.float32)
        payload[:, 1] = values[:, 1].astype(np.float32)
        return self._texture_from_float_rgba(payload)

    def _pack_binary_ops_texture(self, op_bytes: bytes):
        bits = np.frombuffer(op_bytes, dtype=np.uint32)
        k = (bits & np.uint32(0xFFFFFFF8)).view(np.float32)
        payload = np.zeros((bits.shape[0], 4), dtype=np.float32)
        payload[:, 0] = k
        payload[:, 1] = (bits & np.uint32(1)).astype(np.float32)
        payload[:, 2] = ((bits >> np.uint32(1)) & np.uint32(3)).astype(np.float32)
        return self._texture_from_float_rgba(payload)

    def _pack_primitives_texture(self, prim_bytes: bytes):
        words_u32 = np.frombuffer(prim_bytes, dtype=np.uint32).reshape((-1, 24))
        words_f32 = words_u32.view(np.float32)
        payload = np.zeros(
            (words_u32.shape[0] * _PRIMITIVE_TEXELS, 4), dtype=np.float32
        )
        base = np.arange(words_u32.shape[0]) * _PRIMITIVE_TEXELS
        payload[base + 0, 0:3] = words_f32[:, 0:3]
        payload[base + 0, 3] = words_u32[:, 18].astype(np.float32)
        payload[base + 1, :] = words_f32[:, 4:8]
        payload[base + 2, :] = words_f32[:, 8:12]
        payload[base + 3, :] = words_f32[:, 12:16]
        payload[base + 4, 0:2] = words_f32[:, 16:18]
        payload[base + 4, 2] = words_f32[:, 19]
        payload[base + 4, 3] = (words_u32[:, 20] & np.uint32(0xFF)).astype(np.float32)
        payload[base + 5, 0] = (
            (words_u32[:, 20] >> np.uint32(8)) & np.uint32(0xFF)
        ).astype(np.float32)
        payload[base + 5, 1] = (
            (words_u32[:, 20] >> np.uint32(16)) & np.uint32(0xFF)
        ).astype(np.float32)
        payload[base + 5, 2] = (words_u32[:, 3] & np.uint32(0xFF)).astype(np.float32)
        payload[base + 5, 3] = (
            (words_u32[:, 3] >> np.uint32(8)) & np.uint32(0xFF)
        ).astype(np.float32)
        payload[base + 6, 0] = (
            (words_u32[:, 3] >> np.uint32(16)) & np.uint32(0xFF)
        ).astype(np.float32)
        payload[base + 6, 1] = (
            (words_u32[:, 3] >> np.uint32(24)) & np.uint32(0xFF)
        ).astype(np.float32)
        payload[base + 6, 2] = words_f32[:, 21]
        return self._texture_from_float_rgba(payload)

    def _sync_scene_textures(self, packed):
        self._free_scene_textures()
        self.prims_tex, prim_bytes = self._pack_primitives_texture(
            bytes(packed["primitives"])
        )
        self.nodes_tex, node_bytes = self._pack_nodes_texture(bytes(packed["nodes"]))
        self.binary_ops_tex, op_bytes = self._pack_binary_ops_texture(
            bytes(packed["binary_ops"])
        )
        self.scene_texture_bytes = prim_bytes + node_bytes + op_bytes

    def _sync_primitive_texture(self, primitive_bytes: bytes):
        if self.prims_tex is not None:
            try:
                self.prims_tex.free()
            except Exception:
                pass
        self.prims_tex, prim_bytes = self._pack_primitives_texture(primitive_bytes)
        static_bytes = 0
        for texture in (self.nodes_tex, self.binary_ops_tex):
            if texture is not None:
                static_bytes += int(texture.width) * int(texture.height) * 16
        self.scene_texture_bytes = prim_bytes + static_bytes

    def _sync_texture_init_textures(self):
        for texture_name in ("parents_init_tex", "active_nodes_init_tex"):
            texture = getattr(self, texture_name, None)
            if texture is not None:
                try:
                    texture.free()
                except Exception:
                    pass
                setattr(self, texture_name, None)

        parent_count = len(self._parents_init_raw) // np.dtype(np.uint16).itemsize
        active_count = len(self._active_init_raw) // np.dtype(np.uint16).itemsize
        self.parents_init_tex = self._create_r32f_texture_from_uint16(
            self._parents_init_raw, parent_count
        )
        self.active_nodes_init_tex = self._create_r32f_texture_from_uint16(
            self._active_init_raw, active_count
        )

    def _array_texture_capacity_limit(self) -> int:
        try:
            max_size = int(gpu.capabilities.max_texture_size_get())
        except Exception:
            max_size = _TEXTURE_DATA_WIDTH
        return _TEXTURE_DATA_WIDTH * max(1, max_size)

    def _texture_precision_active_limit(self) -> int:
        return min(self._array_texture_capacity_limit(), _FLOAT32_EXACT_UINT_LIMIT)

    def _array_texture_size(self, count: int) -> tuple[int, int]:
        height = max(
            1, (max(int(count), 1) + _TEXTURE_DATA_WIDTH - 1) // _TEXTURE_DATA_WIDTH
        )
        return _TEXTURE_DATA_WIDTH, height

    def _create_scalar_texture(self, count: int, fmt: str):
        size = self._array_texture_size(count)
        return gpu.types.GPUTexture(size, format=fmt)

    def _create_r32f_texture_from_uint16(self, data: bytes, count: int):
        values = np.zeros(max(int(count), 1), dtype=np.uint32)
        if data:
            src = np.frombuffer(data, dtype=np.uint16).astype(np.uint32)
            values[: len(src)] = src
        payload = values.astype(np.float32)
        width, height = self._array_texture_size(count)
        needed = width * height
        if payload.size < needed:
            payload = np.pad(payload, (0, needed - payload.size), constant_values=0.0)
        buffer = gpu.types.Buffer("FLOAT", len(payload), payload)
        return gpu.types.GPUTexture((width, height), format="R32F", data=buffer)

    def _clear_texture(self, texture, data_format: str, value):
        if texture is not None:
            texture.clear(format=data_format, value=value)

    def _ensure_texture_work_buffers(
        self,
        node_count: int,
        grid_level: int,
        active_capacity: int | None = None,
        tmp_capacity: int | None = None,
    ):
        default_active, default_tmp = self._capacity_for_scene(node_count)
        texture_cap = self._array_texture_capacity_limit()
        active_precision_cap = self._texture_precision_active_limit()
        if active_capacity is None:
            active_capacity = max(default_active, self.active_capacity)
        if tmp_capacity is None:
            tmp_capacity = max(default_tmp, self.tmp_capacity)
        active_capacity = min(active_capacity, active_precision_cap)
        tmp_capacity = min(tmp_capacity, texture_cap)
        grid_size = 1 << grid_level
        num_cells = grid_size * grid_size * grid_size
        key = (node_count, grid_level, active_capacity, tmp_capacity, "texture")
        if key == self._buffer_config_key:
            return

        self.active_capacity = active_capacity
        self.tmp_capacity = tmp_capacity
        self.num_cells = num_cells
        self.final_grid_level = grid_level

        for idx in range(2):
            texture = self.parents_tex[idx]
            if texture is not None:
                try:
                    texture.free()
                except Exception:
                    pass
            self.parents_tex[idx] = self._create_scalar_texture(active_capacity, "R32F")

            texture = self.active_nodes_tex[idx]
            if texture is not None:
                try:
                    texture.free()
                except Exception:
                    pass
            self.active_nodes_tex[idx] = self._create_scalar_texture(
                active_capacity, "R32F"
            )

            texture = self.cell_offsets_tex[idx]
            if texture is not None:
                try:
                    texture.free()
                except Exception:
                    pass
            self.cell_offsets_tex[idx] = self._create_scalar_texture(num_cells, "R32F")

            texture = self.num_active_tex[idx]
            if texture is not None:
                try:
                    texture.free()
                except Exception:
                    pass
            self.num_active_tex[idx] = self._create_scalar_texture(num_cells, "R32F")

            texture = self.cell_errors_tex[idx]
            if texture is not None:
                try:
                    texture.free()
                except Exception:
                    pass
            self.cell_errors_tex[idx] = self._create_scalar_texture(num_cells, "R32F")

        for texture_name, count, fmt in (
            ("counters_tex", _COUNTERS_SIZE, "R32UI"),
            ("old_to_new_scratch_tex", tmp_capacity, "R32UI"),
            ("tmp_tex", tmp_capacity, "R32UI"),
        ):
            texture = getattr(self, texture_name, None)
            if texture is not None:
                try:
                    texture.free()
                except Exception:
                    pass
            setattr(self, texture_name, self._create_scalar_texture(count, fmt))

        self._buffer_config_key = key
        self.pruning_key = None
        self._sync_texture_init_textures()
        if (
            active_precision_cap < texture_cap
            and active_capacity == active_precision_cap
        ):
            runtime.debug_log(
                f"Viewport texture active capacity capped at {active_precision_cap:,} to keep float32 cell offsets exact"
            )
        runtime.debug_log(
            f"Viewport texture buffers: active={active_capacity:,}, tmp={tmp_capacity:,}, cells={num_cells:,}"
        )

    def _storage_bytes_for_u16(self, data: bytes) -> bytes:
        if not data:
            return b""
        if self.use_int16_storage:
            return data
        return np.frombuffer(data, dtype=np.uint16).astype(np.uint32).tobytes()

    def _storage_bytes_for_active_nodes(self, data: bytes) -> bytes:
        if not data:
            return b""
        if self.use_int16_storage:
            return data
        values = np.frombuffer(data, dtype=np.uint16).astype(np.uint32)
        if values.size & 1:
            values = np.pad(values, (0, 1), constant_values=0)
        packed = values[0::2] | (values[1::2] << 16)
        return packed.tobytes()

    def _active_entry_stride(self) -> int:
        return 2 if self.use_int16_storage else 4

    def _active_buffer_bytes(self, capacity: int) -> int:
        if self.use_int16_storage:
            return capacity * 2
        return ((capacity + 1) // 2) * 4

    def _capacity_for_scene(self, node_count: int) -> tuple[int, int]:
        active_floor = min(8_000_000, self.max_active_limit)
        tmp_floor = min(32_000_000, self.max_tmp_limit)
        active_default = min(
            max(node_count * 4096, active_floor), self.max_active_limit
        )
        tmp_default = min(
            max(max(active_default * 8, node_count * 1024), tmp_floor),
            self.max_tmp_limit,
        )
        active_capacity = min(
            _int_env("MATHOPS_V2_VIEWPORT_MAX_ACTIVE", active_default),
            self.max_active_limit,
        )
        tmp_capacity = min(
            _int_env("MATHOPS_V2_VIEWPORT_MAX_TMP", tmp_default),
            self.max_tmp_limit,
        )
        if self.texture_scene_mode:
            texture_cap = self._array_texture_capacity_limit()
            active_capacity = min(
                active_capacity, self._texture_precision_active_limit()
            )
            tmp_capacity = min(tmp_capacity, texture_cap)
        return active_capacity, tmp_capacity

    def _ensure_work_buffers(
        self,
        node_count: int,
        grid_level: int,
        active_capacity: int | None = None,
        tmp_capacity: int | None = None,
    ):
        if self.texture_scene_mode:
            self._ensure_texture_work_buffers(
                node_count, grid_level, active_capacity, tmp_capacity
            )
            return
        default_active, default_tmp = self._capacity_for_scene(node_count)
        if active_capacity is None:
            active_capacity = max(default_active, self.active_capacity)
        if tmp_capacity is None:
            tmp_capacity = max(default_tmp, self.tmp_capacity)
        grid_size = 1 << grid_level
        num_cells = grid_size * grid_size * grid_size
        key = (
            node_count,
            grid_level,
            active_capacity,
            tmp_capacity,
            self.use_int16_storage,
        )
        if key == self._buffer_config_key:
            return

        self.active_capacity = active_capacity
        self.tmp_capacity = tmp_capacity
        self.num_cells = num_cells
        self.final_grid_level = grid_level

        for buffer in self.parents_ssbo:
            buffer.ensure_size(active_capacity * self._active_entry_stride())
        for buffer in self.active_nodes_ssbo:
            buffer.ensure_size(self._active_buffer_bytes(active_capacity))
        for buffer in self.cell_offsets_ssbo:
            buffer.ensure_size(num_cells * 4)
        for buffer in self.num_active_ssbo:
            buffer.ensure_size(num_cells * 4)
        for buffer in self.cell_errors_ssbo:
            buffer.ensure_size(num_cells * 4)

        self.counters_ssbo.ensure_size(_COUNTERS_SIZE * 4)
        self.old_to_new_scratch_ssbo.ensure_size(tmp_capacity * 4)
        self.tmp_ssbo.ensure_size(tmp_capacity * 4)

        self._buffer_config_key = key
        self.pruning_key = None
        runtime.debug_log(
            f"Viewport buffers: active={active_capacity:,}, tmp={tmp_capacity:,}, cells={num_cells:,}"
        )

    def _grow_work_buffers(self, node_count: int, grid_level: int) -> bool:
        if self.texture_scene_mode:
            texture_cap = self._array_texture_capacity_limit()
            next_active = min(
                max(self.active_capacity * 2, 1), self._texture_precision_active_limit()
            )
            next_tmp = min(max(self.tmp_capacity * 2, 1), texture_cap)
            if next_active == self.active_capacity and next_tmp == self.tmp_capacity:
                return False
            runtime.debug_log(
                f"Viewport texture pruning overflow; retrying with active={next_active:,}, tmp={next_tmp:,}"
            )
            self._ensure_texture_work_buffers(
                node_count, grid_level, next_active, next_tmp
            )
            return True
        next_active = min(max(self.active_capacity * 2, 1), self.max_active_limit)
        next_tmp = min(max(self.tmp_capacity * 2, 1), self.max_tmp_limit)
        if next_active == self.active_capacity and next_tmp == self.tmp_capacity:
            return False
        runtime.debug_log(
            f"Viewport pruning overflow; retrying with active={next_active:,}, tmp={next_tmp:,}"
        )
        self._ensure_work_buffers(node_count, grid_level, next_active, next_tmp)
        return True

    def _sync_scene(self, scene_path: Path, settings):
        anim_frame_key = bridge.demo_anim_frame_key(settings)
        anim_active = anim_frame_key is not None
        scene_token = bridge.scene_content_token(scene_path)
        scene_static_key = (
            str(scene_path.resolve()),
            scene_token,
            anim_active,
        )
        scene_key = (
            scene_static_key[0],
            scene_static_key[1],
            anim_frame_key,
        )
        if self.scene_key != scene_key:
            native_module = bridge.load_native_module()
            anim_time = bridge.demo_anim_time(settings)
            if anim_time is None:
                packed = native_module.pack_scene_file(str(scene_path))
            else:
                packed = native_module.pack_scene_demo_anim(
                    str(scene_path), float(anim_time)
                )
            full_refresh = self.scene_static_key != scene_static_key
            if full_refresh:
                self._parents_init_raw = bytes(packed["parents"])
                self._active_init_raw = bytes(packed["active_nodes"])
                if self.texture_scene_mode:
                    self._sync_scene_textures(packed)
                    self._sync_texture_init_textures()
                else:
                    self.prims_ssbo.upload(bytes(packed["primitives"]))
                    self.nodes_ssbo.upload(bytes(packed["nodes"]))
                    self.binary_ops_ssbo.upload(bytes(packed["binary_ops"]))
                self.scene_info = {
                    "aabb_min": tuple(packed["aabb_min"]),
                    "aabb_max": tuple(packed["aabb_max"]),
                    "node_count": int(packed["node_count"]),
                }
                self.scene_static_key = scene_static_key
            elif anim_active and self.texture_scene_mode:
                self._sync_primitive_texture(bytes(packed["primitives"]))
            elif anim_active:
                self.prims_ssbo.upload(bytes(packed["primitives"]))
            self.scene_key = scene_key
            self.pruning_key = None
            self.culling_overflow = False
            self.last_overflow_message = ""
            if full_refresh:
                runtime.debug_log(
                    f"Viewport scene packed: {scene_path.name}, nodes={self.scene_info['node_count']}"
                )

        self._ensure_work_buffers(
            self.scene_info["node_count"], bridge.grid_level(settings)
        )

    def _reset_pruning_state(self):
        if self.texture_scene_mode:
            self._clear_texture(self.parents_tex[1], "FLOAT", (0.0,))
            self._clear_texture(self.active_nodes_tex[1], "FLOAT", (0.0,))
            self._clear_texture(self.cell_offsets_tex[0], "FLOAT", (0.0,))
            self._clear_texture(self.cell_offsets_tex[1], "FLOAT", (0.0,))
            self._clear_texture(self.num_active_tex[0], "FLOAT", (0.0,))
            self._clear_texture(self.num_active_tex[1], "FLOAT", (0.0,))
            self._clear_texture(self.cell_errors_tex[0], "FLOAT", (0.0,))
            self._clear_texture(self.cell_errors_tex[1], "FLOAT", (0.0,))
            self._clear_texture(self.counters_tex, "UINT", (0,))
            self._clear_texture(self.old_to_new_scratch_tex, "UINT", (0,))
            self._clear_texture(self.tmp_tex, "UINT", (0,))
            self.max_active_count = 0
            self.max_tmp_count = 0
            self.culling_overflow = False
            self.last_overflow_message = ""
            return
        self.parents_ssbo[0].update(self._storage_bytes_for_u16(self._parents_init_raw))
        self.active_nodes_ssbo[0].update(
            self._storage_bytes_for_active_nodes(self._active_init_raw)
        )
        self.counters_ssbo.update(np.zeros(_COUNTERS_SIZE, dtype=np.int32).tobytes())
        zero_f32 = np.zeros(1, dtype=np.float32).tobytes()
        self.cell_errors_ssbo[0].update(zero_f32)
        self.cell_errors_ssbo[1].update(zero_f32)
        self.max_active_count = 0
        self.max_tmp_count = 0
        self.culling_overflow = False
        self.last_overflow_message = ""

    def _bind_culling_buffers(self, input_idx: int, output_idx: int):
        if self.texture_scene_mode:
            parents_in_tex = (
                self.parents_init_tex
                if getattr(self, "_texture_first_level", False)
                else self.parents_tex[input_idx]
            )
            active_nodes_in_tex = (
                self.active_nodes_init_tex
                if getattr(self, "_texture_first_level", False)
                else self.active_nodes_tex[input_idx]
            )
            self.compute_shader.bind()
            self.compute_shader.uniform_sampler("mopsPrimsTex", self.prims_tex)
            self.compute_shader.uniform_sampler("mopsNodesTex", self.nodes_tex)
            self.compute_shader.uniform_sampler("mopsBinaryOpsTex", self.binary_ops_tex)
            self.compute_shader.uniform_sampler("mopsParentsInTex", parents_in_tex)
            self.compute_shader.uniform_sampler(
                "mopsActiveNodesInTex", active_nodes_in_tex
            )
            self.compute_shader.uniform_sampler(
                "mopsParentCellOffsetsTex", self.cell_offsets_tex[input_idx]
            )
            self.compute_shader.uniform_sampler(
                "mopsParentCellCountsTex", self.num_active_tex[input_idx]
            )
            self.compute_shader.uniform_sampler(
                "mopsCellValueInTex", self.cell_errors_tex[input_idx]
            )
            self.compute_shader.image("mopsParentsOutImg", self.parents_tex[output_idx])
            self.compute_shader.image(
                "mopsActiveNodesOutImg", self.active_nodes_tex[output_idx]
            )
            self.compute_shader.image(
                "mopsChildCellOffsetsImg", self.cell_offsets_tex[output_idx]
            )
            self.compute_shader.image(
                "mopsCellCountsImg", self.num_active_tex[output_idx]
            )
            self.compute_shader.image("mopsCountersImg", self.counters_tex)
            self.compute_shader.image(
                "mopsCellValueOutImg", self.cell_errors_tex[output_idx]
            )
            self.compute_shader.image("mopsOldToNewImg", self.old_to_new_scratch_tex)
            self.compute_shader.image("mopsTmpImg", self.tmp_tex)
            return
        self.prims_ssbo.bind(_BINDING_PRIMS)
        self.nodes_ssbo.bind(_BINDING_NODES)
        self.binary_ops_ssbo.bind(_BINDING_BINARY_OPS)
        self.parents_ssbo[input_idx].bind(_BINDING_PARENTS_IN)
        self.parents_ssbo[output_idx].bind(_BINDING_PARENTS_OUT)
        self.active_nodes_ssbo[input_idx].bind(_BINDING_ACTIVE_NODES_IN)
        self.active_nodes_ssbo[output_idx].bind(_BINDING_ACTIVE_NODES_OUT)
        self.cell_offsets_ssbo[input_idx].bind(_BINDING_PARENT_CELL_OFFSETS)
        self.cell_offsets_ssbo[output_idx].bind(_BINDING_CELL_OFFSETS)
        self.num_active_ssbo[input_idx].bind(_BINDING_PARENT_CELL_COUNTS)
        self.num_active_ssbo[output_idx].bind(_BINDING_CELL_COUNTS)
        self.counters_ssbo.bind(_BINDING_COUNTERS)
        self.cell_errors_ssbo[input_idx].bind(_BINDING_CELL_ERROR_IN)
        self.cell_errors_ssbo[output_idx].bind(_BINDING_CELL_ERROR_OUT)
        self.old_to_new_scratch_ssbo.bind(_BINDING_OLD_TO_NEW_SCRATCH)
        self.tmp_ssbo.bind(_BINDING_TMP)

    def _bind_draw_buffers(self):
        if self.texture_scene_mode:
            self.draw_shader.bind()
            self.draw_shader.uniform_sampler("mopsPrimsTex", self.prims_tex)
            self.draw_shader.uniform_sampler("mopsNodesTex", self.nodes_tex)
            self.draw_shader.uniform_sampler("mopsBinaryOpsTex", self.binary_ops_tex)
            self.draw_shader.uniform_sampler(
                "mopsActiveNodesTex", self.active_nodes_tex[self.final_output_idx]
            )
            self.draw_shader.uniform_sampler(
                "mopsCellOffsetsTex", self.cell_offsets_tex[self.final_output_idx]
            )
            self.draw_shader.uniform_sampler(
                "mopsCellCountsTex", self.num_active_tex[self.final_output_idx]
            )
            self.draw_shader.uniform_sampler(
                "mopsCellErrorsTex", self.cell_errors_tex[self.final_output_idx]
            )
            return
        self.prims_ssbo.bind(_BINDING_PRIMS)
        self.nodes_ssbo.bind(_BINDING_NODES)
        self.binary_ops_ssbo.bind(_BINDING_BINARY_OPS)
        self.active_nodes_ssbo[self.final_output_idx].bind(_BINDING_ACTIVE_NODES_OUT)
        self.cell_offsets_ssbo[self.final_output_idx].bind(_BINDING_CELL_OFFSETS)
        self.num_active_ssbo[self.final_output_idx].bind(_BINDING_CELL_COUNTS)
        self.cell_errors_ssbo[self.final_output_idx].bind(_BINDING_CELL_ERROR_OUT)

    def _memory_barrier(self):
        if self.texture_scene_mode:
            if _TEXTURE_COUNTER_READBACK and self.counters_tex is not None:
                self.counters_tex.read()
            return
        _gl_functions()["glMemoryBarrier"](
            _GL_SHADER_STORAGE_BARRIER_BIT | _GL_BUFFER_UPDATE_BARRIER_BIT
        )

    def _ensure_offscreen(self, width: int, height: int):
        size = (int(width), int(height))
        if self.offscreen is not None and self.offscreen_size == size:
            return
        if self.offscreen is not None:
            try:
                self.offscreen.free()
            except Exception:
                pass
        self.offscreen = gpu.types.GPUOffScreen(*size)
        self.offscreen_size = size

    def _update_pruning(self, scene_path: Path, settings, aabb_min, aabb_max):
        if not settings.culling_enabled:
            self.frame_culling_ms = 0.0
            self.max_active_count = 0
            self.max_tmp_count = 0
            self.culling_overflow = False
            self.last_overflow_message = ""
            return

        pruning_key = (
            self.scene_key,
            bridge.grid_level(settings),
            tuple(round(float(v), 6) for v in aabb_min),
            tuple(round(float(v), 6) for v in aabb_max),
        )
        if self.pruning_key == pruning_key:
            self.frame_culling_ms = 0.0
            return

        runtime.debug_log(f"Viewport pruning recompute: {scene_path.name}")
        start = time.perf_counter()
        grid_level = bridge.grid_level(settings)
        while True:
            self._reset_pruning_state()

            input_idx = 0
            output_idx = 1
            first_lvl = True

            for current_level in range(2, grid_level + 1, 2):
                grid_size = 1 << current_level
                num_groups = (grid_size + 3) // 4

                self.compute_shader.bind()
                self._texture_first_level = first_lvl
                self.compute_shader.uniform_float("aabb_min", (*aabb_min, 0.0))
                self.compute_shader.uniform_float("aabb_max", (*aabb_max, 0.0))
                self.compute_shader.uniform_int(
                    "total_num_nodes", int(self.scene_info["node_count"])
                )
                self.compute_shader.uniform_int("grid_size", grid_size)
                self.compute_shader.uniform_int("first_lvl", 1 if first_lvl else 0)
                self.compute_shader.uniform_int("active_capacity", self.active_capacity)
                self.compute_shader.uniform_int("tmp_capacity", self.tmp_capacity)
                self.compute_shader.uniform_int(
                    "active_counter_idx", _COUNTERS_ACTIVE_BASE + current_level
                )
                self.compute_shader.uniform_int(
                    "old_to_new_counter_idx",
                    _COUNTERS_OLD_TO_NEW_BASE + current_level,
                )
                self.compute_shader.uniform_int("status_counter_idx", _COUNTERS_STATUS)
                self._bind_culling_buffers(input_idx, output_idx)
                gpu.compute.dispatch(
                    self.compute_shader, num_groups, num_groups, num_groups
                )
                self._memory_barrier()
                self._texture_first_level = False

                first_lvl = False
                if current_level != grid_level:
                    input_idx, output_idx = output_idx, input_idx

            self.final_input_idx = input_idx
            self.final_output_idx = output_idx
            self.frame_culling_ms = (time.perf_counter() - start) * 1000.0

            if self.texture_scene_mode and _TEXTURE_COUNTER_READBACK:
                try:
                    counter_values = np.frombuffer(
                        self.counters_tex.read(), dtype=np.uint32
                    )
                    overflow = int(counter_values[_COUNTERS_STATUS]) != 0
                except Exception as exc:
                    runtime.debug_log(
                        f"Viewport texture counter readback failed: {exc}"
                    )
                    counter_values = np.zeros(_COUNTERS_SIZE, dtype=np.uint32)
                    overflow = True
            elif self.texture_scene_mode:
                counter_values = np.zeros(_COUNTERS_SIZE, dtype=np.uint32)
                overflow = False
            else:
                counter_values = np.frombuffer(
                    self.counters_ssbo.read(_COUNTERS_SIZE * 4), dtype=np.int32
                )
                overflow = int(counter_values[_COUNTERS_STATUS]) != 0
            active_counts = counter_values[
                _COUNTERS_ACTIVE_BASE : _COUNTERS_ACTIVE_BASE + 10
            ]
            tmp_counts = counter_values[
                _COUNTERS_OLD_TO_NEW_BASE : _COUNTERS_OLD_TO_NEW_BASE + 10
            ]
            self.max_active_count = (
                int(active_counts.max()) if active_counts.size else 0
            )
            self.max_tmp_count = int(tmp_counts.max()) if tmp_counts.size else 0

            if not overflow:
                self.culling_overflow = False
                self.last_overflow_message = ""
                self.pruning_key = pruning_key
                runtime.debug_log(
                    f"Viewport pruning ready: active={self.max_active_count:,}, tmp={self.max_tmp_count:,}, culling={self.frame_culling_ms:.2f}ms"
                )
                return

            if self._grow_work_buffers(self.scene_info["node_count"], grid_level):
                continue

            self.culling_overflow = True
            self.pruning_key = pruning_key
            self.last_overflow_message = "Viewport pruning buffers overflowed at max capacity; rendering full tree without pruning"
            runtime.debug_log(self.last_overflow_message)
            return

    def _allocated_pruning_bytes(self) -> int:
        if self.texture_scene_mode:
            return sum(
                0 if texture is None else int(texture.width) * int(texture.height) * 4
                for texture in (
                    self.parents_init_tex,
                    self.active_nodes_init_tex,
                    self.parents_tex[0],
                    self.parents_tex[1],
                    self.active_nodes_tex[0],
                    self.active_nodes_tex[1],
                    self.cell_offsets_tex[0],
                    self.cell_offsets_tex[1],
                    self.num_active_tex[0],
                    self.num_active_tex[1],
                    self.cell_errors_tex[0],
                    self.cell_errors_tex[1],
                    self.counters_tex,
                    self.old_to_new_scratch_tex,
                    self.tmp_tex,
                )
            )
        return sum(
            buffer.size
            for buffer in (
                self.parents_ssbo[0],
                self.parents_ssbo[1],
                self.active_nodes_ssbo[0],
                self.active_nodes_ssbo[1],
                self.cell_offsets_ssbo[0],
                self.cell_offsets_ssbo[1],
                self.num_active_ssbo[0],
                self.num_active_ssbo[1],
                self.cell_errors_ssbo[0],
                self.cell_errors_ssbo[1],
                self.counters_ssbo,
                self.old_to_new_scratch_ssbo,
                self.tmp_ssbo,
            )
        )

    def _allocated_tracing_bytes(self) -> int:
        if self.texture_scene_mode:
            return self.scene_texture_bytes + sum(
                0 if texture is None else int(texture.width) * int(texture.height) * 4
                for texture in (
                    self.active_nodes_tex[self.final_output_idx],
                    self.num_active_tex[self.final_output_idx],
                    self.cell_offsets_tex[self.final_output_idx],
                    self.cell_errors_tex[self.final_output_idx],
                )
            )
        return sum(
            buffer.size
            for buffer in (
                self.prims_ssbo,
                self.nodes_ssbo,
                self.binary_ops_ssbo,
                self.active_nodes_ssbo[self.final_output_idx],
                self.num_active_ssbo[self.final_output_idx],
                self.cell_offsets_ssbo[self.final_output_idx],
                self.cell_errors_ssbo[self.final_output_idx],
            )
        )

    def _set_draw_uniforms(
        self,
        context,
        scene,
        settings,
        width,
        height,
        aabb_min,
        aabb_max,
        camera_position,
        camera_target_value,
        camera_up,
        fov_y,
    ):
        region3d = context.space_data.region_3d
        renderer_to_blender = self._renderer_to_blender_matrix()
        mvp = region3d.perspective_matrix @ renderer_to_blender
        view = region3d.view_matrix @ renderer_to_blender
        view_inv = view.inverted()
        proj_inv = region3d.window_matrix.inverted()
        tan_half_fov = max(1e-4, float(np.tan(float(fov_y) * 0.5)))
        background = bridge.world_background_color(scene)
        use_culling = settings.culling_enabled and not self.culling_overflow
        shading = getattr(context.space_data, "shading", None)
        show_specular = int(bool(getattr(shading, "show_specular_highlight", True)))
        params = self.fragment_params_data
        params.aabb_min[:] = (*aabb_min, 0.0)
        params.aabb_max[:] = (*aabb_max, 0.0)
        params.ints0[:] = (
            int(width),
            int(height),
            int(self.scene_info["node_count"]),
            1 << bridge.grid_level(settings),
        )
        params.ints1[:] = (
            {
                "SHADED": 0,
                "HEATMAP": 1,
                "NORMALS": 2,
                "AO": 3,
            }[settings.shading_mode],
            int(use_culling),
            max(settings.num_samples, 1),
            show_specular,
        )
        params.floats0[:] = (
            float(settings.colormap_max),
            1.0,
            float(settings.gamma),
            0.0,
        )
        params.u_mvp[:] = _flatten_matrix_column_major(mvp)
        params.u_view[:] = _flatten_matrix_column_major(view)
        params.u_view_inv[:] = _flatten_matrix_column_major(view_inv)
        params.u_proj_inv[:] = _flatten_matrix_column_major(proj_inv)
        params.u_cam0[:] = (*camera_position, 0.0)
        params.u_cam1[:] = (*camera_target_value, 0.0)
        params.u_cam2[:] = (*camera_up, tan_half_fov)
        params.u_cam3[:] = (*background, 1.0)

        params_buffer = gpu.types.Buffer("UBYTE", ctypes.sizeof(params), params)
        if self.params_ubo is None:
            self.params_ubo = gpu.types.GPUUniformBuf(params_buffer)
        else:
            self.params_ubo.update(params_buffer)

        self.draw_shader.bind()
        self.draw_shader.uniform_block("params", self.params_ubo)
        diffuse_texture, specular_texture = matcap.get_matcap_textures(
            context, getattr(settings, "custom_matcap", "")
        )
        self.draw_shader.uniform_sampler("mopsMatcapTex", diffuse_texture)
        self.draw_shader.uniform_sampler("mopsMatcapSpecularTex", specular_texture)

    def draw(self, context, depsgraph):
        settings = depsgraph.scene.mathops_v2_settings
        try:
            if not self._ensure_shaders(settings.shading_mode):
                return False

            scene_path = bridge.resolve_scene_path(settings, create=True)
            if not scene_path.is_file():
                if not runtime.last_error_message:
                    bridge.set_last_error(f"Scene file not found: {scene_path}")
                return False

            self._sync_scene(scene_path, settings)
            aabb_min, aabb_max, _metadata = bridge.effective_aabb(settings, scene_path)
            self._update_pruning(scene_path, settings, aabb_min, aabb_max)

            width, height = bridge.get_viewport_render_size(context, settings)
            region_width = max(1, int(context.region.width))
            region_height = max(1, int(context.region.height))
            camera_position, camera_target_value, camera_up, fov_y = bridge.view_camera(
                context, depsgraph.scene
            )

            self._bind_draw_buffers()
            self._set_draw_uniforms(
                context,
                depsgraph.scene,
                settings,
                width,
                height,
                aabb_min,
                aabb_max,
                camera_position,
                camera_target_value,
                camera_up,
                fov_y,
            )

            start = time.perf_counter()
            if width != region_width or height != region_height:
                self._ensure_offscreen(width, height)
                runtime.debug_log(
                    f"Viewport internal size: {width}x{height} (region {region_width}x{region_height})"
                )
                with self.offscreen.bind():
                    fb = gpu.state.active_framebuffer_get()
                    fb.clear(color=(0.0, 0.0, 0.0, 0.0))
                    gpu.state.depth_test_set("NONE")
                    gpu.state.depth_mask_set(False)
                    gpu.state.blend_set("NONE")
                    self.batch.draw(self.draw_shader)

                gpu.state.depth_test_set("NONE")
                gpu.state.depth_mask_set(False)
                gpu.state.blend_set("NONE")
                draw_texture_2d(
                    self.offscreen.texture_color, (0, 0), region_width, region_height
                )
            else:
                gpu.state.depth_test_set("NONE")
                gpu.state.depth_mask_set(False)
                gpu.state.blend_set("NONE")
                self.batch.draw(self.draw_shader)
            tracing_ms = (time.perf_counter() - start) * 1000.0
            render_ms = tracing_ms + self.frame_culling_ms

            runtime.last_render_stats.update(
                {
                    "scene_name": scene_path.stem,
                    "scene_path": str(scene_path),
                    "node_count": int(self.scene_info["node_count"]),
                    "render_ms": render_ms,
                    "shader_ms": tracing_ms,
                    "upload_ms": 0.0,
                    "frame_ms": render_ms,
                    "tracing_ms": tracing_ms,
                    "culling_ms": self.frame_culling_ms,
                    "eval_grid_ms": 0.0,
                    "pruning_mem_gb": float(
                        self._allocated_pruning_bytes() / (1024.0 * 1024.0 * 1024.0)
                    ),
                    "tracing_mem_gb": float(
                        self._allocated_tracing_bytes() / (1024.0 * 1024.0 * 1024.0)
                    ),
                    "active_ratio": float(
                        self.max_active_count / self.active_capacity
                        if self.active_capacity
                        else 0.0
                    ),
                    "tmp_ratio": float(
                        self.max_tmp_count / self.tmp_capacity
                        if self.tmp_capacity
                        else 0.0
                    ),
                }
            )
            runtime.last_error_message = self.last_overflow_message
            return True
        except Exception as exc:
            runtime.debug_log(f"Viewport GPU path failed: {exc}")
            runtime.last_error_message = f"Viewport GPU path failed: {exc}"
            self.draw_shader = None
            self.compute_shader = None
            self.batch = None
            self.runtime_failed = True
            return False
