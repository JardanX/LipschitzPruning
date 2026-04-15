import ctypes
import os
import re
import sys
import time
from pathlib import Path

import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix

from .. import runtime
from . import bridge


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

_COMMON_BUFFER_BLOCK_FRAGMENT = f"""
layout(std430, binding = {_BINDING_PRIMS}) buffer PrimitivesBuf {{ Primitive tab[]; }} prims;
layout(std430, binding = {_BINDING_NODES}) buffer NodesBuf {{ Node tab[]; }} nodes;
layout(std430, binding = {_BINDING_BINARY_OPS}) buffer BinaryOpsBuf {{ BinaryOp tab[]; }} binary_ops;
layout(std430, binding = {_BINDING_ACTIVE_NODES_OUT}) buffer ActiveNodesOutBuf {{ ActiveNode tab[]; }} active_nodes_out;
layout(std430, binding = {_BINDING_CELL_OFFSETS}) buffer CellOffsetsBuf {{ int tab[]; }} cells_offset;
layout(std430, binding = {_BINDING_CELL_COUNTS}) buffer CellCountsBuf {{ int tab[]; }} cells_num_active;
layout(std430, binding = {_BINDING_CELL_ERROR_OUT}) buffer CellErrorOutBuf {{ float tab[]; }} cell_error_out;
""".strip()

_COMMON_BUFFER_BLOCK_COMPUTE = f"""
layout(std430, binding = {_BINDING_PRIMS}) buffer PrimitivesBuf {{ Primitive tab[]; }} prims;
layout(std430, binding = {_BINDING_NODES}) buffer NodesBuf {{ Node tab[]; }} nodes;
layout(std430, binding = {_BINDING_BINARY_OPS}) buffer BinaryOpsBuf {{ BinaryOp tab[]; }} binary_ops;
layout(std430, binding = {_BINDING_PARENTS_IN}) buffer ParentsInBuf {{ uint tab[]; }} parents_in;
layout(std430, binding = {_BINDING_PARENTS_OUT}) buffer ParentsOutBuf {{ uint tab[]; }} parents_out;
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
#define mops_viz_max params.floats0.x
#define mops_alpha params.floats0.y
#define mops_gamma params.floats0.z
#define mops_u_mvp params.u_mvp
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
        ("u_cam0", ctypes.c_float * 4),
        ("u_cam1", ctypes.c_float * 4),
        ("u_cam2", ctypes.c_float * 4),
        ("u_cam3", ctypes.c_float * 4),
    ]


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


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


def _replace_common_buffer_refs(source: str, stage: str) -> str:
    block = (
        _COMMON_BUFFER_BLOCK_COMPUTE
        if stage == "compute"
        else _COMMON_BUFFER_BLOCK_FRAGMENT
    )
    replaced, count = _BUFFER_REFERENCE_RE.subn(
        block + "\n\nvoid set_bit", source, count=1
    )
    if count != 1:
        raise RuntimeError("Failed to adapt common.glsl buffer references")
    return replaced


def _replace_push_constants(source: str, stage: str) -> str:
    replaced, count = _PUSH_CONSTANT_RE.subn("", source, count=1)
    if count != 1:
        raise RuntimeError(f"Failed to adapt {stage} shader uniforms")
    return replaced


def _transform_shader_source(
    source: str, stage: str, shading_mode_value: int | None = None
) -> str:
    source = re.sub(r"^\s*#version[^\n]*\n", "", source, flags=re.M)
    source = re.sub(r"^\s*#extension[^\n]*\n", "", source, flags=re.M)
    source = source.replace(
        "layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;", ""
    )
    source = source.replace("uint16_t", "uint")
    source = _replace_common_buffer_refs(source, stage)
    source = _replace_push_constants(source, stage)
    source = _UNUSED_UINT64_RE.sub("", source, count=1)
    source = _PCG_BLOCK_RE.sub(_PCG_BLOCK_32 + "\n", source, count=1)
    source = source.replace("cam.tab[0]", "u_cam0")
    source = source.replace("cam.tab[1]", "u_cam1")
    source = source.replace("cam.tab[2]", "u_cam2")
    source = source.replace("cam.tab[3]", "u_cam3")
    source = source.replace("mvp.m", "u_mvp")

    if stage == "compute":
        source = source.replace(
            "shared ActiveNode s_parent_active_nodes[64];\nshared uint s_parent_node_parents[64];\n\n#define INVALID_INDEX 0xffffu\n",
            "shared int s_tmp_offset_shared[1];\n\n#define INVALID_INDEX 0xffffu\n",
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
        source = source.replace(
            "    int tmp_offset = atomicAdd(counters.tab[old_to_new_counter_idx], num_nodes);\n",
            "    int tmp_offset = -1;\n"
            "    if (gl_LocalInvocationIndex == 0u) {\n"
            "        s_tmp_offset_shared[0] = atomicAdd(counters.tab[old_to_new_counter_idx], 64 * num_nodes);\n"
            "    }\n"
            "    barrier();\n"
            "    tmp_offset = s_tmp_offset_shared[0];\n"
            "    int lane_idx = int(gl_LocalInvocationIndex);\n"
            "    if (tmp_offset + 64 * num_nodes > tmp_capacity) {\n"
            "        counters.tab[status_counter_idx] = 1;\n"
            "        num_active_out.tab[cell_idx] = 0;\n"
            "        child_cells_offset.tab[cell_idx] = 0;\n"
            "        cell_value_out.tab[cell_idx] = 0.0;\n"
            "        return;\n"
            "    }\n",
        )
        source = source.replace(
            "    int cell_offset = atomicAdd(counters.tab[active_counter_idx], cell_num_active);\n",
            "    int cell_offset = atomicAdd(counters.tab[active_counter_idx], cell_num_active);\n"
            "    if (cell_offset + cell_num_active > active_capacity) {\n"
            "        counters.tab[status_counter_idx] = 1;\n"
            "        num_active_out.tab[cell_idx] = 0;\n"
            "        child_cells_offset.tab[cell_idx] = 0;\n"
            "        cell_value_out.tab[cell_idx] = 0.0;\n"
            "        return;\n"
            "    }\n",
        )
        source = source.replace(
            "tmp.tab[tmp_offset + right_entry.idx]",
            "tmp.tab[tmp_offset + 64 * right_entry.idx + lane_idx]",
        )
        source = source.replace(
            "tmp.tab[tmp_offset + left_entry.idx]",
            "tmp.tab[tmp_offset + 64 * left_entry.idx + lane_idx]",
        )
        source = source.replace(
            "tmp.tab[tmp_offset + i]",
            "tmp.tab[tmp_offset + 64 * i + lane_idx]",
        )
        source = source.replace(
            "tmp.tab[tmp_offset + parent_idx]",
            "tmp.tab[tmp_offset + 64 * int(parent_idx) + lane_idx]",
        )
        source = source.replace(
            "old_to_new_scratch.tab[tmp_offset + i]",
            "old_to_new_scratch.tab[tmp_offset + 64 * i + lane_idx]",
        )
        source = source.replace(
            "old_to_new_scratch.tab[tmp_offset + new_parent_old_idx]",
            "old_to_new_scratch.tab[tmp_offset + 64 * new_parent_old_idx + lane_idx]",
        )
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
            "    int num_active = (cells_num_active.tab[cell_idx] + 1 )/ 2;\n",
            "    int num_active = bool(mops_culling_enabled) ? (cells_num_active.tab[cell_idx] + 1) / 2 : (mops_total_num_nodes + 1) / 2;\n",
        )
        source = source.replace(
            "            gl_FragDepth = projected_depth;\n",
            "            gl_FragDepth = projected_depth * 0.5 + 0.5;\n",
        )
        source = source.replace(
            "    outColor /= mops_num_samples;\n",
            "    outColor /= float(max(mops_num_samples, 1));\n",
        )

    return source.strip() + "\n"


def _shader_source(
    filename: str, stage: str, shading_mode_value: int | None = None
) -> str:
    key = (filename, stage, shading_mode_value)
    cached = _SHADER_CACHE.get(key)
    if cached is not None:
        return cached

    source = _expand_includes(_shader_root() / filename, {})
    source = _transform_shader_source(source, stage, shading_mode_value)
    _SHADER_CACHE[key] = source
    return source


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
        self.compute_shader = None
        self.batch = None
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
        self._parents_init_bytes = b""
        self._active_init_bytes = b""
        self.fragment_params_data = _FragmentParamsUBO()
        self.params_ubo = None

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
        self.compute_shader = None
        self.batch = None
        self.scene_key = None
        self.pruning_key = None
        self.scene_info = None
        self.runtime_failed = False
        self._buffer_config_key = None
        self._parents_init_bytes = b""
        self._active_init_bytes = b""
        self.params_ubo = None
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

    def supported(self) -> bool:
        try:
            return gpu.platform.backend_type_get() == "OPENGL"
        except Exception:
            return False

    def _ensure_shaders(self, shading_mode: str) -> bool:
        if self.runtime_failed:
            return False
        if (
            self.draw_shader is not None
            and self.compute_shader is not None
            and self.draw_shader_mode == shading_mode
        ):
            return True
        if not self.supported():
            if not self.logged_backend_warning:
                runtime.debug_log(
                    "Exact viewport pruning requires Blender's OpenGL backend; falling back to Vulkan readback path"
                )
                self.logged_backend_warning = True
            return False

        try:
            compute_shader_info = gpu.types.GPUShaderCreateInfo()
            compute_shader_info.local_group_size(4, 4, 4)
            _declare_push_constants(compute_shader_info, _COMPUTE_PUSH_CONSTANTS)
            compute_shader_info.compute_source(
                _shader_source("culling.comp.glsl", "compute")
            )
            self.compute_shader = gpu.shader.create_from_info(compute_shader_info)

            shading_mode_value = {
                "SHADED": 0,
                "HEATMAP": 1,
                "NORMALS": 2,
                "AO": 3,
            }[shading_mode]

            shader_info = gpu.types.GPUShaderCreateInfo()
            shader_info.vertex_in(0, "VEC2", "pos")
            shader_info.fragment_out(0, "VEC4", "outColor")
            shader_info.depth_write("ANY")
            shader_info.typedef_source(_FRAGMENT_PARAMS_TYPEDEF)
            shader_info.uniform_buf(0, "ViewportParams", "params")
            shader_info.vertex_source(
                "void main(){  gl_Position = vec4(pos, 0.0, 1.0);}"
            )
            shader_info.fragment_source(
                _shader_source(
                    "simple.frag.glsl",
                    "fragment",
                    shading_mode_value=shading_mode_value,
                )
            )
            self.draw_shader = gpu.shader.create_from_info(shader_info)
            self.draw_shader_mode = shading_mode

            if self.batch is None:
                verts = ((-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0))
                self.batch = batch_for_shader(self.draw_shader, "TRIS", {"pos": verts})
            return True
        except Exception as exc:
            if not self.logged_shader_failure:
                runtime.debug_log(
                    f"Failed to compile exact viewport GPU shaders: {exc}"
                )
                self.logged_shader_failure = True
            self.draw_shader = None
            self.draw_shader_mode = None
            self.compute_shader = None
            self.batch = None
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

    def _u32_bytes(self, data: bytes) -> bytes:
        if not data:
            return b""
        return np.frombuffer(data, dtype=np.uint16).astype(np.uint32).tobytes()

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
        return active_capacity, tmp_capacity

    def _ensure_work_buffers(
        self,
        node_count: int,
        grid_level: int,
        active_capacity: int | None = None,
        tmp_capacity: int | None = None,
    ):
        default_active, default_tmp = self._capacity_for_scene(node_count)
        if active_capacity is None:
            active_capacity = max(default_active, self.active_capacity)
        if tmp_capacity is None:
            tmp_capacity = max(default_tmp, self.tmp_capacity)
        grid_size = 1 << grid_level
        num_cells = grid_size * grid_size * grid_size
        key = (node_count, grid_level, active_capacity, tmp_capacity)
        if key == self._buffer_config_key:
            return

        self.active_capacity = active_capacity
        self.tmp_capacity = tmp_capacity
        self.num_cells = num_cells
        self.final_grid_level = grid_level

        for buffer in self.parents_ssbo:
            buffer.ensure_size(active_capacity * 4)
        for buffer in self.active_nodes_ssbo:
            buffer.ensure_size(active_capacity * 4)
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
        scene_key = (str(scene_path.resolve()), scene_path.stat().st_mtime_ns)
        if self.scene_key != scene_key:
            native_module = bridge.load_native_module()
            packed = native_module.pack_scene_file(str(scene_path))
            self.prims_ssbo.upload(bytes(packed["primitives"]))
            self.nodes_ssbo.upload(bytes(packed["nodes"]))
            self.binary_ops_ssbo.upload(bytes(packed["binary_ops"]))
            self._parents_init_bytes = self._u32_bytes(bytes(packed["parents"]))
            self._active_init_bytes = self._u32_bytes(bytes(packed["active_nodes"]))
            self.scene_info = {
                "aabb_min": tuple(packed["aabb_min"]),
                "aabb_max": tuple(packed["aabb_max"]),
                "node_count": int(packed["node_count"]),
            }
            self.scene_key = scene_key
            self.pruning_key = None
            self.culling_overflow = False
            self.last_overflow_message = ""
            runtime.debug_log(
                f"Viewport scene packed: {scene_path.name}, nodes={self.scene_info['node_count']}"
            )

        self._ensure_work_buffers(
            self.scene_info["node_count"], bridge.grid_level(settings)
        )

    def _reset_pruning_state(self):
        self.parents_ssbo[0].update(self._parents_init_bytes)
        self.active_nodes_ssbo[0].update(self._active_init_bytes)
        self.counters_ssbo.update(np.zeros(_COUNTERS_SIZE, dtype=np.int32).tobytes())
        zero_f32 = np.zeros(1, dtype=np.float32).tobytes()
        self.cell_errors_ssbo[0].update(zero_f32)
        self.cell_errors_ssbo[1].update(zero_f32)
        self.max_active_count = 0
        self.max_tmp_count = 0
        self.culling_overflow = False
        self.last_overflow_message = ""

    def _bind_culling_buffers(self, input_idx: int, output_idx: int):
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
        self.prims_ssbo.bind(_BINDING_PRIMS)
        self.nodes_ssbo.bind(_BINDING_NODES)
        self.binary_ops_ssbo.bind(_BINDING_BINARY_OPS)
        self.active_nodes_ssbo[self.final_output_idx].bind(_BINDING_ACTIVE_NODES_OUT)
        self.cell_offsets_ssbo[self.final_output_idx].bind(_BINDING_CELL_OFFSETS)
        self.num_active_ssbo[self.final_output_idx].bind(_BINDING_CELL_COUNTS)
        self.cell_errors_ssbo[self.final_output_idx].bind(_BINDING_CELL_ERROR_OUT)

    def _memory_barrier(self):
        _gl_functions()["glMemoryBarrier"](
            _GL_SHADER_STORAGE_BARRIER_BIT | _GL_BUFFER_UPDATE_BARRIER_BIT
        )

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

                first_lvl = False
                if current_level != grid_level:
                    input_idx, output_idx = output_idx, input_idx

            self.final_input_idx = input_idx
            self.final_output_idx = output_idx
            self.frame_culling_ms = (time.perf_counter() - start) * 1000.0

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
        mvp = region3d.perspective_matrix @ self._renderer_to_blender_matrix()
        tan_half_fov = max(1e-4, float(np.tan(float(fov_y) * 0.5)))
        background = bridge.world_background_color(scene)
        use_culling = settings.culling_enabled and not self.culling_overflow
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
            0,
        )
        params.floats0[:] = (
            float(settings.colormap_max),
            1.0,
            float(settings.gamma),
            0.0,
        )
        params.u_mvp[:] = tuple(value for row in mvp for value in row)
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

    def draw(self, context, depsgraph):
        try:
            settings = depsgraph.scene.mathops_v2_settings
            if not self._ensure_shaders(settings.shading_mode):
                return False

            scene_path = bridge.resolve_scene_path(settings)
            if not scene_path.is_file():
                return False

            self._sync_scene(scene_path, settings)
            aabb_min, aabb_max, _metadata = bridge.effective_aabb(settings, scene_path)
            self._update_pruning(scene_path, settings, aabb_min, aabb_max)

            width, height = bridge.get_viewport_render_size(context, settings)
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
            gpu.state.depth_test_set("LESS_EQUAL")
            gpu.state.depth_mask_set(True)
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
            self.draw_shader = None
            self.compute_shader = None
            self.batch = None
            self.runtime_failed = True
            return False
