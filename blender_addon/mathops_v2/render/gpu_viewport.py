import math
import time

import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from .. import runtime
from ..nodes import sdf_tree
from ..shaders import pruning as pruning_shader
from ..shaders import raymarch
from . import pruning


_COUNTERS_ACTIVE_BASE = 0
_COUNTERS_TMP_BASE = 10
_COUNTER_STATUS = 20
_COUNTERS_SIZE = 21
_FLOAT32_EXACT_UINT_LIMIT = 1 << 24
_TEXTURE_DATA_WIDTH = 4096
_MAX_ACTIVE_CAP = 100_000_000
_MAX_TMP_CAP = 400_000_000


class MathOPSV2GPUViewport:
    def __init__(self):
        self._draw_shader = None
        self._compute_shader = None
        self._batch = None
        self._params_ubo = None
        self._scene_texture = None
        self._scene_hash = ""
        self._dummy_scalar_texture = None
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
        self._compute_shader = None
        self._batch = None
        self._params_ubo = None
        self._scene_texture = None
        self._scene_hash = ""
        self._dummy_scalar_texture = None
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

    def _ensure_draw_shader(self):
        if self._draw_shader is not None:
            return self._draw_shader

        interface = gpu.types.GPUStageInterfaceInfo("mathops_v2_interface")
        interface.smooth("VEC2", "uvInterp")

        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.push_constant("MAT4", "invViewProjectionMatrix")
        shader_info.typedef_source(raymarch.UNIFORMS_SOURCE)
        shader_info.uniform_buf(0, "MathOPSV2ViewParams", "mathops")
        shader_info.sampler(0, "FLOAT_2D", "sceneData")
        shader_info.sampler(1, "FLOAT_2D", "pruneActiveNodes")
        shader_info.sampler(2, "FLOAT_2D", "pruneCellOffsets")
        shader_info.sampler(3, "FLOAT_2D", "pruneCellCounts")
        shader_info.sampler(4, "FLOAT_2D", "pruneCellErrors")
        shader_info.vertex_in(0, "VEC2", "position")
        shader_info.vertex_out(interface)
        shader_info.fragment_out(0, "VEC4", "FragColor")
        shader_info.vertex_source(raymarch.VERTEX_SOURCE)
        shader_info.fragment_source(raymarch.FRAGMENT_SOURCE)
        self._draw_shader = gpu.shader.create_from_info(shader_info)
        return self._draw_shader

    def _ensure_compute_shader(self):
        if self._compute_shader is not None:
            return self._compute_shader

        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.push_constant("VEC4", "aabbMin")
        shader_info.push_constant("VEC4", "aabbMax")
        shader_info.push_constant("INT", "totalNumNodes")
        shader_info.push_constant("INT", "primitiveCount")
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
        bounds_min,
        bounds_max,
        primitive_count,
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

    def _update_pruning(self, settings, compiled, scene_texture):
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
                    bounds_min,
                    bounds_max,
                    primitive_count,
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

    def _ensure_params_ubo(self, settings, compiled, camera_position, light_direction):
        pruning_enabled = bool(self._final_active_nodes is not None)
        bounds_min, bounds_max = pruning.normalized_bounds(compiled)
        grid_size = float(pruning.final_grid_size(settings)) if pruning_enabled else 0.0

        viz_max = float(max(int(getattr(settings, "colormap_max", 25)), 1))
        data = [
            float(camera_position[0]),
            float(camera_position[1]),
            float(camera_position[2]),
            1.0,
            float(light_direction[0]),
            float(light_direction[1]),
            float(light_direction[2]),
            float(settings.surface_epsilon),
            float(settings.max_distance),
            float(settings.max_steps),
            float(compiled["primitive_count"]),
            float(compiled["instruction_count"]),
            float(bounds_min[0]),
            float(bounds_min[1]),
            float(bounds_min[2]),
            float(grid_size),
            float(bounds_max[0]),
            float(bounds_max[1]),
            float(bounds_max[2]),
            1.0 if pruning_enabled else 0.0,
            float(pruning.debug_mode_value(settings)),
            viz_max,
            0.0,
            0.0,
        ]
        buffer = gpu.types.Buffer("FLOAT", len(data), data)
        if self._params_ubo is None:
            self._params_ubo = gpu.types.GPUUniformBuf(buffer)
        else:
            self._params_ubo.update(buffer)
        return self._params_ubo

    def draw(self, context, depsgraph):
        del depsgraph
        scene = context.scene
        settings = runtime.scene_settings(scene)
        if settings is None:
            raise RuntimeError("MathOPS scene settings are unavailable")

        compiled = sdf_tree.compile_scene(scene)
        if compiled.get("message"):
            runtime.set_error(compiled["message"])
        else:
            runtime.clear_error()

        scene_texture = self._ensure_scene_texture(compiled)
        try:
            self._update_pruning(settings, compiled, scene_texture)
        except Exception:
            self._disable_pruning()

        shader = self._ensure_draw_shader()
        batch = self._ensure_batch()
        region_data = context.region_data
        inv_view_projection = region_data.perspective_matrix.inverted()
        camera_position = region_data.view_matrix.inverted().translation
        light_direction = runtime.normalize3(settings.light_direction)
        params_ubo = self._ensure_params_ubo(settings, compiled, camera_position, light_direction)

        gpu.state.depth_test_set("NONE")
        gpu.state.depth_mask_set(False)
        gpu.state.blend_set("NONE")

        shader.uniform_float("invViewProjectionMatrix", inv_view_projection)
        shader.uniform_block("mathops", params_ubo)
        shader.uniform_sampler("sceneData", scene_texture)
        shader.uniform_sampler("pruneActiveNodes", self._final_active_nodes or self._ensure_dummy_scalar_texture())
        shader.uniform_sampler("pruneCellOffsets", self._final_cell_offsets or self._ensure_dummy_scalar_texture())
        shader.uniform_sampler("pruneCellCounts", self._final_cell_counts or self._ensure_dummy_scalar_texture())
        shader.uniform_sampler("pruneCellErrors", self._final_cell_errors or self._ensure_dummy_scalar_texture())
        batch.draw(shader)

        compiled["pruning_active"] = bool(self._pruning_stats["active"])
        compiled["pruning_cells"] = int(self._pruning_stats["cells"])
        compiled["pruning_sequences"] = int(self._max_active_count)
        compiled["pruning_ms"] = float(self._pruning_stats["ms"])
        compiled["pruning_pending"] = False
        return compiled
