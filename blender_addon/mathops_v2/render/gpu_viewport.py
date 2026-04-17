import gpu
from gpu_extras.batch import batch_for_shader

from .. import runtime
from ..nodes import sdf_tree
from ..shaders import raymarch


class MathOPSV2GPUViewport:
    def __init__(self):
        self._shader = None
        self._batch = None
        self._params_ubo = None
        self._scene_texture = None
        self._scene_hash = ""

    def free(self):
        self._shader = None
        self._batch = None
        self._params_ubo = None
        self._scene_texture = None
        self._scene_hash = ""

    def _ensure_shader(self):
        if self._shader is not None:
            return self._shader

        interface = gpu.types.GPUStageInterfaceInfo("mathops_v2_interface")
        interface.smooth("VEC2", "uvInterp")

        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.push_constant("MAT4", "invViewProjectionMatrix")
        shader_info.typedef_source(raymarch.UNIFORMS_SOURCE)
        shader_info.uniform_buf(0, "MathOPSV2ViewParams", "mathops")
        shader_info.sampler(0, "FLOAT_2D", "sceneData")
        shader_info.vertex_in(0, "VEC2", "position")
        shader_info.vertex_out(interface)
        shader_info.fragment_out(0, "VEC4", "FragColor")
        shader_info.vertex_source(raymarch.VERTEX_SOURCE)
        shader_info.fragment_source(raymarch.FRAGMENT_SOURCE)

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

    def _ensure_scene_texture(self, compiled):
        scene_hash = str(compiled["hash"])
        if self._scene_texture is not None and self._scene_hash == scene_hash:
            return self._scene_texture

        rows = compiled["rows"] if compiled["rows"] else [(0.0, 0.0, 0.0, 0.0)]
        flat = []
        for row in rows:
            flat.extend(float(value) for value in row)
        buffer = gpu.types.Buffer("FLOAT", len(flat), flat)
        texture = gpu.types.GPUTexture((1, len(rows)), format="RGBA32F", data=buffer)
        try:
            texture.filter_mode(False)
        except Exception:
            pass
        self._scene_texture = texture
        self._scene_hash = scene_hash
        return texture

    def _ensure_params_ubo(self, settings, compiled, camera_position, light_direction):
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

        texture = self._ensure_scene_texture(compiled)
        shader = self._ensure_shader()
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
        shader.uniform_sampler("sceneData", texture)
        batch.draw(shader)
        return compiled
