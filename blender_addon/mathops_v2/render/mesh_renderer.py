import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from .. import runtime


_VERTEX_SOURCE = """
void main()
{
  vec4 worldPos = modelMatrix * vec4(pos, 1.0);
  worldPos_f = worldPos.xyz;
  worldNor_f = normalize(mat3(modelMatrix) * nor);
  gl_Position = viewProjectionMatrix * worldPos;
}
"""

_FRAGMENT_SOURCE = """
void main()
{
  FragColor = vec4(0.5, 0.5, 0.5, 1.0);
  OutlineId = 0.0;
  HitPosition = vec4(worldPos_f, 1.0);
  SurfaceNormal = vec4(worldNor_f, 1.0);
}
"""

_WIRE_VERTEX = """
void main()
{
  gl_Position = viewProjectionMatrix * modelMatrix * vec4(pos, 1.0);
}
"""

_WIRE_FRAGMENT = """
void main()
{
  FragColor = wireColor;
}
"""


class MathOPSV2MeshRenderer:
    def __init__(self):
        self._shader = None
        self._wire_shader = None

    def free(self):
        self._shader = None
        self._wire_shader = None

    def _ensure_shader(self):
        if self._shader is not None:
            return self._shader

        interface = gpu.types.GPUStageInterfaceInfo("mathops_mesh_iface")
        interface.smooth("VEC3", "worldPos_f")
        interface.smooth("VEC3", "worldNor_f")

        info = gpu.types.GPUShaderCreateInfo()
        info.push_constant("MAT4", "viewProjectionMatrix")
        info.push_constant("MAT4", "modelMatrix")
        info.vertex_in(0, "VEC3", "pos")
        info.vertex_in(1, "VEC3", "nor")
        info.vertex_out(interface)
        info.fragment_out(0, "VEC4", "FragColor")
        info.fragment_out(1, "FLOAT", "OutlineId")
        info.fragment_out(2, "VEC4", "HitPosition")
        info.fragment_out(3, "VEC4", "SurfaceNormal")
        info.depth_write("ANY")
        info.vertex_source(_VERTEX_SOURCE)
        info.fragment_source(_FRAGMENT_SOURCE)
        self._shader = gpu.shader.create_from_info(info)
        return self._shader

    def _ensure_wire_shader(self):
        if self._wire_shader is not None:
            return self._wire_shader

        info = gpu.types.GPUShaderCreateInfo()
        info.push_constant("MAT4", "viewProjectionMatrix")
        info.push_constant("MAT4", "modelMatrix")
        info.push_constant("VEC4", "wireColor")
        info.vertex_in(0, "VEC3", "pos")
        info.fragment_out(0, "VEC4", "FragColor")
        info.depth_write("ANY")
        info.vertex_source(_WIRE_VERTEX)
        info.fragment_source(_WIRE_FRAGMENT)
        self._wire_shader = gpu.shader.create_from_info(info)
        return self._wire_shader

    def _mesh_triangle_data(self, mesh):
        mesh.calc_loop_triangles()
        tris = mesh.loop_triangles
        tri_count = len(tris)
        if tri_count == 0:
            return None, None

        vert_count = len(mesh.vertices)

        vert_pos = np.empty(vert_count * 3, dtype=np.float32)
        mesh.vertices.foreach_get('co', vert_pos)
        vert_pos = vert_pos.reshape(-1, 3)

        vert_nor = np.empty(vert_count * 3, dtype=np.float32)
        mesh.vertices.foreach_get('normal', vert_nor)
        vert_nor = vert_nor.reshape(-1, 3)

        tri_indices = np.empty(tri_count * 3, dtype=np.int32)
        tris.foreach_get('vertices', tri_indices)

        positions = np.ascontiguousarray(vert_pos[tri_indices])
        normals = np.ascontiguousarray(vert_nor[tri_indices])
        return positions, normals

    def _mesh_edge_data(self, mesh):
        edges = mesh.edges
        edge_count = len(edges)
        if edge_count == 0:
            return None

        vert_count = len(mesh.vertices)
        vert_pos = np.empty(vert_count * 3, dtype=np.float32)
        mesh.vertices.foreach_get('co', vert_pos)
        vert_pos = vert_pos.reshape(-1, 3)

        edge_indices = np.empty(edge_count * 2, dtype=np.int32)
        edges.foreach_get('vertices', edge_indices)

        return np.ascontiguousarray(vert_pos[edge_indices])

    def _object_mesh(self, obj, depsgraph, is_edit):
        if is_edit:
            return obj.data, None
        try:
            eval_obj = obj.evaluated_get(depsgraph)
            mesh = eval_obj.to_mesh()
            if mesh is not None:
                mesh.calc_loop_triangles()
            return mesh, eval_obj
        except Exception:
            return None, None

    def _should_render_object(self, obj):
        if obj.type != "MESH":
            return False
        if getattr(obj, "hide_viewport", False):
            return False
        settings = runtime.object_settings(obj)
        if settings is not None and settings.enabled:
            return False
        return True

    def draw(self, context, depsgraph, backface_culling, view_projection):
        shader = self._ensure_shader()

        active_obj = getattr(context, "active_object", None)
        active_in_edit = (
            active_obj is not None
            and active_obj.type == "MESH"
            and active_obj.mode == "EDIT"
        )

        visible = []
        try:
            for obj in depsgraph.objects:
                if self._should_render_object(obj):
                    visible.append(obj)
        except Exception:
            scene = getattr(context, "scene", None)
            if scene is not None:
                for obj in scene.objects:
                    if self._should_render_object(obj):
                        visible.append(obj)

        if not visible:
            return

        if backface_culling:
            gpu.state.face_culling_set("BACK")
        else:
            gpu.state.face_culling_set("NONE")

        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(True)
        gpu.state.blend_set("NONE")

        shader.uniform_float("viewProjectionMatrix", view_projection)

        for obj in visible:
            is_edit = active_in_edit and obj == active_obj
            mesh, eval_obj = self._object_mesh(obj, depsgraph, is_edit)
            if mesh is None:
                continue
            try:
                positions, normals = self._mesh_triangle_data(mesh)
                if positions is None:
                    continue
                batch = batch_for_shader(shader, "TRIS", {"pos": positions, "nor": normals})
                shader.uniform_float("modelMatrix", obj.matrix_world)
                batch.draw(shader)
            except Exception:
                pass
            finally:
                if eval_obj is not None:
                    try:
                        eval_obj.to_mesh_clear()
                    except Exception:
                        pass

        if active_in_edit and active_obj is not None:
            self._draw_edit_wireframe(active_obj, view_projection)

        gpu.state.face_culling_set("NONE")

    def _draw_edit_wireframe(self, obj, view_projection):
        wire_shader = self._ensure_wire_shader()
        mesh = obj.data
        try:
            edge_positions = self._mesh_edge_data(mesh)
            if edge_positions is None:
                return
            batch = batch_for_shader(wire_shader, "LINES", {"pos": edge_positions})
            gpu.state.depth_test_set("LESS_EQUAL")
            gpu.state.depth_mask_set(False)
            gpu.state.blend_set("ALPHA")
            wire_shader.uniform_float("viewProjectionMatrix", view_projection)
            wire_shader.uniform_float("modelMatrix", obj.matrix_world)
            wire_shader.uniform_float("wireColor", (0.0, 0.0, 0.0, 0.15))
            batch.draw(wire_shader)
        except Exception:
            pass
