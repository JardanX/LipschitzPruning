#include "context.h"
#include "scene.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace {

glm::vec3 to_vec3(const std::vector<float>& value) {
    if (value.size() != 3) {
        throw std::runtime_error("camera vectors must contain exactly 3 floats");
    }
    return { value[0], value[1], value[2] };
}

class OffscreenRenderer {
public:
    OffscreenRenderer(
        const std::string& shader_dir,
        int width,
        int height,
        int final_grid_lvl = 8,
        int shading_mode = SHADING_MODE_SHADED,
        bool culling_enabled = true,
        int num_samples = 1,
        float gamma = 1.2f)
    {
        if (width <= 0 || height <= 0) {
            throw std::runtime_error("width and height must be positive");
        }

        spv_dir = shader_dir;
        ctx.initialize(false, final_grid_lvl, (uint32_t)width, (uint32_t)height, shading_mode);
        ctx.render_data.push_constants.alpha = 1.0f;
        ctx.render_data.culling_enabled = culling_enabled;
        ctx.render_data.num_samples = num_samples;
        ctx.render_data.gamma = gamma;
    }

    ~OffscreenRenderer() {
        close();
    }

    void close() {
        if (closed_) {
            return;
        }
        ctx.shutdown();
        scene_loaded = false;
        timings = {};
        closed_ = true;
    }

    void load_scene_file(const std::string& scene_path) {
        ensure_open();
        std::vector<CSGNode> nodes;
        glm::vec3 aabb_min;
        glm::vec3 aabb_max;
        load_json(scene_path.c_str(), nodes, aabb_min, aabb_max);
        upload_scene(nodes, aabb_min, aabb_max);
    }

    void load_scene_json(const std::string& scene_json) {
        ensure_open();
        std::vector<CSGNode> nodes;
        glm::vec3 aabb_min;
        glm::vec3 aabb_max;
        load_json_string(scene_json.c_str(), nodes, aabb_min, aabb_max);
        upload_scene(nodes, aabb_min, aabb_max);
    }

    void configure(
        bool culling_enabled,
        bool compute_culling,
        int num_samples,
        float gamma,
        int colormap_max,
        const std::vector<float>& background_color,
        float background_alpha)
    {
        ensure_open();
        ctx.render_data.culling_enabled = culling_enabled;
        ctx.render_data.compute_culling = compute_culling;
        ctx.render_data.num_samples = num_samples;
        ctx.render_data.gamma = gamma;
        ctx.render_data.colormap_max = colormap_max;
        ctx.render_data.background_color = to_vec3(background_color);
        ctx.render_data.push_constants.alpha = background_alpha;
    }

    void set_aabb(const std::vector<float>& aabb_min, const std::vector<float>& aabb_max) {
        ensure_open();
        ctx.render_data.aabb_min = to_vec3(aabb_min);
        ctx.render_data.aabb_max = to_vec3(aabb_max);
    }

    nb::bytes render_rgba(
        const std::vector<float>& camera_position,
        const std::vector<float>& camera_target,
        const std::vector<float>& camera_up,
        float fov_y,
        bool interactive)
    {
        ensure_open();
        if (!scene_loaded) {
            throw std::runtime_error("load_scene_file() or load_scene_json() must be called before render_rgba()");
        }

        timings = ctx.render(to_vec3(camera_position), to_vec3(camera_target), to_vec3(camera_up), fov_y, !interactive);
        std::vector<uint8_t> pixels = ctx.readback_rgba();
        return nb::bytes(reinterpret_cast<const char*>(pixels.data()), pixels.size());
    }

    nb::dict last_timings() const {
        ensure_open();
        nb::dict result;
        result["culling_ms"] = timings.culling_elapsed_ms;
        result["tracing_ms"] = timings.tracing_elapsed_ms;
        result["render_ms"] = timings.render_elapsed_ms;
        result["eval_grid_ms"] = timings.eval_grid_elapsed_ms;
        result["pruning_mem_gb"] = timings.pruning_mem_usage_gb;
        result["tracing_mem_gb"] = timings.tracing_mem_usage_gb;
        result["active_ratio"] = MAX_ACTIVE_COUNT > 0 ? float(ctx.render_data.max_active_count) / float(MAX_ACTIVE_COUNT) : 0.0f;
        result["tmp_ratio"] = MAX_TMP_COUNT > 0 ? float(ctx.render_data.max_tmp_count) / float(MAX_TMP_COUNT) : 0.0f;
        return result;
    }

    nb::dict scene_info() const {
        ensure_open();
        nb::dict result;
        result["node_count"] = ctx.render_data.total_num_nodes;
        result["aabb_min"] = std::vector<float>{ ctx.render_data.aabb_min.x, ctx.render_data.aabb_min.y, ctx.render_data.aabb_min.z };
        result["aabb_max"] = std::vector<float>{ ctx.render_data.aabb_max.x, ctx.render_data.aabb_max.y, ctx.render_data.aabb_max.z };
        return result;
    }

    int width() const {
        ensure_open();
        return (int)ctx.init.swapchain.extent.width;
    }

    int height() const {
        ensure_open();
        return (int)ctx.init.swapchain.extent.height;
    }

private:
    void ensure_open() const {
        if (closed_) {
            throw std::runtime_error("renderer is closed");
        }
    }

    void upload_scene(const std::vector<CSGNode>& nodes, const glm::vec3& aabb_min, const glm::vec3& aabb_max) {
        ctx.render_data.aabb_min = aabb_min;
        ctx.render_data.aabb_max = aabb_max;
        ctx.alloc_input_buffers((int)nodes.size());
        ctx.upload(nodes, 0);
        scene_loaded = true;
    }

    Context ctx;
    Timings timings{};
    bool scene_loaded = false;
    bool closed_ = false;
};

nb::dict pack_scene_file_impl(const std::string& scene_path) {
    std::vector<CSGNode> nodes;
    glm::vec3 aabb_min;
    glm::vec3 aabb_max;
    load_json(scene_path.c_str(), nodes, aabb_min, aabb_max);

    std::vector<GPUNode> gpu_nodes;
    std::vector<Primitive> primitives;
    std::vector<BinaryOp> binary_ops;
    std::vector<uint16_t> parents;
    std::vector<uint16_t> active_nodes;
    ConvertToGPUTree(0, nodes, gpu_nodes, primitives, binary_ops, parents, active_nodes);

    nb::dict result;
    result["node_count"] = int(gpu_nodes.size());
    result["primitive_count"] = int(primitives.size());
    result["binary_op_count"] = int(binary_ops.size());
    result["aabb_min"] = std::vector<float>{ aabb_min.x, aabb_min.y, aabb_min.z };
    result["aabb_max"] = std::vector<float>{ aabb_max.x, aabb_max.y, aabb_max.z };

    const char* prim_ptr = primitives.empty() ? nullptr : reinterpret_cast<const char*>(primitives.data());
    const char* node_ptr = gpu_nodes.empty() ? nullptr : reinterpret_cast<const char*>(gpu_nodes.data());
    const char* op_ptr = binary_ops.empty() ? nullptr : reinterpret_cast<const char*>(binary_ops.data());
    const char* parent_ptr = parents.empty() ? nullptr : reinterpret_cast<const char*>(parents.data());
    const char* active_ptr = active_nodes.empty() ? nullptr : reinterpret_cast<const char*>(active_nodes.data());

    result["primitives"] = nb::bytes(prim_ptr, primitives.size() * sizeof(Primitive));
    result["nodes"] = nb::bytes(node_ptr, gpu_nodes.size() * sizeof(GPUNode));
    result["binary_ops"] = nb::bytes(op_ptr, binary_ops.size() * sizeof(BinaryOp));
    result["parents"] = nb::bytes(parent_ptr, parents.size() * sizeof(uint16_t));
    result["active_nodes"] = nb::bytes(active_ptr, active_nodes.size() * sizeof(uint16_t));
    return result;
}

}

NB_MODULE(lipschitz_pruning_native, m) {
    m.doc() = "Lipschitz pruning Vulkan renderer bindings";

    m.attr("SHADING_MODE_SHADED") = SHADING_MODE_SHADED;
    m.attr("SHADING_MODE_HEATMAP") = SHADING_MODE_HEATMAP;
    m.attr("SHADING_MODE_NORMALS") = SHADING_MODE_NORMALS;
    m.attr("SHADING_MODE_AO") = SHADING_MODE_BEAUTY;
    m.def("pack_scene_file", &pack_scene_file_impl, nb::arg("scene_path"));

    nb::class_<OffscreenRenderer>(m, "Renderer")
        .def(nb::init<const std::string&, int, int, int, int, bool, int, float>(),
            nb::arg("shader_dir"),
            nb::arg("width"),
            nb::arg("height"),
            nb::arg("final_grid_level") = 8,
            nb::arg("shading_mode") = SHADING_MODE_SHADED,
            nb::arg("culling_enabled") = true,
            nb::arg("num_samples") = 1,
            nb::arg("gamma") = 1.2f)
        .def("load_scene_file", &OffscreenRenderer::load_scene_file, nb::arg("scene_path"))
        .def("load_scene_json", &OffscreenRenderer::load_scene_json, nb::arg("scene_json"))
        .def("close", &OffscreenRenderer::close)
        .def("configure", &OffscreenRenderer::configure,
            nb::arg("culling_enabled"),
            nb::arg("compute_culling"),
            nb::arg("num_samples"),
            nb::arg("gamma"),
            nb::arg("colormap_max"),
            nb::arg("background_color"),
            nb::arg("background_alpha"))
        .def("set_aabb", &OffscreenRenderer::set_aabb, nb::arg("aabb_min"), nb::arg("aabb_max"))
        .def("render_rgba", &OffscreenRenderer::render_rgba, nb::arg("camera_position"), nb::arg("camera_target"), nb::arg("camera_up"), nb::arg("fov_y") = 1.57079632679f, nb::arg("interactive") = false)
        .def("last_timings", &OffscreenRenderer::last_timings)
        .def("scene_info", &OffscreenRenderer::scene_info)
        .def_prop_ro("width", &OffscreenRenderer::width)
        .def_prop_ro("height", &OffscreenRenderer::height);
}
