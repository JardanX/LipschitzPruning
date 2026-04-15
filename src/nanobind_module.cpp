#include "context.h"
#include "scene.h"

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
        ctx.shutdown();
    }

    void load_scene_file(const std::string& scene_path) {
        std::vector<CSGNode> nodes;
        glm::vec3 aabb_min;
        glm::vec3 aabb_max;
        load_json(scene_path.c_str(), nodes, aabb_min, aabb_max);
        upload_scene(nodes, aabb_min, aabb_max);
    }

    void load_scene_json(const std::string& scene_json) {
        std::vector<CSGNode> nodes;
        glm::vec3 aabb_min;
        glm::vec3 aabb_max;
        load_json_string(scene_json.c_str(), nodes, aabb_min, aabb_max);
        upload_scene(nodes, aabb_min, aabb_max);
    }

    nb::bytes render_rgba(const std::vector<float>& camera_position, const std::vector<float>& camera_target) {
        if (!scene_loaded) {
            throw std::runtime_error("load_scene_file() or load_scene_json() must be called before render_rgba()");
        }

        timings = ctx.render(to_vec3(camera_position), to_vec3(camera_target));
        std::vector<uint8_t> pixels = ctx.readback_rgba();
        return nb::bytes(reinterpret_cast<const char*>(pixels.data()), pixels.size());
    }

    nb::dict last_timings() const {
        nb::dict result;
        result["culling_ms"] = timings.culling_elapsed_ms;
        result["tracing_ms"] = timings.tracing_elapsed_ms;
        result["render_ms"] = timings.render_elapsed_ms;
        result["eval_grid_ms"] = timings.eval_grid_elapsed_ms;
        result["pruning_mem_gb"] = timings.pruning_mem_usage_gb;
        result["tracing_mem_gb"] = timings.tracing_mem_usage_gb;
        return result;
    }

    int width() const {
        return (int)ctx.init.swapchain.extent.width;
    }

    int height() const {
        return (int)ctx.init.swapchain.extent.height;
    }

private:
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
};

}

NB_MODULE(lipschitz_pruning_native, m) {
    m.doc() = "Lipschitz pruning Vulkan renderer bindings";

    m.attr("SHADING_MODE_SHADED") = SHADING_MODE_SHADED;
    m.attr("SHADING_MODE_HEATMAP") = SHADING_MODE_HEATMAP;
    m.attr("SHADING_MODE_NORMALS") = SHADING_MODE_NORMALS;
    m.attr("SHADING_MODE_AO") = SHADING_MODE_BEAUTY;

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
        .def("render_rgba", &OffscreenRenderer::render_rgba, nb::arg("camera_position"), nb::arg("camera_target"))
        .def("last_timings", &OffscreenRenderer::last_timings)
        .def_prop_ro("width", &OffscreenRenderer::width)
        .def_prop_ro("height", &OffscreenRenderer::height);
}
