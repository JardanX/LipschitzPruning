#include "context.h"
#include "scene.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef MATHOPS_WITH_OPENEXR
#  include <OpenEXR/ImfChannelList.h>
#  include <OpenEXR/ImfFrameBuffer.h>
#  include <OpenEXR/ImfHeader.h>
#  include <OpenEXR/ImfInputFile.h>
#endif

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace {

#ifdef MATHOPS_WITH_OPENEXR
bool starts_with(const std::string& value, const std::string& prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

bool has_layer_prefix(const OPENEXR_IMF_NAMESPACE::ChannelList& channels, const std::string& prefix) {
    const std::string match = prefix + ".";
    for (auto it = channels.begin(); it != channels.end(); ++it) {
        if (starts_with(it.name(), match)) {
            return true;
        }
    }
    return false;
}

std::vector<float> read_layer_rgba(
    OPENEXR_IMF_NAMESPACE::InputFile& file,
    const OPENEXR_IMF_NAMESPACE::ChannelList& channels,
    const IMATH_NAMESPACE::Box2i& data_window,
    int width,
    int height,
    const std::string& prefix)
{
    static constexpr const char* suffixes[4] = {"R", "G", "B", "A"};
    std::vector<float> planes[4];
    bool present[4] = {false, false, false, false};
    OPENEXR_IMF_NAMESPACE::FrameBuffer frame_buffer;
    const std::int64_t row_stride = std::int64_t(width) * sizeof(float);
    bool has_any_channel = false;

    for (int channel_index = 0; channel_index < 4; ++channel_index) {
        std::string channel_name = prefix.empty() ? suffixes[channel_index] : prefix + "." + suffixes[channel_index];
        if (channels.findChannel(channel_name.c_str()) == nullptr) {
            continue;
        }

        has_any_channel = true;
        present[channel_index] = true;
        planes[channel_index].assign(std::size_t(width) * std::size_t(height), 0.0f);
        char* base = reinterpret_cast<char*>(planes[channel_index].data()) -
                     std::ptrdiff_t(data_window.min.x) * sizeof(float) -
                     std::ptrdiff_t(data_window.min.y) * row_stride;
        frame_buffer.insert(
            channel_name.c_str(),
            OPENEXR_IMF_NAMESPACE::Slice(
                OPENEXR_IMF_NAMESPACE::FLOAT,
                base,
                sizeof(float),
                row_stride));
    }

    if (!has_any_channel) {
        throw std::runtime_error("EXR matcap layer is missing RGBA channels");
    }

    file.setFrameBuffer(frame_buffer);
    file.readPixels(data_window.min.y, data_window.max.y);

    std::vector<float> rgba(std::size_t(width) * std::size_t(height) * 4u, 0.0f);
    for (int y = 0; y < height; ++y) {
        const int source_y = height - 1 - y;
        for (int x = 0; x < width; ++x) {
            const std::size_t src_idx = std::size_t(source_y) * std::size_t(width) + std::size_t(x);
            const std::size_t dst_idx = (std::size_t(y) * std::size_t(width) + std::size_t(x)) * 4u;
            for (int channel_index = 0; channel_index < 4; ++channel_index) {
                rgba[dst_idx + std::size_t(channel_index)] = present[channel_index] ?
                    planes[channel_index][src_idx] : (channel_index == 3 ? 1.0f : 0.0f);
            }
        }
    }

    return rgba;
}

nb::dict extract_matcap_exr_impl(const std::string& path) {
    OPENEXR_IMF_NAMESPACE::InputFile file(path.c_str());
    const OPENEXR_IMF_NAMESPACE::Header& header = file.header();
    const OPENEXR_IMF_NAMESPACE::ChannelList& channels = header.channels();
    const IMATH_NAMESPACE::Box2i& data_window = header.dataWindow();
    const int width = data_window.max.x - data_window.min.x + 1;
    const int height = data_window.max.y - data_window.min.y + 1;
    const bool has_diffuse = has_layer_prefix(channels, "diffuse");
    const bool has_specular = has_diffuse && has_layer_prefix(channels, "specular");

    std::vector<float> diffuse = read_layer_rgba(
        file,
        channels,
        data_window,
        width,
        height,
        has_diffuse ? "diffuse" : "");
    std::vector<float> specular;
    if (has_specular) {
        specular = read_layer_rgba(file, channels, data_window, width, height, "specular");
    }

    nb::dict result;
    result["width"] = width;
    result["height"] = height;
    result["has_specular"] = has_specular;
    result["diffuse"] = nb::bytes(
        reinterpret_cast<const char*>(diffuse.data()),
        diffuse.size() * sizeof(float));
    result["specular"] = has_specular ?
        nb::bytes(reinterpret_cast<const char*>(specular.data()), specular.size() * sizeof(float)) :
        nb::bytes("", 0);
    return result;
}
#else
nb::dict extract_matcap_exr_impl(const std::string& /*path*/) {
    throw std::runtime_error("Native EXR matcap support is unavailable in this build");
}
#endif

float primitive_distance_scale(const glm::mat4& world_to_prim)
{
    glm::vec3 row0(world_to_prim[0][0], world_to_prim[1][0], world_to_prim[2][0]);
    glm::vec3 row1(world_to_prim[0][1], world_to_prim[1][1], world_to_prim[2][1]);
    glm::vec3 row2(world_to_prim[0][2], world_to_prim[1][2], world_to_prim[2][2]);
    float max_inv_scale = std::max(std::max(glm::length(row0), glm::length(row1)), glm::length(row2));
    return max_inv_scale > 0.0f ? 1.0f / max_inv_scale : 1.0f;
}

glm::mat4 payload_matrix_to_glm(const nb::list& values);
void load_payload_dict(
    const nb::dict& payload,
    std::vector<CSGNode>& nodes,
    glm::vec3& aabb_min,
    glm::vec3& aabb_max);

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

    void load_scene_payload(const nb::dict& scene_payload) {
        ensure_open();
        std::vector<CSGNode> nodes;
        glm::vec3 aabb_min;
        glm::vec3 aabb_max;
        load_payload_dict(scene_payload, nodes, aabb_min, aabb_max);
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
            throw std::runtime_error("load_scene_file(), load_scene_json(), or load_scene_payload() must be called before render_rgba()");
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

nb::dict pack_scene_nodes_impl(
    const std::vector<CSGNode>& nodes,
    int root_idx,
    const glm::vec3& aabb_min,
    const glm::vec3& aabb_max)
{
    std::vector<GPUNode> gpu_nodes;
    std::vector<Primitive> primitives;
    std::vector<BinaryOp> binary_ops;
    std::vector<uint16_t> parents;
    std::vector<uint16_t> active_nodes;
    ConvertToGPUTree(root_idx, nodes, gpu_nodes, primitives, binary_ops, parents, active_nodes);

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

glm::mat4 payload_matrix_to_glm(const nb::list& values)
{
    return glm::mat4(
        nb::cast<float>(values[0]), nb::cast<float>(values[4]), nb::cast<float>(values[8]), 0.0f,
        nb::cast<float>(values[1]), nb::cast<float>(values[5]), nb::cast<float>(values[9]), 0.0f,
        nb::cast<float>(values[2]), nb::cast<float>(values[6]), nb::cast<float>(values[10]), 0.0f,
        nb::cast<float>(values[3]), nb::cast<float>(values[7]), nb::cast<float>(values[11]), 1.0f);
}

void load_payload_dict(
    const nb::dict& payload,
    std::vector<CSGNode>& nodes,
    glm::vec3& aabb_min,
    glm::vec3& aabb_max)
{
    nb::list aabb_min_arr = nb::cast<nb::list>(payload["aabb_min"]);
    nb::list aabb_max_arr = nb::cast<nb::list>(payload["aabb_max"]);
    aabb_min = {
        nb::cast<float>(aabb_min_arr[0]),
        nb::cast<float>(aabb_min_arr[1]),
        nb::cast<float>(aabb_min_arr[2]),
    };
    aabb_max = {
        nb::cast<float>(aabb_max_arr[0]),
        nb::cast<float>(aabb_max_arr[1]),
        nb::cast<float>(aabb_max_arr[2]),
    };

    struct StackEntry {
        nb::dict payload;
        glm::mat4 mat;
        int idx;
        bool sign;
    };

    nodes.clear();
    nodes.push_back({});
    std::vector<StackEntry> stack = {{payload, glm::mat4(1.0f), 0, true}};

    while (!stack.empty()) {
        StackEntry entry = stack.back();
        stack.pop_back();

        const nb::dict& node_payload = entry.payload;
        glm::mat4 node_mat = payload_matrix_to_glm(nb::cast<nb::list>(node_payload["matrix"]));
        glm::mat4 mat = entry.mat;
        std::string type = nb::cast<std::string>(node_payload["nodeType"]);
        int node_idx = entry.idx;

        if (type == "primitive") {
            CSGNode& node = nodes[node_idx];
            node.type = NODETYPE_PRIMITIVE;
            node.left = -1;
            node.right = -1;
            node.sign = entry.sign;

            glm::mat4 world_to_prim = mat * node_mat;
            node.primitive.m_row0 = glm::vec4(world_to_prim[0][0], world_to_prim[1][0], world_to_prim[2][0], world_to_prim[3][0]);
            node.primitive.m_row1 = glm::vec4(world_to_prim[0][1], world_to_prim[1][1], world_to_prim[2][1], world_to_prim[3][1]);
            node.primitive.m_row2 = glm::vec4(world_to_prim[0][2], world_to_prim[1][2], world_to_prim[2][2], world_to_prim[3][2]);
            node.primitive.pad0 = primitive_distance_scale(world_to_prim);
            node.primitive.pad1 = 0.0f;
            node.primitive.pad2 = 0.0f;
            node.primitive.extrude_rounding.x = nb::cast<float>(node_payload["round_x"]);
            node.primitive.extrude_rounding.y = nb::cast<float>(node_payload["round_y"]);

            {
                nb::list color = nb::cast<nb::list>(node_payload["color"]);
                node.primitive.color = 0;
                node.primitive.color |= std::min((uint32_t)(nb::cast<float>(color[0]) * 255.99f), 255u) << 0;
                node.primitive.color |= std::min((uint32_t)(nb::cast<float>(color[1]) * 255.99f), 255u) << 8;
                node.primitive.color |= std::min((uint32_t)(nb::cast<float>(color[2]) * 255.99f), 255u) << 16;
            }

            std::string primitive_type = nb::cast<std::string>(node_payload["primitiveType"]);
            if (primitive_type == "sphere") {
                node.primitive.type = PRIMITIVE_SPHERE;
            } else if (primitive_type == "box") {
                node.primitive.type = PRIMITIVE_BOX;
            } else if (primitive_type == "cylinder") {
                node.primitive.type = PRIMITIVE_CYLINDER;
            } else if (primitive_type == "cone") {
                node.primitive.type = PRIMITIVE_CONE;
            } else {
                throw std::runtime_error("Unknown primitive: " + primitive_type);
            }

            switch (node.primitive.type) {
                case PRIMITIVE_SPHERE: {
                    float radius = nb::cast<float>(node_payload["radius"]);
                    node.primitive.sphere.radius = glm::vec4(radius, 0.0f, 0.0f, 0.0f);
                    break;
                }
                case PRIMITIVE_BOX: {
                    nb::list sides = nb::cast<nb::list>(node_payload["sides"]);
                    node.primitive.box.sizes = glm::vec4(
                        nb::cast<float>(sides[0]),
                        nb::cast<float>(sides[1]),
                        nb::cast<float>(sides[2]),
                        0.0f);

                    float scale = fmaxf(node.primitive.box.sizes.x, node.primitive.box.sizes.z) * 2.0f;
                    nb::list bevel = nb::cast<nb::list>(node_payload["bevel"]);
                    glm::vec4 corner_rounding(
                        nb::cast<float>(bevel[0]),
                        nb::cast<float>(bevel[1]),
                        nb::cast<float>(bevel[2]),
                        nb::cast<float>(bevel[3]));

                    uint32_t corner_data = 0;
                    corner_data |= (uint32_t)(corner_rounding.x * 255.0f * 2.0f / scale) << 0;
                    corner_data |= (uint32_t)(corner_rounding.y * 255.0f * 2.0f / scale) << 8;
                    corner_data |= (uint32_t)(corner_rounding.z * 255.0f * 2.0f / scale) << 16;
                    corner_data |= (uint32_t)(corner_rounding.w * 255.0f * 2.0f / scale) << 24;
                    memcpy(&node.primitive.box.sizes.w, &corner_data, sizeof(uint32_t));
                    break;
                }
                case PRIMITIVE_CYLINDER: {
                    node.primitive.cylinder.height = nb::cast<float>(node_payload["height"]);
                    node.primitive.cylinder.radius = nb::cast<float>(node_payload["radius"]);
                    break;
                }
                case PRIMITIVE_CONE: {
                    node.primitive.cone.height = nb::cast<float>(node_payload["height"]);
                    node.primitive.cone.radius = nb::cast<float>(node_payload["radius"]);
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported primitive type");
            }
            continue;
        }

        if (type != "binaryOperator") {
            throw std::runtime_error("Invalid scene payload node type: " + type);
        }

        std::string blend_mode = nb::cast<std::string>(node_payload["blendMode"]);
        uint32_t op = 0;
        if (blend_mode == "union") {
            op = OP_UNION;
        } else if (blend_mode == "sub") {
            op = OP_SUB;
        } else if (blend_mode == "inter") {
            op = OP_INTER;
        } else {
            throw std::runtime_error("Unknown blend mode: " + blend_mode);
        }

        int left_idx = int(nodes.size());
        nodes.push_back({});
        int right_idx = int(nodes.size());
        nodes.push_back({});

        stack.push_back({nb::cast<nb::dict>(node_payload["leftChild"]), mat * node_mat, left_idx, true});
        stack.push_back({nb::cast<nb::dict>(node_payload["rightChild"]), mat * node_mat, right_idx, op != OP_SUB});

        CSGNode& node = nodes[node_idx];
        node.left = left_idx;
        node.right = right_idx;
        node.sign = entry.sign;
        node.binary_op = BinaryOp(nb::cast<float>(node_payload["blendRadius"]), op == OP_UNION, op);
    }
}

nb::dict pack_scene_payload_impl(const nb::dict& scene_payload) {
    std::vector<CSGNode> nodes;
    glm::vec3 aabb_min;
    glm::vec3 aabb_max;
    load_payload_dict(scene_payload, nodes, aabb_min, aabb_max);
    return pack_scene_nodes_impl(nodes, 0, aabb_min, aabb_max);
}

nb::dict pack_scene_packed_impl(
    const std::vector<GPUNode>& gpu_nodes,
    const std::vector<Primitive>& primitives,
    const std::vector<BinaryOp>& binary_ops,
    const std::vector<uint16_t>& parents,
    const std::vector<uint16_t>& active_nodes,
    const glm::vec3& aabb_min,
    const glm::vec3& aabb_max)
{

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

struct PackedSceneCacheEntry {
    std::vector<GPUNode> gpu_nodes;
    std::vector<BinaryOp> binary_ops;
    std::vector<uint16_t> parents;
    std::vector<uint16_t> active_nodes;
    std::vector<int> primitive_node_indices;
};

std::vector<int> primitive_node_indices_in_gpu_order(
    int root_idx,
    const std::vector<CSGNode>& csg_nodes)
{
    std::vector<int> stack = {root_idx};
    std::vector<int> preorder;
    preorder.reserve(csg_nodes.size());
    while (!stack.empty()) {
        int current_idx = stack.back();
        stack.pop_back();

        preorder.push_back(current_idx);
        if (csg_nodes[current_idx].type == NODETYPE_BINARY) {
            stack.push_back(csg_nodes[current_idx].left);
            stack.push_back(csg_nodes[current_idx].right);
        }
    }

    std::vector<int> primitive_indices;
    primitive_indices.reserve(csg_nodes.size());
    for (int i = int(preorder.size()) - 1; i >= 0; --i) {
        int current_idx = preorder[i];
        if (csg_nodes[current_idx].type == NODETYPE_PRIMITIVE) {
            primitive_indices.push_back(current_idx);
        }
    }
    return primitive_indices;
}

PackedSceneCacheEntry& get_packed_scene_cache_entry(
    const std::string& topology_key,
    const std::vector<CSGNode>& nodes)
{
    static std::unordered_map<std::string, PackedSceneCacheEntry> cache;
    auto [it, inserted] = cache.try_emplace(topology_key);
    if (!inserted) {
        return it->second;
    }

    PackedSceneCacheEntry& entry = it->second;
    std::vector<Primitive> primitives;
    ConvertToGPUTree(0, nodes, entry.gpu_nodes, primitives, entry.binary_ops, entry.parents, entry.active_nodes);
    entry.primitive_node_indices = primitive_node_indices_in_gpu_order(0, nodes);
    return entry;
}

std::vector<Primitive> extract_cached_primitives(
    const std::vector<CSGNode>& nodes,
    const PackedSceneCacheEntry& entry)
{
    std::vector<Primitive> primitives;
    primitives.reserve(entry.primitive_node_indices.size());
    for (int node_idx : entry.primitive_node_indices) {
        if (node_idx < 0 || node_idx >= int(nodes.size())) {
            throw std::runtime_error("cached topology no longer matches scene nodes");
        }
        const CSGNode& node = nodes[node_idx];
        if (node.type != NODETYPE_PRIMITIVE) {
            throw std::runtime_error("cached primitive index does not point to a primitive node");
        }
        primitives.push_back(node.primitive);
    }
    return primitives;
}

nb::dict pack_scene_file_impl(const std::string& scene_path) {
    std::vector<CSGNode> nodes;
    glm::vec3 aabb_min;
    glm::vec3 aabb_max;
    load_json(scene_path.c_str(), nodes, aabb_min, aabb_max);
    return pack_scene_nodes_impl(nodes, 0, aabb_min, aabb_max);
}

nb::dict pack_scene_json_impl(const std::string& scene_json) {
    std::vector<CSGNode> nodes;
    glm::vec3 aabb_min;
    glm::vec3 aabb_max;
    load_json_string(scene_json.c_str(), nodes, aabb_min, aabb_max);
    return pack_scene_nodes_impl(nodes, 0, aabb_min, aabb_max);
}

nb::dict pack_scene_json_cached_impl(
    const std::string& scene_json,
    const std::string& topology_key,
    bool include_static)
{
    if (topology_key.empty()) {
        return pack_scene_json_impl(scene_json);
    }

    std::vector<CSGNode> nodes;
    glm::vec3 aabb_min;
    glm::vec3 aabb_max;
    load_json_string(scene_json.c_str(), nodes, aabb_min, aabb_max);

    PackedSceneCacheEntry& entry = get_packed_scene_cache_entry(topology_key, nodes);
    std::vector<Primitive> primitives = extract_cached_primitives(nodes, entry);
    if (include_static) {
        return pack_scene_packed_impl(
            entry.gpu_nodes,
            primitives,
            entry.binary_ops,
            entry.parents,
            entry.active_nodes,
            aabb_min,
            aabb_max);
    }

    nb::dict result;
    result["node_count"] = int(entry.gpu_nodes.size());
    result["primitive_count"] = int(primitives.size());
    result["binary_op_count"] = int(entry.binary_ops.size());
    result["aabb_min"] = std::vector<float>{ aabb_min.x, aabb_min.y, aabb_min.z };
    result["aabb_max"] = std::vector<float>{ aabb_max.x, aabb_max.y, aabb_max.z };
    const char* prim_ptr = primitives.empty() ? nullptr : reinterpret_cast<const char*>(primitives.data());
    result["primitives"] = nb::bytes(prim_ptr, primitives.size() * sizeof(Primitive));
    return result;
}

nb::dict pack_scene_payload_cached_impl(
    const nb::dict& scene_payload,
    const std::string& topology_key,
    bool include_static)
{
    if (topology_key.empty()) {
        return pack_scene_payload_impl(scene_payload);
    }

    std::vector<CSGNode> nodes;
    glm::vec3 aabb_min;
    glm::vec3 aabb_max;
    load_payload_dict(scene_payload, nodes, aabb_min, aabb_max);

    PackedSceneCacheEntry& entry = get_packed_scene_cache_entry(topology_key, nodes);
    std::vector<Primitive> primitives = extract_cached_primitives(nodes, entry);
    if (include_static) {
        return pack_scene_packed_impl(
            entry.gpu_nodes,
            primitives,
            entry.binary_ops,
            entry.parents,
            entry.active_nodes,
            aabb_min,
            aabb_max);
    }

    nb::dict result;
    result["node_count"] = int(entry.gpu_nodes.size());
    result["primitive_count"] = int(primitives.size());
    result["binary_op_count"] = int(entry.binary_ops.size());
    result["aabb_min"] = std::vector<float>{ aabb_min.x, aabb_min.y, aabb_min.z };
    result["aabb_max"] = std::vector<float>{ aabb_max.x, aabb_max.y, aabb_max.z };
    const char* prim_ptr = primitives.empty() ? nullptr : reinterpret_cast<const char*>(primitives.data());
    result["primitives"] = nb::bytes(prim_ptr, primitives.size() * sizeof(Primitive));
    return result;
}

struct DemoAnimCacheEntry {
    std::vector<GPUNode> gpu_nodes;
    std::vector<Primitive> primitives;
    std::vector<BinaryOp> binary_ops;
    std::vector<uint16_t> parents;
    std::vector<uint16_t> active_nodes;
    glm::vec3 aabb_min;
    glm::vec3 aabb_max;
    int root_idx = 0;
    int sphere_idx = -1;
    int sphere_primitive_idx = -1;
};

DemoAnimCacheEntry& get_demo_anim_entry(const std::string& scene_path) {
    static std::unordered_map<std::string, DemoAnimCacheEntry> cache;
    auto [it, inserted] = cache.try_emplace(scene_path);
    if (!inserted) {
        return it->second;
    }

    DemoAnimCacheEntry& entry = it->second;
    std::vector<CSGNode> nodes;
    load_json(scene_path.c_str(), nodes, entry.aabb_min, entry.aabb_max);

    CSGNode sphere_node{};
    sphere_node.primitive.sphere = {.radius = glm::vec4(0.2f)};
    sphere_node.primitive.m_row0 = glm::vec4(1, 0, 0, 0);
    sphere_node.primitive.m_row1 = glm::vec4(0, 1, 0, 0);
    sphere_node.primitive.m_row2 = glm::vec4(0, 0, 1, 0);
    sphere_node.primitive.extrude_rounding = glm::vec2(0.0f);
    sphere_node.primitive.pad0 = 1.0f;
    sphere_node.primitive.type = PRIMITIVE_SPHERE;
    sphere_node.primitive.color = 0xaaaaff;
    sphere_node.type = NODETYPE_PRIMITIVE;
    sphere_node.left = -1;
    sphere_node.right = -1;
    sphere_node.sign = true;
    entry.sphere_idx = int(nodes.size());
    nodes.push_back(sphere_node);

    CSGNode union_node{};
    union_node.binary_op = BinaryOp(1e-1f, true, OP_UNION);
    union_node.type = NODETYPE_BINARY;
    union_node.sign = true;
    union_node.left = 0;
    union_node.right = entry.sphere_idx;
    entry.root_idx = int(nodes.size());
    nodes.push_back(union_node);

    std::vector<int> cpu_to_gpu(nodes.size());
    std::vector<int> stack = {entry.root_idx};
    std::vector<int> preorder;
    while (!stack.empty()) {
        int current_idx = stack.back();
        stack.pop_back();
        preorder.push_back(current_idx);
        if (nodes[current_idx].type == NODETYPE_BINARY) {
            stack.push_back(nodes[current_idx].left);
            stack.push_back(nodes[current_idx].right);
        }
    }

    entry.active_nodes.resize(nodes.size());
    for (int i = int(nodes.size()) - 1; i >= 0; --i) {
        int current_idx = preorder[i];
        const CSGNode& node = nodes[current_idx];
        switch (node.type) {
            case NODETYPE_BINARY: {
                int gpu_left = cpu_to_gpu[node.left];
                int gpu_right = cpu_to_gpu[node.right];
                entry.binary_ops.push_back(node.binary_op);
                GPUNode gpu_node = {
                    .type = node.type,
                    .idx_in_type = int(entry.binary_ops.size()) - 1,
                };
                entry.gpu_nodes.push_back(gpu_node);
                entry.parents[gpu_left] = uint16_t(entry.gpu_nodes.size() - 1);
                entry.parents[gpu_right] = uint16_t(entry.gpu_nodes.size() - 1);
                entry.parents.push_back(0xffff);
                break;
            }
            case NODETYPE_PRIMITIVE: {
                entry.primitives.push_back(node.primitive);
                GPUNode gpu_node = {
                    .type = node.type,
                    .idx_in_type = int(entry.primitives.size()) - 1,
                };
                entry.gpu_nodes.push_back(gpu_node);
                entry.parents.push_back(0xffff);
                if (current_idx == entry.sphere_idx) {
                    entry.sphere_primitive_idx = gpu_node.idx_in_type;
                }
                break;
            }
            default:
                abort();
        }
        uint32_t gpu_idx = uint32_t(entry.gpu_nodes.size() - 1);
        cpu_to_gpu[current_idx] = int(gpu_idx);
        entry.active_nodes[gpu_idx] = uint16_t(gpu_idx);
        if (!node.sign) {
            entry.active_nodes[gpu_idx] |= 1u << 15;
        }
    }
    return entry;
}

nb::dict pack_scene_demo_anim_impl(const std::string& scene_path, float anim_time) {
    DemoAnimCacheEntry& entry = get_demo_anim_entry(scene_path);

    const float radius = 0.2f;
    glm::vec3 center = {
        cosf(anim_time),
        cosf(anim_time * 0.3f),
        sinf(anim_time),
    };
    center *= sinf(anim_time * 0.56f + 123.4f);
    glm::vec3 scale = entry.aabb_max - entry.aabb_min - glm::vec3(2.0f * radius);
    center = entry.aabb_min + glm::vec3(radius) + (center * 0.5f + glm::vec3(0.5f)) * scale;

    Primitive& sphere = entry.primitives[entry.sphere_primitive_idx];
    sphere.m_row0[3] = -center.x;
    sphere.m_row1[3] = -center.y;
    sphere.m_row2[3] = -center.z;

    return pack_scene_packed_impl(
        entry.gpu_nodes,
        entry.primitives,
        entry.binary_ops,
        entry.parents,
        entry.active_nodes,
        entry.aabb_min,
        entry.aabb_max);
}

}

NB_MODULE(lipschitz_pruning_native, m) {
    m.doc() = "Lipschitz pruning Vulkan renderer bindings";

    m.attr("SHADING_MODE_SHADED") = SHADING_MODE_SHADED;
    m.attr("SHADING_MODE_HEATMAP") = SHADING_MODE_HEATMAP;
    m.attr("SHADING_MODE_NORMALS") = SHADING_MODE_NORMALS;
    m.attr("SHADING_MODE_AO") = SHADING_MODE_BEAUTY;
    m.def("pack_scene_file", &pack_scene_file_impl, nb::arg("scene_path"));
    m.def("pack_scene_payload", &pack_scene_payload_impl, nb::arg("scene_payload"));
    m.def("pack_scene_json", &pack_scene_json_impl, nb::arg("scene_json"));
    m.def(
        "pack_scene_json_cached",
        &pack_scene_json_cached_impl,
        nb::arg("scene_json"),
        nb::arg("topology_key"),
        nb::arg("include_static") = true);
    m.def(
        "pack_scene_payload_cached",
        &pack_scene_payload_cached_impl,
        nb::arg("scene_payload"),
        nb::arg("topology_key"),
        nb::arg("include_static") = true);
    m.def("pack_scene_demo_anim", &pack_scene_demo_anim_impl, nb::arg("scene_path"), nb::arg("anim_time"));
    m.def("extract_matcap_exr", &extract_matcap_exr_impl, nb::arg("path"));

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
        .def("load_scene_payload", &OffscreenRenderer::load_scene_payload, nb::arg("scene_payload"))
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
