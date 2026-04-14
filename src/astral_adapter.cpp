#include "astral_adapter.h"

#include "UI/NodeGraph.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace {

constexpr int ASTRAL_TYPE_SPHERE = 0;
constexpr int ASTRAL_TYPE_BOX = 1;
constexpr int ASTRAL_TYPE_TORUS = 2;
constexpr int ASTRAL_TYPE_CYLINDER = 3;
constexpr int ASTRAL_TYPE_CAPSULE = 4;

glm::mat3 rot_x(float a) {
    float c = cosf(a);
    float s = sinf(a);
    return glm::mat3(1, 0, 0, 0, c, -s, 0, s, c);
}

glm::mat3 rot_y(float a) {
    float c = cosf(a);
    float s = sinf(a);
    return glm::mat3(c, 0, s, 0, 1, 0, -s, 0, c);
}

glm::mat3 rot_z(float a) {
    float c = cosf(a);
    float s = sinf(a);
    return glm::mat3(c, -s, 0, s, c, 0, 0, 0, 1);
}

glm::mat3 euler_xyz(const glm::vec3& r) {
    return rot_z(r.z) * rot_y(r.y) * rot_x(r.x);
}

glm::mat3 abs_mat3(const glm::mat3& m) {
    return glm::mat3(
        fabsf(m[0][0]), fabsf(m[0][1]), fabsf(m[0][2]),
        fabsf(m[1][0]), fabsf(m[1][1]), fabsf(m[1][2]),
        fabsf(m[2][0]), fabsf(m[2][1]), fabsf(m[2][2]));
}

glm::vec3 primitive_local_extents(const NodeGraph::PrimData& prim) {
    int type = (int)prim.color_type.a;
    if (type == ASTRAL_TYPE_SPHERE) {
        return glm::vec3(prim.params.x);
    }
    if (type == ASTRAL_TYPE_BOX) {
        return glm::vec3(prim.params.x, prim.params.y, prim.params.z);
    }
    if (type == ASTRAL_TYPE_TORUS) {
        return glm::vec3(prim.params.x + prim.params.y, prim.params.x + prim.params.y, prim.params.y);
    }
    if (type == ASTRAL_TYPE_CYLINDER) {
        return glm::vec3(prim.params.x, prim.params.x, prim.params.y);
    }
    if (type == ASTRAL_TYPE_CAPSULE) {
        return glm::vec3(prim.params.x, prim.params.x, prim.params.x + prim.params.y);
    }
    return glm::vec3(prim.posAndBound.w);
}

void prim_aabb(const NodeGraph::PrimData& prim, glm::vec3& out_min, glm::vec3& out_max) {
    glm::vec3 center = glm::vec3(prim.posAndBound);
    glm::vec3 scale = glm::vec3(prim.scale);
    glm::vec3 local_ext = primitive_local_extents(prim) * scale;
    glm::mat3 rot_abs = abs_mat3(euler_xyz(glm::vec3(prim.rotation)));
    glm::vec3 ext = rot_abs * local_ext;
    out_min = center - ext;
    out_max = center + ext;
}

bool find_ref_aabb(
    int ref,
    const std::vector<NodeGraph::PrimData>& prims,
    const std::vector<NodeGraph::CSGNodeData>& csg,
    std::vector<glm::vec4>& min_cache,
    std::vector<glm::vec4>& max_cache,
    glm::vec3& out_min,
    glm::vec3& out_max) {
    if (ref < 0) {
        int idx = -(ref + 1);
        if (idx < 0 || idx >= (int)prims.size()) {
            return false;
        }
        prim_aabb(prims[idx], out_min, out_max);
        return true;
    }
    if (ref < 0 || ref >= (int)csg.size()) {
        return false;
    }
    if (min_cache[ref].w >= 0.0f && max_cache[ref].w >= 0.0f) {
        out_min = glm::vec3(min_cache[ref]);
        out_max = glm::vec3(max_cache[ref]);
        return true;
    }

    glm::vec3 min_a;
    glm::vec3 max_a;
    glm::vec3 min_b;
    glm::vec3 max_b;
    const auto& node = csg[ref];
    if (!find_ref_aabb(node.childA, prims, csg, min_cache, max_cache, min_a, max_a)) {
        return false;
    }
    if (!find_ref_aabb(node.childB, prims, csg, min_cache, max_cache, min_b, max_b)) {
        return false;
    }

    if (node.op == 0 || node.op == 3) {
        out_min = glm::min(min_a, min_b);
        out_max = glm::max(max_a, max_b);
    } else {
        out_min = min_a;
        out_max = max_a;
    }

    min_cache[ref] = glm::vec4(out_min, 1.0f);
    max_cache[ref] = glm::vec4(out_max, 1.0f);
    return true;
}

uint32_t pack_color(const glm::vec3& color) {
    uint32_t r = std::min(255u, (uint32_t)(glm::clamp(color.r, 0.0f, 1.0f) * 255.99f));
    uint32_t g = std::min(255u, (uint32_t)(glm::clamp(color.g, 0.0f, 1.0f) * 255.99f));
    uint32_t b = std::min(255u, (uint32_t)(glm::clamp(color.b, 0.0f, 1.0f) * 255.99f));
    return r | (g << 8) | (b << 16);
}

bool is_supported_primitive_type(int type) {
    return type >= ASTRAL_TYPE_SPHERE && type <= ASTRAL_TYPE_CAPSULE;
}

Primitive build_primitive(const NodeGraph::PrimData& prim) {
    Primitive out{};
    int type = (int)prim.color_type.a;
    glm::vec3 pos = glm::vec3(prim.posAndBound);
    glm::vec3 rot = glm::vec3(prim.rotation);
    glm::vec3 scale = glm::max(glm::vec3(prim.scale), glm::vec3(0.001f));

    glm::mat3 inv_scale(1.0f);
    inv_scale[0][0] = 1.0f / scale.x;
    inv_scale[1][1] = 1.0f / scale.y;
    inv_scale[2][2] = 1.0f / scale.z;

    glm::mat3 linear = glm::transpose(euler_xyz(rot)) * inv_scale;

    if (type == ASTRAL_TYPE_CYLINDER) {
        glm::mat3 basis_swap(1.0f);
        basis_swap[0] = glm::vec3(1.0f, 0.0f, 0.0f);
        basis_swap[1] = glm::vec3(0.0f, 0.0f, 1.0f);
        basis_swap[2] = glm::vec3(0.0f, 1.0f, 0.0f);
        linear = basis_swap * linear;
    }

    glm::vec3 translation = -(linear * pos);
    glm::mat4 world_to_prim(1.0f);
    world_to_prim[0] = glm::vec4(linear[0], 0.0f);
    world_to_prim[1] = glm::vec4(linear[1], 0.0f);
    world_to_prim[2] = glm::vec4(linear[2], 0.0f);
    world_to_prim[3] = glm::vec4(translation, 1.0f);

    out.m_row0 = glm::vec4(world_to_prim[0][0], world_to_prim[1][0], world_to_prim[2][0], world_to_prim[3][0]);
    out.m_row1 = glm::vec4(world_to_prim[0][1], world_to_prim[1][1], world_to_prim[2][1], world_to_prim[3][1]);
    out.m_row2 = glm::vec4(world_to_prim[0][2], world_to_prim[1][2], world_to_prim[2][2], world_to_prim[3][2]);
    out.color = pack_color(glm::vec3(prim.color_type));
    out.extrude_rounding = glm::vec2(0.0f);
    out.bevel = 0.0f;
    out.pad0 = std::min(scale.x, std::min(scale.y, scale.z));

    switch (type) {
        case ASTRAL_TYPE_SPHERE:
            out.type = PRIMITIVE_SPHERE;
            out.sphere.radius = glm::vec4(prim.params.x, 0.0f, 0.0f, 0.0f);
            break;
        case ASTRAL_TYPE_BOX:
            out.type = PRIMITIVE_BOX;
            out.box.sizes = glm::vec4(prim.params.x * 2.0f, prim.params.y * 2.0f, prim.params.z * 2.0f, 0.0f);
            break;
        case ASTRAL_TYPE_TORUS:
            out.type = PRIMITIVE_TORUS;
            out.torus.major_radius = prim.params.x;
            out.torus.minor_radius = prim.params.y;
            break;
        case ASTRAL_TYPE_CYLINDER:
            out.type = PRIMITIVE_CYLINDER;
            out.cylinder.radius = prim.params.x;
            out.cylinder.height = prim.params.y * 2.0f;
            break;
        case ASTRAL_TYPE_CAPSULE:
            out.type = PRIMITIVE_CAPSULE;
            out.capsule.radius = prim.params.x;
            out.capsule.half_height = prim.params.y;
            break;
        default:
            out.type = PRIMITIVE_SPHERE;
            out.sphere.radius = glm::vec4(0.0f);
            break;
    }

    return out;
}

int append_ref_scene(
    int ref,
    bool sign,
    const std::vector<NodeGraph::PrimData>& prims,
    const std::vector<NodeGraph::CSGNodeData>& csg,
    std::vector<CSGNode>& nodes) {
    if (ref < 0) {
        int idx = -(ref + 1);
        if (idx < 0 || idx >= (int)prims.size()) {
            return -1;
        }
        if (!is_supported_primitive_type((int)prims[idx].color_type.a)) {
            return -1;
        }
        CSGNode node{};
        node.type = NODETYPE_PRIMITIVE;
        node.left = -1;
        node.right = -1;
        node.sign = sign;
        node.primitive = build_primitive(prims[idx]);
        nodes.push_back(node);
        return (int)nodes.size() - 1;
    }

    if (ref >= (int)csg.size()) {
        return -1;
    }

    const auto& src = csg[ref];
    int left = append_ref_scene(src.childA, true, prims, csg, nodes);
    int right = append_ref_scene(src.childB, src.op != 1, prims, csg, nodes);
    if (left < 0 || right < 0) {
        return -1;
    }

    uint32_t op = OP_UNION;
    float blend = src.blend;
    switch (src.op) {
        case 0:
            op = OP_UNION;
            break;
        case 1:
            op = OP_SUB;
            blend = 0.0f;
            break;
        case 2:
            op = OP_INTER;
            blend = 0.0f;
            break;
        case 3:
            op = OP_UNION;
            break;
        default:
            return -1;
    }

    CSGNode node{};
    node.type = NODETYPE_BINARY;
    node.left = left;
    node.right = right;
    node.sign = sign;
    node.binary_op = BinaryOp(blend, op == OP_UNION, op);
    nodes.push_back(node);
    return (int)nodes.size() - 1;
}

int root_ref(const NodeGraph::RootEntry& root) {
    return root._p1 != 0 ? -(root.id + 1) : root.id;
}

}

bool build_scene_from_astral_graph(
    const NodeGraph& graph,
    std::vector<CSGNode>& nodes,
    int& root_idx,
    glm::vec3& aabb_min,
    glm::vec3& aabb_max) {
    const auto& prims = graph.getPrims();
    const auto& csg = graph.getCSGNodes();
    const auto& roots = graph.getRoots();
    if (roots.empty()) {
        nodes.clear();
        root_idx = -1;
        aabb_min = glm::vec3(-1.0f);
        aabb_max = glm::vec3(1.0f);
        return false;
    }

    nodes.clear();
    std::vector<int> root_nodes;
    root_nodes.reserve(roots.size());

    glm::vec3 scene_min(FLT_MAX);
    glm::vec3 scene_max(-FLT_MAX);
    std::vector<glm::vec4> min_cache(csg.size(), glm::vec4(0.0f, 0.0f, 0.0f, -1.0f));
    std::vector<glm::vec4> max_cache(csg.size(), glm::vec4(0.0f, 0.0f, 0.0f, -1.0f));

    for (const auto& root : roots) {
        int ref = root_ref(root);
        glm::vec3 root_min;
        glm::vec3 root_max;
        if (find_ref_aabb(ref, prims, csg, min_cache, max_cache, root_min, root_max)) {
            scene_min = glm::min(scene_min, root_min);
            scene_max = glm::max(scene_max, root_max);
        }

        int node_idx = append_ref_scene(ref, true, prims, csg, nodes);
        if (node_idx >= 0) {
            root_nodes.push_back(node_idx);
        }
    }

    if (root_nodes.empty()) {
        nodes.clear();
        root_idx = -1;
        aabb_min = glm::vec3(-1.0f);
        aabb_max = glm::vec3(1.0f);
        return false;
    }

    root_idx = root_nodes[0];
    for (size_t i = 1; i < root_nodes.size(); ++i) {
        CSGNode node{};
        node.type = NODETYPE_BINARY;
        node.left = root_idx;
        node.right = root_nodes[i];
        node.sign = true;
        node.binary_op = BinaryOp(0.0f, true, OP_UNION);
        nodes.push_back(node);
        root_idx = (int)nodes.size() - 1;
    }

    if (scene_min.x == FLT_MAX || scene_max.x == -FLT_MAX) {
        scene_min = glm::vec3(-1.0f);
        scene_max = glm::vec3(1.0f);
    }

    glm::vec3 pad = glm::max((scene_max - scene_min) * 0.05f, glm::vec3(0.05f));
    aabb_min = scene_min - pad;
    aabb_max = scene_max + pad;
    return true;
}
