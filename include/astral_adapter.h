#ifndef LIPSCHITZ_ASTRAL_ADAPTER_H
#define LIPSCHITZ_ASTRAL_ADAPTER_H

#include <glm/glm.hpp>
#include <vector>

#include "scene.h"

class NodeGraph;

bool build_scene_from_astral_graph(
    const NodeGraph& graph,
    std::vector<CSGNode>& nodes,
    int& root_idx,
    glm::vec3& aabb_min,
    glm::vec3& aabb_max);

#endif
