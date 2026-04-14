#include "context.h"
//#include "example_config.h"
#include <string>
#include "CLI/CLI.hpp"
#include <algorithm>
#include <random>
#include <queue>
#include <array>
#include <system_error>
#include "imgui.h"
#include "backends/imgui_impl_vulkan.h"
#include "backends/imgui_impl_glfw.h"
#include "astral_adapter.h"
#include "scene.h"
#include "UI/NodeGraph.h"
#include <filesystem>

int create_scene(std::vector<CSGNode>& csg_tree, const std::string& input_path, glm::vec3& aabb_min, glm::vec3& aabb_max) {
    csg_tree.clear();
    load_json(input_path.c_str(), csg_tree, aabb_min, aabb_max);
    int root_idx = 0;
    return root_idx;
}

float cam_distance = 3.f;
bool g_viewport_hovered = true;

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
    if (g_viewport_hovered) {
        cam_distance -= (float)yoffset * 0.1f;
        cam_distance = std::max(0.1f, cam_distance);
    }
}

bool show_imgui = true;

std::string find_astral_font_path() {
    const std::array<std::filesystem::path, 5> candidates = {
        std::filesystem::path("C:/Users/bysta/Downloads/ra_mono/Ra-Mono.otf"),
        std::filesystem::path("../Ra-Mono.otf"),
        std::filesystem::path("../../blender-sdf/release/datafiles/fonts/Ra-Mono.otf"),
        std::filesystem::path("../../blender-sdf-portable/5.1/datafiles/fonts/Ra-Mono.otf"),
        std::filesystem::path("D:/Projects/GitHub/blender-sdf/release/datafiles/fonts/Ra-Mono.otf"),
    };

    for (const auto& candidate : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec)) {
            return std::filesystem::absolute(candidate, ec).string();
        }
    }

    return {};
}

void apply_astral_font() {
    ImGuiIO& io = ImGui::GetIO();
    ImFontConfig cfg{};
    cfg.OversampleH = 3;
    cfg.OversampleV = 1;

    std::string font_path = find_astral_font_path();
    if (!font_path.empty()) {
        if (ImFont* mono_font = io.Fonts->AddFontFromFileTTF(font_path.c_str(), 15.0f, &cfg)) {
            io.FontDefault = mono_font;
        }
    }
}

void apply_astral_style() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;
    style.FrameRounding = 0.0f;
    style.GrabRounding = 0.0f;
    style.TabRounding = 0.0f;
    style.ScrollbarRounding = 0.0f;
    style.PopupRounding = 0.0f;
    style.ChildRounding = 0.0f;
    style.WindowPadding = ImVec2(8, 8);
    style.FramePadding = ImVec2(6, 3);
    style.ItemSpacing = ImVec2(6, 4);
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.WindowMenuButtonPosition = ImGuiDir_None;
    style.TabBarBorderSize = 1.0f;

    ImVec4* c = style.Colors;
    c[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.06f, 0.08f, 1.0f);
    c[ImGuiCol_ChildBg] = ImVec4(0.05f, 0.05f, 0.07f, 1.0f);
    c[ImGuiCol_Border] = ImVec4(0.14f, 0.14f, 0.18f, 1.0f);
    c[ImGuiCol_FrameBg] = ImVec4(0.10f, 0.10f, 0.13f, 1.0f);
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.13f, 0.13f, 0.17f, 1.0f);
    c[ImGuiCol_FrameBgActive] = ImVec4(0.16f, 0.16f, 0.20f, 1.0f);
    c[ImGuiCol_TitleBg] = ImVec4(0.05f, 0.05f, 0.07f, 1.0f);
    c[ImGuiCol_TitleBgActive] = ImVec4(0.08f, 0.08f, 0.11f, 1.0f);
    c[ImGuiCol_Tab] = ImVec4(0.08f, 0.08f, 0.11f, 1.0f);
    c[ImGuiCol_TabHovered] = ImVec4(0.15f, 0.15f, 0.19f, 1.0f);
    c[ImGuiCol_TabActive] = ImVec4(0.11f, 0.11f, 0.15f, 1.0f);
    c[ImGuiCol_Button] = ImVec4(0.12f, 0.12f, 0.16f, 1.0f);
    c[ImGuiCol_ButtonHovered] = ImVec4(0.17f, 0.17f, 0.22f, 1.0f);
    c[ImGuiCol_ButtonActive] = ImVec4(0.21f, 0.21f, 0.26f, 1.0f);
    c[ImGuiCol_Header] = ImVec4(0.10f, 0.10f, 0.14f, 1.0f);
    c[ImGuiCol_HeaderHovered] = ImVec4(0.15f, 0.15f, 0.20f, 1.0f);
    c[ImGuiCol_HeaderActive] = ImVec4(0.18f, 0.18f, 0.23f, 1.0f);
    c[ImGuiCol_Separator] = ImVec4(0.16f, 0.16f, 0.20f, 1.0f);
    c[ImGuiCol_TextSelectedBg] = ImVec4(0.35f, 0.35f, 0.55f, 0.40f);
    c[ImGuiCol_ScrollbarBg] = ImVec4(0.10f, 0.10f, 0.12f, 1.0f);
    c[ImGuiCol_ScrollbarGrab] = ImVec4(0.30f, 0.30f, 0.33f, 1.0f);
    c[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.38f, 0.38f, 0.42f, 1.0f);
    c[ImGuiCol_CheckMark] = ImVec4(0.55f, 0.75f, 0.95f, 1.0f);
    c[ImGuiCol_SliderGrab] = ImVec4(0.45f, 0.65f, 0.85f, 1.0f);
    c[ImGuiCol_SliderGrabActive] = ImVec4(0.55f, 0.75f, 0.95f, 1.0f);
}


void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
    if (key == GLFW_KEY_I && action == GLFW_PRESS) {
        show_imgui = !show_imgui;
    }
}

int main(int argc, char** argv) {
    enum class SceneSource {
        Demo,
        Graph,
    };

    glm::vec3 cam_target = glm::vec3(0);
    float cam_yaw = 0.f;
    float cam_pitch = M_PI / 2;

    bool culling_enabled = true;
    int num_samples = 1;
    std::string shading_mode_str = "shaded";

    std::string anim_path = "";
    //std::string input_file = "../build/catalog/guy.json";
    std::string input_file = "../scenes/trees.json";
    spv_dir = ".";
    CLI::App cli{ "Lipschitz Pruning demo" };
    cli.add_option("-i,--input", input_file, "Input");
    cli.add_option("-s,--shaders", spv_dir, "SPIR-V path");
    float min_coord = -1;
    float max_coord = 1;
    cli.add_option("--min", min_coord, "AABB min");
    cli.add_option("--max", max_coord, "AABB min");
    cli.add_option("--cam_yaw", cam_yaw, "Camera yaw");
    cli.add_option("--cam_pitch", cam_pitch, "Camera pitch");
    cli.add_option("--cam_dist", cam_distance, "Camera pitch");
    cli.add_option("--culling", culling_enabled, "Enable culling");
    cli.add_option("--samples", num_samples, "Samples per pixel");
    cli.add_option("--shading", shading_mode_str, "Shading mode");
    cli.add_option("--max-active", MAX_ACTIVE_COUNT, "Max active count");
    cli.add_option("--max-tmp", MAX_TMP_COUNT, "Max tmp count");
    cli.add_option("--anim", anim_path, "Animation directory");
    cli.add_option("--target_x", cam_target.x, "Target X");
    cli.add_option("--target_y", cam_target.y, "Target Y");
    cli.add_option("--target_z", cam_target.z, "Target Z");
    CLI11_PARSE(cli, argc, argv);

    //std::string input_file = "C:\\Users\\schtr\\Documents\\projects\\SDFCulling\\build\\catalog\\guy.json";
    //spv_dir = "C:\\Users\\schtr\\Documents\\projects\\SDFCulling\\build";

    constexpr int NUM_SCENES = 3;
    const char* preset_scenes[NUM_SCENES+1][2] = {
        { "Trees", "trees.json" },
        { "Monument", "monument.json" },
        { "Molecule", "molecule.json" },
        { "Custom", nullptr }
    };
    int preset_scene_idx = 0;

    int num_nodes = 0;
    std::vector<CSGNode> csg_tree;

    NodeGraph node_graph;
    bool show_graph_window = true;
    SceneSource scene_source = SceneSource::Demo;
    bool graph_scene_valid = false;

    Context ctx;
    ctx.initialize(true, 8);
    apply_astral_font();
    apply_astral_style();
    node_graph.init();

    Timings timing = { };
    double anim_start_time = 0.0;
    float anim_speed = 4.0f;
    bool anim_play = false;
    int root_idx = -1;
    bool viewport_navigating = false;
    bool scene_dirty = true;
    bool request_redraw = true;

    auto upload_scene = [&](const glm::vec3& aabb_min, const glm::vec3& aabb_max) {
        num_nodes = (int)csg_tree.size();
        ctx.render_data.aabb_min = aabb_min;
        ctx.render_data.aabb_max = aabb_max;
        ctx.alloc_input_buffers(num_nodes);
        ctx.upload(csg_tree, root_idx);
        anim_play = false;
        ctx.render_data.culling_enabled = true;
        scene_dirty = true;
        request_redraw = true;
    };

    root_idx = create_scene(csg_tree, input_file, ctx.render_data.aabb_min, ctx.render_data.aabb_max);
    upload_scene(ctx.render_data.aabb_min, ctx.render_data.aabb_max);

    auto load_demo_scene = [&]() {
        root_idx = create_scene(csg_tree, input_file, ctx.render_data.aabb_min, ctx.render_data.aabb_max);
        upload_scene(ctx.render_data.aabb_min, ctx.render_data.aabb_max);
        scene_source = SceneSource::Demo;
    };

    auto sync_scene_from_graph = [&]() {
        std::vector<CSGNode> graph_nodes;
        int graph_root = -1;
        glm::vec3 graph_min;
        glm::vec3 graph_max;
        graph_scene_valid = build_scene_from_astral_graph(node_graph, graph_nodes, graph_root, graph_min, graph_max);
        node_graph.clearDirty();
        if (!graph_scene_valid) {
            return false;
        }
        csg_tree = std::move(graph_nodes);
        root_idx = graph_root;
        upload_scene(graph_min, graph_max);
        scene_source = SceneSource::Graph;
        return true;
    };

    ctx.render_data.push_constants.alpha = 1;
    ctx.render_data.culling_enabled = culling_enabled;
    ctx.render_data.num_samples = num_samples;

    if (shading_mode_str == "normals") {
        ctx.render_data.shading_mode = SHADING_MODE_NORMALS;
    }
    else if (shading_mode_str == "shaded") {
        ctx.render_data.shading_mode = SHADING_MODE_SHADED;
    }
    else if (shading_mode_str == "beauty") {
        ctx.render_data.shading_mode = SHADING_MODE_BEAUTY;
    }
    else {
        fprintf(stderr, "Unknown shading mode: %s\n", shading_mode_str.c_str());
        abort();
    }

    glfwSetKeyCallback(ctx.init.window, key_callback);
    glfwSetScrollCallback(ctx.init.window, scroll_callback);

    double last_x, last_y;
    glfwGetCursorPos(ctx.init.window, &last_x, &last_y);

    while (!glfwWindowShouldClose(ctx.init.window)/* && g_frame < 10000*/) {
        if (anim_play || viewport_navigating || request_redraw) {
            glfwPollEvents();
        } else {
            glfwWaitEventsTimeout(0.1);
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        request_redraw = false;

        bool scene_changed = false;
        bool viewport_changed = false;
        float prev_cam_distance = cam_distance;
        float prev_cam_yaw = cam_yaw;
        float prev_cam_pitch = cam_pitch;
        glm::vec3 prev_cam_target = cam_target;

        g_viewport_hovered = !show_imgui;
        ImGuiViewport* main_vp = ImGui::GetMainViewport();
        ImGuiIO& io = ImGui::GetIO();
        ImVec2 work_pos = main_vp->WorkPos;
        ImVec2 work_size = main_vp->WorkSize;
        float right_w = floorf(work_size.x * 0.22f);
        float nodegraph_h = floorf(work_size.y * 0.42f);
        float viewport_h = work_size.y - nodegraph_h;
        float scene_h = floorf(work_size.y * 0.34f);
        float templates_h = floorf(work_size.y * 0.22f);
        float renderer_h = work_size.y - scene_h - templates_h;

        if (show_imgui) {
            ImGui::SetNextWindowPos(work_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(work_size.x - right_w, viewport_h), ImGuiCond_Always);

            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowTitleAlign, ImVec2(0.5f, 0.5f));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            if (ImGui::Begin("Viewport##vp", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
                ImVec2 vp_min = ImGui::GetCursorScreenPos();
                ImVec2 vp_size = ImGui::GetContentRegionAvail();
                ImVec2 vp_max(vp_min.x + vp_size.x, vp_min.y + vp_size.y);
                bool recompute_pruning = scene_dirty || anim_play;
                ImVec2 fb_scale = io.DisplayFramebufferScale;
                ctx.render_data.viewport_offset = glm::ivec2(
                    std::max(0, (int)floorf(vp_min.x * fb_scale.x)),
                    std::max(0, (int)floorf(vp_min.y * fb_scale.y)));
                ctx.render_data.viewport_size = glm::ivec2(
                    std::max(1, (int)floorf(vp_size.x * fb_scale.x)),
                    std::max(1, (int)floorf(vp_size.y * fb_scale.y)));
                g_viewport_hovered = ImGui::IsMouseHoveringRect(vp_min, vp_max, false);

                ImDrawList* draw_list = ImGui::GetWindowDrawList();
                draw_list->AddRect(vp_min, vp_max, IM_COL32(72, 72, 88, 255));
                draw_list->AddText(ImVec2(vp_min.x + 14.0f, vp_min.y + 10.0f), IM_COL32(180, 184, 194, 255), "Lipschitz Viewport");
                draw_list->AddText(ImVec2(vp_min.x + 14.0f, vp_min.y + 28.0f), IM_COL32(120, 124, 136, 255), "Orbit LMB | Pan RMB | Zoom Wheel");
                draw_list->AddText(
                    ImVec2(vp_min.x + 14.0f, vp_min.y + 46.0f),
                    recompute_pruning ? IM_COL32(255, 196, 96, 255) : IM_COL32(124, 164, 120, 255),
                    recompute_pruning ? "Pruning: recomputing" : "Pruning: cached");
            }
            ImGui::End();
            ImGui::PopStyleVar(3);
        } else {
            ctx.render_data.viewport_offset = glm::ivec2(0);
            ctx.render_data.viewport_size = glm::ivec2(0);
        }

        ctx.render_data.show_imgui = show_imgui;

        double cur_x, cur_y;
        glfwGetCursorPos(ctx.init.window, &cur_x, &cur_y);
        double delta_x = cur_x - last_x;
        double delta_y = cur_y - last_y;
        last_x = cur_x;
        last_y = cur_y;

        bool camera_active = !show_imgui || g_viewport_hovered;
        if (camera_active && glfwGetMouseButton(ctx.init.window, GLFW_MOUSE_BUTTON_LEFT)) {
            cam_yaw -= 0.01f * (float)delta_x;
            cam_pitch -= 0.01f * (float)delta_y;
            cam_yaw = fmodf(cam_yaw, 2.f * M_PI);
            cam_pitch = fminf(fmaxf(cam_pitch, 1e-3f), M_PI - 1e-3f);
        }

        glm::vec3 v = glm::vec3{
            cam_distance * sinf(cam_yaw) * sinf(cam_pitch),
            cam_distance * cosf(cam_pitch),
            cam_distance * cosf(cam_yaw) * sinf(cam_pitch),
        };
        glm::vec3 cam_position = cam_target + v;

        if (camera_active && glfwGetMouseButton(ctx.init.window, GLFW_MOUSE_BUTTON_RIGHT)) {
            glm::vec3 vel = glm::vec3(0);
            glm::vec3 right = glm::normalize(glm::cross(v, glm::vec3(0, 1, 0)));
            glm::vec3 cam_up = glm::normalize(glm::cross(right, v));
            vel += right * 0.01f * (float)delta_x;
            vel += cam_up * 0.01f * (float)delta_y;
            cam_target += vel;
            cam_position += vel;
        }

        auto apply_graph_preset = [&](auto&& fn) {
            fn();
            sync_scene_from_graph();
            scene_changed = true;
        };

        if (show_imgui) {
            ImGui::SetNextWindowPos(ImVec2(work_pos.x + work_size.x - right_w, work_pos.y), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(right_w, work_size.y), ImGuiCond_Always);
        }
        if (show_imgui && ImGui::Begin("Controls##sidebar", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove)) {
            ImGui::Text("Source: %s", scene_source == SceneSource::Graph ? "Astral Graph" : "Demo Scene");
            if (scene_source == SceneSource::Graph && !graph_scene_valid) {
                ImGui::TextColored(ImVec4(0.95f, 0.65f, 0.45f, 1.0f), "Graph has no compiled scene yet.");
            }

            if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
                viewport_changed |= ImGui::Checkbox("Show Node Graph", &show_graph_window);
                if (ImGui::Button("Use Current Graph")) {
                    if (sync_scene_from_graph()) {
                        scene_changed = true;
                    }
                }
                if (ImGui::Button("Use Demo Preset")) {
                    load_demo_scene();
                    scene_changed = true;
                }

                if (ImGui::BeginCombo("Demo Preset", preset_scenes[preset_scene_idx][0])) {
                    for (int i = 0; i < NUM_SCENES; i++) {
                        bool is_selected = (i == preset_scene_idx);
                        if (ImGui::Selectable(preset_scenes[i][0], is_selected)) {
                            preset_scene_idx = i;
                            input_file = "../scenes/" + std::string(preset_scenes[i][1]);
                            load_demo_scene();
                            scene_changed = true;
                        }
                    }
                    ImGui::EndCombo();
                }
                if (ImGui::Button("Reload Demo Scene")) {
                    load_demo_scene();
                    scene_changed = true;
                }

                if (ImGui::SliderFloat3("AABB min", &ctx.render_data.aabb_min[0], -3, 0)) {
                    scene_changed = true;
                }
                if (ImGui::SliderFloat3("AABB max", &ctx.render_data.aabb_max[0], 0, 3)) {
                    scene_changed = true;
                }
            }

            if (ImGui::CollapsingHeader("Templates", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImGui::Button("128 Rand")) { apply_graph_preset([&] { node_graph.createStressRandomScene(128); }); }
                ImGui::SameLine();
                if (ImGui::Button("512 Rand")) { apply_graph_preset([&] { node_graph.createStressRandomScene(512); }); }
                ImGui::SameLine();
                if (ImGui::Button("2048 Rand")) { apply_graph_preset([&] { node_graph.createStressRandomScene(2048); }); }

                if (ImGui::Button("32 Multi")) { apply_graph_preset([&] { node_graph.createStressMultiCSGScene(32); }); }
                ImGui::SameLine();
                if (ImGui::Button("128 Multi")) { apply_graph_preset([&] { node_graph.createStressMultiCSGScene(128); }); }
                ImGui::SameLine();
                if (ImGui::Button("384 Multi")) { apply_graph_preset([&] { node_graph.createStressMultiCSGScene(384); }); }

                if (ImGui::Button("Pump Housing")) {
                    apply_graph_preset([&] { node_graph.createComplexPumpHousingScene(); });
                }
                if (ImGui::Button("Clear Graph")) {
                    node_graph.clearScene();
                    node_graph.clearDirty();
                    graph_scene_valid = false;
                    scene_changed = true;
                }
            }

            if (ImGui::CollapsingHeader("Render", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("Nodes: %d", num_nodes);
                ImGui::Text("Graph: %d prims / %d ops / %d roots", (int)node_graph.getPrims().size(), (int)node_graph.getCSGNodes().size(), (int)node_graph.getRoots().size());

                if (scene_source == SceneSource::Graph) {
                    ImGui::TextDisabled("Animation disabled for graph scenes.");
                    ImGui::BeginDisabled();
                }
                if (anim_play) {
                    if (ImGui::Button("Stop Anim")) {
                        anim_play = false;
                        load_demo_scene();
                        scene_changed = true;
                    }
                } else if (ImGui::Button("Play Anim")) {
                    anim_play = true;
                    anim_start_time = glfwGetTime();
                    {
                        CSGNode node{};
                        node.primitive.sphere = {.radius = glm::vec4(0.2f)};
                        node.primitive.m_row0 = glm::vec4(1, 0, 0, 0);
                        node.primitive.m_row1 = glm::vec4(0, 1, 0, 0);
                        node.primitive.m_row2 = glm::vec4(0, 0, 1, 0);
                        node.primitive.type = PRIMITIVE_SPHERE;
                        node.primitive.color = 0xaaaaff;
                        node.type = NODETYPE_PRIMITIVE;
                        node.left = -1;
                        node.right = -1;
                        node.sign = true;
                        csg_tree.push_back(node);
                    }
                    {
                        CSGNode node{};
                        node.binary_op = BinaryOp(1e-1f, true, OP_UNION);
                        node.type = NODETYPE_BINARY;
                        node.sign = true;
                        node.left = root_idx;
                        node.right = (int)csg_tree.size() - 1;
                        csg_tree.push_back(node);
                    }
                    num_nodes = (int)csg_tree.size();
                    root_idx = (int)csg_tree.size() - 1;
                    ctx.alloc_input_buffers(num_nodes);
                    ctx.upload(csg_tree, root_idx);
                    scene_changed = true;
                }
                if (scene_source == SceneSource::Graph) {
                    ImGui::EndDisabled();
                }
                if (ImGui::SliderFloat("Anim speed", &anim_speed, 0, 4.f)) {
                    viewport_changed = true;
                }

                if (num_nodes > 500) {
                    ImGui::BeginDisabled();
                }
                if (ImGui::Checkbox("Enable pruning", &ctx.render_data.culling_enabled)) {
                    viewport_changed = true;
                }
                if (num_nodes > 500) {
                    ImGui::SameLine();
                    ImGui::Text("(forced > 500)");
                    ImGui::EndDisabled();
                }
                ImGui::TextDisabled("Recompute pruning: %s", (scene_dirty || scene_changed || anim_play) ? "On (scene change)" : "Off (camera only)");
                ImGui::TextColored(
                    (scene_dirty || scene_changed || anim_play) ? ImVec4(1.0f, 0.77f, 0.38f, 1.0f) : ImVec4(0.55f, 0.78f, 0.55f, 1.0f),
                    (scene_dirty || scene_changed || anim_play) ? "Pruning recompute active" : "Pruning cache active");

                if (ImGui::Button("Grid -")) {
                    ctx.render_data.final_grid_lvl = std::max(2, ctx.render_data.final_grid_lvl - 2);
                    scene_changed = true;
                }
                ImGui::SameLine();
                if (ImGui::Button("Grid +")) {
                    ctx.render_data.final_grid_lvl = std::min(8, ctx.render_data.final_grid_lvl + 2);
                    scene_changed = true;
                }
                ImGui::SameLine();
                ImGui::Text("1 << %d", ctx.render_data.final_grid_lvl);

                if (ImGui::Combo("Shading", &ctx.render_data.shading_mode, "Shaded\0Heatmap\0Normals\0AO\0")) {
                    ctx.init.disp.destroyPipeline(ctx.render_data.graphics_pipeline, nullptr);
                    create_graphics_pipeline(ctx.init, ctx.render_data);
                    viewport_changed = true;
                }
                if (ImGui::Checkbox("Show grid", &ctx.render_data.show_grid)) {
                    viewport_changed = true;
                }
                if (ctx.render_data.shading_mode == SHADING_MODE_HEATMAP) {
                    if (ImGui::SliderInt("Colormap max", &ctx.render_data.colormap_max, 1, 64)) {
                        viewport_changed = true;
                    }
                }
                if (ImGui::SliderInt("Samples", &ctx.render_data.num_samples, 1, 64)) {
                    viewport_changed = true;
                }
                if (ImGui::SliderFloat("Gamma", &ctx.render_data.gamma, 1, 4)) {
                    viewport_changed = true;
                }
            }

            if (ImGui::CollapsingHeader("Stats", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("%.0f FPS", ImGui::GetIO().Framerate);
                ImGui::Text("Render %.3f ms", ctx.render_data.render_elapsed_ms);
                ImGui::Text("Culling %.3f ms", ctx.render_data.culling_elapsed_ms);
                ImGui::Text("Tracing %.3f ms", ctx.render_data.tracing_elapsed_ms);
                ImGui::Text("Mem %.3f GB", timing.pruning_mem_usage_gb);
                ImGui::Text("Active %.2f", (float)ctx.render_data.max_active_count / (float)MAX_ACTIVE_COUNT);

                VkPhysicalDeviceMemoryBudgetPropertiesEXT budget;
                budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
                budget.pNext = nullptr;
                VkPhysicalDeviceMemoryProperties2 props;
                props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
                props.pNext = &budget;
                vkGetPhysicalDeviceMemoryProperties2(ctx.init.device.physical_device, &props);
                for (int i = 0; i < props.memoryProperties.memoryHeapCount; i++) {
                    if ((props.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) {
                        continue;
                    }
                    ImGui::Text("Heap %i %.2f / %.2f GB", i, (double)budget.heapUsage[i] / (1024. * 1024. * 1024.), (double)budget.heapBudget[i] / (1024. * 1024. * 1024.));
                }

#if 1
                if (ImGui::TreeNode("Shader stats")) {
                    if (ImGui::TreeNode("Pruning")) {
                        char pipeline_stats[4096];
                        get_pipeline_stats(ctx.init, ctx.render_data.culling_pipeline.pipe, 0, pipeline_stats, 4096);
                        ImGui::TextWrapped("%s", pipeline_stats);
                        ImGui::TreePop();
                    }
                    if (ImGui::TreeNode("Fragment")) {
                        char pipeline_stats[4096];
                        get_pipeline_stats(ctx.init, ctx.render_data.graphics_pipeline, 1, pipeline_stats, 4096);
                        ImGui::TextWrapped("%s", pipeline_stats);
                        ImGui::TreePop();
                    }
                    ImGui::TreePop();
                }
#endif
            }
        }
        if (show_imgui) {
            ImGui::End();
        }

        if (show_imgui && show_graph_window) {
            ImGui::SetNextWindowPos(ImVec2(work_pos.x, work_pos.y + viewport_h), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(work_size.x - right_w, nodegraph_h), ImGuiCond_Always);
            node_graph.render("Node Graph##nodes", &show_graph_window);
            node_graph.renderViewportAddMenu();
        }
        if (node_graph.needsUpload()) {
            if (sync_scene_from_graph()) {
                scene_changed = true;
            }
        }

        if (anim_play) {
            auto before = std::chrono::high_resolution_clock::now();
            float anim_time = (float)(glfwGetTime() - anim_start_time);
            anim_time *= anim_speed;
            glm::vec3 center = {cosf(anim_time), cosf(anim_time * 0.3f), sinf(anim_time)};
            center *= sin(anim_time * 0.56f + 123.4f);
            float radius = 0.2f;
            glm::vec3 scale = ctx.render_data.aabb_max - ctx.render_data.aabb_min - 2.f * radius;
            center = ctx.render_data.aabb_min + radius + (center * 0.5f + 0.5f) * scale;
            csg_tree[num_nodes - 2].primitive.m_row0[3] = -center.x;
            csg_tree[num_nodes - 2].primitive.m_row1[3] = -center.y;
            csg_tree[num_nodes - 2].primitive.m_row2[3] = -center.z;
            ctx.upload(csg_tree, root_idx);
            auto after = std::chrono::high_resolution_clock::now();
            (void)std::chrono::duration_cast<std::chrono::microseconds>(after - before).count();
            scene_changed = true;
        }

        bool camera_changed = fabsf(cam_distance - prev_cam_distance) > 1e-6f ||
            fabsf(cam_yaw - prev_cam_yaw) > 1e-6f ||
            fabsf(cam_pitch - prev_cam_pitch) > 1e-6f ||
            glm::length(cam_target - prev_cam_target) > 1e-6f;

        scene_dirty = scene_dirty || scene_changed;
        ctx.render_data.compute_culling = scene_dirty || anim_play;

        timing = ctx.render(cam_position, cam_target);
        if (ctx.render_data.compute_culling && !anim_play) {
            scene_dirty = false;
        }

        viewport_navigating = camera_active &&
            (glfwGetMouseButton(ctx.init.window, GLFW_MOUSE_BUTTON_LEFT) ||
             glfwGetMouseButton(ctx.init.window, GLFW_MOUSE_BUTTON_RIGHT));
        request_redraw = anim_play || viewport_navigating || scene_changed || viewport_changed || camera_changed;
    }
    VK_CHECK(ctx.init.disp.deviceWaitIdle());

    //{
    //    FILE* fp = fopen("timings.csv", "w");
    //    for (int i = 0; i < 10000; i++) {
    //        fprintf(fp, "%f\n", g_timings[i]);
    //    }
    //    fflush(fp);
    //    fclose(fp);
    //}

    return 0;
}
