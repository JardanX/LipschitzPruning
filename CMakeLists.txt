cmake_minimum_required(VERSION 3.18)
project(LipschitzPruning LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(ext/vk-bootstrap)

find_package(Vulkan REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

include(FetchContent)

FetchContent_Declare(
    CLI11
    GIT_REPOSITORY "https://github.com/CLIUtils/CLI11"
    GIT_TAG "v2.5.0"
)
FetchContent_MakeAvailable(CLI11)

FetchContent_Declare(
    nanobind
    GIT_REPOSITORY "https://github.com/wjakob/nanobind.git"
    GIT_TAG "v2.12.0"
)
FetchContent_MakeAvailable(nanobind)

find_package(glfw3)
if (NOT ${glfw3_FOUND})
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY "https://github.com/glfw/glfw"
        GIT_TAG 3.4
    )
    FetchContent_MakeAvailable(glfw)
endif()

set(SHADER_SRCS
    simple.vert.glsl
    simple.frag.glsl
    culling.comp.glsl
    plane.vert.glsl
    plane.frag.glsl
    dense_eval.comp.glsl)
set(SHADER_STAGES
    vert
    frag
    comp
    vert
    frag
    comp)
set(SHADER_BINS
    vert.spv
    frag.spv
    culling.comp.spv
    plane.vert.spv
    plane.frag.spv
    dense_eval.comp.spv)

set(SHARED_SRC
    src/utils.cpp
    src/debug_plane.cpp
    src/context.cpp
    src/scene.cpp
    ext/imgui/imgui.cpp
    ext/imgui/imgui_draw.cpp
    ext/imgui/imgui_demo.cpp
    ext/imgui/imgui_tables.cpp
    ext/imgui/imgui_widgets.cpp
    ext/imgui/backends/imgui_impl_vulkan.cpp
    ext/imgui/backends/imgui_impl_glfw.cpp
)

set(HEADERS
    include/constants.h
    include/context.h
    include/debug_plane.h
    include/scene.h
    include/utils.h
)

set(SHADER_SRCS_COMMON
    eval.glsl
    common.glsl
    common_culling.glsl
)

set(SHADER_OUTPUTS)
foreach(src_file bin_file stage IN ZIP_LISTS SHADER_SRCS SHADER_BINS SHADER_STAGES)
    set(shader_output "${CMAKE_BINARY_DIR}/${bin_file}")
    list(APPEND SHADER_OUTPUTS "${shader_output}")
    add_custom_command(
        OUTPUT "${shader_output}"
        COMMAND ${Vulkan_GLSLC_EXECUTABLE}
        ARGS -fshader-stage=${stage} --target-spv=spv1.4 -g ${CMAKE_SOURCE_DIR}/shaders/${src_file} -o ${shader_output}
        MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/shaders/${src_file}
        DEPENDS
            ${CMAKE_SOURCE_DIR}/shaders/eval.glsl
            ${CMAKE_SOURCE_DIR}/shaders/common.glsl
            ${CMAKE_SOURCE_DIR}/shaders/common_culling.glsl
            ${CMAKE_SOURCE_DIR}/include/constants.h
    )
endforeach()

add_custom_target(lipschitz_shaders DEPENDS ${SHADER_OUTPUTS})

add_library(lipschitz_core STATIC ${SHARED_SRC} ${HEADERS})
target_include_directories(lipschitz_core
    PUBLIC
        include/
        ext/imgui
        ext/json/include
        ext/rapidjson/include
        ext/glm
)
target_link_libraries(lipschitz_core PUBLIC glfw vk-bootstrap Vulkan::Vulkan)
add_dependencies(lipschitz_core lipschitz_shaders)
foreach(src_file IN LISTS SHADER_SRCS_COMMON)
    target_sources(lipschitz_core PRIVATE ${CMAKE_SOURCE_DIR}/shaders/${src_file})
endforeach()

add_executable(LipschitzPruning src/main.cpp)
target_link_libraries(LipschitzPruning PRIVATE lipschitz_core CLI11::CLI11)
add_dependencies(LipschitzPruning lipschitz_shaders)

nanobind_add_module(lipschitz_pruning_native src/nanobind_module.cpp)
target_link_libraries(lipschitz_pruning_native PRIVATE lipschitz_core)
add_dependencies(lipschitz_pruning_native lipschitz_shaders)

set(BLENDER_ADDON_DIR "${CMAKE_SOURCE_DIR}/blender_addon/lipschitz_pruning")
set_target_properties(lipschitz_pruning_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${BLENDER_ADDON_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${BLENDER_ADDON_DIR}"
)

set(ADDON_COPY_COMMANDS
    COMMAND ${CMAKE_COMMAND} -E make_directory "${BLENDER_ADDON_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:lipschitz_pruning_native>" "${BLENDER_ADDON_DIR}/$<TARGET_FILE_NAME:lipschitz_pruning_native>"
)
foreach(bin_file IN LISTS SHADER_BINS)
    list(APPEND ADDON_COPY_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_BINARY_DIR}/${bin_file}" "${BLENDER_ADDON_DIR}/${bin_file}"
    )
endforeach()

add_custom_target(stage_blender_addon ALL
    ${ADDON_COPY_COMMANDS}
    DEPENDS lipschitz_pruning_native lipschitz_shaders
)
