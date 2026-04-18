cmake_minimum_required(VERSION 3.18)
project(LipschitzPruning LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BLENDER_ADDON_DIR "${CMAKE_SOURCE_DIR}/blender_addon/mathops_v2")

set(MATHOPS_HAS_CORE OFF)
if (
    EXISTS "${CMAKE_SOURCE_DIR}/src/context.cpp"
    AND EXISTS "${CMAKE_SOURCE_DIR}/include/context.h"
    AND EXISTS "${CMAKE_SOURCE_DIR}/shaders/simple.frag.glsl"
    AND EXISTS "${CMAKE_SOURCE_DIR}/ext/vk-bootstrap/CMakeLists.txt"
)
    set(MATHOPS_HAS_CORE ON)
    message(STATUS "MathOPS native renderer sources detected")
else()
    message(STATUS "MathOPS native renderer sources not found; building standalone addon extras only")
endif()

set(MATHOPS_BLENDER_LIB_ROOT "${CMAKE_SOURCE_DIR}/../blender-sdf/lib/windows_x64" CACHE PATH "Blender dependency bundle used for native extras")
if (WIN32 AND EXISTS "${MATHOPS_BLENDER_LIB_ROOT}")
    list(PREPEND CMAKE_PREFIX_PATH
        "${MATHOPS_BLENDER_LIB_ROOT}/imath"
        "${MATHOPS_BLENDER_LIB_ROOT}/openjph"
        "${MATHOPS_BLENDER_LIB_ROOT}/openexr"
    )
endif()

if (MATHOPS_HAS_CORE)
    add_subdirectory(ext/vk-bootstrap)
    find_package(Vulkan REQUIRED)
endif()

set(_BLENDER_PYTHON_DEFAULT "")
foreach(_blender_python_candidate IN ITEMS
    "D:/Blender/Blender/versions/blender-5.1.0-windows-x64/5.1/python"
    "D:/Blender/Blender/versions/blender-5.0.1-windows-x64/5.0/python"
)
    if (EXISTS "${_blender_python_candidate}")
        set(_BLENDER_PYTHON_DEFAULT "${_blender_python_candidate}")
        break()
    endif()
endforeach()
set(BLENDER_PYTHON_ROOT "${_BLENDER_PYTHON_DEFAULT}" CACHE PATH "Preferred Blender Python root for addon builds")
if (EXISTS "${BLENDER_PYTHON_ROOT}")
    set(Python_ROOT_DIR "${BLENDER_PYTHON_ROOT}")
    message(STATUS "Using Blender Python from ${BLENDER_PYTHON_ROOT}")
endif()
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(Imath CONFIG QUIET)
find_package(openjph CONFIG QUIET)
find_package(OpenEXR CONFIG QUIET)

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

if (MATHOPS_HAS_CORE)
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
endif()

set(MATHOPS_WITH_OPENEXR OFF)
if (TARGET OpenEXR::OpenEXR AND TARGET Imath::Imath AND TARGET openjph)
    set(MATHOPS_WITH_OPENEXR ON)
    message(STATUS "Matcap EXR support enabled")
else()
    message(STATUS "Matcap EXR support disabled")
endif()

if (MATHOPS_WITH_OPENEXR AND EXISTS "${CMAKE_SOURCE_DIR}/src/matcap_exr_module.cpp")
    nanobind_add_module(mathops_matcap_native src/matcap_exr_module.cpp)
    target_link_libraries(mathops_matcap_native PRIVATE OpenEXR::OpenEXR)
endif()

if (TARGET lipschitz_pruning_native)
    set_target_properties(lipschitz_pruning_native PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${BLENDER_ADDON_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${BLENDER_ADDON_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${BLENDER_ADDON_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${BLENDER_ADDON_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${BLENDER_ADDON_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${BLENDER_ADDON_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${BLENDER_ADDON_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${BLENDER_ADDON_DIR}"
    )
endif()

if (TARGET mathops_matcap_native)
    set_target_properties(mathops_matcap_native PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${BLENDER_ADDON_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${BLENDER_ADDON_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${BLENDER_ADDON_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${BLENDER_ADDON_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${BLENDER_ADDON_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${BLENDER_ADDON_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${BLENDER_ADDON_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${BLENDER_ADDON_DIR}"
    )
endif()

set(ADDON_COPY_COMMANDS
    COMMAND ${CMAKE_COMMAND} -E make_directory "${BLENDER_ADDON_DIR}"
)

if (TARGET lipschitz_pruning_native)
    list(APPEND ADDON_COPY_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:lipschitz_pruning_native>" "${BLENDER_ADDON_DIR}/$<TARGET_FILE_NAME:lipschitz_pruning_native>"
    )
endif()

if (MATHOPS_HAS_CORE)
    foreach(bin_file IN LISTS SHADER_BINS)
        list(APPEND ADDON_COPY_COMMANDS
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_BINARY_DIR}/${bin_file}" "${BLENDER_ADDON_DIR}/${bin_file}"
        )
    endforeach()
endif()

if (TARGET mathops_matcap_native)
    list(APPEND ADDON_COPY_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:mathops_matcap_native>" "${BLENDER_ADDON_DIR}/$<TARGET_FILE_NAME:mathops_matcap_native>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:Imath::Imath>" "${BLENDER_ADDON_DIR}/$<TARGET_FILE_NAME:Imath::Imath>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:openjph>" "${BLENDER_ADDON_DIR}/$<TARGET_FILE_NAME:openjph>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:OpenEXR::Iex>" "${BLENDER_ADDON_DIR}/$<TARGET_FILE_NAME:OpenEXR::Iex>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:OpenEXR::IlmThread>" "${BLENDER_ADDON_DIR}/$<TARGET_FILE_NAME:OpenEXR::IlmThread>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:OpenEXR::OpenEXRCore>" "${BLENDER_ADDON_DIR}/$<TARGET_FILE_NAME:OpenEXR::OpenEXRCore>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:OpenEXR::OpenEXR>" "${BLENDER_ADDON_DIR}/$<TARGET_FILE_NAME:OpenEXR::OpenEXR>"
    )
endif()

set(STAGE_DEPENDS)
if (TARGET lipschitz_pruning_native)
    list(APPEND STAGE_DEPENDS lipschitz_pruning_native)
endif()
if (TARGET lipschitz_shaders)
    list(APPEND STAGE_DEPENDS lipschitz_shaders)
endif()
if (TARGET mathops_matcap_native)
    list(APPEND STAGE_DEPENDS mathops_matcap_native)
endif()

add_custom_target(stage_blender_addon ALL
    ${ADDON_COPY_COMMANDS}
    DEPENDS ${STAGE_DEPENDS}
)
