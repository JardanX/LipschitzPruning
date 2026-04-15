# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "D:/Projects/GitHub/LipschitzPruning/build/_deps/imgui_node_editor-src")
  file(MAKE_DIRECTORY "D:/Projects/GitHub/LipschitzPruning/build/_deps/imgui_node_editor-src")
endif()
file(MAKE_DIRECTORY
  "D:/Projects/GitHub/LipschitzPruning/build/_deps/imgui_node_editor-build"
  "D:/Projects/GitHub/LipschitzPruning/build/_deps/imgui_node_editor-subbuild/imgui_node_editor-populate-prefix"
  "D:/Projects/GitHub/LipschitzPruning/build/_deps/imgui_node_editor-subbuild/imgui_node_editor-populate-prefix/tmp"
  "D:/Projects/GitHub/LipschitzPruning/build/_deps/imgui_node_editor-subbuild/imgui_node_editor-populate-prefix/src/imgui_node_editor-populate-stamp"
  "D:/Projects/GitHub/LipschitzPruning/build/_deps/imgui_node_editor-subbuild/imgui_node_editor-populate-prefix/src"
  "D:/Projects/GitHub/LipschitzPruning/build/_deps/imgui_node_editor-subbuild/imgui_node_editor-populate-prefix/src/imgui_node_editor-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "D:/Projects/GitHub/LipschitzPruning/build/_deps/imgui_node_editor-subbuild/imgui_node_editor-populate-prefix/src/imgui_node_editor-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "D:/Projects/GitHub/LipschitzPruning/build/_deps/imgui_node_editor-subbuild/imgui_node_editor-populate-prefix/src/imgui_node_editor-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
