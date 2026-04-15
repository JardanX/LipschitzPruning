# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "D:/Projects/GitHub/LipschitzPruning/build-vs-imnodes/_deps/imnodes_rokups-src")
  file(MAKE_DIRECTORY "D:/Projects/GitHub/LipschitzPruning/build-vs-imnodes/_deps/imnodes_rokups-src")
endif()
file(MAKE_DIRECTORY
  "D:/Projects/GitHub/LipschitzPruning/build-vs-imnodes/_deps/imnodes_rokups-build"
  "D:/Projects/GitHub/LipschitzPruning/build-vs-imnodes/_deps/imnodes_rokups-subbuild/imnodes_rokups-populate-prefix"
  "D:/Projects/GitHub/LipschitzPruning/build-vs-imnodes/_deps/imnodes_rokups-subbuild/imnodes_rokups-populate-prefix/tmp"
  "D:/Projects/GitHub/LipschitzPruning/build-vs-imnodes/_deps/imnodes_rokups-subbuild/imnodes_rokups-populate-prefix/src/imnodes_rokups-populate-stamp"
  "D:/Projects/GitHub/LipschitzPruning/build-vs-imnodes/_deps/imnodes_rokups-subbuild/imnodes_rokups-populate-prefix/src"
  "D:/Projects/GitHub/LipschitzPruning/build-vs-imnodes/_deps/imnodes_rokups-subbuild/imnodes_rokups-populate-prefix/src/imnodes_rokups-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "D:/Projects/GitHub/LipschitzPruning/build-vs-imnodes/_deps/imnodes_rokups-subbuild/imnodes_rokups-populate-prefix/src/imnodes_rokups-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "D:/Projects/GitHub/LipschitzPruning/build-vs-imnodes/_deps/imnodes_rokups-subbuild/imnodes_rokups-populate-prefix/src/imnodes_rokups-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
