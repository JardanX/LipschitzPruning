# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "D:/Projects/GitHub/MathOPS-v2/.cmake-matcap-check/_deps/nanobind-src")
  file(MAKE_DIRECTORY "D:/Projects/GitHub/MathOPS-v2/.cmake-matcap-check/_deps/nanobind-src")
endif()
file(MAKE_DIRECTORY
  "D:/Projects/GitHub/MathOPS-v2/.cmake-matcap-check/_deps/nanobind-build"
  "D:/Projects/GitHub/MathOPS-v2/.cmake-matcap-check/_deps/nanobind-subbuild/nanobind-populate-prefix"
  "D:/Projects/GitHub/MathOPS-v2/.cmake-matcap-check/_deps/nanobind-subbuild/nanobind-populate-prefix/tmp"
  "D:/Projects/GitHub/MathOPS-v2/.cmake-matcap-check/_deps/nanobind-subbuild/nanobind-populate-prefix/src/nanobind-populate-stamp"
  "D:/Projects/GitHub/MathOPS-v2/.cmake-matcap-check/_deps/nanobind-subbuild/nanobind-populate-prefix/src"
  "D:/Projects/GitHub/MathOPS-v2/.cmake-matcap-check/_deps/nanobind-subbuild/nanobind-populate-prefix/src/nanobind-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "D:/Projects/GitHub/MathOPS-v2/.cmake-matcap-check/_deps/nanobind-subbuild/nanobind-populate-prefix/src/nanobind-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "D:/Projects/GitHub/MathOPS-v2/.cmake-matcap-check/_deps/nanobind-subbuild/nanobind-populate-prefix/src/nanobind-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
