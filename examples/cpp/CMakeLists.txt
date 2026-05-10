# Standalone CMake project for Pulsim C++ examples.
#
# Each example is a single-file binary linking against the header-only
# pulsim::core target. Configure & build:
#
#   cmake -S examples/cpp -B build/examples
#   cmake --build build/examples
#
# Or, if you've already configured the main project at `build/`, the
# examples are also added as targets via the top-level CMakeLists when
# PULSIM_BUILD_EXAMPLES=ON.

cmake_minimum_required(VERSION 3.20)
project(pulsim_examples_cpp LANGUAGES CXX)

# ---- Picking up the main project's pulsim::core ----
# Two paths are supported:
#   1. The user already configured Pulsim at the repo root (`build/`):
#      we re-enter that build via `add_subdirectory` and depend on
#      `pulsim::core` directly.
#   2. Standalone: build pulsim::core first, then this directory.

set(PULSIM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
if(NOT TARGET pulsim::core)
    message(STATUS "pulsim::core not found in this configuration. "
                   "Add this directory from the top-level Pulsim CMakeLists, "
                   "or build the parent project first.")
    add_subdirectory(${PULSIM_ROOT} ${CMAKE_BINARY_DIR}/_pulsim_core)
endif()

set(_examples
    01_saturable_inductor
    02_dc_motor_step
    03_pmsm_foc
    04_three_phase_pll
    05_robustness_profile
)

foreach(name IN LISTS _examples)
    add_executable(pulsim_example_${name} ${name}.cpp)
    target_link_libraries(pulsim_example_${name} PRIVATE pulsim::core)
    target_compile_features(pulsim_example_${name} PRIVATE cxx_std_23)
endforeach()
