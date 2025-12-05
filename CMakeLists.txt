cmake_minimum_required(VERSION 3.20)
project(spicelab VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(SPICELAB_BUILD_TESTS "Build tests" ON)
option(SPICELAB_BUILD_CLI "Build CLI application" ON)
option(SPICELAB_BUILD_PYTHON "Build Python bindings" OFF)
option(SPICELAB_BUILD_GRPC "Build gRPC API server" OFF)

# If project-level tests are disabled, also disable CTest/BUILD_TESTING
# so FetchContent dependencies (Eigen, abseil, etc.) won't add their
# test targets (e.g. "check", "meta") which can conflict with other
# dependencies when pulled into the superproject.
if(NOT SPICELAB_BUILD_TESTS)
    set(BUILD_TESTING OFF CACHE BOOL "Disable tests for dependencies" FORCE)
endif()
# When project tests are disabled, disable BUILD_TESTING and common
# test-related options for third-party FetchContent deps to avoid
# global target name collisions (e.g. Eigen's test targets like
# `check`/`meta` conflict with abseil's targets pulled by gRPC).
if(NOT SPICELAB_BUILD_TESTS)
    message(STATUS "SPICELAB_BUILD_TESTS=OFF -> disabling dependency tests (BUILD_TESTING/EIGEN_* )")
    # Prevent fetched projects from enabling their own testing targets
    set(BUILD_TESTING OFF CACHE BOOL "Disable building tests in dependencies" FORCE)
    # Eigen-specific guards (try both variable names used across versions)
    set(EIGEN_BUILD_TESTS OFF CACHE BOOL "Disable Eigen tests in dependencies" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen testing in dependencies" FORCE)
endif()

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror=return-type)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0)
    else()
        add_compile_options(-O3 -DNDEBUG)
    endif()
elseif(MSVC)
    add_compile_options(/W4)
endif()

# Sanitizers for Debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT MSVC)
    option(SPICELAB_SANITIZERS "Enable sanitizers" OFF)
    if(SPICELAB_SANITIZERS)
        add_compile_options(-fsanitize=address,undefined)
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

# Dependencies via FetchContent
include(FetchContent)

# Eigen
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(eigen)

# nlohmann/json
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(json)

# Catch2 for testing
if(SPICELAB_BUILD_TESTS)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(CTest)
    include(Catch)
endif()

# Core library
add_subdirectory(core)

# CLI application
if(SPICELAB_BUILD_CLI)
    add_subdirectory(cli)
endif()

# Python bindings
if(SPICELAB_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# gRPC API server
if(SPICELAB_BUILD_GRPC)
    add_subdirectory(api-grpc)
endif()

# Custom targets for running examples
add_custom_target(run_buck
    COMMAND $<TARGET_FILE:spicelab> -v run ${CMAKE_SOURCE_DIR}/examples/buck_converter.json -o buck_result.csv
    DEPENDS spicelab
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running buck converter simulation..."
)

add_custom_target(run_rc
    COMMAND $<TARGET_FILE:spicelab> -v run ${CMAKE_SOURCE_DIR}/examples/rc_circuit.json -o rc_result.csv
    DEPENDS spicelab
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running RC circuit simulation..."
)

add_custom_target(run_rlc
    COMMAND $<TARGET_FILE:spicelab> -v run ${CMAKE_SOURCE_DIR}/examples/rlc_circuit.json -o rlc_result.csv
    DEPENDS spicelab
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running RLC circuit simulation..."
)
