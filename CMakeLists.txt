cmake_minimum_required(VERSION 3.20)
project(pulsim VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(PULSIM_BUILD_TESTS "Build tests" ON)
option(PULSIM_BUILD_CLI "Build CLI application" ON)
option(PULSIM_BUILD_PYTHON "Build Python bindings" OFF)
option(PULSIM_BUILD_GRPC "Build gRPC API server" OFF)
option(PULSIM_USE_SUNDIALS "Use SUNDIALS for advanced ODE/DAE solvers" OFF)
option(PULSIM_USE_SUITESPARSE "Use SuiteSparse KLU for faster sparse LU" OFF)

# Note: We use FetchContent_Populate for Eigen (header-only) to avoid
# processing its CMakeLists.txt which includes many test targets.
# Other dependencies have their test options disabled via cache variables.

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
    option(PULSIM_SANITIZERS "Enable sanitizers" OFF)
    if(PULSIM_SANITIZERS)
        add_compile_options(-fsanitize=address,undefined)
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

# Dependencies via FetchContent
include(FetchContent)

# Note: We don't set BUILD_TESTING OFF globally as it would disable our own tests.
# Eigen is handled via FetchContent_Populate (not MakeAvailable) to skip its CMakeLists.txt.

# Eigen - header-only, don't process its CMakeLists.txt to avoid test targets
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
endif()
# Create Eigen3::Eigen target manually (header-only)
if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${eigen_SOURCE_DIR}"
    )
endif()

# nlohmann/json - disable tests
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(json)

# Catch2 for testing
if(PULSIM_BUILD_TESTS)
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

# SUNDIALS - Advanced ODE/DAE solvers (IDA, CVODE, ARKODE)
if(PULSIM_USE_SUNDIALS)
    # First try to find system-installed SUNDIALS
    find_package(SUNDIALS CONFIG QUIET)
    if(NOT SUNDIALS_FOUND)
        message(STATUS "SUNDIALS not found on system, fetching from source...")
        FetchContent_Declare(
            sundials
            GIT_REPOSITORY https://github.com/LLNL/sundials.git
            GIT_TAG v6.7.0
            GIT_SHALLOW TRUE
        )
        # Configure SUNDIALS build options
        set(SUNDIALS_BUILD_SHARED_LIBS OFF CACHE BOOL "Build static SUNDIALS" FORCE)
        set(BUILD_ARKODE ON CACHE BOOL "Enable ARKODE" FORCE)
        set(BUILD_CVODE OFF CACHE BOOL "Disable CVODE" FORCE)
        set(BUILD_CVODES OFF CACHE BOOL "Disable CVODES" FORCE)
        set(BUILD_IDA ON CACHE BOOL "Enable IDA" FORCE)
        set(BUILD_IDAS OFF CACHE BOOL "Disable IDAS" FORCE)
        set(BUILD_KINSOL OFF CACHE BOOL "Disable KINSOL" FORCE)
        set(SUNDIALS_BUILD_WITH_MONITORING OFF CACHE BOOL "Disable monitoring" FORCE)
        set(EXAMPLES_ENABLE_C OFF CACHE BOOL "Disable C examples" FORCE)
        set(EXAMPLES_ENABLE_CXX OFF CACHE BOOL "Disable C++ examples" FORCE)
        set(EXAMPLES_INSTALL OFF CACHE BOOL "Don't install examples" FORCE)
        FetchContent_MakeAvailable(sundials)
    endif()
    add_compile_definitions(PULSIM_HAS_SUNDIALS)
    message(STATUS "SUNDIALS support enabled")
endif()

# SuiteSparse KLU - Fast sparse LU for circuit matrices
if(PULSIM_USE_SUITESPARSE)
    # Try to find system-installed SuiteSparse
    find_package(SuiteSparse CONFIG QUIET)
    if(SuiteSparse_FOUND)
        message(STATUS "Found system SuiteSparse")
        add_compile_definitions(PULSIM_HAS_KLU)
    else()
        # Try pkg-config
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(KLU QUIET klu)
            if(KLU_FOUND)
                message(STATUS "Found KLU via pkg-config")
                add_compile_definitions(PULSIM_HAS_KLU)
            else()
                message(WARNING "SuiteSparse/KLU not found. Install with: brew install suite-sparse (macOS) or apt install libsuitesparse-dev (Linux)")
            endif()
        else()
            message(WARNING "SuiteSparse/KLU not found and pkg-config not available")
        endif()
    endif()
endif()

# Core library
add_subdirectory(core)

# gRPC API server (must come before CLI to provide grpc_proto target)
if(PULSIM_BUILD_GRPC)
    add_subdirectory(api-grpc)
endif()

# CLI application
if(PULSIM_BUILD_CLI)
    add_subdirectory(cli)
endif()

# If tests are enabled, ensure the CLI executable is built before running integration
# tests that invoke it. `pulsim_tests` lives in the `core` subdirectory.
if(PULSIM_BUILD_TESTS AND PULSIM_BUILD_CLI)
    if(TARGET pulsim AND TARGET pulsim_tests)
        add_dependencies(pulsim_tests pulsim)
    endif()
endif()

# Python bindings
if(PULSIM_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# Custom targets for running examples (only when CLI is built)
if(PULSIM_BUILD_CLI)
    add_custom_target(run_buck
        COMMAND $<TARGET_FILE:pulsim> -v run ${CMAKE_SOURCE_DIR}/examples/buck_converter.json -o buck_result.csv
        DEPENDS pulsim
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running buck converter simulation..."
    )

    add_custom_target(run_rc
        COMMAND $<TARGET_FILE:pulsim> -v run ${CMAKE_SOURCE_DIR}/examples/rc_circuit.json -o rc_result.csv
        DEPENDS pulsim
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running RC circuit simulation..."
    )

    add_custom_target(run_rlc
        COMMAND $<TARGET_FILE:pulsim> -v run ${CMAKE_SOURCE_DIR}/examples/rlc_circuit.json -o rlc_result.csv
        DEPENDS pulsim
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running RLC circuit simulation..."
    )
endif()
