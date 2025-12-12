cmake_minimum_required(VERSION 3.20)
project(pulsim VERSION 0.1.0 LANGUAGES CXX)

# =============================================================================
# C++ Standard Configuration
# =============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =============================================================================
# Build Options
# =============================================================================
option(PULSIM_BUILD_TESTS "Build tests" ON)
option(PULSIM_BUILD_PYTHON "Build Python bindings" OFF)
option(PULSIM_USE_SUNDIALS "Use SUNDIALS for advanced ODE/DAE solvers" OFF)
option(PULSIM_USE_SUITESPARSE "Use SuiteSparse KLU for faster sparse LU" OFF)

# =============================================================================
# Performance Optimization Options
# =============================================================================
option(PULSIM_ENABLE_LTO "Enable Link-Time Optimization for Release builds" ON)
option(PULSIM_ENABLE_NATIVE "Enable native architecture optimizations" OFF)
option(PULSIM_ENABLE_PGO_GENERATE "Enable PGO instrumentation (generate profile)" OFF)
option(PULSIM_ENABLE_PGO_USE "Enable PGO optimization (use profile)" OFF)
set(PULSIM_PGO_PROFILE_DIR "${CMAKE_BINARY_DIR}/pgo-profiles" CACHE PATH "Directory for PGO profile data")

# LTO (Link-Time Optimization) - significant performance improvement
if(PULSIM_ENABLE_LTO AND CMAKE_BUILD_TYPE STREQUAL "Release")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_ERROR)
    if(LTO_SUPPORTED)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
        message(STATUS "LTO: Enabled (interprocedural optimization)")
    else()
        message(WARNING "LTO: Not supported - ${LTO_ERROR}")
    endif()
endif()

# PGO (Profile-Guided Optimization) - requires two-pass build
# Pass 1: cmake -DPULSIM_ENABLE_PGO_GENERATE=ON .. && make && ./run_benchmarks
# Pass 2: cmake -DPULSIM_ENABLE_PGO_USE=ON .. && make
if(PULSIM_ENABLE_PGO_GENERATE AND PULSIM_ENABLE_PGO_USE)
    message(FATAL_ERROR "Cannot enable both PGO_GENERATE and PGO_USE simultaneously")
endif()

if(PULSIM_ENABLE_PGO_GENERATE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(-fprofile-generate=${PULSIM_PGO_PROFILE_DIR})
        add_link_options(-fprofile-generate=${PULSIM_PGO_PROFILE_DIR})
        message(STATUS "PGO: Instrumentation enabled (profiles -> ${PULSIM_PGO_PROFILE_DIR})")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_compile_options(-fprofile-generate -fprofile-dir=${PULSIM_PGO_PROFILE_DIR})
        add_link_options(-fprofile-generate)
        message(STATUS "PGO: Instrumentation enabled (GCC, profiles -> ${PULSIM_PGO_PROFILE_DIR})")
    elseif(MSVC)
        add_compile_options(/GL)
        add_link_options(/LTCG /GENPROFILE:PGD=${PULSIM_PGO_PROFILE_DIR}/pulsim.pgd)
        message(STATUS "PGO: Instrumentation enabled (MSVC)")
    endif()
endif()

if(PULSIM_ENABLE_PGO_USE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Merge profiles first: llvm-profdata merge -output=merged.profdata ${PULSIM_PGO_PROFILE_DIR}/*.profraw
        set(PULSIM_PGO_MERGED_PROFILE "${PULSIM_PGO_PROFILE_DIR}/merged.profdata" CACHE FILEPATH "Merged PGO profile")
        if(EXISTS "${PULSIM_PGO_MERGED_PROFILE}")
            add_compile_options(-fprofile-use=${PULSIM_PGO_MERGED_PROFILE})
            add_link_options(-fprofile-use=${PULSIM_PGO_MERGED_PROFILE})
            message(STATUS "PGO: Using profile ${PULSIM_PGO_MERGED_PROFILE}")
        else()
            message(WARNING "PGO: Profile not found at ${PULSIM_PGO_MERGED_PROFILE}. Run 'llvm-profdata merge' first.")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_compile_options(-fprofile-use -fprofile-dir=${PULSIM_PGO_PROFILE_DIR} -fprofile-correction)
        add_link_options(-fprofile-use)
        message(STATUS "PGO: Using profiles from ${PULSIM_PGO_PROFILE_DIR}")
    elseif(MSVC)
        add_compile_options(/GL)
        add_link_options(/LTCG /USEPROFILE:PGD=${PULSIM_PGO_PROFILE_DIR}/pulsim.pgd)
        message(STATUS "PGO: Using MSVC profile")
    endif()
endif()

# Native architecture optimizations (use with caution for distribution)
if(PULSIM_ENABLE_NATIVE AND NOT MSVC)
    add_compile_options(-march=native -mtune=native)
    message(STATUS "Native optimizations: Enabled (-march=native)")
endif()

# =============================================================================
# Ninja Generator Optimization
# =============================================================================
# Recommend Ninja for faster parallel builds:
#   cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
if(CMAKE_GENERATOR STREQUAL "Ninja")
    message(STATUS "Build system: Ninja (parallel compilation optimized)")
    # Ninja handles parallelism automatically - no additional configuration needed
elseif(CMAKE_GENERATOR MATCHES "Makefiles")
    message(STATUS "Build system: Make (use -jN for parallel builds, or switch to Ninja)")
    message(STATUS "Tip: For faster builds, use: cmake -G Ninja ...")
endif()

# =============================================================================
# Compiler Detection and Verification
# =============================================================================
# Verify Clang version for C++23 support (recommended: 17+)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "17.0")
        message(WARNING
            "Clang ${CMAKE_CXX_COMPILER_VERSION} detected. "
            "Version 17+ is recommended for full C++23 support.\n"
            "Consider using the Clang toolchain: cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-clang.cmake ..."
        )
    else()
        message(STATUS "Clang ${CMAKE_CXX_COMPILER_VERSION} - C++23 support OK")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "13.0")
        message(WARNING
            "GCC ${CMAKE_CXX_COMPILER_VERSION} detected. "
            "Version 13+ is recommended for full C++23 support.\n"
            "Consider using Clang 17+: cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-clang.cmake ..."
        )
    else()
        message(STATUS "GCC ${CMAKE_CXX_COMPILER_VERSION} - C++23 support OK")
    endif()
elseif(MSVC)
    if(MSVC_VERSION LESS 1937)
        message(WARNING
            "MSVC ${MSVC_VERSION} detected. "
            "Visual Studio 2022 17.7+ is recommended for full C++23 support."
        )
    else()
        message(STATUS "MSVC ${MSVC_VERSION} - C++23 support OK")
    endif()
endif()

message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard: C++${CMAKE_CXX_STANDARD}")

# Note: We use FetchContent_Populate for Eigen (header-only) to avoid
# processing its CMakeLists.txt which includes many test targets.
# Other dependencies have their test options disabled via cache variables.

# =============================================================================
# Compiler Warnings and Flags
# =============================================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Clang-specific flags (preferred compiler)
    add_compile_options(
        -Wall -Wextra -Wpedantic -Werror=return-type
        -Wno-c++98-compat -Wno-c++98-compat-pedantic
    )
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0 -fno-omit-frame-pointer)
    else()
        add_compile_options(-O3 -DNDEBUG)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC flags
    add_compile_options(-Wall -Wextra -Wpedantic -Werror=return-type)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0)
    else()
        add_compile_options(-O3 -DNDEBUG)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    # AppleClang (Xcode) flags
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

# Core library (v2 header-only)
add_subdirectory(core)

# Python bindings
if(PULSIM_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# =============================================================================
# Benchmark Targets
# =============================================================================
option(PULSIM_BUILD_BENCHMARKS "Build benchmark targets" OFF)

if(PULSIM_BUILD_BENCHMARKS)
    # Compile-time benchmark: measure template instantiation time
    add_custom_target(benchmark_compile_time
        COMMAND ${CMAKE_COMMAND} -E echo "=== Compile-Time Benchmark ==="
        COMMAND ${CMAKE_COMMAND} -E echo "Cleaning build..."
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
        COMMAND ${CMAKE_COMMAND} -E echo "Timing full rebuild..."
        COMMAND ${CMAKE_COMMAND} -E time ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target pulsim_tests -- -j1
        COMMENT "Measuring compile time (single-threaded for consistency)"
    )
endif()

# =============================================================================
# PGO Workflow Helper Targets
# =============================================================================
if(PULSIM_ENABLE_PGO_GENERATE)
    add_custom_target(pgo_merge_profiles
        COMMAND ${CMAKE_COMMAND} -E echo "Merging PGO profiles..."
        COMMAND llvm-profdata merge -output=${PULSIM_PGO_PROFILE_DIR}/merged.profdata ${PULSIM_PGO_PROFILE_DIR}/*.profraw
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Merging LLVM PGO profiles (run after training workload)"
    )
endif()

# =============================================================================
# Build Configuration Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== PulsimCore v2 Build Configuration ===")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:      C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler:          ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Generator:         ${CMAKE_GENERATOR}")
message(STATUS "  LTO:               ${PULSIM_ENABLE_LTO}")
message(STATUS "  PGO Generate:      ${PULSIM_ENABLE_PGO_GENERATE}")
message(STATUS "  PGO Use:           ${PULSIM_ENABLE_PGO_USE}")
message(STATUS "  Native opts:       ${PULSIM_ENABLE_NATIVE}")
message(STATUS "  Build tests:       ${PULSIM_BUILD_TESTS}")
message(STATUS "  Build Python:      ${PULSIM_BUILD_PYTHON}")
message(STATUS "  SUNDIALS:          ${PULSIM_USE_SUNDIALS}")
message(STATUS "  SuiteSparse:       ${PULSIM_USE_SUITESPARSE}")
message(STATUS "=========================================")
