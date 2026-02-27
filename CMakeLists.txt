cmake_minimum_required(VERSION 3.20)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# =============================================================================
# macOS: Force AppleClang for Python bindings (ABI compatibility)
# =============================================================================
# When building Python bindings on macOS, we MUST use AppleClang to avoid
# libc++ ABI mismatches. Homebrew's Clang uses a different libc++ version
# than the Python interpreter, causing "symbol not found" errors at runtime.
if(APPLE AND NOT DEFINED CMAKE_C_COMPILER AND NOT DEFINED CMAKE_CXX_COMPILER)
    # Check if we're building Python bindings (scikit-build-core sets this)
    if(SKBUILD OR PULSIM_BUILD_PYTHON)
        # Force AppleClang for Python ABI compatibility
        find_program(APPLE_CLANG "/usr/bin/clang")
        find_program(APPLE_CLANGXX "/usr/bin/clang++")

        if(APPLE_CLANG AND APPLE_CLANGXX)
            set(CMAKE_C_COMPILER "${APPLE_CLANG}" CACHE FILEPATH "C compiler" FORCE)
            set(CMAKE_CXX_COMPILER "${APPLE_CLANGXX}" CACHE FILEPATH "C++ compiler" FORCE)
            message(STATUS "macOS Python bindings: Using AppleClang for ABI compatibility")
        endif()
    endif()
endif()

project(pulsim VERSION 0.5.0 LANGUAGES CXX)

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
# Note: SuiteSparse/KLU is now a required dependency (default backend for linear solver)

# =============================================================================
# Performance Optimization Options
# =============================================================================
option(PULSIM_ENABLE_LTO "Enable Link-Time Optimization for Release builds" ON)
option(PULSIM_ENABLE_NATIVE "Enable native architecture optimizations" OFF)
option(PULSIM_ENABLE_PGO_GENERATE "Enable PGO instrumentation (generate profile)" OFF)
option(PULSIM_ENABLE_PGO_USE "Enable PGO optimization (use profile)" OFF)
option(PULSIM_SANITIZERS "Enable ASan/UBSan for supported compilers" OFF)
set(PULSIM_PGO_PROFILE_DIR "${CMAKE_BINARY_DIR}/pgo-profiles" CACHE PATH "Directory for PGO profile data")

# Python extension builds link static libraries into a shared module, so
# position-independent code is required for reliable Linux wheel/editable builds.
if(SKBUILD OR PULSIM_BUILD_PYTHON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# Editable/wheel Python builds on Linux can hit linker relocation issues with
# ThinLTO + pybind11 thread-local storage; keep LTO for core C++ builds.
if((SKBUILD OR PULSIM_BUILD_PYTHON) AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(PULSIM_ENABLE_LTO OFF CACHE BOOL "Disable LTO for Python builds on Linux" FORCE)
    message(STATUS "LTO: Disabled for Python build on Linux (pybind11 TLS compatibility)")
endif()

# CI Build Detection - skip timing-sensitive benchmark tests
# Auto-detect CI environments or allow manual override
if(DEFINED ENV{CI} OR DEFINED ENV{GITHUB_ACTIONS})
    option(PULSIM_CI_BUILD "Building in CI environment (skips timing benchmarks)" ON)
else()
    option(PULSIM_CI_BUILD "Building in CI environment (skips timing benchmarks)" OFF)
endif()

if(PULSIM_CI_BUILD)
    add_compile_definitions(PULSIM_CI_BUILD=1)
    message(STATUS "CI Build: Timing benchmarks will be skipped")
endif()

# LTO (Link-Time Optimization) - significant performance improvement
if(PULSIM_ENABLE_LTO AND CMAKE_BUILD_TYPE STREQUAL "Release")
    # Some CI/editable-build environments can expose mismatched CMake module
    # roots (e.g. pip build env cleanup on Windows), making CheckIPOSupported
    # fail while probing. In that case, disable LTO gracefully.
    set(_ipo_template "${CMAKE_ROOT}/Modules/CheckIPOSupported/CMakeLists-CXX.txt.in")
    if(EXISTS "${_ipo_template}")
        include(CheckIPOSupported)
        if(COMMAND check_ipo_supported)
            check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_ERROR)
            if(LTO_SUPPORTED)
                set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
                message(STATUS "LTO: Enabled (interprocedural optimization)")
            else()
                message(WARNING "LTO: Not supported - ${LTO_ERROR}")
            endif()
        else()
            message(WARNING "LTO: check_ipo_supported() unavailable; disabling LTO for this configure")
        endif()
    else()
        message(WARNING
            "LTO: IPO template not found at '${_ipo_template}'. "
            "Disabling LTO for this configure (likely transient CMake module path in CI)."
        )
    endif()
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
# Verify compiler version for C++23 support
if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    # AppleClang versioning: 15.0+ has good C++23 support (ships with Xcode 15+)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "15.0")
        message(WARNING
            "AppleClang ${CMAKE_CXX_COMPILER_VERSION} detected. "
            "Version 15+ (Xcode 15+) is recommended for C++23 support.\n"
            "Update Xcode via: xcode-select --install"
        )
    else()
        message(STATUS "AppleClang ${CMAKE_CXX_COMPILER_VERSION} - C++23 support OK")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # LLVM Clang: 17+ for full C++23
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
# Project-Wide Target-Scoped Build Flags
# =============================================================================
# Keep warnings/opts target-scoped so external dependencies are not affected.
set(PULSIM_TARGET_COMPILE_OPTIONS "")
set(PULSIM_TARGET_LINK_OPTIONS "")

# PGO (Profile-Guided Optimization) - requires two-pass build
# Pass 1: cmake -DPULSIM_ENABLE_PGO_GENERATE=ON .. && make && ./run_benchmarks
# Pass 2: cmake -DPULSIM_ENABLE_PGO_USE=ON .. && make
if(PULSIM_ENABLE_PGO_GENERATE AND PULSIM_ENABLE_PGO_USE)
    message(FATAL_ERROR "Cannot enable both PGO_GENERATE and PGO_USE simultaneously")
endif()

if(PULSIM_ENABLE_PGO_GENERATE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND PULSIM_TARGET_COMPILE_OPTIONS -fprofile-generate=${PULSIM_PGO_PROFILE_DIR})
        list(APPEND PULSIM_TARGET_LINK_OPTIONS -fprofile-generate=${PULSIM_PGO_PROFILE_DIR})
        message(STATUS "PGO: Instrumentation enabled (profiles -> ${PULSIM_PGO_PROFILE_DIR})")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(APPEND PULSIM_TARGET_COMPILE_OPTIONS -fprofile-generate -fprofile-dir=${PULSIM_PGO_PROFILE_DIR})
        list(APPEND PULSIM_TARGET_LINK_OPTIONS -fprofile-generate)
        message(STATUS "PGO: Instrumentation enabled (GCC, profiles -> ${PULSIM_PGO_PROFILE_DIR})")
    elseif(MSVC)
        list(APPEND PULSIM_TARGET_COMPILE_OPTIONS /GL)
        list(APPEND PULSIM_TARGET_LINK_OPTIONS /LTCG /GENPROFILE:PGD=${PULSIM_PGO_PROFILE_DIR}/pulsim.pgd)
        message(STATUS "PGO: Instrumentation enabled (MSVC)")
    endif()
endif()

if(PULSIM_ENABLE_PGO_USE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Merge profiles first: llvm-profdata merge -output=merged.profdata ${PULSIM_PGO_PROFILE_DIR}/*.profraw
        set(PULSIM_PGO_MERGED_PROFILE "${PULSIM_PGO_PROFILE_DIR}/merged.profdata" CACHE FILEPATH "Merged PGO profile")
        if(EXISTS "${PULSIM_PGO_MERGED_PROFILE}")
            list(APPEND PULSIM_TARGET_COMPILE_OPTIONS -fprofile-use=${PULSIM_PGO_MERGED_PROFILE})
            list(APPEND PULSIM_TARGET_LINK_OPTIONS -fprofile-use=${PULSIM_PGO_MERGED_PROFILE})
            message(STATUS "PGO: Using profile ${PULSIM_PGO_MERGED_PROFILE}")
        else()
            message(WARNING "PGO: Profile not found at ${PULSIM_PGO_MERGED_PROFILE}. Run 'llvm-profdata merge' first.")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(APPEND PULSIM_TARGET_COMPILE_OPTIONS -fprofile-use -fprofile-dir=${PULSIM_PGO_PROFILE_DIR} -fprofile-correction)
        list(APPEND PULSIM_TARGET_LINK_OPTIONS -fprofile-use)
        message(STATUS "PGO: Using profiles from ${PULSIM_PGO_PROFILE_DIR}")
    elseif(MSVC)
        list(APPEND PULSIM_TARGET_COMPILE_OPTIONS /GL)
        list(APPEND PULSIM_TARGET_LINK_OPTIONS /LTCG /USEPROFILE:PGD=${PULSIM_PGO_PROFILE_DIR}/pulsim.pgd)
        message(STATUS "PGO: Using MSVC profile")
    endif()
endif()

# Native architecture optimizations (use with caution for distribution)
if(PULSIM_ENABLE_NATIVE AND NOT MSVC)
    list(APPEND PULSIM_TARGET_COMPILE_OPTIONS -march=native -mtune=native)
    message(STATUS "Native optimizations: Enabled (-march=native)")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    list(APPEND PULSIM_TARGET_COMPILE_OPTIONS
        -Wall -Wextra -Wpedantic -Werror=return-type
        -Wno-c++98-compat -Wno-c++98-compat-pedantic
    )
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND PULSIM_TARGET_COMPILE_OPTIONS -g -O0 -fno-omit-frame-pointer)
    else()
        list(APPEND PULSIM_TARGET_COMPILE_OPTIONS -O3 -DNDEBUG)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    list(APPEND PULSIM_TARGET_COMPILE_OPTIONS
        -Wall -Wextra -Wpedantic -Werror=return-type
        -Wno-c++98-compat -Wno-c++98-compat-pedantic
    )
    # On Linux, optionally use libc++ for pure C++ builds.
    # For Python builds (wheels/editable), keep default libstdc++ to avoid
    # ABI mismatches with dependency/toolchain artifacts in CI.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT (SKBUILD OR PULSIM_BUILD_PYTHON))
        list(APPEND PULSIM_TARGET_COMPILE_OPTIONS -stdlib=libc++)
        list(APPEND PULSIM_TARGET_LINK_OPTIONS -stdlib=libc++ -lc++abi)
        message(STATUS "Using libc++ on Linux for Clang")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(STATUS "Using default libstdc++ on Linux for Python build compatibility")
    endif()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND PULSIM_TARGET_COMPILE_OPTIONS -g -O0 -fno-omit-frame-pointer)
    else()
        list(APPEND PULSIM_TARGET_COMPILE_OPTIONS -O3 -DNDEBUG)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND PULSIM_TARGET_COMPILE_OPTIONS -Wall -Wextra -Wpedantic -Werror=return-type)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND PULSIM_TARGET_COMPILE_OPTIONS -g -O0)
    else()
        list(APPEND PULSIM_TARGET_COMPILE_OPTIONS -O3 -DNDEBUG)
    endif()
elseif(MSVC)
    list(APPEND PULSIM_TARGET_COMPILE_OPTIONS /W4)
endif()

if(PULSIM_SANITIZERS)
    if(MSVC)
        message(WARNING "PULSIM_SANITIZERS is enabled, but MSVC sanitizers are not configured in this project.")
    elseif(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(WARNING "PULSIM_SANITIZERS works best with Debug builds; current build type is '${CMAKE_BUILD_TYPE}'.")
    endif()
    if(NOT MSVC)
        list(APPEND PULSIM_TARGET_COMPILE_OPTIONS -fsanitize=address,undefined)
        list(APPEND PULSIM_TARGET_LINK_OPTIONS -fsanitize=address,undefined)
    endif()
endif()

list(REMOVE_DUPLICATES PULSIM_TARGET_COMPILE_OPTIONS)
list(REMOVE_DUPLICATES PULSIM_TARGET_LINK_OPTIONS)

add_library(pulsim_target_defaults INTERFACE)
if(PULSIM_TARGET_COMPILE_OPTIONS)
    target_compile_options(pulsim_target_defaults INTERFACE ${PULSIM_TARGET_COMPILE_OPTIONS})
endif()
if(PULSIM_TARGET_LINK_OPTIONS)
    target_link_options(pulsim_target_defaults INTERFACE ${PULSIM_TARGET_LINK_OPTIONS})
endif()

function(pulsim_apply_target_defaults target_name)
    if(NOT TARGET ${target_name})
        message(FATAL_ERROR "Target '${target_name}' not found for pulsim_apply_target_defaults()")
    endif()
    get_target_property(_is_imported ${target_name} IMPORTED)
    if(_is_imported)
        return()
    endif()
    get_target_property(_target_type ${target_name} TYPE)
    if(_target_type STREQUAL "INTERFACE_LIBRARY")
        target_link_libraries(${target_name} INTERFACE pulsim_target_defaults)
    else()
        target_link_libraries(${target_name} PRIVATE pulsim_target_defaults)
    endif()
endfunction()

# Dependencies via FetchContent
include(FetchContent)

# Note: We don't set BUILD_TESTING OFF globally as it would disable our own tests.

# Eigen - header-only dependency
# Prefer system package in CI to avoid network flakiness, then fall back to fetch.
find_package(Eigen3 3.4 QUIET NO_MODULE)

# Some CMake/Python build environments miss Homebrew prefixes by default.
if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 3.4 QUIET NO_MODULE
        PATHS
            /opt/homebrew/opt/eigen
            /usr/local/opt/eigen
        PATH_SUFFIXES
            share/eigen3/cmake
    )
endif()

if(TARGET Eigen3::Eigen)
    message(STATUS "Eigen3: using system package")
else()
    # Last local fallback for environments with headers installed but no CMake config package.
    find_path(PULSIM_EIGEN3_INCLUDE_DIR
        NAMES Eigen/Core
        PATHS
            /opt/homebrew/include
            /usr/local/include
            /usr/include
        PATH_SUFFIXES
            eigen3
    )

    if(PULSIM_EIGEN3_INCLUDE_DIR)
        message(STATUS "Eigen3: using headers from ${PULSIM_EIGEN3_INCLUDE_DIR}")
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${PULSIM_EIGEN3_INCLUDE_DIR}"
        )
    else()
        # Network fallback uses archive download (no git credentials required).
        set(PULSIM_EIGEN_ARCHIVE_URL
            "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"
            CACHE STRING "Eigen source archive URL"
        )
        message(STATUS "Eigen3: system package not found, downloading ${PULSIM_EIGEN_ARCHIVE_URL}")

        # Use populate-only mode so Eigen's own CMake targets/tests are never added.
        set(_pulsim_eigen_source_dir "${CMAKE_BINARY_DIR}/_deps/eigen-src")
        set(_pulsim_eigen_subbuild_dir "${CMAKE_BINARY_DIR}/_deps/eigen-subbuild")
        if(NOT EXISTS "${_pulsim_eigen_source_dir}/Eigen/Core")
            FetchContent_Populate(
                eigen
                SOURCE_DIR "${_pulsim_eigen_source_dir}"
                SUBBUILD_DIR "${_pulsim_eigen_subbuild_dir}"
                URL ${PULSIM_EIGEN_ARCHIVE_URL}
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE
            )
        endif()

        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${_pulsim_eigen_source_dir}"
        )
    endif()
endif()

# yaml-cpp - YAML parser
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG 0.8.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(yaml-cpp)

# GCC 14 + libstdc++ in C++23 mode can fail when compiling yaml-cpp internals
# (tuple-like conversion in std::map::operator[]). Build yaml-cpp with C++17
# while keeping Pulsim targets on C++23.
if(TARGET yaml-cpp)
    set_target_properties(yaml-cpp PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS OFF
    )
endif()

# Catch2 for testing (v3.8.0+ required for GCC 14 compatibility)
if(PULSIM_BUILD_TESTS)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.8.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Catch2)

    # GCC 14 + libstdc++ can fail compiling Catch2 internals in C++23 mode.
    # Keep our code on C++23, but build Catch2 itself with C++17.
    if(TARGET Catch2)
        set_target_properties(Catch2 PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED YES
            CXX_EXTENSIONS OFF
        )
    endif()
    if(TARGET Catch2WithMain)
        set_target_properties(Catch2WithMain PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED YES
            CXX_EXTENSIONS OFF
        )
    endif()

    # Linux Clang sanitizer jobs use libc++; ensure Catch2 is compiled with the
    # same standard library to avoid ABI mismatches (std::__1 vs std::__cxx11).
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT (SKBUILD OR PULSIM_BUILD_PYTHON))
        foreach(_catch_target Catch2 Catch2WithMain)
            if(TARGET ${_catch_target})
                target_compile_options(${_catch_target} PUBLIC -stdlib=libc++)
                target_link_options(${_catch_target} INTERFACE -stdlib=libc++ -lc++abi)
            endif()
        endforeach()
        message(STATUS "Catch2: forcing libc++ for Linux Clang ABI compatibility")
    endif()

    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(CTest)
    include(Catch)
endif()

# =============================================================================
# SuiteSparse KLU - Required dependency for fast sparse LU (circuit-optimized)
# =============================================================================
# KLU is the default linear solver backend. We try to find it on the system first,
# then fall back to building from source via FetchContent.

set(PULSIM_KLU_FOUND FALSE)

function(pulsim_validate_imported_include_dirs target_name result_var)
    if(NOT TARGET ${target_name})
        set(${result_var} FALSE PARENT_SCOPE)
        return()
    endif()

    get_target_property(_include_dirs ${target_name} INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT _include_dirs)
        set(${result_var} TRUE PARENT_SCOPE)
        return()
    endif()

    foreach(_include_dir IN LISTS _include_dirs)
        if("${_include_dir}" MATCHES "^\\$<")
            continue()
        endif()
        if(IS_ABSOLUTE "${_include_dir}" AND NOT EXISTS "${_include_dir}")
            message(WARNING
                "Imported target '${target_name}' has missing include directory: ${_include_dir}")
            set(${result_var} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()

    set(${result_var} TRUE PARENT_SCOPE)
endfunction()

# Method 1: Try to find system-installed SuiteSparse
find_package(SuiteSparse CONFIG QUIET)
if(SuiteSparse_FOUND AND TARGET SuiteSparse::KLU)
    pulsim_validate_imported_include_dirs(SuiteSparse::KLU _pulsim_klu_target_valid)
    if(_pulsim_klu_target_valid)
        message(STATUS "Found system SuiteSparse with KLU")
        set(PULSIM_KLU_FOUND TRUE)
        set(PULSIM_KLU_LIBRARIES SuiteSparse::KLU)
    else()
        message(WARNING
            "Ignoring broken SuiteSparse::KLU imported target and trying alternate discovery methods")
    endif()
endif()

# Method 2: Try pkg-config
if(NOT PULSIM_KLU_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(KLU QUIET IMPORTED_TARGET klu)
        if(KLU_FOUND AND TARGET PkgConfig::KLU)
            message(STATUS "Found KLU via pkg-config: ${KLU_LIBRARY_DIRS}")
            set(PULSIM_KLU_FOUND TRUE)
            add_library(SuiteSparse::KLU INTERFACE IMPORTED)
            target_link_libraries(SuiteSparse::KLU INTERFACE PkgConfig::KLU)
            set(PULSIM_KLU_LIBRARIES SuiteSparse::KLU)
        endif()
    endif()
endif()

# Method 3: Try direct find_library (common locations)
if(NOT PULSIM_KLU_FOUND)
    find_library(KLU_LIBRARY NAMES klu PATHS
        /usr/local/lib
        /usr/local/lib64
        /opt/homebrew/lib
        /usr/lib
        /usr/lib64
        /usr/lib/aarch64-linux-gnu
        /usr/lib/x86_64-linux-gnu
    )
    find_path(KLU_INCLUDE_DIR NAMES klu.h PATHS
        /usr/local/include
        /usr/local/include/suitesparse
        /opt/homebrew/include
        /opt/homebrew/include/suitesparse
        /usr/include
        /usr/include/suitesparse
    )
    find_library(AMD_LIBRARY NAMES amd PATHS
        /usr/local/lib
        /usr/local/lib64
        /opt/homebrew/lib
        /usr/lib
        /usr/lib64
        /usr/lib/aarch64-linux-gnu
        /usr/lib/x86_64-linux-gnu
    )
    find_library(COLAMD_LIBRARY NAMES colamd PATHS
        /usr/local/lib
        /usr/local/lib64
        /opt/homebrew/lib
        /usr/lib
        /usr/lib64
        /usr/lib/aarch64-linux-gnu
        /usr/lib/x86_64-linux-gnu
    )
    find_library(BTF_LIBRARY NAMES btf PATHS
        /usr/local/lib
        /usr/local/lib64
        /opt/homebrew/lib
        /usr/lib
        /usr/lib64
        /usr/lib/aarch64-linux-gnu
        /usr/lib/x86_64-linux-gnu
    )
    find_library(SUITESPARSE_CONFIG_LIBRARY NAMES suitesparseconfig PATHS
        /usr/local/lib
        /usr/local/lib64
        /opt/homebrew/lib
        /usr/lib
        /usr/lib64
        /usr/lib/aarch64-linux-gnu
        /usr/lib/x86_64-linux-gnu
    )

    if(KLU_LIBRARY AND KLU_INCLUDE_DIR AND AMD_LIBRARY AND COLAMD_LIBRARY AND BTF_LIBRARY AND SUITESPARSE_CONFIG_LIBRARY)
        message(STATUS "Found KLU via find_library: ${KLU_LIBRARY}")
        set(PULSIM_KLU_FOUND TRUE)
        # Create imported target
        add_library(SuiteSparse::KLU INTERFACE IMPORTED)
        set_target_properties(SuiteSparse::KLU PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${KLU_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "${KLU_LIBRARY};${AMD_LIBRARY};${COLAMD_LIBRARY};${BTF_LIBRARY};${SUITESPARSE_CONFIG_LIBRARY}"
        )
        set(PULSIM_KLU_LIBRARIES SuiteSparse::KLU)
    endif()
endif()

# Method 4: Build from source via FetchContent
if(NOT PULSIM_KLU_FOUND)
    message(STATUS "SuiteSparse/KLU not found on system, building from source...")

    # SuiteSparse requires some configuration
    set(SUITESPARSE_ENABLE_PROJECTS "suitesparse_config;amd;colamd;btf;klu" CACHE STRING "" FORCE)
    # KLU-only build avoids CHOLMOD/BLAS coupling and is enough for our solver.
    set(KLU_USE_CHOLMOD OFF CACHE BOOL "" FORCE)
    set(CHOLMOD_CAMD OFF CACHE BOOL "" FORCE)
    set(SUITESPARSE_USE_OPENMP OFF CACHE BOOL "" FORCE)
    set(SUITESPARSE_CONFIG_USE_OPENMP OFF CACHE BOOL "" FORCE)
    # Bypass mandatory BLAS auto-detection for KLU-only builds on platforms where BLAS is unavailable.
    set(BLAS_LIBRARIES "" CACHE STRING "" FORCE)
    set(SUITESPARSE_USE_CUDA OFF CACHE BOOL "" FORCE)
    set(SUITESPARSE_USE_FORTRAN OFF CACHE BOOL "" FORCE)
    set(SUITESPARSE_DEMOS OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)

    FetchContent_Declare(
        suitesparse
        GIT_REPOSITORY https://github.com/DrTimothyAldenDavis/SuiteSparse.git
        GIT_TAG v7.6.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(suitesparse)

    set(PULSIM_KLU_FOUND TRUE)
    set(PULSIM_KLU_LIBRARIES SuiteSparse::KLU)
    message(STATUS "Built SuiteSparse/KLU from source")
endif()

# Define the macro so code knows KLU is available
if(PULSIM_KLU_FOUND)
    add_compile_definitions(PULSIM_HAS_KLU)
    message(STATUS "KLU support: ENABLED")
else()
    message(FATAL_ERROR "SuiteSparse/KLU is required but could not be found or built. "
        "Install with: brew install suite-sparse (macOS) or apt install libsuitesparse-dev (Linux)")
endif()

# =============================================================================
# HYPRE AMG - Optional dependency for AMG preconditioning
# =============================================================================
option(PULSIM_USE_HYPRE "Enable HYPRE AMG backend if available" ON)
set(PULSIM_HYPRE_FOUND FALSE)
set(PULSIM_HYPRE_LIBRARIES "")

if(PULSIM_USE_HYPRE)
    find_package(HYPRE QUIET)
    if(HYPRE_FOUND)
        set(PULSIM_HYPRE_FOUND TRUE)
        if(TARGET HYPRE::HYPRE)
            set(PULSIM_HYPRE_LIBRARIES HYPRE::HYPRE)
        elseif(TARGET HYPRE)
            set(PULSIM_HYPRE_LIBRARIES HYPRE)
        else()
            if(HYPRE_LIBRARIES)
                add_library(HYPRE::HYPRE INTERFACE IMPORTED)
                set_target_properties(HYPRE::HYPRE PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "${HYPRE_INCLUDE_DIRS}"
                    INTERFACE_LINK_LIBRARIES "${HYPRE_LIBRARIES}"
                )
                set(PULSIM_HYPRE_LIBRARIES HYPRE::HYPRE)
            else()
                set(PULSIM_HYPRE_FOUND FALSE)
            endif()
        endif()
    endif()
endif()

if(PULSIM_HYPRE_FOUND)
    add_compile_definitions(PULSIM_HAS_HYPRE)
    message(STATUS "HYPRE AMG support: ENABLED")
else()
    message(STATUS "HYPRE AMG support: DISABLED")
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
message(STATUS "  KLU (required):    ${PULSIM_KLU_FOUND}")
message(STATUS "  HYPRE AMG:         ${PULSIM_HYPRE_FOUND}")
message(STATUS "=========================================")
