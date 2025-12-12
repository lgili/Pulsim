# =============================================================================
# PulsimCore - Clang Toolchain File
# =============================================================================
# Usage:
#   cmake -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-clang.cmake
#
# This toolchain:
#   - Automatically detects LLVM installation path per platform
#   - Configures Clang as the C/C++ compiler
#   - Verifies minimum Clang version (17+) for C++23 support
#   - Sets up platform-specific flags (libc++, rpath, etc.)
# =============================================================================

cmake_minimum_required(VERSION 3.20)

# Minimum required Clang version for C++23
set(PULSIM_CLANG_MIN_VERSION 17)

# =============================================================================
# Detect LLVM Installation Path
# =============================================================================

if(NOT DEFINED LLVM_PREFIX)
    if(APPLE)
        # macOS: Try Homebrew LLVM first
        execute_process(
            COMMAND brew --prefix llvm
            OUTPUT_VARIABLE LLVM_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE BREW_RESULT
        )
        if(NOT BREW_RESULT EQUAL 0 OR NOT EXISTS "${LLVM_PREFIX}/bin/clang")
            # Fallback paths for Apple Silicon and Intel Macs
            if(EXISTS "/opt/homebrew/opt/llvm/bin/clang")
                set(LLVM_PREFIX "/opt/homebrew/opt/llvm")
            elseif(EXISTS "/usr/local/opt/llvm/bin/clang")
                set(LLVM_PREFIX "/usr/local/opt/llvm")
            endif()
        endif()

    elseif(WIN32)
        # Windows: Standard LLVM installation path
        if(EXISTS "C:/Program Files/LLVM/bin/clang.exe")
            set(LLVM_PREFIX "C:/Program Files/LLVM")
        elseif(EXISTS "$ENV{LLVM_PATH}/bin/clang.exe")
            set(LLVM_PREFIX "$ENV{LLVM_PATH}")
        endif()

    else()
        # Linux: Check common LLVM installation paths
        foreach(VERSION RANGE 19 17 -1)
            if(EXISTS "/usr/lib/llvm-${VERSION}/bin/clang")
                set(LLVM_PREFIX "/usr/lib/llvm-${VERSION}")
                break()
            endif()
        endforeach()

        # Fallback to /usr if LLVM is in standard path
        if(NOT DEFINED LLVM_PREFIX)
            if(EXISTS "/usr/bin/clang")
                set(LLVM_PREFIX "/usr")
            endif()
        endif()
    endif()
endif()

# Verify LLVM was found
if(NOT DEFINED LLVM_PREFIX OR NOT EXISTS "${LLVM_PREFIX}")
    message(FATAL_ERROR
        "LLVM installation not found!\n"
        "Please install LLVM/Clang 17+ or set LLVM_PREFIX manually:\n"
        "  cmake -DLLVM_PREFIX=/path/to/llvm ...\n\n"
        "Installation instructions:\n"
        "  macOS:   ./scripts/setup-macos.sh\n"
        "  Windows: .\\scripts\\setup-windows.ps1\n"
        "  Linux:   apt install llvm-17 clang-17 (or equivalent)"
    )
endif()

# =============================================================================
# Set Compilers
# =============================================================================

if(WIN32)
    set(CLANG_EXECUTABLE "${LLVM_PREFIX}/bin/clang.exe")
    set(CLANGXX_EXECUTABLE "${LLVM_PREFIX}/bin/clang++.exe")
else()
    set(CLANG_EXECUTABLE "${LLVM_PREFIX}/bin/clang")
    set(CLANGXX_EXECUTABLE "${LLVM_PREFIX}/bin/clang++")
endif()

# Verify compilers exist
if(NOT EXISTS "${CLANG_EXECUTABLE}")
    message(FATAL_ERROR "Clang not found at: ${CLANG_EXECUTABLE}")
endif()
if(NOT EXISTS "${CLANGXX_EXECUTABLE}")
    message(FATAL_ERROR "Clang++ not found at: ${CLANGXX_EXECUTABLE}")
endif()

# Set CMake compiler variables
set(CMAKE_C_COMPILER "${CLANG_EXECUTABLE}" CACHE FILEPATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "${CLANGXX_EXECUTABLE}" CACHE FILEPATH "C++ compiler" FORCE)

# =============================================================================
# Verify Clang Version
# =============================================================================

execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} --version
    OUTPUT_VARIABLE CLANG_VERSION_OUTPUT
    ERROR_QUIET
)

string(REGEX MATCH "([0-9]+)\\.[0-9]+\\.[0-9]+" CLANG_VERSION_MATCH "${CLANG_VERSION_OUTPUT}")
string(REGEX MATCH "^([0-9]+)" CLANG_MAJOR_VERSION "${CLANG_VERSION_MATCH}")

if(NOT CLANG_MAJOR_VERSION)
    message(WARNING "Could not determine Clang version")
elseif(CLANG_MAJOR_VERSION LESS PULSIM_CLANG_MIN_VERSION)
    message(FATAL_ERROR
        "Clang ${CLANG_MAJOR_VERSION} found, but version ${PULSIM_CLANG_MIN_VERSION}+ is required for C++23 support.\n"
        "Please upgrade your LLVM installation:\n"
        "  macOS:   brew upgrade llvm\n"
        "  Windows: winget upgrade LLVM.LLVM\n"
        "  Linux:   apt install llvm-${PULSIM_CLANG_MIN_VERSION} clang-${PULSIM_CLANG_MIN_VERSION}"
    )
else()
    message(STATUS "Using Clang ${CLANG_MAJOR_VERSION} from ${LLVM_PREFIX}")
endif()

# =============================================================================
# Platform-Specific Configuration
# =============================================================================

if(APPLE)
    # macOS: Use Homebrew's libc++ for full C++23 support
    # Apple's system libc++ may lag behind in feature support

    set(LLVM_LIB_DIR "${LLVM_PREFIX}/lib/c++")

    if(EXISTS "${LLVM_LIB_DIR}")
        # Use LLVM's libc++ instead of Apple's
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++" CACHE STRING "" FORCE)

        # Linker flags for libc++
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${LLVM_LIB_DIR} -Wl,-rpath,${LLVM_LIB_DIR}" CACHE STRING "" FORCE)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${LLVM_LIB_DIR} -Wl,-rpath,${LLVM_LIB_DIR}" CACHE STRING "" FORCE)
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -L${LLVM_LIB_DIR} -Wl,-rpath,${LLVM_LIB_DIR}" CACHE STRING "" FORCE)

        message(STATUS "Using LLVM libc++ from ${LLVM_LIB_DIR}")
    else()
        message(STATUS "Using system libc++")
    endif()

    # Set deployment target
    if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version" FORCE)
    endif()

elseif(WIN32)
    # Windows: Clang can use MSVC runtime or its own
    # Default to MSVC compatibility mode for best library support

    # Use MSVC-compatible mode
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fms-compatibility-version=19.29" CACHE STRING "" FORCE)

    # Windows-specific defines
    add_compile_definitions(
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )

    message(STATUS "Using Clang with MSVC compatibility mode")

else()
    # Linux: Use system libc++ or libstdc++

    # Check if libc++ is available
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -stdlib=libc++ -x c++ -E -
        INPUT_FILE /dev/null
        OUTPUT_QUIET
        ERROR_QUIET
        RESULT_VARIABLE LIBCXX_RESULT
    )

    if(LIBCXX_RESULT EQUAL 0)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++" CACHE STRING "" FORCE)
        message(STATUS "Using libc++ on Linux")
    else()
        message(STATUS "Using libstdc++ on Linux")
    endif()
endif()

# =============================================================================
# Additional Compiler Flags
# =============================================================================

# Color diagnostics
if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics" CACHE STRING "" FORCE)
endif()

# =============================================================================
# Export Variables
# =============================================================================

set(PULSIM_LLVM_PREFIX "${LLVM_PREFIX}" CACHE PATH "LLVM installation prefix" FORCE)
set(PULSIM_CLANG_VERSION "${CLANG_MAJOR_VERSION}" CACHE STRING "Clang major version" FORCE)
set(PULSIM_USING_CLANG_TOOLCHAIN TRUE CACHE BOOL "Using Clang toolchain" FORCE)

# =============================================================================
# Summary
# =============================================================================

message(STATUS "")
message(STATUS "PulsimCore Clang Toolchain Configuration:")
message(STATUS "  LLVM Prefix:    ${LLVM_PREFIX}")
message(STATUS "  Clang Version:  ${CLANG_MAJOR_VERSION}")
message(STATUS "  C Compiler:     ${CMAKE_C_COMPILER}")
message(STATUS "  C++ Compiler:   ${CMAKE_CXX_COMPILER}")
message(STATUS "")
