# Python bindings using pybind11

# If provided via environment (cibuildwheel), hint CMake with explicit Python paths
if(NOT DEFINED Python3_EXECUTABLE AND DEFINED ENV{Python3_EXECUTABLE})
    set(Python3_EXECUTABLE "$ENV{Python3_EXECUTABLE}")
endif()
if(NOT DEFINED Python3_ROOT_DIR AND DEFINED ENV{Python3_ROOT_DIR})
    set(Python3_ROOT_DIR "$ENV{Python3_ROOT_DIR}")
endif()
if(NOT DEFINED Python3_LIBRARY AND DEFINED ENV{Python3_LIBRARY})
    set(Python3_LIBRARY "$ENV{Python3_LIBRARY}")
    set(Python3_LIBRARIES "$ENV{Python3_LIBRARY}")
endif()
if(NOT DEFINED Python3_INCLUDE_DIR AND DEFINED ENV{Python3_INCLUDE_DIR})
    set(Python3_INCLUDE_DIR "$ENV{Python3_INCLUDE_DIR}")
endif()

# Derive libpython path from the interpreter when possible to help FindPython locate Development/Embed components
if(DEFINED Python3_EXECUTABLE)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; libdir=sysconfig.get_config_var('LIBDIR') or ''; soname=sysconfig.get_config_var('INSTSONAME') or ''; incdir=sysconfig.get_paths().get('include', '') or ''; print(libdir); print(soname); print(incdir)"
        OUTPUT_VARIABLE PY_LIBINFO
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX MATCHALL "[^\n]+" PY_LIBINFO_LIST "${PY_LIBINFO}")
    list(LENGTH PY_LIBINFO_LIST PY_LIBINFO_LEN)
    if(PY_LIBINFO_LEN GREATER 2)
        list(GET PY_LIBINFO_LIST 0 PY_LIBDIR)
        list(GET PY_LIBINFO_LIST 1 PY_LIBSONAME)
        list(GET PY_LIBINFO_LIST 2 PY_INCDIR)
        if(NOT DEFINED Python3_LIBRARY AND PY_LIBDIR AND PY_LIBSONAME)
            set(Python3_LIBRARY "${PY_LIBDIR}/${PY_LIBSONAME}" CACHE FILEPATH "")
            set(Python3_LIBRARIES "${Python3_LIBRARY}" CACHE FILEPATH "")
        endif()
        if(NOT DEFINED Python3_INCLUDE_DIR AND PY_INCDIR)
            set(Python3_INCLUDE_DIR "${PY_INCDIR}" CACHE PATH "")
        endif()
    endif()
endif()

set(Python3_FIND_IMPLEMENTATIONS "CPython")
set(Python3_FIND_STRATEGY "LOCATION")
set(Python3_FIND_UNVERSIONED_NAMES "FIRST")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# On macOS prefer the Xcode clang and libc++ to ensure ABI compatibility
if(APPLE)
    find_program(XCRUN_EXECUTABLE xcrun)
    if(XCRUN_EXECUTABLE)
        execute_process(COMMAND xcrun --find clang OUTPUT_VARIABLE XCODE_CLANG OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND xcrun --find clang++ OUTPUT_VARIABLE XCODE_CLANGPP OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND xcrun --sdk macosx --show-sdk-path OUTPUT_VARIABLE XCODE_SYSROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(XCODE_CLANG)
            set(CMAKE_C_COMPILER "${XCODE_CLANG}")
        endif()
        if(XCODE_CLANGPP)
            set(CMAKE_CXX_COMPILER "${XCODE_CLANGPP}")
        endif()
        if(XCODE_SYSROOT)
            set(CMAKE_OSX_SYSROOT "${XCODE_SYSROOT}")
            add_compile_options("-isysroot" "${XCODE_SYSROOT}")
            set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "" FORCE)
        endif()
    endif()
    # Use older libc++ ABI for broader macOS compatibility and avoid newer symbols
    add_compile_definitions(_LIBCPP_ABI_VERSION=1 _LIBCPP_DISABLE_AVAILABILITY)
    add_compile_options(-stdlib=libc++)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++")
endif()

# Prefer pybind11 provided by the build environment (e.g., installed via pip)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.12.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Create the Python module
pybind11_add_module(_pulsim
    bindings.cpp
)

target_link_libraries(_pulsim
    PRIVATE
        pulsim::core
)

# Set the output name to match Python conventions
set_target_properties(_pulsim PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/pulsim"
)

# Copy Python wrapper module
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/__init__.py
    ${CMAKE_BINARY_DIR}/python/pulsim/__init__.py
    COPYONLY
)

# Install targets
install(TARGETS _pulsim
    LIBRARY DESTINATION pulsim
)

install(FILES pulsim/__init__.py
    DESTINATION pulsim
)
