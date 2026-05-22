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

# Create the Python module — v2-only since pulsim 1.0.0. The
# legacy ``bindings.cpp`` (4 697 lines exposing the v1 kernel) was
# removed when v1 retired; only the v2 kernel bindings remain.
pybind11_add_module(_pulsim
    NO_EXTRAS
    bindings_v2_kernel.cpp
)

target_link_libraries(_pulsim
    PRIVATE
        pulsim::v2
)
if(APPLE)
    target_compile_options(_pulsim PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
endif()
if(COMMAND pulsim_apply_target_defaults)
    pulsim_apply_target_defaults(_pulsim)
endif()

# Set the output name to match Python conventions
set_target_properties(_pulsim PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/pulsim"
)

# =============================================================================
# Copy Python wrapper module
# =============================================================================
# The build tree at ${CMAKE_BINARY_DIR}/python/pulsim must mirror the source
# tree at python/pulsim so that `PYTHONPATH=build/python python -c "import
# pulsim"` finds a complete, importable package. Missing files manifest as
# ModuleNotFoundError at import time (the source __init__.py imports
# frequency_analysis, templates, codegen, fmu, sweep at module load).
#
# Top-level .py and .pyi files: glob-discovered so new modules ship
# without a manual edit. snubber.py shipped broken because the old
# explicit list missed it; the install rule at the bottom of this
# file uses the same glob pattern to stay in sync.

file(GLOB _pulsim_top_level_files
    LIST_DIRECTORIES false
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/*.py
    ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/*.pyi
)
foreach(_pulsim_file IN LISTS _pulsim_top_level_files)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${_pulsim_file}
        ${CMAKE_BINARY_DIR}/python/${_pulsim_file}
        COPYONLY
    )
endforeach()

# Submodules: copy every .py inside each directory. A new file requires
# re-running cmake configure (CMAKE_CONFIGURE_DEPENDS would be heavier).
# The `schematic` submodule also ships:
#   - a Node.js bridge script (`elk_bridge.js`) for the ELK layout backend
#   - SVG skin files (`skin/pulsim_analog.svg`) for the netlistsvg backend
# Extend the glob to those extensions and to one level of sub-directory
# (so `schematic/skin/*.svg` gets copied).
set(_pulsim_submodules codegen fmu sweep schematic)
foreach(_pulsim_sub IN LISTS _pulsim_submodules)
    file(GLOB _pulsim_sub_files
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/${_pulsim_sub}/*.py
        ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/${_pulsim_sub}/*.js
        ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/${_pulsim_sub}/*/*.svg
    )
    foreach(_pulsim_sub_file IN LISTS _pulsim_sub_files)
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/${_pulsim_sub_file}
            ${CMAKE_BINARY_DIR}/python/${_pulsim_sub_file}
            COPYONLY
        )
    endforeach()
endforeach()

# =============================================================================
# Install targets
# =============================================================================
install(TARGETS _pulsim
    LIBRARY DESTINATION pulsim
)

# Auto-discover all top-level .py / .pyi files in pulsim/ so we don't
# silently forget new modules (snubber.py shipped broken because it
# wasn't added to the explicit FILES list when it landed in f1b6a86).
file(GLOB _pulsim_top_files
    LIST_DIRECTORIES false
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/*.py
    ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/*.pyi
)
install(FILES ${_pulsim_top_files}
    DESTINATION pulsim
)

foreach(_pulsim_sub IN LISTS _pulsim_submodules)
    install(DIRECTORY pulsim/${_pulsim_sub}/
        DESTINATION pulsim/${_pulsim_sub}
        FILES_MATCHING PATTERN "*.py"
    )
endforeach()
