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

# Create the Python module. All kernel symbols live directly on
# ``_pulsim`` (no submodule); ``python/pulsim/__init__.py`` imports
# them via ``from ._pulsim import ...``.
pybind11_add_module(_pulsim
    NO_EXTRAS
    bindings.cpp
    dsed_bindings.cpp
)

target_link_libraries(_pulsim
    PRIVATE
        pulsim::core
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

# Build-tree package copy — ONE recursive glob over the whole
# ``pulsim/`` tree (Phase-0 CI fix). History shows per-directory
# globs here are a repeat-offender failure mode: the flat-layout
# glob missed ``schematic/`` after the v2 merge (breaking CI's
# ``from . import schematic``), was patched with a second
# schematic-only glob, and then the SAME bug recurred when the
# ``dsed/`` subpackage landed — ``PYTHONPATH=build/python`` imports
# died on ``from . import dsed`` (misreported by CPython as a
# circular import), and the schematic-smoke job has been red since
# 2026-05-29. Recursing over every asset extension in one place
# means the NEXT subpackage ships without anyone editing this file.
file(GLOB_RECURSE _pulsim_pkg_files
    LIST_DIRECTORIES false
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/*.py
    ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/*.pyi
    ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/*.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/*.json
    ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/*.js
    ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/*.svg
    ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/*.h
)
foreach(_pulsim_file IN LISTS _pulsim_pkg_files)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${_pulsim_file}
        ${CMAKE_BINARY_DIR}/python/${_pulsim_file}
        COPYONLY
    )
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

# Install the schematic submodule alongside the top-level Python
# files. Mirrors the build-tree copy above so wheel users get the
# subpackage + YAML templates.
file(GLOB_RECURSE _pulsim_schematic_install_files
    LIST_DIRECTORIES false
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/pulsim
    ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/schematic/*.py
    ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/schematic/*.pyi
    ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/schematic/*.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/schematic/*.json
    ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/schematic/*.js
    ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/schematic/*.svg
)
foreach(_rel_path IN LISTS _pulsim_schematic_install_files)
    get_filename_component(_rel_dir ${_rel_path} DIRECTORY)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/${_rel_path}
            DESTINATION pulsim/${_rel_dir})
endforeach()

# Install the dsed (Path-Based Event-Driven) Python subpackage
# alongside the top-level files. Mirrors the schematic pattern.
# Source: add-path-based-dsed-engine OpenSpec, Gate 2 Phase 2.B-3.
file(GLOB _pulsim_dsed_install_files
    LIST_DIRECTORIES false
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/pulsim
    ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/dsed/*.py
)
foreach(_rel_path IN LISTS _pulsim_dsed_install_files)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/pulsim/${_rel_path}
            DESTINATION pulsim/dsed)
endforeach()
