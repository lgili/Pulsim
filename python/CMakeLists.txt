# Python bindings using pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

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
