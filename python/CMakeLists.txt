# Python bindings using pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Fetch pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(pybind11)

# Create the Python module
pybind11_add_module(_spicelab
    bindings.cpp
)

target_link_libraries(_spicelab
    PRIVATE
        spicelab::core
)

# Set the output name to match Python conventions
set_target_properties(_spicelab PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/spicelab"
)

# Copy Python wrapper module
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/spicelab/__init__.py
    ${CMAKE_BINARY_DIR}/python/spicelab/__init__.py
    COPYONLY
)

# Install targets
install(TARGETS _spicelab
    LIBRARY DESTINATION spicelab
)

install(DIRECTORY spicelab/
    DESTINATION .
    FILES_MATCHING PATTERN "*.py"
)
