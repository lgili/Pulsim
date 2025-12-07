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

install(DIRECTORY pulsim/
    DESTINATION .
    FILES_MATCHING PATTERN "*.py"
)
