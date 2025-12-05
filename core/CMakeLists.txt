add_library(spicelab_core
    src/parser.cpp
    src/circuit.cpp
    src/mna.cpp
    src/solver.cpp
    src/devices.cpp
    src/simulation.cpp
    src/thermal.cpp
)

target_include_directories(spicelab_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(spicelab_core
    PUBLIC
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
)

# Alias for cleaner usage
add_library(spicelab::core ALIAS spicelab_core)

# Tests
if(SPICELAB_BUILD_TESTS)
    add_executable(spicelab_tests
        tests/test_main.cpp
        tests/test_parser.cpp
        tests/test_circuit.cpp
        tests/test_mna.cpp
        tests/test_solver.cpp
        tests/test_simulation.cpp
        tests/test_power_electronics.cpp
        tests/test_thermal.cpp
    )

    target_link_libraries(spicelab_tests
        PRIVATE
            spicelab::core
            Catch2::Catch2WithMain
    )

    catch_discover_tests(spicelab_tests)
endif()
