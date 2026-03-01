# PulsimCore v2 - Header-only library
# The v2 API is implemented entirely in headers using C++23 features

# Create an interface (header-only) library
add_library(pulsim_core INTERFACE)

target_include_directories(pulsim_core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(pulsim_core INTERFACE cxx_std_23)

target_link_libraries(pulsim_core
    INTERFACE
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        ${PULSIM_KLU_LIBRARIES}
)

# Alias for cleaner usage
add_library(pulsim::core ALIAS pulsim_core)

# =============================================================================
# Compiled pulsim library (Circuit, Simulator, MNA, etc.)
# =============================================================================
add_library(pulsim STATIC
    src/circuit.cpp
    src/mna.cpp
    src/simulation.cpp
    src/simulation_control.cpp
    src/solver.cpp
    src/devices.cpp
    src/parser.cpp
    src/advanced_solver.cpp
    src/convergence_aids.cpp
    src/thermal.cpp
    src/ac_analysis.cpp
    src/parallel.cpp
    src/metadata.cpp
    src/validation.cpp
)

target_include_directories(pulsim
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(pulsim PUBLIC cxx_std_23)

target_link_libraries(pulsim
    PUBLIC
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        ${PULSIM_KLU_LIBRARIES}
)

# Alias for cleaner usage
add_library(pulsim::pulsim ALIAS pulsim)

# =============================================================================
# Tests
# =============================================================================
if(PULSIM_BUILD_TESTS)
    # Header-only v1/v2 API tests
    add_executable(pulsim_tests
        tests/test_main.cpp
        tests/test_concepts.cpp
        tests/test_richardson_lte.cpp
        tests/test_benchmarks.cpp
    )

    target_link_libraries(pulsim_tests
        PRIVATE
            pulsim::core
            Catch2::Catch2WithMain
    )

    # Enable profiling for benchmark tests in Debug builds
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_definitions(pulsim_tests PRIVATE PULSIM_ENABLE_PROFILING)
    endif()

    catch_discover_tests(pulsim_tests)

    # Compiled library tests (Circuit, Simulator, power electronics)
    add_executable(pulsim_simulation_tests
        tests/test_main.cpp
        tests/test_circuit.cpp
        tests/test_mna.cpp
        tests/test_solver.cpp
        tests/test_simulation.cpp
        tests/test_power_electronics.cpp
        tests/test_difficult_converters.cpp
        tests/test_thermal.cpp
        tests/test_convergence_aids.cpp
    )

    target_link_libraries(pulsim_simulation_tests
        PRIVATE
            pulsim::pulsim
            pulsim::core
            Catch2::Catch2WithMain
    )

    catch_discover_tests(pulsim_simulation_tests)

    # =============================================================================
    # Custom test targets for convenience
    # =============================================================================

    # Run all simulation tests
    add_custom_target(test-all
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/pulsim_simulation_tests
        DEPENDS pulsim_simulation_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running all simulation tests..."
    )

    # Run only quick tests (converters, etc.)
    add_custom_target(test-quick
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/pulsim_simulation_tests "[quick]"
        DEPENDS pulsim_simulation_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running quick tests..."
    )

    # Run converter tests
    add_custom_target(test-converters
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/pulsim_simulation_tests "[converter]"
        DEPENDS pulsim_simulation_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running converter tests..."
    )

    # Run tests with verbose output
    add_custom_target(test-verbose
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/pulsim_simulation_tests -s -d
        DEPENDS pulsim_simulation_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running all tests with verbose output..."
    )

    # List all available tests
    add_custom_target(test-list
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/pulsim_simulation_tests --list-tests
        DEPENDS pulsim_simulation_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Listing all available tests..."
    )
endif()
