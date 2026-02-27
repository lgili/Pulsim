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
        yaml-cpp
        ${PULSIM_KLU_LIBRARIES}
        ${PULSIM_HYPRE_LIBRARIES}
)

# Alias for cleaner usage
add_library(pulsim::core ALIAS pulsim_core)

# =============================================================================
# Compiled pulsim library (Circuit, Simulator, MNA, etc.)
# =============================================================================
add_library(pulsim STATIC
    src/v1/simulation.cpp
    src/v1/simulation_periodic.cpp
    src/v1/simulation_step.cpp
    src/v1/transient_services.cpp
    src/v1/yaml_parser.cpp
    src/simulation_control.cpp
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
        yaml-cpp
        ${PULSIM_KLU_LIBRARIES}
        ${PULSIM_HYPRE_LIBRARIES}
)
if(COMMAND pulsim_apply_target_defaults)
    pulsim_apply_target_defaults(pulsim)
endif()

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
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_tests)
    endif()

    # Enable profiling for benchmark tests in Debug builds
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_definitions(pulsim_tests PRIVATE PULSIM_ENABLE_PROFILING)
    endif()

    catch_discover_tests(pulsim_tests)

    # Compiled library tests (Circuit, Simulator, power electronics)
    add_executable(pulsim_simulation_tests
        tests/test_main.cpp
        tests/test_linear_solver_selection.cpp
        tests/test_newton_krylov.cpp
        tests/test_v1_diagnostics.cpp
        tests/test_v1_input_validation.cpp
        tests/test_v1_periodic_validation.cpp
        tests/test_v1_stage_context_raii.cpp
        tests/test_extension_registry.cpp
        tests/test_v1_kernel.cpp
        tests/test_stress_simulation.cpp
    )

    target_link_libraries(pulsim_simulation_tests
        PRIVATE
            pulsim::pulsim
            pulsim::core
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_simulation_tests)
    endif()

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
