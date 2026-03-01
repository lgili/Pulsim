add_library(pulsim_core
    src/parser.cpp
    src/circuit.cpp
    src/mna.cpp
    src/solver.cpp
    src/advanced_solver.cpp
    src/convergence_aids.cpp
    src/ac_analysis.cpp
    src/parallel.cpp
    src/devices.cpp
    src/simulation.cpp
    src/simulation_control.cpp
    src/thermal.cpp
    src/parser/spice_parser.cpp
    src/parser/yaml_parser.cpp
    src/parser/subcircuit.cpp
    src/fmu/fmu_export.cpp
    src/models/mosfet_models.cpp
    src/models/magnetic_core.cpp
    src/models/control_blocks.cpp
    src/metadata.cpp
    src/validation.cpp
)

# Enable position-independent code for linking into shared libraries (Python bindings)
set_target_properties(pulsim_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(pulsim_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(pulsim_core
    PUBLIC
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
)

# Optional SUNDIALS support
if(PULSIM_USE_SUNDIALS)
    if(TARGET SUNDIALS::ida)
        target_link_libraries(pulsim_core PUBLIC SUNDIALS::ida SUNDIALS::arkode)
    elseif(TARGET sundials_ida_static)
        target_link_libraries(pulsim_core PUBLIC sundials_ida_static sundials_arkode_static)
    endif()
endif()

# Optional SuiteSparse KLU support
if(PULSIM_USE_SUITESPARSE)
    if(TARGET SuiteSparse::KLU)
        target_link_libraries(pulsim_core PUBLIC SuiteSparse::KLU)
    elseif(KLU_FOUND)
        target_include_directories(pulsim_core PUBLIC ${KLU_INCLUDE_DIRS})
        target_link_libraries(pulsim_core PUBLIC ${KLU_LIBRARIES})
    endif()
endif()

# Alias for cleaner usage
add_library(pulsim::core ALIAS pulsim_core)

# Tests
if(PULSIM_BUILD_TESTS)
    add_executable(pulsim_tests
        tests/test_main.cpp
        tests/test_parser.cpp
        tests/test_circuit.cpp
        tests/test_mna.cpp
        tests/test_solver.cpp
        tests/test_simulation.cpp
        tests/test_power_electronics.cpp
        tests/test_thermal.cpp
        tests/test_cli.cpp
        tests/test_mvp2_validation.cpp
        tests/test_advanced_solver.cpp
        tests/test_convergence_aids.cpp
        tests/test_ac_analysis.cpp
        tests/test_parallel.cpp
        tests/test_mosfet_models.cpp
        tests/test_magnetic_core.cpp
        tests/test_control_blocks.cpp
        tests/test_metadata.cpp
        tests/test_validation.cpp
    )

    target_link_libraries(pulsim_tests
        PRIVATE
            pulsim::core
            Catch2::Catch2WithMain
    )

    # Ensure the CLI executable is built before running integration tests that invoke it
    if(TARGET pulsim)
        add_dependencies(pulsim_tests pulsim)
    endif()

    catch_discover_tests(pulsim_tests)
endif()
