add_library(spicelab_core
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

# Optional SUNDIALS support
if(SPICELAB_USE_SUNDIALS)
    if(TARGET SUNDIALS::ida)
        target_link_libraries(spicelab_core PUBLIC SUNDIALS::ida SUNDIALS::arkode)
    elseif(TARGET sundials_ida_static)
        target_link_libraries(spicelab_core PUBLIC sundials_ida_static sundials_arkode_static)
    endif()
endif()

# Optional SuiteSparse KLU support
if(SPICELAB_USE_SUITESPARSE)
    if(TARGET SuiteSparse::KLU)
        target_link_libraries(spicelab_core PUBLIC SuiteSparse::KLU)
    elseif(KLU_FOUND)
        target_include_directories(spicelab_core PUBLIC ${KLU_INCLUDE_DIRS})
        target_link_libraries(spicelab_core PUBLIC ${KLU_LIBRARIES})
    endif()
endif()

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
        tests/test_cli.cpp
        tests/test_mvp2_validation.cpp
        tests/test_advanced_solver.cpp
        tests/test_convergence_aids.cpp
        tests/test_ac_analysis.cpp
        tests/test_parallel.cpp
    )

    target_link_libraries(spicelab_tests
        PRIVATE
            spicelab::core
            Catch2::Catch2WithMain
    )

    # Ensure the CLI executable is built before running integration tests that invoke it
    if(TARGET spicelab)
        add_dependencies(spicelab_tests spicelab)
    endif()

    catch_discover_tests(spicelab_tests)
endif()
