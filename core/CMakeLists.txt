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
# pulsim::v2 — bootstrap-pulsim-v2-kernel (Layer 0+)
# =============================================================================
# Header-only v2 namespace. Coexists with pulsim_core (v1); zero
# coupling. Built layer by layer:
#   Layer 0 (this commit): numeric/ + sparse/
#   Layer 1 (next OpenSpec): topology/
#   Layer 2: models/
#   ...
add_library(pulsim_v2 INTERFACE)
target_include_directories(pulsim_v2
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(pulsim_v2 INTERFACE cxx_std_23)
target_link_libraries(pulsim_v2
    INTERFACE
        Eigen3::Eigen
)
add_library(pulsim::v2 ALIAS pulsim_v2)

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
        tests/test_switching_mode.cpp
        tests/test_pwl_state_space.cpp
        tests/test_ad_scalar.cpp
        tests/test_ad_diode_stamp.cpp
        tests/test_ad_mosfet_stamp.cpp
        tests/test_ad_igbt_stamp.cpp
        tests/test_ad_vcswitch_stamp.cpp
        tests/test_ad_linear_opt_out.cpp
        tests/test_ad_validate.cpp
        tests/test_ad_stamp_perf.cpp
        tests/test_linear_cache_symbolic_reuse.cpp
        tests/test_magnetic_phase1_primitives.cpp
        tests/test_magnetic_phase2_saturable_inductor.cpp
        tests/test_magnetic_phase3_saturable_transformer.cpp
        tests/test_magnetic_phase4_hysteresis.cpp
        tests/test_magnetic_phase5_catalog.cpp
        tests/test_magnetic_phase6_validation.cpp
        tests/test_catalog_phase1_to_4.cpp
        tests/test_catalog_phase7_yaml_loader.cpp
        tests/test_catalog_phase8_validation.cpp
        tests/test_motor_models.cpp
        tests/test_grid_library.cpp
        tests/test_robustness_profile.cpp
        tests/test_preset.cpp
        tests/test_armijo_line_search.cpp
        tests/test_homotopy_dc.cpp
        tests/test_iterative_refinement.cpp
        tests/test_solver_kind_collapse.cpp
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
        tests/test_model_regularization.cpp
        tests/test_v1_kernel.cpp
        tests/test_stress_simulation.cpp
        tests/test_pwl_segment_primary.cpp
        tests/test_pwl_speedup_benchmark.cpp
        tests/test_consistent_initialization.cpp
        tests/test_linear_cache_numeric_lru.cpp
        tests/test_linear_cache_phase6_benchmarks.cpp
        tests/test_frequency_analysis_phase1.cpp
        tests/test_frequency_analysis_phase2.cpp
        tests/test_frequency_analysis_phase3.cpp
        tests/test_frequency_analysis_phase4.cpp
        tests/test_frequency_analysis_phase7_yaml.cpp
        tests/test_frequency_analysis_phase8_validation.cpp
        tests/test_frequency_analysis_phase9_perf.cpp
        tests/test_converter_templates.cpp
        tests/test_three_phase_source.cpp
        tests/test_dc_motor_device.cpp
        tests/test_mechanical_device.cpp
        tests/test_bldc_motor_device.cpp
        tests/test_induction_motor_device.cpp
        tests/test_pmsm_foc_device.cpp
        tests/test_three_phase_source_b1.cpp
        tests/test_yaml_motor_parser_b1.cpp
        tests/test_bldc_benchmark_yaml.cpp
        tests/test_hysteresis_inductor_device.cpp
        tests/test_compressor_load.cpp
        tests/test_single_phase_induction_motor.cpp
        tests/test_refrigerants.cpp
        tests/test_yaml_compressor_load.cpp
        tests/test_yaml_preset.cpp
        tests/test_simultaneous_events.cpp
        tests/test_mmc_arm_template.cpp
        tests/test_mmc_balancer.cpp
        tests/test_three_phase_rl_load.cpp
        tests/test_pmsm_dynamic.cpp
        tests/test_three_phase_vsi.cpp
        tests/test_diode_loss_thermal.cpp
        tests/test_switch_loss_thermal.cpp
        tests/test_passives_loss_thermal.cpp
        tests/test_switching_phase4.cpp
        tests/test_component_introspection.cpp
        tests/test_mosfet_gate_anchor.cpp
        tests/test_motor_flux_integration.cpp
        tests/test_igbt_vce_sat_stamp.cpp
        tests/test_body_diode.cpp
        tests/test_motor_winding_thermal.cpp
        tests/test_electrothermal_loop_closure.cpp
        tests/test_yaml_passive_thermal.cpp
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
    # pulsim_v2_layer0_tests — bootstrap-pulsim-v2-kernel
    # =============================================================================
    # Layer 0 tests live in a separate binary so v2 can be developed
    # bottom-up with zero v1 coupling. Each future layer (1, 2, 3,
    # ...) gets its own test executable.
    add_executable(pulsim_v2_layer0_tests
        tests/v2/layer0/test_main.cpp
        tests/v2/layer0/test_numeric_types.cpp
        tests/v2/layer0/test_sparse_matrix.cpp
        tests/v2/layer0/test_sparse_solver.cpp
    )
    target_link_libraries(pulsim_v2_layer0_tests
        PRIVATE
            pulsim::v2
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_v2_layer0_tests)
    endif()
    catch_discover_tests(pulsim_v2_layer0_tests)

    # =============================================================================
    # pulsim_v2_layer1_tests — pulsim-v2-topology-and-switch-enumeration
    # =============================================================================
    # Layer 1 = topology (graph + switch combinatorics). Depends only
    # on Layer 0 (numeric types) — compile-time enforced via header
    # includes.
    add_executable(pulsim_v2_layer1_tests
        tests/v2/layer1/test_main.cpp
        tests/v2/layer1/test_graph.cpp
        tests/v2/layer1/test_switch_state.cpp
        tests/v2/layer1/test_enumerator.cpp
        tests/v2/layer1/test_node_equivalence.cpp
        tests/v2/layer1/test_key.cpp
    )
    target_link_libraries(pulsim_v2_layer1_tests
        PRIVATE
            pulsim::v2
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_v2_layer1_tests)
    endif()
    catch_discover_tests(pulsim_v2_layer1_tests)

    # =============================================================================
    # pulsim_v2_layer2_tests — pulsim-v2-device-models-ad-driven
    # =============================================================================
    # Layer 2 = forward-mode AD scalar + DeviceModel concept + first
    # three device models (Resistor, VoltageSource, IdealDiode).
    # Depends only on Layers 0 + 1.
    add_executable(pulsim_v2_layer2_tests
        tests/v2/layer2/test_main.cpp
        tests/v2/layer2/test_ad_scalar.cpp
        tests/v2/layer2/test_device_model_concept.cpp
        tests/v2/layer2/test_resistor.cpp
        tests/v2/layer2/test_voltage_source.cpp
        tests/v2/layer2/test_ideal_diode.cpp
    )
    target_link_libraries(pulsim_v2_layer2_tests
        PRIVATE
            pulsim::v2
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_v2_layer2_tests)
    endif()
    catch_discover_tests(pulsim_v2_layer2_tests)

    # =============================================================================
    # pulsim_v2_layer3_tests — pulsim-v2-generic-stamping-pipeline
    # =============================================================================
    # Layer 3 = generic stamping pipeline. Depends on Layers 0 + 1 + 2.
    # ONE templated stamper for every DeviceModel — the v1 four-stamp
    # duplication is structurally impossible here.
    add_executable(pulsim_v2_layer3_tests
        tests/v2/layer3/test_main.cpp
        tests/v2/layer3/test_branch_coord.cpp
        tests/v2/layer3/test_stamp_device.cpp
        tests/v2/layer3/test_stamp_voltage_source.cpp
        tests/v2/layer3/test_stamp_switch.cpp
        tests/v2/layer3/test_integration_vrgnd.cpp
    )
    target_link_libraries(pulsim_v2_layer3_tests
        PRIVATE
            pulsim::v2
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_v2_layer3_tests)
    endif()
    catch_discover_tests(pulsim_v2_layer3_tests)

    # =============================================================================
    # pulsim_v2_layer4_tests — pulsim-v2-pwl-state-space-cache
    # =============================================================================
    # Layer 4 = the PLECS-killer PWL state-space cache. Depends on
    # all lower layers (0/1/2/3).
    add_executable(pulsim_v2_layer4_tests
        tests/v2/layer4/test_main.cpp
        tests/v2/layer4/test_device_pool.cpp
        tests/v2/layer4/test_segment.cpp
        tests/v2/layer4/test_assemble.cpp
        tests/v2/layer4/test_cache.cpp
        tests/v2/layer4/test_integration_chopper.cpp
    )
    target_link_libraries(pulsim_v2_layer4_tests
        PRIVATE
            pulsim::v2
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_v2_layer4_tests)
    endif()
    catch_discover_tests(pulsim_v2_layer4_tests)

    # =============================================================================
    # pulsim_v2_layer5_tests — pulsim-v2-solver-and-events
    # =============================================================================
    # Layer 5 = the time-stepping orchestrator. Wraps Layer 4's
    # cache.solve(...) in a fixed-dt loop with user-supplied switch
    # schedule + optional b_extra(t) callback. Depends on all lower
    # layers (0/1/2/3/4).
    add_executable(pulsim_v2_layer5_tests
        tests/v2/layer5/test_main.cpp
        tests/v2/layer5/test_options.cpp
        tests/v2/layer5/test_result.cpp
        tests/v2/layer5/test_run_transient.cpp
        tests/v2/layer5/test_integration_chopper_pwm.cpp
    )
    target_link_libraries(pulsim_v2_layer5_tests
        PRIVATE
            pulsim::v2
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_v2_layer5_tests)
    endif()
    catch_discover_tests(pulsim_v2_layer5_tests)

    # =============================================================================
    # pulsim_v2_layer4_v1_tests — pulsim-v2-trapezoidal-companion (Layer 4 V1)
    # =============================================================================
    # Capacitor + Inductor with the trap companion model. Cache
    # becomes dt-aware. Static-only V0 behaviour preserved via the
    # dt=0 path.
    add_executable(pulsim_v2_layer4_v1_tests
        tests/v2/layer4_v1/test_main.cpp
        tests/v2/layer4_v1/test_capacitor.cpp
        tests/v2/layer4_v1/test_inductor.cpp
        tests/v2/layer4_v1/test_device_pool_dynamic.cpp
        tests/v2/layer4_v1/test_stamp_companion.cpp
        tests/v2/layer4_v1/test_dt_aware_cache.cpp
    )
    target_link_libraries(pulsim_v2_layer4_v1_tests
        PRIVATE
            pulsim::v2
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_v2_layer4_v1_tests)
    endif()
    catch_discover_tests(pulsim_v2_layer4_v1_tests)

    # =============================================================================
    # pulsim_v2_layer5_v1_tests — pulsim-v2-trapezoidal-companion (Layer 5 V1)
    # =============================================================================
    # HistoryState + run_transient extended for dynamic devices.
    # Integration: RC charging, RL ramp, RLC ringdown.
    add_executable(pulsim_v2_layer5_v1_tests
        tests/v2/layer5_v1/test_main.cpp
        tests/v2/layer5_v1/test_history_state.cpp
        tests/v2/layer5_v1/test_run_transient_history.cpp
        tests/v2/layer5_v1/test_integration_rc.cpp
        tests/v2/layer5_v1/test_integration_rl.cpp
        tests/v2/layer5_v1/test_integration_rlc.cpp
        tests/v2/layer5_v1/test_integration_buck.cpp
    )
    target_link_libraries(pulsim_v2_layer5_v1_tests
        PRIVATE
            pulsim::v2
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_v2_layer5_v1_tests)
    endif()
    catch_discover_tests(pulsim_v2_layer5_v1_tests)

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
