# Pulsim core — v2-only since 1.0.0
#
# The v2 API is header-only and exposed as a single INTERFACE
# library, ``pulsim::v2``. The legacy v1 surface (``pulsim::core``
# INTERFACE + ``pulsim::pulsim`` STATIC) was removed in Phase 3c.3
# of the v1 retirement, together with ~90 v1 test sources and the
# six ``core/src/v1/*.cpp`` translation units.

# =============================================================================
# pulsim::v2 — header-only kernel
# =============================================================================
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
        yaml-cpp::yaml-cpp
        ${PULSIM_KLU_LIBRARIES}
        ${PULSIM_HYPRE_LIBRARIES}
)
add_library(pulsim::v2 ALIAS pulsim_v2)

# =============================================================================
# Tests
# =============================================================================
if(PULSIM_BUILD_TESTS)
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
        tests/v2/layer2/test_current_source.cpp   # Layer 2 V3
        tests/v2/layer2/test_pwm_voltage_source.cpp  # Layer 2 V4
        tests/v2/layer2/test_sine_voltage_source.cpp # Layer 2 V11
        tests/v2/layer2/test_pulse_voltage_source.cpp # Layer 2 V12
        tests/v2/layer2/test_mosfet_level1.cpp        # Layer 2 V13
        tests/v2/layer2/test_igbt_level1.cpp          # Layer 2 V14
        tests/v2/layer2/test_vcvs.cpp                 # Layer 2 V15
        tests/v2/layer2/test_saturable_inductor.cpp   # Layer 2 V16
        tests/v2/layer2/test_multi_winding_transformer.cpp  # Layer 2 V16
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
    # pulsim_v2_analysis_tests — pulsim-v2-core-loss (Layer 11 V18)
    # =============================================================================
    # Post-process analysis modules: Steinmetz core-loss estimator,
    # future AC sweep / Bode plot machinery.
    add_executable(pulsim_v2_analysis_tests
        tests/v2/analysis/test_main.cpp
        tests/v2/analysis/test_core_loss.cpp
    )
    target_link_libraries(pulsim_v2_analysis_tests
        PRIVATE
            pulsim::v2
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_v2_analysis_tests)
    endif()
    catch_discover_tests(pulsim_v2_analysis_tests)

    # =============================================================================
    # pulsim_v2_sources_tests — pulsim-v2-source-helpers (Layer 2 V5+)
    # =============================================================================
    # Standalone target for `sources/` helpers — currently
    # `make_pwm_switch_fn` (Layer 2 V5). These are pure
    # builders that produce SwitchScheduleFn / b_extra
    # callbacks; they live outside the device-model layer
    # because they don't introduce branches.
    add_executable(pulsim_v2_sources_tests
        tests/v2/sources/test_main.cpp
        tests/v2/sources/test_pwm_switch_fn.cpp
        tests/v2/sources/test_dead_time_pwm_pair_fn.cpp  # Layer 2 V6
        tests/v2/sources/test_spwm_pair_fn.cpp           # Layer 2 V7
        tests/v2/sources/test_three_phase_spwm_fn.cpp    # Layer 2 V8
        tests/v2/sources/test_phase_shift_full_bridge_fn.cpp  # V9
        tests/v2/sources/test_combined_switch_fn.cpp     # Layer 2 V10
    )
    target_link_libraries(pulsim_v2_sources_tests
        PRIVATE
            pulsim::v2
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_v2_sources_tests)
    endif()
    catch_discover_tests(pulsim_v2_sources_tests)

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
        tests/v2/layer4_v1/test_lazy_cache.cpp      # V6 lazy
        tests/v2/layer4_v1/test_multi_dt_cache.cpp  # V7 multi-dt
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
        tests/v2/layer5_v1/test_transformer.cpp          # Layer 2 V2
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
    # pulsim_v2_layer5_v2_tests — pulsim-v2-ideal-diode-auto-commutation
    # =============================================================================
    # SwitchedDiode + DiodeEventState + diode-aware run_transient.
    # Integration: half-wave rectifier (single diode) and boost
    # converter (1 controlled switch + 1 auto-commutating diode).
    add_executable(pulsim_v2_layer5_v2_tests
        tests/v2/layer5_v2/test_main.cpp
        tests/v2/layer5_v2/test_switched_diode.cpp
        tests/v2/layer5_v2/test_diode_event_state.cpp
        tests/v2/layer5_v2/test_integration_half_wave_rectifier.cpp
        tests/v2/layer5_v2/test_integration_boost.cpp   # V2.1 fix
        tests/v2/layer5_v2/test_commutation_events.cpp  # V2.2 substep
    )
    target_link_libraries(pulsim_v2_layer5_v2_tests
        PRIVATE
            pulsim::v2
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_v2_layer5_v2_tests)
    endif()
    catch_discover_tests(pulsim_v2_layer5_v2_tests)

    # =============================================================================
    # pulsim_v2_layer5_v3_tests — pulsim-v2-substep-state-correction
    # =============================================================================
    # V3 ships sub-step state correction on the dynamic path:
    # when an event is detected mid-step, retroactively split
    # the step into two `solve_at` calls (pre-event + post-event)
    # at the linearly-interpolated commutation time.
    add_executable(pulsim_v2_layer5_v3_tests
        tests/v2/layer5_v3/test_main.cpp
        tests/v2/layer5_v3/test_snapshot_restore.cpp
        tests/v2/layer5_v3/test_substep_correction.cpp
    )
    target_link_libraries(pulsim_v2_layer5_v3_tests
        PRIVATE
            pulsim::v2
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_v2_layer5_v3_tests)
    endif()
    catch_discover_tests(pulsim_v2_layer5_v3_tests)

    # =============================================================================
    # pulsim_v2_builder_tests — pulsim-v2-builder-api
    # =============================================================================
    # Layer 6 V0: high-level CircuitBuilder that hides
    # `Graph + DevicePool` and exposes string node names +
    # SI-unit parameter values (ohms, farads, henries).
    add_executable(pulsim_v2_builder_tests
        tests/v2/builder/test_main.cpp
        tests/v2/builder/test_circuit_builder.cpp
    )
    target_link_libraries(pulsim_v2_builder_tests
        PRIVATE
            pulsim::v2
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_v2_builder_tests)
    endif()
    catch_discover_tests(pulsim_v2_builder_tests)

    # =============================================================================
    # pulsim_v2_yaml_tests — pulsim-v2-yaml-parser (Layer 8)
    # =============================================================================
    # YAML circuit loader: parses a YAML file into a
    # CircuitBuilder + SimulationOptions.
    add_executable(pulsim_v2_yaml_tests
        tests/v2/yaml/test_main.cpp
        tests/v2/yaml/test_yaml_loader.cpp
    )
    target_link_libraries(pulsim_v2_yaml_tests
        PRIVATE
            pulsim::v2
            yaml-cpp::yaml-cpp
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_v2_yaml_tests)
    endif()
    catch_discover_tests(pulsim_v2_yaml_tests)

    # =============================================================================
    # pulsim_v2_showcase_tests — pulsim-v2-smps-showcase (Layer 9)
    # =============================================================================
    # End-to-end SMPS showcase: load YAML, drive PWM switch_fn,
    # verify steady-state V_out matches analytical buck math.
    add_executable(pulsim_v2_showcase_tests
        tests/v2/showcases/test_main.cpp
        tests/v2/showcases/test_buck_open_loop.cpp
        tests/v2/showcases/test_flyback_open_loop.cpp           # V5
        tests/v2/showcases/test_three_phase_inverter_open_loop.cpp  # V8
        tests/v2/showcases/test_phase_shift_full_bridge_open_loop.cpp  # V9
        tests/v2/showcases/test_three_phase_diode_rectifier_open_loop.cpp  # V11
        tests/v2/showcases/test_rlc_step_response.cpp           # V12
        tests/v2/showcases/test_common_source_amplifier.cpp     # V13
        tests/v2/showcases/test_pwm_chopper_realistic_mosfet.cpp # V13 SMPS
        tests/v2/showcases/test_boost_realistic_igbt.cpp         # V14 boost (IGBT)
        tests/v2/showcases/test_boost_realistic_mosfet_v14.cpp   # V14 boost (MOSFET)
        tests/v2/showcases/test_ldo_with_opamp.cpp               # V15 LDO + op-amp
        tests/v2/showcases/test_boost_saturable_inductor.cpp     # V18 saturable boost
    )
    target_link_libraries(pulsim_v2_showcase_tests
        PRIVATE
            pulsim::v2
            yaml-cpp::yaml-cpp
            Catch2::Catch2WithMain
    )
    # Expose the examples/v2 dir to the test binary via
    # an env var so test runs from any working directory.
    target_compile_definitions(pulsim_v2_showcase_tests
        PRIVATE
            PULSIM_REPO_EXAMPLES_DIR="${CMAKE_SOURCE_DIR}/examples/v2"
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_v2_showcase_tests)
    endif()
    catch_discover_tests(pulsim_v2_showcase_tests
        PROPERTIES
            ENVIRONMENT "PULSIM_EXAMPLES_DIR=${CMAKE_SOURCE_DIR}/examples/v2"
    )

    # =============================================================================
    # pulsim_v2_layer4_v2_tests — pulsim-v2-dc-operating-point
    # =============================================================================
    # DC operating-point computation + HistoryState/DiodeEventState
    # seeding. Layer 5 V3 run_transient extended with
    # start_from_dc_op flag.
    add_executable(pulsim_v2_layer4_v2_tests
        tests/v2/layer4_v2/test_main.cpp
        tests/v2/layer4_v2/test_dc_assemble.cpp
        tests/v2/layer4_v2/test_seeding.cpp
        tests/v2/layer4_v2/test_integration_rc_dc_op.cpp
    )
    target_link_libraries(pulsim_v2_layer4_v2_tests
        PRIVATE
            pulsim::v2
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_v2_layer4_v2_tests)
    endif()
    catch_discover_tests(pulsim_v2_layer4_v2_tests)

    # =============================================================================
    # pulsim_v2_layer4_v3_tests — pulsim-v2-nonlinear-segment-newton
    # =============================================================================
    # Newton iteration on top of the cached LINEAR factor for
    # circuits with BranchKind::Nonlinear devices (real diodes,
    # future MOSFETs, etc.).
    add_executable(pulsim_v2_layer4_v3_tests
        tests/v2/layer4_v3/test_main.cpp
        tests/v2/layer4_v3/test_nonlinear_solve.cpp
        tests/v2/layer4_v3/test_diode_load_line.cpp
    )
    target_link_libraries(pulsim_v2_layer4_v3_tests
        PRIVATE
            pulsim::v2
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_v2_layer4_v3_tests)
    endif()
    catch_discover_tests(pulsim_v2_layer4_v3_tests)

    # =============================================================================
    # pulsim_v2_layer5_v4_tests — pulsim-v2-newton-in-run-transient
    # =============================================================================
    # Newton iteration wired into the run_transient loop.
    # Integration: half-wave rectifier with smooth-blend
    # (AD-driven) IdealDiode + warm-started Newton per step.
    add_executable(pulsim_v2_layer5_v4_tests
        tests/v2/layer5_v4/test_main.cpp
        tests/v2/layer5_v4/test_integration_smooth_rectifier.cpp
        tests/v2/layer5_v4/test_globalized_rectifier.cpp
        tests/v2/layer5_v4/test_lm_rectifier.cpp        # V5 LM
        tests/v2/layer5_v4/test_continuation_rectifier.cpp  # V8 continuation
        tests/v2/layer5_v4/test_vf0_continuation_rectifier.cpp  # V9 V_F0 cont
        tests/v2/layer5_v4/test_pseudo_transient_rectifier.cpp  # V10 PTC
    )
    target_link_libraries(pulsim_v2_layer5_v4_tests
        PRIVATE
            pulsim::v2
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_v2_layer5_v4_tests)
    endif()
    catch_discover_tests(pulsim_v2_layer5_v4_tests)

endif()
