# Pulsim core — v2-only since 1.0.0
#
# The v2 API is header-only and exposed as a single INTERFACE
# library, ``pulsim::core``. The legacy v1 surface (``pulsim::core``
# INTERFACE + ``pulsim::pulsim`` STATIC) was removed in Phase 3c.3
# of the v1 retirement, together with ~90 v1 test sources and the
# six ``core/src/v1/*.cpp`` translation units.

# =============================================================================
# pulsim::core — header-only kernel
# =============================================================================
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
        yaml-cpp::yaml-cpp
)

add_library(pulsim::core ALIAS pulsim_core)

# =============================================================================
# Tests
# =============================================================================
if(PULSIM_BUILD_TESTS)
    # =============================================================================
    # pulsim_core_dsed_tests — add-path-based-dsed-engine (Gate 2)
    # =============================================================================
    # Path-Based Event-Driven engine unit tests. Header-only modules
    # in core/include/pulsim/dsed/. Depends only on Layer 0 (numeric
    # types) — no Pulsim-cache integration yet (that lands at Gate 4).
    add_executable(pulsim_core_dsed_tests
        tests/dsed/test_main.cpp
        tests/dsed/test_step_controller.cpp
        tests/dsed/test_event_predictor.cpp
        tests/dsed/test_rk45.cpp
        tests/dsed/test_buck_ccm.cpp        # Phase 2.B-1 — C++ end-to-end on buck CCM
        tests/dsed/test_diode_events.cpp    # Gate 3 — buck DCM + body-diode commutation
        tests/dsed/test_bdf2.cpp            # Gate 4 — BDF2 + stiffness detector
        tests/dsed/test_bdf2_scheduler.cpp  # Gate 4.C — end-to-end BDF2 PED scheduler
        tests/dsed/test_mmc_thevenin_arm.cpp
        tests/dsed/test_trbdf2_transient.cpp
        tests/dsed/test_trbdf2_stage_coeffs.cpp
        tests/dsed/test_exact_lti.cpp
        tests/dsed/test_event_projection.cpp
        tests/dsed/test_scheduler_auto.cpp  # Gate 4.D — auto-dispatch RK45↔BDF2 per mode
    )
    target_link_libraries(pulsim_core_dsed_tests
        PRIVATE
            pulsim::core
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_core_dsed_tests)
    endif()
    catch_discover_tests(pulsim_core_dsed_tests)

    # =============================================================================
    # pulsim_core_layer0_tests — bootstrap-pulsim-v2-kernel
    # =============================================================================
    # Layer 0 tests live in a separate binary so v2 can be developed
    # bottom-up with zero v1 coupling. Each future layer (1, 2, 3,
    # ...) gets its own test executable.
    add_executable(pulsim_core_layer0_tests
        tests/layer0/test_main.cpp
        tests/layer0/test_numeric_types.cpp
        tests/layer0/test_logistic.cpp     # v2.0 Phase 2 — overflow-free sigmoid
        tests/layer0/test_sparse_matrix.cpp
        tests/layer0/test_sparse_solver.cpp
        tests/layer0/test_pulsim_lu_solver.cpp
        tests/layer0/test_pulsim_lu_scaling.cpp           # Layer 0 V8 — in-house sparse LU (Real)
        tests/layer0/test_pulsim_lu_solver_complex.cpp   # Layer 0 v1.4.0 — complex specialisation
    )
    target_link_libraries(pulsim_core_layer0_tests
        PRIVATE
            pulsim::core
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_core_layer0_tests)
    endif()
    catch_discover_tests(pulsim_core_layer0_tests)

    # =============================================================================
    # pulsim_core_layer1_tests — pulsim-v2-topology-and-switch-enumeration
    # =============================================================================
    # Layer 1 = topology (graph + switch combinatorics). Depends only
    # on Layer 0 (numeric types) — compile-time enforced via header
    # includes.
    add_executable(pulsim_core_layer1_tests
        tests/layer1/test_main.cpp
        tests/layer1/test_graph.cpp
        tests/layer1/test_switch_state.cpp
        tests/layer1/test_enumerator.cpp
        tests/layer1/test_node_equivalence.cpp
        tests/layer1/test_key.cpp
        tests/layer1/test_identifiers.cpp
    )
    target_link_libraries(pulsim_core_layer1_tests
        PRIVATE
            pulsim::core
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_core_layer1_tests)
    endif()
    catch_discover_tests(pulsim_core_layer1_tests)

    # =============================================================================
    # pulsim_core_layer2_tests — pulsim-v2-device-models-ad-driven
    # =============================================================================
    # Layer 2 = forward-mode AD scalar + DeviceModel concept + first
    # three device models (Resistor, VoltageSource, IdealDiode).
    # Depends only on Layers 0 + 1.
    add_executable(pulsim_core_layer2_tests
        tests/layer2/test_main.cpp
        tests/layer2/test_ad_scalar.cpp
        tests/layer2/test_device_model_concept.cpp
        tests/layer2/test_resistor.cpp
        tests/layer2/test_voltage_source.cpp
        tests/layer2/test_ideal_diode.cpp
        tests/layer2/test_current_source.cpp   # Layer 2 V3
        tests/layer2/test_pwm_voltage_source.cpp  # Layer 2 V4
        tests/layer2/test_sine_voltage_source.cpp # Layer 2 V11
        tests/layer2/test_pulse_voltage_source.cpp # Layer 2 V12
        tests/layer2/test_mosfet_level1.cpp        # Layer 2 V13
        tests/layer2/test_shockley_diode.cpp       # Phase 4 C.1
        tests/layer2/test_igbt_level1.cpp          # Layer 2 V14
        tests/layer2/test_vcvs.cpp                 # Layer 2 V15
        tests/layer2/test_saturable_inductor.cpp   # Layer 2 V16
        tests/layer2/test_multi_winding_transformer.cpp  # Layer 2 V16
        tests/layer2/test_ideal_transformer.cpp         # Phase 4 C.4
    )
    target_link_libraries(pulsim_core_layer2_tests
        PRIVATE
            pulsim::core
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_core_layer2_tests)
    endif()
    catch_discover_tests(pulsim_core_layer2_tests)

    # =============================================================================
    # pulsim_core_analysis_tests — pulsim-v2-core-loss (Layer 11 V18)
    # =============================================================================
    # Post-process analysis modules: Steinmetz core-loss estimator,
    # future AC sweep / Bode plot machinery.
    add_executable(pulsim_core_analysis_tests
        tests/analysis/test_main.cpp
        tests/analysis/test_core_loss.cpp
        tests/analysis/test_mna_sweep.cpp   # v1.4.0 — in-house complex solver integration
    )
    target_link_libraries(pulsim_core_analysis_tests
        PRIVATE
            pulsim::core
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_core_analysis_tests)
    endif()
    catch_discover_tests(pulsim_core_analysis_tests)

    # =============================================================================
    # pulsim_core_sources_tests — pulsim-v2-source-helpers (Layer 2 V5+)
    # =============================================================================
    # Standalone target for `sources/` helpers — currently
    # `make_pwm_switch_fn` (Layer 2 V5). These are pure
    # builders that produce SwitchScheduleFn / b_extra
    # callbacks; they live outside the device-model layer
    # because they don't introduce branches.
    add_executable(pulsim_core_sources_tests
        tests/sources/test_main.cpp
        tests/sources/test_pwm_switch_fn.cpp
        tests/sources/test_dead_time_pwm_pair_fn.cpp  # Layer 2 V6
        tests/sources/test_spwm_pair_fn.cpp           # Layer 2 V7
        tests/sources/test_three_phase_spwm_fn.cpp    # Layer 2 V8
        tests/sources/test_phase_shift_full_bridge_fn.cpp  # V9
        tests/sources/test_combined_switch_fn.cpp     # Layer 2 V10
    )
    target_link_libraries(pulsim_core_sources_tests
        PRIVATE
            pulsim::core
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_core_sources_tests)
    endif()
    catch_discover_tests(pulsim_core_sources_tests)

    # =============================================================================
    # pulsim_core_layer3_tests — pulsim-v2-generic-stamping-pipeline
    # =============================================================================
    # Layer 3 = generic stamping pipeline. Depends on Layers 0 + 1 + 2.
    # ONE templated stamper for every DeviceModel — the v1 four-stamp
    # duplication is structurally impossible here.
    add_executable(pulsim_core_layer3_tests
        tests/layer3/test_main.cpp
        tests/layer3/test_branch_coord.cpp
        tests/layer3/test_stamp_device.cpp
        tests/layer3/test_stamp_voltage_source.cpp
        tests/layer3/test_stamp_switch.cpp
        tests/layer3/test_integration_vrgnd.cpp
    )
    target_link_libraries(pulsim_core_layer3_tests
        PRIVATE
            pulsim::core
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_core_layer3_tests)
    endif()
    catch_discover_tests(pulsim_core_layer3_tests)

    # =============================================================================
    # pulsim_core_layer4_tests — pulsim-v2-pwl-state-space-cache
    # =============================================================================
    # Layer 4 = the PLECS-killer PWL state-space cache. Depends on
    # all lower layers (0/1/2/3).
    add_executable(pulsim_core_layer4_tests
        tests/layer4/test_main.cpp
        tests/layer4/test_device_pool.cpp
        tests/layer4/test_segment.cpp
        tests/layer4/test_assemble.cpp
        tests/layer4/test_row_names.cpp   # v2.0 Phase 1 — named diagnostics
        tests/layer4/test_preflight.cpp    # v2.0 Phase 2 — preflight
        tests/layer4/test_gmin.cpp         # v2.0 Phase 2 — gmin + DC cascade
        tests/layer4/test_cache.cpp
        tests/layer4/test_integration_chopper.cpp
        tests/layer4/test_pwl_cache_rank1.cpp       # Layer 4 V8 — rank-1 fast path
        tests/layer4/test_pwl_cache_parametric.cpp  # Layer 4 v1.4.0 — parametric refactor
    )
    target_link_libraries(pulsim_core_layer4_tests
        PRIVATE
            pulsim::core
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_core_layer4_tests)
    endif()
    catch_discover_tests(pulsim_core_layer4_tests)

    # =============================================================================
    # pulsim_core_layer5_tests — pulsim-v2-solver-and-events
    # =============================================================================
    # Layer 5 = the time-stepping orchestrator. Wraps Layer 4's
    # cache.solve(...) in a fixed-dt loop with user-supplied switch
    # schedule + optional b_extra(t) callback. Depends on all lower
    # layers (0/1/2/3/4).
    add_executable(pulsim_core_layer5_tests
        tests/layer5/test_main.cpp
        tests/layer5/test_dt_retry.cpp     # v2.0 Phase 2 — local dt reduction
        tests/layer5/test_options.cpp
        tests/layer5/test_result.cpp
        tests/layer5/test_state_trajectory.cpp  # v2.0 Phase 1 — contiguous storage
        tests/layer5/test_run_transient.cpp
        tests/layer5/test_integration_chopper_pwm.cpp
    )
    target_link_libraries(pulsim_core_layer5_tests
        PRIVATE
            pulsim::core
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_core_layer5_tests)
    endif()
    catch_discover_tests(pulsim_core_layer5_tests)

    # =============================================================================
    # pulsim_core_layer4_v1_tests — pulsim-v2-trapezoidal-companion (Layer 4 V1)
    # =============================================================================
    # Capacitor + Inductor with the trap companion model. Cache
    # becomes dt-aware. Static-only V0 behaviour preserved via the
    # dt=0 path.
    add_executable(pulsim_core_layer4_v1_tests
        tests/layer4_v1/test_main.cpp
        tests/layer4_v1/test_capacitor.cpp
        tests/layer4_v1/test_inductor.cpp
        tests/layer4_v1/test_device_pool_dynamic.cpp
        tests/layer4_v1/test_stamp_companion.cpp
        tests/layer4_v1/test_dt_aware_cache.cpp
        tests/layer4_v1/test_lazy_cache.cpp      # V6 lazy
        tests/layer4_v1/test_multi_dt_cache.cpp  # V7 multi-dt
    )
    target_link_libraries(pulsim_core_layer4_v1_tests
        PRIVATE
            pulsim::core
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_core_layer4_v1_tests)
    endif()
    catch_discover_tests(pulsim_core_layer4_v1_tests)

    # =============================================================================
    # pulsim_core_perf_tests — v2.0 Phase 1 zero-allocation lock-in
    # =============================================================================
    # SEPARATE executable on purpose: it replaces global operator
    # new/delete with counting versions and redefines eigen_assert +
    # EIGEN_RUNTIME_NO_MALLOC (which must precede every Eigen include
    # in the TU) to catch Eigen-side heap traffic in release builds.
    add_executable(pulsim_core_perf_tests
        tests/perf/test_zero_alloc.cpp
    )
    target_link_libraries(pulsim_core_perf_tests
        PRIVATE
            pulsim::core
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_core_perf_tests)
    endif()
    catch_discover_tests(pulsim_core_perf_tests)

    # =============================================================================
    # pulsim_core_layer5_v1_tests — pulsim-v2-trapezoidal-companion (Layer 5 V1)
    # =============================================================================
    # HistoryState + run_transient extended for dynamic devices.
    # Integration: RC charging, RL ramp, RLC ringdown.
    add_executable(pulsim_core_layer5_v1_tests
        tests/layer5_v1/test_main.cpp
        tests/layer5_v1/test_history_state.cpp
        tests/layer5_v1/test_run_transient_history.cpp
        tests/layer5_v1/test_integration_rc.cpp
        tests/layer5_v1/test_integration_rl.cpp
        tests/layer5_v1/test_integration_rlc.cpp
        tests/layer5_v1/test_integration_buck.cpp
        tests/layer5_v1/test_transformer.cpp          # Layer 2 V2
    )
    target_link_libraries(pulsim_core_layer5_v1_tests
        PRIVATE
            pulsim::core
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_core_layer5_v1_tests)
    endif()
    catch_discover_tests(pulsim_core_layer5_v1_tests)

    # =============================================================================
    # pulsim_core_layer5_v2_tests — pulsim-v2-ideal-diode-auto-commutation
    # =============================================================================
    # SwitchedDiode + DiodeEventState + diode-aware run_transient.
    # Integration: half-wave rectifier (single diode) and boost
    # converter (1 controlled switch + 1 auto-commutating diode).
    add_executable(pulsim_core_layer5_v2_tests
        tests/layer5_v2/test_main.cpp
        tests/layer5_v2/test_switched_diode.cpp
        tests/layer5_v2/test_diode_event_state.cpp
        tests/layer5_v2/test_integration_half_wave_rectifier.cpp
        tests/layer5_v2/test_integration_boost.cpp   # V2.1 fix
        tests/layer5_v2/test_commutation_events.cpp  # V2.2 substep
    )
    target_link_libraries(pulsim_core_layer5_v2_tests
        PRIVATE
            pulsim::core
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_core_layer5_v2_tests)
    endif()
    catch_discover_tests(pulsim_core_layer5_v2_tests)

    # =============================================================================
    # pulsim_core_layer5_v3_tests — pulsim-v2-substep-state-correction
    # =============================================================================
    # V3 ships sub-step state correction on the dynamic path:
    # when an event is detected mid-step, retroactively split
    # the step into two `solve_at` calls (pre-event + post-event)
    # at the linearly-interpolated commutation time.
    add_executable(pulsim_core_layer5_v3_tests
        tests/layer5_v3/test_main.cpp
        tests/layer5_v3/test_snapshot_restore.cpp
        tests/layer5_v3/test_substep_correction.cpp
    )
    target_link_libraries(pulsim_core_layer5_v3_tests
        PRIVATE
            pulsim::core
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_core_layer5_v3_tests)
    endif()
    catch_discover_tests(pulsim_core_layer5_v3_tests)

    # =============================================================================
    # pulsim_core_builder_tests — pulsim-v2-builder-api
    # =============================================================================
    # Layer 6 V0: high-level CircuitBuilder that hides
    # `Graph + DevicePool` and exposes string node names +
    # SI-unit parameter values (ohms, farads, henries).
    add_executable(pulsim_core_builder_tests
        tests/builder/test_main.cpp
        tests/builder/test_circuit_builder.cpp
    )
    target_link_libraries(pulsim_core_builder_tests
        PRIVATE
            pulsim::core
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_core_builder_tests)
    endif()
    catch_discover_tests(pulsim_core_builder_tests)

    # =============================================================================
    # pulsim_core_yaml_tests — pulsim-v2-yaml-parser (Layer 8)
    # =============================================================================
    # YAML circuit loader: parses a YAML file into a
    # CircuitBuilder + SimulationOptions.
    add_executable(pulsim_core_yaml_tests
        tests/yaml/test_main.cpp
        tests/yaml/test_yaml_loader.cpp
    )
    target_link_libraries(pulsim_core_yaml_tests
        PRIVATE
            pulsim::core
            yaml-cpp::yaml-cpp
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_core_yaml_tests)
    endif()
    catch_discover_tests(pulsim_core_yaml_tests)

    # =============================================================================
    # pulsim_core_showcase_tests — pulsim-v2-smps-showcase (Layer 9)
    # =============================================================================
    # End-to-end SMPS showcase: load YAML, drive PWM switch_fn,
    # verify steady-state V_out matches analytical buck math.
    add_executable(pulsim_core_showcase_tests
        tests/showcases/test_main.cpp
        tests/showcases/test_buck_open_loop.cpp
        tests/showcases/test_flyback_open_loop.cpp           # V5
        tests/showcases/test_three_phase_inverter_open_loop.cpp  # V8
        tests/showcases/test_phase_shift_full_bridge_open_loop.cpp  # V9
        tests/showcases/test_three_phase_diode_rectifier_open_loop.cpp  # V11
        tests/showcases/test_rlc_step_response.cpp           # V12
        tests/showcases/test_common_source_amplifier.cpp     # V13
        tests/showcases/test_pwm_chopper_realistic_mosfet.cpp # V13 SMPS
        tests/showcases/test_boost_realistic_igbt.cpp         # V14 boost (IGBT)
        tests/showcases/test_boost_realistic_mosfet_v14.cpp   # V14 boost (MOSFET)
        tests/showcases/test_ldo_with_opamp.cpp               # V15 LDO + op-amp
        tests/showcases/test_boost_saturable_inductor.cpp     # V18 saturable boost
    )
    target_link_libraries(pulsim_core_showcase_tests
        PRIVATE
            pulsim::core
            yaml-cpp::yaml-cpp
            Catch2::Catch2WithMain
    )
    # Expose the examples dir to the test binary via
    # an env var so test runs from any working directory.
    target_compile_definitions(pulsim_core_showcase_tests
        PRIVATE
            PULSIM_REPO_EXAMPLES_DIR="${CMAKE_SOURCE_DIR}/examples"
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_core_showcase_tests)
    endif()
    catch_discover_tests(pulsim_core_showcase_tests
        PROPERTIES
            ENVIRONMENT "PULSIM_EXAMPLES_DIR=${CMAKE_SOURCE_DIR}/examples"
    )

    # =============================================================================
    # pulsim_core_layer4_v2_tests — pulsim-v2-dc-operating-point
    # =============================================================================
    # DC operating-point computation + HistoryState/DiodeEventState
    # seeding. Layer 5 V3 run_transient extended with
    # start_from_dc_op flag.
    add_executable(pulsim_core_layer4_v2_tests
        tests/layer4_v2/test_main.cpp
        tests/layer4_v2/test_dc_assemble.cpp
        tests/layer4_v2/test_seeding.cpp
        tests/layer4_v2/test_integration_rc_dc_op.cpp
    )
    target_link_libraries(pulsim_core_layer4_v2_tests
        PRIVATE
            pulsim::core
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_core_layer4_v2_tests)
    endif()
    catch_discover_tests(pulsim_core_layer4_v2_tests)

    # =============================================================================
    # pulsim_core_layer4_v3_tests — pulsim-v2-nonlinear-segment-newton
    # =============================================================================
    # Newton iteration on top of the cached LINEAR factor for
    # circuits with BranchKind::Nonlinear devices (real diodes,
    # future MOSFETs, etc.).
    add_executable(pulsim_core_layer4_v3_tests
        tests/layer4_v3/test_main.cpp
        tests/layer4_v3/test_nonlinear_solve.cpp
        tests/layer4_v3/test_diode_load_line.cpp
        tests/layer4_v3/test_lti_recovery.cpp
        tests/layer4_v3/test_lti_state_space.cpp   # Bridge.3 — DSED bridge
    )
    target_link_libraries(pulsim_core_layer4_v3_tests
        PRIVATE
            pulsim::core
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_core_layer4_v3_tests)
    endif()
    catch_discover_tests(pulsim_core_layer4_v3_tests)

    # =============================================================================
    # pulsim_core_layer5_v4_tests — pulsim-v2-newton-in-run-transient
    # =============================================================================
    # Newton iteration wired into the run_transient loop.
    # Integration: half-wave rectifier with smooth-blend
    # (AD-driven) IdealDiode + warm-started Newton per step.
    add_executable(pulsim_core_layer5_v4_tests
        tests/layer5_v4/test_main.cpp
        tests/layer5_v4/test_integration_smooth_rectifier.cpp
        tests/layer5_v4/test_globalized_rectifier.cpp
        tests/layer5_v4/test_lm_rectifier.cpp        # V5 LM
        tests/layer5_v4/test_continuation_rectifier.cpp  # V8 continuation
        tests/layer5_v4/test_vf0_continuation_rectifier.cpp  # V9 V_F0 cont
        tests/layer5_v4/test_pseudo_transient_rectifier.cpp  # V10 PTC
    )
    target_link_libraries(pulsim_core_layer5_v4_tests
        PRIVATE
            pulsim::core
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_core_layer5_v4_tests)
    endif()
    catch_discover_tests(pulsim_core_layer5_v4_tests)

    # =============================================================================
    # pulsim_benchmarks — analytical-validation + perf measurement
    # =============================================================================
    # Catch2 binary that pairs each circuit (RC, RL, RLC, half-wave
    # rectifier, buck steady-state, …) with a closed-form analytical
    # reference and measures max abs / max rel error alongside the
    # cold-start cache build time + per-step solve cost.
    #
    # Deliberately NOT registered with ctest (``catch_discover_tests``)
    # so a normal ``ctest`` run stays focused on correctness tests.
    # Run manually:
    #
    #     ./build/core/pulsim_benchmarks                # all benchmarks
    #     ./build/core/pulsim_benchmarks "[rc]"          # one tag
    #     ./build/core/pulsim_benchmarks --reporter console
    #
    # Or via the convenience target:
    #
    #     cmake --build build --target bench
    add_executable(pulsim_benchmarks
        tests/benchmarks/test_main.cpp
        tests/benchmarks/test_bench_rc.cpp
        tests/benchmarks/test_bench_rl.cpp
        tests/benchmarks/test_bench_rlc.cpp
        tests/benchmarks/test_bench_half_wave.cpp
        tests/benchmarks/test_bench_buck.cpp
        tests/benchmarks/test_bench_boost.cpp
        tests/benchmarks/test_bench_pwl_rank1.cpp        # Layer 4 V8 — rank-1 fast path
        tests/benchmarks/test_bench_ac_sweep.cpp         # v1.4.0 — complex solver per-freq cost
        tests/benchmarks/test_bench_multi_bit_rank1.cpp  # v1.4.0 — multi-bit path-union routing
        tests/benchmarks/test_bench_parametric_sweep.cpp # v1.4.0 — refactor_parametric for sweeps
        tests/benchmarks/test_bench_dsed.cpp             # PED Gate 2 Phase 2.B-2 — variable-step buck CCM
    )
    target_include_directories(pulsim_benchmarks
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/benchmarks
    )
    target_link_libraries(pulsim_benchmarks
        PRIVATE
            pulsim::core
            Catch2::Catch2WithMain
    )
    if(COMMAND pulsim_apply_target_defaults)
        pulsim_apply_target_defaults(pulsim_benchmarks)
    endif()
    # Convenience: ``cmake --build build --target bench`` builds + runs.
    add_custom_target(bench
        COMMAND $<TARGET_FILE:pulsim_benchmarks>
        DEPENDS pulsim_benchmarks
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running pulsim_benchmarks (analytical + timing)…"
    )

endif()
