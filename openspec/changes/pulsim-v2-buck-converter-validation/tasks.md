## Phase 1 — Buck converter test scaffold (~0.25 days)

### 1.1 `tests/v2/layer5_v1/test_integration_buck.cpp`
- [ ] 1.1.1 `struct BuckConverter` helper that owns the graph,
      pool, and built cache. Constructor takes `dt` for the cache
      build.
- [ ] 1.1.2 Circuit nodes: `v_in`, `v_sw`, `v_out` (plus GND
      sentinel).
- [ ] 1.1.3 Branches (in this exact order to lock the
      branch_var_id layout):
      - 0: V_in source on (v_in, GND)
      - 1: Q1 switch on (v_in, v_sw)
      - 2: Q2 switch on (v_sw, GND)
      - 3: L on (v_sw, v_out)
      - 4: R_load on (v_out, GND)
      - 5: C on (v_out, GND)
- [ ] 1.1.4 DevicePool registrations:
      - V_in = 12 V, switches with `g_on = 1e3`, `g_off = 1e-9`,
        L = 100 µH, R = 10 Ω (G = 0.1), C = 100 µF.
- [ ] 1.1.5 Cache `build(dt = 100 ns)` produces exactly 4
      segments (2-switch combinatorial space).

## Phase 2 — Complementary PWM schedule (~0.1 days)

### 2.1 PWM schedule lambda inside the test
- [ ] 2.1.1 Closure capturing `f_sw = 100 kHz` and `D = 0.5`.
- [ ] 2.1.2 Returns `mask` with bit 0 (Q1) ON when phase < D·T,
      bit 1 (Q2) ON when phase ≥ D·T. Always exactly one ON
      (no dead-time, no shoot-through).

## Phase 3 — Integration run (~0.1 days)

### 3.1 SimulationOptions + run_transient call
- [ ] 3.1.1 `t_start = 0`, `t_end = 1 ms`, `dt = 100 ns` →
      10001 samples (one PWM period = 100 samples).
- [ ] 3.1.2 Call `run_transient(cache, graph, pool, opts,
      pwm_schedule)` with no `b_extra_fn`.
- [ ] 3.1.3 Wall-clock measurement around the call. Assert
      `< 5 s` (very generous; Debug builds are ~10× slower than
      Release).

## Phase 4 — Steady-state metrics (~0.25 days)

### 4.1 Last 10 PWM periods analysis
- [ ] 4.1.1 Compute `t_meas_start = t_end - 10 * T_sw`. Compute
      `k_start` from `t_meas_start`.
- [ ] 4.1.2 Loop k = k_start..N-1 and compute:
      - `mean_v_out`, `mean_i_L`
      - `min_v_out`, `max_v_out`, `delta_v_out`
      - `min_i_L`, `max_i_L`, `delta_i_L`

### 4.2 Validation REQUIRE blocks
- [ ] 4.2.1 `mean_v_out ≈ V_in · D = 6.0 V` within 5 %.
- [ ] 4.2.2 `mean_i_L ≈ V_out / R_load = 0.6 A` within 5 %.
- [ ] 4.2.3 `delta_i_L ≈ V_in · D · (1-D) / (L · f_sw) = 0.3 A`
      within 15 %.
- [ ] 4.2.4 `delta_v_out ≈ delta_i_L / (8 · C · f_sw) = 3.75 mV`
      within 30 %.

### 4.3 Helpful INFO output
- [ ] 4.3.1 Print V_out: mean, min, max, ripple, analytical
      target, relative error.
- [ ] 4.3.2 Print I_L: same pattern.
- [ ] 4.3.3 Print wall-clock time.
- [ ] 4.3.4 Print num_segments (sanity: 4 for 2 switches).

## Phase 5 — Documentation (~0.1 days)

### 5.1 `docs/pulsim-v2/layer5-v1.5-buck-validation.md`
- [ ] 5.1.1 Section "What this test proves" — every layer
      working in concert.
- [ ] 5.1.2 Section "Topology + parameter choices" — why
      complementary PWM, why these L/C values.
- [ ] 5.1.3 Section "Analytical expectations" — the four
      formulas + their derivations (very brief, link to a
      power-electronics textbook for full derivation).
- [ ] 5.1.4 Section "Test output sample" — paste the INFO
      printout from a successful run for documentation.
- [ ] 5.1.5 Section "Limitations" — open-loop, no diode, no
      DCM, no comparison-with-PLECS.

## Phase 6 — CMake + commit (~0.1 days)

### 6.1 CMake target update
- [ ] 6.1.1 Add `test_integration_buck.cpp` to the
      `pulsim_v2_layer5_v1_tests` sources list.

### 6.2 Validation gates
- [ ] 6.2.1 `openspec validate pulsim-v2-buck-converter-validation
      --strict` MUST pass.
- [ ] 6.2.2 `pulsim_v2_layer5_v1_tests` MUST pass with the new
      test included.
- [ ] 6.2.3 All previous layer tests (0-5 V0, 4-5 V1) MUST stay
      green.
- [ ] 6.2.4 v1 `pulsim_tests` MUST stay green.

### 6.3 Commit + push
- [ ] 6.3.1 Stage all new files + the CMake edit, commit with
      descriptive message, push to remote.
