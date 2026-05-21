## Phase 1 — `SimulationOptions` + `SimulationResult` (~0.25 days)

### 1.1 `solver/options.hpp`
- [ ] 1.1.1 `struct SimulationOptions`:
      - `Real t_start = 0`
      - `Real t_end = 0`
      - `Real dt = 0`
- [ ] 1.1.2 `[[nodiscard]] bool valid() const noexcept` — returns
      `dt > 0 && t_end > t_start && std::isfinite(t_start) &&
      std::isfinite(t_end) && std::isfinite(dt)`.
- [ ] 1.1.3 `[[nodiscard]] Size expected_step_count() const noexcept`
      — returns `floor((t_end - t_start) / dt) + 1` (number of
      output samples including both endpoints, defensive against
      tiny FP overshoot of the last step).

### 1.2 `solver/result.hpp`
- [ ] 1.2.1 `struct SimulationResult`:
      - `std::vector<Real> times`
      - `std::vector<Vector> states` — `states[k]` is the state
        vector at `times[k]`.
- [ ] 1.2.2 `[[nodiscard]] Size num_steps() const noexcept` —
      returns `times.size()`.
- [ ] 1.2.3 `[[nodiscard]] bool empty() const noexcept`.
- [ ] 1.2.4 `void reserve(Size n)` — pre-allocates `times` and
      `states`.

### 1.3 Tests `tests/v2/layer5/test_options.cpp`
- [ ] 1.3.1 Default-constructed options report `valid() == false`.
- [ ] 1.3.2 `{t_start=0, t_end=1, dt=0.1}` reports `valid() == true`
      and `expected_step_count() == 11` (0, 0.1, ..., 1.0).
- [ ] 1.3.3 Negative `dt` reports invalid.
- [ ] 1.3.4 `t_end <= t_start` reports invalid.
- [ ] 1.3.5 NaN inputs report invalid.

### 1.4 Tests `tests/v2/layer5/test_result.cpp`
- [ ] 1.4.1 Default-constructed result is empty.
- [ ] 1.4.2 After `reserve(100)`, `num_steps() == 0` (capacity not
      counted as size).
- [ ] 1.4.3 Manually pushing 3 (time, state) pairs reports
      `num_steps() == 3`.

## Phase 2 — `run_transient` main entry point (~0.5 days)

### 2.1 `solver/run_transient.hpp`
- [ ] 2.1.1 Type aliases:
      - `using SwitchScheduleFn = std::function<topology::SwitchStateMask(Real)>;`
      - `using BExtraFn = std::function<Vector(Real)>;`
- [ ] 2.1.2 Function signature:
      ```cpp
      SimulationResult run_transient(
          const pwl::PwlStateSpaceCache& cache,
          Size state_size,
          const SimulationOptions& opts,
          const SwitchScheduleFn& switch_fn,
          const BExtraFn& b_extra_fn = {});
      ```
- [ ] 2.1.3 Implementation:
      - Throws `std::invalid_argument` if `!opts.valid()`.
      - Throws `std::invalid_argument` if `state_size == 0`.
      - Throws `std::invalid_argument` if `!switch_fn`.
      - Pre-allocates output vectors via
        `result.reserve(opts.expected_step_count())`.
      - Initialises `Vector x = Vector::Zero(state_size)`.
      - Pre-allocates a reusable `b_extra` buffer of `state_size`
        zeros for the no-`b_extra_fn` path.
      - Loops `t = t_start; t <= t_end; t += dt`:
        - `mask = switch_fn(t)`
        - If `b_extra_fn`: `b_extra = b_extra_fn(t)`; else use the
          zero buffer.
        - `cache.solve(mask, b_extra, x)`
        - Pushes `(t, x)` into the result (copies x — caller may
          inspect every step).
      - Returns the result.
- [ ] 2.1.4 The loop uses an integer step counter
      `k = 0..expected_step_count() - 1` and computes
      `t = t_start + k * dt` to avoid floating-point drift
      from accumulated `t += dt` over thousands of steps.

### 2.2 Tests `tests/v2/layer5/test_run_transient.cpp`
- [ ] 2.2.1 Invalid options throw `std::invalid_argument`.
- [ ] 2.2.2 `state_size == 0` throws.
- [ ] 2.2.3 Empty `switch_fn` (default-constructed
      `std::function`) throws.
- [ ] 2.2.4 V-R-GND circuit (no switches) — 10-step transient
      returns 10 time samples and a constant state vector
      across all steps (it's a DC circuit, output doesn't move).
- [ ] 2.2.5 V-Switch-R-GND chopper, switch_fn that returns
      `false` for the first half and `true` for the second half —
      verify the state-vector transition happens at the correct
      time and the on-state output matches the cached ON answer
      from Layer 4.
- [ ] 2.2.6 `b_extra_fn` is consulted every step — supply a fn
      that returns a known time-varying vector, verify the solver
      output reflects it.

## Phase 3 — Integration test: chopper at 10 kHz PWM (~0.5 days)

### 3.1 `tests/v2/layer5/test_integration_chopper_pwm.cpp`
- [ ] 3.1.1 Build the V_dc-Switch-R-GND chopper from Layer 4's
      integration test (`V_dc=12 V`, `g_on=1e3`, `g_off=1e-9`,
      `G_R=0.1`).
- [ ] 3.1.2 Define a `pwm_schedule(t)` lambda with a 10 kHz PWM
      signal (`T_pwm = 100 µs`, `duty = 0.5`) — returns ON when
      `fmod(t, T_pwm) < duty * T_pwm`, OFF otherwise.
- [ ] 3.1.3 Run `run_transient` with `t_start=0`, `t_end=1e-3`
      (1 ms), `dt=1e-6` (1 µs). That's 1001 simulation steps,
      10 PWM periods.
- [ ] 3.1.4 Verify `result.num_steps() == 1001`.
- [ ] 3.1.5 Verify the average of `v_out(t)` across the full 1 ms
      equals `V_dc · duty = 6.0` within `< 1 %` tolerance.
      Computation: `mean = sum(states[k][vout_idx]) / num_steps`.
- [ ] 3.1.6 Verify the waveform is a clean square wave:
      - For all `k` where the schedule says ON: `v_out[k]` is
        within `1e-6` of the analytical ON value.
      - For all `k` where the schedule says OFF: `v_out[k]` is
        within `1e-6` of the analytical OFF value.
- [ ] 3.1.7 Performance smoke: total wall-clock time for the
      1000-step simulation under `< 1000 ms` (very generous;
      should be under 100 ms on the test host).

## Phase 4 — Documentation (~0.25 days)

### 4.1 `docs/pulsim-v2/layer5-solver-and-events.md`
- [ ] 4.1.1 Section "What V0 is and isn't": fixed dt, externally-
      scheduled switching, static-only circuits.
- [ ] 4.1.2 Section "The run_transient loop": one diagram of the
      per-step hot path (switch_fn → b_extra_fn → cache.solve →
      record).
- [ ] 4.1.3 Section "Worked example: 10 kHz PWM chopper" — the
      Catch2 test inlined, showing duty = 0.5 → mean v_out = 6 V.
- [ ] 4.1.4 Section "What lands in V1+":
      - Event detection (zero-crossing, Vth crossing)
      - Adaptive dt + LTE estimation
      - Trapezoidal companion for caps/inductors
      - Newton for nonlinear devices
      - Strided output recording

## Phase 5 — Validation gates

- [ ] 5.1 `pulsim_v2_layer5_tests` MUST pass with zero failures.
      Initial target: ≥ 15 assertions / ≥ 8 test cases.
- [ ] 5.2 Layers 0 + 1 + 2 + 3 + 4 tests MUST stay green.
- [ ] 5.3 v1 suites MUST stay green.
- [ ] 5.4 `openspec validate pulsim-v2-solver-and-events --strict`
      MUST pass.
