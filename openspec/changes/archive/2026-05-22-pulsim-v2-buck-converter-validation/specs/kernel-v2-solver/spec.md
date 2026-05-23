## ADDED Requirements

### Requirement: Synchronous Buck Converter Integration Test

The Layer 5 V1 OpenSpec MUST include an integration test that
exercises the full v2 kernel on a synchronous buck converter
topology and validates the steady-state output against analytical
expectations.

The test MUST:

1. Build a synchronous buck graph with V_in source, two
   complementary switches (Q1 high-side, Q2 low-side), an
   inductor L, an output capacitor C, and a load resistor
   R_load — assembled per the topology documented in
   `design.md`.
2. Drive the switches with a complementary PWM schedule at
   100 kHz and 50 % duty: Q1 ON during the first half-period,
   Q2 ON during the second half-period (no dead-time, no
   shoot-through).
3. Simulate from rest (all-zero IC) for `t_end = 1 ms` at
   `dt = 100 ns` (10 001 samples, 100 samples per PWM period,
   100 PWM periods total).
4. Analyze steady-state metrics over the LAST 10 PWM periods
   (skipping the startup transient).
5. Validate four analytical metrics with the tolerances listed
   below.

#### Scenario: Buck converter cache builds 4 segments

- **GIVEN** the synchronous buck graph (2 switches)
- **WHEN** the user constructs `PwlStateSpaceCache` and calls
  `build(dt = 100 ns)`
- **THEN** `cache.num_segments()` SHALL equal `4` (2^2
  combinations).

#### Scenario: Mean V_out matches V_in · duty

- **GIVEN** the buck converter simulating for 1 ms
- **WHEN** the user computes `mean(v_out)` over the last 10 PWM
  periods (t ∈ [0.9 ms, 1 ms])
- **THEN** the result SHALL be within `5 %` of `V_in · D =
  12 · 0.5 = 6.0 V`.

#### Scenario: Mean inductor current matches V_out / R_load

- **GIVEN** the same simulation
- **WHEN** the user computes `mean(i_L)` over the last 10 PWM
  periods
- **THEN** the result SHALL be within `5 %` of `V_out / R_load
  = 6.0 / 10 = 0.6 A`.

#### Scenario: Inductor current ripple matches CCM formula

- **GIVEN** the same simulation
- **WHEN** the user computes peak-to-peak `ΔI_L = max(i_L) −
  min(i_L)` over the measurement window
- **THEN** the result SHALL be within `15 %` of the analytical
  CCM ripple `V_in · D · (1 − D) / (L · f_sw) = 0.3 A`.

#### Scenario: Output voltage ripple is at least the right order of magnitude

- **GIVEN** the same simulation
- **WHEN** the user computes peak-to-peak `ΔV_out =
  max(v_out) − min(v_out)` over the measurement window
- **THEN** the result SHALL be within `30 %` of the analytical
  formula `ΔI_L / (8 · C · f_sw) = 3.75 mV`. (The analytical
  formula ignores higher-order harmonics; the simulation
  captures them, so the simulated ripple is allowed to be
  somewhat larger than the analytical value.)

#### Scenario: 10 000-step simulation completes within wall-clock budget

- **GIVEN** the buck converter simulation
- **WHEN** the user runs `run_transient` and times it
- **THEN** the total wall-clock time SHALL be less than `5
  seconds` on the test host. (Conservative bound for Debug
  builds; Release builds will be much faster.)
