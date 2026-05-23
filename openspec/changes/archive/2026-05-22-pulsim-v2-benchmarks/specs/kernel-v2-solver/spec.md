## ADDED Requirements

### Requirement: v1 vs v2 wall-clock benchmark suite

A `pulsim_v2_benchmarks` Catch2 target SHALL exist that builds the same circuit in v1 (`pulsim::v1::Circuit + Simulator`) and v2 (`CircuitBuilder + PwlStateSpaceCache + run_transient`) for at least four scenarios:

1. V_dc + R (linear, no switches, no dynamic devices).
2. RC charging (1 capacitor, no switches).
3. Half-wave rectifier (1 switching diode).
4. PWM chopper (1 controlled switch with periodic toggle).

For each scenario, the benchmark MUST measure wall-clock time of the **time-stepping loop only** (excluding setup / matrix factorization) for both v1 and v2, and `INFO` a markdown table row with the comparison.

Both v1 and v2 SHALL run with the same `dt`, `t_end`, and input signals. v1 MUST be configured with `adaptive_timestep = false` and the `Trapezoidal` integrator (matching v2's trap companion) using the KLU linear solver.

The benchmark MUST sanity-assert that the final node voltages from v1 and v2 match within a loose tolerance (10 % of the expected value), confirming both solvers compute the same physical answer.

#### Scenario: Benchmark binary exists and reports measurements

- **GIVEN** a Release build of pulsim
- **WHEN** the user runs `pulsim_v2_benchmarks`
- **THEN** the binary SHALL execute all four scenarios without throwing
- **AND** each scenario's `INFO` output SHALL contain wall-clock measurements for both v1 and v2 (in milliseconds)
- **AND** the final v1 and v2 state vectors SHALL agree on the primary observable (e.g. v_node) within 10 %.
