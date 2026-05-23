## Why

v2's strategic claim is: **PWL state-space caching beats v1
(and PSIM/PLECS) on circuits with switching topology**.
The architectural intuition: pre-factor the MNA matrix
once per switch combination, then each timestep is just a
back-substitution. v1 (and SPICE-style solvers) re-factor
per timestep — much slower for repetitive PWM workloads.

So far we've VALIDATED v2's correctness (4894 C++
assertions + 22 Python tests across 14 binaries) and built
the full surface (CircuitBuilder, YAML, Python bindings,
MOSFETs, transformer, end-to-end SMPS showcase). But we
haven't MEASURED the speedup claim.

V10 ships head-to-head benchmarks: same circuit built in
v1 and v2, same simulation, wall-clock comparison. The
output is a markdown table with measured speedup factors
per scenario.

## What Changes

**Scope decision — Layer 10 V0** (v1 vs v2 wall-clock
benchmarks):

- New C++ benchmark target `pulsim_v2_benchmarks` linking
  against BOTH `pulsim::pulsim` (v1) and `pulsim::v2` to
  enable side-by-side comparisons in the same binary.

- Scenarios (~4 cases):
  1. **V_dc + R** (linear, no switches, no dynamic devices)
     — sanity baseline.
  2. **RC charging** (1 cap, no switches) — tests
     dynamic-device handling.
  3. **Half-wave rectifier** (V_sine + diode + R) — tests
     the WIN case: PWL cache vs per-step refactor.
  4. **PWM buck-style chopper** (V_dc + switch + R, no
     L/C for simplicity) — tests heavy switching.

- For each scenario:
  - Build the same circuit in v1 and v2.
  - Configure v1 with `adaptive_timestep = false`,
    `Integrator::BDF1` (or `Trapezoidal`), KLU solver
    to match v2's numerical setup.
  - Configure v2 with the same `dt` and `t_end`.
  - Wall-clock measure ONLY the time-stepping loop
    (excluding cache build / matrix factorization
    setup — that's a one-shot cost).
  - Report: total time, time per step, speedup factor
    (`v1_time / v2_time`).

- Output: markdown table printed via `INFO` in each
  Catch2 case, plus a docs page summarising the results.

- V0 expectation: v2 wins by ~2-20× on switching
  scenarios; loses or ties on tiny linear circuits where
  setup overhead dominates.

## Impact

- **Affected specs**: ADDED requirement on
  `kernel-v2-solver` for measured benchmarks.
- **Affected code** (~250 LOC):
  - NEW `core/tests/v2/benchmarks/test_main.cpp`
  - NEW `core/tests/v2/benchmarks/test_v1_vs_v2.cpp`
  - MODIFIED `core/CMakeLists.txt` (new benchmark target)
  - NEW `docs/pulsim-v2/layer10-benchmarks.md`
- **Migration**: zero. Pure additive (benchmark target
  is opt-in).
- **Risk**: low. The benchmark numbers depend on
  hardware + build flags; results are reported with a
  caveat (single Mac, Release build, etc.) rather than
  as absolute guarantees.
