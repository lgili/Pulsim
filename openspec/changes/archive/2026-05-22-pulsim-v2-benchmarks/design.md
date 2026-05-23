# Design — `pulsim-v2-benchmarks` (Layer 10 V0)

## What we're measuring (and what we're not)

### Measured

For each scenario, the **time-stepping loop only**:

- **v1**: `Simulator sim(circuit, opts); auto result = sim.run_transient();`
  Setup (Simulator construction, initial Jacobian factor) is
  EXCLUDED. The internal loop walks steps via Newton +
  whatever the integrator + linear-solver stack does.

- **v2**: `cache.build(dt)` + `run_transient(cache, graph,
  pool, opts, switch_fn, ...)`. The `cache.build(dt)`
  cost is EXCLUDED (one-shot per dt). The
  time-stepping `run_transient` itself is measured.

### NOT measured (V0)

- Memory footprint (interesting but separate dimension).
- Newton convergence iterations (varies per scenario).
- Cache miss rate (would require perf counters).
- Compilation time of v1 vs v2 stamps.

## Comparison fairness

To make the comparison apples-to-apples, both solvers run
with:
- Same `dt` (fixed-step).
- Same `t_end`.
- Same input signal (constant V_dc, or sinusoidal V_sine
  via b_extra in v2 / equivalent in v1).
- KLU as the linear solver in both (matches v2's default).
- Trap rule (v2 uses trap-companion; v1's `Trapezoidal`
  integrator matches).

The scenarios are intentionally SMALL (1-3 nodes) so that
the comparison stays clean and reproducible. Larger
circuits (RC ladders, dense topologies) are more variable
across hardware and aren't the primary v2 claim — v2's
edge comes from SWITCHING repetition, not absolute matrix
size.

## Scenarios

### S1 — V_dc + R (sanity baseline)

```
Vin (10V) ─── R (100Ω) ─── gnd
```

- dt = 1 µs, t_end = 1 ms → 1000 steps.
- No dynamic devices, no switches.
- v2 has 1 cached segment (static); v1 has a single
  static Jacobian (no per-step refactor).
- Expected: ~tie (v2 slightly faster due to less per-
  step bookkeeping).

### S2 — RC charging (dynamic, no switches)

```
Vin (5V) ─── R (1kΩ) ─── n1 ─── C (1µF) ─── gnd
```

- dt = 1 µs, t_end = 10 ms → 10 000 steps.
- 1 cap → 1 dynamic device.
- v2: 1 cached segment (built once for dt). Each step =
  history compute + 1 back-substitution.
- v1: trap companion changes the matrix per step? Or
  does v1 also pre-factor a static matrix for constant-
  dt linear circuits? Measurement reveals the truth.
- Expected: v2 modestly faster; the gap depends on
  whether v1 amortizes the dt-fixed factor.

### S3 — Half-wave rectifier (1 switch)

```
V_sine(60Hz, 10V) ─── diode ─── n1 ─── R (10Ω) ─── gnd
```

- dt = 100 µs, t_end = 16.67 ms (1 cycle) → 167 steps.
- 1 switching diode. v2 enumerates 2 segments.
- v2: per step = 1 cache lookup (2 segments) + 1
  back-substitution + diode-state update.
- v1: per step = refactor + Newton iterate.
- Expected: v2 wins clearly (3-10×).

### S4 — PWM chopper (heavy switching)

```
V_dc (24V) ─── Switch ─── n1 ─── R (10Ω) ─── gnd
```

- dt = 1 µs, t_end = 1 ms → 1000 steps.
- 100 kHz PWM, 50 % duty → ~100 PWM transitions over
  the run.
- v2 cache has 2 segments (1 switch); per step = 1
  back-substitution. Switch transitions are free for
  v2 (just pick the other cached factor).
- v1: refactor at each PWM transition.
- Expected: v2 wins biggest (5-20×) since switching is
  v2's strong suit.

## Output format

Each Catch2 test case `INFO`s a markdown row that the
build system can grep into a results file. Example:

```
| Scenario             | v1 (ms) | v2 (ms) | speedup |
|----------------------|---------|---------|---------|
| V_dc + R             |     0.5 |     0.4 |   1.3×  |
| RC charging          |    12.0 |     8.0 |   1.5×  |
| Half-wave rectifier  |     5.0 |     0.8 |   6.3×  |
| PWM chopper          |    18.0 |     0.9 |  20.0×  |
```

The numbers are illustrative — actual measurements come
from running the benchmark.

## Caveats documented in the report

- Single Mac (M-series), Release build, AppleClang 17,
  C++23 mode.
- Numbers vary across builds (LTO on/off, PGO on/off,
  cold/warm cache). The benchmark reports the FIRST
  run's wall-clock — sufficient for the architectural
  argument but not for sub-percent precision.
- v1 is configured to disable adaptive timestep + order
  control to match v2's fixed-dt model. Production v1
  use cases may be faster or slower depending on
  workload.
- The architectural argument is about SCALING with
  switching frequency, NOT absolute speed on a 1-step
  benchmark.

## Implementation

The benchmark binary `pulsim_v2_benchmarks` links both
v1 and v2:

```cmake
target_link_libraries(pulsim_v2_benchmarks PRIVATE
    pulsim::pulsim       # v1
    pulsim::v2
    Catch2::Catch2WithMain
)
```

Each scenario has a `TEST_CASE` that:
1. Builds the circuit in v1.
2. Builds the circuit in v2.
3. Runs both, captures `std::chrono::high_resolution_clock`
   durations.
4. `INFO`s the comparison row.
5. Sanity-asserts the final node voltages match within
   loose tolerance (both solvers should give the same
   physical answer).

## What V0 deliberately does NOT do

- **Catch2 microbenchmarks (`BENCHMARK` macro)** with
  per-run statistical analysis. V0 measures a single
  run's wall-clock. Statistical rigor is V1.
- **External tooling** (Google Benchmark, Hyperfine).
  V0 keeps it Catch2-internal.
- **Cross-platform reporting** (Linux + Windows
  measurements). V0 is Mac-only at the macro level.
- **PSIM / PLECS comparison**. Out of scope (those are
  closed-source commercial tools; the comparison
  point is v1 within the pulsim repo).
- **Memory benchmarks**. V0 is wall-clock only.

## Files

- NEW `core/tests/v2/benchmarks/test_main.cpp`
- NEW `core/tests/v2/benchmarks/test_v1_vs_v2.cpp`
- MODIFIED `core/CMakeLists.txt` (benchmark target)
- NEW `docs/pulsim-v2/layer10-benchmarks.md`
