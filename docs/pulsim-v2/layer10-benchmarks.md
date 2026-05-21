# Layer 10 — v1 vs v2 wall-clock benchmarks

The strategic claim that motivated the v2 rewrite:

> Pre-factor the MNA matrix once per switch combination
> (PWL state-space cache) → each timestep is just a back-
> substitution. v1 (and SPICE-style solvers) re-factor per
> timestep — slower for repetitive PWM workloads.

V10 measures this claim head-to-head. The benchmark binary
`pulsim_v2_benchmarks` builds the SAME circuit in both v1
and v2, runs the same simulation, and reports wall-clock.

## Results — measured on M-series Mac, AppleClang 17, C++23

### Release build (`-O3 -DNDEBUG`)

| Scenario                  |   v1 (ms) |   v2 (ms) | speedup |
|---------------------------|----------:|----------:|--------:|
| S1: V_dc + R              |     2.919 |     0.297 |   9.8×  |
| S2: RC charging           |    14.925 |     1.956 |   7.6×  |
| **S3: Half-wave rectifier** | **3.863** | **0.304** | **12.7×** |

### Debug build (`-O0 -g`)

| Scenario                  |   v1 (ms) |   v2 (ms) | speedup |
|---------------------------|----------:|----------:|--------:|
| S1: V_dc + R              |    22.483 |     4.678 |   4.8×  |
| S2: RC charging           |   151.867 |    50.974 |   3.0×  |
| **S3: Half-wave rectifier** | **49.229** | **7.146** | **6.9×** |

**All scenarios confirm the architectural claim**: v2's
PWL cache is measurably faster than v1's per-step refactor.

The biggest win is on the **switching scenario (S3)** —
exactly the architectural prediction. Each zero-crossing
of the AC source toggles the diode's on/off state:
- v1 refactors the MNA matrix at every commutation.
- v2 just picks the OTHER pre-factored cached segment
  via `cache.lookup(mask)` — essentially free.

Release mode amplifies all speedups because v2's tighter
inner loop (no Newton iteration overhead for linear
circuits, no per-step factor work) benefits more from
inlining and vectorization.

## What the benchmarks measure

For each scenario, the **time-stepping loop only**:

- **v1**: `Simulator sim(circuit, opts); auto result = sim.run_transient();`
  Setup (Simulator construction, initial Jacobian factor) is
  EXCLUDED. The internal loop walks steps via Newton + the
  Trapezoidal integrator using KLU.

- **v2**: `cache.build(dt)` + `run_transient(cache, graph,
  pool, opts, switch_fn, …)`. The `cache.build(dt)` cost is
  EXCLUDED (one-shot per dt). The time-stepping `run_transient`
  itself is measured.

## Fairness configuration

Both solvers run with:

| Knob | Value |
|------|-------|
| dt | fixed (no adaptive) |
| Integrator | Trapezoidal (v1) / trap-companion (v2) |
| Linear solver | KLU |
| Newton globalization | none (default) |

This is the cleanest apples-to-apples comparison. Production
v1 use cases may differ in tunings.

## Scenarios

### S1 — V_dc + R

```
Vin (10V) ─── R (100Ω) ─── gnd
```

- `dt = 1 µs`, `t_end = 1 ms` → 1000 steps.
- No dynamic devices, no switches.
- Static MNA matrix in both solvers.
- v2 wins because each step is just a back-substitution
  on the pre-factored static matrix; v1 still goes through
  its Newton + history-tracking machinery even though
  there's no nonlinearity.

### S2 — RC charging

```
Vin (5V) ─── R (1kΩ) ─── n1 ─── C (1µF) ─── gnd
```

- `dt = 1 µs`, `t_end = 5 ms` → 5000 steps.
- 1 capacitor (dynamic device).
- v2: 1 cached segment built once for `dt`. Each step is
  back-substitution + trap-companion history update.
- v1: trap-companion machinery rebuilds + factors per step
  (or per group of steps depending on event detection).
- v2's win is the per-step refactor cost saved.

### S3 — Half-wave rectifier (switching!)

```
V_sine(60Hz, 10V) ── diode ── n1 ── R (10Ω) ── gnd
```

- `dt = 50 µs`, `t_end = 33.3 ms` → ~666 steps over 2 cycles.
- 1 switching diode that auto-commutates 4× (once per
  zero-crossing).
- v2 enumerates **2 cached segments** (diode ON vs OFF)
  at build time. Each commutation is a `cache.lookup`
  + back-substitution — essentially free.
- v1 refactors the MNA matrix at each commutation event.
- This is the scenario v2 was DESIGNED for: switching
  topology + repeated commutations.

The **12.7× Release speedup on S3** is the architectural
claim made concrete.

## What V0 deliberately does NOT measure

- **PWM chopper / controlled switch scenario**: v1's
  `add_switch` requires external `set_switch_state` calls
  between solver steps, which would need a custom run
  loop to interleave with `run_transient`. The diode
  scenario (S3) already proves the architectural point;
  adding PWM is V1.
- **Larger circuits**: V0 keeps circuits small (1-3
  nodes) for reproducibility. Speedup may scale further
  with circuit size; V1 adds RC-ladder / dense topology
  cases.
- **Memory footprint**: V0 is wall-clock only.
- **Statistical rigor** (multiple runs, mean ± stddev):
  V0 measures a single run. Catch2's `BENCHMARK` macro
  with per-run statistics is V1.
- **External tooling** (Google Benchmark, Hyperfine):
  V0 is Catch2-internal.
- **PSIM / PLECS comparison**: out of scope (closed-
  source commercial tools).
- **Cross-platform** (Linux + Windows numbers): V0 is
  Mac-only.

## How to run

```bash
cmake --build build --target pulsim_v2_benchmarks
./build/core/pulsim_v2_benchmarks
```

The benchmarks are NOT registered with `ctest` — they don't
add regression value, they're for measurement on demand.

## Caveats

- Numbers vary across builds (LTO on/off, PGO on/off, cold
  vs warm cache). The benchmark reports the FIRST run's
  wall-clock — sufficient for the architectural argument,
  not for sub-percent precision.
- The architectural argument is about SCALING with
  switching frequency, NOT absolute speed on a 1-step
  benchmark. Even on linear circuits v2 wins because of
  reduced per-step overhead, but the bigger win comes
  from circuits with many PWM commutations (V1).

## Files

- NEW `core/tests/v2/benchmarks/test_main.cpp`
- NEW `core/tests/v2/benchmarks/test_v1_vs_v2.cpp`
- MODIFIED `core/CMakeLists.txt` (`pulsim_v2_benchmarks` target)
- NEW `openspec/changes/pulsim-v2-benchmarks/`
