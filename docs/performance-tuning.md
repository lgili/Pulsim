# Performance Tuning

Pulsim is fast by default — the PWL state-space cache pre-factors
every reachable switch configuration into a sparse LU once, then
the transient loop is one back-substitution per step. Most circuits
just work. This guide is for the cases that don't.

## Build-time knobs

Set these on the CMake configure line; details in
[build-system.md](build-system.md).

| Flag | Default | Effect |
|---|---|---|
| `CMAKE_BUILD_TYPE=Release` | — | Always required for benchmarking. |
| `PULSIM_ENABLE_LTO` | `ON` (Release) | Link-time optimization; 10–20 % wins on the heavy templated code. Auto-disabled for Linux Python wheels (pybind11 TLS interaction). |
| `PULSIM_ENABLE_NATIVE` | `OFF` | Adds `-march=native -mtune=native`. Don't ship binaries built with this. |
| `PULSIM_ENABLE_PGO_GENERATE` / `..._USE` | `OFF` | Two-pass profile-guided optimization; comments in `CMakeLists.txt` explain the workflow. |
| `PULSIM_USE_HYPRE` | `ON` | Optional AMG backend for very large systems. |

## Runtime knobs

All on `SimulationOptions`:

```python
import pulsim as p

opts = p.SimulationOptions(t_start=0.0, t_end=1e-3, dt=1e-6)

# Nonlinear-device refresh — Newton iteration on top of the cached
# LU when a smooth-blend diode / MOSFET / IGBT / saturable inductor
# is in the circuit. ``p.simulate(...)`` auto-detects this; pass
# explicitly to override.
opts.enable_nonlinear_refresh = True

# Newton solver tolerances + iteration cap.
opts.max_newton_iterations = 50
opts.tol_newton_dx  = 1e-9       # |Δx|∞ termination
opts.tol_newton_res = 1e-9       # |F(x)|∞ termination

# Globalization strategies (off by default; turn on for stiff
# converters that diverge from a cold start).
opts.enable_newton_line_search = True   # Armijo backtracking
opts.enable_newton_lm = True            # Levenberg-Marquardt trust region

# Sub-step state correction — when a commutation event lands inside
# a fixed-dt step, split the step in two at the linearly-interpolated
# crossing instant. Big accuracy win on PWM converters; small cost.
opts.enable_substep_state_correction = True

# Event-detection iteration cap.
opts.max_event_iterations = 32
```

The ``p.simulate(...)`` ergonomic wrapper accepts each of these as
a keyword argument:

```python
res = p.simulate(
    b, t_end=1e-3, dt=1e-6,
    enable_nonlinear_refresh=True,
    enable_newton_line_search=True,
    enable_substep_state_correction=True,
    max_newton_iterations=50,
)
```

## Cache + scaling

The PWL cache lives on `PwlStateSpaceCache(graph, pool).build(dt)`.
Key properties:

- **One sparse LU per reachable switch combination.** A buck with
  one switch + one diode = 4 combinations, all 4 factored once at
  setup. The transient loop never touches a sparse solver again
  for linear circuits.
- **Lazy expansion.** Combinations not reached in the simulation
  are never factored. Cold-start cost ≈ (num reached configs) ×
  (single LU cost).
- **Multi-dt cache.** Several `dt` values can coexist in the same
  cache (`cache.build_at_dt(dt_1)`, `cache.build_at_dt(dt_2)`);
  the solver picks the right factor by `dt` key.

For circuits with many switches (3-φ VSI with 6 IGBTs has 64
reachable states; PFC + boost cascade can have hundreds), the
cache can become memory-heavy. Profile with `cache.num_entries()`
and `cache.factor_bytes_estimate()`.

## DC operating-point seeding

A converter that doesn't converge from `x=0` often converges from
its DC OP. Two ways to ask:

```python
# 1) p.simulate(...) flag
res = p.simulate(b, t_end=..., dt=..., start_from_dc_op=True)

# 2) Explicit compute_dc_op with a strategy
from pulsim import compute_dc_op, PseudoTransientConfig
x0 = compute_dc_op(
    b, t_eval=0.0,
    config=PseudoTransientConfig(num_steps=50, dt_initial=1e-6),
)
```

The strategies (`compute_dc_op` calls them under the hood):

- **`SourceStepConfig`** — ramp sources from 0 → final value over N
  steps. Robust for stiff non-linear circuits.
- **`PseudoTransientConfig`** — Newton iteration with an added
  pseudo-time damping term that vanishes at convergence; good for
  systems with multiple Newton basins.

See [gotchas.md](gotchas.md) for which one to reach for first.

## Profiling

The C++ side has no internal profiling hooks — the kernel is
header-only and any allocation or branch happens inline. To measure:

- **Wall-clock per simulation:** `time.perf_counter()` around
  `p.simulate(...)`.
- **Per-step cost:** wrap with a `step_observer` and instrument the
  callback (note: this re-enters Python per step, biasing the
  measurement; the C++ path via `MixedDomainBlockChain` is the
  measurement-quality option).
- **Compiler-level**: `samply`, `perf`, or Instruments.app on macOS.
  The hottest function tends to be `pwl::Cache::solve_at`.

## Common pitfalls

- **`dt` too small.** Pulsim doesn't enforce `dt ≪ τ_min`. If `dt`
  is below `1e-9` you're paying for sub-nanosecond sampling and
  Newton accuracy with no physical reason. 10 % of the smallest
  rise/fall time is usually enough.
- **`dt` too large.** Sub-step event correction patches some
  inaccuracy, but a `dt` that misses the entire ON portion of a
  PWM pulse will lose duty cycle. Sample at ≥ 20 points per
  switching period.
- **Sat-inductor + nonlinear refresh.** Saturable magnetics need
  `enable_nonlinear_refresh=True`. ``p.simulate(...)`` detects this
  automatically; the explicit `run_transient` does not.
- **Smooth diode + Newton.** Hard-edge `IdealDiode` (PWL `g_on` /
  `g_off`) doesn't need Newton; the smooth-blend variant
  (`IdealDiodeParams(blend_width=...)`) does. Enable
  `enable_nonlinear_refresh` for the latter.

## Benchmarks

The `benchmark_compile_time` custom target (`PULSIM_BUILD_BENCHMARKS=ON`
configure) times a clean rebuild of the heaviest test binary
(`pulsim_layer5_v4_tests` — Newton in run_transient). Useful when
auditing a kernel-header change for compile-time regressions.

There's no in-tree wall-clock harness for transient simulation yet
— individual test binaries time themselves with
``BENCHMARK("…") { … }`` macros from Catch2.
