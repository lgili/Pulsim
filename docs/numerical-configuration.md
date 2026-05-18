# Numerical Configuration

> **Status: shipped.** Single canonical entry point for configuring
> Pulsim's transient simulation engine — `SimulationOptions.from_preset(...)`
> + 4 named profiles cover 95% of use cases in one decision.

This page is the **user-facing reference** for the
`simplify-and-harden-numerical-surface` change. It replaces the
"hand-tune 50 fields" workflow of earlier releases with a small set
of named presets, hides the convergence machinery behind sensible
auto-tuning, and exposes the new multilevel-convergence aids
(Armijo line search, simultaneous event coalescence, iterative
refinement, homotopy continuation) without any user-facing knobs.

If you're looking for the historical audit that drove this design,
see [`numerical-modes-audit.md`](numerical-modes-audit.md) (kept
in-tree as the working document).

## TL;DR — pick a preset

```python
import pulsim as ps

opts = ps.SimulationOptions.from_preset(ps.Preset.Auto,
                                          dt=1e-6, tstop=1e-3)
ps.Simulator(circuit, opts).run_transient()
```

The four presets:

| Preset | Use when | What it sets |
|---|---|---|
| `Auto` (default) | I don't know which one — let the simulator pick | Robust profile today; tracks the production recommendation |
| `Fast` | Pure-switching topology (buck, boost, full-bridge, basic 3φ VSI) | PWL Ideal + Trapezoidal + KLU + Fixed step |
| `Robust` | Motor drives, mixed-domain, magnetics, thermal feedback | TRBDF2 + KLU + Variable + stiffness on + 12 retries |
| `HighFidelity` | Parity validation vs PLECS / PSIM / SPICE | TRBDF2 + 10× tighter LTE + small dt_max |

YAML:

```yaml
simulation:
  preset: robust          # case-insensitive
  tstop: 1e-3
  dt: 1e-6
```

C++:

```cpp
#include "pulsim/v1/simulation.hpp"
using namespace pulsim::v1;
auto opts = SimulationOptions::from_preset(Preset::Auto, 1e-6, 1e-3);
Simulator(circuit, opts).run_transient();
```

That's the entire learning curve for normal users.

## What the presets cover

Each preset materialises a **complete** numerical configuration. You
don't need to set anything else for the preset's target use case.
The full per-preset table:

| Field | `Fast` | `Auto` / `Robust` | `HighFidelity` |
|---|---|---|---|
| `switching_mode` | `Ideal` | (default) | (default) |
| `integrator` | `Trapezoidal` | `TRBDF2` | `TRBDF2` |
| `step_mode` | `Fixed` | `Variable` | `Variable` |
| `enable_bdf_order_control` | `False` | `True` (min=1, max=2) | `True` |
| `stiffness_config.enable` | `False` | `True` (switch → BDF1) | `True` |
| `max_step_retries` | 2 | 12 | 24 |
| `timestep_config.error_tolerance` | n/a | 5e-3 | 5e-4 |
| `lte_config.voltage_tolerance` | n/a | 5e-3 | 5e-4 |
| `timestep_config.dt_max` | = `dt` | 20·`dt` | 2·`dt` |
| `fallback_policy.enable_transient_gmin` | `False` | `True` | `True` |

## Overriding individual fields

Pick a preset, then override only what genuinely differs:

```python
opts = ps.SimulationOptions.from_preset(ps.Preset.Robust, 1e-6, 1e-3)
opts.integrator = ps.Integrator.BDF1   # override TRBDF2 default
opts.max_step_retries = 50              # bigger retry budget
```

Order of resolution:

1. **C++ defaults**: `SimulationOptions{}` constructs with the legacy
   raw defaults (Trapezoidal + Variable + stiffness on + retries=6).
2. **`from_preset(...)`**: overwrites every field on the struct with
   the preset's tuned values.
3. **Explicit field assignments**: override individual fields after
   the factory call.

YAML follows the same order:

```yaml
simulation:
  preset: robust
  integrator: bdf1         # overrides Robust's TRBDF2
  max_step_retries: 50     # overrides Robust's 12
  tstop: 1e-3
  dt: 1e-6
```

The parser logs a single deprecation warning per process if you set
`adaptive_timestep` or `direct_formulation_fallback` — both are
deprecated in favor of `step_mode` and "always-on" semantics
respectively.

## Convergence aids — what runs under the hood

Pulsim's convergence layer ships with **four** automatic aids that
fire on hard problems without user intervention. The first three are
always-on by default; the fourth is on the `DCStrategy::Auto` ladder.

### 1. Armijo line search (Newton)

Inside every Newton iteration, the line search uses the textbook
**Armijo descent condition** to decide whether to accept the
trial step:

    ||f(x + α·dx)|| ≤ (1 − σ·α) · ||f(x)||    (σ = 1e-4 default)

If the trial fails, α is halved and the step retries up to 10
backtracks. This is strictly stronger than the legacy "any
reduction" criterion and provides provable convergence guarantees on
problems where the full Newton step overshoots (the textbook
counter-example: `arctan(x)` from x₀ > 1, which pure Newton diverges
on but Armijo converges to 0).

**Tuning** (rarely needed):

```python
opts.newton_options.armijo_line_search = True   # default
opts.newton_options.armijo_sigma = 1e-4         # default
opts.newton_options.min_damping = 0.01          # backtrack floor
```

**Telemetry**: `result.newton_result.telemetry.line_search_backtracks`
counts backtracks across all Newton iterations of the DC OP.

### 2. Simultaneous event coalescence (PWL engine)

When two or more PWL switches commute within the bisection tolerance
of each other (e.g. all six MOSFETs of a 3φ VSI flipping on the same
PWM edge, or all submodules of an MMC arm switching synchronously),
the PWL bisection coalesces them into a **single atomic Newton
solve** at the bisected event time instead of processing them
serially across N consecutive steps.

This used to be the dominant convergence-failure mode on dense
switching topologies like MMC.

**No tuning**: the coalescence window is fixed at `16 ·
bisection_tolerance` (well past machine-precision spread of
synchronously-commanded gates).

**Telemetry**: `result.backend_telemetry.simultaneous_event_groups`
counts steps where ≥ 2 commutations were coalesced. Zero on
single-event steps; grows on densely-switching topologies.

### 3. Iterative refinement on direct linear solve

After every successful direct-solver back-substitution (`SparseLU`,
`EnhancedSparseLU`, `KLU`), the solver computes the relative residual
`||b − A·x|| / ||b||`. If it exceeds `10·ε_machine`, **one round of
iterative refinement** fires automatically:

```
r = b − A·x
A · δ = r        (cheap: re-uses existing factorization)
x ← x + δ
```

Triggers on **ill-conditioned MNA** — flying-cap, MMC,
dense-floating-cap topologies where KLU's partial-pivoting back-sub
accumulates round-off. Common case (well-conditioned RC, buck, boost)
sees zero refinement cost.

**No user tuning**.

**Telemetry**:
`result.linear_solver_telemetry.linear_refinement_steps` counts the
number of refinement rounds. Zero on well-conditioned MNA; non-zero
on multilevel.

### 4. Homotopy continuation (DC operating point)

When `DCStrategy::Auto` exhausts the first four strategies (Direct →
SourceStepping → GminStepping → PseudoTransient), the orchestrator
falls back to **homotopy continuation**: it steps a parameter λ from
0 (all nonlinear devices replaced by their `g_off` conductance —
fully linear MNA, solves in one direct call) to 1 (full nonlinear
model) in 5 increments, warm-starting Newton at each step from the
previous step's solution.

This is the strategy for cold-start multilevel converters (NPC, MMC,
flying-cap) where the nonlinear residual landscape has too many
local minima for the earlier strategies to escape.

**Tuning** (rarely needed):

```python
opts.dc_config.homotopy_config.enable = True       # default
opts.dc_config.homotopy_config.ladder_steps = 5    # default (10 in HighFidelity)
opts.dc_config.homotopy_config.max_newton_per_step = 10
```

**Telemetry**:
`result.dc_result.homotopy_steps` (number of λ increments executed),
`result.dc_result.homotopy_ladder_completed` (True if reached λ=1).

## When to override the preset

If you find yourself overriding more than 3-4 fields after the
preset, you're probably picking the wrong preset. Quick chooser:

| Problem | Preset |
|---|---|
| Newton fails to converge on DC OP | `Robust` (gets the full convergence-aid ladder) |
| Newton OK but transient overshoots | `Robust` (TRBDF2 instead of Trapezoidal) |
| Validating against SPICE / PLECS golden CSV | `HighFidelity` |
| Pure switching converter, no nonlinear devices | `Fast` |
| Simulation is too slow | `Fast` (if topology allows) |
| I have no idea | `Auto` |

Beyond the preset, the **single most impactful knob** is
`switching_mode`. Setting it to `SwitchingMode.Ideal` engages the PWL
state-space fast path on every switching device that opts in
(MOSFETs, IGBTs, diodes, voltage-controlled switches). For
power-electronics circuits this is usually a 5-20× speedup vs the
Behavioral path.

```python
opts.switching_mode = ps.SwitchingMode.Ideal
```

## Available integrators

After the v0.11 surface cleanup, the **curated** integrator set is:

| Value | Use when |
|---|---|
| `Integrator.Trapezoidal` | Power-electronics default (A-stable order 2) |
| `Integrator.BDF1` | L-stable stiff fallback (backward Euler) |
| `Integrator.BDF2` | General-purpose stiff (A-stable order 2) |
| `Integrator.TRBDF2` | Trapezoidal + BDF2 hybrid (stiff-stable, default in Robust preset) |
| `Integrator.RosenbrockW` | Linearly implicit stiff (no Newton inner loop) |

**Deprecated** (still parse with a warning, removed in next major release):

| Value | Replacement |
|---|---|
| `BDF3`, `BDF4`, `BDF5` | `BDF2` / `TRBDF2` / `RosenbrockW` — these were not A-stable for switching topologies |
| `Gear` | `BDF2` (literal alias) |
| `SDIRK2` | `TRBDF2` (research-grade, no benchmark coverage) |

## Telemetry reference

Every transient run exposes a `result.backend_telemetry` and
`result.linear_solver_telemetry` with diagnostic counters. The most
useful for understanding what the convergence aids did:

```python
result = ps.Simulator(circuit, opts).run_transient()
t = result.backend_telemetry

# Newton activity
print("Newton iters total:           ", t.newton_iterations_total)
print("Line-search backtracks:       ", result.newton_iterations_total)

# PWL event scheduler
print("PWL commutations:             ", t.pwl_event_commutations)
print("PWL bisections:               ", t.pwl_event_bisections)
print("Simultaneous-event groups:    ", t.simultaneous_event_groups)

# Linear solver
print("Active solver:                ", result.linear_solver_telemetry.last_solver)
print("Total solve calls:            ", result.linear_solver_telemetry.total_solve_calls)
print("Iterative refinement steps:   ", result.linear_solver_telemetry.linear_refinement_steps)

# DC OP
if result.dc_result is not None:
    print("DC strategy used:           ", result.dc_result.strategy_used)
    print("DC homotopy steps:          ", result.dc_result.homotopy_steps)
```

If a circuit reports `simultaneous_event_groups > 0` or
`linear_refinement_steps > 0` or
`homotopy_ladder_completed == True`, that's a signal the
convergence aids fired and saved the run.

## Migration from earlier releases

If you have existing code that uses `make_robust_options(...)` (the
Python helper that lived in `python/bindings.cpp`):

```python
# Before (still works):
opts = ps.make_robust_options(circuit, 0.0, 1e-3, 1e-6, ps.NewtonOptions(),
                                ps.LinearSolverStackConfig.defaults())

# After (recommended):
opts = ps.SimulationOptions.from_preset(ps.Preset.Robust, dt=1e-6, tstop=1e-3)
```

If you have YAML that sets `adaptive_timestep`:

```yaml
# Before (works with deprecation warning):
simulation:
  adaptive_timestep: true

# After:
simulation:
  step_mode: variable
```

If you have YAML that sets `direct_formulation_fallback`:

```yaml
# Before (deprecation warning — fallback is always-on now):
simulation:
  direct_formulation_fallback: true

# After: just remove the field — the fallback is unconditionally on.
```

If you have YAML / Python that picks `Integrator::BDF5`, `Gear`, or
`SDIRK2`:

```yaml
# Before (deprecation warning):
simulation:
  integrator: bdf5

# After:
simulation:
  integrator: trbdf2     # for stiff problems
  # or: bdf2 / rosenbrockw
```

## See also

- [`numerical-modes-audit.md`](numerical-modes-audit.md) — the
  historical audit that motivated this design (kept as the working
  document)
- [`convergence-tuning-guide.md`](convergence-tuning-guide.md) —
  diagnosing and fixing convergence failures
- [`multilevel-converters.md`](multilevel-converters.md) — the MMC
  topology template + benchmarks
- [`pwl-switching-migration.md`](pwl-switching-migration.md) — the
  PWL state-space engine
- [`device-models.md`](device-models.md) — per-device modelling
  details
