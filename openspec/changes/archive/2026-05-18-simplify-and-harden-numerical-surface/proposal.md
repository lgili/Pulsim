## Why

Pulsim's numerical configuration surface has grown to **~50 user-tunable
knobs across 9 axes** (switching mode, integrator, linear solver,
Newton, step control, formulation, DC strategy, stiffness, analysis
type). The audit in `docs/numerical-modes-audit.md` shows:

- **5 dead integrators** (`BDF3`, `BDF4`, `BDF5`, `Gear` alias, `SDIRK2`)
  that are unstable for power-electronics switching and never exercised
  in production benchmarks.
- **Default split-brain**: raw `SimulationOptions{}` uses `Trapezoidal +
  SparseLU + no stiffness`, while `make_robust_options()` uses
  `TRBDF2 + KLU + stiffness + 12 retries`. Two different "defaults" for
  the same class.
- **Overlapping axes**: `adaptive_timestep` (bool) and `step_mode`
  (enum) control the same thing; `direct_formulation_fallback` overlaps
  with `formulation_mode`.
- **Leaky abstractions**: users have to know what ILU0 vs ILUT is, or
  what `GminStepping` vs `PseudoTransient` means, to simulate a buck
  converter.
- **No `Preset` entry point**: a new user must either accept the slow
  raw defaults or hand-tune a 50-field struct.

At the same time, **convergence on multilevel converters (NPC, T-type,
flying-cap, MMC)** is fragile: many simultaneous switching events, ill-
conditioned MNA matrices from large floating-cap networks, and stiff
controller transients regularly produce Newton non-convergence.

Source files are spread across the kernel without a clear "numerical
core" boundary — `simulation.hpp`, `transient_services.hpp`,
`convergence_aids.hpp`, `high_performance.hpp`, and `integration.hpp`
each own a slice of the picture, so future improvements (e.g., adding
SUNDIALS hooks, a line-search Newton, or a sparser MMC topology
template) have no obvious home.

This proposal does three things in one coordinated PR sequence:

1. **Slim the user-facing surface** to one `Preset` enum (`Auto`,
   `Fast`, `Robust`, `HighFidelity`) + 4 essential knobs (`tstop`,
   `dt`, `switching_mode`, `integrator`). Everything else moves to
   `options.advanced.*` and stops being a top-level kwarg.
2. **Remove dead code** (BDF3+, Gear, SDIRK2, redundant booleans,
   precondition-enum leak) — net negative LOC.
3. **Add convergence robustness for multilevel converters** —
   damped-Newton line search, simultaneous event detection, iterative
   refinement on ill-conditioned KLU factorizations, homotopy
   continuation on DC OP, and an MMC topology template + benchmark
   suite to validate PLECS / PSIM parity.

The reorganization also collapses the numerical layer into a single
`core/include/pulsim/v1/numerical/` directory so future improvements
have a home.

## What Changes

### NEW

- **`pulsim::v1::Preset` enum** — `Auto` (default), `Fast`, `Robust`,
  `HighFidelity`. Each materialises a `SimulationOptions` profile
  derived from the audit's recommended defaults.
- **`SimulationOptions::from_preset(Preset, dt, tstop)` factory** —
  the canonical entry point. The raw `SimulationOptions{}` ctor still
  works for power users but is no longer documented as the starting
  point.
- **`options.advanced.*` namespace** — collapses `newton_options`,
  `timestep_config`, `lte_config`, `bdf_config`, `dc_config`,
  `stiffness_config`, `fallback_policy`, `formulation_mode` under one
  nested struct. Top-level field aliases stay for one release with
  deprecation warnings.
- **Damped-Newton line search** — Armijo backtracking inside the
  inner Newton iteration, fires when the residual norm increases.
  Currently we damp the *step* but never *backtrack mid-iteration*;
  multilevel topologies need both.
- **Simultaneous event detection** — when N switches cross threshold
  within one timestep, the PWL engine detects them all at the earliest
  crossing instant instead of processing them one at a time. Currently
  serialised event handling causes Newton non-convergence on MMC with
  hundreds of submodules.
- **Iterative refinement on the linear stage** — if the KLU residual
  norm exceeds a threshold (large floating-cap networks produce
  ill-conditioned MNA), apply one round of iterative refinement
  automatically. Hidden from the user; logged in telemetry.
- **Homotopy continuation on DC OP** — when `DCStrategy::Auto` exhausts
  Direct + Gmin + Source-stepping + Pseudo-transient, try a homotopy
  path on the nonlinear terms (smoothly turn on diodes / MOSFETs from
  off to behavioral). Fixes hard cold-start cases on multilevel.
- **`numerical/` directory** — moves the scattered numeric headers
  into `core/include/pulsim/v1/numerical/{preset.hpp, integrator.hpp,
  newton.hpp, linear_solver.hpp, dc_strategy.hpp, timestep_control.hpp,
  stiffness.hpp, formulation.hpp}`. Old paths keep redirecting
  `#include`s through deprecated headers for one release.
- **MMC topology template** — `pulsim::v1::templates::mmc(...)` builder
  generating an N-submodule three-phase MMC with arm inductors and
  pole-to-ground caps. Used by the new multilevel benchmark.
- **Multilevel benchmark suite** — 4 reference circuits (3-level NPC,
  T-type, 5-level flying-cap, 9-level MMC) with PLECS / PSIM golden
  waveforms checked in. Closes the "compete with PLECS / PSIM" goal.

### MODIFIED

- **`Integrator` enum** — keeps only `Trapezoidal`, `BDF1`, `BDF2`,
  `TRBDF2`, `RosenbrockW`. **BREAKING** removal of `BDF3`, `BDF4`,
  `BDF5`, `Gear`, `SDIRK2` in v2 schema (one-release deprecation in
  v1).
- **Default `Integrator`** — flips from `Trapezoidal` to `TRBDF2` when
  user goes through `Preset::Auto`. Raw `SimulationOptions{}` still
  defaults to `Trapezoidal` for back-compat.
- **`LinearSolverKind` user surface** — exposes only `Auto` (alias
  for the platform-best direct solver), `Direct`, and `Iterative`.
  Internal enum still distinguishes `SparseLU` / `EnhancedSparseLU` /
  `KLU` / `GMRES` / `BiCGSTAB` / `CG` for the auto-selector, but those
  values stop being part of the documented user API.
- **Preconditioner enum** — replaced by a single `solver_quality:
  Fast|Default|Best` knob on `options.advanced.linear_solver`. The
  enum (`None_`, `Jacobi`, `ILU0`, `ILUT`, `AMG`) becomes an internal
  type the auto-selector picks.
- **`DCStrategy` user surface** — collapsed to `Auto` (default) and
  `Override`. `Direct`, `GminStepping`, `SourceStepping`,
  `PseudoTransient` still exist as overrides under
  `options.advanced.dc.strategy` for power users.
- **`adaptive_timestep` (bool)** — **DEPRECATED**, emits a warning at
  load. `step_mode: Fixed|Variable` is the canonical replacement.
- **`direct_formulation_fallback` (bool)** — **DEPRECATED**, always
  on internally. No user-facing knob.
- **`SwitchingMode::Auto`** — flips its resolution default from
  `Behavioral` to `Ideal` (the PWL roadmap flip mentioned in
  `docs/pwl-switching-migration.md`). **BREAKING** behavioral
  change for any circuit relying on the implicit Behavioral fallback.

### REMOVED

- **`Integrator::BDF3`, `BDF4`, `BDF5`, `Gear`, `SDIRK2`** — dead
  code paths, unstable for switching topologies.
- **`SimulationOptions::adaptive_timestep` field** — redundant with
  `step_mode`.
- **`SimulationOptions::direct_formulation_fallback` field** —
  always-on internally, no user value in toggling it.
- **`LinearSolverStackConfig::auto_select`,
  `LinearSolverStackConfig::size_threshold`,
  `LinearSolverStackConfig::nnz_threshold`** — auto is always on; no
  user-facing thresholds.

## Impact

### Affected specs

- `kernel-v1-core` — adds `Preset`, multilevel-convergence
  requirements, deprecates 4 fields.
- `transient-timestep` — removes 5 dead integrators, adds line-search
  + simultaneous event detection requirements.
- `linear-solver` — replaces preconditioner taxonomy with
  `solver_quality`, adds iterative-refinement requirement.
- `dc-operating-point` — collapses 5-strategy enum to `Auto +
  Override`, adds homotopy continuation requirement.
- `python-bindings` — `SimulationOptions` Python class gains
  `from_preset()` factory + `advanced.*` namespace + deprecates 4
  top-level fields.
- `netlist-yaml` — `simulation:` block schema gains `preset:` key,
  deprecates `adaptive_timestep`.

### Affected code

- **Reorganization**: moves `simulation.hpp`, `transient_services.hpp`,
  `convergence_aids.hpp`, `high_performance.hpp`, `integration.hpp`
  numerical pieces into `core/include/pulsim/v1/numerical/`. Keeps
  deprecated forwarder headers at old paths for one release.
- **New code**: `numerical/preset.hpp` (Preset enum + factory),
  `numerical/line_search.hpp` (Armijo backtracking),
  `numerical/event_detector.hpp` (simultaneous crossing detection),
  `numerical/iterative_refinement.hpp` (KLU refinement wrapper),
  `numerical/homotopy.hpp` (DC OP homotopy path).
- **New template**: `core/include/pulsim/v1/templates/mmc.hpp` for the
  MMC builder.
- **New tests**: `test_preset.cpp`, `test_line_search.cpp`,
  `test_simultaneous_events.cpp`, `test_iterative_refinement.cpp`,
  `test_homotopy_dc.cpp`, `test_multilevel_npc.cpp`,
  `test_multilevel_ttype.cpp`, `test_multilevel_flying_cap.cpp`,
  `test_multilevel_mmc.cpp`.
- **New benchmarks**: `benchmarks/multilevel/` golden waveforms vs
  PLECS / PSIM exports.
- **Python bindings**: ~200 LOC of pybind11 for `Preset`,
  `SimulationOptions.from_preset(...)`, `options.advanced` namespace.
- **YAML parser**: adds `preset:` key handling, deprecation warnings
  for removed fields, validation of `advanced.*` sub-tree.
- **Docs**: rewrites `docs/numerical-modes-audit.md` from "audit /
  plan" to "shipped — numerical configuration guide"; updates
  `docs/configuration.md`, `docs/getting-started.md`,
  `docs/convergence-tuning-guide.md`, and every notebook /
  python-example that constructs a `SimulationOptions`.

### Migration impact for users

| User type | Impact |
|---|---|
| New user starting fresh | Just learns `Preset.Auto` + 3 essentials. Massively smaller surface. |
| User of `make_robust_options(...)` | Aliased to `Preset.Robust` — no code change. |
| User explicitly setting deprecated fields | Gets a warning for one release, then breaks. Migration guide ships with the change. |
| User using `BDF3`/`BDF4`/`BDF5`/`Gear`/`SDIRK2` | Warning in v1, error in v2. We don't believe any production user picks these — the audit found zero benchmark coverage. |
| User relying on `SwitchingMode::Auto` → Behavioral | **BREAKING**: now defaults to Ideal. Override with `Behavioral` explicitly to preserve old behavior. |

### Benchmark target

After this change, Pulsim must pass:

- **3-level NPC at 10 kHz, 600 V DC link, 50 Hz output, 5 kW load** —
  within 0.5% RMS error vs PLECS golden CSV
- **5-level flying-cap at 5 kHz, 800 V DC, 100 Hz output, 10 kW** —
  within 0.5% RMS error vs PLECS
- **9-submodule MMC half-arm at 2 kHz arm-switching, 400 V/sub, 50 Hz
  output, balanced cap voltages** — within 1% RMS error vs PSIM
- **All four** converge from cold start (zero state) without manual
  intervention, with `Preset.Auto`, in under 2× the PLECS wall-clock.
