## Context

The current numerical layer evolved over ~12 OpenSpec changes
(`refactor-pwl-switching-engine`, `improve-convergence-algorithms`,
`add-advanced-solvers`, `refactor-cpp23-high-performance`,
`add-sundials-direct-dae-runtime`, etc.). Each change added the
fields, enums, and config structs it needed without a holistic owner,
so the result is a 50-field union of well-meaning additions with
overlapping defaults and no orchestrator.

The audit in `docs/numerical-modes-audit.md` is the working
inventory; this design document explains *how* we slim the surface
without losing capability, *how* we make multilevel converters
converge, and *how* we reorganize so the next layer of improvements
(SUNDIALS hooks, GPU offload, JIT-compiled stamping) has a home.

Three stakeholders:

- **End user (95% case)**: Wants one decision: "I have a buck" / "I
  have a motor drive" / "I'm validating against SPICE". Then run.
- **Power user (4% case)**: Already knows what TRBDF2 + KLU +
  `dt_max = 1e-5` means. Wants direct access without ceremony.
- **Pulsim contributor (1% case)**: Adding a new integrator, a new
  DC strategy, or porting SUNDIALS. Needs a clear extension point
  and a single `numerical/` directory to grep.

## Goals / Non-Goals

### Goals

- **G1**: Reduce user-visible config surface from ~50 to ~6 top-level
  fields, with a `Preset` enum covering 95% of cases in one call.
- **G2**: Eliminate dead integrators (`BDF3`, `BDF4`, `BDF5`, `Gear`,
  `SDIRK2`) — they fail in switching topologies and have zero
  benchmark coverage.
- **G3**: Eliminate redundant booleans (`adaptive_timestep`,
  `direct_formulation_fallback`) — single source of truth per axis.
- **G4**: Pass cold-start convergence on 4 reference multilevel
  topologies (3-level NPC, T-type, flying-cap, 9-level MMC) with no
  manual tuning, matching PLECS / PSIM golden waveforms within 0.5%
  RMS.
- **G5**: Move the scattered numerical code into a single
  `core/include/pulsim/v1/numerical/` directory so the next set of
  improvements has an obvious home.
- **G6**: Keep one-release back-compat for every deprecation —
  deprecated fields emit warnings, old `#include` paths forward to
  new ones via deprecated header stubs.

### Non-Goals

- **NG1**: We are NOT removing the underlying engines (KLU, GMRES,
  SparseLU, Newton-Raphson, trapezoidal, BDF1/2, etc.). They keep
  working. We are only collapsing the *user-facing* surface that
  exposes them.
- **NG2**: We are NOT porting to SUNDIALS in this change. The
  reorganization clears the way for the existing
  `add-sundials-direct-dae-runtime` change, but SUNDIALS hooks land
  in that change, not this one.
- **NG3**: We are NOT adding GPU / SIMD acceleration in this change.
- **NG4**: We are NOT replacing the PWL switching engine. PWL stays
  as the dominant fast path; this change improves event detection
  inside the existing PWL engine but does not rewrite it.

## Decisions

### D1 — Single `Preset` enum + factory, additive

**Decision**: Add `Preset { Auto, Fast, Robust, HighFidelity }` plus a
static factory `SimulationOptions::from_preset(Preset, dt, tstop)`.
Keep the raw `SimulationOptions{}` constructor unchanged.

**Why**: The current "two defaults" split (raw vs `make_robust_options`)
already exists; we just promote `make_robust_options` to a first-class
named factory and add three sibling profiles. Existing code keeps
compiling; we don't have to migrate every caller in one PR.

**Alternatives considered**:

- *Builder pattern* (`SimulationOptions::builder().tstop(...).build()`):
  more idiomatic in C++23 but Python bindings get ugly. Rejected.
- *Plain functions* (`make_fast_options(dt, tstop)`): no central
  registry, hard to introspect. Rejected — we want the enum so users
  can do `for preset in Preset: ...`.

### D2 — Collapse advanced fields under one `advanced.*` struct

**Decision**: `SimulationOptions` gains a new field
`AdvancedOptions advanced{};` that owns `newton_options`,
`timestep_config`, `lte_config`, `bdf_config`, `dc_config`,
`stiffness_config`, `fallback_policy`, `formulation_mode`,
`linear_solver`. Top-level field aliases (`opts.newton_options.*`,
etc.) stay for one release, forwarding to `opts.advanced.*` with
deprecation warnings.

**Why**: Discoverability. Today, `dir(opts)` in Python shows 28 top-
level fields; after this change it shows ~6, with `opts.advanced.*`
the explicit power-user namespace.

**Alternatives considered**:

- *Hide advanced under a separate object* (`opts.tune.newton.*`): more
  ceremony, less Python-idiomatic. Rejected.
- *Move advanced fields to a sibling class* (`AdvancedSimulationOptions
  adv; ps.Simulator(ckt, opts, adv)`): breaks the single-arg
  `Simulator(ckt, opts)` contract. Rejected.

### D3 — Deprecate-not-remove the redundant booleans (one release)

**Decision**: `adaptive_timestep` and `direct_formulation_fallback`
stay readable / writable for one release with a runtime warning
logged on access. v2 removes them.

**Why**: Existing benchmarks, examples, notebooks set these. A hard
removal breaks the CI matrix on day one.

### D4 — Damped Newton with Armijo line search

**Decision**: Add an inner backtracking line-search loop inside Newton.
Each iteration:

```
solve  J · Δx = −r            // current Newton step
α = 1.0
while α > α_min:
    x_trial = x + α · Δx
    r_trial = residual(x_trial)
    if ||r_trial|| < (1 − σ·α) · ||r||:
        break
    α *= 0.5
x = x_trial
```

with `σ ≈ 1e-4` (Armijo constant), `α_min = 2^-8 = 0.004`.

**Why**: Current Newton damps the *step* (`opts.advanced.newton.
initial_damping`) but never checks whether the trial state actually
*reduced* the residual. On multilevel topologies the step direction
can be valid but oversized — backtracking recovers without rejecting
the whole iteration.

**Cost**: ~2× residual evaluations per Newton step in the worst case,
zero overhead in the common case (α = 1.0 accepted immediately).

**Alternatives considered**:

- *Trust region* (Dogleg, Levenberg-Marquardt): higher implementation
  cost, larger memory footprint per Newton iteration. Rejected for
  this change; revisit if Armijo alone doesn't close the multilevel
  gap.

### D5 — Simultaneous event detection in the PWL engine

**Decision**: When ≥ 2 switches cross threshold in a single timestep,
the PWL engine:

1. Computes the crossing instant `t*_i` for each switch via linear
   interpolation on the pre / post-step branch currents / node
   voltages.
2. Sorts by `t*_i`.
3. Takes ALL events whose crossing falls within `t*_min + ε` (where
   `ε = 1e-12 · dt`) as simultaneous.
4. Applies all simultaneous switch-state changes atomically, then
   does ONE Newton solve at `t*_min`.
5. Continues the timestep from `t*_min` to `t_now + dt`.

Currently we process one event at a time and re-Newton; on MMC with
hundreds of submodules switching at the same arm-PWM edge, the
serialised re-Newton loop fails to converge.

**Why**: Real hardware switches simultaneously when commanded
simultaneously. Serialising the events is a numerical artifact that
breaks MMC and any other dense-switching topology.

**Cost**: Algorithmic — one extra sort per step. Negligible for typical
< 50 switches per step.

### D6 — Iterative refinement on KLU when ill-conditioned

**Decision**: After KLU back-solve, compute residual `r = b − A·x`.
If `||r||/||b|| > 10·ε_machine`, do one round of iterative refinement
(`solve A·δ = r; x ← x + δ`). Hidden from user; counter exposed in
telemetry as `linear_refinement_steps`.

**Why**: Floating-cap topologies have cap-to-cap loops that produce
ill-conditioned KCL submatrices. KLU's partial pivoting handles them
but the back-substitution accumulates round-off. One refinement step
recovers ≥ 5 decimal digits at ~10% overhead.

**Cost**: ~10% per timestep when triggered; zero when not.

### D7 — Homotopy continuation as last-resort DC OP

**Decision**: When `DCStrategy::Auto` exhausts the existing four
strategies (Direct → Gmin → SourceStepping → PseudoTransient), try
**homotopy on the nonlinear devices**:

- Parameter λ ∈ [0, 1]
- At λ = 0: all diodes / MOSFETs / IGBTs replaced by their **linear
  off-state conductance** (`g_off`) — purely linear MNA, solves in
  one direct solve.
- At λ = 1: full nonlinear behavioural model.
- Step λ from 0 → 1 in 5-10 increments, doing one Newton solve at
  each λ with the previous solution as warm start.

**Why**: Hard cold-start cases on multilevel (especially MMC with all
caps initially equal) often hang at PseudoTransient because the
nonlinear residual landscape has too many local minima. Homotopy
walks the solution path continuously and is provably convergent for
diode / FET models if started linear.

**Alternatives considered**:

- *Continuation on V_in / I_in*: equivalent to SourceStepping which we
  already have. Rejected — adds nothing.
- *Continuation on Gmin*: equivalent to GminStepping. Rejected.

### D8 — Numerical directory reorganization

**Decision**: Move (or forward) these headers:

```
core/include/pulsim/v1/numerical/
├── preset.hpp                  (NEW)
├── integrator.hpp              (moved from integration.hpp; old path forwards)
├── newton.hpp                  (extracted from convergence_aids.hpp)
├── line_search.hpp             (NEW)
├── linear_solver.hpp           (extracted from high_performance.hpp)
├── iterative_refinement.hpp    (NEW)
├── dc_strategy.hpp             (moved from convergence_aids.hpp; old path forwards)
├── homotopy.hpp                (NEW)
├── timestep_control.hpp        (moved from transient_services.hpp; old path forwards)
├── stiffness.hpp               (extracted from convergence_aids.hpp)
├── event_detector.hpp          (NEW — simultaneous event detection)
├── formulation.hpp             (extracted from simulation.hpp)
└── advanced_options.hpp        (NEW — the `advanced.*` namespace struct)
```

`SimulationOptions` (in `simulation.hpp`) gets thin wrapper structs:
`AdvancedOptions { ... };` that aggregates everything in
`numerical/advanced_options.hpp`.

**Why**: Today, finding "where Newton is implemented" requires
grepping across 5 files. After this, `core/include/pulsim/v1/numerical/`
is the obvious place. The forwarder headers at old paths preserve
back-compat for one release.

**Migration cost**: ~30 `#include` updates in our own codebase;
zero downstream impact because old paths keep working.

### D9 — `LinearSolverKind` user surface

**Decision**: The public Python / YAML enum becomes:

```
LinearSolverKind {
    Auto,        # default — let the auto-selector pick
    Direct,      # forces SparseLU / EnhancedSparseLU / KLU (best of)
    Iterative,   # forces GMRES / BiCGSTAB / CG (best of)
}
```

The full 6-value enum still exists internally for the auto-selector,
but is not part of the documented user API. Same for the preconditioner
enum (`None_`, `Jacobi`, `ILU0`, `ILUT`, `AMG`) which is replaced by a
single `solver_quality: Fast|Default|Best` knob in
`options.advanced.linear_solver`.

**Why**: A user simulating a buck converter does not need to learn the
sparse-linear-algebra taxonomy. The auto-selector already does the
right thing today; we just stop exposing the underlying choices.

### D10 — `DCStrategy` user surface

**Decision**: Public enum becomes `DCStrategy { Auto, Override }`. The
full 5-value enum (`Direct`, `GminStepping`, `SourceStepping`,
`PseudoTransient`, `Homotopy`) is accessible under
`options.advanced.dc.strategy_override` for power users.

`Auto` orchestrates the strategies in this order:

1. Direct (one Newton from `x0 = 0`)
2. Source-stepping (ramp V/I sources 0 → nominal)
3. Gmin-stepping (start with large parallel conductance, ramp down)
4. Pseudo-transient continuation
5. **Homotopy** (new — D7)

**Why**: Same reason as D9 — the user does not need to know the
strategy taxonomy to get an operating point.

### D11 — `SwitchingMode::Auto` flips its resolution to `Ideal`

**Decision**: `Auto` resolves to `Ideal` (the PWL fast path) unless the
circuit contains a device that has explicitly opted into
`Behavioral` via the device-level override. This is the long-promised
flip in `docs/pwl-switching-migration.md`.

**Why**: PWL Ideal is faster, more accurate (zero Newton iters in
stable topos), and has been the recommended path for two releases.
Defaulting to it lets new users get the better path automatically.

**Back-compat**: Users who relied on the implicit Behavioral fallback
must either:

- Set `opts.switching_mode = SwitchingMode::Behavioral` explicitly, or
- Set `circuit.set_default_switching_mode(SwitchingMode::Behavioral)`

This is the one **BREAKING** change in this proposal. Documented with
a migration note in `docs/pwl-switching-migration.md`.

### D12 — MMC topology template + golden benchmarks

**Decision**: Add `pulsim::v1::templates::mmc(num_submodules, V_dc,
f_arm, ...)` returning a `Circuit` pre-wired with:

- 6 arms (upper + lower for each of 3 phases)
- N submodules per arm, each = a half-bridge MOSFET pair with floating
  capacitor
- Arm inductors
- Optional pole-to-ground capacitors and grid-side L-filter

Plus 4 reference benchmarks:

- `benchmarks/multilevel/3level_npc.yaml` — 3-level Neutral-Point-
  Clamped, 600 V DC, 10 kHz carrier, 50 Hz output, 5 kW load
- `benchmarks/multilevel/5level_flying_cap.yaml` — 800 V DC, 5 kHz,
  100 Hz, 10 kW
- `benchmarks/multilevel/ttype_3level.yaml` — T-type, same DC link
- `benchmarks/multilevel/mmc_9sub.yaml` — 9 submodules per arm, 400 V
  per sub, 2 kHz arm switching, balanced cap voltages

Each ships with a `golden_*.csv` exported from PLECS (or PSIM for
MMC) and a `test_multilevel_*.cpp` that asserts ≤ 0.5% RMS error
versus the golden.

**Why**: Closes the "compete with PLECS / PSIM" goal with measurable
gates. Without a benchmark, "we support multilevel" is just a claim.

## Risks / Trade-offs

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Breaking change on `SwitchingMode::Auto` → Ideal surprises existing users | High | Med | Migration note in pwl-switching-migration.md, warning logged for one release when a circuit's resolved mode flips |
| `from_preset(Robust)` is slower than today's `make_robust_options()` because of extra checks | Med | Low | Benchmark before / after; iterate-refinement and event-detector only fire when triggered |
| Iterative refinement on KLU adds nondeterminism (different binaries refine differently) | Low | Med | Tie threshold to `ε_machine`, log refinement steps in telemetry; determinism tests gate on counter, not on bitwise output |
| Multilevel benchmarks vary across PLECS versions | Med | Med | Pin to a single PLECS export + version-tag the golden CSVs |
| Numerical directory reorganization breaks downstream `#include`s | High | Low | Forwarder headers at old paths for one release with `#pragma message` warning |
| Line-search may double-count residual evaluations in user telemetry | Low | Low | Add a `line_search_backtracks` counter; existing `newton_residual_evaluations` stays as-is |

## Migration Plan

### Phase 1 — Reorganization (no behavior change)

1. Create `core/include/pulsim/v1/numerical/` directory + new headers.
2. Move existing types into the new headers; leave forwarder stubs
   at old paths with `#pragma message("deprecated, use numerical/")`.
3. Update internal `#include`s.
4. Run full test suite — must be green with zero behavior change.

### Phase 2 — `Preset` + `advanced.*` surface (additive)

1. Implement `Preset` enum + `SimulationOptions::from_preset(...)`.
2. Implement `AdvancedOptions` aggregate; top-level field aliases
   forward to `advanced.*` with deprecation warnings.
3. Add Python bindings + YAML `preset:` key.
4. Update docs + notebooks + examples to use `Preset`.

### Phase 3 — Convergence hardening

1. Add Armijo line search inside Newton.
2. Add simultaneous event detection in PWL engine.
3. Add iterative refinement on KLU.
4. Add homotopy continuation as the 5th DC strategy.
5. Run existing convergence test suite — must still be green; should
   pick up wins on previously-flaky tests.

### Phase 4 — Multilevel templates + benchmarks

1. Implement `templates::mmc(...)` builder.
2. Add 4 YAML benchmark files.
3. Export PLECS / PSIM golden CSVs (manual one-time step).
4. Add 4 `test_multilevel_*.cpp` tests gating on ≤ 0.5% RMS error.

### Phase 5 — Deprecations + flip

1. Mark `BDF3`/`BDF4`/`BDF5`/`Gear`/`SDIRK2` as deprecated in the
   enum (still parseable; warns).
2. Mark `adaptive_timestep` and `direct_formulation_fallback`
   deprecated.
3. Flip `SwitchingMode::Auto` resolution from Behavioral to Ideal.
4. Update release notes; ship as a minor-version bump.

### Phase 6 — Removal (next release after deprecation cycle)

1. Drop the 5 deprecated integrator values from the enum.
2. Drop the deprecated bool fields.
3. Drop the forwarder headers at old paths.

## Open Questions

- **Q1**: Should `Preset.HighFidelity` use Richardson LTE or
  step-doubling LTE? Richardson is 3× cheaper but slightly less
  accurate. Lean: step-doubling for the highest preset.
- **Q2**: For the MMC benchmark, do we use PSIM golden (more
  industry-standard for MMC) or generate our own reference via a
  high-resolution Pulsim run with extremely tight tolerances? Lean:
  PSIM for v1; revisit when PSIM licensing constrains CI.
- **Q3**: Should homotopy be `Preset.Robust`'s default, or only fire
  in `Preset.HighFidelity`? Lean: included in `Robust` but with a
  short λ ladder (5 steps); `HighFidelity` uses a longer ladder (10
  steps).
- **Q4**: Does the line-search trigger inside DC OP iterations too, or
  only in transient Newton? Lean: both; the cost is the same.
