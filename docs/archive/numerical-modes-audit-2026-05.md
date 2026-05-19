# Numerical Execution Modes — Audit & Simplification Plan

> **Internal working document.** Catalogues every user-selectable
> "compute mode" / "execution path" we expose today, scores them by
> quality + actual usage, and proposes the slimmed surface that should
> ship to end users. Not yet wired into the nav — this is the planning
> document for a follow-up simplification PR.

## TL;DR

A user who opens `SimulationOptions` today has to make **~50 separate
choices** distributed across 7 axes. About **6 of them matter**; the
rest are either undocumented, redundant, or wrong-by-default. The
recommendation is to collapse the surface to:

1. One `Preset` enum (`Auto` / `Fast` / `Robust` / `HighFidelity`) +
2. Keep the 7 axes as **advanced overrides** behind a single
   `options.advanced.*` namespace + 3 escape-hatch fields users always
   want (`tstop`, `dt`, `switching_mode`).

Everything else moves to documented profiles (`make_robust_options`,
`make_fast_options`, etc.) and stops being a top-level kwarg.

## Inventory — every choice the user can make today

### Axis 1 — Switching path

| Value | Default? | Quality | Where |
|---|---|---|---|
| `SwitchingMode::Auto` | ✅ | Resolves to `Behavioral` today; will flip to `Ideal` in the roadmap | `components/base.hpp:??` |
| `SwitchingMode::Ideal` | | ⭐⭐⭐⭐⭐ — PWL state-space fast path, zero Newton iters in stable topos | |
| `SwitchingMode::Behavioral` | | ⭐⭐⭐ — Smooth tanh stamps, Newton-per-step. Works everywhere but slow | |

**3 values → only 2 real paths.** `Auto` is currently a one-line alias
for `Behavioral`.

### Axis 2 — Integrator (time stepping)

`Integrator` enum (`integration.hpp`):

| Value | Default? | Quality | Use case |
|---|---|---|---|
| `Trapezoidal` | ✅ (raw `SimulationOptions{}`) | ⭐⭐⭐⭐ A-stable order 2 — the textbook standard | Power-electronics default |
| `BDF1` | | ⭐⭐⭐ L-stable order 1 — backward Euler | Stiff fallback |
| `BDF2` | | ⭐⭐⭐⭐ A-stable order 2 | General-purpose stiff |
| `BDF3` | | ⭐⭐ Order 3 — rarely used in switching topologies | Smooth dynamics only |
| `BDF4` | | ⭐ Order 4 — unstable in switching transients | Theoretical only |
| `BDF5` | | ⭐ Order 5 — unstable in switching transients | Theoretical only |
| `Gear` | | ⭐ — Alias for `BDF2`. **Duplicate.** | Back-compat |
| `TRBDF2` | ✅ (`make_robust_options()`) | ⭐⭐⭐⭐⭐ — Trapezoidal + BDF2 hybrid, multi-stage | **Power-electronics default we should advertise** |
| `RosenbrockW` | | ⭐⭐⭐ Linearly-implicit, no Newton inner loop | Specialised stiff |
| `SDIRK2` | | ⭐⭐ Singly-DIRK order 2 | Specialised stiff |

**10 values → ~3 that are actually production-ready** (`Trapezoidal`,
`BDF2`, `TRBDF2`). `BDF3/4/5` and `SDIRK2` are research vestiges.
`Gear` is a duplicate of `BDF2`. Default disagrees between raw API
(`Trapezoidal`) and "robust" preset (`TRBDF2`).

### Axis 3 — Linear solver

`LinearSolverKind` enum (`high_performance.hpp`):

| Value | Default? | Quality | Notes |
|---|---|---|---|
| `SparseLU` | ✅ (raw default) | ⭐⭐⭐ Native Eigen, reliable, slow at scale | OK up to ~1k nodes |
| `EnhancedSparseLU` | | ⭐⭐ Improved pivoting | Marginal benefit over SparseLU |
| `KLU` | ✅ (`make_robust_options()`) | ⭐⭐⭐⭐⭐ — Fast, sparse-friendly | **Should be default** |
| `GMRES` | (fallback) | ⭐⭐⭐⭐ Iterative — needed for very large systems | Restart=40 |
| `BiCGSTAB` | | ⭐⭐⭐ Iterative — alternative to GMRES | |
| `CG` | | ⭐⭐ Symmetric only — almost never matches MNA structure | Rare |

Plus **3 stack-config knobs** (`order`, `fallback_order`,
`auto_select`), 2 thresholds (`size_threshold`, `nnz_threshold`), and
**5 preconditioners** (`None_`, `Jacobi`, `ILU0`, `ILUT`, `AMG`). The
user that just wants "use whatever's fast" has to learn the entire
sparse-linear-algebra taxonomy.

### Axis 4 — Newton

Single class `NewtonOptions` with ~10 knobs:
`max_iterations`, `initial_damping`, `min_damping`, `auto_damping`,
`enable_limiting`, `max_voltage_step`, `max_current_step`,
`track_history`, `check_per_variable`, plus `num_nodes`/`num_branches`
that should *never* be user-set (they get auto-filled from the circuit).

**Quality**: the core Newton is solid. The problem is the surface — no
"profile" picker, so users either accept defaults blindly or have to
tune a 10-field struct.

### Axis 5 — Step-mode (timestep control)

`TransientStepMode` enum (`transient_services.hpp`):
- `Fixed` — Deterministic uniform `dt`
- `Variable` — Adaptive (default)

PLUS three sub-configs that affect the same axis:

| Sub-config | Purpose |
|---|---|
| `AdvancedTimestepConfig` | Newton-feedback + LTE-weighted adaptive (5 weights/thresholds) |
| `RichardsonLTEConfig` | Step-doubling vs Richardson extrapolation for LTE estimation |
| `BDFOrderConfig` | Min/max BDF order + auto-adaptation |

Two enums that overlap with the high-level mode (`adaptive_timestep`
boolean is the legacy back-compat field; `step_mode` is canonical).
**User can set conflicting values without warning.**

### Axis 6 — Formulation

`FormulationMode` enum (`simulation.hpp`):
- `ProjectedWrapper` (default) — DAE wrapper, robust on mixed-domain
- `Direct` — Plain DAE, faster but fragile on hybrid blocks

Plus `direct_formulation_fallback` boolean. **In practice nobody picks
`Direct` explicitly** — it's diagnostic-only.

### Axis 7 — DC operating point strategy

`DCStrategy` enum (`convergence_aids.hpp`):
- `Direct`
- `GminStepping`
- `SourceStepping`
- `PseudoTransient`
- `Auto` (default) — orchestrates the 4 above

Each strategy has its own sub-config struct (`GminConfig`,
`SourceSteppingConfig`, `PseudoTransientConfig`,
`InitializationConfig`) with 3-5 knobs each. Total ~20 DC-OP fields
that 95% of users never touch.

### Axis 8 — Analysis pipelines (`Simulator::run_*`)

Not a config field but a method choice — five entry points:

| Method | Purpose | Quality |
|---|---|---|
| `run_transient` | Time-domain | ⭐⭐⭐⭐⭐ — the workhorse |
| `run_periodic_shooting` | Period steady-state via shooting | ⭐⭐⭐ — works but undocumented in Python examples |
| `run_harmonic_balance` | Frequency-domain periodic | ⭐⭐⭐ — research-grade |
| `run_ac_sweep` | Linear AC sweep | ⭐⭐⭐⭐⭐ — well-tested |
| `run_fra` | Frequency response via transient + DFT | ⭐⭐⭐⭐⭐ — well-tested |

These five are legitimately different analyses, not redundant
implementations of the same thing. Keep them all.

### Axis 9 — Stiffness mitigation

`StiffnessConfig` with ~6 fields (`enable`, three thresholds,
`switch_integrator`, `stiff_integrator`, `dt_backoff`,
`monitor_conditioning`, `cooldown_steps`). Auto-switches integrator
behind the user's back. **Helpful but invisible** — users see
their `integrator: Trapezoidal` choice silently get overridden mid-run.

---

## Quality issues

1. **Default split-brain.** `SimulationOptions()` defaults differ
   between raw C++ (`Trapezoidal`, `SparseLU`, `Variable`) and the
   `make_robust_options()` preset (`TRBDF2`, `KLU` stack, full
   robustness). Users who learn from one path get burned switching
   to the other.

2. **Dead code paths.** `BDF3`/`BDF4`/`BDF5`/`SDIRK2`/`Gear` are
   essentially never exercised in production benchmarks. They inflate
   the API surface for zero user value.

3. **Overlapping axes.** `adaptive_timestep` (bool) and `step_mode`
   (enum) control the same thing. `direct_formulation_fallback` + the
   `formulation_mode` enum overlap. Setting one inconsistent with the
   other produces silent surprises.

4. **No "preset" entry point.** A new user has to either:
   - Accept the raw `SimulationOptions{}` (which is slow + fragile)
   - Discover `make_robust_options(...)` (Python-only, undocumented in
     the components reference)
   - Hand-tune 50 fields

5. **Auto-promotion hides decisions.** `SwitchingMode::Auto` + the
   stiffness fallback both silently override user choices. Logged but
   not surfaced.

6. **Preconditioner taxonomy leak.** The user shouldn't need to know
   what ILU0 vs ILUT is to simulate a buck converter.

---

## Recommendation — proposed user-facing surface

### Tier 1 — what most users should see

A single `Preset` choice + 4 essentials:

```python
opts = pulsim.SimulationOptions(
    preset = pulsim.Preset.Auto,    # NEW — replaces the 50-field tuning
    tstop = 1e-3,
    dt = 1e-6,
    switching_mode = pulsim.SwitchingMode.Ideal,   # opt-in PWL
)
```

**`Preset` enum (proposed):**

| Value | Equivalent today | When to use |
|---|---|---|
| `Auto` (default) | `make_robust_options()` w/ Behavioral | Unknown circuit — let the simulator pick |
| `Fast` | PWL Ideal + Trapezoidal + KLU + fixed-step | Pure switching topology, no nonlinear devices |
| `Robust` | TRBDF2 + KLU + Variable + stiffness on + retries | Complex nonlinear (motors, magnetics, thermal) |
| `HighFidelity` | TRBDF2 + tighter LTE tolerance + dt_max small | Validation runs, parity checks against SPICE |

That covers 95% of users with **one decision**.

### Tier 2 — explicit overrides

Users who know what they need keep direct access to the canonical
fields, but the API doesn't lead with them:

```python
opts.tstop = 1e-3
opts.dt = 1e-6
opts.switching_mode = ps.SwitchingMode.Ideal
opts.integrator = ps.Integrator.TRBDF2           # advanced
opts.linear_solver_kind = ps.LinearSolverKind.KLU # advanced
```

### Tier 3 — power-user namespace (`options.advanced.*`)

Everything else moves under one namespace:

```python
opts.advanced.newton.max_iterations = 80
opts.advanced.timestep.error_tolerance = 5e-3
opts.advanced.dc.strategy = ps.DCStrategy.PseudoTransient
opts.advanced.stiffness.enable = False
opts.advanced.formulation = ps.FormulationMode.ProjectedWrapper
```

This stops `SimulationOptions` from being a 50-field god-class.

### Tier 4 — deprecate / hide

- **Remove from public enum:** `BDF3`, `BDF4`, `BDF5`, `Gear`, `SDIRK2`
  (move to a debug/research namespace, document as "not supported").
- **Mark `adaptive_timestep` deprecated** (replaced by `step_mode`).
- **Drop preconditioner enum from public surface** — let the linear
  solver auto-pick. Expose `solver_quality: Fast|Default|Best` instead.
- **Hide `direct_formulation_fallback`** — just always have the
  fallback active.
- **Hide `auto_select` / `size_threshold` / `nnz_threshold`** — auto
  is on by default; no knob.

---

## Migration plan (proposed PR sequence)

| Step | Deliverable | Effort |
|---|---|---|
| 1 | Add `Preset` enum + `SimulationOptions.from_preset(...)` factory in C++ + Python | S |
| 2 | Update every doc example + notebook to use `Preset` | M |
| 3 | Add deprecation warnings on `adaptive_timestep`, `direct_formulation_fallback`, `auto_select`, BDF3+ | S |
| 4 | Move advanced fields under `options.advanced.*` namespace (additive; keep top-level aliases through one release) | M |
| 5 | Remove the 5 dead integrators (`BDF3/4/5`, `Gear`, `SDIRK2`) from the public enum after a deprecation cycle | S |
| 6 | Hide preconditioner enum behind a `solver_quality` selector | M |
| 7 | Add a "Numerical configuration" docs page that just teaches the 4 presets | S |

End state: a user-facing `SimulationOptions` with **~6 top-level
fields** (vs ~50 today) + one `advanced` namespace for the long tail.

---

## Quick-reference: what each "axis" is actually solving

For readers who want the physics behind the choices:

| Axis | Physical problem |
|---|---|
| Switching path | How to handle the discontinuity at a switch turn-on/off |
| Integrator | How to march `dx/dt = f(x)` through one timestep |
| Step mode | How big should `dt` be — fixed by user, or chosen by error metric? |
| Newton | How to converge the nonlinear residual at each timestep |
| Linear solver | How to solve `A·dx = -r` inside Newton |
| DC strategy | How to find the initial operating point before transient |
| Formulation | How the DAE is assembled (mixed-domain wrapping vs direct) |
| Stiffness | What to do when the integrator hits a stiff region mid-run |

Most users only need to make **one** of these decisions explicitly
(switching path: PWL or Behavioral). The other seven should be picked
by the preset.
