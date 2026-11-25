## Why

Between two events a piecewise-linear circuit is linear time-invariant,
and its trajectory has a closed form. `engine='dsed'` already steps it
that way (`core/include/pulsim/dsed/exact_lti.hpp`) and is the most
accurate and the cheapest path the kernel has for that class of circuit
(paths below are relative to `core/include/pulsim/` unless they start
with `python/`):

| Sync / diode buck, audit circuit | Error at 40 periods | Samples per period | 2000 periods (perf row 22) |
|---|---|---|---|
| `pwl`, fixed trapezoidal | 1.18e-2 A at T/50, 7.4e-4 A at T/800 (order 1.00) | 50 … 800 | 91 ms at dt = 1e-7, 7.0 ms at dt = 1e-6 |
| `trbdf2` | 8e-4 V at rtol 1e-4, 2e-6 V at rtol 1e-8 | 19 | 69 ms |
| `dsed` exact propagator | 6e-10 (CCM), 2e-8 (DCM) | 2 … 10 | 14.5 ms |

Sources: audit `raw/numerics.md` §1 and items 9, 10; `raw/perf.md` row
22, item 7 and question 3; probes `numerics/p2_buck.py`,
`numerics/p9_dsed_dcm.py`, `perf/p9_dsed.py`. The 2e-8 DCM figure is
mostly the reference's own error: the audit's reference used
`scipy.linalg.expm`, which is 3.3e-8 V off at the DCM fixed point; the
kernel's `dsed` reads 8.996247087 V after 4000 periods against the
50-digit value 8.99624708728943 V
(`probes/references_50_digits.py`).

It is nevertheless a side engine with holes, and 97 % of its run time is
not arithmetic:

- **Rest state with a zero band.** A PWL diode with `V_th = 0` in a
  circuit that starts with every switch open and every state at zero
  aborts: "diode configuration did not settle after 64 rounds at
  t = 0.000000" (`dsed/scheduler.hpp:925-957`; reproduced by
  `probes/vth0_rest_state.py`; `V_th = 1e-9` runs). At `v_D = 0`
  exactly the law turns the diode ON (`v ≥ V_on`) and OFF (`i > 0` is
  false) on alternate rounds. The message prints `std::to_string(t)`,
  six decimals.
- **DCM idle chatter for some parameter values.** The audit's DCM buck
  (`V_th = 1e-6`) runs at L = 100 µH but aborts when L is changed by a
  relative 1e-5, 1e-4 or 1e-2: "D1 is chattering at t = 0.000255 — it
  commutated 32 times inside 0.000000 s"
  (`../add-parameter-sensitivities/probes/fd_gradient_today.py`, which
  hit it while differencing two runs).
- **Builder initial conditions are refused**
  (`python/pulsim/__init__.py:2054-2082`): the reduced state of a
  floating capacitor is not a row of the MNA vector.
- **Event tolerance is an absolute 1e-10**, applied to both the time
  bracket and the predicate value (`dsed/event_predictor.hpp:115, 132,
  158, 253`), whatever the period or the unit of the predicate.
- **The eigenbasis is accepted on `isInvertible()`**
  (`dsed/exact_lti.hpp:76-79`). On a series RLC at critical damping
  `eig` returns an invertible basis with condition number 2.3e9 and the
  modal formula is 7.9e-9 off (relative); at 7e7 it is 2.3e-10 off
  (`probes/propagator_accuracy.py`).
- **φ₁ loses digits beside its series switch**: `(eˢ − 1)/s` is used
  from `|s| = 1e-6` (`dsed/exact_lti.hpp:90-97`), relative error 6.5e-11
  at `s = 1.01e-6` (same prototype).
- **Coupled inductors are refused** (`pwl/cache.hpp:991-1010`), PWM and
  pulse sources fall back to explicit RK45
  (`dsed/builder_adapter.hpp:352-365`), Newton devices are refused
  (`python/pulsim/_dsed_dispatch.py:578-585`), and the work per mode is
  dense O(n³).
- **Overhead**: 3.6 µs per step, of which the triangular solve is 3 %,
  malloc/free 28 % and Python/pybind 47.5 % (`raw/perf.md`, time-split
  table). `exact_advance` recomputes `V⁻¹b` and allocates four vectors
  per call (`dsed/exact_lti.hpp:100-114`); the event search calls it
  once per Illinois iteration and again to build the event
  (`dsed/scheduler.hpp:632-672`); samples are a `std::vector<Vector>`
  (`dsed/scheduler.hpp:89`) copied into per-sample Python lists
  (`python/pulsim/_dsed_dispatch.py:763-765, 829-831`).
- **The fixed engine cannot use it at all**: a buck step costs
  380–460 ns with 13 % in the solve (`raw/perf.md` row 1), where the
  update `x ← Φx + Γ` for two states is eight multiply-adds and has no
  truncation error (`probes/phi_gamma_fixed_step.py`: 2.5e-12 at
  T/100 against a 50-digit reference, where `pwl` is 1.18e-2 A at T/50).

## What Changes

- **A propagator kernel that states its own accuracy.** Per mask:
  eigendecomposition in energy coordinates, accepted only when
  `u·κ₂(V) ≤ 1e-10`; otherwise a Padé scaling-and-squaring exponential
  of the augmented matrix, accepted only when `u·‖A·h‖₁ ≤ 1e-10`;
  otherwise that mask is integrated by the implicit strategy. The tier
  used, with the reason, is in the result for every mask. φ₁ and φ₂ are
  evaluated through `expm1`.
- **Modal stepping.** Inside a segment the state stays in modal
  coordinates (`z ← e^{Λh}z + h·φ₁(Λh)·w`, O(n) per step); `w = V⁻¹b` is
  cached per mask; predicates are rows in modal coordinates (O(n) per
  Illinois iteration); the physical state is formed only for a recorded
  sample or an event. No heap allocation per step after warm-up.
- **Event location with a relative tolerance** on the time bracket and a
  scan step bounded by the fastest oscillatory mode that the predicate
  can see, so a double crossing inside one segment is not skipped.
- **Diode mode selection by derivative** when a crossing signal is zero
  to tolerance, replacing the value test that cannot decide a zero band;
  times are printed in scientific notation.
- **Builder initial conditions** mapped onto the reduced state through
  the per-device values (capacitor branch voltage, inductor current),
  with the charge-conserving projection when a capacitor loop makes them
  inconsistent.
- **Coupled inductors** through the inductance matrix in the reduced
  mass matrix; `|k| = 1` is refused by name.
- **Sources**: piecewise-constant sources (PWM, pulse) are exact, their
  breakpoints being time events; sine sources keep the existing
  oscillator augmentation; a piecewise-linear source uses the φ₂ term;
  an arbitrary `b_extra_fn` selects the implicit strategy, with the
  reason recorded.
- **Fixed-step discrete update** `x ← Φ_m·x + Γ_m` per mask with a
  cached `(mask, h)` pair and a dense fast path, as the fixed-step form
  of the same strategy; an event between grid points is taken by an
  exact sub-step.
- **Hybrid with Newton devices**: a device that reports an affine
  region is part of the mode key and is propagated exactly inside the
  region; a step in which any device is outside an affine region is an
  implicit step. Nothing is deleted or frozen.
- **Applicability rule** by state count and bytes; above it `auto` does
  not select the strategy and an explicit request is refused by name.
- **BREAKING:** explicit RK45 is no longer the fallback of the exact
  path; the fallback is the implicit (L-stable) strategy. `integrator=
  'rk45'` and `'bdf2'` on `engine='dsed'` are deprecated for one release
  and then removed (owner decision, see design.md Open Questions).
- **BREAKING:** results move at the 1e-10 level where the propagator
  tier or the φ₁ branch changes, and at the event-tolerance level
  (1e-10 s) where an event is now located more tightly.

## Impact

- Affected specs: `exact-propagator` (new), `transient-timestep`.
- Affected code: `dsed/{exact_lti,scheduler,event_predictor,
  event_projection,builder_adapter}.hpp`, `pwl/cache.hpp`
  (`compute_lti_state_space`: inductance matrix, initial-condition map,
  affine device regions), `python/dsed_bindings.cpp`,
  `python/pulsim/_dsed_dispatch.py`, `python/pulsim/__init__.py` (the
  dsed refusals), docs (`docs/dsed.md`, engine guide), stubs.
- Depends on:
  - `fix-dsed-auto-integrator-selection` (Phase 0): `auto` already
    prefers the exact stepper.
  - `add-engine-conformance-matrix` (Phase 0): the refusal type and the
    device-kind table used by the applicability rule.
  - `restore-reference-benchmark-harness` (Phase 0): the counter-based
    gate.
  - `refactor-unified-transient-stepper`, `add-unified-event-queue`,
    `add-unified-device-interface` (Phase 3): for the strategy inside
    the stepper, the fixed-step update, source breakpoints as events and
    affine device regions.
  - `add-probes-and-recorders` (Phase 3): the contiguous result buffer.
- Ships on today's structure, inside `dsed/` (tasks 1–7): the kernel
  tiers, φ accuracy, modal stepping and allocation removal, relative
  event tolerance, zero-band mode selection, initial conditions, coupled
  inductors.
- Must wait for Phase 3 (tasks 8–11): piecewise-constant sources as
  queue events, the fixed-step Φ/Γ form, the hybrid with Newton devices,
  retirement of the RK45 / BDF2 schedulers.
- Speed figures: "about 10× for dsed" and "5–10× for Φ/Γ" are the
  audit's estimates from a time split, not measurements; tasks.md gates
  them on counters. No comparison with PLECS or PSIM was measured; the
  statement that ideal-switch simulators step segments in closed form is
  general knowledge.

## Re-centred scope, 2026-11-13

> Appended; the sections above are unchanged and slice 1 (task groups
> 1–6, being implemented on `feat/core-p4-exact-propagator`) is not
> re-planned. Where this section and an earlier one disagree, this
> section supersedes it, and the superseded statement is named.

Exact propagation is now one of the two pillars of the core roadmap
(`openspec/CORE_ROADMAP.md`, "The centre of the roadmap"). The
strategy stops being an engine a user selects and becomes the route
`engine='auto'` takes whenever it can, with the implicit integrators as
the strategies for what it cannot take. Measured reasons, beyond the
ones above:

- With ideal switches and the complementarity mode selection
  (`add-branch-form-ideal-switch`, `add-complementarity-mode-selection`)
  the exact route reproduces 40-digit and mpmath references at the
  1e-12 level on every family prototyped: DCM buck `v_C` +4.5e-12 V
  and turn-off +2.0e-12·T after 4000 periods; boost CCM orbit
  −2.5e-12 V, −6.2e-13 A after 20 000 periods; bridge event times
  within 1.1e-14 s (single-phase) and 8.1e-15 s (three-phase)
  (`../add-complementarity-mode-selection/probes/lcp_cases.out`).
- The implicit routes cannot reach that at any affordable step: the
  ideal DCM buck on the fixed grid with the best restart rule is at
  4.3e-8 V with 400 steps per period
  (`../add-branch-form-ideal-switch/probes/sdirk_restart_ideal.out`).
- Large circuits need not leave the exact route
  (`probes/expm_action_ladder.out`, RLC ladder, `h = 1 µs`): at
  `‖A·h‖₁ = 2` the action of the exponential costs 20 matrix-vector
  products per step and agrees with the dense propagator to 3.2e-15;
  its flop count per step is 0.25 of the dense step at n = 200 and
  0.02 at n = 2000, and it needs no O(n³) setup (dense setup 0.9 ms
  at n = 200, 355 ms at n = 2000, scipy, single thread). With a 1 ns
  snubber (`‖A·h‖₁ = 1e3`) the action costs 2492 products per step,
  5–25 times the dense step, and the dense setup is repaid after
  203–6027 steps: that is the case for dense or for the hand-over.
- Sources held over a step are first order: a 100 V, 50 Hz sine into
  an RL at `h = T/20` is 17 A off a 95.4 A peak when held, 1.6e-12 A
  through the augmented oscillator states (same probe).

### What changes (appended)

- **Default route.** `engine='auto'` selects exact propagation for a
  segment whenever every active device is linear or PWL (ideal or
  resistor-pair switches, PWL diodes, devices on an affine region) and
  every source is exact-capable. Supersedes "selected by `auto` when
  … `n_state ≤ 64`" of the Applicability rules: the size limit is
  replaced by the scaling rule below.
- **Hybrid rule, precise.** Segments are classified at events only; an
  implicit segment lasts to the next event; the hand-over in either
  direction is an event with the restart rule (three consecutive
  Alexander SDIRK(2,2) steps) on entry to an implicit segment.
- **Scaling rule.** Per mode: dense modal (tier 1) or dense Padé/Schur
  (tier 2) while the dense setup is repaid inside the mode's expected
  steps and fits the cache budget; the action of the exponential
  (truncated Taylor with scaling, Al-Mohy–Higham; Krylov variants as an
  implementation choice) when `‖A·h‖₁` is moderate and `n` large;
  hand-over to the implicit strategy otherwise. Every choice and its
  inputs are counters in the result. Supersedes the Non-Goal "Krylov
  exponentials for large sparse systems".
- **Sources exact.** Constant, polynomial-in-time (ramps, PWL
  segments), exponential and sinusoidal sources through an augmented
  state (or the equivalent φ-functions); breakpoints are time events.
  An arbitrary Python source function still makes its segments
  implicit, now as a recorded segment-level decision.
- **Output grid and controller ticks are events.** A recorded sample
  is an evaluation of the exact flow at the sample time, not a step; a
  controller tick is an event that updates its outputs (zero-order
  hold) and may change gates.
- **Ideal switches are the native switch model** of the exact route:
  per-mode reduced state and explicit projection from
  `add-branch-form-ideal-switch` (its task group 8), mode at every
  event from `add-complementarity-mode-selection`. This supersedes the
  slice-1 zero-band rule (D4) as the decision rule once that change is
  the default; the zero-band rule stays as the slice-1 behaviour until
  then.
- **Engine names.** `engine='exact'` names the strategy explicitly;
  `'dsed'` stays as its alias for the legacy period. `'pwl'` and
  `'trbdf2'` select the implicit strategies (`refactor-unified-transient-stepper`).

### Impact (appended)

- Additional affected code: `dsed/exact_lti.hpp` (action kernel),
  a route selector beside `python/pulsim/_dsed_dispatch.py`, the
  recorders (samples as events), the control-tick scheduler.
- Additional dependencies: `add-complementarity-mode-selection`,
  `add-branch-form-ideal-switch` (task group 8), `add-unified-event-queue`
  (sample and tick events), `fix-control-tick-semantics` (done on
  `feat/core-integration`).
- The default flip of the route is one of the number-changing defaults
  that flip together in one release, re-baselined through the
  reference harness.
