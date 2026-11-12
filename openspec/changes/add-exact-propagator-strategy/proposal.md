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
