## Why

The fixed engine (`engine='pwl'`, `solver/run_transient.hpp`) is a
trapezoidal companion with no rule for discontinuities. The companion
keeps each inductor's `v_prev` and each capacitor's `i_prev` from the
topology that existed before the step (`pwl/history_state.hpp:360-381`,
committed at `run_transient.hpp:1516`), and the step ending at `t` uses
`switch_fn(t)` (`run_transient.hpp:956`). Three measured consequences
(core review 2026-11, `raw/numerics.md` item 1; probes in
`/Users/lgili/pulsim_wt/core_review_2026-11/numerics/`):

- **Every gate edge is half a step off, so the engine is first order.**
  `p1_edge.py`, RL plus switch, re-run on this branch: the timing error
  is +0.500 dt or −0.500 dt for an edge on the grid (by the closure's
  convention) and ∓0.200 dt at +0.3 / +0.7 dt, order 1.00. `p2_buck.py`,
  buck CCM: 1.18e-2, 5.9e-3, 2.9e-3, 1.5e-3, 7.4e-4 A at T/50 … T/800.
- **Ringing at the Nyquist frequency that does not shrink with dt.**
  `p3_dcm.py`: in the idle interval of a DCM buck `v(sw)` alternates
  0.50 V / 26.15 V where the true value is 13.31 V, at T/100 and at
  T/400. `p8_caploop.py`: 1 nF across a 1 mΩ switch, true current 10 A,
  reads −9.8 / +29.8 A at dt = 1e-8 and −79 / +99 A at dt = 1e-9; `i²R`
  reports 0.197 W against 0.049 W.
- **The run is first order from t = 0** (`seed_from_dc_op`,
  `history_state.hpp:396`; pinned by four strict xfails in
  `python/tests/test_first_step_order.py`), and a source jump on the
  grid is 0.475 V off on a 10 V step at dt = τ/10 (`p13.py`).

A numpy re-implementation of the step (`p18_cda_proto.py`) reproduces
these numbers and shows the remedy: with the rule below the RL error
goes from −6.07e-3 A to −5.05e-6 A at dt = 2 µs (1200×, order 2.00) and
the buck CCM error from 1.28e-2 … 7.97e-4 A to 2.20e-5 … 8.60e-8 A
(order 2.00). These are prototype figures, not engine measurements.

## What Changes

- **Restart rule (critical damping adjustment, Marti & Lin 1989).** A
  step that follows a discontinuity is taken as two backward-Euler
  half-steps, `t → t + h/2 → t + h`, instead of one trapezoidal step.
  Backward Euler over `h/2` has companion conductance `C/(h/2) = 2C/h`
  and `L/(h/2) = 2L/h`, the trapezoidal values at `h`, so the matrix,
  its sparsity and every cached factor are the nominal ones; only the
  right-hand side changes (the stored derivative is left out). The
  trapezoidal rule is unchanged inside smooth segments.
- **Triggers:** a mask change (gate or diode), a source jump, and the
  first step of a run that does not start from a consistent state. A
  `resume_from` run and a `start_from_dc_op` run whose first mask is the
  DC mask do not restart.
- **Event iteration per half-step.** Each half-step runs its own diode
  event iteration; after the last mask change two half-steps are taken
  under the settled mask, so a flip settled in the second half-step
  makes the next step a restart step as well.
- **`TrBdf2Stage::BackwardEuler`** beside `Trapezoidal` and
  `Bdf2Stage2`, taken by the five stateful devices (nonlinear capacitor,
  saturable inductor, Lauritzen diode, IGBT tail, MNA-native PMSM) and
  by `HistoryState`.
- **BREAKING — gate sampling.** With the rule on, the step
  `(t_{k−1}, t_k]` is stamped with `switch_fn(t_{k−1})` (the
  right-continuous state at the step start), not `switch_fn(t_k)`.
  Without this the rule doubles the timing error of the natural PWM
  closure (prototype: +1.21e-2 A against +6.06e-3 A today).
- **BREAKING — source jumps.** A step that ends on a jump reads the
  source's left limit; the jump applies from the next step, which
  restarts.
- **BREAKING — numerical results.** Every fixed-step result with an
  event, a source jump or a non-DC start changes. The rule ships behind
  `SimulationOptions::discontinuity_restart` (Python
  `simulate(discontinuity_restart=...)`), off by default first, then on.
- `SimulationResult::n_restart_steps` records how many steps restarted.

## Impact

- Affected specs: `transient-timestep`.
- Affected code: `solver/run_transient.hpp` (step loop, event
  iteration, dt-retry, commit, first step), `solver/options.hpp`,
  `solver/result.hpp`, `pwl/history_state.hpp`, `pwl/trbdf2_stage.hpp`,
  `pwl/{nonlinear_capacitor,saturable_inductor,lauritzen_diode,igbt_tail,pmsm_mna}_history.hpp`,
  `pwl/nonlinear_refresh_*.hpp` of those five, `sources/` (jump table,
  left-limit evaluation), `python/bindings.cpp`,
  `python/pulsim/__init__.py`, stubs, `AI_GUIDE.md`,
  `docs/known-limitations.md`, `CHANGELOG.md`.
- Depends on: `restore-reference-benchmark-harness` (the golden hashes
  and the analytic reference set are how the re-baselining is
  reviewed). Touches the same lines as
  `update-fixed-step-loop-bookkeeping` (one `switch_fn` call per step)
  and overlaps `fix-control-tick-semantics` on what the step observer
  is told; see design.md.
- Needed by: `add-fixed-step-event-splitting`,
  `add-periodic-steady-state-shooting`.
- Golden baselines: with the flag off every fixed-engine hash is
  unchanged. With it on, every baseline that contains an event moves;
  DC-started runs before their first event are bit-identical. Users are
  told through a CHANGELOG **BREAKING** entry with the before/after
  error table, the release notes, and guide traps 9, 10 and §6.3.
- `engine='trbdf2'` and `engine='dsed'` are not touched.

## Superseded rule, 2026-11-13

> Appended note; the sections above are kept as the record of the first
> form of this change.

The rule is now three consecutive Alexander SDIRK(2,2) steps
(`γ = 1 − 1/√2`, one cached factor per mode at `dt_eq = 2γh`) at every
mask change, source breakpoint and t = 0, not the two backward-Euler
half-steps described above (maintainer decision of 2026-11-13,
`openspec/CORE_ROADMAP.md` decision 2). The backward-Euler pair drains
resonant modes; on the ideal DCM buck three SDIRK(2,2) steps are about
400 times more accurate at the same step
(`../add-branch-form-ideal-switch/probes/sdirk_restart_ideal.out` (3):
4.345e-8 V against 1.707e-5 V at T/400). The rule is opt-in until the
single number-changing release; it is being re-implemented on
`feat/core-phase0`, where this change's documents are updated with it.
