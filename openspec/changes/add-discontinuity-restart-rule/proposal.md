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

A numpy re-implementation of the step (`p18_cda_proto.py`) showed the
first remedy, a pair of backward-Euler half-steps (Marti & Lin 1989).
That pair was implemented (commit 41711488) and then REPLACED: it drains
resonant modes by (ωh)²/2 per event. A Q = 50 tank driven by a PWM
source lost 11.9 % / 3.1 % of its peak at T/50 / T/100 (0.86 % / 0.05 %
without it), and an ideal LC beside an electrically isolated PWM source
kept 0.019 of its energy after 100 periods (0.996 without it).

## What Changes

- **Restart rule: three Alexander SDIRK(2,2) steps.** The step that
  contains a discontinuity and the next two are taken as SDIRK(2,2)
  steps, `γ = 1 − 1/√2` (L-stable, second order). Each is two
  backward-Euler-like stages of `γh`:
  - stage 1 ends at `t0 + γh`;
  - stage 2 ends at `t0 + h` and starts from the pseudo previous state
    `x0* = x0 + ((1 − γ)/γ)(x1 − x0)`.
  The stage conductance `C/(γh)` is the trapezoidal companion at
  `2γh`. Both stages share one factor per mask, kept by a second lazy
  cache at that fixed step. Stateful devices extrapolate their
  charge/flux the same way; the flux row stays in conductance form. The
  trapezoidal rule is unchanged inside smooth segments.
- **Triggers:**
  - a mask change (gate or diode);
  - a source jump;
  - a jump a `b_extra_fn` declares through `next_jump_after(t)`
    (undeclared callback jumps are not restarted, documented);
  - the first step of a run that does not start from a consistent state.
  A `resume_from` run and a `start_from_dc_op` run do not restart.
  Each trigger (re-)arms a window of three steps; the count survives
  snapshot and pickle (`SolverSnapshot::restart_steps_left`).
- **Event iteration per stage.** Each stage runs its own diode event
  iteration. A flip in either stage extends the window to the next two
  steps. Decided once per step instead, the turn-off current has no path
  in stage 1 and a buck collapses (measured: DCM output 0.83 V instead
  of 13.3 V).
- **`TrBdf2Stage::BackwardEuler`** beside `Trapezoidal` and
  `Bdf2Stage2`. It is taken by `HistoryState` and by the five stateful
  devices: nonlinear capacitor, saturable inductor, Lauritzen diode,
  IGBT tail and MNA-native PMSM. Each also gets `extrapolate_previous`.
- **BREAKING — gate sampling.** With the rule on, the step
  `(t_{k−1}, t_k]` is stamped with `switch_fn(t_{k−1})` (the
  right-continuous state at the step start), on the dynamic and the
  static path alike.
- **BREAKING — source jumps.** A step that ends on a jump reads the
  source's left limit; the jump applies from the next step, which
  restarts.
- **BREAKING — numerical results.** Every fixed-step result with an
  event, a source jump or a non-DC start changes. The rule ships behind
  `SimulationOptions::discontinuity_restart` (Python
  `simulate(discontinuity_restart=...)`), off by default.
- **Refusal.** The rule together with `enable_substep_state_correction`
  is refused by name.
- `SimulationResult::n_restart_steps` and `n_restart_factorizations`.

## Impact

- Affected specs: `transient-timestep`.
- Affected code: `solver/run_transient.hpp` (step loop, event
  iteration, dt-retry, commit, first step), `solver/options.hpp`,
  `solver/result.hpp`, `pwl/history_state.hpp`, `pwl/trbdf2_stage.hpp`, `python/pulsim/__init__.py` (`next_jump_after` composition),
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
