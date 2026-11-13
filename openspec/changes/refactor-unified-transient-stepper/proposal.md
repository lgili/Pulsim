## Why

The two MNA transient engines are the same numerical kernel inside two
hand-written loops, and every place the loops differ has been a defect.
Counted on this branch (`core/…` = `core/include/pulsim/…`,
RT = `core/solver/run_transient.hpp`, TB = `core/solver/trbdf2_transient.hpp`,
PY = `python/pulsim/__init__.py`):

- **Four transient drivers and one legacy overload:**
  `run_transient` (one function, RT:284-1793, 1510 lines),
  `run_transient_trbdf2` (one function, TB:264-1849, 1586 lines), the
  DSED schedulers (`core/dsed/*.hpp`, 3846 lines, three schedulers),
  `run_transient_bdf1` (`core/solver/bdf1.hpp`, 434 lines) and the V0
  overload with no history (RT:179-265).
- **Ten integrator implementations** (audit `raw/arch.md` §b):
  trapezoidal, TR-BDF2, backward Euler, DSED BDF2, exact LTI,
  Dormand–Prince three times (`core/dsed/rk45_dormand_prince.hpp`,
  `core/integrators/dormand_prince5.hpp`, `python/pulsim/integrators.py`),
  Radau twice; plus `python/pulsim/adaptive.py` (273 lines), which wraps
  `simulate()`. `core/integrators/` (528 lines) is bound and used by no
  module except its own tests and a re-export.
- **What the two MNA drivers already share:** both TR-BDF2 stages go
  through `cache.solve_at(mask, γh)`, the trapezoidal kernel
  (TB:25-37); the fixed engine's dt-retry already runs a variable step
  on the same history (RT:1075-1160); device stamps already take a
  `TrBdf2Stage`. They differ in the loop only.
- **What they duplicate:** source accumulation (RT:777-793, TB:480-500,
  `core/dsed/builder_adapter.hpp`), the Newton refresh chain (RT:471-600,
  TB:588-697), initial-condition and resume handling (RT:345-467,
  TB:420-585), the diode settle and commit sequences
  (`update_from_state` appears 24 times in RT and 17 in TB), and four
  result-push sites in RT against three in TB.
- **Twelve features exist on one loop only.** `_trbdf2_blockers`
  (PY:3177-3289) refuses on the variable-step engine: `on_cache`, C
  blocks, `live_stream`, `progress`, `start_from_dc_op`,
  `strict_event_iterations`, `max_dt_halvings`,
  `enable_substep_state_correction`, `store_every`, `mmc_arms`,
  `inductor_freeze_di_max`, `inductor_abs_clamp`; plus an observer
  without a cadence and an undriven controlled switch.
- **`bdf1` supports almost nothing** and has its own copy of the
  nonlinear guard (`core/solver/bdf1.hpp:297`).

Defects that are divergences between the loops (audit `raw/arch.md` §0
and the owner's notes):

| Symptom | Divergence | Stop-gap |
|---|---|---|
| D3: `bdf1` deletes a nonlinear capacitor (6.3002 V with and without Coss) | third copy of the device guard | `add-engine-conformance-matrix` |
| D5, D6, D7: two loops crash, multi-rate wrong, `controller_period` ignored | tick handling per loop | `fix-control-tick-semantics` |
| SIGSEGV in the TR-BDF2 diode settle beside a saturable inductor | probe solves did not set the device step that the stage solves set | fixed in `trap_solve` |
| Builder ICs honoured on one engine only until 2026-09-04 | IC seeding per loop | moved in Python |
| Resume exact on the fixed engine, "to tolerance" on TR-BDF2 | the controller state is not in the snapshot | — |
| Saturable inductor could not take a variable step on the fixed engine's retry ladder (`dt_retry_can_help`, RT:763) | its history ignored the step it was given | refusal |
| `store_every`, `start_from_dc_op`, progress refused on the default engine when no `dt` is given | features implemented inside one loop | refusal |

Probes: `core_review_2026-11/arch/{matrix,probe3,probe5,probe7}.py`.

## What Changes

- **One stepper.** `Stepper<Integrator, Controller>` over a
  non-template `StepperCore` that owns the cache, the device state, the
  mode set, the event queue and the workspace.
  - Integrators (stage tables, an LTE estimate where they have one):
    `BackwardEuler`, `Trapezoidal`, `TrBdf2`.
  - Controllers: `Fixed{dt}` (integer step counter, optional retry
    ladder) and `Lte{rtol, atol}` (accept / reject, step memory).
- **One solve point.** Every solve of the run (stage, probe, settle,
  landing, retry sub-step) is `solve_stage(mask, StageSpec, t)`, which
  builds the step context, the history right-hand side and the sources
  for exactly that solve.
- **Policies instead of forks.** Event handling
  (`IterateAtStepEnd`, `Localize`), restart after a discontinuity
  (`None`, `Cda`, `DerivativeRefresh`: the rules of
  `add-discontinuity-restart-rule` and
  `update-trbdf2-post-event-restart`), retry on a failed step
  (`HalvingLadder`, `ShrinkAndRepeat`) are small policy types selected
  by the controller.
- **Features become plugins** of the one loop: recorder hooks, guard
  hooks, tick handlers, initial-condition sources. A feature is then
  either available on both controllers or refused for a structural
  reason that the refusal states.
- **Proof before switch.** `Stepper<TrBdf2, Lte>` must reproduce the
  current variable-step engine step for step, and
  `Stepper<Trapezoidal, Fixed>` must reproduce the fixed-engine goldens
  bit for bit, before `engine='trbdf2'` and `engine='pwl'` are pointed
  at them. The old drivers stay reachable as `engine='trbdf2-legacy'`
  and `engine='pwl-legacy'` for one release.
- **Retired:** `core/solver/bdf1.hpp` (replaced by
  `Stepper<BackwardEuler, Fixed>`), the V0 overload (RT:179-265),
  `core/integrators/`, `python/pulsim/integrators.py`,
  `python/pulsim/adaptive.py`.
- **BREAKING (Python API):** `pulsim.DormandPrince5`, `RadauIIA3`,
  `AdaptiveSolution`, `AdaptiveResult`, `run_transient_adaptive` and the
  bound `dopri5_solve` / `radau_iia3_solve` are removed after one
  deprecation release; `run_transient_bdf1` becomes a deprecated alias.
- **BREAKING (snapshot):** the snapshot gains an optional controller
  block (last accepted step, step memory), which makes a variable-step
  resume exact. Snapshots without it still resume to tolerance.
- **BREAKING (behaviour, improvement):** `bdf1` runs Newton devices and
  PWL diodes instead of refusing them; `store_every`,
  `start_from_dc_op`, `progress`, `live_stream`,
  `strict_event_iterations` and the inductor guards stop being refused
  on the variable-step engine as each is ported.
- **Not changed:** the DSED schedulers. The exact propagator becomes a
  strategy of this stepper in `add-exact-propagator-strategy`.

## Impact

- Affected specs: `transient-timestep`.
- Affected code: new `core/include/pulsim/solver/step/`;
  `solver/{run_transient,trbdf2_transient,bdf1,options,result}.hpp`;
  `pwl/{history_state,nonlinear_solve,diode_event_state}.hpp`;
  `integrators/`; `python/bindings.cpp`;
  `python/pulsim/{__init__,_dsed_dispatch,integrators,adaptive}.py`;
  stubs; `docs/solvers.md`, `docs/mental-model.md`,
  `docs/known-limitations.md`.
- Depends on: `add-unified-device-interface` (steps 1–3 at least: one
  history fold, kernel-built stamps, the flat device state),
  `add-unified-event-queue` (time events, `ModeSet`, the locator),
  `restore-reference-benchmark-harness` (golden hashes, the
  differential runner, the counter-based perf gate).
- Carries over, as policies, whatever has landed from Phase 1 and
  Phase 2: `update-fixed-step-loop-bookkeeping`,
  `update-trbdf2-factor-reuse`, `add-discontinuity-restart-rule`,
  `add-fixed-step-event-splitting`, `update-trbdf2-post-event-restart`,
  `update-trbdf2-error-control`, `add-newton-roundoff-acceptance`,
  `add-fixed-step-resolution-diagnostics`.
- Conflicts with the pending change `add-adaptive-runge-kutta-solvers`
  (it extends `core/integrators/`, which this change retires); see the
  design's Open Questions.
- Unblocks: `add-exact-propagator-strategy`,
  `add-high-order-stiff-integrator`, `add-parameter-sensitivities`,
  `Simulation::step()` of `add-kernel-simulation-object`.
- Performance constraints: TR-BDF2 keeps one factorisation per
  `(mask, γh)`; `Fixed` + `Trapezoidal` uses the primary cached segment
  and performs no LTE work and no rollback copy; zero heap allocations
  per step (`core/tests/perf/test_zero_alloc.cpp` ported to the
  stepper); thread-local workspaces; the counter gate (`n_solves`,
  factorisations, cache hits) is equal to the legacy drivers on every
  harness circuit, and ns/step on audit rows 1–3 and 5–9 is not above
  the legacy value. Compile time: the template layer is the stage
  sequencing only; measured, with a 10 % budget on the extension build.
- Results: bit-identical on the fixed engine and step-identical on the
  variable-step engine at the switch; nothing moves until the legacy
  drivers are deleted.

## Re-centred scope, 2026-11-13

> Appended; the sections above stand. Where this section disagrees
> with them it supersedes them, and says so.

The core roadmap now has two pillars, the ideal switch with
complementarity mode selection and exact propagation
(`openspec/CORE_ROADMAP.md`, "The centre of the roadmap"). Phase 3 is
designed around them: the one kernel is **event-driven first**. Its
loop is "advance exactly to the next event, then handle the event"; an
integrator is how a segment is advanced when it cannot be propagated
exactly, not the shape of the loop.

What changes in this change's plan:

- **The kernel loop**: pop the next event from the queue
  (`add-unified-event-queue`); advance the state to it with the
  segment's strategy; at the event, select the mode
  (`add-complementarity-mode-selection`), re-initialise
  (`add-branch-form-ideal-switch` D4), restart the strategy. The step
  loop of D4 above becomes the inner loop of an implicit segment.
- **Strategies**: the exact propagator (`add-exact-propagator-strategy`,
  the default for linear and PWL segments); the trapezoidal rule on a
  fixed grid with the restart rule of three consecutive Alexander
  SDIRK(2,2) steps after every discontinuity; TR-BDF2 with error
  control. The two implicit ones serve nonlinear segments and segments
  the propagator declines.
- **One mode-selection service and one re-initialisation service**,
  called at every event by every strategy. The per-strategy event
  policies of D4 (`IterateAtStepEnd`, `Localize`) keep their location
  part and lose their decision part: `modes.settle` is replaced by the
  selection service once it is the default.
- **Engine names survive as strategy selectors**: `engine='auto'` (the
  default: exact where eligible, TR-BDF2 elsewhere), `'exact'` (alias
  `'dsed'`), `'pwl'` (trapezoid on a fixed grid with the SDIRK restart),
  `'trbdf2'`; `'dsed'` and the hidden legacy names keep working for a
  legacy period with a deprecation warning.
- `RestartPolicy::Cda` (the backward-Euler pair) is superseded by
  `RestartPolicy::Sdirk22x3`; this supersedes the row "`Cda` —
  `add-discontinuity-restart-rule`" of D4.

The equality proofs of D7 (bit-identical fixed engine, step-identical
variable-step engine) are unchanged: they are proved with the legacy
decision and restart policies selected, before the defaults flip.
The defaults flip in the single number-changing release of the
roadmap.
