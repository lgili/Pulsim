# Core roadmap — index of OpenSpec changes (re-centred 2026-11-13)

Source: the core review of 2026-11-12 (accuracy, speed, models,
architecture) on branch `feat/diode-forward-drop`. Raw audit reports and
the probe scripts that produced every number quoted in the proposals:
`/Users/lgili/pulsim_wt/core_review_2026-11/` (`raw/*.md`, `numerics/`,
`perf/`, `models/`, `arch/`, `CORE_REVIEW_2026-11.md`). Re-centred on
2026-11-13 by the maintainer under the owner's delegation; the
prototypes behind the re-centring are committed in the probes folders
of `add-complementarity-mode-selection`, `add-branch-form-ideal-switch`
and `add-exact-propagator-strategy`.

Each row is one change under `openspec/changes/<id>/`. Phases are
ordered; inside a phase, changes are independent unless "Needs" says
otherwise. Item codes (D, P, V, M, A) refer to the review.

Legend: effort S / M / L. **B** = changes numerical results or public
behaviour (deliberate release, migration note). Status, as of
2026-11-13: **integrated** = done and integrated on
`feat/core-integration`; **side branch** = done on a side branch,
awaiting integration; **in progress**; **not started**. A bracket
narrows the status to the part it names.

## The centre of the roadmap

The roadmap has two pillars. Everything in Phase 3 is designed around
them and Phase 4 is ordered by what they make possible.

**Pillar A — the ideal switch, with complementarity mode selection**
(`add-complementarity-mode-selection`, `add-branch-form-ideal-switch`).
A switch is today two conductances (`1e3 / 1e-9 S`), and a diode is
decided by re-solving and re-applying a value law, with round caps,
cycle detectors, chatter guards and an anti-chatter band `V_th > V_f`
as the user's remedy. Measured on today's kernel
(`add-complementarity-mode-selection/probes/today_dsed.out`): with
`V_th = 0` the event-driven engine fails a buck at rest ("did not
settle after 64 rounds"), chatters in DCM at 0.996 ms, and fails the
boost and both bridge rectifiers at t = 0; three of six `L·(1 ± δ)`
runs abort; the periodic-steady-state solver cannot settle a boost
with `V_th ≤ 1e-3`. The resistor pair puts picosecond modes into every
circuit (idle DCM eigenvalue −5.0e12 s⁻¹), costs the exact propagator
its conditioning (κ = 2.3e9 → 7.9e-9 relative error), and with
`g_off = 0` the kernel rings at ±397 V (`pwl`) or 7.0e10 V (`trbdf2`)
without a warning. With ideal switches, one LCP per event (Lemke,
lexicographic rule; a passive network gives a positive-semidefinite,
hence copositive-plus, port matrix, so it terminates) and a
conservative re-initialisation, the prototype runs the five failing
families with no band: at most 6 pivots per event, no wrong mode, the
DCM buck within 4.5e-12 V of its 40-digit orbit after 4000 periods,
bridge event times within 1.1e-14 s of the closed form.

**Pillar B — exact propagation as the default**
(`add-exact-propagator-strategy`). Between events a linear or PWL
circuit has a closed-form solution. Measured: `dsed` is about 1e-10
against a 50-digit DCM reference and 6e-10 on CCM at 10 samples per
period; with ideal switches the exact route reaches 1e-12 on every
prototyped family, while the best implicit route (trapezoid with the
SDIRK restart) is at 4.3e-8 V with 400 steps per period
(`add-branch-form-ideal-switch/probes/sdirk_restart_ideal.out`). Large
circuits stay exact through the action of the exponential (20
matrix-vector products per step on a 2000-state ladder, 0.02 of the
dense step) until stiffness makes a hand-over cheaper
(`add-exact-propagator-strategy/probes/expm_action_ladder.out`). Its
overhead today is plumbing (97 %: `pwl` 91 ms, `trbdf2` 69 ms, `dsed`
14.5 ms for 2000 periods), not mathematics.

**Why these two**: they remove the two sources of wrong answers that
no tolerance fixes — a diode state chosen by iteration, and an event
placed on a grid — and they make every KPI smooth in the design
parameters, which is what the optimisation-first Phase 4 needs (a
central finite difference through the fixed engine is off by relative
errors from −12.9 to +9.1, `add-parameter-sensitivities`).

The implicit integrators remain, as strategies: the trapezoidal rule on
a fixed grid with the restart rule (three consecutive Alexander
SDIRK(2,2) steps after every discontinuity), and TR-BDF2 with error
control, for nonlinear segments and for circuits the propagator
declines.

## Order and dependencies

Phase 0 → Phase 1 (speed, bit-identical) → Phase 2 (accuracy) →
CORE A (ideal switch + complementarity) → CORE B (exact propagation
default) → Phase 3 (one kernel, designed around the core) → Phase 4
(optimisation-first).

```
Phase 0  conformance matrix ─┬─ harness ─────────────────────────────────────┐
         (refusals, table)   │                                               │
Phase 1  linear-time assembly, factor reuse, Newton solver (bit-identical)  │
Phase 2  restart rule (SDIRK(2,2)×3) ─ event splitting ─ TR-BDF2 restart ───┤
                                                                             │
CORE A   complementarity mode selection ──┐ (shadow mode needs only Phase 0) │
         branch-form ideal switch ────────┤ (needs the restart rule for      │
                                          │  g_off = 0 on implicit routes)   │
CORE B   exact propagator: slice 1 ───────┴─ default route, scaling rule,    │
                                             ideal switches native           │
Phase 3  event queue ─ device interface ─ stepper (event-driven first) ──────┘
         simulation object ─ probes/recorders
Phase 4  in-kernel event losses ─ IR parameter re-binding ─ sensitivities ─
         PSS completion ─ design-optimisation workflow ─ the rest
```

CORE A and CORE B ship slice by slice on today's structure (the
selection service and the reduced state live beside the existing
engines); Phase 3 then folds them into one kernel. All number-changing
defaults flip together in one release (decision 3 below).

## Phase 0 — stop the silent wrong answers

| Change id | Scope | Items | Capabilities | Effort | Needs | Status |
|---|---|---|---|---|---|---|
| `add-engine-conformance-matrix` | Per-device-kind capability table replacing the hand-kept `has_nonlinear_devices()` list; one `UnsupportedError` naming device + engine; conformance test (device × engine, feature × engine) with the invariant "no silent cell". Fixes: dsed and bdf1 drop `NonlinearCapacitor`; pwl with refresh off treats Newton devices as open; `NativeMultiMaskPwm` loses exact edges on trbdf2 | D3, D4, D10 | `engine-conformance` (new), `transient-timestep` | M | — | integrated |
| `fix-nonlinear-diode-passivity` | Retires `add_nonlinear_diode` (decision 5); SPICE importer maps Ron/Roff/Vfwd diodes to the PWL diode with `V_f` | D1 | `device-models`, `python-bindings` | S | — | integrated [as a retirement] |
| `fix-mna-sweep-correctness` | Kernel AC sweep: mutual inductance in `E`, caller-supplied switch mask, refusal when a Newton kind is present, no broad `except` | D2 | `ac-analysis` | S | — | integrated |
| `fix-control-tick-semantics` | Two closed loops without `dt`; per-observer rates on trbdf2 (multi-rate); `controller_period` honoured on pwl | D5, D6, D7 | `transient-timestep`, `python-bindings` | M | — | integrated |
| `fix-dsed-auto-integrator-selection` | dsed `integrator='auto'` uses the exact stepper whenever available; BDF2 bootstrap is backward Euler | D8 | `transient-timestep` | S | — | integrated |
| `update-composite-diode-forward-drop` | `add_mosfet` / `add_igbt` / Level-1 body and freewheel diodes take their `V_F` as a conduction drop (**B**) | D9 | `device-models`, `python-bindings` | S | — | side branch |
| `restore-reference-benchmark-harness` | Harness-only reader for the 126 v1 benchmark circuits; differential runner circuit × engine; analytic reference set; golden checkpoints of the fixed engine (tolerances, exact for discrete data); counter-based perf gate; nightly ngspice job | D11, A1 | `benchmark-suite` | M | — | side branch |

## Phase 1 — speed with bit-identical results

| Change id | Scope | Items | Capabilities | Effort | Needs | Status |
|---|---|---|---|---|---|---|
| `update-fixed-step-loop-bookkeeping` | Hoist `masks_seen`; one `switch_fn` call per step; memoise the last mask/segment; C++ constant schedule when no `switch_fn` is given; sparse source passes; zero-copy `res.times` | V3, V4, V12 | `transient-timestep`, `python-bindings` | S | harness | in progress |
| `update-newton-persistent-solver` | One in-house LU per run for the Newton path, analysed once, values written in place; hoisted work vectors | V1 (stage A) | `linear-solver`, `transient-timestep` | M | harness | side branch [stage A1]; in progress [stage A2/B] |
| `update-trbdf2-factor-reuse` | Byte-budgeted event cache; allocation-free refactor; step size on a geometric grid with a dt-keyed factor cache (**B**: step sequence) | V2 | `transient-timestep`, `linear-solver` | S + M | harness | side branch [parts 1–2] |
| `update-linear-time-assembly` | Triplet/reserved assembly, per-circuit stamp-slot map; one symbolic analysis shared by all masks; single-bit flips through `partial_refactor`; fused triangular solve | V5, V6, V11 | `linear-solver`, `pwl-rank1-update` | M | harness | side branch [tier 1]; in progress [tier 2] |
| `update-sweep-threading-and-import-time` | Thread-pool sweeps when callbacks are C++ schedules; lazy import of the scope GUI stack | V9 | `parameter-sweep`, `python-bindings` | S | — | integrated |

## Phase 2 — accuracy of the method

| Change id | Scope | Items | Capabilities | Effort | Needs | Status |
|---|---|---|---|---|---|---|
| `add-discontinuity-restart-rule` | Fixed engine: the restart rule — three consecutive Alexander SDIRK(2,2) steps at every mask change, source breakpoint and t = 0, one factor per mode at `dt_eq = 2γh` (supersedes the backward-Euler pair, decision 2); gate mask sampled at the step start (**B**) | P1 | `transient-timestep` | S–M | harness | side branch [opt-in; being reworked to SDIRK(2,2)×3] |
| `add-fixed-step-event-splitting` | Fixed engine: PSIM-style step split at diode zero crossings and gate edges, then the restart rule (**B**) | P2 | `transient-timestep` | M | restart rule | in progress |
| `update-trbdf2-post-event-restart` | TR-BDF2: derivative refresh (or SDIRK(2,2) first step) after an event; step memory stores the controller proposal | P3 | `transient-timestep` | M | harness | in progress |
| `update-trbdf2-error-control` | Estimator scale factor; Hosea–Shampine stiff filter; step bounds independent of `t_end`; source breakpoints as events (**B**) | P6, P7 | `transient-timestep` | S | — | in progress |
| `add-newton-roundoff-acceptance` | Transient-only acceptance on stagnated dx plus row-wise backward error; step floor at the roundoff bound; DC cascade unchanged | P4 | `transient-timestep` | S–M | — | in progress |
| `add-fixed-step-resolution-diagnostics` | Per-step normalised LTE recorded in the result; Nyquist-ringing detector; warning when under-resolved | P5 | `transient-timestep`, `python-bindings` | S–M | — | in progress |

## CORE A — the ideal switch with complementarity mode selection

| Change id | Scope | Items | Capabilities | Effort | Needs | Status |
|---|---|---|---|---|---|---|
| `add-complementarity-mode-selection` | One LCP per event on the diode ports of the event-step companion network, solved by Lemke (lexicographic), shared by every engine and the DC operating point; enumeration fallback and typed refusal for non-passive ports; ties by widened event step; floating islands as a gauge; shadow mode, then opt-in, then default (**B** at the flip) | — | `mode-selection` (new), `device-models` | M–L | conformance; harness (shadow comparison) | not started |
| `add-branch-form-ideal-switch` | Branch-form switch and diode with legal `R_on = 0`, `g_off = 0` on a mask-invariant pattern; per-mode floating-node and ideal-loop refusals; one re-initialisation service (explicit projection on the exact route, SDIRK(2,2)×3 on the implicit routes) with impulse records; switching energies from tables at the event; the default switch model at the flip (**B**) | P11 | `device-models`, `linear-solver` | L | conformance; complementarity; restart rule and TR-BDF2 post-event restart (for `g_off = 0` on implicit routes) | not started |

## CORE B — exact propagation as the default

| Change id | Scope | Items | Capabilities | Effort | Needs | Status |
|---|---|---|---|---|---|---|
| `add-exact-propagator-strategy` | Slice 1: robustness gaps (V_th = 0, ICs, relative event tolerance, conditioning), overhead removal. Re-centred scope: the default route of `engine='auto'`; hybrid segment rule; dense / expm-action / hand-over scaling rule with counters; exact sources; samples and ticks as events; ideal switches native; per-mask Φ/Γ fixed-step form (**B** at the flip) | P9, V7, V8 | `exact-propagator` (new), `transient-timestep` | L | slice 1: none; default route: CORE A, event queue, stepper | in progress [slice 1] |

## Phase 3 — one kernel, designed around the core

The kernel is event-driven first: advance exactly to the next event,
then select the mode (CORE A) and re-initialise (CORE A), with the
implicit integrators as strategies for the segments the propagator
declines (re-centred scope sections of the changes below).

| Change id | Scope | Items | Capabilities | Effort | Needs | Status |
|---|---|---|---|---|---|---|
| `add-unified-event-queue` | One queue for time events (gate edges, control ticks, source breakpoints, output samples) and state events (one Illinois); declarative modulators with analytic `next_edge`; one mode selection per event instant | A4, V10 | `event-scheduling` (new) | M | conformance | not started |
| `add-unified-device-interface` | Device = `f(x,t) + d/dt q(x)` with AD Jacobians, optional state / events / outputs / field table; flat device-state vector; generic `res.i` and loss outputs; AC and DC derived from f and q; the PWL category carries the complementarity form | M1, M4, A2 | `device-interface` (new), `device-models`, `ac-analysis` | M–L | conformance, harness | not started |
| `add-kernel-simulation-object` | `RunPlan` resolved once above the dispatch, then a C++ `Simulation{Circuit, RunOptions}`; typed error model; one options struct | A3, A8 | `simulation-kernel` (new), `python-bindings` | M–L | conformance | not started |
| `refactor-unified-transient-stepper` | Event-driven outer loop; integrators (trapezoid + SDIRK×3 on a fixed grid, TR-BDF2 with error control, exact) as segment strategies; one mode-selection and one re-initialisation service; engine names as strategy selectors with a legacy period; fixed-h trap mode proven equal to the pwl goldens; retire `bdf1`, the V0 overload and `integrators/` | A5, A11 | `transient-timestep` | L | device interface, event queue, harness, CORE A | not started |
| `add-probes-and-recorders` | Netlist-declared probes computed in-kernel; recorders (all / selected / decimated / `t_eval`); one result type that knows its grid | A7 | `simulation-results` (new) | M | device interface | not started |

## Phase 4 — optimisation first

In this order: in-kernel event losses, parameter re-binding in the IR,
sensitivities, PSS completion, the design-optimisation workflow; then
the rest.

| Change id | Scope | Items | Capabilities | Effort | Needs | Status |
|---|---|---|---|---|---|---|
| `add-in-kernel-event-losses` | Loss tables in C++; energy injected into the thermal node at each commutation (the event records of the ideal switch); closed-loop T_j on every engine | M3 | `loss-thermal` (new) | M | event queue; branch-form loss records | not started |
| `add-immutable-circuit-ir` | Immutable `Circuit` with named parameters and expressions, hierarchy kept, clone / `with_params` (re-binding without rebuild) / `to_yaml`, domain-tagged nodes | A9 | `circuit-ir` (new), `netlist-yaml` | M–L | simulation object | not started |
| `add-parameter-sensitivities` | Forward sensitivities through the cached LU with event jump correction; exact-propagator form; steady-state gradients | A10 | `parameter-sweep` | L | device interface, stepper | not started |
| `add-periodic-steady-state-shooting` | One Newton shooting solver on the true state with located events and saltation matrices; works in DCM; completion: every strategy, nonlinear devices, ideal-switch reset maps | P8 | `periodic-steady-state` (new) | M–L | slice 1: none; completion: event queue, device interface, CORE A | integrated [slice 1] |
| `add-design-optimization-workflow` | `evaluate(design) -> EvaluationRecord`: deterministic, thread-safe; typed refused / failed / infeasible records; T_j fixed point until in-kernel losses; batch on the thread sweep backend; worked buck Pareto example with an optional optimiser | — | `design-optimization` (new) | M | first slice: none; then in-kernel losses, IR re-binding, sensitivities | not started |
| `add-power-mosfet-datasheet-device` | `add_power_mosfet` (channel + Cgs, Cgd(v), Coss(v), R_g, Lauritzen body diode); VDMOS import with capacitances; datasheet fitters; real validated parts; ngspice cross-validation | M2, M10, M11 | `device-models`, `python-bindings` | M | conformance | side branch [machinery, no real parts] |
| `add-pwl-igbt-and-thyristor` | PWL IGBT with V_ce0 and reverse blocking; kernel thyristor / GTO latch (a gated complementarity port) | M5 | `device-models` | S–M | conformance; CORE A | not started |
| `add-mna-native-machines` | DC machine as an ideal-transformer composite; induction machine with rotor-flux states; BLDC via `PmsmMna` with a flux table | M6 | `motor-models` | S + M | device interface | integrated [DC machine only] |
| `add-controlled-source-set` | VCCS, CCVS, CCCS, gyrator; sine / pulse / table current sources | M7 | `device-models`, `netlist-yaml` | S | — | integrated |
| `add-permeance-magnetic-domain` | Permeance–capacitor magnetic networks on the gyrator; nonlinear permeance; eddy and excess loss | M8 | `magnetic-models` | M–L | controlled sources, power MOSFET (charge-law variant), device interface | not started |
| `add-external-device-abi` | User devices through a C ABI (f, q, Jacobians), then sympy-to-C codegen, then an OSDI host | M9 | `device-interface` | M, then L | device interface | not started |
| `add-compiled-control-graph` | Control graph with rates, sorted schedule, loop detection, discrete blocks on tick events, C export | A6 | `control-graph` (new) | L | event queue | not started |
| `add-high-order-stiff-integrator` | Single-factor ESDIRK4(3) as an integrator strategy for resonant dynamics | P10 | `transient-timestep` | L | stepper | not started |
| `add-realistic-passives` | ESR / ESL parameters, C(V) table, R_ac(f) ladder fit, transmission-line model | M12 | `device-models` | S–M | — | integrated [without C(V) and Bergeron] |

## Rules that apply to every change

- No implementation starts before its proposal is approved: by the
  owner, or by the maintainer under the owner's delegation of
  2026-11-13.
- Every proposal quotes the measurement that motivates it and names the
  probe script that reproduces it. Speed-up figures taken from a time
  split are labelled as estimates and each `tasks.md` carries a
  before/after measurement gate. Competitor behaviour is stated as
  general knowledge, not measured.
- Every refusal is `pulsim.UnsupportedError(engine, feature, device,
  kind)` backed by the capability table.
- The restart rule is three consecutive Alexander SDIRK(2,2) steps in
  every text and every implementation.
- Verification set for every implementation: ruff, pytest, ctest
  (`-j2` on the shared machine), `openspec validate --all --strict`,
  stub check.
- Decisions already taken by the owner stay: the absolute dx rule in the
  DC cascade; the flux row in conductance form; no "trapezoid at 2h"
  start; PSIM-style diode drop; refusals instead of silent fallbacks.

## Corrections found while the proposals were written

The proposals are authoritative where they differ from the review; the
rows above were updated for dependencies. The points below changed the
scope or the claim of a change.

- `add-discontinuity-restart-rule`: the rule only works with the gate
  mask sampled at the step START (with today's end-sampling it doubles
  the error) — a second breaking change. Prototype of the first form
  (two backward-Euler half-steps): buck CCM order 2.00, error
  1.28e-2 → 2.20e-5 A at the coarsest step. **Superseded 2026-11-13**:
  the pair drains resonant modes (Q = 50 tank at T/50: −11.9 % against
  −0.86 %); the rule is three consecutive Alexander SDIRK(2,2) steps
  (tank −0.56 % / −0.04 %, RL edge error 30–48 times smaller; on the
  ideal DCM buck about 400 times more accurate than the pair,
  `add-branch-form-ideal-switch/probes/sdirk_restart_ideal.out`).
- `add-fixed-step-event-splitting`: with the restart rule alone the DCM
  buck is already second order in the prototype; the measured value of
  the split is the off-grid gate edge (8.0e-2 V → 7.3e-6 V).
- `update-trbdf2-error-control`: the review's "error floor of 1.8e-4 V
  on a pulse-driven RC" is wrong (the error follows rtol^(2/3)); source
  breakpoints are motivated by 32 rejects and 1.3e-14 s steps instead.
- `update-trbdf2-post-event-restart`: prototype 28.2 → 8.1 solves per
  period (3.5×). The proposal memory without the derivative refresh is
  worse, so the landing order is fixed.
- `add-newton-roundoff-acceptance`: the step floor uses the point where
  roundoff exceeds 1e-6 of the solution, not the review's bound (which
  grows with stored flux). The link to the documented `h_min` stall is
  still an inference; both changes carry a "reproduce and report" task.
- `update-newton-persistent-solver`: the Newton path uses
  `Eigen::SparseLU`, so only stage A1 (persistent workspace, one
  symbolic analysis, same backend) is bit-identical; A2 (in-house LU)
  and B (low-rank update) change rounding and are gated separately
  (decision 4).
- `update-linear-time-assembly`: only assembly, shared symbolic analysis
  and fused passes are bit-identical; `partial_refactor` on mask flips
  and reciprocal pivots change rounding (decision 4).
- `update-composite-diode-forward-drop`: `add_igbt` has no diode; the
  change covers `add_mosfet`, `add_mosfet_with_body_diode` and the two
  Level-1 builders.
- `add-engine-conformance-matrix`: `has_nonlinear_devices()` is one of
  four diverging rules. `NativeMultiMaskPwm` on trbdf2 was measured: a
  1 µs window in a 1 ms period sees 0 of 40 gate events with no warning.
- `add-periodic-steady-state-shooting`: saltation is not what makes DCM
  work with today's resistor-pair diode (the field is continuous at
  turn-off); Newton on the true state with located events is. Saltation
  matters for state-dependent gate edges and ideal switches. Slice 1:
  DCM buck 8.996247087236 V against the 50-digit 8.99624708728943 V in
  5 map evaluations.
- `add-exact-propagator-strategy`: the review's 2e-8 DCM error was
  mostly its own reference; against a 50-digit reference dsed is about
  1e-10. A Padé fallback is not safe on stiff masks, so a stiff and
  defective mask goes to the implicit tier. The `V_f > V_th` case gave
  a 10.37 V error and is refused on `feat/core-integration`; slice 1
  (side branch) removes the refusal with its look-ahead on the exact
  OFF trajectory. Re-centred 2026-11-13: the
  `n_state ≤ 64` limit is replaced by the scaling rule; Krylov-type
  exponentials are no longer a Non-Goal.
- `add-branch-form-ideal-switch`: re-centred 2026-11-13 — default at
  the flip instead of opt-in; SDIRK(2,2)×3 instead of the
  backward-Euler pair; diode-bounded islands are a gauge instead of an
  error; LCP mode selection instead of "a Non-Goal".
- `add-high-order-stiff-integrator`: helps only where steps are
  accuracy-limited (LC tank: 4.7–7.6× fewer solves than TR-BDF2); on
  event-limited circuits it is about 2.5× slower per step.
- `add-mna-native-machines`: BLDC uses a PM-flux harmonic series, not a
  flux table (a trapezoidal back-EMF table is only C¹).
- `refactor-unified-transient-stepper` retires `core/integrators/`,
  which the change `add-adaptive-runge-kutta-solvers` built on; that
  change is not on this branch (see decision 9).
- `add-external-device-abi` adds requirements to `device-interface`,
  which exists only after `add-unified-device-interface` is archived.

## Decisions taken (maintainer, delegated by the owner on 2026-11-13)

1. **Two-pillar re-centring.** The ideal switch with complementarity
   mode selection (CORE A) and exact propagation as the default
   (CORE B) are the centre of the roadmap; Phase 3 is designed around
   them; implicit integrators are strategies.
2. **Restart rule** = three consecutive Alexander SDIRK(2,2) steps
   (`γ = 1 − 1/√2`), superseding the backward-Euler pair; opt-in until
   the default flip, which is deferred to decision 3's release.
3. **One number-changing release.** All number-changing defaults flip
   together in one release with one re-baseline through the reference
   harness: the restart rule on, gate sampling at the step start, the
   composite diode drop, the TR-BDF2 estimator factor and step grid,
   the ideal switch as the default (with the complementarity selection
   and the exact route of `auto`).
4. **Rounding-changing linear algebra now.** The stages that change
   rounding (Newton stages A2/B, `partial_refactor` on mask flips,
   reciprocal pivots) are taken now, each behind a gate: the same
   iteration counts on the harness, and states within 1e-9 relative of
   the bit-identical path.
5. **`add_nonlinear_diode` retired** (`fix-nonlinear-diode-passivity`
   ships as its retirement).
6. **Typed refusals everywhere**: `pulsim.UnsupportedError(engine,
   feature, device, kind)` and the capability table; no silent
   fallback, no broad `except` that turns a refusal into a value.
7. **Optimisation-first Phase 4**: in-kernel event losses, parameter
   re-binding in the IR, sensitivities, PSS completion and the
   design-optimisation workflow before the other Phase 4 changes.

Where the earlier list of owner questions now stands:

8. Releases (old 1) → decision 3. Typed errors (old 2) → decision 6;
   `UnsupportedError` stays a `ValueError` subclass (as integrated).
   Undriven controlled switch (old 3) → refused on every engine
   (decision 6). Composite diodes (old 4) → the drop flips with
   decision 3; the keyword question stays in that change. Newton ports
   to the device interface (old 5) → 1e-10 relative with one
   re-baseline per kind, by the rule of decision 4. Benchmarks (old 6)
   → harness-only v1 reader (the side-branch implementation).
   Engine names (old 8) → strategy selectors: `'auto'` (default),
   `'exact'` (alias `'dsed'` for a legacy period), `'pwl'`, `'trbdf2'`;
   the `-legacy` engines live until the stepper's task group 13.
   Control delay (old 10) → stays an open question of
   `add-compiled-control-graph`.
9. `add-adaptive-runge-kutta-solvers` (old 7): withdrawn in favour of
   the stepper's strategies if it is reintroduced; `core/integrators/`
   retires with the stepper.

## Still the owner's

- **The project name.** A rename away from the P…SIM name skeleton is
  recommended before the first external users publish with it; the
  migration (package, imports, docs, repository) is a separate proposal
  once a name is chosen.
- **Parts**: which real power devices ship, and whether digitised
  datasheet curves may be redistributed (the shipped `lib/data` parts
  are synthetic).
- **Pushing and publishing**: branches, releases, PyPI.
