# Core roadmap — index of OpenSpec changes (2026-11)

Source: the core review of 2026-11-12 (accuracy, speed, models,
architecture) on branch `feat/diode-forward-drop`. Raw audit reports and
the probe scripts that produced every number quoted in the proposals:
`/Users/lgili/pulsim_wt/core_review_2026-11/` (`raw/*.md`, `numerics/`,
`perf/`, `models/`, `arch/`, `CORE_REVIEW_2026-11.md`).

Each row is one change under `openspec/changes/<id>/`. Phases are
ordered; inside a phase, changes are independent unless "Needs" says
otherwise. Item codes (D, P, V, M, A) refer to the review.

Legend: effort S / M / L. **B** = changes numerical results or public
behaviour (deliberate release, migration note).

## Phase 0 — stop the silent wrong answers

| Change id | Scope | Items | Capabilities | Effort | Needs |
|---|---|---|---|---|---|
| `add-engine-conformance-matrix` | Per-device-kind capability table replacing the hand-kept `has_nonlinear_devices()` list; one `UnsupportedError` naming device + engine; conformance test (device × engine, feature × engine) with the invariant "no silent cell". Fixes: dsed and bdf1 drop `NonlinearCapacitor`; pwl with refresh off treats Newton devices as open; `NativeMultiMaskPwm` loses exact edges on trbdf2 | D3, D4, D10 | `engine-conformance` (new), `transient-timestep` | M | — |
| `fix-nonlinear-diode-passivity` | `IdealDiode` law made passive and monotone (softplus form); SPICE importer maps Ron/Roff/Vfwd diodes to the PWL diode with `V_f` | D1 | `device-models`, `python-bindings` | S | — |
| `fix-mna-sweep-correctness` | Kernel AC sweep: mutual inductance in `E`, caller-supplied switch mask, refusal when a Newton kind is present, no broad `except` | D2 | `ac-analysis` | S | — |
| `fix-control-tick-semantics` | Two closed loops without `dt`; per-observer rates on trbdf2 (multi-rate); `controller_period` honoured on pwl | D5, D6, D7 | `transient-timestep`, `python-bindings` | M | — |
| `fix-dsed-auto-integrator-selection` | dsed `integrator='auto'` uses the exact stepper whenever available; BDF2 bootstrap is backward Euler | D8 | `transient-timestep` | S | — |
| `update-composite-diode-forward-drop` | `add_mosfet` / `add_igbt` / Level-1 body and freewheel diodes take their `V_F` as a conduction drop (**B**) | D9 | `device-models`, `python-bindings` | S | — |
| `restore-reference-benchmark-harness` | Harness-only reader for the 126 v1 benchmark circuits; differential runner circuit × engine; analytic reference set; golden checkpoints of the fixed engine (tolerances, exact for discrete data); counter-based perf gate; nightly ngspice job | D11, A1 | `benchmark-suite` | M | — |

## Phase 1 — speed with bit-identical results

| Change id | Scope | Items | Capabilities | Effort | Needs |
|---|---|---|---|---|---|
| `update-fixed-step-loop-bookkeeping` | Hoist `masks_seen`; one `switch_fn` call per step; memoise the last mask/segment; C++ constant schedule when no `switch_fn` is given; sparse source passes; zero-copy `res.times` | V3, V4, V12 | `transient-timestep`, `python-bindings` | S | harness |
| `update-newton-persistent-solver` | One in-house LU per run for the Newton path, analysed once, values written in place; hoisted work vectors | V1 (stage A) | `linear-solver`, `transient-timestep` | M | harness |
| `update-trbdf2-factor-reuse` | Byte-budgeted event cache; allocation-free refactor; step size on a geometric grid with a dt-keyed factor cache (**B**: step sequence) | V2 | `transient-timestep`, `linear-solver` | S + M | harness |
| `update-linear-time-assembly` | Triplet/reserved assembly, per-circuit stamp-slot map; one symbolic analysis shared by all masks; single-bit flips through `partial_refactor`; fused triangular solve | V5, V6, V11 | `linear-solver`, `pwl-rank1-update` | M | harness |
| `update-sweep-threading-and-import-time` | Thread-pool sweeps when callbacks are C++ schedules; lazy import of the scope GUI stack | V9 | `parameter-sweep`, `python-bindings` | S | — |

## Phase 2 — accuracy of the method

| Change id | Scope | Items | Capabilities | Effort | Needs |
|---|---|---|---|---|---|
| `add-discontinuity-restart-rule` | Fixed engine: two backward-Euler half-steps (critical damping adjustment) at every mask change, source breakpoint and t = 0, reusing the nominal factor; gate mask sampled at the step start (**B**) | P1 | `transient-timestep` | S–M | harness |
| `add-fixed-step-event-splitting` | Fixed engine: PSIM-style step split at diode zero crossings and gate edges, then the restart rule (**B**) | P2 | `transient-timestep` | M | restart rule |
| `update-trbdf2-post-event-restart` | TR-BDF2: derivative refresh (or SDIRK(2,2) first step) after an event; step memory stores the controller proposal | P3 | `transient-timestep` | M | harness |
| `update-trbdf2-error-control` | Estimator scale factor; Hosea–Shampine stiff filter; step bounds independent of `t_end`; source breakpoints as events (**B**) | P6, P7 | `transient-timestep` | S | — |
| `add-newton-roundoff-acceptance` | Transient-only acceptance on stagnated dx plus row-wise backward error; step floor at the roundoff bound; DC cascade unchanged | P4 | `transient-timestep` | S–M | — |
| `add-fixed-step-resolution-diagnostics` | Per-step normalised LTE recorded in the result; Nyquist-ringing detector; warning when under-resolved | P5 | `transient-timestep`, `python-bindings` | S–M | — |

## Phase 3 — one kernel

| Change id | Scope | Items | Capabilities | Effort | Needs |
|---|---|---|---|---|---|
| `add-unified-device-interface` | Device = `f(x,t) + d/dt q(x)` with AD Jacobians, optional state / events / outputs / field table; flat device-state vector; generic `res.i` and loss outputs; AC and DC derived from f and q | M1, M4, A2 | `device-interface` (new), `device-models`, `ac-analysis` | M–L | conformance, harness |
| `add-unified-event-queue` | One queue for time events (gate edges, control ticks, source breakpoints, output grid) and state events (one Illinois); declarative modulators with analytic `next_edge` | A4, V10 | `event-scheduling` (new) | M | conformance |
| `add-kernel-simulation-object` | `RunPlan` resolved once above the dispatch, then a C++ `Simulation{Circuit, RunOptions}`; typed error model; one options struct | A3, A8 | `simulation-kernel` (new), `python-bindings` | M–L | conformance |
| `refactor-unified-transient-stepper` | `Stepper<Integrator, Controller>`; fixed-h trap mode proven equal to the pwl goldens; retire `bdf1`, the V0 overload and `integrators/` | A5, A11 | `transient-timestep` | L | device interface, event queue, harness |
| `add-probes-and-recorders` | Netlist-declared probes computed in-kernel; recorders (all / selected / decimated / `t_eval`); one result type that knows its grid | A7 | `simulation-results` (new) | M | device interface |
| `add-immutable-circuit-ir` | Immutable `Circuit` with named parameters and expressions, hierarchy kept, clone / `with_params` / `to_yaml`, domain-tagged nodes | A9 | `circuit-ir` (new), `netlist-yaml` | M–L | simulation object |

## Phase 4 — differentiators

| Change id | Scope | Items | Capabilities | Effort | Needs |
|---|---|---|---|---|---|
| `add-exact-propagator-strategy` | Exact LTI propagator as a first-class strategy: robustness gaps (V_th = 0, ICs, relative event tolerance, conditioning), overhead removal, per-mask Φ/Γ update and dense fast path | P9, V7, V8 | `exact-propagator` (new), `transient-timestep` | M–L | stepper |
| `add-periodic-steady-state-shooting` | One Newton shooting solver on the true state with located events and saltation matrices; works in DCM | P8 | `periodic-steady-state` (new) | M–L | first slice: none (ships on today's dsed); full scope: event queue, device interface |
| `add-power-mosfet-datasheet-device` | `add_power_mosfet` (channel + Cgs, Cgd(v), Coss(v), R_g, Lauritzen body diode); VDMOS import with capacitances; datasheet fitters; real validated parts; ngspice cross-validation | M2, M10, M11 | `device-models`, `python-bindings` | M | conformance |
| `add-in-kernel-event-losses` | Loss tables in C++; energy injected into the thermal node at each commutation; closed-loop T_j on every engine | M3 | `loss-thermal` (new) | M | event queue |
| `add-pwl-igbt-and-thyristor` | PWL IGBT with V_ce0 and reverse blocking; kernel thyristor / GTO latch | M5 | `device-models` | S–M | conformance |
| `add-mna-native-machines` | DC machine as an ideal-transformer composite; induction machine with rotor-flux states; BLDC via `PmsmMna` with a flux table | M6 | `motor-models` | S + M | device interface |
| `add-controlled-source-set` | VCCS, CCVS, CCCS, gyrator; sine / pulse / table current sources | M7 | `device-models`, `netlist-yaml` | S | — |
| `add-permeance-magnetic-domain` | Permeance–capacitor magnetic networks on the gyrator; nonlinear permeance; eddy and excess loss | M8 | `magnetic-models` | M–L | controlled sources, power MOSFET (charge-law variant), device interface |
| `add-external-device-abi` | User devices through a C ABI (f, q, Jacobians), then sympy-to-C codegen, then an OSDI host | M9 | `device-interface` | M, then L | device interface |
| `add-compiled-control-graph` | Control graph with rates, sorted schedule, loop detection, discrete blocks on tick events, C export | A6 | `control-graph` (new) | L | event queue |
| `add-parameter-sensitivities` | Forward sensitivities through the cached LU with event jump correction; steady-state gradients | A10 | `parameter-sweep` | L | device interface, stepper |
| `add-high-order-stiff-integrator` | Single-factor ESDIRK4(3) as an integrator strategy for resonant dynamics | P10 | `transient-timestep` | L | stepper |
| `add-branch-form-ideal-switch` | Branch-form switch stamp allowing true `R_on = 0` and `g_off = 0` on a fixed sparsity pattern | P11 | `device-models`, `linear-solver` | L | conformance (`R_on = 0` only); restart rule and TR-BDF2 post-event restart for `g_off = 0` |
| `add-realistic-passives` | ESR / ESL parameters, C(V) table, R_ac(f) ladder fit, transmission-line model | M12 | `device-models` | S–M | — |

## Rules that apply to every change

- No implementation starts before the owner approves the proposal.
- Every proposal quotes the measurement that motivates it and names the
  probe script that reproduces it. Speed-up figures taken from a time
  split are labelled as estimates and each `tasks.md` carries a
  before/after measurement gate.
- Verification set for every implementation: ruff, pytest, ctest,
  `openspec validate --all --strict`, stub check.
- Decisions already taken by the owner stay: the absolute dx rule in the
  DC cascade; the flux row in conductance form; no "trapezoid at 2h"
  start; PSIM-style diode drop; refusals instead of silent fallbacks.

## Corrections found while the proposals were written

The proposals are authoritative where they differ from the review; the
rows above were updated for dependencies. The points below changed the
scope or the claim of a change.

- `add-discontinuity-restart-rule`: the rule only works with the gate
  mask sampled at the step START (with today's end-sampling it doubles
  the error) — a second breaking change. Two half-steps are needed after
  the last flip, and the pair cannot be atomic for the diode event
  iteration. Prototype: buck CCM order 2.00, error 1.28e-2 → 2.20e-5 A
  at the coarsest step.
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
  and B (low-rank update) change rounding and are gated separately.
- `update-linear-time-assembly`: only assembly, shared symbolic analysis
  and fused passes are bit-identical; `partial_refactor` on mask flips
  and reciprocal pivots change rounding.
- `update-composite-diode-forward-drop`: `add_igbt` has no diode; the
  change covers `add_mosfet`, `add_mosfet_with_body_diode` and the two
  Level-1 builders.
- `add-engine-conformance-matrix`: `has_nonlinear_devices()` is one of
  four diverging rules. `NativeMultiMaskPwm` on trbdf2 was measured: a
  1 µs window in a 1 ms period sees 0 of 40 gate events with no warning.
- `add-periodic-steady-state-shooting`: saltation is not what makes DCM
  work with today's resistor-pair diode (the field is continuous at
  turn-off); Newton on the true state with located events is. Saltation
  matters for state-dependent gate edges and ideal switches. Prototype:
  DCM buck from zero converges in 5 map evaluations to 8.9962470872 V.
- `add-exact-propagator-strategy`: the review's 2e-8 DCM error was
  mostly its own reference; against a 50-digit reference dsed is about
  1e-10. A Padé fallback is not safe on stiff masks, so a stiff and
  defective mask goes to the implicit tier.
- `add-high-order-stiff-integrator`: helps only where steps are
  accuracy-limited (LC tank: 4.7–7.6× fewer solves than TR-BDF2); on
  event-limited circuits it is about 2.5× slower per step.
- `add-mna-native-machines`: BLDC uses a PM-flux harmonic series, not a
  flux table (a trapezoidal back-EMF table is only C¹).
- `refactor-unified-transient-stepper` retires `core/integrators/`,
  which the open change `add-adaptive-runge-kutta-solvers` builds on:
  one of the two has to give way.
- `add-external-device-abi` adds requirements to `device-interface`,
  which exists only after `add-unified-device-interface` is archived.

## Decisions the owner has to take

Each `design.md` ends with its own Open Questions. These are the ones
that cut across changes:

1. Releases: which release carries the numerically breaking changes
   (restart rule, gate sampling at the step start, composite diode drop,
   TR-BDF2 estimator factor and step grid), and do they ship together so
   references move once?
2. Typed errors: `UnsupportedError` as a `ValueError` subclass with a
   bridge for the two `NotImplementedError` sites, or one class deriving
   from both; how long the builtin bases stay.
3. Undriven controlled switch: refuse on every engine, or keep today's
   engine-dependent defaults.
4. Composite diodes: keep the keyword `V_F` or rename to `V_f`; default
   0.7 V (every default `add_mosfet` call changes) or 0.
5. Newton ports to the device interface: 1e-10 relative with one
   re-baseline per kind, or bit-identical (which rules out AD for the
   hand-stamped kinds).
6. Benchmarks: harness-only v1 reader (proposed) or rewrite the 126
   files; what to do with the 70 legacy baselines.
7. `add-adaptive-runge-kutta-solvers`: withdraw, or keep
   `core/integrators/`.
8. Engine names: keep `engine='pwl'` or add `fixed` / `variable`
   aliases; how long the `-legacy` engines live.
9. Parts: which three to five real power devices ship, and whether
   digitised datasheet curves may be redistributed.
10. Control: default computational delay (zero or one sample); whether
    `step_observer` / `b_extra_fn` stay as sugar.
