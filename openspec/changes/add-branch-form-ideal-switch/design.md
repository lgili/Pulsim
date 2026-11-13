## Context

> **Revised 2026-11-13** (see the note at the top of proposal.md). D1
> and D2 are unchanged from the first version; D3–D10 replace the
> earlier D3–D7.

Switches and PWL diodes are `BranchKind::Switch` branches stamped as a
conductance chosen by the mask bit. The matrix pattern is the same for
every mask (`pwl/cache.hpp:1348-1350`), which is what the lazy per-mask
factor cache, the event-dt entries and the planned shared symbolic
analysis rely on. Diode decisions go through one law
(`SwitchedDiode::decide`, `crossing_signal`), to be replaced by the
complementarity service of `add-complementarity-mode-selection`. The
DSED path already contains the mathematics of an ideal commutation in
its stiff-limit form: `dsed/event_projection.hpp` projects onto the
slow manifold, which "in the DAE limit" is `argmin ‖x⁺ − x⁻‖_M` under
the new mode's constraints, `M = diag(C, L)` (owner note
*v2-roadmap-2026-08*).

Decisions that bound this change: PSIM-style diode (ideal switch plus
forward drop; owner); typed refusals; no "trapezoid at 2h" start; the
saturable inductor's flux row stays in conductance form (owner; this
change touches switch branches only); the restart rule is three
consecutive Alexander SDIRK(2,2) steps; every number-changing default
flips in one release (maintainer, delegated by the owner on
2026-11-13).

## Goals / Non-Goals

- Goals: legal `R_on = 0` and `g_off = 0`; a mask-invariant sparsity
  pattern; named refusals for structurally singular modes; one
  conservative re-initialisation service with the impulse reported;
  per-mode state dimension handled on every strategy; the ideal switch
  as the default at the end of the migration; losses that do not
  depend on a resistor pair.
- Non-Goals: variable-dimension matrices on the implicit route
  (merging nodes when a switch closes); arc or avalanche models for an
  interrupted inductor; automatic snubbers; switching transients
  resolved in time (the ideal switch commutes instantly, and the
  switching energy comes from tables); the decision rule for diodes
  (that is `add-complementarity-mode-selection`).

## Decisions

### D1. The stamp

Unknowns: node voltages and one branch current `i_s` per branch-form
switch, oriented from `a` to `b`.

```
KCL(a): … + i_s          KCL(b): … − i_s
row s :  α·(v_a − v_b) + β·i_s = γ
         ON :  α = 1,       β = −R_on,   γ = V_f
         OFF:  α = −g_off,  β = 1,       γ = 0
```

Structural entries, allocated in both states even when a value is
zero: `(a, s)`, `(b, s)`, `(s, a)`, `(s, b)`, `(s, s)`. An ON switch
with `R_on = 0` is a 0 V source; an OFF switch with `g_off = 0` is the
row `i_s = 0`.

`V_f` enters `γ` only (the right-hand side), as in
`add-switched-diode-forward-drop`.

- Alternatives: nodal conductances with a very large ratio (today) —
  the stiff modes; merging and splitting nodes per mask (how
  state-space ideal-switch simulators are usually described) — the
  dimension and pattern change with the mask, which discards the factor
  cache; a current-controlled form only (`i_s` always the unknown, `v =
  R·i`) — cannot express `g_off = 0`.

### D2. Linear algebra

- The pattern is the union above, identical for every mask, so one
  column ordering and symbolic analysis serve all masks.
- The diagonal entry `(s, s)` is zero when ON with `R_on = 0`, as for a
  voltage source, so numeric factorization needs partial pivoting; the
  in-house LU has it (`sparse/pulsim_lu_solver.hpp`, section 3). A
  refactor "without pivot search" is therefore not valid across a flip
  of a branch-form switch; it stays valid across a change of `dt`
  within one mask.
- A single flip replaces row `s` (three entries): a rank-one change
  `e_s·δᵀ`. `solve_rank1` applies; the path-based partial refactor
  applies when the pivot order is unchanged and otherwise falls back to
  a full numeric refactor, counted in `CacheMetrics`.
- The coefficients of row `s` are O(1) in both states (`1, −R_on` and
  `−g_off, 1`), so no row equilibration is planned; task 3.3 measures
  the pivot growth to confirm it.


### D3. Structural checks per mode, with named refusals

With `g_off = 0` an open switch is no path and with `R_on = 0` a closed
one is a 0 V source. For each mode, on its first visit, a union-find
over the branches that connect in that mode (resistors, inductors,
voltage sources, closed switches and ON diodes, capacitors for the
transient matrix) finds the node sets with no path to ground, and a
second pass finds loops of closed ideal switches and voltage sources.

| Finding | Result |
|---|---|
| island whose boundary contains an OFF diode | gauge (`add-complementarity-mode-selection` D10): the potential is reported at the centre of the interval the diodes allow, the nodes are listed in `diagnostics['gauge_nodes']` |
| island bounded only by open gated switches | `UnsupportedError(engine, 'floating_node', switch, 'switch')`, naming the mode, the nodes, every switch on the boundary and the two remedies (`g_off > 0` on one of them, or a component to ground) |
| loop of closed ideal switches (`R_on = 0`) and voltage sources | `UnsupportedError(engine, 'ideal_voltage_loop', switch, 'switch')`, naming the switches and sources and the remedy (`R_on > 0`); a shoot-through gate pattern is the usual cause |
| loop of capacitors closed by ideal switches | legal: dependent states, D4 |
| cut-set of inductors opened by ideal switches | legal: dependent states, D4 |

The checks run before factorization, cost O(branches) per new mode
and are cached with the mode. The DC operating point uses the same
pass with capacitors excluded and inductors conducting (the rule of
`pwl/preflight.hpp`). This supersedes "a floating node is always an
error" of the first version for islands bounded by diodes.

### D4. Re-initialisation: one service, two realisations

A mode can make states dependent:

| Structure in the new mode | Constraint | If violated at `t⁻` |
|---|---|---|
| cut-set of inductors, open ideal switches, OFF ideal diodes and current sources | `Σ ±i_L = Σ ±I_src` | voltage impulse |
| loop of capacitors, closed ideal switches, ON ideal diodes and voltage sources | `Σ ±v_C = Σ ±V_src` | current impulse |

Let the constraints be `K·x = d` on the state `x = (v_C, i_L)` and
`M = diag(C, L)` (with the inductance matrix for coupled windings). The
consistent state is the charge- and flux-conserving projection

```
x⁺ = x⁻ − M⁻¹·Kᵀ·(K·M⁻¹·Kᵀ)⁻¹·(K·x⁻ − d)
Λ  = (K·M⁻¹·Kᵀ)⁻¹·(K·x⁻ − d)         transferred flux (V·s) / charge (C) per constraint
ΔW = ½·(x⁻ − x⁺)ᵀ·M·(x⁻ − x⁺) ≥ 0    energy removed by the impulse
```

The service (`ReinitService::apply(mode, x⁻, t) -> {x⁺, records}`)
owns the detection (`K` from the D3 pass), the record
`{t, devices, kind: cut-set | loop, Λ, ΔW}`, the policy
`on_impulse = 'warn' (default) | 'error' | 'allow'`, and the marking of
the pulse sample in the result (averages and losses use `ΔW`, never
the height of a one-step pulse). It is called after the mode selection
at every event; a consistent commutation (`K·x⁻ = d`, e.g. a diode that
stops at `i = 0` in DCM) produces no record.

**State dimension per strategy.**

- *Exact propagator* (`add-exact-propagator-strategy`): per mode, the
  reduced state has dimension `n_x − rank(K)` (gauge directions move
  no state and do not count). The
  projection is applied explicitly at mode entry; the flow continues in
  the reduced coordinates; the saltation matrix of
  `periodic_steady_state` uses the projection as the reset map. The
  prototype (`../add-complementarity-mode-selection/probes/lcp_core.py`,
  class `Mode`) builds the reduced dynamics from the state-held MNA by
  SVD: left null space → constraints, right null space resolved by
  differentiating the constraints, remaining free directions → gauge.
- *Implicit strategies* (fixed trapezoid, TR-BDF2): the MNA dimension
  never changes (D1 keeps the pattern). The projection is realised by
  the restart rule: three consecutive Alexander SDIRK(2,2) steps
  (`γ = 1 − 1/√2`) after the event, then the trapezoidal rule. The
  stage matrix `E/(γh) − J` is the trapezoidal matrix at
  `dt_eq = 2γh`, so the factor cache serves it. Measured
  (`probes/sdirk_restart_ideal.out`): the flux removed in the cut-set
  equals the SDIRK quadrature of the node voltage over the first step
  (−9.93096483e-3 V·s both), the capacitor loop moves 6.0 µC and loses
  24.0 µJ exactly; the first post-event sample of the algebraic unknown
  carries the impulse (959.018799 V; −289.7 A) and is marked; from the
  second step on it is 0 (below 1e-14 A). The trapezoidal rule must not
  take the first step after such an event: it rings for ever
  (±397.307628 V, ±120 A).

The backward-Euler pair of the first version also conserves flux and
charge (prototype (1), (2)) but drains resonant modes: on the Q = 50
tank the energy error after the restart is −11.75 %/−3.13 %/−0.78 %
for the BE pair against −0.56 %/−0.04 %/+0.01 % for three SDIRK(2,2)
steps, and the RL edge error is 30–48 times smaller with SDIRK
(prototypes `p30_restart_sdirk_proto.py`, `p31_restart_sdirk_multi.py`
behind the 2026-11-13 decision, recorded in the maintainer's plan
outside this repository; the rule is being re-implemented on
`feat/core-phase0`), and on the ideal DCM buck it is about 400 times less
accurate than SDIRK×3 at the same step (prototype (3): 1.71e-5 V
against 4.35e-8 V at N = 400). It is superseded.

- Alternatives: explicit zero-time projection on the implicit route as
  well — possible (the service computes it anyway for the record) and
  kept as a test oracle, but the SDIRK restart is needed after every
  discontinuity for accuracy regardless, and realises the projection
  without a second code path; refuse every dependent-state mode — would
  refuse DCM; a small parasitic added automatically — the stiff modes
  again, and a hidden model.

### D5. Diode in branch form, decided by complementarity

```
ON :  v = V_f + R_on·i_s           OFF:  i_s = g_off·v
crossing signals:  ON → i_s (amperes);  OFF → w = V_f + R_on·i_s − v (volts)
mode at an event:  add-complementarity-mode-selection (one LCP)
```

`i_s` is read from the solution vector, so the ON decision no longer
multiplies a rounding-level voltage by `g_on`. With `R_on = 0` and
`g_off = 0` this is the ideal diode `0 ≤ i ⟂ V_f − v ≥ 0`. Between
events the signals are watched and located (on the exact trajectory or
the step polynomial); at the located time the LCP decides, for all
diodes together. This supersedes the first version's "derivative rule
or the band".

### D6. Interface and defaults

| Call | Before the flip | After the flip |
|---|---|---|
| `add_switch(name, a, b)` | conductance form, `g_on = 1e3`, `g_off = 1e-9` | branch form, ideal |
| `add_switch(name, a, b, g_on, g_off)` | conductance form, validated | branch form, `R_on = 1/g_on`, `g_off` (agrees within 1e-10, task 3.4) |
| `add_ideal_switch(name, a, b, R_on=0.0, g_off=0.0)` | branch form | alias of `add_switch` |
| `add_diode(…, form=…, R_on=…, g_off=…, V_f=0.0, V_th=0.0)` | `form='conductance'` default | `form='branch'` default |
| topology helpers (`add_three_phase_vsi`, `add_bridge_rectifier`, `add_mosfet`) | conductance form | follow the default; `form=` accepted |
| YAML `switch` / `diode` | `form: conductance` default | `form: branch` default |

`form='conductance'` stays accepted for the legacy period with a
deprecation warning after the flip, then is removed; its numbers are
reproduced by the branch form with the same values.

### D7. Engines

| Strategy | Branch form | State dimension | Restart |
|---|---|---|---|
| Exact propagator (`dsed`, `engine='auto'` exact route) | supported (task 8) | reduced per mode, explicit projection | none needed (exact flow) |
| Fixed trapezoid (`pwl`) | supported | full MNA, fixed | SDIRK(2,2) × 3 |
| TR-BDF2 | supported | full MNA, fixed | post-event restart (`update-trbdf2-post-event-restart`) |
| DC operating point, AC sweep | supported (same stamp) | — | — |
| Newton segments with nonlinear devices | supported (the switch is linear) | full MNA | as the implicit strategy |

Before task 5 lands (the restart on the implicit strategies), the
capability table refuses `g_off = 0` and `R_on = 0` across a capacitor
on the implicit strategies with `UnsupportedError(engine,
'dependent_states', switch, 'switch')`.

### D8. Losses

- Conduction: `R_on·i_s² + V_f·i_s`, integrated on the solved current
  (time-weighted: `np.trapezoid` on adaptive grids, owner note
  *time-weight-averages-on-adaptive-grids*).
- Switching: the ideal switch has no transition waveform, so the
  energy of each commutation is read from the device's tables at the
  event: `E_on(i, v, T_j)` and `E_off(i, v, T_j)` with the commutated
  current and the blocked voltage taken from the states at `t⁻` and
  `t⁺`, and `E_rr` for a diode turn-off from forward conduction. The
  energy is booked at the event in the loss record. This is the
  in-kernel event-loss record Phase 4 builds on (`CORE_ROADMAP.md`);
  in this change it is a per-event record and the existing
  post-processing reads it.
- Impulses: `ΔW` of D4, reported as a separate category, never mixed
  into device losses.

### D9. Why resistor-pair behaviour remains

A finite `R_on` and a finite `g_off` are physical (a MOSFET channel, a
leakage current). The branch form keeps both, as explicit parameters:
`R_on` is a series resistance in row `s`, `g_off` a parallel leak. What
disappears is the implicit pair that every switch carries whether the
user wants it or not, and with it the stiff modes. A user who needs
today's numbers states the values (task 3.4 shows the agreement).

### D10. Why the default flips

With the complementarity service the ideal switch needs no band and no
leak to run (prototype: every family runs with `R_on = 0`, `g_off = 0`,
`V_th = 0`), the exact route gets smaller, non-stiff modes, and ports
from PSIM and PLECS agree with their reference. The price is new
refusals on circuits that relied on the leak for a path (D3) and one
more unknown per switch on the implicit route (D7). Both are measured
before the flip (tasks 7 and 9.1): every example, tutorial and test
circuit is built with the ideal default and every refusal is listed
with its remedy.

## Applicability rules

Use explicit `R_on` and `g_off` when the device has them and they
matter (channel conduction, leakage-dominated idle intervals). Keep the
ideal default otherwise.

Refused: `R_on < 0`, `g_off < 0`, non-finite values; the D3 findings.

## Risks / Trade-offs

- Pivot order differs between masks → shared symbolic analysis only;
  the partial-refactor fallback is counted and gated.
- Structural zeros dropped by an assembly cleanup would change the
  pattern → a test compares the pattern of all `2^k` masks of a small
  bridge.
- New refusals at the flip → the migration audit of task 9.1, every
  message names the element and the remedy.
- The pulse sample is large (959 V for 9.93 A interrupted at
  `h = τ/20`) → marked in the result and documented; losses use `ΔW`.
- Order reduction: `i_s` of an ideal ON switch across a capacitor is an
  index-2 unknown; its first post-event sample carries the impulse.
  States keep the engine's order (prototype (3): order 1.89 → 1.98 on
  `v_C`).
- Event splitting on the fixed grid creates a sub-step factor per event
  (prototype (3): 603–606 factorisations over 300 periods) → counted;
  the event-dt cache of `update-fixed-step-loop-bookkeeping` bounds it.
- Cost on the implicit route at the flip → gate in task 7.

## Migration Plan

1. Characterisation tests of today's behaviour (`g_on = inf`,
   `g_off = 0`).
2. Validation of the conductance form (**BREAKING** for invalid
   values).
3. Branch stamp in the transient and DC assemblers; pattern-invariance
   test; pivoting and rank-one tests.
4. Structural checks; builder, YAML and Python API; `R_on = 0` with
   `g_off > 0`.
5. With the restart rule: `g_off = 0` and capacitor loops on the
   implicit strategies; the re-initialisation service and its record.
6. Branch-form diode with the complementarity service; outputs and
   losses.
7. Counter gate.
8. Exact route: reduced state per mode, explicit projection.
9. Default flip in the single number-changing release; legacy period
   for `form='conductance'`.

Rollback: before step 9 the form is opt-in; step 9 is reverted by
restoring the default, the branch form staying available.

## Decisions taken (2026-11-13)

The first version's open questions, decided by the maintainer under the
owner's delegation:

1. `on_impulse` default `'warn'` for both kinds; the record carries the
   energy, so a user can make it an error.
2. `g_off = 0` in the conductance form is refused with a pointer to the
   branch form until the flip; after the flip the conductance form
   maps to the branch form.
3. Floating nodes: islands bounded by OFF diodes are a gauge; islands
   bounded only by open gated switches are refused (D3).
4. The topology helpers follow the default and accept `form=` (D6).
5. The exact route with ideal switches is part of this change (task 8).
6. Name: `add_ideal_switch` now; `add_switch` builds the ideal switch
   after the flip and `add_ideal_switch` becomes an alias.
