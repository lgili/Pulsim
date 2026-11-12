## Context

Switches and PWL diodes are `BranchKind::Switch` branches stamped as a
conductance chosen by the mask bit. The matrix pattern is the same for
every mask (`pwl/cache.hpp:1348-1350`), which is what the lazy per-mask
factor cache, the event-dt entries and the planned shared symbolic
analysis rely on. Diode decisions go through one law
(`SwitchedDiode::decide`, `crossing_signal`). The DSED path already
contains the mathematics of an ideal commutation in its stiff-limit
form: `dsed/event_projection.hpp` projects onto the slow manifold,
which "in the DAE limit" is `argmin ‖x⁺ − x⁻‖_M` under the new mode's
constraints, `M = diag(C, L)` (owner note *v2-roadmap-2026-08*).

Owner decisions that bound this change: PSIM-style diode (ideal switch
plus forward drop); refusals instead of silent fallbacks; no
"trapezoid at 2h" start. The saturable inductor's flux row stays in
conductance form (owner decision); this change touches switch branches
only.

## Goals / Non-Goals

- Goals: legal `R_on = 0` and `g_off = 0`; a mask-invariant sparsity
  pattern; a named error for floating nodes; a defined, conservative
  treatment of dependent states at commutation, with the impulse
  reported; no change to circuits that do not opt in.
- Non-Goals: changing the default switch model; variable-dimension
  matrices (merging nodes when a switch closes); arc or avalanche
  models for an interrupted inductor; automatic snubbers; ideal
  switches on the exact strategy in the first release; complementarity
  (LCP) solvers for simultaneous ideal diodes.

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

### D3. Floating nodes

With `g_off = 0` an open switch is no path. For each mask, on its first
visit, a union-find over the branches that connect in that mask
(resistors, inductors, voltage sources, closed switches and diodes,
capacitors for the transient matrix) finds the node sets with no path
to ground. A non-empty set is an `UnsupportedError`-family error
naming the mask, the nodes and the open switches on the boundary, with
the two remedies (a finite `g_off` on one of them, or a component to
ground). It runs before factorization, costs O(branches), and is cached
with the segment. The DC operating point uses the same pass with
capacitors excluded and inductors conducting (the rule of
`pwl/preflight.hpp`).

### D4. Dependent states at a commutation

A mask can make states dependent:

| Structure in the new mask | Constraint | If violated at `t⁻` |
|---|---|---|
| Cut-set of inductors, open ideal switches and current sources | `Σ ±i_L = Σ ±I_src` | voltage impulse |
| Loop of capacitors, closed ideal switches (`R_on = 0`) and voltage sources | `Σ ±v_C = Σ ±V_src` | current impulse |

Let the constraints be `K·x = d` on the state `x = (v_C, i_L)` and
`M = diag(C, L)` (with the inductance matrix for coupled windings). The
consistent state is the charge- and flux-conserving projection

```
x⁺ = x⁻ − M⁻¹·Kᵀ·(K·M⁻¹·Kᵀ)⁻¹·(K·x⁻ − d)
Λ  = (K·M⁻¹·Kᵀ)⁻¹·(K·x⁻ − d)         transferred flux (V·s) / charge (C) per constraint
ΔW = ½·(x⁻ − x⁺)ᵀ·M·(x⁻ − x⁺) ≥ 0    energy removed by the impulse
```

Realisation on the implicit engines: one backward-Euler step of size
`h` from `x⁻` under the new mask solves `(C/h + G)·x = (C/h)·x⁻ + b`,
whose differential part equals the projection up to `O(h)` and whose
algebraic unknown carries the impulse as a pulse of area `Λ` (measured
in the prototype: the area is `L·i(t⁻)` and `6.0 µC` to rounding at
finite `h`). The two backward-Euler half-steps of
`add-discontinuity-restart-rule` are exactly that. The trapezoidal rule
must not take the first step after such an event: it delivers `2Λ` and
then alternates for ever (prototype: ±397.3 V, ±120 A).

What this change adds on top of the restart rule:

- detection, from the state jump across the restart sub-steps, that a
  commutation was inconsistent (`‖x⁺ − x⁻‖_M` above the event
  tolerance);
- an event record `{t, switch, kind: cut-set | loop, Λ, ΔW}`;
- the policy `on_impulse`: `'warn'` (default; a warning naming the
  switch and the energy), `'error'`, `'allow'`;
- the pulse sample is marked in the result so that averages and losses
  use `ΔW`, not the height of a one-step pulse.

A consistent commutation (a diode that stops at `i = 0` in DCM) has
`K·x⁻ = d`, hence no impulse and no record.

- Alternatives: explicit zero-time projection with `K` from a spanning
  tree — needed for the exact strategy (task 8), heavier than reusing
  the restart on the implicit engines; refuse every dependent-state
  mask — would refuse DCM; a small parasitic added automatically —
  the stiff modes again, and a hidden model.

### D5. Diode law in branch form

```
ON :  v = V_f + R_on·i_s       stays ON while i_s > 0
OFF:  i_s = g_off·v            turns ON when v ≥ V_on
crossing signal:  ON → i_s (amperes);  OFF → v − V_on (volts)
```

`i_s` is read from the solution vector, so the ON decision no longer
multiplies a rounding-level voltage by `g_on`. With `R_on = 0` and
`g_off = 0` this is the ideal diode `0 ≤ i ⟂ (V_f − v) ≥ 0`. The law
lives in `SwitchedDiode` beside the conductance form; decision sites
call the same functions with a form tag. Zero-band ties are resolved by
the derivative rule of `add-exact-propagator-strategy` where that
exists; elsewhere the existing advice (a band `V_th > V_f`) holds.

### D6. Interface and defaults

| Call | Form | Defaults |
|---|---|---|
| `add_switch(name, a, b, g_on, g_off)` | conductance (unchanged, bit-identical) | validated |
| `add_ideal_switch(name, a, b, R_on=0.0, g_off=0.0)` | branch | ideal |
| `add_diode(…, form='branch', R_on=0.0, g_off=0.0, V_f=0.0, V_th=0.0)` | branch | ideal |
| YAML `switch` / `diode` with `form: branch`, `R_on`, `g_off` | branch | |

Topology helpers keep the conductance form; a `form=` argument is added
only where a user asks (Open Questions).

### D7. Engines

| Engine | Branch form |
|---|---|
| Fixed trapezoidal | supported once the restart rule exists; before that, `R_on = 0` with `g_off > 0` and no capacitor loop only |
| TR-BDF2 | supported with the post-event restart |
| DC operating point, AC sweep | supported (same stamp) |
| Exact strategy (`dsed`) | refused by name until task 8 |

## Applicability rules

Use the branch form when the model must be ideal (a port from PSIM or
PLECS, a comparison with a textbook result), when a leak or an
on-resistance is the dominant error, or when the stiff modes hurt (the
exact strategy's Padé tier, explicit methods).

Do not use it by default: it adds an unknown per switch and turns
modelling mistakes that a leak hides (an inductor with no freewheel
path, two switches in series with nothing between them) into errors the
user must resolve.

Refused: `R_on < 0`, `g_off < 0`, non-finite values; `R_on = 0` and
`g_off = 0` on a switch in parallel with another ideal closed path
(a loop of closed ideal switches and voltage sources: structurally
singular in the ON mask), reported by D3's pass extended to loops.

## Risks / Trade-offs

- Pivot order differs between masks → shared symbolic analysis only;
  the partial-refactor fallback is counted and gated.
- Structural zeros dropped by an assembly cleanup would change the
  pattern → a test compares the pattern of all `2^k` masks of a small
  bridge.
- Users see new errors (floating nodes, impulses) on circuits that
  "ran" → every message names the element and the remedy; the default
  form is unchanged.
- The pulse of area `Λ` is `Λ/h` high and looks alarming in a plot →
  marked in the result and documented; losses use `ΔW`.
- Order reduction: `i_s` of an ideal ON switch across a capacitor is an
  index-2 unknown; its accuracy is first order after an event → states
  keep the engine's order (test), and the documentation says which
  unknowns do not.
- Simultaneous ideal diodes may have no unique mode from value tests →
  the derivative rule, or the band; a true LCP solver is a Non-Goal.

## Migration Plan

1. Characterisation tests of today's behaviour (`g_on = inf`,
   `g_off = 0`).
2. Validation of the conductance form (**BREAKING** for invalid
   values).
3. Branch stamp in the transient and DC assemblers; pattern-invariance
   test; pivoting and rank-one tests.
4. Floating-node pass; builder, YAML and Python API; `R_on = 0` with
   `g_off > 0`.
5. After Phase 2: `g_off = 0` and capacitor loops on the restart rule;
   impulse detection, record and policy.
6. Branch-form diode law; outputs and losses.
7. Counter gate; docs.
8. Exact strategy: per-mask constraints in the extraction and the
   explicit projection.

Rollback: the form is opt-in; removing the API leaves the conductance
form untouched.

## Open Questions

1. Default `on_impulse`: `'warn'` for both kinds (proposed), or
   `'error'` for an interrupted inductor and `'warn'` for capacitor
   charge sharing?
2. Should `g_off = 0` in the conductance form silently select the
   branch form, stay as it is, or be refused with a pointer to
   `add_ideal_switch`?
3. A floating node: always an error (proposed), or an optional "hold
   the last voltage" policy?
4. Should the topology helpers (`add_three_phase_vsi`,
   `add_bridge_rectifier`, `add_mosfet`) gain `form='branch'` in this
   change?
5. Is the exact strategy with ideal switches (task 8) part of this
   change or a follow-up after both changes have shipped?
6. Name: `add_ideal_switch`, or a `form=` keyword on `add_switch`
   only?
