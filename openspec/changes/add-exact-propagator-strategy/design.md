## Context

`compute_lti_state_space(mask)` (`pwl/cache.hpp:453-1040`) reduces the
MNA system of one switch mask to `ẋ = A·x + b` on the capacitor voltages
and inductor currents, with a recovery map back to the full MNA vector.
`dsed/exact_lti.hpp` advances that system by any `h` through the complex
eigendecomposition; `dsed/event_projection.hpp` re-initialises the state
on the slow manifold at an event; `dsed/scheduler.hpp` locates predicate
roots on the analytic trajectory. Owner notes (*v2-roadmap-2026-08*):
explicit integrators are stability-limited on the stiff idle mode
(`τ = L·g_off ≈ 1e-13 s`), which is why the exact stepper exists; the
eigendecomposition was once built eagerly and cost 252 of 267 ms, so it
is lazy; sine sources are already exact through oscillator augmentation.

Constraints: no silent fallback (owner rule); the fixed engine's golden
hashes stay until the stepper change replaces them; shared 16 GB
machine, so gates are counters, not wall time.

## Goals / Non-Goals

- Goals: one propagator kernel with a stated accuracy per mask; no
  failure on a zero-band diode at rest; initial conditions, coupled
  inductors and piecewise-constant sources covered; per-step cost O(n)
  with no allocation; a fixed-step `Φ/Γ` form; a defined hybrid with
  Newton devices; an applicability rule that refuses instead of
  degrading.
- Non-Goals: Krylov exponentials for large sparse systems (`expmv`,
  `phipm`); a block Schur–Parlett exponential; exactness for arbitrary
  time-varying sources; replacing the implicit strategies; opaque
  Python `switch_fn` polling (owned by `add-unified-event-queue`).

## Decisions

### D1. State, coordinates and the step formula

Per mask `m`: `ẋ = A_m·x + B_m·u(t)`, `x ∈ ℝⁿ`. With `u` constant on the
step (`b = B·u`):

```
x(t+h) = Φ·x(t) + Γ₀·b            Φ = e^{Ah},  Γ₀ = h·φ₁(Ah)
u linear on the step:  + Γ₁·B·(u₁ − u₀),     Γ₁ = h·φ₂(Ah)
φ₁(z) = (e^z − 1)/z,   φ₂(z) = (e^z − 1 − z)/z²
```

Modal form, `A = VΛV⁻¹`, `z = V⁻¹x`, `w = V⁻¹b`:

```
z(t+h) = e^{Λh}·z(t) + h·φ₁(Λh)·w          (component-wise, O(n))
```

Energy coordinates before the eigenproblem: `x̃ = D·x`, `D = diag(√C_i,
√L_j)` (for coupled inductors the Cholesky factor of the inductance
matrix). In these coordinates a lossless network has a skew-symmetric
`Ã`, hence an orthogonal eigenbasis, and the condition number measures
physics, not units. The scaling is done explicitly instead of relying
on whatever balancing the eigen-solver applies.

φ₁ for complex `s = a + ib`, no cancellation at any `|s|`:

```
e^s − 1 = expm1(a)·cos b − 2·sin²(b/2) + i·e^a·sin b
φ₁(s) = (e^s − 1)/s   for s ≠ 0;   φ₁(0) = 1
φ₂(s) = (φ₁(s) − 1)/s for |s| ≥ 0.5; Taylor to s¹² below (remainder < 1e-16)
```

- Alternatives: keep the series switch at 1e-6 and widen it — rejected,
  any fixed switch has a cancellation shoulder (measured 6.5e-11 at
  `s = 1.01e-6`).

### D2. Three tiers per mask, each with an acceptance bound

| Tier | Method | Accepted when | Cost |
|---|---|---|---|
| 1 | eigendecomposition in energy coordinates, modal stepping | `u·κ₂(Ṽ) ≤ 1e-10` (κ₂ by SVD, once per mask) | O(n³) once, O(n) per step, O(n²) per recorded sample |
| 2 | Padé [13/13] scaling and squaring (Higham 2005) of `[[A, b],[0, 0]]·h`, cached per `(mask, h)` | tier 1 refused and `u·‖A·h‖₁ ≤ 1e-10` | O(n³) per distinct `h` |
| 3 | implicit strategy for the steps taken under that mask | otherwise | as the implicit engine |

`u = 2.2e-16`. Measured basis for the two bounds
(`probes/propagator_accuracy.py`, `probes/phi_gamma_fixed_step.py`):

| Case | Quantity | Error |
|---|---|---|
| RLC, `R/R_crit − 1 = 1e-9`, κ(V) = 7.1e5 | modal against Padé | 6.1e-13 |
| `R/R_crit − 1 = 1e-13`, κ(V) = 7.1e7 | same | 2.3e-10 |
| `R = R_crit` exactly, κ(V) = 2.3e9 (still "invertible") | same | 7.9e-9 |
| buck idle mode, `‖A·h‖ = 1e7`, `Φ₂₂` against 50 digits | eigen | 6.2e-17 |
| same | Padé | 8.2e-11 |

Padé loses about `u·‖A·h‖` on the slow modes of a stiff matrix, and its
error is a staircase in `h`: in the shooting prototype it made Newton
converge linearly (rate 0.3) instead of quadratically until the eigen
form replaced it. With resistor-pair switches every open switch in
series with an inductor makes `‖A‖ ≈ 5e12`, so tier 2 serves non-stiff
defective masks only; a mask that is both stiff and near-defective goes
to tier 3. `add-branch-form-ideal-switch` removes the stiff modes and so
widens tier 2.

- Alternatives: block Schur–Parlett (Davies & Higham 2003) covers stiff
  and defective together but needs a reordered Schur form and Sylvester
  solves that Eigen does not provide — deferred, see Open Questions.
  Jordan form — numerically unusable. Always Padé — loses the O(n) step
  and 6 digits on stiff masks.

### D3. Event location on the analytic trajectory

A predicate is affine in the state, `g(x) = cᵀx + d` (diode `v − V_on`
or `v − V_f`, a comparator). In modal coordinates
`g(τ) = Σᵢ (cᵀV)ᵢ·[e^{λᵢτ}zᵢ + τ·φ₁(λᵢτ)·wᵢ] + d`, O(n) per evaluation
with the row `cᵀV` cached per `(mask, predicate)`.

- Root finder: Illinois with the bisection fallback, unchanged.
- Stop rule: `|t_b − t_a| ≤ max(4u·|t|, 1e-12·h_seg)`, or `g = 0`
  exactly. The value test `|g| < 1e-10` is removed: it is unit-dependent
  and stops early on a micro-amp predicate.
- Scan step: the segment is scanned for a sign change at
  `h_scan = min(h_seg, π/(4·ω_max))`, `ω_max = max |Im λᵢ|` over modes
  with `|(cᵀV)ᵢ·zᵢ| > 1e-12·g_scale`. A resonant mode cannot cross twice
  between two scan points spaced an eighth of its period.
- Alternatives: a guaranteed root isolation for exponential sums —
  no cheap algorithm; Hermite interpolation — less accurate than the
  closed form already available.

### D4. Mode selection when a crossing signal is zero

At a located event, or at `t = 0`, a diode whose crossing signal is
within the event tolerance of zero is decided by the sign of the first
non-zero derivative of that signal in the candidate mode (lexicographic
rule of linear complementarity systems):

```
OFF is consistent  ⇔  d/dt (v_D − V_on) |_OFF ≤ 0
ON  is consistent  ⇔  d/dt (v_D − V_f)  |_ON  ≥ 0
σ̇ = rᵀ(A_m·x + b_m)        r = the node-pair recovery row of the diode
```

up to the second derivative; the settle is Gauss–Seidel (the mask is
updated after each diode). A diode away from zero keeps
`SwitchedDiode::decide`. If no candidate is consistent, the error names
the diodes and prints `t` in `%.9e`.

At rest with every switch open the rule is expected to select OFF, the
state `V_th = 1e-9` reaches today; task 1.3 confirms it before the rule
is written.

- Alternatives: require a band (today's advice) — keeps a failure for
  the default `V_th = 0`; enumerate all `2^k` masks — exponential.

### D5. Initial conditions

The reduced state is given per device, not gathered from the MNA
vector: `x_s[cap] = c0` (branch voltage `v_from − v_to`),
`x_s[ind] = i0`. Where the extraction merged dependent states (a
capacitor loop), the declared values are projected with the mass-norm
projection `argmin ‖x⁺ − x⁰‖_M` subject to the loop constraint (charge
conservation: two capacitors at `v₁, v₂` start at
`(C₁v₁ + C₂v₂)/(C₁ + C₂)`), and the adjustment is reported in the result
diagnostics. `set_initial` on a node voltage that is not a state is
refused by name.

### D6. Coupled inductors

Step 7 of the extraction applies the mass matrix per row (`1/C`,
`−1/L`). With mutual inductance the inductor block of the mass matrix is
the symmetric inductance matrix `𝐋` (`L_ij = k_ij·√(L_i·L_j)`); the
block is applied as `𝐋⁻¹` through its Cholesky factor. `𝐋` not positive
definite (`|k| = 1`) is refused: it is an algebraic constraint between
currents and needs the index reduction of an ideal transformer.

### D7. Sources

| Source | Treatment |
|---|---|
| DC | `b` constant per mask |
| PWM, pulse, square | piecewise constant; each breakpoint is a time event from the event queue; `b` constant between events |
| Sine | existing oscillator augmentation (`builder_adapter.hpp:405-499`) |
| Piecewise-linear table, control output under first-order hold | φ₂ term |
| Control output under zero-order hold | constant between ticks; a tick is a time event |
| Arbitrary `b_extra_fn(t)` | not exact: implicit strategy, reason recorded |

### D8. Fixed-step form

With a fixed `dt` and gate edges on the grid, `(Φ_m, Γ_m)` are dense
real matrices cached per `(mask, dt)` in a byte-budgeted LRU; a step is
`x ← Φ_m·x + Γ_m` (n² multiply-adds; 8 for a buck). A diode event
between grid points is located on the analytic trajectory and taken as
two exact sub-steps from the modal form (not cached). The full MNA
vector is recovered only for recorded samples and probes. This is the
`Fixed{dt}` controller of the unified stepper with the exact integrator;
it has no truncation error, so it is not bit-compatible with the
trapezoidal goldens and is never selected implicitly for `engine='pwl'`.

### D9. Hybrid with Newton devices

A device kind may declare `affine_region(x) → id | none`: its `f` and
`q` are affine on a region around `x` (a PWL table device on one table
segment, a flux-table inductor on one flux segment, a nonlinear
capacitor with a PWL `q(v)`). The mode key is `(mask, region ids)`; the
extraction uses the region's affine model; the region boundaries are
state events. A step in which any device returns `none` (Shockley
diode, MOSFET Level 1) is taken by the implicit strategy. Both
strategies advance the same physical state (capacitor voltages,
inductor currents, device states), so alternating between them needs no
conversion; on entry to an exact segment the reduced state is gathered
per device as in D5.

### D10. Result and counters

`result.exact` carries: per mask the tier and reason; `n_eig`, `n_pade`,
`n_phi_cache_hits`, `n_implicit_steps`, `n_predicate_evals`,
`n_alloc_steady` (heap allocations after warm-up, from the existing
operator-new counting harness).

## Applicability rules

The strategy is **selected by `auto`** when all hold:

1. every device is linear, a switch, a PWL diode, or declares an affine
   region at the current state;
2. every time-varying source is in the first five rows of D7;
3. `n_state ≤ 64` and `48·n²·(masks visited) ≤` the cache byte budget
   (complex `V`, `V⁻¹` and a real `Φ`: 48 bytes per entry);
4. the extraction succeeds (no inductor cut-set or capacitor loop it
   cannot reduce, no `|k| = 1`).

It is **refused** (explicit request, `UnsupportedError` naming the rule
that failed) or **not selected** (`auto`, with the reason in
`engine_route_reason`) otherwise. Rule 3 is the "when not to use it"
bound: a sparse step costs about 23 ns per unknown (perf row 4) and a
dense `Φx` about `n²` multiply-adds, which by the audit's estimate
crosses near n ≈ 50; 64 is a placeholder that task 12.2 replaces with a
measured value. Inside a run, a mask that fails D2 falls to tier 3 for
that mask only and is listed in the result.

## Risks / Trade-offs

- A wrong tier bound gives a confident wrong "exact" answer →
  the 50-digit references in the tests, and the tier and κ in every
  result.
- Modal stepping accumulates rounding in `z` over many steps of one
  segment → `z` is rebuilt from `x` at every event and every 1024 steps.
- The derivative rule changes which state a tied diode takes → it
  applies only inside the event tolerance; outside it the shared
  `decide` law is untouched, so the three engines still agree.
- The scan step raises predicate evaluations on resonant circuits →
  O(n) each; gated by the counter.
- Dense memory on many-mask circuits (a three-phase bridge visits 87
  masks) → rule 3 and the LRU.
- Tier 3 inside an "exact" run surprises a user → it is never silent,
  and a strict option turns it into a refusal.

## Migration Plan

1. Characterisation tests and 50-digit references (no behaviour
   change).
2. Kernel: φ accuracy, energy coordinates, κ bound, Padé tier, tier
   record. Results move at 1e-10.
3. Modal stepping, cached `w` and predicate rows, allocation removal,
   counters.
4. Event tolerance and scan step.
5. Zero-band mode selection.
6. Initial conditions; coupled inductors.
7. After Phase 3: sources through the event queue; exact integrator in
   the unified stepper; fixed-step `Φ/Γ`; contiguous buffer through the
   recorders.
8. Hybrid with affine-region devices.
9. Deprecate, then remove, the RK45 / BDF2 / auto schedulers.

Each step is shippable; rollback is per step (the previous kernel stays
selectable through a private option until step 9).

## Open Questions

1. Retire `integrator='rk45'` and `'bdf2'` on `engine='dsed'`
   (deprecate one release, then remove), or keep RK45 as a documented
   research integrator?
2. Tier 3 inside a run: record and continue (proposed default), or
   refuse unless the user opts in?
3. `n_state ≤ 64` as the `auto` limit until measured — acceptable, or
   should `auto` never pick the exact strategy above a lower bound such
   as 32?
4. Is a block Schur–Parlett exponential worth a later change, or does
   `add-branch-form-ideal-switch` make stiff-and-defective masks rare
   enough?
5. Should the fixed-step `Φ/Γ` form get its own engine name (for
   example `engine='exact'` with `dt=`), given that it must not answer
   to `engine='pwl'`?
6. First-order hold for control outputs: default zero-order hold
   (matches a digital controller), first-order hold opt-in?

## Re-centred scope, 2026-11-13

> Appended; D1–D10 stand, except where a decision below names what it
> supersedes. Prototypes: `probes/expm_action_ladder.py` (scaling,
> sources), `../add-complementarity-mode-selection/probes/lcp_core.py`
> (ideal switches, reduced state per mode, mode selection, exact flow
> between events), `../add-branch-form-ideal-switch/probes/sdirk_restart_ideal.py`
> (the implicit comparison).

### D11. Exact propagation is the default route

`engine='auto'` decides per segment (the interval between two
consecutive events). A segment is **exact-eligible** when, at its
start event:

1. every device is linear, a switch (branch or conductance form), a
   PWL diode, or returns a region id from `affine_region(x)` (D9);
2. every source is in the exact classes of D14;
3. the mode's extraction succeeds (dependent states reduced by the
   projection of `add-branch-form-ideal-switch` D4; coupled-inductor
   matrix positive definite, D6);
4. the scaling rule (D13) finds an exact kernel within its budget.

An eligible segment is propagated exactly; otherwise it is taken by
the implicit strategy (D12) and the reason is recorded. With an
explicit `engine='exact'` (alias `'dsed'`) a segment that is not
eligible is refused with `UnsupportedError(engine, feature, device,
kind)` naming the failed rule in `feature` (`'device'`, `'source'`,
`'extraction'`, `'cost'`) and the offending device and its kind where
there is one (`device = ''`, `kind = ''` for `'cost'` and for a source
function), unless `allow_implicit_segments=True`.
This supersedes Applicability rule 3 (`n_state ≤ 64`) and task 12.2's
placeholder: size is a cost question, answered by D13. The spec
requirement "Applicability Limit" keeps its form (a limit, a route
reason, a refusal with both numbers); after task group 15 the limit it
names is the D13 cost rule, not a state count.

### D12. Hybrid rule for Newton segments

- **Classification happens at events only.** A device leaves an
  affine region at a region-boundary event; a device that has no
  affine region at the current state (a Shockley diode, a MOSFET
  Level 1 in its square-law region) makes the segment implicit.
- **An implicit segment lasts until the next event** of the queue
  (gate edge, source breakpoint, controller tick, located crossing of a
  PWL device, region boundary of an affine device, or a declared
  region re-entry of a smooth device: `affine_region` is re-tested at
  each accepted implicit step and its first success schedules an
  event at that step's end). There is no mid-step switch.
- **Entry to an implicit segment** is a discontinuity of the vector
  field, so the implicit strategy starts with its post-event restart
  from the exact state: the restart rule (three consecutive Alexander
  SDIRK(2,2) steps) on the fixed trapezoid, and on TR-BDF2 the restart
  of `update-trbdf2-post-event-restart` — or SDIRK(2,2) × 3 when the
  entry event has dependent states (`add-branch-form-ideal-switch`
  D4). **Entry
  to an exact segment** gathers the reduced state per device (D5) and
  applies the re-initialisation service if the new mode has dependent
  states.
- **Which implicit strategy**: TR-BDF2 with error control by default;
  the fixed-grid trapezoid with the SDIRK restart when the run has a
  fixed `dt`. Newton tolerances are the implicit strategy's own (the
  absolute `dx` test stays; owner note
  *newton-absolute-dx-is-load-bearing*).
- **No flapping**: a device that re-enters and leaves an affine region
  more than 8 times inside one switching period is held implicit for
  the rest of that period; counted.
- **Counters**: `n_exact_segments`, `n_implicit_segments`,
  `t_exact_fraction`, and per hand-over `{t, device, reason}`.

### D13. Scaling rule: dense, action of the exponential, or hand-over

For a mode with `n` states, `nnz` non-zeros in `A`, step or segment
length `h`, and `N_m` steps expected in the mode (from previous visits;
first visit: segment length / `h`):

| Kernel | Setup | Per step | Exact? |
|---|---|---|---|
| dense modal (tier 1) / dense Padé (tier 2) | ≈ `2n³·(6 + log₂‖A·h‖₁)` flops, `48n²` bytes cached | `2n²` | yes |
| action of the exponential (truncated Taylor with scaling, Al-Mohy & Higham 2011) | none | ≈ `2·nnz·m` with `m` the kernel's own predicted matvec count, growing about linearly in `‖A·h‖₁` (measured: 20 at 2, 2492 at 1e3) | yes (to the tolerance `u`) |
| implicit hand-over (TR-BDF2, or trapezoid on a fixed grid) | one sparse LU | one solve | no (second order) |

Rule, per mode at its first visit (re-evaluated when `N_m` changes by a
factor 4):

1. If the dense setup fits the cache budget and
   `setup_dense + N_m·step_dense ≤ N_m·step_action`, use dense
   (tier 1, or tier 2 when D2 refuses tier 1).
2. Else, if `m ≤ m_max` (`m_max = 200` matvecs, an estimate to be
   measured), use the action.
3. Else hand over to the implicit strategy for this mode, with reason
   `'cost'`.

Measured basis (`probes/expm_action_ladder.out`, flop model; wall time
in the probe is dominated by scipy's per-call overhead of about 300 µs
and is not used):

| Ladder | n | `‖A·h‖₁` | matvecs/step | action/dense flops per step | dense setup repaid after |
|---|---|---|---|---|---|
| RLC | 200 / 500 / 1000 / 2000 | 2 | 20 | 0.25 / 0.10 / 0.05 / 0.02 | never (action is cheaper per step) |
| RLC + 1 ns snubbers | 300 / 750 / 1500 | 1e3 | 2492 | 24.7 / 9.9 / 5.0 | 203 / 1342 / 6027 steps |

`|dense − action| ≤ 3.2e-15` (non-stiff) and `≤ 1.6e-13` (stiff). The
implicit trapezoid on the same ladder costs 0.2–0.5 ms of LU and
7–28 µs per step and is 2.3e-2 off at this `h`.

Reading: the non-stiff large circuit stays exact through the action;
the stiff large circuit stays exact through dense when its modes last
long enough, and otherwise hands over. Ideal switches remove the
switch-induced stiffness (a resistor-pair switch gives `‖A‖ ≈ 5e12`,
proposal of `add-branch-form-ideal-switch`), which is what makes
rule 2 the common case for large converters. A rational Krylov
(shift-and-invert) action would cover stiff large modes at a few
sparse solves per step; it is not prototyped and is not in this
change; the hand-over is the fallback.

Counters per mode in `result.exact['routes']`: kernel, reason, `n`,
`nnz`, `‖A·h‖₁`, matvecs per step, estimated setup flops, steps taken.

### D14. Sources exact through augmented states

The source vector is `b(t) = B_s·s(t)` with `s` the state of an
autonomous source system `ṡ = S·s`, so that `[x; s]` evolves under
`[[A, B_s], [0, S]]`:

| Source | `s` | `S` |
|---|---|---|
| DC | `s₀ = 1` | 0 |
| ramp, polynomial of degree p | `(1, t, …, t^p/p!)` | nilpotent shift |
| PWL table | ramp on each segment; each corner a time event | nilpotent |
| `e^{−t/τ}` | one state | `−1/τ` |
| sine, cosine at ω | `(cos ωt, sin ωt)` | `[[0, −ω], [ω, 0]]` (one pair per frequency, shared) |
| controller output, zero-order hold | constant between ticks | 0 |

Equivalently, for the dense kernel, `x(t+h) = e^{Ah}x + Σ_k h^{k+1}φ_{k+1}(Ah)·b^{(k)}` with
φ-functions; the augmented form is used because it shares the dense
and action kernels. Measured: sine into RL at `h = T/20`, 1.6e-12 A
augmented against 17 A held (`probes/expm_action_ladder.out`); every
LCP prototype family with a sine source uses this form (event times
within 1.1e-14 s). An arbitrary Python `b_extra_fn(t)` is not exact:
the segment is implicit with reason `'source'`.

### D15. Output grid and controller ticks are events

- A recorded sample at `t_k` is an evaluation of the flow from the last
  event: `x(t_k) = e^{A(t_k − t_e)}·x(t_e)` (modal: O(n); dense with a
  fixed sample spacing: one cached `Φ(dt_out)`). It is not a step and
  does not touch the event tolerance or the mode.
- A controller tick is a time event of the queue: the controller reads
  the state at the tick, its outputs update with a zero-order hold, and
  any gate change is a gate edge at the tick or at the computed PWM
  edge (`fix-control-tick-semantics`, done on `feat/core-integration`,
  fixes the sampling instant).
- The output grid never shortens the propagation: the number of
  exact segments is set by events, not by `dt_out`.

### D16. Ideal switches as the native switch model

The exact route takes branch-form ideal switches and diodes as its
first-class case: per mode the reduced state of dimension
`n_x − rank(K)` (`add-branch-form-ideal-switch` D4), gauge directions
excluded (`add-complementarity-mode-selection` D10), the mode at every
event from one LCP. The resistor-pair form stays accepted and goes
through D2's tiers as today. The slice-1 zero-band rule (D4) remains
the decision rule until the complementarity service is the default;
then D4 is retired.

### Decisions taken on the Open Questions (2026-11-13)

Taken by the maintainer under the owner's delegation:

1. `integrator='rk45'|'bdf2'` on `engine='dsed'`: deprecated for one
   release, then removed (as task 12.1).
2. A non-eligible segment inside an `auto` run: recorded and continued
   (implicit); with explicit `engine='exact'` it is refused unless
   `allow_implicit_segments=True`.
3. The `n_state ≤ 64` limit is replaced by D13.
4. Block Schur–Parlett: not in this change; the counters of D13 show
   whether stiff-and-defective modes remain after the ideal-switch
   default.
5. Name: `engine='exact'` with `'dsed'` as a legacy alias; the
   fixed-step Φ/Γ form is `engine='exact', dt=…`, never `'pwl'`.
6. Controller outputs: zero-order hold by default; first-order hold
   opt-in through the PWL source class.

### Risks (appended)

- The cost model of D13 mis-ranks kernels → it decides on counted
  quantities (`n`, `nnz`, `‖A·h‖₁`, `N_m`), and task 15 calibrates the
  constants against measured C++ times; a wrong ranking costs time, not
  accuracy.
- `m_max` and the dense budget are estimates → measured in task 15.
- Hybrid flapping on a device that hovers at a region boundary → the
  per-period hold of D12, counted.
- Default flip changes numbers on every circuit `auto` now routes to
  the exact strategy → flips with the other number-changing defaults,
  re-baselined.
