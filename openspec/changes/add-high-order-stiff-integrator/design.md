## Context

The variable-step engine is TR-BDF2 with `γ = 2 − √2`: a three-stage
ESDIRK of order 2 whose two implicit stages share the matrix
`G + 2C/(γh)`, i.e. the trapezoidal companion at `dt = γh`, served by
`cache.solve_at(mask, γh)` from the `(G, C, b)` split. Its history is
physical (voltages and currents), so a step of any size is legal.
Phase 3 turns the driver into `Stepper<Integrator, Controller>` and
moves the time formula of stateful devices into the stepper
(`f(x, t) + d/dt q(x) = 0`).

Constraints: a single real factorization per (mask, h); L-stability
(the resistor-pair switch puts modes at 1e-13 s into every circuit);
no explicit method as a fallback; tolerances mean what they say.

## Goals / Non-Goals

- Goals: a fourth-order implicit integrator that reuses the existing
  factor, with an embedded error estimate, for dynamics whose step is
  limited by accuracy; no per-device code.
- Non-Goals: replacing TR-BDF2 as the default; orders above four;
  fully implicit methods (Radau, Gauss); symplectic integration;
  multistep (BDF/NDF) with variable order; a continuous extension of
  fourth order.

## Decisions

### D1. The method

ESDIRK4(3)6L[2]SA — Kennedy & Carpenter, *Additive Runge–Kutta schemes
for convection–diffusion–reaction equations*, Appl. Numer. Math. 44
(2003) 139–181: the implicit tableau of ARK4(3)6L[2]SA (to my knowledge
the table SUNDIALS ships as `ARKODE_ARK436L2SA_DIRK_6_3_4`; task 1.1
checks the coefficients against it). Six stages, the first explicit,
`γ = 1/4` on the diagonal of the other five.

```
c = [0, 1/2, 83/250, 31/50, 17/20, 1]

a21 = 1/4
a31 = 8611/62500          a32 = −1743/31250
a41 = 5012029/34652500    a42 = −654441/2922500     a43 = 174375/388108
a51 = 15267082809/155376265600   a52 = −71443401/120774400
a53 = 730878875/902184768        a54 = 2285395/8070912
a61 = 82889/524892   a62 = 0   a63 = 15625/83664   a64 = 69875/102672   a65 = −2260/8211
a_ii = 1/4 (i = 2 … 6)

b  = row 6 (stiffly accurate)
b̂ = [4586570599/29645900160, 0, 178811875/945068544,
      814220225/1159782912, −3700637/11593932, 61727/225920]
```

Verified in exact rational arithmetic
(`probes/esdirk43_lc.py`, `probes/esdirk43_stability.py`):

| Property | Result |
|---|---|
| Row sums | equal `c` |
| Order conditions of `b`, orders 1–4 (8 conditions) | all exactly 0 |
| Order conditions of `b̂`, orders 1–3 | exactly 0; order-4 residuals 1.4e-3, 7.2e-4, −4.5e-4, −2.2e-4 |
| Stage order 2 (`A·c = c²/2`, stages 2–6) | exactly 0 |
| Stiff accuracy | `b = a₆` |
| `R(−∞)` | exactly 0 |
| `max |R(iy)|`, `y ∈ [1e-3, 1e6]` | 1.0 (A-stable on the sampled axis; with `R(∞) = 0`, L-stable) |
| `R̂(−∞)` | −3/20 |

Why each property is wanted:

- *Single diagonal*: one matrix for all implicit stages.
- *Stiff accuracy*: the step ends on the last stage, so algebraic
  unknowns of the index-1 MNA system are consistent at `t_{n+1}` and
  `R(∞) = 0`.
- *L-stability*: the picosecond modes of open switches are damped in
  one step instead of ringing.
- *Explicit first stage*: stage order 2, which limits order reduction
  on stiff problems, and the derivative at the end of a step is the
  first stage of the next (no evaluation).
- *Embedded pair*: an error estimate with no extra solve.

- Alternatives: SDIRK4 of Hairer & Wanner (five implicit stages, stage
  order 1, stronger order reduction); ESDIRK4(3)6L[2]SA of the 2016
  NASA review and the 2019 `_C(3)` variants (later tunings of the same
  family; their coefficients were not verified here, see Open
  Questions); Radau IIA(3)/(5) (fully implicit, a 2n or 3n system);
  Rosenbrock–W methods (one factor, no Newton, but they need an exact
  Jacobian at every step or lose order, and the event machinery is
  built on converged stage solves); BDF3–5 (not A-stable, removed from
  the kernel by the `transient-timestep` spec).

### D2. Stages on the MNA system

For `d/dt q(x) + f(x, t) = 0`, stages `i = 2 … 6`:

```
ψ_i  = q(x_n) + h·Σ_{j<i} a_ij·q̇_j
(q(X_i) − ψ_i)/(γh) + f(X_i, t_n + c_i·h) = 0          solve for X_i
q̇_i = (q(X_i) − ψ_i)/(γh)
q̇_1 = −f(x_n, t_n)        (= q̇_6 of the previous step if no event intervened)
x_{n+1} = X_6
```

Newton matrix of every stage: `(1/(γh))·∂q/∂x + ∂f/∂x`. For linear
capacitors and inductors this is `C/(γh) + G = G + 2C/(h/2)`: the
trapezoidal companion matrix at `dt = h/2`, so
`cache.solve_at(mask, h/2)` provides the factor and only the right-hand
side changes from stage to stage. The stepper stores five `q̇` vectors
(the stage derivatives) in place of TR-BDF2's one.

The stage is the device interface's general stage `(β·h, ψ)` with
`β = γ`; no device implements anything for this method.

### D3. Error estimate and step control

```
e_q = h·Σ_i (b_i − b̂_i)·q̇_i                         (charge / flux units)
(C/(γh) + G)·ẽ = e_q/(γh)                              (stiff filter, one back-solve)
err = RMS over capacitor voltages and inductor currents of ẽ_k/(atol + rtol·max(|x_n,k|, |x_{n+1},k|))
accept if err ≤ 1;   h ← h·clamp(0.9·err^{−1/4}, 0.2, 5)
```

The filter is the Hosea–Shampine device that
`update-trbdf2-error-control` introduces for TR-BDF2: without it the
−3/20 of `R̂(−∞)` makes every stiff mode look like an error. The step
grid and the dt-keyed factor cache of `update-trbdf2-factor-reuse`
apply unchanged, with `h/2` as the key.

### D4. Events

- After any event the first stage is recomputed from the consistent
  post-event solve (the derivative refresh of
  `update-trbdf2-post-event-restart`); it is never carried across a
  discontinuity.
- State events: predicate signs are checked at the accepted step's end
  and on the cubic Hermite interpolant built from `(x_n, ẋ_n, x_{n+1},
  ẋ_{n+1})`, both derivatives being available. A sign change is refined
  by Illinois on the interpolant and then landed by re-taking the step
  to the located instant (the factor for the shortened `h` comes from
  the event cache), so the committed state is fourth order and only the
  bracket is third.
- Time events (gate edges, ticks, source breakpoints) cap the step, as
  for TR-BDF2.

### D5. Cost model and selection

Per step: 5 stage solves plus 1 filter solve, against 2 (plus 1 with
the filter) for TR-BDF2 — a factor 2 to 2.5. The method pays when it
takes at least that many times fewer steps, which requires steps
limited by accuracy, not by events.

- `integrator='esdirk43'`: explicit.
- `engine='auto'`: unchanged in this change. A selection rule is
  proposed for measurement, not shipped blind: over a probe window,
  count accepted TR-BDF2 steps whose size was set by the error
  controller rather than by an event cap; if more than half are, and
  the exact strategy does not apply, switch to `Esdirk43`.

## Applicability rules

Selected only on request (D5). Refused, naming the reason:

- with the `Fixed{dt}` controller in this change (a fixed-step
  fourth-order mode is possible but has no user yet);
- when the exact strategy applies and the user asked for `auto`
  (never a downgrade from exact to approximate);
- on a device kind that the conformance table does not list for the
  implicit stepper.

Documented limits rather than refusals: on index-2 unknowns (ideal
switches of `add-branch-form-ideal-switch`, voltage-source loops with
capacitors) and in the stiff limit the observed order of the algebraic
components falls towards the stage order (2–3); on event-dominated
circuits it is slower than TR-BDF2.

## Risks / Trade-offs

- A coefficient typo gives a method of lower order that still runs →
  the tableau test evaluates all order conditions in rational
  arithmetic and the convergence test measures order 4 on a smooth
  problem.
- Newton per stage: five solves that may each iterate → stage
  predictors from the previous stages; iteration counts are in the
  counter gate.
- Order reduction on stiff nonlinear problems hides in a tolerance →
  the convergence test includes a stiff Prothero–Robinson problem and
  records the observed order.
- Larger steps make event bracketing coarser → Hermite bracket plus
  landing; the event tests of TR-BDF2 are run on this integrator too.
- A third integrator to maintain → it is a tableau and a stage loop in
  the unified stepper; nothing per device.

## Migration Plan

1. Tableau constants and the rational-arithmetic property tests.
2. Linear kernel: stages on `(G, C, b)` with `solve_at(mask, h/2)`,
   embedded error and filter; closed-form tests; not exposed in
   `simulate()`.
3. After Phase 3: integrator in the unified stepper; nonlinear stages
   through `f`, `q`; events; `integrator='esdirk43'`.
4. Measurement of the selection rule on the reference set; a separate
   decision on `auto`.

Rollback: remove the option; no other code path changes.

## Open Questions

1. Tableau: the 2003 ARK4(3)6L[2]SA implicit part verified here, or one
   of the later Kennedy–Carpenter ESDIRK4(3)6L[2]SA variants (2016
   report; 2019 `_C(3)`), which should be verified the same way before
   being chosen?
2. Should `auto` ever switch integrators inside a run (D5), or should
   the choice be per run and explicit?
3. Close the remaining tasks of `add-adaptive-runge-kutta-solvers`
   (DOPRI5, Radau IIA(3)) in favour of this change and the exact
   propagator?
4. Is a fixed-step fourth-order mode wanted (for HIL-style runs with
   resonant plants)?
5. A second-derivative-free dense output of order 3 from the published
   continuous extension, instead of the Hermite interpolant, if event
   bracketing proves too coarse?
