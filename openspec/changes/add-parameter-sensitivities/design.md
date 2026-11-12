## Context

The transient kernel solves `d/dt q(x, p) + f(x, t, p) = 0` (the form
of `add-unified-device-interface`) with implicit one-step methods whose
stage matrix is factored once per (mask, step) or once per Newton
iteration. Device laws are single templated functions on a scalar, with
Jacobians by forward AD (`ad::ADRealN<N>`, `ad/ad_scalar.hpp`). Events
are located in time on TR-BDF2 and on the exact path; on the fixed
engine they will be once `add-fixed-step-event-splitting` lands.

Constraints: no silent wrong answers — a sensitivity that cannot be
computed correctly is refused; a run without `sensitivities` stays
bit-identical; owner note *newton-devices-need-smooth-laws-to-1e-12*
(device laws must be smooth for Newton, which is also what makes them
differentiable in `p`).

## Goals / Non-Goals

- Goals: `∂x/∂p` along a transient at the cost of one back-substitution
  per parameter per stage; correct across switching events; gradients
  of the periodic steady state; gradients of scalar KPIs; an interface
  an optimiser can call.
- Non-Goals: adjoint (reverse) sensitivities; second derivatives;
  sensitivities with respect to table breakpoints, topology or integer
  parameters; an optimiser; differentiating the adaptive step-size
  sequence; GPU batching.

## Decisions

### D1. Differentiate the discrete step (staggered direct method)

One implicit stage solves, for the stage value `X`,

```
R(X; ψ, p) = (q(X, p) − ψ)/(β·h) + f(X, t, p) = 0
J = ∂R/∂X = (1/(β·h))·∂q/∂x + ∂f/∂x          (the factored stage matrix)
```

where `ψ` is the method's known combination of earlier values of `q`
and `q̇`, and `β·h` its diagonal coefficient (trapezoidal `h/2`,
backward Euler `h`, TR-BDF2 `γh/2`, ESDIRK `γh`). Differentiating with
respect to one parameter:

```
J·S_X = (1/(β·h))·(dψ/dp − ∂q/∂p) − ∂f/∂p ,     S_X = dX/dp
```

- `dψ/dp` is the same combination applied to the stored
  `d q/dp = (∂q/∂x)·S + ∂q/∂p` of earlier stages, so the stepper keeps
  one extra vector per parameter per history slot.
- `∂f/∂p`, `∂q/∂p` are evaluated at the converged `X` with the
  parameter seeded on the AD scalar (D2); for a linear element they are
  the element's stamp differentiated, four entries.
- The solve reuses the factor of `J`: after a Newton solve, the factor
  of the last iteration when the iteration converged with a fresh
  Jacobian; on the linear path the cached per-mask factor. One forward
  and one back substitution per parameter per stage.

This is the derivative of the numerical solution for the step sequence
actually taken (Caracotsios & Stewart 1985; the staggered corrector of
Maly & Petzold 1996, as in CVODES/IDAS). With a modified-Newton factor
(a stale `J`), the right-hand side is still exact but the solve is not:
the stepper then iterates the sensitivity solve with the same factor
until its residual meets the Newton tolerance.

- Alternatives: finite differences of whole runs — measured wrong by a
  factor of ten on the fixed engine and `2·n_p` runs; differentiate the
  continuous equations and integrate them with their own error control
  — a second integrator and results that differ from the difference of
  two runs; reverse-mode AD of the step loop — needs the whole
  trajectory stored and a backward pass through events.

### D2. Parameters on the AD scalar

`template <class S> struct Params` for each device: every `Real` field
that is a physical parameter becomes `S`. The device is evaluated twice
per converged stage when sensitivities are requested: once as today
(seeds on terminal voltages, for `J`), once with `S = ADRealN<n_local>`
seeded on the requested parameters of that device (for `∂f/∂p`,
`∂q/∂p`). Devices with no requested parameter are not re-evaluated.
`Params<Real>` is layout-compatible with today's `Params`, so the
default path does not change.

Table-based laws expose the derivative with respect to a scale
parameter (for example a `Coss` multiplier), not to individual
breakpoints.

### D3. Jump correction at events

State event `g(x, t, p) = 0` at `τ(p)`, with a reset `x⁺ = R(x⁻, t, p)`
(identity for today's switches):

```
dτ/dp = −(∇gᵀ·S⁻ + g_p) / (∇gᵀ·f⁻ + g_t)
S⁺ = R_x·S⁻ + R_p + (R_x·f⁻ + R_t − f⁺)·dτ/dp
continuous state:   S⁺ = S⁻ + (f⁻ − f⁺)·dτ/dp
```

(Rozenvasser 1967; Hiskens & Pai, IEEE TCAS-I 2000.) `f⁻`, `f⁺` are the
time derivatives of the differential variables just before and after
the event. The stepper already has both: `f⁻` from the last stage
before the event and `f⁺` from the consistent post-event solve that
`update-trbdf2-post-event-restart` introduces. The sensitivities of the
algebraic unknowns after the event follow from one solve with the
post-event stage matrix.

Time event whose instant depends on a parameter (PWM duty `d`, dead
time, phase, period): `dτ/dp` is analytic from the modulator
(`dτ/dd = T` for a trailing-edge modulator) and the same formula
applies. `∇gᵀ·f⁻ + g_t = 0` (grazing) makes the sensitivity undefined;
the run stops with an error naming the event.

For a law that is continuous at the event (today's resistor-pair diode
with `V_th = 0`) `f⁻ = f⁺` to rounding and the correction is zero; it
is non-zero for a gate edge, a hysteresis band and the ideal switch of
`add-branch-form-ideal-switch`, and it is the entire sensitivity for a
timing parameter.

### D4. Exact-propagator form

```
S(t+h) = Φ·S(t) + (∂Φ/∂p)·x(t) + ∂(Γ₀·b)/∂p
∂Φ/∂p = V·(F ∘ (V⁻¹·(∂A/∂p)·V))·V⁻¹                 (Daleckii–Krein)
F_ij = h·e^{λ_j h}·φ₁((λ_i − λ_j)·h)                   (F_ii = h·e^{λ_i h})
```

with `∂(Γ₀b)/∂p` from the same formula on the augmented matrix
`[[A, b],[0, 0]]`. Writing the divided difference through φ₁ keeps it
accurate for close eigenvalues. `∂A/∂p`, `∂b/∂p` are obtained by
differentiating the extraction (`A = −M⁻¹·(G_ss − G_sa·G_aa⁻¹·G_as)`)
with the sparse `∂G/∂p`, `∂C/∂p`, `∂b/∂p` of the element; for a stamp
that is affine in the parameter (`1/R`, `C`, `L`, a source value) the
derivative of the stamp is exact from the difference of two assemblies.

### D5. Steady-state gradients

`x*(p)` solves `P(x*, p) = x*`. With the monodromy matrix `M` of the
steady-state solver:

```
(I − M)·dx*/dp = ∂P/∂p |_{x*}
```

`∂P/∂p` is `S(T)` of one period started at `x*` with `S(0) = 0`. A KPI
over the orbit, `K = (1/T)·∫ k(x, p) dt`, needs one more sensitivity
period with `S(0) = dx*/dp`:

```
dK/dp = (1/T)·∫ (k_x·S + k_p) dt + (1/T)·Σ_events (k⁻ − k⁺)·dτ/dp
```

### D6. KPIs

| KPI | Gradient |
|---|---|
| final value `x_i(t_end)` | row of `S(t_end)` |
| time-weighted average, energy | quadrature of `k_x·S + k_p` on the run's own grid, plus the event terms of D5 |
| RMS | `d(rms)/dp = (1/(rms·T))·∫ y·(dy/dp) dt` |
| peak, ripple | `dy/dp` at the arg-max (arg-min); flagged `non_unique` when two extrema are within the tolerance |

Quadratures use the time-weighted rules already in `stats.py`
(owner note *time-weight-averages-on-adaptive-grids*).

### D7. Interface

```python
res = pulsim.simulate(b, t_end=…, sensitivities=["L1.L", "Rload.R", "pwm.duty"])
res.sens("vout", "L1.L")            # array, same length as res.times
res.sensitivity.states              # (n_samples, n_state, n_p), zero-copy
value, grad = res.kpi("vout", "avg", t_from=…, with_grad=True)   # grad: dict by parameter

ss = pulsim.periodic_steady_state(b, period=T, sensitivities=[…])
ss.dx0["L1.L"]; ss.kpi("vout", "ripple", with_grad=True)

f = pulsim.value_and_grad(objective, params=["L1.L", "C1.C"])   # f(p_vector) -> (value, grad)
```

Parameter names are `"<component>.<field>"` (the fields of
`components()`), `"<modulator>.<field>"`, `"<block>.<field>"`; a name
that does not resolve, or resolves to a non-differentiable field, is
refused. The step-size controller ignores `S` by default;
`sensitivity_error_control=True` adds it to the error norm.

## Applicability rules

Computed:

- any parameter that enters a device's `f` or `q`, a source value, a
  modulator's timing, a control-graph block parameter, an initial
  condition;
- on TR-BDF2, ESDIRK, the exact strategy, and the fixed engine with
  event splitting.

Refused, naming the parameter or the feature:

- the fixed engine when an event is not located in time (legacy mode
  without event splitting): the solution is a staircase in the
  parameter, measured −9.3 … +2.0 relative error;
- a parameter that changes the sparsity pattern or the topology, an
  integer, a table breakpoint;
- a circuit with a `PythonBlock`, an opaque `switch_fn` or a
  `b_extra_fn` between the parameter and the output;
- a grazing event (D3);
- more than 64 parameters in one run (forward mode costs one
  back-substitution each; the limit is a guard, not a design point).

## Risks / Trade-offs

- A device evaluated on the parameter-seeded scalar may hide a
  non-smooth branch (`abs`, a clamp) → the forward value is checked
  against a central finite difference of the device law in the unit
  tests of every device, to 1e-8.
- Stale Newton factor → the sensitivity solve is iterated to the Newton
  tolerance (D1); its iteration count is a counter.
- Memory: `n_samples × n_state × n_p` doubles → sensitivities are
  recorded only for requested probes unless `sensitivity.states` is
  asked for.
- The gradient is that of the discrete solution, which differs from the
  continuous one by the truncation error of the method → documented;
  the tests compare with finite differences of the same discrete
  scheme on engines where that is meaningful, and with closed forms
  elsewhere.
- Peak and ripple are not differentiable where two extrema tie → the
  `non_unique` flag.

## Migration Plan

1. Reference tests from the prototypes; finite-difference harness.
2. Exact path, affine-stamp parameters and duty, Python over the `dsed`
   adapter: transient `S`, `∂P/∂p`, steady-state gradients.
3. `Params<S>` and parameter seeding in the AD layer (no behaviour
   change; a compile-time and a bit-identity test).
4. After Phase 3: the recurrence of D1 in the unified stepper; the jump
   correction in the event queue.
5. KPI gradients; `value_and_grad`; one worked optimisation example.
6. Controller gains (after the control graph).

Rollback: every step is additive; `sensitivities=None` is the previous
behaviour.

## Open Questions

1. Is forward mode with a 64-parameter guard the right first scope, or
   is a use case with hundreds of parameters (and few outputs) already
   in view, which would argue for planning the adjoint now?
2. Parameter naming before `add-immutable-circuit-ir`:
   `"<component>.<field>"` strings (proposed), or wait for named
   parameters?
3. Should sensitivities enter the step-size error norm by default?
   (It costs steps; proposed default: no.)
4. Temperature as a parameter (through the electrothermal coupling): in
   this change or with `add-in-kernel-event-losses`?
5. Is a worked optimisation example enough, or is a thin
   `pulsim.optimize` wrapper over scipy wanted?
