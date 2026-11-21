## Why

A design question is usually a derivative: how much does the output
ripple move with the inductance, the efficiency with the dead time, the
junction temperature with the gate resistance. Pulsim can answer only by
running the simulation again with a parameter nudged, and on the default
fixed engine that answer is wrong.

Measured on this branch (`probes/fd_gradient_today.py`; the audit's
DCM buck from zero, `∂v_C(40T)/∂L`, reference 1758.1254 V/H from the
exact-propagator prototype, which the kernel's `dsed` reproduces to
1.75e-7):

| Engine | Relative error of a central finite difference, relative step 1e-6 … 1e-2 |
|---|---|
| `pwl`, dt = T/100 | −9.3, −9.5, −12.9, +2.0, −0.46 |
| `pwl`, dt = T/400 | +9.1, +9.1, +2.6, −0.040, +0.041 |
| `trbdf2`, default tolerances | −1.1e-4, −1.1e-4, −8.8e-5, −2.4e-3, +2.9e-3 |
| `dsed` | +1.75e-7 at 1e-6; three of the five steps abort with "D1 is chattering" |

On the fixed engine the difference is off by a factor of ten and
changes sign with the step: the diode commutation is placed on the grid
(audit `raw/numerics.md` item 3), so the result is a staircase in any
parameter that moves an event. Each gradient also costs two full runs
per parameter, and sweeps start one process per run
(`python/pulsim/sweep.py:185, 290`; an 8-job sweep is 695 ms sequential,
`raw/perf.md` row 23).

The kernel has what a direct method needs and does not use it:

- every implicit step ends with a factored Jacobian (the per-mask cached
  factor on the linear path, the converged Newton matrix otherwise);
- device laws are templated on a scalar and differentiated by forward
  AD, but only with respect to terminal voltages
  (`core/include/pulsim/models/device_model.hpp:75-90`); `Params` are
  plain `Real` (`:44-52`);
- parameters can be changed in place for four element kinds only
  (`python/bindings.cpp:1000-1040`: `update_resistor_R`,
  `update_inductor_L`, `update_capacitor_C`, `update_voltage_source_V`).

Prototypes (audit `raw/arch.md` ranked change 10 proposed the method;
these are the first measurements of it):

- **Implicit path** (`probes/forward_sensitivity_mna.py`: sine source,
  resistor, Shockley diode, RC load; trapezoidal with Newton). Forward
  sensitivities of the capacitor voltage, one extra back-substitution
  per parameter per step on the converged Jacobian, against central
  finite differences: R 3.4e-7, C and R_load below 1e-8, `I_s` 1.1e-8,
  `n·V_t` 8e-9 (relative).
- **Exact path with events**
  (`../add-periodic-steady-state-shooting/probes/shooting_dcm.py`).
  Transient `∂x(40T)/∂p` on the DCM buck against central differences of
  the same exact map: L 1.0e-8, R 2.2e-9, duty 2.4e-10. Steady-state
  gradients `dx*/dp = (I − M)⁻¹·∂P/∂p` against differences of the
  steady-state solve: L 3.4e-10, R 5.5e-9, duty 3.5e-10
  (`∂v_C*/∂L = −3.4575154e4 V/H`, `∂v_C*/∂R = 6.92255e-2 V/Ω`,
  `∂v_C*/∂D = 23.0895697 V`).
- A parameter that only moves an event instant (duty, dead time, phase
  shift) has no sensitivity at all without the jump term at the event:
  its whole effect is `(f⁻ − f⁺)·dτ/dp`.

## What Changes

- **Forward sensitivities in the transient kernel.**
  `simulate(…, sensitivities=["L1.L", "Rload.R", "pwm.duty", …])`
  propagates `S = ∂x/∂p` with the state, by differentiating the
  discrete step: one back-substitution per parameter per implicit stage
  with the factor already computed.
- **Parameters on the AD scalar.** Device `Params` become templates on
  the scalar, so `∂f/∂p` and `∂q/∂p` come from the same single
  definition of the device law as the Jacobian.
- **Jump correction at events.** At a state event `g(x, t, p) = 0`:
  `dτ/dp = −(∇gᵀS⁻ + g_p)/(∇gᵀf⁻ + g_t)` and
  `S⁺ = S⁻ + (f⁻ − f⁺)·dτ/dp` (with the reset terms when the state
  jumps); at a time event with a parameter-dependent instant, `dτ/dp`
  comes from the modulator.
- **Exact-propagator form**: `S(t+h) = Φ·S + (∂Φ/∂p)·x + ∂(Γb)/∂p`,
  with `∂Φ/∂p` from the eigenbasis already cached.
- **Steady-state gradients**: `dx*/dp = (I − M)⁻¹·∂P/∂p` with the
  monodromy matrix of `add-periodic-steady-state-shooting`.
- **Gradients of scalar KPIs**: final value, time-weighted average, RMS
  and energy of any recorded signal, over a window or over the periodic
  orbit, returned with the value (`res.kpi("vout", "avg",
  with_grad=True)`); peak and ripple with the arg-max rule and a flag
  when the extremum is not unique.
- **Use in optimisation**: `pulsim.value_and_grad(objective, params)`
  returns `(value, gradient)` in the form `scipy.optimize.minimize(…,
  jac=True)` expects. No optimiser is added.
- **Refusals**: sensitivities on a run whose events are not located in
  time (the fixed engine without event splitting), on a parameter that
  is not differentiable (an integer, a topology choice, a table
  breakpoint), or through an opaque block, are refused by name.
- No breaking change: with `sensitivities` absent the run is unchanged
  and bit-identical.

## Impact

- Affected specs: `parameter-sweep`.
- Affected code: `core/include/pulsim/ad/ad_scalar.hpp` (parameter
  seeding), `models/*.hpp` (`Params<S>`), the unified stepper
  (sensitivity recurrence beside the state), the event queue (jump
  correction), `dsed/exact_lti.hpp` (`∂Φ/∂p`), `python/bindings.cpp`,
  `python/pulsim/{__init__,sweep,kpi,stats}.py`, a new
  `python/pulsim/sensitivity.py`, docs, stubs.
- Depends on:
  - `add-unified-device-interface` (Phase 3): `f` and `q` with AD; this
    change extends the AD seed to parameters.
  - `refactor-unified-transient-stepper` (Phase 3): one place for the
    recurrence instead of three drivers.
  - `add-unified-event-queue` (Phase 3): one event path for the jump
    correction, modulators with analytic `dτ/dp`.
  - `add-fixed-step-event-splitting` (Phase 2): events located in time
    on the fixed engine.
  - `add-periodic-steady-state-shooting`: `M` and the variational
    recurrence, which is this recurrence with the initial state as the
    parameter.
  - `add-exact-propagator-strategy`: the exact form.
  - `add-immutable-circuit-ir` (Phase 3): named parameters; until then
    parameter names are `"<component>.<field>"` from `components()`.
  - `add-compiled-control-graph`: controller gains as parameters.
- Ships on today's structure (tasks 1–3): transient and steady-state
  sensitivities on the exact path for the parameters whose stamp is
  affine (R, L, C, source value) and for the duty of the C++ PWM
  schedules, in Python over the `dsed` adapter, together with slice 1 of
  the steady-state change.
- Must wait for Phase 3: Newton-device parameters, the implicit
  integrators, control gains, KPI gradients on every engine.
- Whether PLECS, PSIM or SIMBA offer parameter sensitivities was not
  checked (the audit says so too); "differentiator" is a hypothesis.
  Adjoint sensitivities are out of scope.
