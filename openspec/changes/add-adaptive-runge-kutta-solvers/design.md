# Design — Adaptive Runge-Kutta Integrators

## Why a separate design doc

Two integrators interacting with the existing step controller, event detector, Newton refresh, and dense-output consumer is enough cross-cutting concern to warrant explicit decisions before code.

## Architectural decisions

### 1. Integrator interface

All four integrators (Tustin, BDF1, DOPRI5(4), RadauIIA(3)) implement the same internal C++ concept:

```cpp
struct IntegratorStep {
    StepResult step(Real t, Real dt, const Vector& x,
                    const StepCallbacks& cb,
                    Vector& x_new, Real& err_norm);
    DenseOutput interpolant(Real t, Real dt,
                            const Vector& x_old, const Vector& x_new);
    int order() const;
    bool is_stiff() const;
};
```

`StepResult` carries `{accepted, n_jacobian_solves, n_function_evals}` so the step controller has all the data it needs.

### 2. Step controller stays where it is

The existing `transient-timestep` step controller (LTE-based + Newton-feedback) consumes `err_norm` from any integrator. We do not re-implement the controller — we just have DOPRI5 and Radau feed it real error estimates instead of Richardson extrapolation (which becomes the fallback for Tustin / BDF1 only).

### 3. Event handling

When the existing event detector flags a switch transition inside `[t, t+dt]`:

1. Bisect on the dense-output interpolant to localise the event time `t_evt` to within 1 ns.
2. Replace the current step with a sub-step `[t, t_evt]` using the same integrator and `dt_event = t_evt − t`.
3. Apply the switch state change at `t_evt`.
4. Resume from `t_evt` with the controller's recommended dt for the new dynamics.

No integrator-specific logic at the bisection layer — the dense-output formula does all the work.

### 4. Radau Newton — Jacobian reuse

Radau IIA(3) is implicit: each step requires solving a 2s × N nonlinear system. We reuse the existing `enable_nonlinear_refresh` Jacobian path:

- Same MNA sparse Jacobian.
- Same Kelley-style trust-region globalization.
- Newton iteration count fed back into the step controller (already there for BDF1).

Failure modes (no convergence in `max_iter`) reject the step and shrink dt by the controller's "Newton-feedback" branch.

### 5. Dense output formulas

- **DOPRI5(4)**: standard Hermite-type 4th-order continuous interpolant using `(x_old, x_new, k_1, k_3, k_5, k_6)`. Cost: free — we already evaluated those stages.
- **RadauIIA(3)**: the natural 3rd-order interpolant from the Lagrange basis on the Radau nodes. Cost: free — we have the stage values.

Both stored alongside `x_new` so the event detector and `step_observer` consume them without re-solving.

### 6. Public API

User-visible knobs in `SimulationOptions`:

```python
opts = SimulationOptions(
    t_start=0.0, t_end=1.0, dt=1e-7,
    step_mode=TransientStepMode.Variable,
)
opts.advanced.timestep.integrator = Integrator.DormandPrince54
opts.advanced.timestep.rtol = 1e-5
opts.advanced.timestep.atol = 1e-8
opts.advanced.timestep.dt_min = 1e-9
opts.advanced.timestep.dt_max = 1e-3
```

Plus the shorthand:

```python
res = p.simulate(b, t_end=1.0, dt=1e-7, integrator="dopri5", step_mode="variable")
```

The default (no overrides) remains `Tustin` + fixed-step — no behavioural break for existing users.

### 7. What we are NOT doing in this change

- No DOPRI8(7) (8th-order). Future change if customers ask.
- No implicit Rosenbrock / Rodas (no Newton, just one linear solve per step). DOPRI5 + Radau cover the common cases.
- No Krylov-Newton-AMG for Radau on huge systems — small N stays cheap with the existing direct sparse solver.

## Testing strategy

- Manufactured-solution accuracy tests against scipy `solve_ivp` (treated as oracle since it implements the same methods).
- Order-of-convergence test: refine `rtol` by 10× and verify error drops by ~10⁵ for DOPRI5 / ~10³ for Radau IIA.
- Event-localization test on a half-wave rectifier zero-crossing — must land within 1 ns of the analytical zero.
- Wall-clock benchmark: PSFB at full load, variable-step DOPRI5 vs fixed-step Tustin at 10 ns. Target ≥ 5× speedup.
- Stiff regression: RC with `RC = 1 µs` simulated to `t = 10 ms`. Radau IIA should accept ~50 steps; Tustin with the same rtol would need ~10 k.

## Open questions deferred

- Should the public Python API surface a `solver = "fast"` / `"accurate"` / `"stiff"` shorthand that maps to DOPRI5 / DOPRI5+tighter-tol / Radau? Decide after first user feedback.
- Should we expose dense-output query directly to users (`res.x_at(t)`)? Likely yes in a follow-up — too large a UX redesign to bundle here.
