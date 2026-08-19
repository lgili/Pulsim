# Numerical Solvers

Pulsim's transient kernel supports two **fixed-step** integrators
(Tustin / BDF1) baked into `pulsim.simulate(...)`, plus two
**adaptive Runge-Kutta** integrators (`DormandPrince5` and
`RadauIIA3`) usable standalone against any user-supplied
`f(t, x) → dx/dt` callback.

## Quick decision table

| Need | Use |
|---|---|
| Default SMPS / converter sim | `pulsim.simulate(...)` (Tustin fixed-step) |
| Stiff circuit (RC ≪ dt) | `pulsim.simulate(..., integrator="bdf1")` (L-stable, fixed) |
| Non-stiff ODE outside pulsim | `DormandPrince5(f=...).solve(...)` |
| Stiff ODE outside pulsim | `RadauIIA3(f=...).solve(...)` |
| Variable step inside pulsim | **shipped** — `engine='dsed'` (event-driven RK45/BDF2 auto-dispatch) for LTI-per-mask circuits; in-kernel adaptive trap for the PWL engine still planned |

## Fixed-step integrators (in-kernel)

### Tustin (trapezoidal companion) — default

- 2nd-order, A-stable, NOT L-stable.
- Cheap: 1 sparse solve per step.
- Used for the bread-and-butter PWM-switched-mode workflow.

### BDF1 (implicit Euler)

- 1st-order, L-stable.
- Used for stiff systems where Tustin would oscillate at large `dt`.
- Enabled by `opts.advanced.timestep.integrator = Bdf1`.

## Adaptive Runge-Kutta integrators (Phase 2.4, standalone)

Both integrators ship as pure-Python classes in
`python/pulsim/integrators.py`. They run against any
`f(t, x) → dx/dt` callback (numpy array in, numpy array out),
**not yet coupled to the in-kernel `pulsim.simulate` solve** —
that requires reformulating the cache to expose continuous
matrices, queued for v1.6.0.

### `DormandPrince5` — RK 5(4)

7-stage FSAL embedded pair (the "RK45" of MATLAB / SciPy fame).

- 5th-order solution + 4th-order embedded error estimate.
- FSAL → 6 effective `f` evals per step (vs 7).
- PI step controller with safety factor 0.9, growth ≤ 5×.
- Default `rtol=1e-5`, `atol=1e-8`.

```python
import pulsim as p
import numpy as np

def lorenz(t, x, sigma=10, rho=28, beta=8/3):
    return np.array([sigma*(x[1]-x[0]),
                       x[0]*(rho-x[2]) - x[1],
                       x[0]*x[1] - beta*x[2]])

solver = p.DormandPrince5(f=lorenz, rtol=1e-8, atol=1e-10)
res = solver.solve(t_span=(0.0, 10.0), x0=np.array([1.0, 1.0, 1.0]))
print(f"{res.n_accepted} accepted, {res.n_rejected} rejected, "
      f"{res.n_f_evals} f evals")
```

### `RadauIIA3` — implicit, L-stable

2-stage Radau IIA, order 3.

- **L-stable** — handles stiff ODEs that crash RK45.
- Newton inner solve on the 2n × 2n block system.
- Step-doubling error estimate (Richardson) — robust + simple.
- Optional analytical Jacobian via `jac=callable`; defaults to
  central finite differences.

## Speedup demo

`examples/scripts/run_adaptive_rk_speedup.py` — van der Pol stiff
oscillator (μ=100):

| Integrator | Steps | Wall-clock | vs Euler |
|---|---|---|---|
| Fixed-step Euler (dt=10 µs) | 49,999 | 86.3 ms | baseline |
| DormandPrince5 (rtol=1e-4) | 56 | 3.7 ms | **23× faster** |
| RadauIIA3 (rtol=1e-4) | 18 | 7.0 ms | **12× faster** |

DOPRI5 vs Radau final states agree to 1.49e-7. DOPRI5 wins
wall-clock because each step is cheap (explicit). Radau wins
step count because L-stability lets dt grow several orders of
magnitude through the stiff transitions, at the cost of a
Newton solve per step.

## Tuning guide

| Knob | DormandPrince5 | RadauIIA3 | Effect |
|---|---|---|---|
| `rtol` | 1e-5 | 1e-5 | Larger → fewer steps, lower accuracy. |
| `atol` | 1e-8 | 1e-8 | Floor on the error scale (for states near zero). |
| `dt_min` | 1e-12 | 1e-12 | Raises if controller wants smaller (typically means tolerances too tight). |
| `dt_max` | ∞ | ∞ | Cap for non-stiff regions to avoid skipping events. |
| `safety` | 0.9 | 0.9 | Multiplier on optimal step estimate. |
| `growth_max` | 5.0 | 5.0 | Max single-step growth factor. |
| `newton_max_iter` | n/a | 8 | Radau: max Newton iterations per step. |
| `newton_tol` | n/a | 1e-8 | Radau: Newton convergence threshold. |
| `jac` | n/a | None | Radau: analytical Jacobian callable, else finite diff. |

## What's NOT here (v1.5)

- **In-kernel coupling** — replace Tustin / BDF1 inside `pulsim.simulate`
  with `DormandPrince5` / `RadauIIA3`. Needs `PwlStateSpaceCache` to
  expose continuous `A, b` matrices (currently Tustin-discretized).
  Scope: 3-4 weeks kernel refactor. Queued for v1.6.0.
- **Dense-output interpolants** — needed for precise switching-event
  localisation. Both algorithms support them (Hermite for DOPRI5,
  Lagrange for Radau); not exposed in this release.
- **DOPRI8(7)** higher-order — only added on demand.
- **Rosenbrock / Rodas** semi-implicit (no Newton, one linear solve) —
  DOPRI5 + Radau cover the common cases.

## See also

- `python/pulsim/integrators.py` — source.
- `python/pulsim/adaptive.py` — coarse-grain Python-level adaptive
  driver wrapping fixed-step `simulate()` segments. Useful when
  you need adaptive stepping today without waiting for v1.6.0.
- `openspec/changes/add-adaptive-runge-kutta-solvers/` — proposal +
  design + delta spec.
