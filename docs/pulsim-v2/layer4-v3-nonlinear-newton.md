# Layer 4 V3 — Newton iteration on the cached linear factor

Layers 4 V0-V2 cached `J_lin` and `b_lin` per switch state and
solved `J_lin · x = -b_lin` in one triangular pass. That works
for PIECEWISE-LINEAR circuits but skips `BranchKind::Nonlinear`
branches entirely — the smooth-blend `IdealDiode` and other
nonlinear models in Layer 2 had nowhere to plug in.

**Layer 4 V3 adds Newton iteration** on top of the cached
linear factor:

```
f(x) = J_lin · x + b_lin + g(x)
J(x) = J_lin + ∂g/∂x

Newton:
  J(x_k) · dx = -f(x_k)
  x_{k+1} = x_k + dx
```

Each iteration re-factors `J_lin + J_nl(x_k)` (since `J_nl`
changes with the iterate). The CACHED linear base is still
amortized across many simulation steps in the same switch
state — only the small nonlinear delta is recomputed per
iteration.

## API

```cpp
namespace pulsim::v2::pwl {

/// Callback that, given the current Newton iterate, fills the
/// nonlinear Jacobian + residual deltas.
using NonlinearRefreshFn = std::function<
    Real(const Vector& x,
         sparse::Matrix& J_nl,
         Vector& f_nl,
         const topology::Graph& graph,
         const DevicePool& pool)>;

/// Built-in refresh for the smooth-blend `IdealDiode`.
Real refresh_smooth_diodes(...);

/// No-op refresh — converges in 1 iteration on linear circuits.
Real noop_refresh(...);

/// Newton-iterated solve. Throws on non-convergence.
Vector solve_with_newton(
    const PwlSegment& seg,
    const NonlinearRefreshFn& refresh,
    const topology::Graph& graph,
    const DevicePool& pool,
    const Vector& x_init,
    Size max_iters = 50,
    Real tol_dx  = 1e-9,
    Real tol_res = 1e-9);

}  // namespace pulsim::v2::pwl
```

`DevicePool` adds `add_nonlinear_diode(branch_id, IdealDiode::Params)`
to register the smooth-blend diode on a `BranchKind::Nonlinear`
branch.

## Verified

- **Linear-only circuit + `noop_refresh`**: converges in
  1 iteration, result identical to `cache.solve`.
- **V-R divider + Newton**: bit-identical to `cache.solve`.
- **DC diode load-line** (V_dc = 2 V, smooth-blend diode,
  R = 1 kΩ): Newton converges, V_diode ≈ V_F0 + I·R_d, current
  matches `(V_dc − V_F0)/(R + R_d) ≈ 1.3 mA` within 5 %.
- **Reverse-biased diode**: source constraint holds, v_n1 is
  bounded (no numerical blow-up).
- **Non-convergence**: oscillating refresh + tight tolerances
  throws `std::runtime_error` after `max_iters`.

## V0 limitations

- **Per-iteration refactor**. Each Newton iter does a full
  factorization of `J_lin + J_nl`. Sherman-Morrison rank-1
  updates between iterations are a future perf OpenSpec.
- **No globalization**. Pure Newton — no line search, no
  damping, no trust region. Works well for PE-typical
  problems; fragile for very stiff nonlinearities.
- **Not yet integrated with `run_transient`**. The Newton
  primitive is exposed as `solve_with_newton(...)`; wiring it
  into the V3 `run_transient` loop is a V0.5 follow-up
  (one-line change + careful test design around the
  event-iteration interplay).

## Why this still beats v1

v1's Newton refactors the FULL Jacobian per step. v2 V3:
- Refactors `J_lin + J_nl` per Newton iteration WITHIN a step.
- But `J_lin` is identical across MANY simulation steps in the
  same switch state.
- AND the assembly of `J_lin` (the expensive part) is done
  ONCE per switch state at cache build time.

So Newton circuits pay the factorization cost per iter (as in
v1), but the assembly cost amortizes across thousands of steps
(unlike v1). On steady-state PE workloads where the switch
state changes rarely, this is a significant net win.

## Status

| Layer | Cases | Assertions |
|---|---|---|
| 0 | 19 | 80 |
| 1 | 36 | 126 |
| 2 | 36 | 93 |
| 3 | 16 | 61 |
| 4 V0 | 24 | 58 |
| 5 V0 | 21 | 2069 |
| 4 V1 | 32 | 76 |
| 5 V1 | 17 | 59 |
| 5 V2.1 | 18 | 42 |
| 4 V2 | 9 | 520 |
| **4 V3** | **5** | **13** ← NEW |
| **Total** | **233** | **3197** |

v1 regression: 304 / 4214, all green.
