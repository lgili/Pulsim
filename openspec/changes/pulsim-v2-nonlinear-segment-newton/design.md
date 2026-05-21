# Design — `pulsim-v2-nonlinear-segment-newton` (Layer 4 V3)

## The Newton loop on the cached linear factor

For a circuit with both LINEAR (R, V, Switch, C with companion,
L with companion, Diode binary) and NONLINEAR (smooth-blend
IdealDiode, real MOSFET, etc.) devices, the MNA residual is:

```
f(x) = J_lin · x + b_lin + g(x)
```

where `J_lin` and `b_lin` are the cached linear contributions
(constant for a given switch state) and `g(x)` is the
nonlinear contribution from all `BranchKind::Nonlinear` branches.

Newton iteration:

```
x_{k+1} = x_k − [J_lin + ∂g/∂x|_{x_k}]^{−1} · [J_lin · x_k + b_lin + g(x_k)]
```

The matrix `J_lin + ∂g/∂x` changes each iteration (because g
depends on x). We re-factor it each iteration.

**Why this still beats v1**: in v1, Newton refactors the FULL
Jacobian per step. In v2, Newton refactors per Newton iteration
WITHIN a step, AND many simulation steps re-use the same switch
state (so the LINEAR base is identical across them). Each
nonlinear-segment Newton iter costs one factor; v1 would have
done the same factor + the linear-base assembly, which is much
slower.

## API

```cpp
namespace pulsim::v2::pwl {

/// Refresh function: given the current state `x`, fills J_nl
/// (sparse delta to be ADDED to the cached J_lin) and f_nl
/// (dense delta to be ADDED to the cached b_lin).
///
/// Returns the SUM(|g_i|) norm of the residual contributions,
/// for convergence diagnostics.
using NonlinearRefreshFn = std::function<
    Real(const Vector& x,
         sparse::Matrix& J_nl,
         Vector& f_nl,
         const topology::Graph& graph,
         const DevicePool& pool)>;

/// Solve `cache_segment.J · x + cache_segment.b + g(x) = 0`
/// using Newton iteration. Each iter re-factors J_lin + J_nl(x_k).
///
/// Convergence: ||x_{k+1} − x_k||_∞ < tol AND
///              ||residual||_∞ < tol_residual.
///
/// Throws std::runtime_error if max iters exhausted.
[[nodiscard]] Vector solve_with_newton(
    const PwlSegment& seg,
    const NonlinearRefreshFn& refresh,
    const topology::Graph& graph,
    const DevicePool& pool,
    const Vector& x_init,
    Size max_iters = 50,
    Real tol_dx = 1e-9,
    Real tol_res = 1e-9);

}  // namespace pulsim::v2::pwl
```

## The refresh function

For each nonlinear branch, we use Layer 2's
`evaluate_current_and_jacobian<T>` (which uses AD internally) to
get `i_branch` and the partial derivatives:

```cpp
Real refresh_smooth_diode(const Vector& x,
                           sparse::Matrix& J_nl, Vector& f_nl,
                           const Graph& graph,
                           const DevicePool& pool) {
    Real total_residual = 0;
    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        const auto& branch = graph.branch(b_id);
        if (branch.kind != BranchKind::Nonlinear) continue;
        // Look up the device, evaluate i + partials via AD.
        // Stamp the standard 2-terminal pattern (like
        // stamp_device).
        ...
    }
    return total_residual;
}
```

For V0, we ship ONE built-in refresh function — for the smooth-
blend `IdealDiode`. The user can supply their own via the
`NonlinearRefreshFn` typedef.

## Convergence criteria

Newton converges when BOTH:
- `||dx||_∞ < tol_dx`
- `||residual||_∞ < tol_res`

For PE workloads at double precision, `1e-9` is comfortable.
The user can tighten or loosen.

If max_iters is hit without converging, the function throws
`std::runtime_error` with the last `||dx||` and `||residual||`
in the message.

## Why no run_transient integration in V0

The Newton-aware `run_transient` is a one-line addition (call
`solve_with_newton` instead of `cache.solve` when nonlinear
branches are present), but it has subtler design questions:
- Does the user supply the refresh function or does
  `run_transient` build one automatically by walking the pool?
- How do we report Newton stats (per-step iteration count)?
- Should the event-iteration loop wrap Newton or vice versa?

Those questions deserve their own OpenSpec. V0 ships
`solve_with_newton` as a primitive and lets the user (or a
future spec) wire it up.

## Test coverage

1. **DC diode load line**: V_dc(2 V) → Diode → R(1 kΩ) → GND.
   At DC, `V_dc = V_diode(I) + I·R`. With Shockley model
   `I = I_s · (exp(V_diode/V_T) − 1)`, this transcendental
   equation has a known solution (look up "diode load-line"
   in any electronics textbook). Verify Newton converges in
   < 20 iterations and the answer matches within 1 % of the
   numerical-textbook answer.

2. **DC convergence on a circuit with NO nonlinear branches**:
   Newton converges in exactly 1 iteration (since g(x) = 0 and
   J_nl = 0, the first Newton step IS the linear solve).

3. **DC singularity handling**: a circuit with two parallel
   diodes (degenerate) should either converge or throw cleanly.

## What V0 deliberately does NOT include

- Sherman-Morrison rank-1 updates between Newton iterations.
  (V1 perf optimization.)
- Damped Newton / line search. (Globalization OpenSpec.)
- run_transient integration. (V0.5 follow-up.)
- Sub-Newton iteration policies (e.g., "only do Newton when
  V_diode > V_th"). (V1 efficiency.)
