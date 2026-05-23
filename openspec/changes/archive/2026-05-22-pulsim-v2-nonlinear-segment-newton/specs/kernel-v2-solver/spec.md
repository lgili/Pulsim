## ADDED Requirements

### Requirement: NonlinearRefreshFn — nonlinear-stamp callback

`pulsim::v2::pwl::NonlinearRefreshFn` SHALL be a callable type
that, given the current Newton iterate `x`, fills the nonlinear
Jacobian + residual deltas and returns the residual norm:

```cpp
using NonlinearRefreshFn = std::function<
    Real(const Vector& x,
         sparse::Matrix& J_nl,
         Vector& f_nl,
         const topology::Graph& graph,
         const DevicePool& pool)>;
```

The function MUST:
- Zero `J_nl` and `f_nl` before stamping.
- Walk every `BranchKind::Nonlinear` branch in `graph`.
- For each such branch, evaluate the device's current + Jacobian
  via Layer 2's `evaluate_current_and_jacobian<T>`.
- Stamp the standard 2-terminal pattern: `+i` on from-row, `-i`
  on to-row, `+G` on (from, from) and (to, to), `-G` on the
  off-diagonals.
- Return `max(|i_branch|)` across all nonlinear branches — used
  by Newton's convergence check.

#### Scenario: Refresh function returns 0 for linear-only circuit

- **GIVEN** a graph with no `Nonlinear`-kind branches
- **WHEN** `refresh(x, J_nl, f_nl, graph, pool)` is called
- **THEN** the result SHALL be `0.0`
- **AND** `J_nl` SHALL contain no non-zero entries
- **AND** `f_nl` SHALL be all zero.

### Requirement: solve_with_newton — Newton-iterated cached solve

`solve_with_newton` SHALL iterate the Newton
loop on top of a cached PwlSegment's LINEAR base:

```
f(x) = J_linear · x + b_linear + g(x)
J(x) = J_linear + ∂g/∂x

Newton:
  x_{k+1} = x_k − J(x_k)^{-1} · f(x_k)
```

Each iteration re-factors `J_linear + J_nl(x_k)` (since
`J_nl` changes with the iterate). The first iteration MAY warm-
start with the cached linear factor when `J_nl(x_0) = 0` (no
nonlinear branches).

Convergence is reached when BOTH:
- `||x_{k+1} − x_k||_∞ < tol_dx`
- `||residual||_∞ < tol_res`

If `max_iters` is exhausted without convergence, the function
SHALL throw `std::runtime_error` containing the last `||dx||`
and `||residual||` in the message.

#### Scenario: Linear-only circuit converges in 1 iteration

- **GIVEN** a circuit with no nonlinear branches and a built
  PwlSegment
- **WHEN** the user calls `solve_with_newton(seg, refresh,
  graph, pool, x_init=0)`
- **THEN** the function SHALL return after 1 Newton iteration
- **AND** the result SHALL be bit-identical to
  `cache.solve(mask, b_extra=0, x)`.

#### Scenario: DC diode load-line converges

- **GIVEN** a circuit V_dc(2V) → smooth-blend IdealDiode →
  R(1kΩ) → GND
- **WHEN** the user calls `solve_with_newton` from x_init = 0
- **THEN** the function SHALL converge in < 20 iterations
- **AND** the V_diode + I·R sum SHALL equal V_dc within tol_res.

#### Scenario: Non-convergence throws

- **GIVEN** a circuit that produces a non-converging Newton
  sequence
- **WHEN** the user calls `solve_with_newton` with
  `max_iters = 5`
- **THEN** the call SHALL throw `std::runtime_error`
  containing the last `||dx||` and `||residual||` values.
