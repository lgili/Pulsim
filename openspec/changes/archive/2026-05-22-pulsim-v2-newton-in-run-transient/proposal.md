## Why

Layer 4 V3 added `solve_with_newton` as a primitive — Newton
iteration on top of the cached linear factor — but the V3
`run_transient` loop still calls `cache.solve` directly. So
nonlinear devices (smooth-blend diodes, future MOSFET/IGBT
behavioral models) can't be used in a transient simulation:
they're only callable through the primitive in isolation.

This OpenSpec wires Newton into `run_transient`. When the user
supplies a `NonlinearRefreshFn`, every step uses Newton instead
of the direct cached solve. Newton warm-starts from the previous
step's `x` — for typical PE workloads where the state moves
slowly per step, this converges in 1-2 iterations.

## What Changes

**Scope decision — Layer 5 V4**:

- Extend `SimulationOptions`:
  - `Size max_newton_iterations = 50`
  - `Real tol_newton_dx  = 1e-9`
  - `Real tol_newton_res = 1e-9`

- Extend `run_transient` with an optional final parameter:
  ```cpp
  const pwl::NonlinearRefreshFn& nl_refresh = {}
  ```
  When `nl_refresh` is non-empty, the inner solve becomes
  `solve_with_newton(seg, refresh, graph, pool, x, ...)`
  (warm-starting from the current `x`).

- Extend `solve_with_newton` to accept an optional `b_extra`
  parameter so it composes with the trap-companion history
  terms. `nullptr` defaults to "no b_extra" (the original
  Layer 4 V3 behaviour).

- New test: **Half-wave rectifier with smooth-blend
  IdealDiode**. Same topology as the Layer 5 V2 binary-diode
  test, but using the AD-driven nonlinear model and Newton in
  the time loop. Compares the output against the binary-diode
  reference within 5 % (the smooth-blend should reproduce the
  same shape with slightly rounded edges).

## Impact

- **Affected specs**: ADDED requirements on `kernel-v2-solver`
  (Newton-in-run_transient).
- **Affected code**:
  - MODIFIED `solver/options.hpp` (~10 LOC)
  - MODIFIED `solver/run_transient.hpp` (~50 LOC for the
    Newton inner-call branch)
  - MODIFIED `pwl/nonlinear_solve.hpp` (~5 LOC for b_extra
    overload)
  - NEW `tests/v2/layer5_v4/test_integration_smooth_rectifier.cpp`
- **Migration**: zero. Default `nl_refresh = {}` makes Newton
  opt-in. All existing tests stay green.
- **Risk**: low. Newton is well-tested as a primitive (Layer 4
  V3); this OpenSpec is plumbing.
