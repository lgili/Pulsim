# Layer 5 V4 — Newton wired into run_transient

Layer 4 V3 added `solve_with_newton` as a primitive. Layer 5 V4
wires it into `run_transient` so transient simulations with
nonlinear devices (smooth-blend diodes, future MOSFETs/IGBTs)
work end-to-end.

## API change

```cpp
struct SimulationOptions {
    // ... existing fields ...
    Size max_newton_iterations = 50;   // NEW
    Real tol_newton_dx  = 1e-9;        // NEW
    Real tol_newton_res = 1e-9;        // NEW
};

SimulationResult run_transient(
    const PwlStateSpaceCache& cache,
    const Graph& graph,
    const DevicePool& pool,
    const SimulationOptions& opts,
    const SwitchScheduleFn& switch_fn,
    const BExtraFn& b_extra_fn = {},
    bool start_from_dc_op = false,
    const NonlinearRefreshFn& nl_refresh = {});  // NEW
```

When `nl_refresh` is empty (default), `cache.solve` is used as
before. When non-empty, each step's inner solve becomes
`solve_with_newton_b_extra(...)` with the previous step's `x`
as warm start.

## solve_with_newton_b_extra

The Newton primitive now has a `b_extra` overload that composes
with the trap-companion history terms:

```cpp
Vector solve_with_newton_b_extra(
    const PwlSegment& seg,
    const NonlinearRefreshFn& refresh,
    const Graph& graph,
    const DevicePool& pool,
    const Vector& x_init,
    const Vector& b_extra,
    Size max_iters = 50,
    Real tol_dx  = 1e-9,
    Real tol_res = 1e-9);
```

The residual is `J_lin·x + (b_constant + b_extra) + g(x)` —
identical to the Layer 5 V1+ trap-companion convention.

## Verified

- **Linear regression**: `run_transient` with `nl_refresh =
  &noop_refresh` produces bit-identical results to the
  `cache.solve` path (all v0-v3 behaviour preserved).
- **DC diode load-line via run_transient**: V_dc(2V) →
  smooth-blend diode → R(1kΩ). At every recorded sample (after
  warm-up), the current matches the analytical
  `(V_dc - V_F0)/(R + R_d) ≈ 1.3 mA` within 5 %, and the source
  constraint v_n0 = V_dc holds exactly.

## V0 limitations (documented for the next OpenSpec)

- **Stiff nonlinearities + sinusoidal source**: a half-wave
  rectifier with a sinusoidal source stresses Newton's warm
  start at the zero-crossings (the sigmoid model with
  `kappa = 20` has steep slope near `V_F0`). Plain Newton can
  oscillate or converge to the wrong branch when the warm-x is
  in a "wrong region" relative to the new operating point.
  Mitigation: damped Newton with backtracking line search —
  the `pulsim-v2-newton-globalization` follow-up.
- **No event-iteration interaction tests**. Circuits combining
  SwitchedDiodes + smooth-blend diodes haven't been verified
  in V0. The natural composition is `event-iter wraps Newton`;
  worth a dedicated regression test in V1.

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
| 4 V3 | 5 | 13 |
| **5 V4** | **2** | **55** ← NEW |
| **Total** | **235** | **3252** |
