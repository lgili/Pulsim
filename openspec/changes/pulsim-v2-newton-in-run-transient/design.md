# Design — `pulsim-v2-newton-in-run-transient` (Layer 5 V4)

## API change

```cpp
struct SimulationOptions {
    Real t_start = 0;
    Real t_end   = 0;
    Real dt      = 0;
    Size max_event_iterations = 16;
    // NEW (Layer 5 V4)
    Size max_newton_iterations = 50;
    Real tol_newton_dx  = 1e-9;
    Real tol_newton_res = 1e-9;
};

inline SimulationResult run_transient(
    const PwlStateSpaceCache& cache,
    const Graph& graph,
    const DevicePool& pool,
    const SimulationOptions& opts,
    const SwitchScheduleFn& switch_fn,
    const BExtraFn& b_extra_fn = {},
    bool start_from_dc_op = false,
    const pwl::NonlinearRefreshFn& nl_refresh = {});   // NEW
```

When `nl_refresh` is empty (default), the loop runs Layer 5 V3
behaviour exactly (cache.solve). When non-empty, each step's
inner solve becomes Newton-iterated.

## Composition with event iteration

```
for each step:
  do (event-iter):
    mask = combine(switch_fn(t), diode_mask, diode_owned)
    if (nl_refresh):
        x = solve_with_newton(seg[mask], nl_refresh, graph, pool,
                               x_warmstart=x, b_extra, ...)
    else:
        cache.solve(mask, b_extra, x)
    flipped = diodes.update_from_state(x)
  while (flipped && iter < max_event)
```

Each event iteration runs Newton to convergence on the
continuous state, THEN checks if the switch state should flip.
This is the right composition order — we don't waste Newton
iters on a soon-to-be-stale mask.

## Warm starting

Newton's initial `x` is whatever was the previous step's
solution. For a circuit close to steady state, this is a
near-converged guess and Newton finishes in 1-2 iterations.

For the first step (or just after a DC-OP seed), `x` is either
all-zero or the DC operating point — both reasonable warm
starts for typical PE workloads.

## b_extra plumbing

`solve_with_newton` currently does:

```cpp
Vector f_combined = seg.J * x + seg.b_constant + f_nl;
```

The MNA residual at convergence is `J·x + b_total + g(x) = 0`
where `b_total = b_constant + b_extra`. So we extend to:

```cpp
Vector f_combined = seg.J * x + seg.b_constant + b_extra + f_nl;
```

We add a new overload that takes `b_extra` explicitly. The
original signature (no b_extra) routes through the overload
with a zero vector.

## Tests

### Smooth-blend half-wave rectifier

Same topology as Layer 5 V2's binary-diode test, but using the
AD-driven smooth-blend `IdealDiode` instead:

```cpp
V_sine ──[smooth-blend Diode]── R ── GND
```

V_amp = 10, f = 60 Hz, R = 10. Simulate 2 cycles at dt = 100 µs.

Validation:
- Positive half-cycle: V_out tracks V_sine minus V_F0 (≈ 0.7 V
  drop). Within 1 V of `max(V_sine − V_F0, 0)`.
- Negative half-cycle: V_out ≈ 0.
- Mean output power within 10 % of the analytical half-wave
  rectifier formula adjusted for V_F0:
  `P = (V_amp − V_F0)² / (4 · R)` ≈ 2.16 W for our values.

## Risks

1. **Newton non-convergence on transients**. For stiff
   transitions (e.g., the diode going from deep OFF to deep
   ON in one step), the sigmoid model can be hard for plain
   Newton. Mitigation: large `max_newton_iterations = 50` by
   default; warm start from previous x.
2. **Eigen sparse matrix add operations**. `J_lin + J_nl` in
   each Newton iter does a sparse-pattern union — not the
   fastest path. For dense circuits (where J_nl pattern
   matches J_lin), this is cheap. For very large N, we'd want
   to share the sparsity pattern. V0 doesn't optimise; future
   perf OpenSpec might.
