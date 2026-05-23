# Design — `pulsim-v2-newton-globalization` (Layer 4 V4)

## Backtracking line search

At each Newton iteration:

```
1. Build J_combined and f_combined as in V3.
2. Solve J_combined · dx = -f_combined.
3. baseline_norm = ||f_combined||_inf
4. alpha = 1
5. For backtrack_iter in 0..max_backtrack:
       x_trial = x + alpha · dx
       refresh(x_trial, J_nl_trial, f_nl_trial, ...)
       f_trial = seg.J · x_trial + seg.b_constant + b_extra + f_nl_trial
       if (||f_trial||_inf < baseline_norm):
           // Accept this step.
           x = x_trial
           break
       alpha *= 0.5
6. If no alpha was accepted (residual didn't decrease at any α):
       // Accept α = 1 anyway. This is the "no progress" case —
       // plain Newton would have done the same. Iterator
       // counter still increments; we don't loop forever.
       x = x_initial + dx
```

This is intentionally simple (no Wolfe conditions, no
trust-region). For PE workloads where the residual landscape is
generally well-behaved away from sharp transitions, full Newton
steps are accepted ~always. Line search only kicks in at the
transition events.

## When line search rejects every alpha

If the smallest alpha (≈ 1/256) still doesn't reduce the
residual, the Newton direction itself is bad. Options:
1. Accept alpha = 1 anyway (V0 choice — moves forward, even if
   slightly).
2. Throw to alert the caller (might lose useful info).

V0 picks option 1 because:
- It matches plain Newton's behaviour exactly in pathological
  cases (no worse than V3).
- The outer convergence check still runs — if the residual
  doesn't go to zero, the max_iter throw catches it.
- Throwing per-iter on backtrack failure would mask the case
  where ONE bad step is followed by good progress.

## Cost analysis

Each backtrack iteration costs:
- 1 sparse matrix-vector multiply (`seg.J * x_trial`).
- 1 dense vector add (`+ b_constant + b_extra + f_nl_trial`).
- 1 refresh callback (which does an AD eval per nonlinear
  branch — typically < 10 µs for PE devices).
- 1 norm computation.

NO factor / solve — those happen once per Newton iter, not per
backtrack. So a typical "1 backtrack" step costs maybe 20-30 %
extra over plain Newton. For circuits where line search is
unnecessary (the common case), there's zero overhead because
the first `alpha = 1` trial succeeds.

## SimulationOptions extension

```cpp
struct SimulationOptions {
    // ... existing fields ...
    bool enable_newton_line_search = false;   // NEW
};
```

`run_transient` passes this through to
`solve_with_newton_b_extra`.

## API extension

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
    Real tol_res = 1e-9,
    bool enable_line_search = false);   // NEW
```

When `enable_line_search = false`, the loop runs plain Newton
(bit-identical to V3).

## Test plan

Re-enable the sinusoidal smooth-blend half-wave rectifier
(the failing V4 test). With line search:
- Positive-half: V_out tracks `max(V_sine − V_F0, 0)` within
  1 V on > 95 % of samples.
- Negative-half: V_out within 0.5 V of zero on > 95 % of samples.
- Mean output power matches `(V_amp − V_F0)² / (4·R)` within
  10 %.

If line search fixes it, we've shown the V0 globalization is
enough for PE-typical workloads.

## What V0 deliberately does NOT do

- **Wolfe conditions** (sufficient decrease + curvature). The
  V0 backtrack only checks "residual decreased". Good enough
  for non-pathological problems.
- **Trust region** methods (Dogleg, Levenberg-Marquardt). More
  robust but more complex. Future research OpenSpec.
- **Continuation methods** (homotopy from a known easy
  solution to the hard one). Useful for very stiff problems
  but out of scope here.
- **Adaptive line-search step size** (e.g., golden-section
  on alpha). Halving is simple and the residual landscape is
  not smooth enough for fancier methods to pay off.
