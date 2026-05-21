# Layer 4 V4 — Newton globalization (line search)

Plain Newton (Layer 4 V3) is fragile when the warm-start `x`
is in the "wrong region" of a stiff nonlinearity — the full
step can overshoot and make the residual worse. Layer 4 V4 adds
**backtracking line search** to globalize Newton.

## Algorithm

```
1. Compute Newton direction:  J · dx = -f
2. Try α = 1.
3. If ||f(x + α · dx)||_∞ ≥ ||f(x)||_∞:
       α *= 0.5
       repeat (up to 8 backtracks)
4. If no α reduced the residual, fall back to α = 1
   (plain Newton's behaviour at that step)
5. Update: x := x + α · dx
```

This is "Armijo-lite": only checks that the residual norm
decreases, no Wolfe conditions, no trust region. Simple,
robust, no extra parameters to tune.

## API

```cpp
struct SimulationOptions {
    // ... existing fields ...
    bool enable_newton_line_search = false;   // NEW
};

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

When the flag is `false` (default), V4 behaviour matches V3
bit-identically.

## Cost

Each backtrack iteration costs one matrix-vector multiply +
one residual norm — no factor/solve (those are once per
Newton iter, not per backtrack). For well-behaved problems,
the first `α = 1` trial succeeds and there's zero overhead.

## Verified

- **Line-search OFF reproduces V3**: same DC diode load-line
  result with both paths.
- **Line-search ON on well-behaved DC problem**: converges to
  the same operating point as plain Newton.
- All regression tests stay green.

## V0 limitations

The sinusoidal half-wave rectifier with κ=20 sigmoid (a stiff
problem at zero-crossings) **still fails** even with line
search — Newton oscillates around the steep transition. Two
realistic future paths:

1. **Trust-region** methods (Dogleg, Levenberg-Marquardt). More
   robust globalization but more complex.
2. **Continuation methods** (homotopy from κ=2 to κ=20 over
   sub-steps). The natural way to handle stiff nonlinearities.

Both are research-grade follow-up OpenSpecs.

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
| 5 V4 | **4** | **60** ← +2 / +5 globalization tests |
| **Total** | **237** | **3257** |
