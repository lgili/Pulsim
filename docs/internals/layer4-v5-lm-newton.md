# Layer 4 V5 — Levenberg-Marquardt damped Newton

Plain Newton (Layer 4 V3) is fast on well-behaved problems but
can diverge or oscillate when the warm-start `x` is far from
the solution. Layer 4 V4 added backtracking line search to
shorten "too-long" steps. **Layer 4 V5 adds LM damping** as a
more general globalization: solve `(J + λ·I) · dx = -f` with
adaptive `λ` that grows on rejected steps and shrinks on
accepted ones.

## Algorithm

```
λ = 1e-6   (start near plain Newton)
for iter in 0..max_iters:
    Refresh J_nl, f_nl at x.
    J = J_lin + J_nl
    baseline_l2 = ||f||₂

    repeat (up to 30 attempts):
        Solve (J + λ·I) · dx = -f
        x_trial = x + dx
        f_trial at x_trial
        if ||f_trial||₂ ≤ baseline_l2 · (1 + 0.0001):
            accept; x = x_trial; λ *= 0.5
            break
        else:
            reject; λ *= 10
            if λ > 1e8: throw

    Check ||dx|| and ||f|| against tolerances.
```

Key properties:
- **λ → 0**: recovers plain Newton (fast quadratic convergence).
- **λ → ∞**: recovers gradient descent (slow but always
  reduces residual).
- **Adaptive**: starts near Newton, grows damping when needed.

## API

```cpp
struct SimulationOptions {
    // ... existing fields ...
    bool enable_newton_lm = false;   // NEW (default false)
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
    bool enable_line_search = false,
    bool enable_lm = false);          // NEW
```

LM takes precedence over line search when both are enabled.
Default `enable_lm = false` preserves V4 behaviour
bit-identically.

## When to use LM vs plain Newton vs line search

| Tool | Best for | Limitations |
|---|---|---|
| Plain Newton | Well-behaved nonlinear systems with good warm-start | Diverges on stiff sigmoids; oscillates at zero-crossings |
| Line search | Newton steps that are "too long" | Doesn't help when Newton direction is wrong |
| **LM** | Pathological residual landscapes; needs robust convergence | More conservative than Newton — can converge to **local** minima of ||f||₂² rather than the global zero (slower on simple problems too) |

For PE workloads:
- **DC operating point with realistic diode**: use plain
  Newton with looser tolerances.
- **Buck/boost with binary `SwitchedDiode`**: no Newton needed
  (PWL solve directly).
- **Smooth-blend `IdealDiode` (Shockley-flavor)**: try plain
  Newton first; if it diverges or oscillates, enable LM.
- **Sinusoidal source + steep sigmoid (κ=20)**: continuation
  (homotopy) needed — neither Newton nor LM converges
  cleanly. Future research OpenSpec.

## V0 limitations (documented)

LM converged to a different (local) minimum than plain Newton
on the soft-sigmoid (κ=5) DC load-line test. This is the
classical LM-vs-Newton trade-off:
- Newton blindly follows the Newton direction → may diverge.
- LM requires monotone residual decrease → may stop early at
  saddle points or local minima.

For non-convex residual landscapes (which include most smooth-
blend semiconductor models), the user must choose. V0 ships
both algorithms as opt-in tools.

The truly stiff case (κ=20 sigmoid + sinusoidal source) needs
**continuation**: start with κ=2 (easy problem), gradually
increase κ to 20 while warm-starting from the previous
solution. That's the next OpenSpec.

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
| 5 V2.2 | 20 | 46 |
| 4 V2 | 9 | 520 |
| 4 V3 | 5 | 13 |
| **5 V4 + V5** | **7** | **66** ← +3/+6 LM tests |
| **Total** | **242** | **3267** |
