# Design — `pulsim-v2-trust-region-newton` (Layer 4 V5)

## Levenberg-Marquardt algorithm

```
λ = λ_init   (typical 1e-6)
for iter in 0..max_iters:
    Refresh J_nl, f_nl at current x.
    J_full = J_lin + J_nl
    f_full = J_lin·x + b_total + f_nl
    baseline_norm = ||f_full||_∞

    LM inner loop:
        accepted = false
        for attempt in 0..max_lm_attempts:
            Solve (J_full + λ·I) · dx = -f_full
            x_trial = x + dx
            Refresh nl at x_trial → J_nl_trial, f_nl_trial
            f_trial = J_lin·x_trial + b_total + f_nl_trial
            if ||f_trial||_∞ < baseline_norm:
                x = x_trial
                λ *= 0.5      (shrink — closer to Newton)
                λ = max(λ, λ_min)
                accepted = true
                break
            else:
                λ *= 10      (grow — closer to gradient descent)
                if λ > λ_max:
                    throw
        if not accepted: throw

    Check convergence on ||dx||, ||f_full||.
```

Key properties:
- **λ → 0**: recovers Newton (fast quadratic convergence near
  the solution).
- **λ → ∞**: recovers gradient descent with step `1/λ` (always
  reduces residual, but slow).
- **Adaptive**: starts with small λ (let Newton work), grows
  when needed.
- **Sparse-friendly**: `J + λ·I` only adds to the diagonal —
  preserves the sparsity pattern of J.

## Implementation: adding λ·I to a sparse matrix

`Eigen::SparseMatrix` supports `+= λ·Identity` via:

```cpp
for (Index i = 0; i < J.rows(); ++i) {
    J.coeffRef(i, i) += λ;
}
```

This modifies the existing diagonal entries (or creates them if
they don't exist). Cost: O(n) for the diagonal pass + the LU
factor cost.

## When line search vs LM?

Line search (Layer 4 V4):
- Pro: super simple, zero numerical overhead per iter when
  α = 1 succeeds.
- Con: only helps "Newton step too long" cases. If the Newton
  DIRECTION is wrong, line search can't recover.

LM (Layer 4 V5):
- Pro: handles both "wrong direction" and "step too long".
- Con: extra factorisation per LM attempt (each retry costs
  one factor).

The two are complementary. Line search wraps LM cheaply: try
α=1 with the LM direction; if that doesn't reduce, try α=0.5,
... before bumping λ. V0 of LM doesn't combine them — just
ships pure LM. Future combo OpenSpec can layer them.

## Tests

### Sinusoidal smooth-blend half-wave rectifier (κ=20)

The test that failed in V4 with line search alone:

```cpp
V_sine ──[smooth-blend Diode (κ=20)]── R ── GND
```

V_amp = 10, f = 60 Hz. With LM enabled, every step converges,
and the rectifier produces the expected half-wave output:
- Positive half: V_out ≈ V_sine − V_F0 (within 1 V on > 95 %)
- Negative half: V_out ≈ 0 (within 0.5 V on > 95 %)
- Mean output power within 10 % of `(V_amp − V_F0)² / (4·R)`

### Linear regression

`enable_lm = true` on a linear-only circuit converges in 1
iteration (LM at λ ≈ 0 IS Newton; the first iter accepts the
full step which reduces the residual to numerical zero, then
λ shrinks toward 0). Result is bit-identical to the plain
Newton path.

## Why this beats trust-region (Dogleg, etc.)

Trust-region methods are more sophisticated but require
choosing a "trust radius" parameter and computing a
Cauchy-point + Newton-point blend. LM achieves comparable
robustness with a single parameter (λ) and is more directly
implementable.

For PE workloads (well-behaved most of the time, with stiff
transitions at switching events), LM is the sweet spot.
Sophisticated trust-region might be needed for chemistry or
optimisation problems but not here.
