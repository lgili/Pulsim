## Why

Layer 4 V4 added backtracking line search to globalize Newton.
Line search helps with "Newton step too long" cases but is
**not** sufficient for "Newton direction is wrong" cases — most
notably the κ=20 sinusoidal rectifier from the V4 OpenSpec,
where Newton oscillates around the steep zero-crossing region
because the full step direction itself flips between iterations.

The classical fix is **Levenberg-Marquardt damping**: instead of
solving `J · dx = -f`, solve `(J + λ·I) · dx = -f`. The damping
parameter λ shrinks (toward 0, recovering plain Newton) when
steps reduce the residual, and grows (toward ∞, recovering
gradient descent direction with small step) when they don't.
This handles both "wrong direction" AND "step too long" cases
robustly.

## What Changes

**Scope decision — Layer 4 V5** (LM damping):

- Extend `solve_with_newton_b_extra` with an optional
  `bool enable_lm` parameter (default `false`). When `true`:
  1. Maintain a running `λ` (starts at `λ_init = 1e-6`).
  2. Each iter: solve `(J + λ·I) · dx = -f`.
  3. If `||f(x + dx)||_∞ < ||f(x)||_∞`: accept, shrink `λ ×= 0.5`.
  4. Else: reject, grow `λ ×= 10`, retry (re-solve with new λ).
  5. Cap `λ ≤ 1e8` before declaring failure.

- `enable_lm = true` SUPERSEDES `enable_line_search`. When both
  are set, LM takes precedence (LM is strictly more general).

- Extend `SimulationOptions` with:
  - `bool enable_newton_lm = false`
  - (Internal: `lm_init = 1e-6`, `lm_shrink = 0.5`, `lm_grow = 10`,
    `lm_max = 1e8` — defaults baked in, not exposed.)

- `run_transient` plumbs `enable_newton_lm` through to
  `solve_with_newton_b_extra`.

- **Test**: re-enable the sinusoidal smooth-blend half-wave
  rectifier (κ=20) that failed in V4. With LM, Newton converges
  every step and the rectifier produces the expected output.

## Impact

- **Affected specs**: ADDED requirement on `kernel-v2-solver`
  (LM flag + behaviour).
- **Affected code**:
  - MODIFIED `pwl/nonlinear_solve.hpp` (~40 LOC for the LM loop)
  - MODIFIED `solver/options.hpp` (+1 field)
  - MODIFIED `solver/run_transient.hpp` (~5 LOC plumb-through)
  - NEW `tests/v2/layer5_v4/test_lm_rectifier.cpp` (~150 LOC)
- **Migration**: zero. Default `enable_newton_lm = false`
  preserves V4 behaviour bit-identically.
- **Risk**: low. LM is decades-old, well-understood, and the
  damping bound (`λ ≤ 1e8`) ensures the algorithm terminates
  even on pathological problems.
