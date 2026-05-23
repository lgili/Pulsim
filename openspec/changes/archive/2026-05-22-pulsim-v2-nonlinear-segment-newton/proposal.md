## Why

Layers 4-5 V0-V3 handle every PIECEWISE-LINEAR PE circuit:
R, V, C, L, switches, ideal diodes (binary on/off). The
`BranchKind::Nonlinear` branches (smooth-blend `IdealDiode`,
future MOSFET/IGBT behavioral models, etc.) are **silently
SKIPPED** at assembly. That means circuits with real (Shockley-
exponential) diodes can't be simulated by the v2 kernel — only
their idealized binary cousins.

Real PE designs sometimes need:
- **Real diode reverse recovery** — captured via the smooth I-V
  curve, not a binary state machine.
- **MOSFET I_D(V_GS, V_DS) characteristics** — quadratic and
  saturation regions, not just an on/off switch.
- **Active-region BJT/MOSFET models** — for linear amplifier or
  signal-conditioning blocks within a PE design.
- **Continuous IGBT models** — the more accurate behavioral
  variants (PSIM "level 2", PLECS "Detailed switch model").

This OpenSpec adds **Newton iteration on top of the cached
linear factor** to bring nonlinear devices into the v2 fold.

## What Changes

**Scope decision — Layer 4 V3** (the nonlinear-Newton
capability):

- After assembling the per-segment LINEAR matrix as Layer 4 V1
  does, the cache ALSO records a "nonlinear refresh function"
  per segment: a callable that, given the current state `x`,
  computes the Jacobian contribution and residual contribution
  from every `BranchKind::Nonlinear` branch.
- The `cache.solve` hot path becomes:
  ```
  while not converged AND iter < N_max:
      J_nl, f_nl = refresh_nonlinear(x)
      J_combined = J_linear + J_nl
      f_combined = f_linear + f_nl
      solve J_combined · dx = -f_combined
      x += dx
      check ||dx|| < tol
      ++iter
  ```
- We **re-factor** the matrix per Newton iteration when nonlinear
  devices are present (the J changes). The "PWL cache" cost
  advantage is preserved for the LINEAR part of each segment;
  nonlinear circuits pay the factorization cost per Newton
  iteration, which is still better than v1's per-step refactor
  (because we still amortize across many steps within ONE
  switch state).
- **AD-driven Jacobian** — Layer 2's `evaluate_current_and_jacobian<T>`
  already does this. The Newton refresh just stamps the
  Layer 2 output into J_nl.

**Test cases**:

1. **DC diode + R**: V_dc(2V) → IdealDiode(Shockley) →
   R(1kΩ) → GND. Verify the DC operating point matches the
   transcendental equation `V_dc = V_diode(I) + I·R` solved
   numerically. This is the canonical "diode load-line" test.

2. **Half-wave rectifier with real diode**: same topology as
   Layer 5 V2's SwitchedDiode test but using the smooth-blend
   `IdealDiode` instead. The output power should match the
   analytical formula adjusted for the diode's V_F drop.

## Impact

- **Affected specs**: ADDED capability `kernel-v2-newton` (new
  delta spec).
- **Affected code** (estimated):
  - NEW `pwl/nonlinear_refresh.hpp` (~150 LOC) — the Newton-
    iteration helper.
  - MODIFIED `pwl/cache.hpp` (~50 LOC) — `solve_with_newton(...)`
    overload.
  - NEW `tests/v2/layer4_v3/*` (~250 LOC across 3 files).
- **Migration**: zero. Existing tests use `cache.solve(...)`
  which doesn't iterate Newton (no nonlinear branches in those
  circuits).
- **Risk**: medium. Newton can diverge. We ship:
  - Max-iteration safety (`max_newton_iterations`, default 50).
  - Norm-based convergence (`||dx||₂ < tol`, default `1e-9`).
  - A throw on non-convergence with diagnostic info.
  - The first Newton iteration uses the LINEAR-only cached
    factor — that's a "warm start" if the nonlinear contribution
    is small.

## What this proposal does NOT do

- **No `run_transient` integration** in V0. The Newton loop is
  exposed as `cache.solve_with_newton(...)`, and the caller
  decides when to use it. Updating Layer 5 V3's run_transient
  to auto-detect nonlinear branches and use Newton is a V0.5
  one-line change but I'm keeping it scope-separated for
  reviewability.
- **No Sherman-Morrison rank-1 updates** between Newton
  iterations. Each iter does a full refactor. Sherman-Morrison
  is a future performance OpenSpec.
- **No globalization** (line search, damping, trust regions).
  Pure Newton. Sufficient for well-conditioned PE workloads;
  globalization is a future OpenSpec.
