# Design — `pulsim-v2-vf0-continuation` (Layer 4 V9)

## What V9 ships

Two parameter-override factories for the smooth-blend
`IdealDiode`, completing the V8 set:

| Factory | Overrides | Pool-sourced |
|---------|-----------|--------------|
| `make_kappa_override_refresh` (V8) | κ | V_F0, R_d, G_off |
| `make_vf0_override_refresh` (V9) | V_F0 | R_d, G_off, κ |
| `make_kappa_vf0_override_refresh` (V9) | κ, V_F0 | R_d, G_off |

All three return a `NonlinearRefreshFn` usable directly by
`solve_with_newton_b_extra` or `continuation_solve`.

## Why this matters

V8 left `make_kappa_override_refresh` as the only override.
That covers κ-homotopy chains AND κ-sweeps, but not V_F0
sweeps (a common parameter study) or combined homotopies.

V9's V_F0 + combined helpers complete the set:

- **Single-V_F0 override**: parameter sweeps. Evaluate the
  same circuit at multiple V_F0 values without rebuilding
  the `DevicePool`.
- **Combined κ+V_F0 override**: synchronized homotopies
  where stiffness is jointly modulated.

## Honest scope: stiff-rectifier-from-zero is NOT solved by V9

The V8 motivation was to crack the κ=20 sinusoidal rectifier
from `x = 0`. V8 ended up requiring a load-line warm-start.

Empirical exploration during V9 implementation pursued:

- **Pure V_F0 chain at κ=20**: `{−10, −5, −2, −0.5, 0, 0.3,
  0.5, 0.7}` and finer variants. All fail at intermediate
  steps via "combined matrix is numerically singular" —
  the κ=20 sigmoid causes Newton overshoot from any chain-
  step's warm-start into a basin where the Jacobian
  becomes singular.
- **Combined κ+V_F0 chain at κ_target=20**: `(2,−10) → (20,
  0.7)` with various interpolation curves. Same failure
  mode at intermediate κ values once the sigmoid sharpens.
- **κ=10 combined chain DC at V_dc=3V**: failed at step 3.
- **LM (Levenberg-Marquardt) damping**: stalls at local
  minima of ‖f‖² where the residual gradient is non-zero
  but no descent direction reduces it under the LM step
  rule.

**Conclusion**: for κ ≥ ~8 sinusoidal rectifiers from `x =
0`, no continuation chain (with the currently-available
overrides) converges at every time step. The V8 load-line
warm-start remains the recommended path for those problems.

## What V9 IS verified on

The DC diode load-line from Layer 4 V3 (V_dc=2V, R_load=1kΩ,
κ=20, V_F0=0.7). Plain Newton from `x=0` already solves
this. V9 confirms the override factories produce the
expected operating points:

```cpp
for (Real vf0 : {0.3, 0.5, 0.7}) {
    auto refresh = make_vf0_override_refresh(vf0);
    Vector x = solve_with_newton_b_extra(seg, refresh, ...);
    // x[1] ≈ V_dc − vf0 within 50 mV
}
```

Three additional unit tests verify:
- Override-vs-default residuals differ at a known x.
- The single-element continuation invariant
  (`continuation_solve({refresh}) == solve_with_newton`).
- The combined override produces a residual different
  from EACH single override (confirming both params are
  taking effect).

## Cost

The override factories run in O(num_nonlinear_branches) per
refresh — same complexity as `refresh_smooth_diodes`. They
allocate one `IdealDiode::Params` per branch per call but no
heap.

## What V0 deliberately does NOT do

- **Solve the κ=20 stiff sinusoidal rectifier from x=0**.
  (Empirically not solvable with current overrides; load-
  line warm-start is V8's tool.)
- **Auto-tune V_F0 sequences**.
- **Adaptive continuation step refinement** on inner-solver
  failure.
- **R_d / G_off overrides**. Could be added trivially; not
  needed for any motivating use case identified to date.

## Files

- **NEW** code:
  - `core/include/pulsim/v2/pwl/nonlinear_refresh_diode.hpp`
    extended with two new factories (V9.0 + V9.1).
- **NEW** test:
  - `core/tests/v2/layer5_v4/test_vf0_continuation_rectifier.cpp`
    — 4 cases.
- **NEW** docs:
  - `docs/pulsim-v2/layer4-v9-vf0-continuation.md`.
