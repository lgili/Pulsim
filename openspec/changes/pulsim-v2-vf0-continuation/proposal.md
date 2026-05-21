## Why

V8 shipped `continuation_solve` + `make_kappa_override_refresh`.
The honest V8 finding: κ-only continuation does NOT solve the
κ=20 stiff sinusoidal rectifier from a naive warm-start. The
load-line warm-start is what made V8 pass.

This leaves a gap: users who want to **vary a single diode
parameter** in a homotopy chain or a parameter sweep have only
the κ helper. Two natural extensions:

1. **`make_vf0_override_refresh(V_F0)`**: companion to V8's
   κ-override. Overrides the diode's `V_F0` while leaving κ,
   R_d, G_off from the pool. Useful for V_F0 homotopy chains
   AND for parameter sweeps (e.g. evaluate the same circuit at
   V_F0 = {0.3, 0.5, 0.7}).
2. **`make_kappa_vf0_override_refresh(κ, V_F0)`**: combined
   override. Lets a chain ramp BOTH parameters simultaneously,
   which is more robust than ramping either alone for problems
   where κ and V_F0 contribute jointly to stiffness.

**HONEST scope decision**: V9 does NOT claim to solve the κ=20
sinusoidal rectifier from `x = 0`. Empirical exploration during
V9 implementation found that:
- Pure V_F0 continuation at fixed κ=20 produces Newton
  overshoots large enough to cause Jacobian singularity.
- Combined κ + V_F0 continuation still hits singularity near
  diode commutation events for the sinusoidal case.
- For κ=20 sinusoidal rectifiers, the V8 LOAD-LINE WARM-START
  remains the recommended path.

V9 ships the helpers as composable parameter-override
primitives, validated on the DC load-line problem (the V3
benchmark). This is honest, useful, and additive.

## What Changes

**Scope decision — Layer 4 V9** (V_F0 override + combined
helpers):

- Add `make_vf0_override_refresh(V_F0_override)` to
  `pwl/nonlinear_refresh_diode.hpp`:
  ```cpp
  [[nodiscard]] NonlinearRefreshFn
  make_vf0_override_refresh(Real V_F0_override);
  ```
  Stamps the smooth-blend `IdealDiode` with `V_F0 =
  V_F0_override`. Other params (R_d, G_off, κ) come from the
  pool unchanged.

- Add `make_kappa_vf0_override_refresh(κ, V_F0)` to
  `pwl/nonlinear_refresh_diode.hpp`:
  ```cpp
  [[nodiscard]] NonlinearRefreshFn
  make_kappa_vf0_override_refresh(Real kappa_override,
                                    Real V_F0_override);
  ```
  Stamps with BOTH overrides simultaneously. Other params
  (R_d, G_off) come from the pool unchanged.

- **Tests** (4 cases):
  - `make_vf0_override_refresh` uses the override V_F0 (the
    residual at a known x differs from the pool-default
    refresh).
  - `make_kappa_vf0_override_refresh` overrides both (the
    combined residual differs from each single override).
  - Single-element V_F0 continuation sequence ==
    `solve_with_newton_b_extra` directly (sanity invariant).
  - V_F0 sweep across {0.3, 0.5, 0.7} on the DC load-line
    matches analytical `v_n1 ≈ V_dc − V_F0` within 50 mV.

## Impact

- **Affected specs**: ADDED requirement on `kernel-v2-solver`
  for `make_vf0_override_refresh` and
  `make_kappa_vf0_override_refresh`.
- **Affected code** (~100 LOC):
  - MODIFIED `pwl/nonlinear_refresh_diode.hpp` (+ two factories)
  - NEW `tests/v2/layer5_v4/test_vf0_continuation_rectifier.cpp`
- **Migration**: zero. All additive.
- **Risk**: low. The factories are direct extensions of V8's
  κ-override pattern; the V_F0 / κ overrides are exposed
  through the same `NonlinearRefreshFn` interface and tested
  on the same DC load-line circuit.
