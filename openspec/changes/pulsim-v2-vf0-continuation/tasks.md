## Phase 1 — V_F0 override helper (~0.2 days)

- [x] 1.1 Add `make_vf0_override_refresh(Real V_F0)` to
      `pwl/nonlinear_refresh_diode.hpp`.
- [x] 1.2 Implementation: mirrors `make_kappa_override_refresh`,
      walks Nonlinear branches, constructs a temporary
      `IdealDiode::Params` with `V_F0 = V_F0_override`,
      and stamps using that.

## Phase 2 — Combined κ+V_F0 helper (~0.1 days)

- [x] 2.1 Add `make_kappa_vf0_override_refresh(Real kappa,
      Real V_F0)` to `pwl/nonlinear_refresh_diode.hpp`.
- [x] 2.2 Implementation: stamps with BOTH overrides
      simultaneously.

## Phase 3 — Tests (~0.2 days)

- [x] 3.1 Unit test: `make_vf0_override_refresh` uses the
      override V_F0 (residual differs from pool-default).
- [x] 3.2 Unit test: `make_kappa_vf0_override_refresh`
      overrides both params (residual differs from each
      single override).
- [x] 3.3 Sanity: single-element V_F0 continuation
      sequence == `solve_with_newton_b_extra` directly.
- [x] 3.4 V_F0 sweep on DC load-line: V_F0 ∈ {0.3, 0.5, 0.7}
      → analytical match within 50 mV.

## Phase 4 — Regression + docs (~0.1 days)

- [x] 4.1 All previous tests stay green.
- [x] 4.2 `openspec validate pulsim-v2-vf0-continuation
      --strict` passes.
- [x] 4.3 `docs/pulsim-v2/layer4-v9-vf0-continuation.md`.

## What was tried and DROPPED

(Recorded in design.md "Honest scope" section.)

- κ=20 sinusoidal rectifier from `x = 0` via V_F0 chain
  (with and without LM): inner Newton hits matrix
  singularity / local-min stall.
- Combined κ+V_F0 chain at κ_target=20: same.
- κ=10 combined chain at DC=3V: still fails near sigmoid
  knees.

These attempts informed the design.md scope statement: V9
ships the OVERRIDE FACTORIES (validated on the DC load-line),
NOT a "stiff-rectifier from zero" claim.
