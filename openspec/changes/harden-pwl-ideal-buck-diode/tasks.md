## 1. Phase 1 — Reproduce the 3 gap classes as red tests

- [ ] 1.1 Build a minimal closed-loop buck reproducer outside the existing
      test suite so the gap is debuggable without 2200 lines of fixture
      around it. Pin the duty-step that triggers the overshoot.
- [ ] 1.2 Same for the diode-loss forward-bias spike. The reproducer should
      print the device's `pwl_state_` per step and the admissibility
      decision per step so the actual failure mode is visible.
- [ ] 1.3 Same for the buck stress megavolt spike. Confirm the
      auto-parasitics analyzer's CRIT message fires and identify the exact
      step where the switch-pole voltage diverges.
- [ ] 1.4 Document each reproducer's expected vs measured waveform in
      `docs/pwl-ideal-stability-audit.md`.

## 2. Phase 2 — Freewheel-diode commutation hardening

- [ ] 2.1 In `runtime_circuit.hpp::force_inductor_driven_diode_commutations`
      add a "duty-drop pre-emption" path: when the upstream switch commits
      OFF on this step, force the inductor-driven diode to ON regardless of
      the voltage observable.
- [ ] 2.2 Pin via `test_pwl_ideal_buck_closed_loop.cpp` — same setup as
      `v1 buck closed-loop callback tracks reference without divergence`
      but PWL Ideal active.
- [ ] 2.3 Remove the `opts.switching_mode = SwitchingMode::Behavioral` pin
      from `test_v1_kernel.cpp` Buck closed-loop test case.
- [ ] 2.4 Verify the existing buck reproducer passes.

## 3. Phase 3 — Diode forward-state admissibility refinement

- [ ] 3.1 In `IdealDiode::is_pwl_admissible(...)` (or the equivalent
      `runtime_circuit.hpp` admissibility check), when the current
      `pwl_state_` is OFF AND the forward voltage exceeds
      `V_F0 + R_d * I_threshold` (default 1 µA), flag the state as
      non-admissible and trigger a Newton retry with ON-state.
- [ ] 3.2 Pin via `test_pwl_ideal_diode_loss_forward_bias.cpp`.
- [ ] 3.3 Remove the `opts.switching_mode = SwitchingMode::Behavioral` pin
      from `test_diode_loss_thermal.cpp` and its Python mirror.
- [ ] 3.4 Verify the existing diode-loss tests pass.

## 4. Phase 4 — Auto-parasitics enforcement

- [ ] 4.1 In `auto_parasitics.hpp`'s analyzer, when the per-pair report
      contains `"PWL Ideal infeasible"`, call
      `device.set_switching_mode(SwitchingMode::Behavioral)` on each
      affected switch and diode. Emit an INFO log explaining the downgrade.
- [ ] 4.2 Add `enable_auto_downgrade` flag (default true) on
      `AutoParasiticsConfig` so power users can disable.
- [ ] 4.3 Pin via `test_auto_parasitics_downgrade.cpp` — the existing
      buck-stress fixture should now run cleanly in default Auto mode.
- [ ] 4.4 Remove the `opts.switching_mode = SwitchingMode::Behavioral` pin
      from `test_stress_simulation.cpp` Buck section.
- [ ] 4.5 Verify the existing buck stress test passes.

## 5. Phase 5 — Test pin migration

After Phases 2/3/4 close, drop the explicit `opts.switching_mode = Behavioral`
pins from the kernel tests:

- [ ] 5.1 `core/tests/test_v1_kernel.cpp` (multi-event timing + buck
      closed-loop — pin from Phase 2 + 11 of `simplify-and-harden-numerical-surface`)
- [ ] 5.2 `core/tests/test_stress_simulation.cpp` Buck section (pin from
      Phase 4)
- [ ] 5.3 `core/tests/test_pwl_segment_primary.cpp` — KEEP the pin (this
      test specifically exercises the DAE fallback path, which only fires
      with Behavioral).
- [ ] 5.4 `core/tests/test_frequency_analysis_phase1.cpp` — KEEP the pin
      (tests the Behavioral linearize bail-out specifically).
- [ ] 5.5 `core/tests/test_diode_loss_thermal.cpp` + `python/tests/...py` —
      remove pin (Phase 3 fix).
- [ ] 5.6 `core/tests/test_v1_input_validation.cpp` — KEEP the pin (test
      starves Newton; only meaningful on Behavioral).
- [ ] 5.7 `core/tests/test_ad_*.cpp` — KEEP all pins. These tests
      cross-validate Behavioral closed-form Jacobians vs centered FD; the
      pin is structurally correct.
- [ ] 5.8 `core/tests/test_concepts.cpp`, `test_switching_mode.cpp` —
      KEEP all pins (same as 5.7).

## 6. Phase 6 — Validation

- [ ] 6.1 Run full `ctest --output-on-failure` — green except for the
      pre-existing `test_switching_phase4` failure documented in
      `simplify-and-harden-numerical-surface`.
- [ ] 6.2 If the multilevel benchmark suite (Phase 13 of
      `simplify-and-harden-numerical-surface`) has shipped, run it in PWL
      Ideal mode and gate ≤ 0.5 % RMS error vs the PLECS golden on the
      buck reference circuit.
- [ ] 6.3 Update `docs/pwl-switching-migration.md` Phase 11 status from
      "BLOCKED" → "shipped".
- [ ] 6.4 Update `docs/numerical-configuration.md` to remove the
      "consider pinning Behavioral on legacy buck circuits" call-out.

## 7. Phase 7 — Archive

- [ ] 7.1 `openspec archive harden-pwl-ideal-buck-diode --yes`.
