## Phase A — High-impact, low-cost (~1.5 days)

### A1. IGBT V_CE_sat Norton-shift in Behavioral stamp
- [ ] A1.1 Add `V_ce_sat_at_Tj()` accessor return value to the
      `i_C(V_CE)` Behavioral expression in
      `igbt.hpp::stamp_jacobian_behavioral` (~line 343). Mirror the
      diode pattern: `i_C = (V_CE − V_CE_sat·alpha_gate) / R_CE_on` with
      `R_CE_on` the existing Params field.
- [ ] A1.2 Update the AD path (`stamp_jacobian_via_ad`) to match — the
      AD expression must autodiff the same template the manual stamp
      encodes so `test_ad_igbt_stamp.cpp` continues to cross-validate.
- [ ] A1.3 Add `test_igbt_vce_sat_stamp.cpp`: 50 A through an ON IGBT
      with `v_ce_sat = 1.5 V` produces `V_CE ≈ 1.5 V + 50·R_CE_on`, not
      `50/g_on ≈ 5 mV`.
- [ ] A1.4 Update `test_ad_igbt_stamp.cpp` cross-validation to use the
      new expression; the AD vs manual `1e-12 margin` assertion must
      still pass.

### A2. Gate-row diagonal anchor on MOSFET + IGBT
- [ ] A2.1 Add `MOSFETParams::g_gate_leak = 1e-9` (S). Stamp
      `J(n_gate, n_gate) += g_gate_leak` unconditionally at the top
      of `stamp_jacobian_behavioral`, `stamp_jacobian_ideal`, and
      `stamp_jacobian_via_ad` in `mosfet.hpp`.
- [ ] A2.2 Mirror on `IGBTParams::g_gate_leak = 1e-9` and the three
      IGBT stamp paths.
- [ ] A2.3 Add `test_mosfet_gate_anchor.cpp`: build an NMOS with the
      gate floating (no PWM source, no R_g to ground), put a pull-up
      on drain → simulation MUST succeed and V_drain MUST equal
      V_dc (within 0.1 V). Without the anchor, the linear row is
      singular and the test fails.
- [ ] A2.4 Verify the auto-parasitics + Catch2 sweep stays green
      (4081 assertions across 274 cases).

### A3. Reduce smooth-blend κ to 20 (or clamp σ_g)
- [ ] A3.1 In `mosfet.hpp`, change
      `static constexpr Real kSmoothRegionSharpness = Real{50.0};` (line
      ~397) to `Real{20.0}`. Document the trade-off (transition window
      widens from ~120 mV to ~300 mV at V_th=2 V, still narrow vs
      typical PWM amplitudes).
- [ ] A3.2 Mirror in `igbt.hpp::kSmoothRegionSharpness`.
- [ ] A3.3 Add `test_mosfet_cutoff_float_underflow.cpp`: assemble the
      stamp in float-precision build with V_gs=0, V_th=4 — the
      `sigma_g` value MUST remain non-denormal (~`exp(-80) ≈ 1.8e-35`,
      well above the 1.4e-45 float denormal floor).
- [ ] A3.4 Visually confirm that
      `examples/notebooks/31_single_phase_to_three_phase_vsi.ipynb`
      still produces clean SPWM after the κ change. (The audit case is
      not directly κ-sensitive, but a regression check is cheap.)

### A4. Bump diode + VCSwitch event_hysteresis defaults
- [ ] A4.1 In `ideal_diode.hpp` line ~653, change
      `event_hysteresis_ = Real{1e-2}` to `Real{5e-2}` (50 mV). Update
      the comment to explain the bus-noise tolerance reasoning.
- [ ] A4.2 In `voltage_controlled_switch.hpp` line ~268, change
      `event_hysteresis_ = Real{1e-9}` to `Real{1e-3}` (1 mV — three
      orders looser; still tighter than the diode but no longer
      anomalously tight vs the rest of the family).
- [ ] A4.3 If existing tests assert at the 10 mV threshold exactly,
      update them to use 100 mV input span so they pass with either
      old or new defaults. (Likely candidates:
      `test_diode_threshold_*.cpp`, `test_vcswitch_*.cpp`.)
- [ ] A4.4 Add `test_diode_chatter_on_noisy_bus.cpp`: put 20 mV pp
      noise on a 400 V bus; the diode MUST NOT register a switching
      event (chatter test).

### A5. Trapezoidal integration on motor flux + cap states
- [ ] A5.1 In `induction_motor_device.hpp::advance_state` (lines
      ~282–289), replace forward-Euler on `ψ_r` with one-iteration
      trapezoidal: compute `dψ_r_old` and `dψ_r_new`, average. Document
      that this stabilises high-slip DOL-start (`ω_e · ψ_rβ` cross-
      coupling at ω_e ≈ 314 rad/s and `dt ≈ 100 µs`).
- [ ] A5.2 In `single_phase_induction_motor_device.hpp` line ~181,
      replace `V_cap += dt·i_aux/C_run` with `V_cap += 0.5·dt·(i_aux +
      i_aux_prev)/C_run`. Cache `i_aux_prev` if not already.
- [ ] A5.3 Add `test_induction_motor_high_slip_start.cpp`: 50 Hz, full
      voltage applied at t=0 with motor at standstill, simulate 200 ms,
      assert ω converges monotonically (no oscillation > 1% of ω_sync).
- [ ] A5.4 Add `test_psc_motor_run_cap_voltage.cpp`: starting transient
      with C_run = 4 µF, assert V_cap waveform is smooth (no sub-step
      kinks > 1 V at dt = 100 µs).

### A6. Flip ModelRegularizationOptions default
- [ ] A6.1 In `simulation.hpp::ModelRegularizationOptions` line ~108,
      change `bool apply_only_in_recovery = true;` to `false`. Document
      that g_off_min floors now apply at the first Newton step, masking
      floating-gate ill-conditioning without code changes.
- [ ] A6.2 In `simulation.hpp` line ~107, change `bool enable_auto =
      false;` to `true`. Pair with A6.1 to make the regularization
      always-on by default.
- [ ] A6.3 Verify `test_pwl_segment_primary.cpp` and existing PWL
      regression tests stay green — they rely on exact `g_off` so any
      `g_off_min` floor must produce equivalent results within
      tolerance.
- [ ] A6.4 Document the default flips in `docs/component-models-audit.md`
      § 6 and `docs/convergence-tuning-guide.md` "model regularization"
      sub-section.

---

## Phase B — Body diode + thermal uniformity (~3 days)

### B1. MOSFET body diode
- [ ] B1.1 Add `MOSFETParams::body_diode_enable = true` (default ON —
      every real power MOSFET has a body diode), plus
      `body_diode_V_F0 = 0.8`, `body_diode_R_d = 25e-3`,
      `body_diode_Qrr = 0`, `body_diode_R_th_ja = 0` (shares the FET's
      R_th_ja by default).
- [ ] B1.2 In `mosfet.hpp` stamping paths, conditionally stamp an
      antiparallel `IdealDiode` companion (anode = source, cathode =
      drain) using the same `R_th_ja` and `T_amb` as the FET. The body
      diode shares the FET's switching mode (auto-promotes to Behavioral
      when the FET runs Behavioral, PWL Ideal otherwise) — match what
      `ThreePhaseVsiParams::add_body_diodes` already does in the VSI
      helper.
- [ ] B1.3 Update Python binding to expose the new fields.
- [ ] B1.4 Add `test_mosfet_synchronous_rectification.cpp`: half-bridge
      synchronous buck with `body_diode_enable = true` — during dead
      time V_sw must clamp at +V_F (or −V_F), not −V_dc. With
      `body_diode_enable = false`, V_sw goes to −V_dc as today.

### B2. IGBT antiparallel diode
- [ ] B2.1 Mirror B1 on `IGBTParams::antiparallel_diode_*`. Default ON.
- [ ] B2.2 Python binding update.
- [ ] B2.3 Add `test_igbt_inverter_freewheel.cpp`: 3φ inverter on an RL
      load — line currents must freewheel through the antiparallel
      diodes during dead time, not crash.

### B3. Motor winding thermal model
- [ ] B3.1 Add to `DcMotorParams`, `PmsmParams`, `BldcMotorParams`,
      `InductionMotorParams`, `SinglePhaseInductionMotorParams`:
      `R_th_winding_to_ambient` (K/W), `T_amb` (°C), `R_s_tc` (1/K),
      `T_ref_winding` (°C, default 20). Default `R_th_winding_to_ambient
      = 0` keeps thermal model OFF (back-compat).
- [ ] B3.2 In each motor's stamping path, scale `R_s` by `(1 + R_s_tc ·
      (T_winding − T_ref_winding))` when `R_th_winding_to_ambient > 0`.
      Mirror the diode/MOSFET pattern.
- [ ] B3.3 Add motor `accumulate_loss(...)` integrating `I_a² · R_s` (DC
      motor) and `I_s_rms² · R_s_eff` (synchronous + induction motors)
      into the unified loss pipeline.
- [ ] B3.4 Expose `<motor_kind>_steady_state_winding_temperature(name)`
      Circuit accessors for all five motor families.
- [ ] B3.5 Test: motor at rated load for several time constants,
      assert `T_winding → T_amb + I²·R·R_th_winding_to_ambient` within
      5%.

### B4. Automatic shaft coupling
- [ ] B4.1 Add `Circuit::couple_shaft(motor_name, mechanical_name,
      gear_ratio = 1.0)` to `runtime_circuit.hpp`. Stores the pair +
      gear ratio in a small `shaft_couplings_` vector.
- [ ] B4.2 At the start of each timestep (or end of
      `update_history_impl`), iterate `shaft_couplings_` and apply
      `mech.set_tau_input(motor.tau_em() / gear_ratio)` +
      `motor.set_tau_load(mech.reaction_torque() · gear_ratio)`. Wire
      through `motors::GearBox::reflect_load`.
- [ ] B4.3 Python binding `circuit.couple_shaft(motor, mech, gear)`.
- [ ] B4.4 Add `test_motor_shaft_coupling.cpp`: PMSM at 100 rad/s
      mechanically coupled to a `MechanicalDevice` (J=0.01, b=0.001) —
      apply 5 N·m load step, assert ω drops by the expected amount
      (steady-state from `τ_em = τ_load + b·ω`).

---

## Phase C — Magnetics fidelity (~4 days)

### C1. SaturableTransformer device variant
- [ ] C1.1 Create `core/include/pulsim/v1/components/saturable_transformer_device.hpp`
      wrapping `magnetic/saturable_transformer.hpp`. Mirror the
      `HysteresisInductorDevice` pattern.
- [ ] C1.2 Params: turns ratio matrix, per-winding leakage L_l_k,
      per-winding R_w_k, B-H curve params (Jiles-Atherton or polynomial
      saturation), R_th_ja, T_amb.
- [ ] C1.3 Stamp the multi-winding flux-linkage equations as a sparse
      block in MNA. Use trapezoidal companion for each winding's
      leakage L; magnetizing branch couples through λ_m state advanced
      in `update_history_impl`.
- [ ] C1.4 Add `Circuit::add_saturable_transformer(name, ...)` helper +
      Python binding.
- [ ] C1.5 Add `test_saturable_transformer_device.cpp`: open-circuit
      magnetizing current with low secondary voltage exhibits sinusoidal
      magnetizing current; at twice rated flux, magnetizing current is
      heavily distorted (peaky, saturation visible).
- [ ] C1.6 Add benchmark: flyback converter with `SaturableTransformer`
      vs PSIM golden output. Validate flux-density peak + secondary
      current shape within 5%.

### C2. Steinmetz + iGSE core loss on Inductor + SaturableTransformer
- [ ] C2.1 Add to `InductorParams`: `core_loss_k_h`, `core_loss_alpha`,
      `core_loss_beta`, `core_volume`, `effective_area`. Default all to
      0 (no core loss — back-compat).
- [ ] C2.2 In `inductor.hpp::accumulate_loss`, when `core_volume > 0`
      and `core_loss_k_h > 0`, compute B(t) = λ(t)/(N·A_e), call
      `magnetic/bh_curve.hpp::steinmetz_loss_density` or
      `igse_loss_density`, multiply by core_volume, integrate. Sum into
      `e_cond_` (same accessor — users see total iron + DCR loss).
- [ ] C2.3 Mirror on SaturableTransformer (per-winding for the
      magnetizing branch — but core loss is on the iron, not per
      winding, so a single Steinmetz call on the magnetizing flux).
- [ ] C2.4 Test: 100 µH inductor with Steinmetz k_h = 1.5e-3, α = 1.4,
      β = 2.5, V_e = 1e-6 m³, A_e = 5e-5 m², driven at 100 kHz square
      wave 10 V amplitude — Steinmetz predicts ~X W; assert
      `inductor_average_power("L1")` is within 10%.

### C3. HysteresisInductor incremental L_eff
- [ ] C3.1 In `hysteresis_inductor_device.hpp::update_history_impl` (or
      a new `assemble_state_space_hook`), recompute `L_eff = ∂λ/∂i` at
      the current Jiles-Atherton operating point. Pull from the existing
      `magnetic/hysteresis_inductor.hpp::incremental_inductance(i)`
      accessor (add it if missing).
- [ ] C3.2 Re-stamp the trapezoidal companion with the new `L_eff` each
      step (this is the key change: today it uses a constant `L_eff`
      from Params).
- [ ] C3.3 Promote `HysteresisInductorDevice` to expose the standard
      `accumulate_loss / total_energy / peak_power /
      junction_temperature` quartet. The hysteresis loop area gives
      `e_hyst`, and Steinmetz provides core loss if those params are set
      — both feed `e_cond_`.
- [ ] C3.4 Test: saturating inductor driven with a triangular current —
      `L_eff` must drop sharply above the I_sat knee; the resulting V_L
      waveform must reflect the falling inductance (V_L = L_eff · dI/dt).

---

## Phase D — Documentation + migration

- [ ] D1 Update `docs/component-models-audit.md` § 4 (Top 10) to mark
      each item with a checkbox showing its implementation status. As
      tasks complete, check them off.
- [ ] D2 Update `docs/convergence-tuning-guide.md` with a new section
      "Model Regularization Defaults" explaining the
      `apply_only_in_recovery = false` flip and how to restore the
      legacy behaviour for parity-with-SPICE testing.
- [ ] D3 Update `docs/electrothermal-workflow.md` with motor winding
      thermal coupling examples.
- [ ] D4 New doc `docs/magnetics-saturation-and-core-loss.md` covering
      `SaturableTransformer` + Steinmetz/iGSE on Inductor +
      HysteresisInductor's incremental-L behaviour.
- [ ] D5 Migration note in `docs/migration-guide.md`: two default-flip
      callouts (diode `event_hysteresis`, `apply_only_in_recovery`) +
      back-compat overrides.

---

## Phase E — Validation

- [ ] E1 Add 5 PSIM/PLECS golden-comparison benchmarks under
      `core/tests/benchmarks/`:
      - `bench_flyback_saturable_xfmr.yaml` (Phase C1)
      - `bench_synchronous_buck_body_diode.yaml` (Phase B1)
      - `bench_3phase_inverter_body_diode.yaml` (Phase B2)
      - `bench_induction_motor_dol_start.yaml` (Phase A5)
      - `bench_pmsm_drive_with_thermal.yaml` (Phase B3)
- [ ] E2 For each, capture PSIM or PLECS reference waveforms (CSV) +
      run Pulsim → compare with 5% tolerance on key KPIs (peak V/I,
      RMS, efficiency).
- [ ] E3 Run the full 50-benchmark dashboard at the end of Phase C →
      assert no regressions on the existing benches.
- [ ] E4 Full Catch2 suite: 4081 assertions / 274 cases must stay green
      throughout. Phase A1 / A2 / A3 add ~6 new cases; B1–B4 add ~8;
      C1–C3 add ~6. Final target ~294 cases / ~4100 assertions.
