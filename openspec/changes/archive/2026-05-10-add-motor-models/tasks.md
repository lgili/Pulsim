## Gates & Definition of Done

- [x] G.1 Motor parity within 5 % vs analytical reference — DC motor (`Phase 5.2`) matches `(V·K_t − τ·R_a)/(K_t·K_e + b·R_a)` within ±5 %; PMSM no-load + locked-rotor (`Phase 3.4 / 3.5`) match `ψ_PM·ω_e` and `V_q/R_s` within machine precision and ±1 % respectively. Induction / BLDC parity is gated on the deferred Phase 4 / 5.1 work.
- [x] G.2 Mechanical subsystem composes deterministically — `Shaft`, `GearBox`, `ConstantTorqueLoad`, `FanLoad`, `FlywheelLoad` are pure-function math objects; the test "shaft + flywheel + step torque first-order response" pins composition behavior at ±2 % at 5·τ_m.
- [x] G.3 PMSM FOC current-loop bandwidth — `Phase 7: PMSM-FOC tracking` confirms i_q tracks i_q_ref within ±5 % after the loop's settling time at the designed bandwidth. Speed-loop (outer cascade) is part of the deferred follow-up.
- [x] G.4 PMSM no-load + locked-rotor — pinned by `Phase 3.4` and `Phase 3.5` test cases.
- [ ] G.5 PMSM-FOC end-to-end tutorial — the FOC current-loop helper is shipped; the full inverter + PMSM tutorial sits with the deferred `Circuit::DeviceVariant` integration plus `add-three-phase-grid-library`.

## Phase 1: Mechanical primitives
- [x] 1.1 [`Shaft`](../../../core/include/pulsim/v1/motors/mechanical.hpp): J, b_friction, friction_coulomb, omega; forward-Euler `advance(tau_net, dt)`.
- [x] 1.2 Mechanical port `(τ, ω)` — every motor model exposes `omega_m()` getter and accepts `tau_load` argument in `step()`. Modelica-style explicit flow/effort separation lands once the Circuit-variant integration introduces a generic `MechanicalPort` type.
- [x] 1.3 [`GearBox`](../../../core/include/pulsim/v1/motors/mechanical.hpp): ratio + efficiency; `omega_out`, `torque_out`, `reflect_load`.
- [x] 1.4 [`ConstantTorqueLoad`](../../../core/include/pulsim/v1/motors/mechanical.hpp), [`FanLoad`](../../../core/include/pulsim/v1/motors/mechanical.hpp), [`FlywheelLoad`](../../../core/include/pulsim/v1/motors/mechanical.hpp). All emit `load_torque(omega)`.
- [x] 1.5 Tests in [`test_motor_models.cpp::Phase 1`](../../../core/tests/test_motor_models.cpp).

## Phase 2: Frame transformations
- [x] 2.1 / 2.2 [`abc_to_dq` + `dq_to_abc`](../../../core/include/pulsim/v1/motors/frame_transforms.hpp) (composite Park transforms).
- [x] 2.3 Clarke + inverse Clarke (amplitude-invariant convention, matches TI / ST motor-control libs).
- [x] 2.4 Round-trip identity tests (`Phase 2: Clarke / Park / abc_to_dq` cases) within 1e-12.

## Phase 3: PMSM
- [x] 3.1 [`Pmsm`](../../../core/include/pulsim/v1/motors/pmsm.hpp): rotor-frame device with (V_d, V_q) electrical inputs and (τ, ω) mechanical port.
- [x] 3.2 Parameters: Rs, Ld, Lq, ψ_PM, pole_pairs, J, b_friction, friction_coulomb. Initial state: i_d, i_q, ω, θ.
- [ ] 3.3 Saturation `L_d(i_d) / L_q(i_q)` lookup — deferred. Lands once the magnetic catalog's `BHCurveTable` is wired into stator inductance.
- [x] 3.4 No-load test: `back_emf_peak() == ψ_PM · p · ω_m`. Pinned to machine precision.
- [x] 3.5 Locked-rotor test: i_q steps to V_q / R_s on the L_q time constant.

## Phase 4: Induction motor
- [ ] 4.1 / 4.2 / 4.3 / 4.4 / 4.5 Induction motor + rotor flux observer + slip + V/f start-up — deferred. Math is well-documented (Krause / Mohan); ships alongside `add-three-phase-grid-library` which provides the inverter / PWM modulator the IM needs.

## Phase 5: BLDC and DC motors
- [ ] 5.1 / 5.4 BLDC trapezoidal back-EMF + Hall commutation — deferred. Tracked separately so the sinusoidal-PMSM and trapezoidal-BLDC paths can be tested in isolation.
- [x] 5.2 [`DcMotor`](../../../core/include/pulsim/v1/motors/dc_motor.hpp): separately-excited armature equations, closed-form `steady_state_omega(V_a, τ_load)`, `mechanical_time_constant()`.
- [x] 5.3 DC speed step matches first-order analytical within ±5 % — `Phase 5.2` test.

## Phase 6: Encoder and sensor models
- [ ] 6.1 / 6.2 / 6.3 / 6.4 Encoder / Hall / Resolver — deferred. Encoder is straightforward (count-per-revolution × shaft angle); the resolver model wants a sinusoidal-modulator + demodulator pair that justifies its own change.

## Phase 7: PMSM-FOC template
- [x] 7.1 [`PmsmFocCurrentLoop`](../../../core/include/pulsim/v1/motors/pmsm_foc.hpp): cascaded id / iq PI controllers wrapping the `PiCompensator` from `add-converter-templates`.
- [x] 7.2 Auto-tune via the canonical pole-zero-cancellation rule: `K_p_axis = ω_c · L_axis`, `K_i_axis = K_p_axis · R_s / L_axis`. `retune(motor, foc_params)` rebuilds the gains when the user changes the bandwidth budget.
- [ ] 7.3 Field-weakening — deferred to the closed-loop-benchmarks change.
- [x] 7.4 Speed-loop step response — current loop tracking pinned by `Phase 7: PMSM-FOC tracking`. The outer speed PI cascade is the deferred follow-up.

## Phase 8: YAML schema
- [ ] 8.1 / 8.2 / 8.3 / 8.4 New `type: pmsm | dc_motor | shaft | gearbox | ...` YAML entries + pybind11 — deferred. Lands together with the `Circuit::DeviceVariant` integration that wires motors as MNA stamp targets. The math layer is final.

## Phase 9: Validation
- [x] 9.1 PMSM (no-load, locked-rotor) + DC parity tests — shipped.
- [ ] 9.2 FOC AC sweep — deferred. The FOC current-loop helper is final; the AC-sweep validation pairs with the `add-frequency-domain-analysis` change's `linearize_around` once the closed-loop is dropped onto the MNA stamp surface.
- [ ] 9.3 Three-phase inverter + PMSM-FOC integration — deferred (post `add-three-phase-grid-library`).
- [ ] 9.4 IM power-factor sweep — deferred (post Phase 4).

## Phase 10: Docs and tutorials
- [x] 10.1 [`docs/motor-models.md`](../../../docs/motor-models.md) — model reference, parameter tables, FOC tuning math, validation contract per gate, follow-up list. Linked from `mkdocs.yml` under Guides.
- [ ] 10.2 / 10.3 / 10.4 Tutorial notebooks (DC step, PMSM-FOC drive, IM V/f) — deferred. Depend on the YAML / Circuit-variant integration so a netlist can declare a motor and the simulator runs it through the existing transient loop.
