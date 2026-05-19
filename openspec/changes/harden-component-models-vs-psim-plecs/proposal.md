## Why
The May 2026 component-models audit (`docs/component-models-audit.md`)
benchmarked every device family in `core/include/pulsim/v1/components/`
(~5400 LOC) against PSIM and PLECS reference models and surfaced three
critical fidelity gaps and four convergence-side numerical-hardening
opportunities:

1. **IGBT `V_CE_sat` is never stamped.** The Params field (`igbt.hpp:36`)
   feeds only `accumulate_loss` (line 174); the Jacobian path
   (`stamp_jacobian_behavioral`, line 343) gives `V_CE = I_C / g_on ≈
   5 mV` instead of the datasheet ≈ 1.5 V. A user putting our IGBT into a
   three-phase inverter and reading line-to-line voltages sees ~10 mV
   switch drops where they expect ~3 V. Highest-visibility fidelity bug
   in the library.
2. **Transformer is ideal-N-turns only.** `transformer.hpp` (122 LOC) has
   no Lm dynamics, no leakage, no saturation, no winding resistance. The
   saturable + multi-winding math already exists in
   `magnetic/saturable_transformer.hpp` (168 LOC) — it just isn't wired
   to a device-variant wrapper. Single biggest missing capability vs
   PSIM/PLECS for any magnetics-heavy converter (flyback, LLC, DAB).
3. **MOSFET / IGBT have no body diode.** Synchronous-rectification
   topologies (the entire EV / SMPS world) get V_sw = −V_dc instead of
   clamping at one diode drop. Every PSIM / PLECS power MOSFET ships
   with a body diode included.
4. **Behavioral MOSFET / IGBT smooth blend underflows at deep cutoff.**
   At κ=50 and V_gs=0, V_th=4, `sigma_g = 1/(1+exp(200))` is denormal
   in float and zero in flush-to-zero builds — the partial-derivative
   chain dies silently and Newton cannot move the gate-driven state.
5. **Floating-gate singularity on MOSFET / IGBT.** Neither device
   stamps a diagonal on `J(n_gate, n_gate)`. A high-impedance gate driver
   (or a free-floating gate during a sim builder error) makes the linear
   row structurally singular — Newton chases NaN.
6. **Threshold chatter on diode + VCSwitch.** `event_hysteresis_ = 10 mV`
   on the diode (`ideal_diode.hpp:653`) is too tight against typical
   400 V-bus ESL ringing; `VCSwitch`'s `1e-9 V` (line 268) is seven orders
   of magnitude tighter than the rest of the family — a uniformity bug.
7. **No motor has winding thermal binding.** Every diode, MOSFET, IGBT,
   resistor, capacitor, and inductor carries `R_th_ja + T_amb +
   T_j(t)` — every motor does not. Largest single uniformity hole.
8. **Forward-Euler on `ψ_r` (3φ IM) and `V_cap` (1φ PSC).** At high slip
   the cross-coupling `ω_e · ψ_rβ` is the dominant rate; FE is marginally
   stable. The PSC motor's run-cap voltage at start steps 25 V per
   100 µs sample at full current — should be trapezoidal.

Plus three smaller convergence / uniformity items that ride along
naturally with the above:

9. **`HysteresisInductorDevice` advances Jiles-Atherton state but uses a
   constant `L_eff` in the MNA stamp.** Saturation is tracked but not in
   the loop — a half-finished feature.
10. **No automatic shaft coupling between motors and `MechanicalDevice`.**
    Users have to call `mech.set_tau_input(motor.electromagnetic_torque())`
    and `motor.set_tau_load(mech.reaction_torque())` every step manually.
    `motors::GearBox::reflect_load` exists but is dead code.

The audit estimates **~9 days of focused C++ work** to close the gap to
PSIM/PLECS across all 10 items. This proposal scopes the work, lands it
in three phases by impact × cost, and adds the regression tests that
make the new behaviour permanent.

## What Changes

**Phase A — High-impact, low-cost convergence + fidelity wins** (~1.5
days, no API breaks):

- Add Norton-shifted `V_CE_sat` to IGBT's Behavioral stamp so the
  Jacobian path produces real datasheet drops.
- Stamp a `g_gate_leak = 1e-9` diagonal on MOSFET + IGBT gate row to
  anchor floating-gate scenarios.
- Reduce smooth-blend κ from 50 to 20 (or clamp `sigma_g ≥ 1e-30`) to
  prevent float underflow at deep cutoff.
- Bump diode `event_hysteresis_` default from 10 mV to 50 mV; bump
  VCSwitch from 1e-9 V to 1e-3 V for family consistency.
- Trapezoidal integration on InductionMotor's `ψ_r` and 1φ-PSC's
  `V_cap`.
- Flip `ModelRegularizationOptions::apply_only_in_recovery` to FALSE by
  default so the per-device-class `g_off_min` floors mask floating-gate
  ill-conditioning on the first Newton step (not just on retry).

**Phase B — Body-diode + thermal uniformity** (~3 days, additive flags):

- New `MOSFETParams::body_diode_*` fields (`V_F0`, `R_d`, `Qrr`,
  `enable`) plus auto-stamp of the antiparallel diode when enabled
  (defaults to ON for compatibility with the
  `boost-pfc-auto-parasitics` story).
- Mirror on IGBT (`IGBTParams::antiparallel_diode_*`).
- Promote `R_th_winding_to_ambient` + `T_winding` (with `R_s(T_winding)`
  scaling) to all five motor families (`DcMotorParams`, `PmsmParams`,
  `BldcMotorParams`, `InductionMotorParams`, `SinglePhaseInductionMotorParams`).
- Add `Circuit::couple_shaft(motor_name, mechanical_name,
  gear_ratio=1.0)` declarative API and wire `motors::GearBox::reflect_load`.

**Phase C — Magnetics fidelity** (~4 days, new device variants):

- New `SaturableTransformer` device variant wrapping
  `magnetic/saturable_transformer.hpp` (the multi-winding saturable math
  that already exists).
- Wire `magnetic/bh_curve.hpp`'s Steinmetz + iGSE core-loss math into
  `Inductor.hpp` and `SaturableTransformer` via new
  `Params::core_loss_*` fields, integrated through the standard
  `accumulate_loss` pipeline.
- Make `HysteresisInductorDevice` recompute incremental `L_eff` per
  Newton step from the Jiles-Atherton operating point (closing the
  feedback loop the device already started).
- Add `Circuit::add_saturable_transformer(...)` helper + Python binding.

## Impact

**Affected specs:** `device-models` (Phases A + B + C), `kernel-v1-core`
(Phase A solver-side regularization flip), `python-bindings` (Phase B + C
new Python-side surface for body-diode flag, motor thermal accessors,
shaft coupling, saturable transformer).

**Affected code:**
- `core/include/pulsim/v1/components/{mosfet,igbt,ideal_diode,voltage_controlled_switch,inductor,transformer,hysteresis_inductor_device}.hpp`
- `core/include/pulsim/v1/components/{dc_motor,pmsm,bldc_motor,induction_motor,single_phase_induction_motor,mechanical}_device.hpp`
- `core/include/pulsim/v1/runtime_circuit.hpp` (Circuit-side accessors +
  shaft coupling)
- `core/include/pulsim/v1/simulation.hpp` (default flip on
  `ModelRegularizationOptions`)
- `python/bindings.cpp` (Python-side body-diode flag, motor T_winding,
  shaft coupling, saturable transformer)
- New test files: `test_igbt_vce_sat_stamp.cpp`,
  `test_mosfet_gate_anchor.cpp`, `test_motor_winding_thermal.cpp`,
  `test_saturable_transformer_device.cpp`.

**Backward compatibility:** all new Params fields default to OFF
(`enable_body_diode = false`, `R_th_winding_to_ambient = 0`, etc.) so
existing user circuits see no behaviour change unless they opt in.
The two default-flips that ARE behaviour-affecting:

- Diode `event_hysteresis_` 10 mV → 50 mV (5× wider band — eliminates
  chatter on real noisy buses but a unit test that pokes at the exact
  10 mV threshold will fail).
- `ModelRegularizationOptions::apply_only_in_recovery` true → false
  (g_off_min floors now apply at first Newton step; runs that *required*
  exact 1e-12 g_off to validate a topology will see ~1e-7 leakage).

Both default-flips are documented in the migration guide; users can
restore the prior behaviour with a one-line override.

**Build / test:** standard cmake rebuild + Python extension reinstall.
No new third-party dependencies.

**Estimated effort: ~9 days of focused C++ work** broken into the three
phases above. Phase A alone (1.5 days) closes the most common
convergence failure modes; Phase B (3 days) closes the body-diode
fidelity gap; Phase C (4 days) closes the transformer gap.
