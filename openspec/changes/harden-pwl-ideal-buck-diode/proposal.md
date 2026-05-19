# Harden PWL Ideal buck/diode-loss path so `SwitchingMode::Auto = Ideal` is universally safe

## Why

The OpenSpec change `simplify-and-harden-numerical-surface` (v0.11) flipped
the default `SwitchingMode::Auto` resolution from **Behavioral** to **Ideal**
inside `core/include/pulsim/v1/components/base.hpp::resolve_switching_mode()`.
The PWL state-space "Ideal" engine is preferred whenever every switching
device declares `supports_pwl` — it skips the smoothing tanh / Shichman-Hodges
nonlinear stamp, runs no Newton iteration in stable topology windows, and is
the right default for new users.

Phase 11 of `simplify-and-harden-numerical-surface` shipped the flip and
documented known stability gaps on three classes of legacy circuits. The
affected tests now pin `SwitchingMode::Behavioral` explicitly, and the
follow-up work to close those gaps is tracked here.

The three gap classes documented in the Phase 11 audit:

1. **Buck-converter closed-loop overshoots** — `test_v1_kernel.cpp`'s
   "v1 buck closed-loop callback tracks reference without divergence"
   produces `vout ≈ 20 V` for a 12 V setpoint when the PWL Ideal path is
   active. Behavioral mode settles within ±1 V. The freewheel-diode
   commutation logic in the PWL engine doesn't handle the rapid duty-cycle
   adjustment the closed-loop PI generates in some operating windows.

2. **Diode forward-bias spikes** — `test_diode_loss_thermal.cpp`'s
   `V_F0 + R_d` stamp test produces `V_diode = −1067 V` on a hard-driven
   forward-biased silicon diode when PWL Ideal is the active path. The PWL
   engine's diode admissibility check rejects the forward state at the very
   first step and the stamp picks up the `g_off` blocking conductance,
   producing the nonsensical voltage.

3. **Buck stress switch-pole megavolt spikes** — `test_stress_simulation.cpp`
   buck-converter section produces ±85 MV switch-pole voltages on a
   2-phase / 20 kHz buck. The auto-parasitics analyzer already flags this
   topology as "PWL Ideal infeasible at canonical 100 kHz / D_on = 25%" but
   the analyzer's recommendation isn't enforced automatically — the
   simulator runs Ideal anyway and produces the unphysical waveform.

## What Changes

This change does NOT revert the v0.11 default flip. The Ideal default is
correct for ~90% of circuits and the new-user UX win is real. Instead,
this change closes the three stability gaps so the legacy circuits that
currently need explicit `Behavioral` opt-in can drop the pin.

### Concrete deliverables

1. **Freewheel-diode commutation hardening** for closed-loop buck. The PWL
   engine's `force_inductor_driven_diode_commutations()` should pre-empt
   the diode-OFF state when the upstream switch turns OFF, regardless of
   the diode voltage observable at that instant. The current
   admissibility-then-bisect path lets the diode stay OFF for one full
   timestep when the duty-cycle dropped between the last two solve points,
   which is the symptom behind the closed-loop overshoot.

2. **Diode forward-state admissibility refinement**. The `g_on / g_off`
   PWL stamp's admissibility check uses the device's last-committed state
   and the current voltage observable. When the initial guess is a stale
   `g_off` from a cold-start DC, a forward-bias voltage gets multiplied by
   `g_off ≈ 1e-9` producing the megavolt artifact. The fix: when the
   admissibility check sees a forward voltage above some threshold AND the
   current state is OFF, force an ON-state retry before stamping.

3. **Auto-parasitics enforcement**. The analyzer at
   `openspec/changes/.../core/include/pulsim/v1/auto_parasitics.hpp` already
   detects the "PWL Ideal infeasible" cases and emits a CRIT log. Make this
   automatically downgrade the affected device's `mode_` to Behavioral
   when the analyzer reports `PWL Ideal infeasible`, rather than just
   logging the recommendation.

4. **Test contract migration**. Once gaps (1)–(3) close, remove the
   explicit `opts.switching_mode = SwitchingMode::Behavioral` pins added by
   `simplify-and-harden-numerical-surface` Phase 11 from:

   - `core/tests/test_v1_kernel.cpp` (multi-event + buck closed-loop)
   - `core/tests/test_stress_simulation.cpp` (buck)
   - `core/tests/test_pwl_segment_primary.cpp` (non-admissible fallback)
   - `core/tests/test_frequency_analysis_phase1.cpp` (Behavioral linearize)
   - `core/tests/test_diode_loss_thermal.cpp` (V_F0 stamp)
   - `core/tests/test_v1_input_validation.cpp` (hard nonlinear failure)
   - `core/tests/test_ad_diode_stamp.cpp`, `test_ad_mosfet_stamp.cpp`,
     `test_ad_igbt_stamp.cpp`, `test_ad_vcswitch_stamp.cpp`,
     `test_ad_validate.cpp`, `test_concepts.cpp`, `test_switching_mode.cpp`
     (AD cross-validation contracts that need Behavioral)
   - `python/tests/test_diode_loss_thermal.py`

   For the AD cross-validation tests, the Behavioral pin is structurally
   correct (the cross-validation tests Behavioral stamps) — those pins
   stay. For the kernel tests, the goal is to remove the pins.

## Impact

- **Affected specs**: `device-models` (PWL Ideal admissibility + commutation
  contract), `kernel-v1-core` (auto-parasitics enforcement), `dc-operating-point`
  (admissibility-aware DC).
- **Affected code**:
  - `core/include/pulsim/v1/runtime_circuit.hpp` (`force_inductor_driven_diode_commutations`,
    `scan_pwl_commutations`, `bisect_pwl_event_alpha`)
  - `core/include/pulsim/v1/components/diode.hpp` (admissibility +
    `g_on / g_off` stamp guard)
  - `core/include/pulsim/v1/auto_parasitics.hpp` (enforce
    "PWL Ideal infeasible" downgrade)
- **Migration**: zero user-visible API changes. Behavior change is "PWL
  Ideal works on more circuits than before"; tests that currently pin
  `Behavioral` can drop the pin once the gap closes.
- **Risk**: low. The fix only changes how the PWL engine handles cases that
  currently produce numerically invalid output (megavolt spikes,
  closed-loop divergence). No path that currently produces valid output is
  altered.
- **Validation contract**:
  1. The 6 test files listed above pass with the explicit
     `SwitchingMode::Behavioral` pins removed.
  2. The PLECS / PSIM golden CSVs from `simplify-and-harden-numerical-surface`
     Phase 13 (deferred there pending external work) produce ≤ 0.5 % RMS
     error on the buck-converter benchmark in PWL Ideal mode.
  3. The auto-parasitics analyzer's "PWL Ideal infeasible" CRIT messages
     drop to zero on the existing benchmark suite (because the analyzer
     now enforces the downgrade rather than warning).
