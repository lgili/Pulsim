## 1. Three-phase sine-PWM inverter
- [ ] 1.1 Create `three_phase_inverter_svpwm.yaml`: 24 V DC bus, three half-bridges (6 vcswitches), each gate pair driven by a sine modulating signal (one of three 60 Hz sine sources at 0°/120°/240°) compared to a single triangular carrier at the switching frequency (e.g. 10 kHz).
- [ ] 1.2 Implement the sine-vs-carrier comparison using existing `comparator` blocks (channel output drives the PWM generator's `duty_from_channel`).
- [ ] 1.3 Wye-connected RL load (3 R + 3 L_load to a common neutral).
- [ ] 1.4 Capture baseline of V(load_a) and the three phase currents.
- [ ] 1.5 KPI: per-phase current THD, phase imbalance (max − min phase RMS).

## 2. Grid-tied single-phase inverter with PLL
- [ ] 2.1 Define a "grid" voltage source (60 Hz, 110 V_pk).
- [ ] 2.2 Build a Park-transform-based single-phase PLL using existing virtual blocks: `gain` + `integrator` + `sum` to lock the inverter's reference sine to the grid.
- [ ] 2.3 H-bridge inverter (4 vcswitches) modulated by the PLL output to track the grid sine.
- [ ] 2.4 Capture baseline of phase error (PLL output − grid angle) and the inverter output voltage.
- [ ] 2.5 KPI: steady-state phase error < 1° after PLL settles; settling time < 50 ms.

## 3. Back-to-back AC-DC-AC
- [ ] 3.1 Input stage: single-phase AC source (60 Hz, 100 V_pk) → 4-diode bridge → DC-link cap (1 mF, IC = 100 V).
- [ ] 3.2 Output stage: three-phase 6-switch inverter (reusing the SVPWM modulator from step 1) driving a wye RL load at variable frequency (e.g. 30 Hz output).
- [ ] 3.3 Capture baseline of V(dc_link) and one output phase current.
- [ ] 3.4 KPI: DC-link ripple peak-to-peak, output current THD, end-to-end power flow check (P_in ≈ P_out + losses).

## 4. Wiring + dashboard
- [ ] 4.1 Register all three benchmarks in `benchmarks.yaml` with `validation: type: reference` and the relevant observable per spec.
- [ ] 4.2 Generate baselines.
- [ ] 4.3 Confirm `closed_loop_dashboard.py` picks them up and they pass deterministically.
- [ ] 4.4 Document the PLL construction pattern in `docs/` (which blocks compose into a SOGI-style PLL) for reuse in future benchmarks.
