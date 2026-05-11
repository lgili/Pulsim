## Why
PSIM and PLECS are sold primarily to **motor drive** designers — that's where their revenue lives. Pulsim today has zero motor-drive benchmarks. The `motor-models` spec exists but is unexercised in the regression matrix.

A serious power-electronics simulator must validate:
- **PMSM** (Permanent Magnet Synchronous) — the dominant industrial drive
- **BLDC** trapezoidal — common in low-cost / fan / pump applications
- **Three-phase induction** — workhorse of industry
- **DC brush** — simplest, used as sanity baseline

For each: torque ripple, current THD, locked-rotor inrush, regen-braking response are the standard datasheet figures.

## What Changes
- Define motor models as YAML circuits using existing primitives (voltage sources for back-EMF, coupled_inductor for stator windings, current_source / resistor for the rotor circuit). No new C++ device types needed.
- New benchmark family `motor_*`:
  - `motor_dc_brush_step_load` — DC motor with sudden 50 % load step; observe i_armature transient
  - `motor_pmsm_dq_open_loop` — PMSM modeled in dq frame; constant V_d / V_q, observe i_d, i_q reaching steady state
  - `motor_bldc_six_step` — trapezoidal back-EMF with a 6-step commutation pattern; observe phase currents
  - `motor_induction_locked_rotor` — three-phase induction at locked rotor (s = 1); observe inrush current
- KPI extraction (depends on `add-kpi-measurement-suite`): torque ripple % at steady-state, current THD per phase, average torque.

## Impact
- Affected specs: `motor-models` (concrete benchmark requirements), `benchmark-suite`.
- Affected code: new YAML circuits + baselines under `benchmarks/circuits/` and `benchmarks/baselines/`.
- The motors are built from existing primitives (inductors, coupled inductors, voltage sources, current sources). No C++ changes required.
- Validation: baseline-against-Pulsim regression (deterministic), plus analytical checks (e.g. PMSM steady-state torque T = (3/2)·p·λ_pm·i_q matches measured torque to within 2 %).
