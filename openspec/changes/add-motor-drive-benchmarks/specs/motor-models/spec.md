## ADDED Requirements

### Requirement: Motor-Drive Benchmark Coverage
The benchmark suite SHALL include validated benchmarks for the four dominant motor types in industrial power electronics — DC brush, PMSM, BLDC, and three-phase induction — built from existing Pulsim primitives (inductors, coupled inductors, voltage sources, vcswitches) with no new C++ device types.

#### Scenario: DC brush motor handles a load step
- **WHEN** running `motor_dc_brush_step_load`
- **THEN** the simulator captures the i_armature transient when a load resistor is switched in at t = 5 ms
- **AND** the steady-state current before and after the step matches `(V − V_BE)/R_a` to within 2 %

#### Scenario: PMSM in dq frame reaches predicted i_q
- **WHEN** running `motor_pmsm_dq_open_loop` with V_d = 0 and V_q at nominal
- **THEN** the steady-state i_q matches `(V_q − ω·λ_pm)/R_s` to within 2 %
- **AND** the measured electromagnetic torque (3/2)·p·λ_pm·i_q is emitted as a KPI

#### Scenario: BLDC six-step commutation produces expected phase currents
- **WHEN** running `motor_bldc_six_step` driving a three-phase trapezoidal-back-EMF motor through a 3-phase inverter
- **THEN** the simulator captures one full electrical period of i_a, i_b, i_c
- **AND** the per-phase current THD KPI is emitted

#### Scenario: Locked-rotor induction motor inrush
- **WHEN** running `motor_induction_locked_rotor` (s = 1)
- **THEN** the measured per-phase RMS current matches `V_phase / √(R² + ω²L²)` to within 3 %
