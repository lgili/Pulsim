## ADDED Requirements

### Requirement: Three-Phase Sine-PWM Inverter Benchmark
The benchmark suite SHALL include a three-phase six-switch inverter producing sine-PWM modulated three-phase output by composing existing control blocks (sine sources, comparators, pwm_generator).

#### Scenario: SVPWM-modulated inverter driving wye RL load
- **WHEN** running `three_phase_inverter_svpwm`
- **THEN** the three output line voltages exhibit the expected 120°-shifted fundamental at the modulator frequency (60 Hz)
- **AND** the per-phase current THD is reported as a KPI
- **AND** the phase-balance KPI (max RMS − min RMS) / mean RMS is below 5 %

### Requirement: Grid Synchronization via PLL
The benchmark suite SHALL include a closed-loop benchmark that synchronizes an inverter's output to an external "grid" sine reference using a phase-locked loop composed from existing virtual control blocks.

#### Scenario: PLL locks to 60 Hz grid reference
- **WHEN** running `grid_tied_single_phase_pll`
- **THEN** the steady-state phase error (PLL output angle − grid angle) is below 1° after the loop settles
- **AND** the PLL settling time (10 %–90 % of final error) is reported as a KPI and is below 50 ms

### Requirement: Back-to-Back AC-DC-AC Converter
The benchmark suite SHALL include a back-to-back rectifier+inverter benchmark that exercises bidirectional power flow through a shared DC-link capacitor.

#### Scenario: AC-DC-AC end-to-end power balance
- **WHEN** running `back_to_back_rectifier_inverter`
- **THEN** the DC-link voltage settles within a documented operating range
- **AND** the DC-link ripple (peak-to-peak) is reported as a KPI
- **AND** the end-to-end power balance check (P_in − P_out_filtered) deviates by less than 10 % from the documented topology losses
