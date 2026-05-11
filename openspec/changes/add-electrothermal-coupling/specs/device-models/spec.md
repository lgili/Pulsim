## ADDED Requirements

### Requirement: Temperature-Dependent On-State Resistance
Switching devices (IdealSwitch, MOSFET, IGBT) SHALL optionally accept a temperature coefficient `r_on_tcr` and reference temperature `t_ref_celsius`, and look up their bonded thermal node's temperature when computing the on-state conductance during Jacobian assembly.

#### Scenario: MOSFET R_DS_on rises with junction temperature
- **GIVEN** a MOSFET declares `r_on_ref: 5e-3, r_on_tcr: 0.005, t_ref_celsius: 25.0` and is bonded to a thermal node
- **WHEN** the thermal node's temperature reaches 75 °C
- **THEN** the device's effective on-state resistance is `5e-3 · (1 + 0.005·50) = 6.25 mΩ`
- **AND** Newton solves the electrical network with this updated R_DS_on

### Requirement: Electrothermal Benchmark Coverage
The benchmark suite SHALL include at least three benchmarks that exercise the electrothermal coupling: a steady-state thermal characterization, a thermal transient under a load step, and a self-heating resistor.

#### Scenario: Buck reaches documented junction temperature in steady state
- **WHEN** running `electrothermal_buck_steady` to thermal steady-state
- **THEN** the junction temperature of the high-side MOSFET reaches the value predicted by the analytical R_th_ja · P_diss model within 3 °C
- **AND** the converter's V(out) is consistent with the higher R_DS_on at the elevated junction temperature
