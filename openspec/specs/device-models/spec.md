# device-models Specification

## Purpose
TBD - created by archiving change improve-convergence-algorithms. Update Purpose after archive.
## Requirements
### Requirement: Diode Stamp with Limiting

The diode stamp function SHALL apply voltage limiting before computing current and conductance.

The stamp SHALL:
- Retrieve previous diode voltage from device state
- Apply voltage limiting to new voltage
- Compute I and G using limited voltage
- Store new voltage in device state

#### Scenario: Diode stamp with limiting

- **GIVEN** MNA assembly with voltage limiting enabled
- **WHEN** stamp_diode() is called with V_new from Newton
- **THEN** V_limited = limit_diode_voltage(V_new, V_old)
- **AND** I and G are computed using V_limited
- **AND** V_old is updated to V_limited

### Requirement: MOSFET Stamp with Limiting

The MOSFET stamp function SHALL apply voltage limiting before computing currents.

#### Scenario: MOSFET stamp with Vgs and Vds limiting

- **GIVEN** MNA assembly with voltage limiting enabled
- **WHEN** stamp_mosfet() is called
- **THEN** Vgs_limited = limit_mosfet_vgs(Vgs_new, Vgs_old)
- **AND** Vds_limited = limit_mosfet_vds(Vds_new, Vds_old)
- **AND** drain current is computed using limited voltages

