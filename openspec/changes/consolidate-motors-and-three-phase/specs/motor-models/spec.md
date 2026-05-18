## ADDED Requirements

### Requirement: Single Source of Truth Per Motor Type

For every motor type the library supports (PMSM, DC brush, BLDC, three-phase induction), exactly **one** `DynamicDeviceBase` subclass SHALL exist in `core/include/pulsim/v1/components/` and SHALL be registered in the `DeviceVariant` union of `RuntimeCircuit`. The library SHALL NOT expose parallel POD parameter structs with their own stamping path for the same motor type.

#### Scenario: PMSM single class
- **GIVEN** the consolidated codebase
- **WHEN** a user searches for PMSM-related types
- **THEN** only `PmsmDevice` (in `components/pmsm_device.hpp`) and its math object `motors::Pmsm` are present
- **AND** no `PmsmSteadyStateParams`, `Circuit::add_pmsm_steady_state`, or equivalent helper survives in the codebase

#### Scenario: Steady-state operating point via the canonical device
- **GIVEN** a `PmsmDevice` registered through `Circuit::add_pmsm`
- **WHEN** the user wants a steady-state operating point with fixed mechanical speed
- **THEN** the user pins `omega_m` via the device's initial condition or by holding the mechanical input torque to balance load torque
- **AND** the dq currents reach the same steady-state values previously produced by the removed `PmsmSteadyStateParams` path (cross-validated by the migrated tests within 1e-6 absolute)

#### Scenario: BLDC device available
- **GIVEN** the consolidated codebase
- **WHEN** the user calls `Circuit::add_bldc_motor(name, nodes, params)`
- **THEN** a `BldcMotorDevice` is registered in the circuit's `DeviceVariant`
- **AND** the device produces a trapezoidal back-EMF waveform with 120° flat-top per phase shifted by 120° between phases

#### Scenario: Induction device available
- **GIVEN** the consolidated codebase
- **WHEN** the user calls `Circuit::add_induction_motor(name, nodes, params)`
- **THEN** an `InductionMotorDevice` is registered in the circuit's `DeviceVariant`
- **AND** the device computes electromagnetic torque from rotor flux × stator current per the squirrel-cage stationary-αβ model
