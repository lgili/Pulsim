## ADDED Requirements

### Requirement: PMSM device integration
The simulator SHALL accept `motors::PMSM` and `motors::PMSM_FOC` as first-class
Circuit devices via new `Circuit::add_pmsm(...)` and `Circuit::add_pmsm_foc(...)`
methods. Each device contributes electrical branch stamps and mechanical state
variables (rotor speed ω and rotor angle θ).

#### Scenario: PMSM-FOC no-load spin-up
- **GIVEN** a `PMSM_FOC` device with default parameters and a speed reference of
  100 rad/s
- **WHEN** the user runs a transient
- **THEN** the rotor speed signal converges to the reference within the configured
  settling time

#### Scenario: Mechanical scheduling order
- **GIVEN** a PMSM device is registered in the Circuit
- **THEN** `Circuit::mixed_domain_phase_order` SHALL place the mechanical state
  update for the device after the corresponding electrical solve in each step

### Requirement: DC motor and mechanical load device integration
The simulator SHALL accept `motors::DC_Motor` and `motors::Mechanical` (inertia +
viscous friction) via new `Circuit::add_dc_motor(...)` and
`Circuit::add_mechanical(...)` methods.

#### Scenario: DC motor coupled to a mechanical load
- **GIVEN** a DC motor connected to a Mechanical load
- **WHEN** the user runs a transient under nominal supply
- **THEN** the steady-state rotor speed reflects the torque-balance set by the
  load's inertia and friction coefficients

### Requirement: Mechanical scheduler domain tag
The Circuit's mixed-domain phase order SHALL include a `Mechanical` domain tag
distinct from the existing `Electrical` and `Control` tags.

#### Scenario: Phase order exposes Mechanical
- **WHEN** the caller invokes `Circuit::mixed_domain_phase_order`
- **THEN** the returned list SHALL contain a `Mechanical` entry placed deterministically
  relative to `Electrical` and `Control`
