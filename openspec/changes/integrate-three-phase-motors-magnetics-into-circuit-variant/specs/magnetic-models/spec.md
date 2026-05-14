## ADDED Requirements

### Requirement: Saturable transformer device integration
The simulator SHALL accept `magnetic::SaturableTransformer` as a first-class Circuit
device via a new `Circuit::add_saturable_transformer(...)` method. The Jacobian
stamping SHALL use the device's BH curve to model the nonlinear flux-current
relationship.

#### Scenario: Saturation flat-tops the flux waveform
- **GIVEN** a SaturableTransformer configured with a finite saturation point
- **WHEN** the user drives the primary beyond knee voltage and runs a transient
- **THEN** the magnetizing flux waveform shows the flattening characteristic of
  saturation, distinguishable from the linear ideal-transformer reference

### Requirement: Hysteresis inductor device integration
The simulator SHALL accept `magnetic::HysteresisInductor` as a first-class Circuit
device via a new `Circuit::add_hysteresis_inductor(...)` method.

#### Scenario: BH-loop closes around an AC cycle
- **GIVEN** a HysteresisInductor driven by a sinusoidal AC source
- **WHEN** the user records B(t) and H(t) across one steady-state cycle
- **THEN** the BH plot traces a closed loop rather than a single line
