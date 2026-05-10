## ADDED Requirements

### Requirement: Automatic Differentiation for Nonlinear Device Jacobians
Nonlinear device models (`MOSFET`, `IGBT`, `IdealDiode` in Behavioral mode, `VoltageControlledSwitch` in Behavioral mode) SHALL derive their Jacobian via forward-mode automatic differentiation rather than hand-coded stamping by default.

#### Scenario: AD-derived Jacobian
- **GIVEN** a nonlinear device using AD-derived Jacobian (default)
- **WHEN** the kernel assembles the system at a Newton iteration
- **THEN** the Jacobian entries are computed by evaluating `residual_at<ADReal>()` once
- **AND** the resulting derivatives match finite-difference reference within `1e-8` relative

#### Scenario: Manual Jacobian retained behind feature flag
- **GIVEN** `PULSIM_LEGACY_MANUAL_JACOBIAN=1` is set
- **WHEN** a simulation runs
- **THEN** devices use their hand-coded `stamp_jacobian_impl` paths
- **AND** a deprecation warning is emitted once per simulation

### Requirement: AD Validation at Startup
The kernel SHALL provide a Jacobian validation routine that compares AD-derived Jacobians against finite-difference references for each device on user-supplied or curated operating points.

#### Scenario: Validation passes for all devices
- **GIVEN** `--validate-jacobians` flag is set
- **WHEN** the simulator initializes
- **THEN** each device's Jacobian is checked against FD at curated operating points
- **AND** all devices pass within `1e-8` relative tolerance

#### Scenario: Validation flags a discrepancy
- **GIVEN** validation is enabled and a device's manual stamp is incorrect
- **WHEN** the simulator initializes
- **THEN** the kernel returns a deterministic diagnostic naming the failing device, terminal pair, and worst delta
- **AND** the simulation aborts before transient stepping

### Requirement: Linear Device Stamps Bypass AD
Linear device models (`Resistor`, `Capacitor`, `Inductor`, `VoltageSource`, `CurrentSource`) SHALL retain direct stamping rather than using AD.

#### Scenario: Linear device stamp
- **GIVEN** a circuit containing only linear devices
- **WHEN** the kernel assembles
- **THEN** no AD scalar evaluation occurs (verifiable via build-time flag or runtime telemetry)
- **AND** stamping performance matches the pre-AD baseline within numerical noise

### Requirement: PWL Mode Bypasses AD
Devices operating in `SwitchingMode::Ideal` SHALL stamp constant per-topology values without AD evaluation.

#### Scenario: PWL stamp without AD
- **GIVEN** a device in `SwitchingMode::Ideal`
- **WHEN** the topology segment model is built
- **THEN** the stamped values are direct constants (`g_on` or `g_off`)
- **AND** no AD scalar arithmetic is performed for that device

### Requirement: Residual Function Contract
Devices using AD SHALL implement `template <typename Scalar> Scalar residual_at(span<const Scalar> x_terminals, Real t, Real dt, const Params& p) const`.

#### Scenario: Residual at evaluates with `Real`
- **WHEN** `residual_at<Real>` is called with concrete terminal voltages
- **THEN** it returns the device's contribution to the residual vector

#### Scenario: Residual at evaluates with `ADReal`
- **WHEN** `residual_at<ADReal>` is called with seeded AD scalars
- **THEN** the returned `ADReal` carries both the value and the local Jacobian row entries
