## ADDED Requirements

### Requirement: Saturable Magnetic Component Types
The YAML schema SHALL accept `saturable_inductor` and `saturable_transformer` component types with magnetic-specific parameters.

#### Scenario: Saturable inductor with table B-H
- **GIVEN** a netlist:
```yaml
- type: saturable_inductor
  name: L1
  nodes: [n1, n2]
  bh_curve:
    type: table
    points: [[0, 0], [50, 0.1], [100, 0.18], [200, 0.22]]  # H[A/m], B[T]
  n_turns: 35
  area_m2: 1.5e-4
```
- **WHEN** the parser loads
- **THEN** the device is instantiated with the table-based B-H curve
- **AND** the magnetic effective parameters are computed from N and Ae

#### Scenario: Saturable transformer with core model reference
- **GIVEN** a netlist with `core_model: ferroxcube/N87` and per-winding leakage values
- **WHEN** the parser loads
- **THEN** the catalog material parameters are loaded
- **AND** Steinmetz parameters are wired from the catalog

#### Scenario: Inline Steinmetz override
- **GIVEN** a saturable device with both `core_model:` and inline `steinmetz: { k: 3.0 }`
- **WHEN** parser loads
- **THEN** the inline parameter overrides the catalog default
- **AND** the override is reflected in `BackendTelemetry.magnetic_overrides`

### Requirement: Hysteresis Configuration in YAML
The YAML schema SHALL support `simulation.hysteresis_model: none | jiles_atherton` and per-device `jiles_atherton:` parameter overrides.

#### Scenario: Global hysteresis off
- **GIVEN** a netlist with `simulation.hysteresis_model: none`
- **WHEN** parser loads
- **THEN** all saturable devices use anhysteretic curves only

#### Scenario: Per-device hysteresis enable
- **GIVEN** global `hysteresis_model: none` but a device with `jiles_atherton: {...}`
- **WHEN** parser loads
- **THEN** that specific device uses Jiles-Atherton; others remain anhysteretic
- **AND** strict validation reports the override in diagnostics

### Requirement: Strict Validation of Magnetic Parameters
Strict validation SHALL reject malformed magnetic parameters (non-monotonic B-H, negative Steinmetz coefficients, missing required fields) with deterministic diagnostics.

#### Scenario: Non-monotonic B-H table
- **GIVEN** a `bh_curve.points` with non-monotonic H values
- **WHEN** strict parsing runs
- **THEN** parsing fails with reason `non_monotonic_bh_curve`
- **AND** the diagnostic identifies the offending point indices

#### Scenario: Missing Ae or N
- **GIVEN** a `saturable_inductor` lacking `area_m2` or `n_turns`
- **WHEN** strict parsing runs
- **THEN** parsing fails with `missing_required_parameter` and the field name
