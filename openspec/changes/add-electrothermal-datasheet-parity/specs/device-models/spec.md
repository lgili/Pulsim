## ADDED Requirements
### Requirement: Semiconductor Loss Characterization Profiles
Thermal-capable semiconductor device models SHALL support explicit loss-characterization profiles for conduction and switching phenomena.

#### Scenario: MOSFET/IGBT datasheet characterization profile
- **GIVEN** a MOSFET or IGBT instance with datasheet-grade characterization
- **WHEN** runtime evaluates losses
- **THEN** conduction and switching terms are evaluated from the configured profile deterministically
- **AND** profile parameters are available to telemetry and diagnostics with stable identity

#### Scenario: Diode characterization profile with reverse recovery
- **GIVEN** a diode instance with reverse-recovery characterization
- **WHEN** runtime observes valid transition conditions
- **THEN** reverse-recovery energy is computed from profile data and included in loss decomposition

### Requirement: Gate-Condition and Calibration Scaling Semantics
Semiconductor loss profiles SHALL support deterministic scaling semantics for gate-condition or calibration factors.

#### Scenario: Gate-resistance scaling on switching energy
- **GIVEN** a profile with gate-condition reference and scaling coefficients
- **WHEN** runtime evaluates switching energy under configured gate conditions
- **THEN** scaling is applied deterministically before energy commit
- **AND** applied scaling factors are bounded by configured policy limits

### Requirement: Thermal Network Model Families per Component
Thermal-capable models SHALL support `single_rc`, `foster`, and `cauer` thermal-network families with consistent state semantics.

#### Scenario: Network family selection
- **WHEN** a component selects thermal network family in configuration
- **THEN** runtime instantiates matching network-state semantics for that component
- **AND** unsupported family selection fails validation deterministically

### Requirement: Deterministic Out-of-Range Handling for Loss Surfaces
Loss-surface model evaluation SHALL define deterministic out-of-range behavior for operating-variable queries.

#### Scenario: Operating point outside table bounds
- **WHEN** runtime queries a loss surface outside configured axis bounds
- **THEN** evaluation follows configured policy (for example clamp) deterministically
- **AND** strict policy modes can fail with typed diagnostics instead of silent extrapolation
