## ADDED Requirements

### Requirement: Top-Level Analysis Section
The YAML schema SHALL accept a top-level `analysis:` array, parallel to `simulation:`, allowing one or more frequency-domain analyses on the same netlist.

#### Scenario: Single AC analysis
- **GIVEN** a netlist with:
```yaml
analysis:
  - type: ac
    f_start: 1
    f_stop: 1e6
    points_per_decade: 20
    perturbation_source: Vin
    measurement_nodes: [vout]
```
- **WHEN** the parser loads
- **THEN** an `AcSweepOptions` is constructed and bound to the simulation
- **AND** the analysis runs after the transient (or alone if no transient is requested)

#### Scenario: AC and FRA combined
- **GIVEN** an `analysis:` array with one `ac` entry and one `fra` entry
- **WHEN** the simulator runs
- **THEN** both analyses execute sequentially using the same DC operating point
- **AND** results are emitted under named keys in the result bundle

### Requirement: Strict Validation of Analysis Configuration
Strict validation SHALL reject malformed analysis blocks with deterministic diagnostics.

#### Scenario: Unknown analysis type
- **GIVEN** `analysis: [{ type: bode_diagram }]` (not a recognized type)
- **WHEN** strict parsing runs
- **THEN** parsing fails with reason `unknown_analysis_type`
- **AND** the diagnostic suggests the supported types

#### Scenario: Inconsistent frequency range
- **GIVEN** an `ac` analysis with `f_start >= f_stop`
- **WHEN** strict parsing runs
- **THEN** parsing fails with `invalid_frequency_range`
- **AND** the diagnostic shows the offending values

#### Scenario: Missing perturbation source
- **GIVEN** an analysis lacking `perturbation_source`
- **WHEN** strict parsing runs
- **THEN** parsing fails with `missing_required_parameter` and the field name
