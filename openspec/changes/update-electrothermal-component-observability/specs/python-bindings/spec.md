## ADDED Requirements
### Requirement: Unified Per-Component Electrothermal Telemetry in Python
Python bindings SHALL expose a unified per-component electrothermal telemetry surface in `SimulationResult`.

#### Scenario: Access per-component losses and temperatures
- **WHEN** Python runs a transient simulation with electrothermal options
- **THEN** `SimulationResult` exposes per-component entries keyed by component identity
- **AND** each entry includes both loss and temperature fields with deterministic schema and ordering

#### Scenario: Thermal-disabled entry shape remains stable
- **WHEN** a component has no enabled thermal port
- **THEN** the component entry still includes thermal fields
- **AND** thermal status is explicit (`thermal_enabled=false`) with deterministic default values

### Requirement: Backward-Compatible Summary Surfaces
Python bindings SHALL keep existing `loss_summary` and `thermal_summary` surfaces while introducing unified per-component telemetry.

#### Scenario: Existing tooling reads legacy summaries
- **WHEN** Python tooling continues to consume `loss_summary` and `thermal_summary`
- **THEN** behavior remains backward compatible
- **AND** aggregate values remain consistent with reductions of the unified per-component telemetry
