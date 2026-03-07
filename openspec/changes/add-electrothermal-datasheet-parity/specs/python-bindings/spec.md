## ADDED Requirements
### Requirement: Typed Electrothermal Characterization Bindings
Python bindings SHALL expose typed structures for datasheet-grade loss characterization and thermal-network configuration.

#### Scenario: Configure datasheet characterization from Python
- **WHEN** Python code configures semiconductor loss and thermal-network structures via typed bindings
- **THEN** runtime receives equivalent backend configuration without requiring YAML-only pathways
- **AND** invalid assignments fail with deterministic typed errors

### Requirement: Canonical Electrothermal Channel Metadata in Python
Python simulation results SHALL expose canonical loss and thermal channels with structured metadata sufficient for frontend routing.

#### Scenario: Frontend adapter reads channels via Python
- **WHEN** Python tooling enumerates `result.virtual_channels` and metadata
- **THEN** it can identify electrothermal channels by metadata fields (domain, quantity, source component, unit)
- **AND** no name-regex heuristic is required for channel classification

### Requirement: Backward-Compatible Summary and Telemetry Surface
Python bindings SHALL preserve existing summary payloads while adding richer per-sample electrothermal channels.

#### Scenario: Existing script consumes summaries only
- **WHEN** a script reads legacy `loss_summary`, `thermal_summary`, and `component_electrothermal`
- **THEN** behavior remains backward compatible
- **AND** summary values are consistent with reductions over canonical electrothermal channels
