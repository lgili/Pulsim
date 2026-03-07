## MODIFIED Requirements
### Requirement: Modular Component Model Library
The system SHALL define each built-in electrical component model in a dedicated component file under a stable component-library path, expose model integration through registry/module contracts, and preserve compatibility aggregator includes for legacy callers.

#### Scenario: Legacy include compatibility after modularization
- **GIVEN** existing code that includes `pulsim/v1/device_base.hpp`
- **WHEN** the project is built after model modularization
- **THEN** all existing built-in component types remain available
- **AND** no caller migration is required for include-path compatibility

#### Scenario: Isolated model evolution per component
- **GIVEN** a change to one component model file
- **WHEN** tests and benchmarks are executed
- **THEN** only that component module and declared integration paths are impacted
- **AND** unrelated models do not require structural edits in the same file

#### Scenario: Registry-driven model integration
- **GIVEN** a new component model that satisfies device contract requirements
- **WHEN** it is added through model registry/module contracts
- **THEN** runtime discovers the model without mandatory edits in central orchestrator files
- **AND** incompatible model metadata is rejected with deterministic diagnostics
