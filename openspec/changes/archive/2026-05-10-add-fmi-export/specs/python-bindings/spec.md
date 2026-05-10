## ADDED Requirements

### Requirement: Python FMU Export and Import API
Python bindings SHALL expose `pulsim.fmu.export(circuit_or_path, fmu_path, **opts)` and `pulsim.fmu.load(fmu_path)`.

#### Scenario: Programmatic FMU export
- **GIVEN** a `Circuit` and an export configuration dict
- **WHEN** Python calls `pulsim.fmu.export(circuit, "buck.fmu", version="2.0", type="cs", inputs=[...], outputs=[...])`
- **THEN** the file `buck.fmu` is created
- **AND** the function returns an `FmuExportSummary` with the model description and validation result

#### Scenario: Programmatic FMU import as block
- **WHEN** Python calls `pulsim.fmu.load("foreign.fmu")` and uses the returned object as a `Circuit` block
- **THEN** the FMU is instantiated and its inputs/outputs are accessible by name
- **AND** the block participates in the simulation as a signal-domain component
