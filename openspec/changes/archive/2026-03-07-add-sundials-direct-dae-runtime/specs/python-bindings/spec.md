## ADDED Requirements
### Requirement: SUNDIALS Formulation Control in Python
Python bindings SHALL expose SUNDIALS formulation mode controls so users can explicitly select projected-wrapper or direct runtime formulation.

#### Scenario: Configure direct formulation from Python
- **WHEN** Python code sets `SimulationOptions.sundials.formulation = Direct`
- **THEN** the transient run executes using direct SUNDIALS callbacks in the runtime backend
- **AND** procedural APIs remain backward-compatible when formulation is not set

### Requirement: SUNDIALS Formulation Telemetry in Python
Python bindings SHALL expose formulation mode and SUNDIALS internal counters through structured result telemetry.

#### Scenario: Read direct formulation telemetry
- **WHEN** Python code inspects telemetry after a SUNDIALS transient run
- **THEN** it can read selected formulation mode and key SUNDIALS counters
- **AND** can use these fields for benchmark and parity reports without parsing logs
