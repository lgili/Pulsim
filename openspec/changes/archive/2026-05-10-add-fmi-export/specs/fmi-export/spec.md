## ADDED Requirements

### Requirement: FMI 2.0 Co-Simulation Export
The library SHALL produce FMI 2.0 Co-Simulation FMUs that pass the standard `fmuCheck` validator.

#### Scenario: Minimal FMU export
- **GIVEN** a Pulsim netlist with declared inputs and outputs
- **WHEN** `pulsim fmu-export --version 2.0 --type cs` runs
- **THEN** the output `.fmu` is a valid zip with `modelDescription.xml`, platform binary, and sources
- **AND** running `fmuCheck` against it succeeds without error

#### Scenario: FMU runs in OMSimulator
- **GIVEN** a buck-template FMU exported by Pulsim
- **WHEN** loaded into OpenModelica's OMSimulator and stepped
- **THEN** outputs match Pulsim native simulation within 1% over the test trace

### Requirement: FMI 3.0 Co-Simulation Export
The library SHALL support FMI 3.0 Co-Simulation export with `fmi3Float64` types and the FMI 3.0 model description schema.

#### Scenario: FMI 3.0 export
- **GIVEN** the same netlist
- **WHEN** `--version 3.0` is passed
- **THEN** the FMU's `modelDescription.xml` validates against the FMI 3.0 schema
- **AND** binary callbacks use FMI 3.0 ABI

### Requirement: Variable Mapping from Netlist
The export SHALL allow declaring inputs, outputs, and tunable parameters from any YAML netlist field via an `fmu_export:` configuration section.

#### Scenario: Input mapping
- **GIVEN** an `fmu_export.inputs` entry mapping `vref` to a source named `Vref`
- **WHEN** the FMU is exported
- **THEN** `modelDescription.xml` contains a variable `vref` with `causality="input"` referencing that source
- **AND** `fmi2SetReal` on `vref` updates the source value at runtime

#### Scenario: Output mapping
- **GIVEN** an output entry `{ name: vout, node: vout }`
- **WHEN** the FMU is exported
- **THEN** `modelDescription.xml` contains a variable `vout` with `causality="output"` referencing the node voltage
- **AND** `fmi2GetReal` on `vout` returns the latest computed node voltage

#### Scenario: Parameter mapping
- **GIVEN** a parameter entry `{ name: kp_ctrl, parameter: PI_ctrl.kp }`
- **WHEN** the FMU is instantiated and `fmi2SetReal` is called on `kp_ctrl` before initialization
- **THEN** the underlying parameter is updated in the Pulsim runtime
- **AND** subsequent `fmi2DoStep` calls reflect the new value

### Requirement: State Serialization
The exported FMU SHALL implement `fmi2GetFMUstate` and `fmi2SetFMUstate` to support host-driven state save/restore.

#### Scenario: State save/restore round-trip
- **GIVEN** an FMU that has been stepped to time `t1`
- **WHEN** `fmi2GetFMUstate` saves state, `fmi2DoStep` advances to `t2`, and `fmi2SetFMUstate` restores
- **THEN** subsequent steps from the restored state produce results identical (within numerical noise) to the original trajectory

### Requirement: Co-Simulation FMU Import (Master Mode)
The library SHALL import foreign FMUs and orchestrate them as signal-domain blocks within a Pulsim study.

#### Scenario: PLECS FMU imported
- **GIVEN** a co-simulation FMU exported from PLECS (or any FMI 2.0 CS source)
- **WHEN** Pulsim loads the FMU and exchanges values via Gauss-Seidel master orchestration
- **THEN** the FMU's outputs are propagated to Pulsim signal-domain blocks each macro-step
- **AND** the simulation runs without ABI errors

#### Scenario: FMU step-size mismatch handling
- **GIVEN** an imported FMU with declared `defaultExperiment.stepSize`
- **WHEN** Pulsim's macro-step differs from FMU step-size
- **THEN** the master subdivides as needed
- **AND** telemetry records sub-step counts per macro-step

### Requirement: Compliance and Cross-Tool Testing
The CI SHALL run nightly `fmuCheck` validation and at least one cross-tool parity test (Pulsim ↔ OMSimulator) where licenses permit.

#### Scenario: Nightly compliance
- **WHEN** the nightly CI job runs
- **THEN** the reference buck FMU is exported and validated by `fmuCheck`
- **AND** any compliance regression blocks merge to main

#### Scenario: Cross-tool parity
- **WHEN** the cross-tool nightly runs
- **THEN** the same scenario simulated natively and as FMU-in-OMSimulator agrees within 1%
- **AND** divergence beyond tolerance fails the build
