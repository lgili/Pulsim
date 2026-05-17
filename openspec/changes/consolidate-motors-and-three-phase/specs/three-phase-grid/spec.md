## ADDED Requirements

### Requirement: Composition Over Duplication in Three-Phase Source Wiring

Circuit-side parameter structs for three-phase sources (`ThreePhaseSourceParams`, `ThreePhaseProgrammableSourceParams`, `ThreePhaseHarmonicSourceParams`) SHALL compose the corresponding math object from `pulsim::v1::grid::` (`ThreePhaseSource`, `ThreePhaseSourceProgrammable`, `ThreePhaseHarmonicSource`) rather than duplicate its field set. Forwarding accessors and setters on the params struct delegate to the embedded math object so that source-of-truth lives in `grid/`.

#### Scenario: ThreePhaseSourceParams composes the math object
- **GIVEN** the consolidated codebase
- **WHEN** a user constructs a `Circuit::ThreePhaseSourceParams`
- **THEN** the struct holds a `grid::ThreePhaseSource source` member
- **AND** the field accessors (`v_rms`, `frequency`, `phase_rad`, `sequence`) read and write through that member

#### Scenario: Direct math-object overload exists
- **GIVEN** a user who already constructed a `grid::ThreePhaseSource`
- **WHEN** the user calls `Circuit::add_three_phase_source(name, nodes, source)` passing the math object directly
- **THEN** the device is registered with the same stamping behavior as the params-struct overload
- **AND** the API surface is available both from C++ and from the Python binding
