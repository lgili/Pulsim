## ADDED Requirements

### Requirement: Three-phase grid source device integration
The simulator SHALL accept `grid::ThreePhaseSource` and
`grid::ProgrammableThreePhaseSource` as first-class Circuit devices via a new
`Circuit::add_three_phase_source(...)` method.

#### Scenario: Drop into Circuit and simulate
- **GIVEN** a Circuit with three named nodes (a, b, c) and a ground reference
- **WHEN** the caller invokes `Circuit::add_three_phase_source("Vgrid", {a, b, c}, params)`
  and runs a transient
- **THEN** the result contains three voltage waveforms 120° apart at the requested
  fundamental frequency
- **AND** the waveforms participate in the simulator's matrix stamping the same way
  scalar voltage sources do

#### Scenario: Programmable per-phase scaling
- **GIVEN** a `ProgrammableThreePhaseSource` is added to the Circuit
- **WHEN** the caller updates the per-phase gain envelope at run-time (g_a, g_b, g_c)
- **THEN** the next transient step uses the new gains for the next sample
