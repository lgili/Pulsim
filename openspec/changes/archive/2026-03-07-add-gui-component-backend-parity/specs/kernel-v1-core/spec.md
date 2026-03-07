## ADDED Requirements

### Requirement: Mixed-Domain Execution Scheduler

The v1 kernel SHALL support deterministic mixed-domain execution for electrical devices, behavioral control blocks, and virtual instrumentation/routing blocks.

#### Scenario: Electrical-control coupling in one timestep
- **GIVEN** a circuit containing electrical devices and control blocks
- **WHEN** an accepted timestep is processed
- **THEN** electrical solve, control update, and event-state updates execute in deterministic order
- **AND** resulting signals are consistent and reproducible across runs

### Requirement: Event-Driven Stateful Device Transitions

The v1 kernel SHALL support event-driven state transitions for latching and trip-based components (`THYRISTOR`, `TRIAC`, `FUSE`, `CIRCUIT_BREAKER`, `RELAY`).

#### Scenario: Stateful transition with event localization
- **GIVEN** a component whose state change depends on threshold crossing
- **WHEN** crossing occurs within a step
- **THEN** event localization refines transition timing
- **AND** state change is applied without non-deterministic ordering

### Requirement: Virtual Instrumentation Graph

The v1 kernel SHALL support virtual instrumentation components (`VOLTAGE_PROBE`, `CURRENT_PROBE`, `POWER_PROBE`, `ELECTRICAL_SCOPE`, `THERMAL_SCOPE`) that do not directly stamp the MNA system.

#### Scenario: Probe and scope extraction
- **GIVEN** a circuit with probes and scopes bound to electrical/thermal signals
- **WHEN** simulation runs
- **THEN** configured channel signals are captured and emitted in result data
- **AND** instrumentation components do not alter electrical matrix topology

### Requirement: Signal Routing Blocks

The v1 kernel SHALL support virtual signal routing components (`SIGNAL_MUX`, `SIGNAL_DEMUX`) for deterministic channel mapping.

#### Scenario: Mux/demux channel routing
- **GIVEN** signal routing blocks with explicit channel ordering
- **WHEN** upstream signals update
- **THEN** mux/demux outputs reflect configured mapping deterministically
