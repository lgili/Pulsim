# converter-templates Specification

## Purpose
TBD - created by archiving change add-converter-templates. Update Purpose after archive.
## Requirements
### Requirement: Converter Template Registration and Expansion
The library SHALL maintain a registry of converter templates, each with deterministic parameter-to-netlist expansion.

#### Scenario: Template registered and discovered
- **GIVEN** a template registered as `buck_template` in the kernel
- **WHEN** a YAML netlist references `type: buck_template`
- **THEN** the parser invokes the registered expansion
- **AND** the result is a sub-netlist of primitive components

#### Scenario: Deterministic expansion
- **GIVEN** the same parameter set
- **WHEN** expansion runs twice
- **THEN** the resulting sub-netlists are bit-identical
- **AND** node naming follows a documented convention (`<template_name>_<role>`)

#### Scenario: Unknown template diagnostic
- **GIVEN** a netlist referencing `type: unknown_template`
- **WHEN** strict parsing runs
- **THEN** parsing fails with `unknown_template` reason
- **AND** the diagnostic suggests templates with similar names

### Requirement: Auto-Design from High-Level Specs
Each template SHALL produce a runnable design when given only `(Vin, Vout, Pout, fsw)` without explicit component values.

#### Scenario: Buck auto-design
- **GIVEN** `buck_template` with `Vin=48, Vout=12, Pout=240, fsw=100e3`
- **WHEN** expansion runs without explicit `inductor`/`output_cap`
- **THEN** inductor is sized for ≤30% peak-to-peak ripple
- **AND** output cap is sized for ≤1% Vout ripple
- **AND** the simulation reaches steady state within 10 ms simulated time

#### Scenario: Override default
- **GIVEN** auto-design heuristics computed and `inductor.value: 100e-6` provided explicitly
- **WHEN** expansion runs
- **THEN** the explicit value overrides the auto-design
- **AND** the auto-designed value is reported in `BackendTelemetry.template_expansions[name].auto_designed_overridden`

### Requirement: First-Wave Template Coverage
The library SHALL ship 10 first-wave templates: buck, boost, buck-boost, flyback, forward, two-switch forward, half-bridge, full-bridge, LLC half-bridge, dual-active-bridge, totem-pole PFC, interleaved-2φ buck.

#### Scenario: Template availability
- **WHEN** `pulsim.list_templates()` is called
- **THEN** the returned list contains all 10 first-wave templates with version metadata

#### Scenario: Template documentation page exists
- **WHEN** the docs site is built
- **THEN** each first-wave template has a docs page with schematic, defaults, and design equations

### Requirement: Control Compensator Sub-Templates
The library SHALL provide control compensator sub-templates: `voltage_mode_pi`, `current_mode_peak`, `current_mode_average`, `type_2_compensator`, `type_3_compensator`.

#### Scenario: PI compensator inside buck template
- **GIVEN** a `buck_template` with `control: { type: voltage_mode_pi, kp: 0.5, ki: 1500 }`
- **WHEN** expansion runs
- **THEN** the expanded netlist includes a `PIController` block wired to the duty-cycle input
- **AND** the closed-loop AC sweep shows expected compensator transfer function within 1 dB / 5°

#### Scenario: Type-III auto-tuning
- **GIVEN** `control: { type: type_3_compensator, crossover_hz: 5000, phase_margin_deg: 60 }`
- **WHEN** expansion runs
- **THEN** the compensator zeros and poles are placed to achieve the requested crossover and margin within 20%
- **AND** the design notes are emitted as part of expansion metadata

### Requirement: Reference-Design Parity Validation
At least 5 templates SHALL pass parity tests against published reference designs (vendor application notes or evaluation kits).

#### Scenario: Buck parity vs TI WEBENCH
- **GIVEN** a `buck_template` configured to match TI WEBENCH design XYZ
- **WHEN** the simulation runs
- **THEN** steady-state inductor current ripple matches WEBENCH within 10%
- **AND** crossover frequency matches WEBENCH within 20%

#### Scenario: LLC parity vs published gain plot
- **GIVEN** an `llc_half_bridge_template` configured per published reference
- **WHEN** AC sweep runs
- **THEN** the gain at the resonant frequency matches the reference within 10%

### Requirement: Template Expansion Performance
Template expansion SHALL complete in ≤10 ms per instance for the largest template (DAB).

#### Scenario: DAB expansion timing
- **GIVEN** a `dab_template` with full default parameters
- **WHEN** the parser runs expansion
- **THEN** expansion latency is measured ≤10 ms on the CI baseline machine
- **AND** the latency is recorded in `BackendTelemetry.template_expansions[name].expansion_time_ms`

