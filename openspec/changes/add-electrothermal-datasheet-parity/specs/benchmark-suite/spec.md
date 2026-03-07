## ADDED Requirements
### Requirement: Closed-Loop Converter Electrothermal Parity Gate
The benchmark suite SHALL include closed-loop converter scenarios that validate control behavior together with non-trivial electrothermal behavior.

#### Scenario: Closed-loop buck with PWM and PI
- **WHEN** the closed-loop buck electrothermal benchmark runs
- **THEN** control channels demonstrate bounded PI/PWM behavior
- **AND** semiconductor switching-loss components are non-zero when switching-loss models are configured
- **AND** thermal traces show physically consistent time evolution

### Requirement: Component-Minimum Electrothermal Theory Matrix
The validation suite SHALL include per-component minimum circuits with expected electrothermal behavior checks.

#### Scenario: Thermal-enabled component minimum circuit
- **WHEN** each supported thermal-capable component is simulated in a minimum deterministic circuit
- **THEN** simulated temperatures and losses are compared against theoretical or reference expectations
- **AND** errors must remain within configured tolerances

### Requirement: Electrothermal Channel/Summary Consistency Regression
Benchmark regression SHALL verify deterministic consistency between electrothermal time-series channels and summary payloads.

#### Scenario: Channel-to-summary reduction check
- **WHEN** an electrothermal benchmark completes
- **THEN** reductions over `P*` and `T*` channels match summary fields within configured tolerance
- **AND** mismatch fails gate deterministically

### Requirement: Electrothermal Performance Non-Regression Gate
Electrothermal benchmark gating SHALL include runtime and memory/allocation stability thresholds for rich datasheet-mode scenarios.

#### Scenario: Rich electrothermal benchmark run
- **WHEN** benchmark scenarios use datasheet-grade tables and multi-stage thermal networks
- **THEN** runtime and allocation telemetry are compared against approved thresholds
- **AND** regressions beyond thresholds fail the gate
