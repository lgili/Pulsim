## ADDED Requirements
### Requirement: Control Node Writeback Registry
The v1 kernel SHALL maintain a control-node registry that is distinct from electrical node values for mixed-domain control evaluation.

#### Scenario: Resolve inputs from control registry first
- **GIVEN** a control block input mapped to a node that has been written by a prior control block
- **WHEN** the control block is evaluated
- **THEN** the input resolves to the control-node registry value
- **AND** falls back to the electrical node voltage only when no control value exists.

#### Scenario: Writeback of control outputs
- **GIVEN** a control block with node outputs
- **WHEN** the block produces output values
- **THEN** the backend publishes channels and writes the same values to the control-node registry.

### Requirement: Deterministic Control Ordering and Loop Detection
The v1 kernel SHALL evaluate control blocks in a deterministic dependency order and fail fast on algebraic loops without state.

#### Scenario: Deterministic evaluation order
- **GIVEN** a cascaded control chain with no cycles
- **WHEN** the control phase executes
- **THEN** each block sees upstream outputs from the same control phase
- **AND** output channels are deterministic for identical inputs.

#### Scenario: Algebraic loop detection
- **GIVEN** a control topology that introduces a zero-delay algebraic loop without state
- **WHEN** the control phase executes
- **THEN** the kernel fails with a deterministic error message that includes the loop path.

### Requirement: Compatibility with Discrete Control Semantics
The control-node registry SHALL respect per-block discrete timing and hold behavior.

#### Scenario: Discrete block holds output between ticks
- **GIVEN** a control block with `sample_time > 0`
- **WHEN** the simulation time advances between ticks
- **THEN** the control-node registry retains the last output value.
