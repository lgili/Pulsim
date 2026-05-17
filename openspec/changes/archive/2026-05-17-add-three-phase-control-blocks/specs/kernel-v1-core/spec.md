## ADDED Requirements

### Requirement: Three-Phase Transform Control Blocks
The runtime SHALL provide virtual control blocks for Clarke and Park transformations and their inverses, plus a phase-locked-loop block that produces a synchronization angle θ.

#### Scenario: Clarke transform of a balanced three-phase sine source
- **GIVEN** three sine voltage sources at 0°, 120°, 240° with equal amplitude V_pk
- **WHEN** a `clarke_transform` block is connected to the three source nodes [a, b, c]
- **THEN** the block emits channel values `<name>.alpha`, `<name>.beta`, `<name>.gamma`
- **AND** `alpha` is a 60 Hz sine of amplitude V_pk (amplitude-invariant convention)
- **AND** `beta` is the 90°-shifted sine of the same amplitude
- **AND** `gamma` is approximately zero for a balanced source

#### Scenario: Park transform takes θ from a channel
- **GIVEN** a `park_transform` block with metadata `theta_from_channel: PLL.theta`
- **WHEN** the simulation runs and the PLL has produced a value `PLL.theta` for the current step
- **THEN** the Park block reads θ from `virtual_signal_state_["PLL.theta"]`
- **AND** emits `<name>.d` and `<name>.q` according to the standard Park rotation matrix

### Requirement: PLL Block
The runtime SHALL provide a `pll` virtual block that locks to a single-phase sinusoidal input via a PI loop on the q-axis projection and emits the recovered phase angle.

#### Scenario: PLL locks to a 60 Hz sine
- **GIVEN** a `pll` block with `kp` and `ki` configured for 60 Hz nominal
- **WHEN** the input is a 60 Hz sine
- **THEN** the PLL's `theta` channel converges to the input's actual phase (within ±1° after settling)
- **AND** the `lock_error` channel drops below 0.01 V_pk in steady state

### Requirement: Space-Vector Modulation Block
The runtime SHALL provide an `svm` virtual block that takes a stationary-frame reference (α, β) plus a DC bus voltage and emits three half-bridge duties.

#### Scenario: SVM produces sinusoidal duties from rotating reference
- **GIVEN** an `svm` block with `alpha_from_channel` / `beta_from_channel` referencing a rotating reference vector at 60 Hz
- **WHEN** the simulation runs for several fundamental periods
- **THEN** each of `<name>.d_a`, `<name>.d_b`, `<name>.d_c` is a sinusoid at 60 Hz with the appropriate 120°-shifted phase
- **AND** the duties are clamped to [0, 1]
