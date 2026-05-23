## ADDED Requirements

### Requirement: Small-Signal Linearisation at Operating Point

The system SHALL provide an `ac::linearise_at` function that, given a graph, device pool, switch state mask, and operating-point state vector `x_op`, returns the linearised state-space matrices `(A, B, C, D)` representing the small-signal dynamics around `x_op`.

#### Scenario: Linear circuit linearisation is exact
- **WHEN** a purely linear RLC circuit is linearised
- **THEN** the returned `A` matrix SHALL equal the matrix that `assemble_segment` produces with the same switch state (within floating-point exactness)

#### Scenario: Nonlinear device contributes its Jacobian
- **WHEN** a circuit contains a `MosfetLevel1` device biased at a specific operating point
- **THEN** the linearisation SHALL include the device's `∂I/∂V_drain`, `∂I/∂V_source`, `∂I/∂V_gate` partial derivatives at that operating point

### Requirement: AC Frequency Sweep

The system SHALL provide a `run_ac_sweep` function that, given a linearised system and a list of frequencies, computes the complex transfer function `H(jω)` from a specified input node to a specified output node.

#### Scenario: First-order RC low-pass
- **WHEN** an RC low-pass filter (R = 1 kΩ, C = 1 µF) is swept from 10 Hz to 100 kHz
- **THEN** the returned `|H(jω)|` at `f = 1/(2π·RC) ≈ 159 Hz` SHALL be within 0.1 dB of -3 dB
- **AND** the phase at that frequency SHALL be within 1° of -45°

#### Scenario: Log-spaced frequency grid
- **WHEN** the user specifies `f_start = 10`, `f_end = 1e6`, `points_per_decade = 20`
- **THEN** the returned `freqs` array SHALL contain 5 × 20 + 1 = 101 frequency points logarithmically spaced

### Requirement: Bode Plot Utilities

The system SHALL provide `bode_data` accessors that convert an `AcSweepResult` into magnitude-in-decibels and phase-in-degrees arrays suitable for plotting.

#### Scenario: Magnitude conversion
- **WHEN** an `AcSweepResult` contains `H(jω) = 0.5 + 0j` at some frequency
- **THEN** `bode_data` SHALL return `20·log10(|H|) = -6.02 dB` at that frequency

### Requirement: YAML Schema for AC Sweep

The YAML loader SHALL parse an optional top-level `analysis:` block with `ac_sweep:` parameters (`f_start`, `f_end`, `points_per_decade`, `input_node`, `output_node`).

#### Scenario: Round-trip parsing
- **WHEN** a YAML file contains an `analysis: ac_sweep` block with valid parameters
- **THEN** `yaml::load_string()` SHALL return a `LoadedCircuit` with an `ac_sweep_options` field populated

### Requirement: Buck AC Sweep Showcase

The repository SHALL include a showcase that performs an AC sweep on an open-loop buck converter and verifies the LC double-pole and -40 dB/decade slope past resonance.

#### Scenario: Buck control-to-output Bode matches analytical
- **WHEN** the showcase runs an AC sweep on a buck with `L = 100 µH, C = 100 µF, R_load = 5 Ω`
- **THEN** the magnitude at `f_LC = 1591 Hz` SHALL show the expected resonant peak (within 3 dB of analytical)
- **AND** the slope past resonance SHALL be -40 ± 2 dB/decade
