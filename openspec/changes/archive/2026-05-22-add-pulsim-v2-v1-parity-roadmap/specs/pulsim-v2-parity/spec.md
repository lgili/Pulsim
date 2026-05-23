## ADDED Requirements

### Requirement: Mixed-Domain Virtual Block Library

Pulsim v2 SHALL provide a library of virtual (non-electrical) control / signal-processing blocks that can be wired together to build closed-loop controllers, signal conditioners, and modulators. The library SHALL be accessible from BOTH the Python builder API (as `pulsim.v2` classes) AND the YAML netlist schema (as `type:` entries under `circuit.devices`), and the two paths SHALL be functionally equivalent — the same block instance can be authored either way.

Each block in the library SHALL:
- Be implemented as a stateful Python class that mirrors the corresponding v1 block name
- Provide a `reset()` method that returns it to its initial state
- Provide an `update(...)` method that consumes input signals, advances internal state by one simulation timestep, and returns the output
- Be evaluable within the v2 kernel's `step_observer(t, x)` callback (so multiple blocks can be chained between solver steps)

Initial library (Phase A.1):
- Math: `Gain`, `Sum`, `Subtract`, `MathBlock`
- Standalone control: `Integrator`, `Differentiator`, `TransferFunction`, `StateMachine`, `OpAmp` (with rail saturation), `MovingAverageFilter`
- Signal: `Limiter`, `DelayBlock` (Hysteresis already provided by existing `Comparator`)
- Modulation: `PwmGenerator`, `SpaceVectorModulator`
- Transforms: `ClarkeTransform`, `InverseClarkeTransform`, `ParkTransform`, `InverseParkTransform`
- Synchronization: `PLL`
- Routing: `SignalMux`, `SignalDemux`

#### Scenario: PI controller authored in Python is equivalent to YAML-loaded PI

- **GIVEN** a closed-loop buck where the controller is a single `pulsim.v2.PIController` instance with `Kp=0.05, Ki=200`
- **WHEN** the same circuit is also expressed as a YAML netlist containing a `type: pi_controller` block with the same parameters
- **THEN** both circuits SHALL produce simulation results matching within numerical precision (sub-millivolt on V_out for a 5 ms buck transient)

#### Scenario: Clarke-Park transform chain in YAML

- **GIVEN** a YAML netlist with the chain `clarke_transform → park_transform → pi_controller → inverse_park_transform → inverse_clarke_transform`
- **WHEN** loaded via `p.load_yaml_file(...)` and simulated
- **THEN** the dq-frame current SHALL track a step reference within 5 % steady-state error after settling
- **AND** the same chain authored in Python via `ClarkeTransform()`, `ParkTransform()`, etc. SHALL produce identical results

### Requirement: DC Operating Point Strategies

Pulsim v2 SHALL provide three numerical strategies to find a DC operating point for circuits where the naive direct solve fails: Gmin ramp, source-stepping, and pseudo-transient continuation. These SHALL be exposed via a single `compute_dc_op_with_strategy(graph, pool, mask, options)` entry point with a `DCStrategy` enum selector. The current `compute_dc_op` (snapshot of transient at `t_eval`) MUST remain available unchanged.

#### Scenario: Gmin ramp resolves a non-convergent stiff diode bridge

- **GIVEN** a 6-diode bridge rectifier where the direct Newton solve at zero initial state diverges
- **WHEN** `compute_dc_op_with_strategy(..., strategy=DCStrategy::GminRamp)` is called
- **THEN** Newton SHALL converge to a physically reasonable DC operating point
- **AND** the resulting `x_op` SHALL be usable as a starting state for `run_transient`

### Requirement: MNA-based AC Sweep

Pulsim v2 SHALL provide a fast linearised AC sweep that takes a DC operating point `x_op` and computes the closed-form frequency response `H(jω) = output / input` by solving `(jωE − A) X = B u` at each sampled frequency. This SHALL coexist with the existing swept-sine `run_ac_sweep` (which works for any plant including strongly nonlinear ones).

The MNA-based sweep SHALL:
- Linearise the cached MNA matrix at `x_op` by querying each nonlinear device for its `∂I/∂V` Jacobian
- Identify the descriptor mass matrix `E` from the trap-companion contributions of capacitors and inductors
- Solve the resulting generalised eigenproblem per frequency using the existing KLU backend (complex factorisation)
- Return complex `H(jω)` matching the format of `run_ac_sweep`

#### Scenario: MNA sweep matches swept-sine on a buck plant

- **GIVEN** a buck converter at `D_op = 0.5`, linearised around the DC operating point
- **WHEN** both `run_mna_sweep(...)` and `run_ac_sweep(...)` are evaluated on the same frequency grid 100 Hz → 10 kHz
- **THEN** the two results SHALL agree within 0.5 dB magnitude and 2° phase across the entire range
- **AND** `run_mna_sweep` SHALL complete at least 50× faster than `run_ac_sweep` for an equivalent point count

### Requirement: v1-to-v2 Parity Roadmap

A roadmap document SHALL exist that enumerates every v1 feature missing from v2, grouped into five sequenced phases (A: solver + virtual blocks, B: convergence, C: domain models, D: motors, E: tooling). Each future phase SHALL be implementable as a separate OpenSpec proposal without requiring this proposal to be re-opened.

#### Scenario: Future-phase items are documented

- **WHEN** a reader opens `openspec/changes/add-pulsim-v2-v1-parity-roadmap/tasks.md`
- **THEN** they SHALL find a complete listing of Phase B/C/D/E items with rough effort estimates (in person-weeks)
- **AND** each future phase SHALL be marked as unchecked, indicating it is tracked but not yet implemented
