## ADDED Requirements

### Requirement: Control Graph Representation

The kernel SHALL represent a controller as a `ControlGraph` of typed
blocks whose signals are integer slots in one array and whose states
live in flat discrete and continuous arrays that are part of the solver
snapshot. Every public way of describing a controller (block chains,
closed-loop helpers, C blocks, YAML chains, Python callables) SHALL
build this one representation, and a block kind the kernel does not
implement SHALL be a build error, never a fallback to another
evaluator.

#### Scenario: One representation behind the helpers
- **GIVEN** a loop built with `bind_pi_to_switch` and the same loop
  built block by block
- **WHEN** both are simulated
- **THEN** their duty histories are bit-identical

#### Scenario: Unknown block
- **GIVEN** a chain containing a block kind with no kernel
  implementation
- **WHEN** the graph is built
- **THEN** an error names the block and no Python evaluator is used

#### Scenario: Controller state survives a snapshot
- **GIVEN** a run with a PI controller stopped in the middle of a
  period
- **WHEN** it is resumed from the snapshot
- **THEN** the continuation equals the uninterrupted run bit for bit

### Requirement: Sorted Schedule and Algebraic-Loop Detection

The graph SHALL be evaluated in a topological order of its
direct-feedthrough dependencies, outputs of all blocks first and state
updates second, so that the result does not depend on the order in
which blocks were added. A cycle of direct feedthrough SHALL be refused
when the graph is built, with an error that names the blocks of the
cycle.

#### Scenario: Insertion order does not matter
- **GIVEN** a `Sum → Gain → Limiter` chain added in order and in
  reverse
- **WHEN** both are simulated
- **THEN** the results are bit-identical

#### Scenario: Algebraic loop
- **GIVEN** a `Gain` whose output feeds a `Sum` that feeds the same
  `Gain`
- **WHEN** the graph is built
- **THEN** the error names both blocks and suggests a `DelayBlock`
- **AND** with a `DelayBlock` in the loop the graph builds and runs

### Requirement: Block Rates and Exact Tick Instants

Every block group SHALL have a rate — discrete with a period and an
offset, continuous, or triggered by an event. A discrete group SHALL
execute at `offset + k·period` for integer `k`, at the same instants on
every engine, with `dt = period` passed to its blocks. A group SHALL
accept a declared computational delay of zero, a time, or one sample,
which defers when its outputs take effect.

#### Scenario: Tick count is engine-independent
- **GIVEN** a 10 kHz controller and a 20 ms run
- **WHEN** it is simulated on each engine
- **THEN** the controller executes exactly 200 times (198 on the fixed
  engine before this change)

#### Scenario: Two rates
- **GIVEN** a 2 kHz PI and a 20 kHz monitor in one 20 ms run
- **WHEN** it is simulated on the variable-step engine
- **THEN** the PI executes 40 times and the monitor 400 times (the PI
  executed 400 times before, moving the output from 13.03 V to 12.09 V)

#### Scenario: Discrete loop against its recurrence
- **GIVEN** a P controller (`K_p = 2`, 100 µs) driving an RC (τ = 1 ms)
  through a zero-order-hold source, set-point 1 V
- **WHEN** it is simulated with the exact strategy
- **THEN** the capacitor voltage at the tenth tick is 0.643545722992 V
  within 1e-9
- **AND** with a one-sample computational delay it is 0.659695654069 V

### Requirement: Continuous Control States in the Solver

Blocks of the continuous rate SHALL be integrated by the circuit solver
as part of the same implicit system, with an exact Jacobian, so that a
continuous controller has no step of lag against the circuit. A linear
continuous subgraph on a piecewise-linear circuit SHALL be advanced by
the exact propagator together with the circuit states.

#### Scenario: Continuous PI on an RC
- **GIVEN** a continuous PI (`K_p = 1`, `K_i = 2000 s⁻¹`) closing a loop
  around an RC (τ = 1 ms), unit step
- **WHEN** it is simulated with the exact strategy
- **THEN** `v(0.5 ms) = 0.467719269784` and `v(1 ms) = 0.801233889654`
  within 1e-9 (`v = 1 − e^{−1000t}·cos 1000t`)

#### Scenario: Result does not depend on the step
- **GIVEN** the same loop on the fixed engine, with the restart rule of
  `add-discontinuity-restart-rule`, at dt = 1e-6 and 1e-7
- **WHEN** both runs are compared with the closed form at 1 ms
- **THEN** the error ratio is at least 50 (second order gives 100; a
  one-step controller lag gives 10)

### Requirement: Control Outputs to the Circuit

The graph SHALL act on the circuit only through declared sinks: a
modulator whose duty register a block writes and whose edges are
computed analytically, a controlled source, and a gate driven by a
comparator. A comparator of the continuous rate SHALL switch at a
located state event; a discrete one SHALL switch at its ticks.

#### Scenario: Realised duty
- **GIVEN** a closed-loop buck whose PI writes a PWM modulator
- **WHEN** it is simulated on a variable-step engine
- **THEN** the realised duty differs from the commanded one by less
  than 1 ppm and the number of gate events equals the analytic edge
  count

#### Scenario: Hysteresis current control
- **GIVEN** an RL load (10 V, 1 Ω, 1 mH) with an ideal freewheel path
  and a continuous comparator holding 5 A ± 0.5 A
- **WHEN** it reaches its limit cycle
- **THEN** the rise and fall times are 200.6707 µs within 1e-9 s and the
  current ripple is 1.000 A within 1e-9

### Requirement: C Export of the Control Graph

The graph SHALL export its discrete-rate groups as one C99 translation
unit with a state structure, an initialiser and one step function per
rate, using no dynamic allocation. For a single-rate graph the export
SHALL also implement the C-block ABI. Export SHALL be refused, naming
the block, when the graph contains a Python block, a C block without
source or a continuous-rate block.

#### Scenario: Round trip
- **GIVEN** a PI, a limiter and a PWM block at one rate
- **WHEN** the export is compiled with `-ffp-contract=off` and loaded
  back through `add_c_block`
- **THEN** its outputs equal the in-kernel graph bit for bit over 10⁴
  ticks

#### Scenario: Strict C
- **WHEN** the export is compiled with `-std=c99 -Wall -Werror
  -pedantic`
- **THEN** it compiles and contains no call to `malloc`

#### Scenario: Opaque block
- **GIVEN** a graph containing a Python block
- **WHEN** `export_c` is called
- **THEN** it raises an error naming that block

### Requirement: Control Evaluation Cost

A graph SHALL be evaluated only at the ticks of its groups, and a run
of a graph with no Python block SHALL make no Python call inside the
step loop. The result SHALL expose the number of block evaluations per
group.

#### Scenario: Evaluations are counted per tick
- **GIVEN** a 10 kHz controller of five blocks on a fixed run with
  dt = 1e-7 over 20 ms
- **WHEN** the run completes
- **THEN** the group reports 1000 block evaluations (200 ticks × 5) and
  zero Python calls
