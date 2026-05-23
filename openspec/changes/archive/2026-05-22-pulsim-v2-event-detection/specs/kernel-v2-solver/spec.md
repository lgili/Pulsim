## ADDED Requirements

### Requirement: SimulationOptions — Event Iteration Limit

`SimulationOptions` SHALL include a `max_event_iterations` field
(default `16`) that bounds how many times each step can re-solve
to converge on a self-consistent diode state. Setting it to `0`
disables the iteration (matching Layer 5 V2 behaviour).

The field SHALL NOT affect `valid()` — any non-negative value is
acceptable.

#### Scenario: Default max_event_iterations is 16

- **GIVEN** a default-constructed `SimulationOptions`
- **WHEN** the user reads `opts.max_event_iterations`
- **THEN** the value SHALL equal `16`.

#### Scenario: Custom value preserved

- **GIVEN** `SimulationOptions{.dt=1e-6, .t_end=1e-3,
  .max_event_iterations=4}`
- **WHEN** the user reads the field
- **THEN** the value SHALL equal `4`
- **AND** `valid()` SHALL still return `true`.

### Requirement: SimulationResult — Event Iteration Diagnostics

`SimulationResult` SHALL include a `std::vector<Size>
event_iteration_count` parallel to `times` and `states`.
Element k counts how many cache.solve calls step k needed to
reach a consistent diode state. Zero means "the first solve was
already consistent" (no commutation, or commutation matched the
previous step's state).

`reserve(n)` SHALL also reserve this vector.

#### Scenario: Default-constructed event_iteration_count is empty

- **GIVEN** a default-constructed `SimulationResult`
- **WHEN** the user reads `result.event_iteration_count.empty()`
- **THEN** the result SHALL be `true`.

#### Scenario: reserve sizes all three vectors

- **GIVEN** a default `SimulationResult`
- **WHEN** the user calls `reserve(100)`
- **THEN** the capacity of all three internal vectors
  (`times`, `states`, `event_iteration_count`) SHALL be at
  least `100`.

### Requirement: run_transient — Event-Iteration Loop

`run_transient` SHALL iterate the diode state to consistency
when the circuit contains any `SwitchedDiode`. The iteration MUST:

1. Solve via `cache.solve` with the current mask.
2. Call `diodes.update_from_state(x)`.
3. If any diode flipped AND iteration count < max, re-solve.
4. Otherwise, record the sample with the final mask.

If the loop hits `max_event_iterations` without converging, the
function SHALL throw `std::runtime_error` with a message
identifying the simulation time at which divergence occurred.

#### Scenario: Diode-free circuit has zero event iterations

- **GIVEN** a Layer 5 V1 circuit with no diodes (e.g., the RC
  charging test or the synchronous buck)
- **WHEN** the user runs `run_transient`
- **THEN** `result.event_iteration_count[k]` SHALL be `0` for
  every k.

#### Scenario: Boost converter steady state matches analytical

- **GIVEN** a boost converter with V_in = 12 V, L = 100 µH,
  C = 100 µF, R_load = 20 Ω, PWM at 100 kHz, D = 0.5
- **WHEN** the user runs a 10 ms simulation
- **THEN** the mean V_out over the last 1 ms SHALL be within
  10 % of `V_in / (1 − D) = 24 V`
- **AND** `max(event_iteration_count)` SHALL be ≤ 4
- **AND** no step SHALL have hit the iteration limit.
