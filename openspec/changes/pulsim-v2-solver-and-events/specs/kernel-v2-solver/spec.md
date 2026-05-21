## ADDED Requirements

### Requirement: SimulationOptions — Fixed-dt Time-Stepping Inputs

`pulsim::v2::solver::SimulationOptions` SHALL be a value-type
aggregate that holds the time-stepping inputs for `run_transient`:

```cpp
struct SimulationOptions {
    Real t_start = 0;
    Real t_end   = 0;
    Real dt      = 0;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] Size expected_step_count() const noexcept;
};
```

The `valid()` method SHALL return `true` if and only if all
three fields are finite, `dt > 0`, and `t_end > t_start`.

The `expected_step_count()` method SHALL return the number of
output samples that `run_transient` will record for valid options,
computed as `floor((t_end - t_start) / dt) + 1`. For invalid
options the return value is unspecified.

#### Scenario: Default options are invalid

- **GIVEN** a default-constructed `SimulationOptions`
- **WHEN** the user calls `opts.valid()`
- **THEN** the result SHALL be `false` (default `dt = 0`).

#### Scenario: Valid options compute the right step count

- **GIVEN** `opts{t_start=0, t_end=1, dt=0.1}`
- **WHEN** the user calls `opts.valid()` and
  `opts.expected_step_count()`
- **THEN** `valid()` SHALL return `true`
- **AND** `expected_step_count()` SHALL return `11`
  (samples at t = 0, 0.1, 0.2, …, 1.0).

#### Scenario: Negative dt is invalid

- **GIVEN** `opts{t_start=0, t_end=1, dt=-0.01}`
- **WHEN** the user calls `opts.valid()`
- **THEN** the result SHALL be `false`.

#### Scenario: t_end ≤ t_start is invalid

- **GIVEN** `opts{t_start=1, t_end=1, dt=0.1}` and
  `opts2{t_start=2, t_end=1, dt=0.1}`
- **WHEN** the user calls `valid()` on each
- **THEN** both SHALL return `false`.

#### Scenario: NaN inputs are invalid

- **GIVEN** `opts{t_start=NaN, t_end=1, dt=0.1}`
- **WHEN** the user calls `opts.valid()`
- **THEN** the result SHALL be `false`.

### Requirement: SimulationResult — Time-Series Output Container

`pulsim::v2::solver::SimulationResult` SHALL be a value-type
aggregate holding the recorded transient output:

```cpp
struct SimulationResult {
    std::vector<Real>   times;
    std::vector<Vector> states;

    [[nodiscard]] Size num_steps() const noexcept;
    [[nodiscard]] bool empty()     const noexcept;
    void reserve(Size n);
};
```

The two parallel vectors MUST always satisfy
`times.size() == states.size()` at the end of `run_transient`;
`states[k]` is the solution at `times[k]`.

The `num_steps()` method SHALL return `times.size()`.

The `reserve(n)` method SHALL call `reserve(n)` on both internal
vectors so callers can pre-allocate without re-allocation
overhead.

#### Scenario: Default result is empty

- **GIVEN** a default-constructed `SimulationResult`
- **WHEN** the user calls `result.empty()` and
  `result.num_steps()`
- **THEN** `empty()` SHALL return `true`
- **AND** `num_steps()` SHALL return `0`.

#### Scenario: Reserve does not change size

- **GIVEN** a default-constructed `SimulationResult`
- **WHEN** the user calls `result.reserve(100)`
- **THEN** `result.num_steps()` SHALL still equal `0`
- **AND** `result.empty()` SHALL still return `true`.

### Requirement: run_transient — Fixed-dt Transient Simulation Loop

`pulsim::v2::solver::run_transient` SHALL be the V0 entry point
for time-stepping a pre-built PWL cache:

```cpp
using SwitchScheduleFn =
    std::function<topology::SwitchStateMask(Real)>;
using BExtraFn = std::function<Vector(Real)>;

SimulationResult run_transient(
    const pwl::PwlStateSpaceCache& cache,
    Size state_size,
    const SimulationOptions& opts,
    const SwitchScheduleFn& switch_fn,
    const BExtraFn& b_extra_fn = {});
```

The function MUST:

1. Throw `std::invalid_argument` if `!opts.valid()`.
2. Throw `std::invalid_argument` if `state_size == 0`.
3. Throw `std::invalid_argument` if `!switch_fn` (the schedule
   callback is required).
4. Pre-allocate the output via
   `result.reserve(opts.expected_step_count())`.
5. Initialise `Vector x = Vector::Zero(state_size)`.
6. Iterate `k = 0 .. expected_step_count() - 1`:
   a. Compute `t = opts.t_start + k * opts.dt` (integer counter
      to avoid floating-point drift).
   b. Compute `mask = switch_fn(t)`.
   c. If `b_extra_fn` is non-empty, compute
      `b_extra = b_extra_fn(t)`; otherwise use a pre-allocated
      zero buffer of size `state_size`.
   d. Call `cache.solve(mask, b_extra, x)`.
   e. Append `(t, x)` to `result` (a copy of `x` is stored).
7. Return the populated `SimulationResult`.

The function MUST NOT mutate any of its input arguments. The
returned `SimulationResult` MUST own its data and be safely
returnable by value.

#### Scenario: Invalid options throw

- **GIVEN** a valid cache and `opts{dt=-1}` (or any other
  configuration that fails `opts.valid()`)
- **WHEN** the user calls `run_transient`
- **THEN** the call SHALL throw `std::invalid_argument`.

#### Scenario: Zero state_size throws

- **GIVEN** a valid cache, valid `opts`, and `state_size == 0`
- **WHEN** the user calls `run_transient`
- **THEN** the call SHALL throw `std::invalid_argument`.

#### Scenario: Empty switch_fn throws

- **GIVEN** a valid cache, valid `opts`, a positive `state_size`,
  and a default-constructed `SwitchScheduleFn`
- **WHEN** the user calls `run_transient`
- **THEN** the call SHALL throw `std::invalid_argument`.

#### Scenario: Step count matches expected_step_count

- **GIVEN** valid inputs with
  `opts{t_start=0, t_end=1e-3, dt=1e-6}`
- **WHEN** the user calls `run_transient` successfully
- **THEN** `result.num_steps()` SHALL equal
  `opts.expected_step_count()` (1001 samples).

#### Scenario: Times grid is t_start + k · dt

- **GIVEN** valid inputs with `opts{t_start=0, t_end=1, dt=0.1}`
- **WHEN** the user calls `run_transient` successfully
- **THEN** for every `k` in `[0, num_steps)`,
  `result.times[k]` SHALL equal `opts.t_start + k * opts.dt`
  exactly (the implementation uses an integer counter, not
  accumulated `t += dt`).

#### Scenario: switch_fn is consulted every step

- **GIVEN** a cache built for 1-switch circuit
- **AND** a `switch_fn` that returns `mask{open}` for
  `t < t_mid` and `mask{closed}` for `t >= t_mid`
- **WHEN** the user runs the simulation
- **THEN** the state vectors for `times[k] < t_mid` SHALL
  match the analytical "open" solution
- **AND** the state vectors for `times[k] >= t_mid` SHALL
  match the analytical "closed" solution.

#### Scenario: b_extra_fn is consulted every step when supplied

- **GIVEN** valid inputs and a `b_extra_fn(t)` that returns a
  known time-varying vector (e.g., a sinusoidal disturbance
  on the source constraint row)
- **WHEN** the user runs the simulation
- **THEN** the state-vector samples SHALL reflect the
  contribution of `b_extra_fn(t)` at every step (verified by
  reproducing the expected analytical answer for a known
  time-varying source).

### Requirement: Chopper Integration Test — 10 kHz PWM Validation

The Layer 5 V0 OpenSpec MUST include an integration test that
runs a chopper circuit (V_dc → Switch → R → GND) under a 10 kHz
PWM schedule with `duty = 0.5` and validates the result.

The test MUST:

1. Build the chopper graph + device pool (V_dc = 12 V,
   g_on = 1e3, g_off = 1e-9, G_R = 0.1).
2. Build the PWL cache.
3. Define a PWM schedule `mask(t)` with period `T = 100 µs` and
   duty `0.5`.
4. Call `run_transient` with `t_start = 0`, `t_end = 1 ms`,
   `dt = 1 µs` (1001 steps, 10 full PWM periods).
5. Verify `result.num_steps() == 1001`.
6. Verify the mean of `v_out(t)` across the full record equals
   `V_dc · duty = 6.0 V` within `< 1 %`.
7. Verify the waveform shape — every sampled `v_out[k]` lies
   within `1e-6` of either the analytical ON value
   (`V_dc · g_on / (g_on + G_R)`) or the analytical OFF value
   (`V_dc · g_off / (g_off + G_R)`), depending on the PWM
   schedule at that step.

#### Scenario: 10 kHz PWM produces mean v_out = V_dc · duty

- **GIVEN** the chopper circuit with `V_dc=12`, `duty=0.5`
- **WHEN** the user runs the 1 ms / 1 µs simulation
- **THEN** `mean(v_out[k] across k=0..N-1)` SHALL be within
  `1 %` of `6.0 V`.

#### Scenario: 10 kHz PWM produces a clean square wave

- **GIVEN** the chopper circuit and PWM schedule from the
  previous scenario
- **WHEN** the user inspects every sample `v_out[k]`
- **THEN** each sample SHALL lie within `1e-6` of either the
  analytical ON value `V_dc · g_on / (g_on + G_R)` or the
  analytical OFF value `V_dc · g_off / (g_off + G_R)`,
  matching the PWM schedule's state at `times[k]`.
