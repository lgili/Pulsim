## ADDED Requirements

### Requirement: Integrator Selection Enum

`SimulationOptions::Advanced::Timestep` SHALL expose an `Integrator` enum with four values: `Tustin`, `Bdf1`, `DormandPrince54`, and `RadauIIA3`. The simulator SHALL dispatch the per-step solve to the configured integrator. The default SHALL remain `Tustin` paired with `TransientStepMode::Fixed` to preserve byte-identical behaviour for users who do not opt in.

#### Scenario: Default integrator is Tustin fixed-step

- **GIVEN** a user constructs a `SimulationOptions` without touching the `advanced.timestep.integrator` field
- **WHEN** the simulator runs
- **THEN** the selected integrator is `Tustin`
- **AND** `step_mode` defaults to `Fixed`
- **AND** the simulation output matches the pre-change result byte-for-byte on a regression circuit

#### Scenario: Variable-step DOPRI5 selection

- **GIVEN** a user sets `opts.step_mode = Variable` and `opts.advanced.timestep.integrator = DormandPrince54`
- **WHEN** the simulator runs
- **THEN** the kernel uses the Dormand-Prince 5(4) integrator
- **AND** the existing LTE-based step controller adjusts `dt` based on the integrator's embedded error estimate

### Requirement: Dormand-Prince 5(4) Integrator

The library SHALL provide a 7-stage FSAL explicit Runge-Kutta integrator with the standard Dormand-Prince coefficients, producing both a 5th-order solution and a 4th-order embedded error estimate per step. The integrator SHALL expose a 4th-order Hermite-style dense-output interpolant computed from the stage values already evaluated during the step.

#### Scenario: DOPRI5 matches scipy oracle on Lorenz system

- **GIVEN** the Lorenz attractor with classic parameters (σ=10, ρ=28, β=8/3) integrated from `t=0` to `t=10` from a fixed initial condition
- **WHEN** the user selects `Integrator::DormandPrince54` with `rtol = 1e-8`, `atol = 1e-10`
- **THEN** the trajectory matches `scipy.integrate.solve_ivp(method='RK45', rtol=1e-8, atol=1e-10)` within 1e-5 relative L2 norm
- **AND** the step controller produces wall-clock at most 2× the scipy reference

#### Scenario: DOPRI5 dense output for event localisation

- **GIVEN** a half-wave rectifier with a diode whose forward-current zero crossing occurs analytically at `t_zc`
- **WHEN** the simulator runs with `Integrator::DormandPrince54` and event-driven step trimming enabled
- **THEN** the detected diode-off event time matches `t_zc` within 1 ns regardless of the integrator's accepted step length

#### Scenario: DOPRI5 speedup on PSFB regression

- **GIVEN** the PSFB ZVS benchmark
- **WHEN** the simulator runs once with fixed-step Tustin at the validated `dt = 10 ns` and once with variable-step DOPRI5 at `rtol = 1e-6`
- **THEN** the DOPRI5 wall-clock is at least 5× faster
- **AND** both runs report the same ZVS fraction within 1 percentage point

### Requirement: Radau IIA(3) Implicit Integrator

The library SHALL provide a 2-stage implicit Runge-Kutta integrator with the Radau IIA coefficients (order 3, L-stable, A-stable). The integrator SHALL reuse the existing MNA sparse Jacobian backend for Newton iteration and SHALL surface Newton iteration count back to the step controller via the existing Newton-feedback hook.

#### Scenario: Radau handles stiff RC discharge in few steps

- **GIVEN** an RC circuit with `RC = 1 µs` discharging from 100 V over 10 ms
- **WHEN** the simulator runs with `Integrator::RadauIIA3` and `rtol = 1e-6`
- **THEN** the controller accepts ≤ 100 steps over the 10 ms span
- **AND** the final voltage matches the analytical exponential within 0.5 %

#### Scenario: Radau Newton failure shrinks the step

- **GIVEN** a circuit where the Jacobian briefly becomes near-singular (e.g. a switch transition)
- **WHEN** the Newton iteration inside `RadauIIA3` fails to converge within `max_iter`
- **THEN** the step is rejected and the step controller shrinks `dt` by the configured Newton-feedback factor
- **AND** the next attempt resumes from the same accepted state without integrator state corruption

#### Scenario: Radau preserves dense output for events

- **GIVEN** the same half-wave rectifier scenario used for DOPRI5
- **WHEN** `Integrator::RadauIIA3` is selected
- **THEN** the detected diode-off event time matches the analytical zero-crossing within 1 ns
