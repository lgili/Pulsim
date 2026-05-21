## ADDED Requirements

### Requirement: High-Level `simulate` Python Wrapper

The Python `pulsim.v2` module SHALL expose a `simulate(builder, t_end, dt, **kwargs)` function that auto-wires the cache, simulation options, and Newton refresh, returning the `SimulationResult` in a single call.

#### Scenario: One-liner transient on a simple RC
- **WHEN** a user writes `simulate(builder, t_end=1e-3, dt=1e-5)`
- **THEN** the function SHALL automatically create a `PwlStateSpaceCache`, build `SimulationOptions`, default `switch_fn` to an all-OFF mask, and return the `SimulationResult`

#### Scenario: Auto-detect nonlinear devices
- **WHEN** a user calls `simulate(...)` on a circuit containing a MOSFET or IGBT (Nonlinear branches)
- **THEN** the wrapper SHALL automatically pass `enable_nonlinear_refresh=True` to `run_transient`
- **AND** Newton convergence SHALL succeed without the user explicitly specifying the flag

#### Scenario: Lift SimulationOptions kwargs
- **WHEN** a user passes `simulate(builder, t_end, dt, max_newton_iterations=200, tol_newton_dx=1e-6)`
- **THEN** those kwargs SHALL be forwarded into `SimulationOptions` before the `run_transient` call
