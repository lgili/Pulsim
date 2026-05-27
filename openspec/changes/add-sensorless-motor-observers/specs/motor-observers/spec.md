## ADDED Requirements

### Requirement: Sliding-Mode Observer (SMO) for PMSM

The library SHALL provide a `SlidingModeObserver` block that estimates rotor electrical position and mechanical speed for a PMSM using only stator-frame voltage commands and current measurements. The observer SHALL plug into `MixedDomainBlockChain` via the standard block interface (`.step(...)` + named input/output channels).

#### Scenario: SMO locks onto rotor angle within startup window

- **GIVEN** a `PmsmDevice` running at 50 % rated speed driven by a known voltage command
- **WHEN** the `SlidingModeObserver` is initialized with default gains and runs for at least 100 ms after the motor reaches steady-state speed
- **THEN** the angle error `|theta_hat − theta_true|` is below 5 ° electrical
- **AND** the estimated speed `omega_hat` is within 1 % of the true mechanical speed

#### Scenario: SMO tracks through a speed reversal

- **GIVEN** a closed-loop sensorless PMSM-FOC drive using SMO for the Park transform angle
- **WHEN** the speed reference is reversed from +25 % to −25 % of rated speed
- **THEN** the controller maintains commutation throughout (no loss-of-lock)
- **AND** the steady-state speed error after the reversal settles below 1 %

#### Scenario: SMO degrades gracefully at very low speed

- **GIVEN** the same PMSM-FOC sensorless drive
- **WHEN** the speed reference is reduced below 5 % of rated speed
- **THEN** the observer either continues to track within a documented degraded tolerance (e.g. 15 ° angle error) or surfaces a `low_speed_flag` output that downstream code can act on
- **AND** the drive does not produce numerical instability (no NaN, no torque oscillations exceeding 2× steady-state amplitude)

### Requirement: Flux-MRAS Observer for Induction Motor

The library SHALL provide a `FluxMRASObserver` block that estimates rotor speed of a squirrel-cage induction motor by comparing voltage-model and current-model rotor-flux estimates and driving the speed-adjustment loop. The observer SHALL conform to the same block-interface used by other `MixedDomainBlockChain` building blocks.

#### Scenario: MRAS tracks rotor speed within tolerance

- **GIVEN** an `InductionMotorDevice` operating between 10 % and 100 % rated speed under any load
- **WHEN** the `FluxMRASObserver` has been running for at least 1 second
- **THEN** the speed error `|omega_hat − omega_true|` stays below 2 % of rated speed across the full operating window

#### Scenario: MRAS startup from standstill

- **GIVEN** an `InductionMotorDevice` initially at standstill driven by a V/f-ramp open-loop start
- **WHEN** the observer is enabled and the closed loop switches over after the speed crosses the documented hand-off threshold (default 10 % rated)
- **THEN** the closed-loop transition completes without torque pulsation exceeding 50 % of rated torque
- **AND** the steady-state speed reaches the commanded reference within 200 ms of the switchover

### Requirement: Observer Wiring via BlockChain Conventions

Both observers SHALL expose a constructor with motor parameters (Rs, Ls / Lr / Lm, pole_pairs) plus tuning gains, a `step(...)` method matching the existing `BlockChain` block convention, named input channels (`v_alpha`, `v_beta`, `i_alpha`, `i_beta`) and named output channels (`theta_hat`, `omega_hat`, plus observer-specific extras), and a `reset()` method that returns the internal state to the documented initial condition.

#### Scenario: Observer wired into a closed-loop chain

- **GIVEN** an instance of either `SlidingModeObserver` or `FluxMRASObserver`
- **WHEN** the user adds it to a `MixedDomainBlockChain` via `chain.add("obs", obs, inputs={"v_alpha":..., "v_beta":..., "i_alpha":..., "i_beta":...}, output="theta_hat")`
- **THEN** the chain validates successfully (no unresolved-input errors)
- **AND** subsequent blocks can reference the observer outputs via the same `channel:theta_hat` mechanism used by every other chain block
