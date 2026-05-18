# dc-operating-point Specification

## Purpose
TBD - created by archiving change improve-convergence-algorithms. Update Purpose after archive.
## Requirements
### Requirement: Multi-Strategy DC Solver

The system SHALL implement a multi-strategy DC solver that automatically tries different convergence algorithms in sequence until the circuit reaches DC operating point.

The solver SHALL support the following strategies in configurable order:
1. **Newton-Raphson**: Standard Newton-Raphson with voltage limiting
2. **GMIN Stepping**: Add conductance to ground, reduce exponentially
3. **Source Stepping**: Ramp sources from 0 to final value
4. **Pseudo-Transient**: Use transient simulation to find DC

#### Scenario: Simple circuit converges with Newton

- **GIVEN** a circuit with linear and weakly nonlinear components
- **WHEN** `dc_operating_point_robust()` is called
- **THEN** the solver converges using Newton-Raphson strategy
- **AND** the result indicates `strategy_used = DCStrategy::Newton`

#### Scenario: Complex circuit uses GMIN stepping fallback

- **GIVEN** a power electronics circuit with multiple MOSFETs in cutoff
- **WHEN** Newton-Raphson fails to converge within max iterations
- **THEN** the solver automatically falls back to GMIN stepping
- **AND** GMIN stepping achieves convergence
- **AND** the final solution is verified without GMIN

#### Scenario: Source stepping for feedback circuits

- **GIVEN** a circuit with strong internal feedback (e.g., op-amp)
- **WHEN** both Newton and GMIN stepping fail
- **THEN** the solver uses source stepping
- **AND** sources are ramped from 0% to 100% in steps
- **AND** convergence is achieved at full source values

#### Scenario: Pseudo-transient for stiff circuits

- **GIVEN** a circuit where all other strategies fail
- **WHEN** Newton, GMIN stepping, and source stepping fail
- **THEN** the solver runs a short pseudo-transient simulation
- **AND** the final transient state is used as DC solution

### Requirement: GMIN Stepping Algorithm

The system SHALL implement GMIN stepping as a convergence aid for DC analysis.

GMIN stepping SHALL:
- Add small conductances (GMIN) from each node to ground
- Start with GMIN = 1e-3 S
- Reduce GMIN exponentially by factor of 10 after each convergence
- Continue until GMIN = 1e-12 S
- Perform final Newton solve without GMIN to verify solution

#### Scenario: GMIN sequence progression

- **GIVEN** a circuit that requires GMIN stepping
- **WHEN** GMIN stepping is initiated
- **THEN** the GMIN sequence follows [1e-3, 1e-4, 1e-5, ..., 1e-12]
- **AND** convergence is attempted at each GMIN level
- **AND** the algorithm progresses only after successful convergence

#### Scenario: GMIN stepping with convergence failure at intermediate step

- **GIVEN** a circuit with GMIN stepping in progress at level 1e-6
- **WHEN** Newton fails to converge at current GMIN level
- **THEN** the algorithm increases GMIN to previous level
- **AND** attempts intermediate GMIN values
- **AND** eventually finds converging path or reports failure

#### Scenario: Final solution without GMIN

- **GIVEN** GMIN stepping has converged at minimum GMIN (1e-12)
- **WHEN** the final Newton solve is performed
- **THEN** GMIN is completely removed from the system
- **AND** the solution is verified to converge without GMIN
- **AND** the returned solution represents the true DC operating point

### Requirement: Source Stepping Algorithm

The system SHALL implement source stepping as a convergence aid for DC analysis.

Source stepping SHALL:
- Save original values of all independent sources
- Scale all sources by a factor from 0.0 to 1.0
- Start at factor 0.0 (all sources off)
- Increment factor in steps of 0.1
- Perform Newton solve at each step
- Use solution from previous step as initial guess

#### Scenario: Successful source ramp-up

- **GIVEN** a circuit requiring source stepping
- **WHEN** source stepping is initiated
- **THEN** all voltage and current sources are scaled to 0
- **AND** Newton converges at each scale factor [0.0, 0.1, 0.2, ..., 1.0]
- **AND** the final solution is at full source values

#### Scenario: Adaptive step insertion on failure

- **GIVEN** source stepping at factor 0.5 has converged
- **WHEN** Newton fails to converge at factor 0.6
- **THEN** the algorithm inserts intermediate step at 0.55
- **AND** continues with finer steps until convergence
- **AND** eventually reaches factor 1.0

#### Scenario: Source restoration after completion

- **GIVEN** source stepping has completed successfully
- **WHEN** the DC solution is returned
- **THEN** all sources are restored to their original values
- **AND** the circuit state reflects full source values

### Requirement: GMIN Floor Conductance

The system SHALL add a minimum floor conductance (GMIN floor) to all nodes to prevent floating node numerical issues.

#### Scenario: GMIN floor applied to voltage nodes

- **GIVEN** a circuit with N voltage nodes
- **WHEN** the MNA system is assembled
- **THEN** a conductance of 1e-12 S is added to diagonal entries G(i,i)
- **AND** this applies to all nodes except the ground reference

#### Scenario: Floating node stabilization

- **GIVEN** a circuit with a node connected only through capacitors
- **WHEN** DC analysis is performed
- **THEN** the GMIN floor prevents singular matrix
- **AND** the node voltage is determined by leakage currents
- **AND** the solution converges successfully

### Requirement: DC Options Configuration

The system SHALL provide a `DCOptions` structure to configure DC analysis behavior.

DCOptions SHALL include:
- `strategy_order`: List of strategies to try (default: Newton, GMIN, Source, Pseudo)
- `max_iterations`: Maximum Newton iterations per attempt (default: 100)
- `tolerance`: Convergence tolerance (default: 1e-9)
- `voltage_limiting`: Enable device voltage limiting (default: true)
- `gmin_floor`: Minimum floor conductance (default: 1e-12)

#### Scenario: Custom strategy order

- **GIVEN** DCOptions with `strategy_order = [SourceStepping, Newton]`
- **WHEN** `dc_operating_point_robust(options)` is called
- **THEN** source stepping is tried first
- **AND** Newton is tried only if source stepping fails

#### Scenario: Disabled voltage limiting

- **GIVEN** DCOptions with `voltage_limiting = false`
- **WHEN** DC analysis is performed
- **THEN** device voltage limiting is not applied
- **AND** Newton may take larger voltage steps per iteration

#### Scenario: Custom tolerance

- **GIVEN** DCOptions with `tolerance = 1e-12`
- **WHEN** DC analysis is performed
- **THEN** convergence requires residual norm < 1e-12
- **AND** solution is more accurate but may require more iterations

### Requirement: Collapsed Public DC Strategy Enum

The system SHALL expose exactly two values in its public-facing
`DCStrategy` enumeration: `Auto` (default) and `Override`.

The internal 5-value implementation enum (`Direct`, `GminStepping`,
`SourceStepping`, `PseudoTransient`, `Homotopy`) SHALL remain
accessible under `opts.advanced.dc.strategy_override` for power
users who need to force a specific strategy.

#### Scenario: Default Auto orchestrates the full ladder

- **GIVEN** a user accepts the default `DCStrategy::Auto`
- **WHEN** Newton on the first attempt (Direct) fails to converge
- **THEN** the orchestrator falls back through SourceStepping →
  GminStepping → PseudoTransient → Homotopy until convergence or
  the full ladder is exhausted

#### Scenario: Override forces a single strategy

- **GIVEN** a user sets
  `opts.advanced.dc.strategy_override = DCStrategyImpl::PseudoTransient`
  with no fallback
- **WHEN** the DC analysis runs
- **THEN** only PseudoTransient is attempted
- **AND** failure of PseudoTransient surfaces as a fatal DC error
  without trying any other strategy

### Requirement: Homotopy Continuation as Last Resort

The system SHALL implement homotopy continuation as the fifth and
final strategy in the `DCStrategy::Auto` fallback ladder.

The homotopy SHALL step a parameter `λ` from `0` to `1` in
configurable increments. At `λ = 0`, all nonlinear devices (diodes,
MOSFETs, IGBTs, BJTs, behavioural switches) SHALL be replaced by
their linear off-state conductance (`g_off`), producing a fully
linear MNA system solvable in a single direct solve. At `λ = 1`,
the full nonlinear device models SHALL be active.

Each step in the ladder SHALL solve Newton-Raphson with the previous
λ's solution as warm-start. The default ladder length SHALL be 5
increments; `Preset::HighFidelity` SHALL use 10 increments.

#### Scenario: Cold-start 3-level NPC succeeds via homotopy

- **GIVEN** a 3-level NPC converter at cold start where Direct,
  SourceStepping, GminStepping, and PseudoTransient all fail
- **WHEN** the `DCStrategy::Auto` orchestrator reaches homotopy
- **THEN** the λ ladder runs from 0 to 1 in 5 steps
- **AND** each step solves Newton in ≤ 10 iterations using the prior
  λ's solution as warm-start
- **AND** the final DC operating point is established
- **AND** `homotopy_ladder_completed == true` is reported

#### Scenario: Simple linear circuit never invokes homotopy

- **GIVEN** a passive RC circuit
- **WHEN** the DC analysis runs with `DCStrategy::Auto`
- **THEN** Direct converges on iteration 1
- **AND** homotopy is never attempted
- **AND** `homotopy_steps == 0` is reported

#### Scenario: HighFidelity preset uses longer ladder

- **GIVEN** a user calls `from_preset(Preset::HighFidelity, ...)`
- **WHEN** the DC analysis ladder reaches homotopy
- **THEN** the ladder runs in 10 increments instead of 5

#### Scenario: Homotopy disabled via advanced override

- **GIVEN** a user sets `opts.advanced.dc.homotopy.enable = false`
- **WHEN** `DCStrategy::Auto` exhausts the first four strategies
- **THEN** homotopy is skipped
- **AND** the DC analysis fails with `DcConvergenceFailure` rather
  than attempting homotopy

### Requirement: Strategy Telemetry

The system SHALL report which DC strategy successfully produced the
operating point in `result.dc_result.strategy_used`, plus per-strategy
timing and iteration counts.

The `strategy_used` field SHALL accept any of the five internal
strategy values: `Direct`, `GminStepping`, `SourceStepping`,
`PseudoTransient`, `Homotopy`.

Additionally, the system SHALL report:
- `homotopy_steps` — number of λ increments executed (0 if homotopy
  not invoked)
- `homotopy_ladder_completed` — true if homotopy reached `λ = 1`,
  false if it gave up partway
- `total_dc_newton_iterations` — sum across all strategies attempted

#### Scenario: Telemetry reports the winning strategy

- **GIVEN** a DC analysis that succeeded via SourceStepping after
  Direct failed
- **WHEN** the user inspects `result.dc_result`
- **THEN** `strategy_used == DCStrategyImpl::SourceStepping`
- **AND** `total_dc_newton_iterations` reflects the sum of Direct's
  failed-iteration count plus SourceStepping's successful iterations

#### Scenario: Telemetry reports homotopy progress

- **GIVEN** a DC analysis that succeeded via homotopy after 5 λ steps
- **WHEN** the user inspects `result.dc_result`
- **THEN** `strategy_used == DCStrategyImpl::Homotopy`
- **AND** `homotopy_steps == 5`
- **AND** `homotopy_ladder_completed == true`

