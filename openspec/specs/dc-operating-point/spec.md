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

