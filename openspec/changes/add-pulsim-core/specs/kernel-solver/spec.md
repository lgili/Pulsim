## ADDED Requirements

### Requirement: Linear System Solver
The kernel SHALL solve sparse linear systems Ax = b efficiently.

#### Scenario: Direct solver
- **WHEN** a linear system is solved
- **THEN** LU factorization with partial pivoting is used
- **AND** SuiteSparse (KLU) or Eigen is the backend

#### Scenario: Factorization caching
- **WHEN** multiple solves use the same matrix structure
- **THEN** symbolic factorization is reused
- **AND** only numeric factorization is repeated

#### Scenario: Singular matrix handling
- **WHEN** the matrix is singular or near-singular
- **THEN** the solver reports the condition and problematic node
- **AND** suggests potential circuit issues (floating node, short circuit)

### Requirement: Newton-Raphson Iteration
The kernel SHALL solve nonlinear systems using Newton-Raphson iteration.

#### Scenario: Basic iteration
- **WHEN** nonlinear components are present
- **THEN** the system iterates: solve J*dx = -f, x = x + dx
- **AND** continues until convergence criteria are met

#### Scenario: Convergence criteria
- **WHEN** checking convergence
- **THEN** both absolute (|dx| < abstol) and relative (|dx/x| < reltol) tolerances are checked
- **AND** default abstol = 1e-12, reltol = 1e-3

#### Scenario: Damping
- **WHEN** Newton step causes divergence
- **THEN** damping is applied: x = x + alpha*dx where 0 < alpha <= 1
- **AND** alpha is adjusted to ensure residual reduction

#### Scenario: Iteration limit
- **WHEN** iteration count exceeds maxiter (default 50)
- **THEN** the solver reports non-convergence
- **AND** provides the current residual and iteration history

### Requirement: Convergence Aids
The kernel SHALL provide convergence aids for difficult circuits.

#### Scenario: Gmin stepping
- **WHEN** Newton fails to converge for DC operating point
- **THEN** a conductance Gmin is added from each node to ground
- **AND** Gmin is gradually reduced until the original circuit converges

#### Scenario: Source stepping
- **WHEN** DC analysis fails to converge
- **THEN** sources are ramped from 0 to their final value
- **AND** previous solution is used as initial guess for each step

#### Scenario: Pseudo-transient continuation
- **WHEN** other methods fail
- **THEN** a pseudo time-step is used to march toward DC solution
- **AND** equivalent capacitors damp the Newton iteration

### Requirement: Time Integration
The kernel SHALL integrate differential equations in time using implicit methods.

#### Scenario: Backward Euler integration
- **WHEN** using Backward Euler (default for stability)
- **THEN** dy/dt is approximated as (y_n - y_{n-1}) / dt
- **AND** the method is first-order accurate, L-stable

#### Scenario: Trapezoidal integration
- **WHEN** using Trapezoidal rule
- **THEN** dy/dt = 2*(y_n - y_{n-1})/dt - dy/dt_{n-1}
- **AND** the method is second-order accurate, A-stable

#### Scenario: Gear integration (BDF)
- **WHEN** using Gear 2nd order (BDF2)
- **THEN** multi-step formula is used for stiff systems
- **AND** the method is second-order accurate, L-stable

#### Scenario: Variable order method
- **WHEN** SUNDIALS IDA/CVODE is selected
- **THEN** variable order BDF (1-5) is used
- **AND** order is adjusted based on error estimates

### Requirement: Adaptive Timestep
The kernel SHALL support adaptive timestep control.

#### Scenario: Error-based adaptation
- **WHEN** adaptive timestep is enabled
- **THEN** the local truncation error is estimated
- **AND** timestep is adjusted to keep error within tolerance

#### Scenario: Timestep limits
- **WHEN** computing the next timestep
- **THEN** dt is bounded by dtmin and dtmax
- **AND** default dtmin = 1e-15, dtmax = tstop/10

#### Scenario: Event-triggered reduction
- **WHEN** a switching event is approaching
- **THEN** timestep is reduced to hit the event precisely
- **AND** steps back if event was missed

### Requirement: Initial Conditions
The kernel SHALL support various initial condition methods.

#### Scenario: DC operating point
- **WHEN** no initial conditions are specified
- **THEN** a DC analysis finds the operating point
- **AND** this is used as the initial state for transient

#### Scenario: User-specified IC
- **WHEN** initial voltages/currents are specified
- **THEN** these values initialize the state vector
- **AND** overrides are applied before first timestep

#### Scenario: Zero IC
- **WHEN** uic (use initial conditions) flag is set
- **THEN** all states start at zero (or specified IC)
- **AND** no DC operating point is computed

### Requirement: DC Analysis
The kernel SHALL perform DC operating point analysis.

#### Scenario: DC operating point
- **WHEN** DC analysis is requested
- **THEN** capacitors are open, inductors are shorted
- **AND** the nonlinear algebraic system is solved

#### Scenario: DC sweep
- **WHEN** DC sweep is requested with parameter and range
- **THEN** DC analysis is repeated for each parameter value
- **AND** results include the swept parameter

### Requirement: AC Analysis
The kernel SHALL perform small-signal AC analysis.

#### Scenario: AC frequency sweep
- **WHEN** AC analysis is requested with frequency range
- **THEN** the circuit is linearized at the DC operating point
- **AND** complex impedance matrix is solved at each frequency

#### Scenario: Bode plot data
- **WHEN** AC analysis completes
- **THEN** magnitude (dB) and phase (degrees) are available
- **AND** for specified output/input node pairs
