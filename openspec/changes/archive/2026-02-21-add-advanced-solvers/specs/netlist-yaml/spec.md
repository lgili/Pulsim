## ADDED Requirements

### Requirement: Solver Order Configuration
The YAML netlist SHALL allow specifying primary and fallback solver orders.

#### Scenario: Separate orders
- **WHEN** `simulation.solver.order` and `simulation.solver.fallback_order` are provided
- **THEN** the parser SHALL map them to distinct primary and fallback orders

### Requirement: Advanced Solver Options
The YAML netlist SHALL allow configuration for JFNK, preconditioners, and stiff integrators.

#### Scenario: JFNK in YAML
- **WHEN** the netlist enables `simulation.solver.nonlinear.jfnk`
- **THEN** the solver SHALL use the JFNK path

#### Scenario: Preconditioner selection
- **WHEN** the netlist sets `solver.iterative.preconditioner` to `ilut` or `amg`
- **THEN** the parser SHALL accept the value or error with a clear diagnostic if unavailable

#### Scenario: Stiff integrator selection
- **WHEN** the netlist sets `simulation.integration` (or `simulation.integrator`) to `tr-bdf2` or `rosenbrock`
- **THEN** the parser SHALL apply the selected integrator

#### Scenario: Periodic steady-state options
- **WHEN** the netlist sets `simulation.shooting` or `simulation.harmonic_balance` (`simulation.hb`)
- **THEN** the parser SHALL map the options to periodic steady-state configuration

#### Scenario: Residual cache tuning
- **WHEN** the netlist sets `simulation.newton.krylov_residual_cache_tolerance`
- **THEN** the parser SHALL apply the specified residual cache tolerance
