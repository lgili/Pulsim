# Linear Solver Optimization

## ADDED Requirements

### Requirement: KLU Default Solver

The system SHALL use KLU (SuiteSparse) as the default linear solver when available, falling back to Eigen SparseLU otherwise.

KLU selection SHALL:
- Auto-detect KLU availability at compile time
- Use KLU when `PULSIM_HAS_KLU` is defined
- Fall back to Eigen SparseLU when KLU is unavailable
- Allow explicit backend selection via options

#### Scenario: Auto-selection with KLU available

- **GIVEN** PulsimCore compiled with SuiteSparse/KLU
- **WHEN** LinearSolver is created with Backend::Auto
- **THEN** KLU backend is selected
- **AND** log indicates "Using KLU sparse solver"

#### Scenario: Auto-selection without KLU

- **GIVEN** PulsimCore compiled without SuiteSparse
- **WHEN** LinearSolver is created with Backend::Auto
- **THEN** Eigen SparseLU backend is selected
- **AND** log indicates "Using Eigen SparseLU solver"

#### Scenario: Explicit Eigen selection

- **GIVEN** LinearSolverOptions with backend = Backend::Eigen
- **WHEN** LinearSolver is created
- **THEN** Eigen SparseLU is used regardless of KLU availability

#### Scenario: Explicit KLU selection without KLU

- **GIVEN** LinearSolverOptions with backend = Backend::KLU
- **WHEN** PulsimCore is compiled without KLU
- **THEN** an error is raised or fallback to Eigen with warning

### Requirement: Symbolic Factorization Caching

The system SHALL cache and reuse symbolic matrix factorization across multiple numeric solves.

Symbolic caching SHALL:
- Perform symbolic factorization once per matrix structure
- Reuse symbolic factorization for subsequent numeric solves
- Detect when matrix structure changes and refactorize
- Limit maximum reuses to prevent numerical degradation

#### Scenario: Initial symbolic factorization

- **GIVEN** a new MNA matrix for first solve
- **WHEN** LinearSolver.solve() is called
- **THEN** symbolic factorization is performed
- **AND** result is cached for future solves

#### Scenario: Numeric solve with cached symbolic

- **GIVEN** cached symbolic factorization from previous solve
- **WHEN** matrix values change but structure is same
- **THEN** only numeric factorization is performed
- **AND** symbolic factorization is reused
- **AND** solve time is reduced by ~30-50%

#### Scenario: Structure change triggers refactorization

- **GIVEN** cached symbolic factorization
- **WHEN** matrix structure changes (new nonzeros)
- **THEN** structure change is detected
- **AND** new symbolic factorization is performed
- **AND** cache is updated

#### Scenario: Maximum reuse limit

- **GIVEN** symbolic factorization reused 50 times
- **WHEN** 51st solve is requested
- **THEN** symbolic factorization is refreshed
- **AND** this prevents numerical drift

### Requirement: Condition Number Monitoring

The system SHALL monitor matrix condition number to detect ill-conditioned systems.

Condition monitoring SHALL:
- Estimate condition number during solve
- Log warning for condition number > 1e10
- Trigger symbolic refactorization when condition degrades significantly
- Provide condition number in solve result

#### Scenario: Well-conditioned matrix

- **GIVEN** a circuit matrix with condition number ~1e6
- **WHEN** solve is performed
- **THEN** no warnings are logged
- **AND** solve completes normally

#### Scenario: Ill-conditioned matrix warning

- **GIVEN** a circuit matrix with condition number > 1e10
- **WHEN** solve is performed
- **THEN** warning is logged: "Matrix is ill-conditioned"
- **AND** solve continues but may have reduced accuracy

#### Scenario: Condition degradation refactorization

- **GIVEN** initial condition number 1e6, current estimate 1e9
- **WHEN** condition has degraded by factor > 100
- **THEN** symbolic refactorization is triggered
- **AND** condition number is recomputed

### Requirement: Pivot Tolerance Configuration

The system SHALL support configurable pivot tolerance for numeric stability.

#### Scenario: Default pivot tolerance

- **GIVEN** LinearSolverOptions with default settings
- **WHEN** linear system is solved
- **THEN** pivot tolerance of 1e-13 is used
- **AND** very small pivots are treated as zero

#### Scenario: Custom pivot tolerance

- **GIVEN** LinearSolverOptions with pivot_tolerance = 1e-10
- **WHEN** linear system is solved
- **THEN** pivots smaller than 1e-10 are treated as zero
- **AND** this may improve stability for some circuits

### Requirement: Linear Solver Options Structure

The system SHALL provide `LinearSolverOptions` for configuring solver behavior.

LinearSolverOptions SHALL include:
- `backend`: Auto, Eigen, or KLU (default: Auto)
- `reuse_symbolic`: Enable symbolic caching (default: true)
- `max_symbolic_reuses`: Maximum reuses before refresh (default: 50)
- `pivot_tolerance`: Minimum pivot value (default: 1e-13)
- `condition_warning_threshold`: Log warning above this (default: 1e10)

#### Scenario: Disable symbolic reuse

- **GIVEN** LinearSolverOptions with reuse_symbolic = false
- **WHEN** multiple solves are performed
- **THEN** symbolic factorization is performed for each solve
- **AND** this is slower but may be more accurate

#### Scenario: Increased reuse limit

- **GIVEN** LinearSolverOptions with max_symbolic_reuses = 100
- **WHEN** symbolic is reused 100 times
- **THEN** refactorization occurs at 101st solve
- **AND** longer reuse reduces overhead

### Requirement: Matrix Structure Change Detection

The system SHALL detect when the sparsity pattern of the matrix changes.

#### Scenario: Sparsity pattern unchanged

- **GIVEN** two consecutive MNA matrices with same structure
- **WHEN** structure comparison is performed
- **THEN** matrices are identified as structurally equivalent
- **AND** symbolic factorization is reused

#### Scenario: New nonzero element

- **GIVEN** MNA matrix with new switch closing (new connection)
- **WHEN** structure comparison is performed
- **THEN** new nonzero pattern is detected
- **AND** symbolic refactorization is triggered

#### Scenario: Removed element handling

- **GIVEN** switch opening removes matrix element
- **WHEN** structure comparison is performed
- **THEN** smaller structure is still compatible with cached symbolic
- **AND** numeric solve can proceed with zero in removed position

## ADDED Requirements

### Requirement: AdvancedLinearSolver Enhancement

The existing AdvancedLinearSolver SHALL be enhanced with the new optimization features.

#### Scenario: AdvancedLinearSolver with KLU

- **GIVEN** AdvancedLinearSolver configured with Backend::Auto
- **WHEN** KLU is available
- **THEN** KLU is used for all solves
- **AND** symbolic caching is enabled by default

#### Scenario: Backward compatibility

- **GIVEN** existing code using AdvancedLinearSolver
- **WHEN** no options are specified
- **THEN** behavior is compatible with previous version
- **AND** new optimizations are applied transparently
