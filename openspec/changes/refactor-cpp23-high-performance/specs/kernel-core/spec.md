## ADDED Requirements

### Requirement: C++23 Standard Compliance
The kernel SHALL be implemented using C++23 standard with Clang 17+ as the primary compiler.

#### Scenario: Compilation with C++23 features
- **WHEN** the project is compiled with Clang 17+
- **THEN** all C++23 features (std::expected, deducing this, constexpr improvements) SHALL compile without errors

#### Scenario: Cross-platform compilation
- **WHEN** the project is compiled on Linux, macOS, or Windows
- **THEN** compilation SHALL succeed with the specified C++23 compiler

### Requirement: Static Polymorphism for Device Models
All device models SHALL use CRTP (Curiously Recurring Template Pattern) for zero-overhead static dispatch.

#### Scenario: Device stamping performance
- **WHEN** a device stamps into the MNA matrix
- **THEN** no virtual function calls SHALL be made
- **AND** the compiler SHALL be able to inline the stamp operation

#### Scenario: Compile-time Jacobian pattern
- **WHEN** a circuit is defined with typed devices
- **THEN** the Jacobian sparsity pattern SHALL be computed at compile time

### Requirement: Expression Templates for Matrix Operations
Matrix operations SHALL use expression templates for lazy evaluation and fusion.

#### Scenario: Matrix expression evaluation
- **WHEN** multiple matrix operations are chained (A * B + C)
- **THEN** no intermediate temporaries SHALL be allocated
- **AND** the expression SHALL be evaluated in a single pass

### Requirement: Memory Pool Allocation
Simulation hot paths SHALL use arena allocators for zero-allocation operation.

#### Scenario: Timestep execution
- **WHEN** a simulation timestep is executed
- **THEN** no heap allocations SHALL occur in the hot path
- **AND** memory SHALL be obtained from a pre-allocated pool

#### Scenario: Memory pool reset
- **WHEN** a simulation completes
- **THEN** the memory pool SHALL be reset in O(1) time

### Requirement: SIMD Optimization
Matrix assembly and device evaluation SHALL be SIMD-optimized where beneficial.

#### Scenario: AVX2 vectorization
- **WHEN** compiled on x86-64 with AVX2 support
- **THEN** hot loops SHALL use AVX2 SIMD instructions

#### Scenario: ARM NEON vectorization
- **WHEN** compiled on ARM64
- **THEN** hot loops SHALL use NEON SIMD instructions

### Requirement: Correct Trapezoidal Integration
The Trapezoidal (GEAR-2) integration method SHALL correctly include history terms for reactive elements.

#### Scenario: Capacitor companion model accuracy
- **WHEN** simulating an RC circuit with Trapezoidal integration
- **THEN** the equivalent current source SHALL include both (2C/dt)*V_{n-1} AND I_{n-1}
- **AND** simulation results SHALL match analytical solution within 0.1%

#### Scenario: Inductor companion model accuracy
- **WHEN** simulating an RL circuit with Trapezoidal integration
- **THEN** the equivalent voltage source SHALL include both (2L/dt)*I_{n-1} AND V_{n-1}
- **AND** simulation results SHALL match analytical solution within 0.1%

### Requirement: BDF Integration Methods
The solver SHALL support BDF methods of orders 1-5 with automatic order selection.

#### Scenario: BDF2 accuracy
- **WHEN** simulating with BDF2 method
- **THEN** the local truncation error SHALL be O(dt^3)
- **AND** results SHALL match analytical solution within 0.1%

#### Scenario: Automatic order selection
- **WHEN** adaptive_order is enabled
- **THEN** the solver SHALL automatically select the optimal BDF order based on error estimates

### Requirement: Newton Solver Convergence
The Newton solver SHALL correctly report convergence status after the final iteration.

#### Scenario: Final step convergence check
- **WHEN** the maximum number of iterations is reached
- **THEN** the solver SHALL evaluate the residual one more time
- **AND** if residual < abstol, status SHALL be Success

#### Scenario: Armijo line search
- **WHEN** a Newton step would increase the residual
- **THEN** the solver SHALL perform backtracking line search
- **AND** the step size SHALL satisfy the Armijo condition

### Requirement: Adaptive Timestep Control
The simulation SHALL support error-based adaptive timestep control with PI controller.

#### Scenario: Timestep increase
- **WHEN** the local truncation error is well below tolerance
- **THEN** the timestep SHALL be increased (up to dtmax)

#### Scenario: Timestep decrease
- **WHEN** the local truncation error exceeds tolerance
- **THEN** the step SHALL be rejected and timestep reduced

#### Scenario: Event handling
- **WHEN** a switching event is detected
- **THEN** the timestep SHALL be reduced to accurately capture the event

### Requirement: Convergence Aids
The solver SHALL provide multiple convergence aid strategies for difficult circuits.

#### Scenario: Gmin stepping
- **WHEN** initial DC operating point fails to converge
- **THEN** Gmin stepping SHALL be attempted automatically

#### Scenario: Source stepping
- **WHEN** Gmin stepping fails
- **THEN** source stepping (continuation) SHALL be attempted

#### Scenario: Pseudo-transient
- **WHEN** source stepping fails
- **THEN** pseudo-transient continuation SHALL be attempted as last resort

### Requirement: Policy-Based Solver Configuration
Solver components SHALL be configurable through compile-time policies.

#### Scenario: Linear solver policy
- **WHEN** the user specifies KLU linear solver
- **THEN** the KLU implementation SHALL be used without runtime branching

#### Scenario: Integration method policy
- **WHEN** the user specifies Trapezoidal integration
- **THEN** the Trapezoidal implementation SHALL be instantiated at compile time

### Requirement: Validation Test Coverage
All simulation results SHALL be validated against analytical solutions and SPICE references.

#### Scenario: Analytical validation
- **WHEN** running the validation test suite
- **THEN** all circuits SHALL produce results within 0.1% of analytical solutions

#### Scenario: SPICE comparison
- **WHEN** comparing against ngspice results
- **THEN** all circuits SHALL produce equivalent results (accounting for numerical differences)

### Requirement: Performance Benchmarks
The new implementation SHALL demonstrate measurable performance improvement.

#### Scenario: Speedup verification
- **WHEN** running benchmark suite
- **THEN** simulation time SHALL be at least 2x faster than current implementation

#### Scenario: Memory efficiency
- **WHEN** running benchmark suite
- **THEN** peak memory usage SHALL not exceed 1.5x the circuit data size

## MODIFIED Requirements

### Requirement: Device Model Interface
Device models SHALL implement a unified interface with compile-time dispatch.

#### Scenario: Device stamping
- **WHEN** a device is stamped into the MNA matrix
- **THEN** the stamp_impl() method SHALL be called via static dispatch
- **AND** the device SHALL provide jacobian_pattern_impl() as constexpr

#### Scenario: Device parameter validation
- **WHEN** a device is created with parameters
- **THEN** parameters SHALL be validated at construction time
- **AND** invalid parameters SHALL result in a compile-time or immediate runtime error

### Requirement: Error Handling
Error conditions SHALL be reported using std::expected<T, Error>.

#### Scenario: Simulation error
- **WHEN** a simulation encounters an error
- **THEN** std::unexpected with error details SHALL be returned
- **AND** no exceptions SHALL be thrown in the hot path

#### Scenario: Error propagation
- **WHEN** a function returns an error
- **THEN** the calling function SHALL propagate or handle the error explicitly
