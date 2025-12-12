## Gates & Definition of Done

- [ ] G.1 Analytical accuracy: RC/RL/RLC (underdamped/critical/overdamped) <0.1% vs analytical; diode/rectifier within tolerance envelope
- [ ] G.2 SPICE parity: ngspice comparison suite <=0.1% RMS error (aligned timestep, same init)
- [ ] G.3 Performance: >=2x speedup vs current main on benchmark suite; peak memory <=1.5x circuit data size
- [ ] G.4 Determinism: solver produces repeatable results with fixed seed/order on CPU targets (x86-64, arm64)
- [ ] G.5 Compatibility: v1 API shim passes smoke tests (CLI + Python) with representative netlists
- [ ] G.6 CI coverage: accuracy, perf, memory regression checks gated in CI (nightly perf allowed, but fail on regressions > threshold)
- [ ] G.7 Docs: migration guide v1->v2, perf tuning guide, updated notebooks/examples and README

## Phase 1: C++23 Foundation & Build System

### 1.1 Build System Configuration
- [x] 1.1.1 Update CMakeLists.txt for C++23 standard
- [x] 1.1.2 Add Clang 17+ detection and requirement (toolchain-clang.cmake)
- [x] 1.1.3 Configure LTO (Link-Time Optimization) for Release builds
- [x] 1.1.4 Add PGO (Profile-Guided Optimization) support
- [x] 1.1.5 Configure sanitizers (ASan, UBSan, TSan) for debug builds
- [x] 1.1.6 Add compile-time benchmark targets
- [x] 1.1.7 Configure parallel compilation (ninja generator)
- [x] 1.1.8 Update CI for Clang 17+ on all platforms

### 1.2 Modern C++23 Features Integration
- [x] 1.2.1 Replace std::optional error handling with std::expected
- [x] 1.2.2 Implement Result<T, E> type alias for consistent error handling
- [x] 1.2.3 Add std::mdspan for multidimensional state arrays
- [x] 1.2.4 Use deducing this for CRTP base classes
- [x] 1.2.5 Add constexpr to all compile-time evaluable functions
- [x] 1.2.6 Implement concepts for device model constraints
- [x] 1.2.7 Add static reflection preparation (for future C++26)

### 1.3 Project Structure Reorganization
- [x] 1.3.1 Create `core/include/pulsim/v2/` for new API
- [x] 1.3.2 Define new namespace `pulsim::v2`
- [x] 1.3.3 Create type traits header `type_traits.hpp`
- [x] 1.3.4 Create concepts header `concepts.hpp`
- [x] 1.3.5 Create compile-time utilities `constexpr_utils.hpp`
- [x] 1.3.6 Add backward compatibility shim header
- [x] 1.3.7 Define rollout flagging (feature macro / namespace alias) to enable v1/v2 side-by-side

## Phase 2: Template Metaprogramming Architecture

### 2.1 Type System Foundation
- [x] 2.1.1 Define `Real` as template parameter (double/float)
- [x] 2.1.2 Define `Index` type with configurable width
- [x] 2.1.3 Create `StaticVector<T, N>` for fixed-size vectors
- [x] 2.1.4 Create `StaticMatrix<T, Rows, Cols>` for small matrices
- [x] 2.1.5 Implement `SparsityPattern<N>` for compile-time patterns
- [x] 2.1.6 Add `Units` type for dimensional analysis
- [x] 2.1.7 Add normalization/scaling helpers for mixed units (volt/amp) to stabilize solvers

### 2.1A Vertical Slice (MVP) - RC/RL correctness + perf
- [ ] 2.1A.1 Fix Trapezoidal capacitor/inductor with history (3.2.1/3.2.2/3.2.3) for RC/RL
- [ ] 2.1A.2 Implement LTE + PI controller minimal path for RC/RL (3.4.x + 3.5.x reduced scope)
- [ ] 2.1A.3 Wire arena allocator + SoA layout for RC/RL device data (4.3.x/4.5.x minimal)
- [ ] 2.1A.4 Add analytical tests RC/RL + perf benchmark (small/medium) to prove <0.1% + speedup
- [ ] 2.1A.5 Document findings and adjust defaults (dtmin/dtmax/safety factors)

### 2.2 CRTP Device Architecture
- [x] 2.2.1 Create `DeviceBase<Derived>` CRTP base class
- [x] 2.2.2 Define device concept `IsDevice<T>` (StampableDevice)
- [x] 2.2.3 Implement `stamp()` with static dispatch
- [x] 2.2.4 Implement `jacobian_pattern()` as constexpr
- [x] 2.2.5 Add compile-time pin count validation
- [x] 2.2.6 Create device trait `device_traits<T>` for introspection

### 2.3 CRTP Device Implementations
- [x] 2.3.1 Implement `Resistor` with CRTP
- [x] 2.3.2 Implement `Capacitor` with CRTP and correct companion model
- [x] 2.3.3 Implement `Inductor` with CRTP and correct companion model
- [x] 2.3.4 Implement `VoltageSource` with CRTP
- [x] 2.3.5 Implement `CurrentSource` with CRTP
- [x] 2.3.6 Implement `Diode` with CRTP (ideal + Shockley)
- [x] 2.3.7 Implement `Switch` with CRTP
- [x] 2.3.8 Implement `MOSFET` with CRTP (Level 1)
- [x] 2.3.9 Implement `IGBT` with CRTP
- [x] 2.3.10 Implement `Transformer` with CRTP
- [x] 2.3.11 Unit tests for all CRTP devices

### 2.4 Expression Templates
- [x] 2.4.1 Create `Expression<Op, Lhs, Rhs>` base template
- [x] 2.4.2 Implement `AddExpr`, `SubExpr`, `MulExpr`, `ScaleExpr`
- [x] 2.4.3 Implement lazy evaluation with `eval()` method
- [x] 2.4.4 Add SIMD-optimized evaluation kernel
- [ ] 2.4.5 Benchmark against Eigen expressions
- [x] 2.4.6 Provide fallback to Eigen expressions and toggle to compare correctness/perf

### 2.5 Compile-Time Circuit Analysis
- [x] 2.5.1 Create `CircuitGraph<Devices...>` variadic template
- [x] 2.5.2 Implement compile-time node counting
- [x] 2.5.3 Implement compile-time branch counting
- [x] 2.5.4 Generate static Jacobian sparsity pattern
- [x] 2.5.5 Validate circuit topology at compile time
- [x] 2.5.6 Add static/dynamic cross-check (compile-time pattern vs runtime assembled pattern) in tests

## Phase 3: Precision & Numerical Robustness

### 3.1 Fix DC Operating Point Convergence
- [x] 3.1.1 Add residual evaluation after final Newton step (solver.cpp)
- [x] 3.1.2 Update default abstol to 1e-9 (SPICE standard) (types.hpp, solver.hpp, advanced_solver.hpp)
- [x] 3.1.3 Implement weighted norm for mixed voltage/current
- [x] 3.1.4 Add per-variable convergence checking
- [x] 3.1.5 Implement convergence history tracking
- [x] 3.1.6 Unit tests for DC OP edge cases
- [ ] 3.1.7 Add deterministic ordering for assembly/solves to ensure repeatability

### 3.2 Fix Trapezoidal Integration
- [x] 3.2.1 Add `i_prev` to capacitor companion model
- [x] 3.2.2 Add `v_prev` to inductor companion model
- [x] 3.2.3 Implement correct Trapezoidal coefficients
- [x] 3.2.4 Add state history storage for reactive elements
- [x] 3.2.5 Validate against analytical RC/RL/RLC solutions
- [x] 3.2.6 Achieve <0.1% error on standard test circuits
- [x] 3.2.7 Add clamps/limits to prevent overflow/underflow in reactive updates

### 3.3 BDF Methods Implementation
- [x] 3.3.1 Implement BDF1 (Backward Euler) with correct formulation
- [x] 3.3.2 Implement BDF2 with proper coefficients
- [x] 3.3.3 Implement BDF3-5 for higher accuracy
- [ ] 3.3.4 Add automatic order selection based on error
- [x] 3.3.5 Implement startup sequence (BDF1 -> BDF2 -> target)
- [ ] 3.3.6 Add order reduction on convergence failure

### 3.4 Local Truncation Error Estimation
- [x] 3.4.1 Implement LTE for Trapezoidal method
- [x] 3.4.2 Implement LTE for BDF methods
- [x] 3.4.3 Add per-state-variable error tracking
- [x] 3.4.4 Implement error-based timestep prediction
- [x] 3.4.5 Add safety factor configuration
- [ ] 3.4.6 Add logging hook to export LTE metrics for debugging (guarded, off by default)

### 3.5 Adaptive Timestep Controller
- [x] 3.5.1 Implement PI controller for timestep
- [x] 3.5.2 Add dtmin/dtmax enforcement
- [x] 3.5.3 Implement step rejection with halving
- [x] 3.5.4 Add event-aware timestep adjustment
- [x] 3.5.5 Implement timestep history for stability
- [x] 3.5.6 Unit tests for adaptive stepping
- [x] 3.5.7 Add configurables for controller gains and safety factors with documented defaults

## Phase 4: High-Performance Solvers

### 4.1 Linear Solver Optimization
- [x] 4.1.1 Create `LinearSolverPolicy` concept
- [x] 4.1.2 Implement `SparseLUPolicy` (Eigen wrapper)
- [ ] 4.1.3 Implement `KLUPolicy` (SuiteSparse wrapper)
- [ ] 4.1.4 Add factorization reuse detection
- [ ] 4.1.5 Implement symbolic analysis caching
- [ ] 4.1.6 Add pivot tolerance configuration
- [ ] 4.1.7 Benchmark linear solver policies
- [ ] 4.1.8 Add deterministic pivoting/path guarantees (within solver capabilities)

### 4.2 Newton Solver Template
- [x] 4.2.1 Create `NewtonSolver<LinearPolicy, ConvergencePolicy>` template
- [x] 4.2.2 Implement `BasicConvergencePolicy`
- [ ] 4.2.3 Implement `ArmijoLineSearchPolicy`
- [ ] 4.2.4 Implement `TrustRegionPolicy`
- [x] 4.2.5 Add iteration callback for debugging
- [x] 4.2.6 Implement damping schedule (start aggressive, relax)

### 4.3 Memory Optimization
- [ ] 4.3.1 Create `ArenaAllocator` with bump allocation
- [ ] 4.3.2 Implement per-simulation memory pool
- [ ] 4.3.3 Add aligned allocation for SIMD
- [ ] 4.3.4 Implement workspace reuse across timesteps
- [ ] 4.3.5 Profile and eliminate heap allocations in hot path
- [ ] 4.3.6 Add memory usage tracking
- [ ] 4.3.7 Add debug-mode poison/guards to detect overruns in arenas

### 4.4 SIMD Optimization
- [ ] 4.4.1 Detect SIMD capabilities at compile time
- [ ] 4.4.2 Implement SIMD matrix assembly kernels
- [ ] 4.4.3 Implement SIMD device evaluation
- [ ] 4.4.4 Add AVX2 specializations
- [ ] 4.4.5 Add AVX-512 specializations (optional)
- [ ] 4.4.6 Add ARM NEON specializations
- [ ] 4.4.7 Benchmark SIMD improvements
- [ ] 4.4.8 Add runtime fallback path for non-SIMD targets with identical results

### 4.5 Cache-Friendly Data Layout
- [ ] 4.5.1 Implement SoA (Structure of Arrays) for device data
- [ ] 4.5.2 Add cache line alignment (64 bytes)
- [ ] 4.5.3 Implement data prefetching hints
- [ ] 4.5.4 Profile cache miss rates
- [ ] 4.5.5 Optimize memory access patterns
- [ ] 4.5.6 Add layout fuzz test to ensure determinism and correctness across AoS/SoA toggles

## Phase 5: Advanced Convergence Aids

### 5.1 Gmin Stepping
- [ ] 5.1.1 Implement exponential Gmin schedule
- [ ] 5.1.2 Add Gmin to ground for all nodes
- [ ] 5.1.3 Implement automatic Gmin reduction
- [ ] 5.1.4 Add Gmin stepping as fallback strategy
- [ ] 5.1.5 Log/trace Gmin ramp parameters for debugging (optional)

### 5.2 Source Stepping
- [ ] 5.2.1 Implement source scaling from 0 to 1
- [ ] 5.2.2 Add continuation parameter tracking
- [ ] 5.2.3 Implement adaptive step size for continuation
- [ ] 5.2.4 Add source stepping as primary DC strategy
- [ ] 5.2.5 Define abort/rollback criteria and reporting for failed continuation

### 5.3 Pseudo-Transient Continuation
- [ ] 5.3.1 Implement pseudo-timestep for DC analysis
- [ ] 5.3.2 Add capacitor-to-ground for DC convergence
- [ ] 5.3.3 Implement automatic pseudo-dt adjustment
- [ ] 5.3.4 Integrate with Newton solver
- [ ] 5.3.5 Add safety clamps for pseudo-dt growth/shrink and logging hooks

### 5.4 Robust Initialization
- [ ] 5.4.1 Implement node voltage initialization heuristics
- [ ] 5.4.2 Add device-specific initial guess
- [ ] 5.4.3 Implement warm start from previous solution
- [ ] 5.4.4 Add randomized restart on convergence failure
- [ ] 5.4.5 Allow deterministic seeding for randomized restarts

## Phase 6: Validation & Benchmarking

### 6.1 Analytical Validation Suite
- [ ] 6.1.1 Create RC circuit tests (all time constants)
- [ ] 6.1.2 Create RL circuit tests
- [ ] 6.1.3 Create RLC circuit tests (underdamped, overdamped, critical)
- [ ] 6.1.4 Create diode rectifier tests
- [ ] 6.1.5 Create buck converter tests
- [ ] 6.1.6 Create boost converter tests
- [ ] 6.1.7 Create full-bridge inverter tests
- [ ] 6.1.8 All tests must pass with <0.1% error
- [ ] 6.1.9 Export metrics (error, max dev) as CSV/JSON for regression tracking

### 6.2 SPICE Reference Comparison
- [ ] 6.2.1 Set up ngspice reference runner
- [ ] 6.2.2 Create automated comparison framework
- [ ] 6.2.3 Compare 10 standard power electronics circuits
- [ ] 6.2.4 Document any deviations with justification
- [ ] 6.2.5 Align timestep/initial conditions and capture deltas in artifacts

### 6.3 Performance Benchmarks
- [ ] 6.3.1 Create benchmark suite with various circuit sizes
- [ ] 6.3.2 Measure simulation time vs ngspice
- [ ] 6.3.3 Measure simulation time vs PSIM (if available)
- [ ] 6.3.4 Profile hot paths with perf/vtune
- [ ] 6.3.5 Document performance characteristics
- [ ] 6.3.6 Achieve >2x speedup over current implementation
- [ ] 6.3.7 Store benchmark outputs (time/mem) per commit for regression detection
- [ ] 6.3.8 Add deterministic benchmark harness (fixed seeds/order)

### 6.4 Regression Testing
- [ ] 6.4.1 Add all validation tests to CI
- [ ] 6.4.2 Add performance regression detection
- [ ] 6.4.3 Add memory regression detection
- [ ] 6.4.4 Create automated nightly benchmarks
- [ ] 6.4.5 Add wave-shape regression (tolerance envelopes) for selected circuits
- [ ] 6.4.6 Gate merges on accuracy/perf regressions beyond threshold

## Phase 7: Python Integration

### 7.1 Binding Updates
- [ ] 7.1.1 Update pybind11 bindings for new API
- [ ] 7.1.2 Add compatibility layer for old API
- [ ] 7.1.3 Expose new solver configuration options
- [ ] 7.1.4 Add Python-side type hints
- [ ] 7.1.5 Update all Python tests
- [ ] 7.1.6 Add Python smoke tests for v1 shim (CLI + basic circuits)

### 7.2 Documentation
- [ ] 7.2.1 Document new API in docstrings
- [ ] 7.2.2 Create migration guide from v1 to v2
- [ ] 7.2.3 Update Jupyter notebook examples
- [ ] 7.2.4 Add performance tuning guide
- [ ] 7.2.5 Document deterministic/repro flags and logging hooks

## Phase 8: Cleanup & Release

### 8.1 Code Cleanup
- [ ] 8.1.1 Remove deprecated v1 code
- [ ] 8.1.2 Update all documentation
- [ ] 8.1.3 Final code review
- [ ] 8.1.4 Update README with new features
- [ ] 8.1.5 Confirm gates G.1-G.7 satisfied before release

### 8.2 Release Preparation
- [ ] 8.2.1 Update version to 2.0.0
- [ ] 8.2.2 Create changelog
- [ ] 8.2.3 Tag release
- [ ] 8.2.4 Publish to PyPI
