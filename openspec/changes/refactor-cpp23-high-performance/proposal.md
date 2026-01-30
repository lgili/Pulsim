## Why

PulsimCore has achieved functional completeness but suffers from several architectural limitations that prevent it from competing with industry-standard simulators like PSIM, PLECS, and LTSpice. Current issues include:

1. **Numerical Precision**: DC operating point convergence bugs, Trapezoidal integration accuracy issues (~8% error), incorrect companion model implementations
2. **Performance**: No use of modern C++ compile-time optimizations, inefficient runtime polymorphism, suboptimal matrix operations
3. **Architecture**: Monolithic solver design makes it hard to add new integration methods, tightly coupled components
4. **Robustness**: Many edge cases cause convergence failures, inadequate error recovery mechanisms

A ground-up refactoring using C++23 with template metaprogramming will deliver a simulator that matches or exceeds PSIM performance and accuracy.

**Dependency**: This change builds on `unify-v1-core` (single v1 kernel + YAML netlist + unified simulator pipeline).

## What Changes

### **BREAKING** - Complete Kernel Rewrite

#### Phase 1: C++23 Foundation & Build System
- Migrate from C++20 to C++23 with Clang 17+
- Enable modules support for faster compilation
- Configure LTO (Link-Time Optimization) and PGO (Profile-Guided Optimization)
- Add SIMD intrinsics support (AVX2/AVX-512, ARM NEON)

#### Phase 2: Template Metaprogramming Architecture
- Static polymorphism for device models (CRTP pattern)
- Compile-time circuit graph analysis
- Expression templates for matrix operations
- Constexpr evaluation of stamp patterns
- Policy-based design for solver configuration

#### Phase 3: Precision & Numerical Robustness
- Fix DC operating point convergence (residual check after final Newton step)
- Fix Trapezoidal integration (correct companion model with history terms)
- Implement BDF2/GEAR with proper coefficient handling
- Add adaptive order selection (BDF1-5)
- Implement proper local truncation error estimation
- Add automatic timestep control with PI controller

#### Phase 4: High-Performance Solvers
- Template-based Newton solver with compile-time Jacobian sparsity
- SIMD-optimized matrix assembly
- Memory pool allocators for zero-allocation hot paths
- Cache-friendly data layout (SoA vs AoS)
- Parallel device evaluation with work stealing

#### Phase 5: Advanced Convergence
- Gmin stepping with exponential ramp
- Source stepping with continuation
- Pseudo-transient analysis
- Automatic damping with Armijo line search
- Trust region methods for difficult circuits

#### Phase 6: Validation & Benchmarking
- Comprehensive test suite against analytical solutions
- SPICE reference comparisons (ngspice, LTSpice)
- Performance benchmarks vs PSIM/PLECS
- Automated regression testing

### Cross-Cutting: Reliability & Compatibility
- Deterministic execution guarantees on x86-64/arm64 with fixed seeds and ordered reductions
- Observability: machine-readable artifacts for accuracy/perf/memory regressions stored per run
- Backward-compat shim for v1 API/CLI/Python routed through v2 core

## Impact

### Affected Capabilities (Complete Rewrite)
- `kernel-mna` - MNA assembly with template metaprogramming
- `kernel-solver` - Newton/linear solvers with static polymorphism
- `kernel-devices` - Device models with CRTP
- `kernel-events` - Event handling with compile-time dispatch
- `kernel-losses` - Loss calculation with expression templates
- `kernel-thermal` - Thermal coupling with cache-friendly layout

### Affected Code
- `core/include/pulsim/*.hpp` - Complete API redesign
- `core/src/*.cpp` - Implementation rewrite
- `python/` - Binding updates for new API
- `CMakeLists.txt` - C++23 configuration

### Migration Path
- Python API will maintain backward compatibility where possible
- JSON netlist format unchanged
- New API alongside old during transition period

### Risk Mitigation
- Incremental phases allow validation at each step
- Comprehensive test suite prevents regression
- Performance benchmarks ensure improvements

## Success Criteria

1. **Accuracy**: All circuits match analytical/SPICE results within 0.1%
2. **Performance**: 2-5x speedup over current implementation
3. **Robustness**: Zero convergence failures on standard test circuits
4. **Compilation**: Full build < 30 seconds with modules
5. **Determinism**: Repeatable results across runs on the same hardware class (x86-64/arm64) with fixed seeds
6. **Compatibility**: v1 API/CLI/Python flows operate via v2 core without breaking changes during transition
