## Context

PulsimCore needs to compete with commercial power electronics simulators (PSIM, PLECS, Simplis) and open-source alternatives (ngspice, Xyce). Current architecture limitations prevent achieving required performance and accuracy.

### Stakeholders
- Power electronics engineers needing accurate simulation
- GUI developers integrating the Python API
- Future contributors extending device models

### Constraints
- Must maintain Python binding compatibility
- Cross-platform: Linux, macOS, Windows
- No GPL dependencies in core
- Clang 17+ required (GCC 14+ secondary)

## Goals / Non-Goals

### Goals
- Achieve SPICE-level accuracy for all supported circuits
- 2-5x performance improvement through compile-time optimization
- Zero-allocation hot paths during simulation
- Extensible architecture for new solvers and devices
- Production-ready robustness for all power electronics topologies

### Non-Goals
- Full SPICE compatibility (we focus on power electronics)
- GPU acceleration (future scope)
- Distributed simulation (future scope)
- Real-time HIL support (future scope)

## Decisions

### 1. C++23 with Clang

**Decision**: Use C++23 standard with Clang 17+ as primary compiler.

**Rationale**:
- `std::expected` for error handling without exceptions
- `std::mdspan` for multidimensional array views
- `constexpr` improvements for compile-time computation
- Better `concepts` support for template constraints
- Deducing `this` for CRTP simplification

**Alternatives Considered**:
- C++20: Lacks key features (`std::expected`, improved constexpr)
- GCC: Slower compile times, less mature C++23 support

### 2. CRTP for Device Models

**Decision**: Use Curiously Recurring Template Pattern for static polymorphism.

```cpp
template<typename Derived>
class DeviceBase {
public:
    void stamp(MNAMatrix& G, Vector& b, const State& state) {
        static_cast<Derived*>(this)->stamp_impl(G, b, state);
    }

    constexpr auto jacobian_pattern() const {
        return static_cast<const Derived*>(this)->jacobian_pattern_impl();
    }
};

class Resistor : public DeviceBase<Resistor> {
    friend class DeviceBase<Resistor>;
    void stamp_impl(MNAMatrix& G, Vector& b, const State& state);
    static constexpr auto jacobian_pattern_impl() { /* compile-time pattern */ }
};
```

**Rationale**:
- Zero virtual call overhead
- Enables compile-time Jacobian sparsity analysis
- Allows inlining of hot stamp() calls

**Alternatives Considered**:
- Virtual functions: 5-10% overhead in inner loops
- std::variant: Good but loses compile-time analysis

### 3. Expression Templates for Matrix Operations

**Decision**: Implement lazy evaluation for matrix expressions.

```cpp
// Instead of creating temporaries:
// auto temp = A * B; result = temp + C;

// Expression templates defer computation:
auto expr = A * B + C;  // No computation yet
result = expr;           // Single pass evaluation
```

**Rationale**:
- Eliminates temporary allocations
- Enables SIMD optimization across fused operations
- Eigen already uses this pattern

### 4. Policy-Based Solver Design

**Decision**: Use policy classes for solver configuration.

```cpp
template<
    typename LinearSolver = SparseLUPolicy,
    typename Timestep = AdaptiveTimestepPolicy,
    typename Integration = TrapezoidalPolicy,
    typename Convergence = ArmijoLineSearchPolicy
>
class Simulator {
    LinearSolver linear_solver_;
    Timestep timestep_;
    Integration integration_;
    Convergence convergence_;
};
```

**Rationale**:
- Compile-time selection eliminates branching
- Easy to add new policies
- Policies can be composed

### 5. Memory Pool Allocators

**Decision**: Use arena allocators for simulation hot paths.

```cpp
class SimulationArena {
    alignas(64) std::byte buffer_[64 * 1024];  // 64KB L1-friendly
    size_t offset_ = 0;

public:
    template<typename T>
    T* allocate(size_t n) {
        // Fast bump allocation
    }

    void reset() { offset_ = 0; }  // O(1) deallocation
};
```

**Rationale**:
- Allocation is O(1) pointer bump
- Deallocation is O(1) reset
- Cache-friendly contiguous memory

### 6. SoA Data Layout for Devices

**Decision**: Structure of Arrays for vectorization.

```cpp
// Instead of AoS:
// struct Device { double v, i, g; };
// std::vector<Device> devices;

// Use SoA:
struct DeviceArrays {
    std::vector<double> voltages;
    std::vector<double> currents;
    std::vector<double> conductances;
};
```

**Rationale**:
- SIMD can process 4-8 elements at once
- Better cache utilization
- Enables parallel reduction

### 7. Companion Model Implementation

**Decision**: Correct implementation with history terms.

**Trapezoidal (GEAR-2) for Capacitor**:
```cpp
// Current companion model:
// I_eq = (2C/dt) * V_n - (2C/dt) * V_{n-1}   // WRONG - missing I_{n-1}

// Correct model:
// I_eq = (2C/dt) * V_n - (2C/dt) * V_{n-1} - I_{n-1}
// Where I_{n-1} is the capacitor current at previous timestep

struct CapacitorState {
    double v_prev;      // Previous voltage
    double i_prev;      // Previous current (CRITICAL!)
    double g_eq;        // Equivalent conductance = 2C/dt
};

void stamp_trapezoidal(double C, double dt, CapacitorState& state,
                       int n_plus, int n_minus) {
    double g_eq = 2.0 * C / dt;
    double i_eq = g_eq * state.v_prev + state.i_prev;  // Include i_prev!

    // Stamp conductance
    G(n_plus, n_plus) += g_eq;
    G(n_plus, n_minus) -= g_eq;
    G(n_minus, n_plus) -= g_eq;
    G(n_minus, n_minus) += g_eq;

    // Stamp equivalent current source
    b(n_plus) += i_eq;
    b(n_minus) -= i_eq;
}
```

**BDF2 Coefficients**:
```cpp
// BDF2: (3/2) * y_n - 2 * y_{n-1} + (1/2) * y_{n-2} = dt * f(y_n)
constexpr double BDF2_ALPHA = 3.0 / 2.0;
constexpr double BDF2_BETA = -2.0;
constexpr double BDF2_GAMMA = 1.0 / 2.0;
```

### 8. Newton Solver Convergence

**Decision**: Proper residual checking and line search.

```cpp
NewtonResult solve(SystemFunction f, Vector x0) {
    Vector x = x0;

    for (int iter = 0; iter < max_iters; ++iter) {
        auto [residual, jacobian] = f(x);

        double norm = residual.norm();
        if (norm < abstol) {
            return {x, Success, iter};  // Converged
        }

        Vector dx = solve_linear(jacobian, -residual);

        // Armijo line search
        double alpha = line_search(f, x, dx, residual);
        x += alpha * dx;
    }

    // CRITICAL: Check convergence after final step
    auto [final_residual, _] = f(x);
    if (final_residual.norm() < abstol) {
        return {x, Success, max_iters};
    }

    return {x, MaxIterationsReached, max_iters};
}
```

## Risks / Trade-offs

### Risk 1: Compile Time Increase
- **Mitigation**: C++20 modules, precompiled headers, parallel compilation
- **Monitoring**: CI tracks compile times

### Risk 2: Template Error Messages
- **Mitigation**: Concepts for clear constraints, static_assert messages
- **Monitoring**: Developer documentation

### Risk 3: Debugging Difficulty
- **Mitigation**: Debug builds disable optimizations, logging infrastructure
- **Monitoring**: Comprehensive test coverage

### Risk 4: Breaking Changes
- **Mitigation**: Python API compatibility layer, deprecation warnings
- **Monitoring**: Integration tests

## Migration Plan

### Phase 1 (Foundation)
1. Update CMake for C++23
2. Add Clang requirement
3. Create new header structure alongside old

### Phase 2 (Core Rewrite)
1. Implement new type system
2. Implement CRTP device base
3. Port device models one by one

### Phase 3 (Solver Rewrite)
1. Implement policy-based solver
2. Fix companion models
3. Add convergence aids

### Phase 4 (Integration)
1. Update Python bindings
2. Deprecate old API
3. Remove old implementation

### Rollback Strategy
- Git tags at each phase completion
- Old implementation preserved until Phase 4
- Feature flags for gradual rollout

## Open Questions

1. **Module Support**: Should we fully adopt C++20 modules or stay with headers?
   - Decision: Start with headers, evaluate modules in Phase 1

2. **SIMD Library**: Use explicit intrinsics, xsimd, or Eigen's vectorization?
   - Decision: Start with Eigen, add xsimd for custom operations

3. **Allocator Strategy**: Single global arena or per-simulation arena?
   - Decision: Per-simulation arena for thread safety

## References

- SPICE3 Implementation Guide (Berkeley)
- "Numerical Methods for VLSI Simulation" - Pillage et al.
- "Direct Methods for Sparse Linear Systems" - Tim Davis
- Eigen Documentation: Expression Templates
- CppCon 2023: CRTP and Static Polymorphism
