# Design: Advanced Convergence Algorithms

## Context

PulsimCore is a circuit simulator focused on power electronics (buck, boost, flyback converters, H-bridges). These circuits contain:

- **Highly nonlinear devices**: MOSFETs, diodes, IGBTs with exponential I-V characteristics
- **Fast switching events**: PWM frequencies of 100kHz+ with sub-microsecond transitions
- **Stiff systems**: Large differences in time constants (switching ns vs thermal ms)
- **Multiple operating regions**: Devices transition between cutoff, linear, and saturation

The current Newton-Raphson solver with basic auto-damping achieves ~70% DC convergence and ~80% transient convergence on complex circuits, which is insufficient for production use.

### Stakeholders

- **Library users**: Need reliable simulations without manual convergence tuning
- **Power electronics engineers**: Need accurate switching waveforms for design validation
- **Control engineers**: Need stable transient simulations for control loop design

### Constraints

- C++23 with Eigen for linear algebra
- Optional SuiteSparse/KLU dependency
- Must maintain backward compatibility with existing API
- Memory footprint should not increase significantly
- Single-threaded performance is priority (parallel is separate proposal)

## Goals / Non-Goals

### Goals

1. **DC convergence >95%** on ngspice-compatible test circuits
2. **Transient convergence >95%** on power electronics benchmarks
3. **2-5x simulation speedup** through efficient timestep control
4. **No API breaking changes** - new features are opt-in or automatic fallbacks
5. **Configurable algorithm selection** - users can choose strategies

### Non-Goals

- Parallel/distributed simulation (separate proposal)
- New device models (BSIM4, etc.)
- AC/frequency domain improvements
- GUI integration

## Decisions

### Decision 1: Multi-Strategy DC Solver

**What**: Implement a cascading DC solver that tries multiple strategies in sequence until convergence.

**Why**: Different circuit topologies respond better to different algorithms. A single algorithm cannot handle all cases optimally.

**Algorithm cascade (default order)**:
```
1. Newton-Raphson with voltage limiting (fast, works for ~70% of circuits)
2. GMIN stepping (add conductance to ground, reduce exponentially)
3. Source stepping (ramp sources from 0 to final value)
4. Pseudo-transient (short transient simulation to find DC)
5. Homotopy (parameter continuation, most robust but slowest)
```

**Alternatives considered**:
- Single robust algorithm: Rejected because no single algorithm works universally
- User-selected algorithm: Still supported, but auto-cascade is better UX
- Parallel algorithm attempts: Future work, adds complexity

### Decision 2: GMIN Stepping Algorithm

**What**: Add small conductances (GMIN) from each node to ground, then reduce exponentially.

**Why**: GMIN prevents floating nodes and provides a path for Newton to converge even with cutoff devices.

**Algorithm**:
```cpp
Real gmin_sequence[] = {1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12};
for (Real gmin : gmin_sequence) {
    add_gmin_to_diagonal(gmin);
    if (newton_solve().converged) {
        if (gmin == gmin_sequence.back()) return success;
        continue;  // Try smaller GMIN
    }
}
// Final solve without GMIN
remove_gmin();
return newton_solve();
```

**Key parameters**:
- `gmin_initial = 1e-3` (starting conductance)
- `gmin_final = 1e-12` (minimum conductance)
- `gmin_factor = 0.1` (reduction factor per step)

### Decision 3: Source Stepping Algorithm

**What**: Ramp all independent sources from 0 to their final value in steps.

**Why**: Allows the circuit to "boot up" gradually, avoiding large initial nonlinearities.

**Algorithm**:
```cpp
Real steps[] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
for (Real factor : steps) {
    scale_all_sources(factor);
    if (!newton_solve().converged) {
        // Backtrack and use smaller steps
        insert_intermediate_steps();
    }
}
```

**Considerations**:
- Works well for circuits with strong power supply dependence
- May fail if circuit has internal feedback that requires full supply

### Decision 4: Efficient Timestep Control

**What**: Replace step-doubling LTE estimation with Richardson extrapolation using stored history.

**Why**: Step-doubling requires 3x computational work per step. Richardson extrapolation reuses previous solution vectors.

**Current (step-doubling)**:
```
1. Take step with dt → x1
2. Take step with dt/2 → x_half
3. Take step with dt/2 → x2
4. LTE = |x1 - x2| / (2^p - 1)
```

**Proposed (Richardson)**:
```
// Using BDF2 solutions at t_{n}, t_{n-1}, t_{n-2}
LTE ≈ (1/3) * |x_n - x_predicted|
where x_predicted comes from polynomial extrapolation
```

**Timestep adjustment formula**:
```cpp
Real safety = 0.9;
Real p_order = 2.0;  // BDF2 order
Real dt_new = safety * dt * pow(tolerance / lte, 1.0 / (p_order + 1));

// Newton iteration feedback
if (newton_iters > target_iters) {
    dt_new *= 0.5;  // Cut step if convergence is slow
}
if (newton_iters < target_iters / 2) {
    dt_new *= 1.5;  // Increase step if convergence is fast
}
```

### Decision 5: Event-Driven Timestep Control

**What**: Detect switching events and adjust timestep to hit them precisely.

**Why**: Missing switch transitions causes accuracy loss and potential convergence failures.

**Event detection**:
```cpp
struct SwitchEvent {
    size_t device_id;
    Real crossing_time;
    bool is_on_to_off;
};

// After each step, check for crossings
for (auto& sw : switches) {
    Real v_prev = sw.control_voltage(t_prev);
    Real v_curr = sw.control_voltage(t_curr);
    if (crosses_threshold(v_prev, v_curr, sw.threshold)) {
        // Binary search for exact crossing time
        Real t_cross = bisection_find_crossing(t_prev, t_curr, sw);
        // Force timestep to hit this point
        next_dt = min(next_dt, t_cross - t_curr);
    }
}
```

### Decision 6: Voltage Limiting for Nonlinear Devices

**What**: Limit voltage changes per Newton iteration to prevent divergence.

**Why**: Large voltage swings cause exponential functions to overflow or Newton to overshoot.

**Diode limiting** (based on ngspice):
```cpp
Real limit_diode_voltage(Real v_new, Real v_old, Real v_thermal) {
    Real v_critical = v_thermal * log(v_thermal / (M_SQRT2 * Is));

    if (v_new > v_critical && abs(v_new - v_old) > 2 * v_thermal) {
        if (v_old > 0) {
            Real arg = 1 + (v_new - v_old) / v_thermal;
            if (arg > 0) return v_old + v_thermal * log(arg);
        }
        return v_thermal * log(v_new / v_thermal);
    }
    return v_new;  // No limiting needed
}
```

**MOSFET limiting**:
```cpp
Real limit_mosfet_vgs(Real vgs_new, Real vgs_old, Real vth) {
    Real max_change = 0.5;  // Max 500mV per iteration
    Real dv = vgs_new - vgs_old;
    if (abs(dv) > max_change) {
        return vgs_old + copysign(max_change, dv);
    }
    return vgs_new;
}

Real limit_mosfet_vds(Real vds_new, Real vds_old) {
    Real max_change = 2.0;  // Max 2V per iteration
    Real dv = vds_new - vds_old;
    if (abs(dv) > max_change) {
        return vds_old + copysign(max_change, dv);
    }
    return vds_new;
}
```

### Decision 7: KLU as Default Linear Solver

**What**: Enable KLU sparse solver by default when SuiteSparse is available.

**Why**: KLU is 2-5x faster than Eigen SparseLU for circuit matrices due to optimized ordering for circuit-like sparsity patterns.

**Implementation**:
```cpp
struct LinearSolverOptions {
    enum class Backend { Auto, Eigen, KLU };
    Backend backend = Backend::Auto;  // Was: Eigen

    bool reuse_symbolic = true;    // Cache symbolic factorization
    int max_symbolic_reuses = 50;  // Refactor after N numeric solves
    Real pivot_tolerance = 1e-13;  // Minimum pivot value
};

// Auto-selection logic
Backend select_backend() {
    #ifdef PULSIM_HAS_KLU
    return Backend::KLU;
    #else
    return Backend::Eigen;
    #endif
}
```

**Matrix reuse strategy**:
- Symbolic factorization: Reuse until matrix structure changes
- Numeric factorization: Refactor when condition estimate degrades
- Track condition number growth between refactorizations

### Decision 8: GMIN Floor Conductance

**What**: Add minimum conductance to all nodes to prevent floating node issues.

**Why**: Floating nodes cause singular matrices and numerical instability.

**Implementation**:
```cpp
const Real GMIN_FLOOR = 1e-12;  // S (siemens)

void MNAAssembler::add_gmin_floor() {
    for (size_t i = 0; i < num_voltage_nodes; ++i) {
        if (i != ground_node) {
            // Add small conductance from node to ground
            G(i, i) += GMIN_FLOOR;
        }
    }
}
```

This is always applied, not just during GMIN stepping.

## Architecture Changes

### Modified Files

| File | Changes |
|------|---------|
| `core/include/pulsim/solver.hpp` | Add DCStrategy enum, voltage limiting options |
| `core/src/solver.cpp` | Implement multi-strategy DC solver |
| `core/include/pulsim/convergence_aids.hpp` | Add GMIN stepping, source stepping classes |
| `core/src/convergence_aids.cpp` | Implement convergence aid algorithms |
| `core/include/pulsim/simulation.hpp` | Add TimestepController, event detection |
| `core/src/simulation.cpp` | Replace step-doubling with Richardson |
| `core/include/pulsim/mna.hpp` | Add device voltage limiting hooks |
| `core/src/mna.cpp` | Implement voltage limiting in stamps |
| `core/include/pulsim/advanced_solver.hpp` | Enhanced linear solver options |
| `core/src/advanced_solver.cpp` | KLU default, symbolic reuse |

### New Classes

```cpp
// convergence_aids.hpp
class GminStepping : public ConvergenceAid {
    Real gmin_current_;
    std::vector<Real> gmin_sequence_;
public:
    void apply(MNASystem& mna) override;
    void reduce() override;
    bool is_complete() const override;
};

class SourceStepping : public ConvergenceAid {
    Real scale_factor_;
    std::vector<Real> scale_sequence_;
    std::vector<SourceSnapshot> original_values_;
public:
    void apply(MNASystem& mna) override;
    void advance() override;
    bool is_complete() const override;
};

// simulation.hpp
class TimestepController {
    std::deque<SolutionHistory> history_;
    Real tolerance_;
    int target_newton_iters_;
public:
    Real compute_lte(const Vector& x_n, Real dt);
    Real suggest_next_dt(Real current_dt, Real lte, int newton_iters);
    std::optional<Real> find_event_crossing(Real t_prev, Real t_curr,
                                            const Circuit& circuit);
};

// solver.hpp
struct VoltageLimiter {
    static Real limit_diode(Real v_new, Real v_old, Real vt, Real is);
    static Real limit_mosfet_vgs(Real vgs_new, Real vgs_old, Real vth);
    static Real limit_mosfet_vds(Real vds_new, Real vds_old);
    static Real limit_bjt_vbe(Real vbe_new, Real vbe_old, Real vt);
};
```

### API Additions (Backward Compatible)

```cpp
// New options structures
struct DCOptions {
    std::vector<DCStrategy> strategy_order = {
        DCStrategy::Newton,
        DCStrategy::GminStepping,
        DCStrategy::SourceStepping,
        DCStrategy::PseudoTransient
    };
    int max_iterations = 100;
    Real tolerance = 1e-9;
    bool voltage_limiting = true;
    Real gmin_floor = 1e-12;
};

struct TransientOptions {
    // Existing options remain unchanged...

    // New options with defaults that maintain current behavior
    TimestepMethod timestep_method = TimestepMethod::Richardson;  // Was: StepDoubling
    bool event_detection = true;
    Real event_tolerance = 1e-9;
    int target_newton_iters = 5;
    Real timestep_increase_factor = 1.5;
    Real timestep_decrease_factor = 0.5;
};

// New simulation method (existing methods unchanged)
DCResult dc_operating_point_robust(const DCOptions& opts = {});
```

## Risks / Trade-offs

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Regression in simple circuits | Medium | High | Extensive test suite with baseline comparisons |
| Performance overhead from extra checks | Low | Medium | Optional features, profile-guided optimization |
| Voltage limiting too aggressive | Medium | Medium | Configurable limits, auto-adjustment based on convergence |
| GMIN stepping changes solution | Low | High | Verify final solution without GMIN |
| KLU dependency issues | Low | Low | Fallback to Eigen always available |

### Trade-off: Speed vs Robustness

- **Richardson LTE**: Faster (1x work vs 3x) but slightly less accurate
- **Voltage limiting**: More iterations but prevents divergence
- **GMIN stepping**: Slower DC solve but much higher success rate

Default configuration optimizes for robustness. Users can tune for speed.

## Migration Plan

### Phase 1: Core Algorithms
1. Implement voltage limiting in device models
2. Implement GMIN floor conductance
3. Add GminStepping and SourceStepping classes
4. Wire into existing NewtonSolver

### Phase 2: Timestep Control
1. Implement Richardson LTE estimation
2. Add TimestepController class
3. Implement event detection
4. Replace step-doubling in Simulator

### Phase 3: Linear Solver
1. Make KLU default when available
2. Implement symbolic factorization caching
3. Add condition number monitoring

### Phase 4: Testing & Tuning
1. Create convergence test suite (100+ circuits)
2. Benchmark against ngspice/Xyce
3. Tune default parameters
4. Document configuration options

### Rollback Strategy

Each phase is independently revertible:
- Compile flags can disable new algorithms
- Options default to new behavior but can be overridden
- Old step-doubling code preserved behind flag

## Open Questions

1. **Homotopy methods**: Should we implement lambda-based continuation for the most difficult circuits? (Deferred to future enhancement)

2. **Automatic parameter tuning**: Should voltage limits auto-adjust based on device parameters? (Start with fixed values, add auto-tuning later)

3. **Sparse matrix library**: Should we consider other libraries like UMFPACK or PARDISO? (KLU is best for circuit matrices, defer others)

## References

1. [ngspice Manual - Chapter 15: Convergence](https://ngspice.sourceforge.io/docs/ngspice-html-manual/manual.xhtml)
2. [Xyce Theory Manual - Nonlinear Solver](https://xyce.sandia.gov/documentation/XyceTheoryGuide.pdf)
3. [KLU: A Direct Sparse Solver for Circuit Simulation Problems](https://dl.acm.org/doi/10.1145/1824801.1824814)
4. [Homotopy Methods for Circuit Simulation](https://jaijeet.github.io/research/PDFs/2006-01-TCAD-Roychowdhury-Melville-MOS-homotopy.pdf)
5. [SPICE3 Implementation Techniques](https://ptolemy.berkeley.edu/projects/embedded/pubs/downloads/spice/)
