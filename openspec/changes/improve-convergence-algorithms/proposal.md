# Proposal: Advanced Convergence Algorithms for Complex Circuits

## Summary

Implement advanced convergence algorithms inspired by ngspice, Xyce, and academic research to significantly improve simulation reliability for complex power electronics circuits (buck/boost converters, half-bridges, full-bridges, flyback converters, etc.).

## Motivation

### Current Pain Points

1. **DC Operating Point Failures**: Complex circuits with multiple nonlinear devices (MOSFETs, diodes) often fail to find a valid DC operating point using simple Newton-Raphson.

2. **Transient Convergence Issues**: Circuits with fast switching (PWM at 100kHz+) frequently fail to converge during switching transitions.

3. **Timestep Inefficiency**: Current adaptive timestep is based on step-doubling LTE estimation, which is computationally expensive (3x more work per step).

4. **Limited DC Startup Options**: Only basic Newton-Raphson with auto-damping; no GMIN stepping, source stepping, or pseudo-transient analysis.

5. **Device Model Limitations**: Voltage limiting for diodes/MOSFETs is minimal, leading to Newton divergence with large voltage swings.

### Expected Benefits

| Metric | Current | Target |
|--------|---------|--------|
| DC convergence rate | ~70% | >95% |
| Transient failures | ~20% | <5% |
| Simulation speed | Baseline | 2-5x faster |
| Memory usage | Baseline | No increase |

## Scope

### In Scope

1. **DC Operating Point Algorithms**
   - GMIN stepping (adaptive conductance injection)
   - Source stepping (ramp sources from 0 to final value)
   - Pseudo-transient analysis (use transient to find DC)
   - Homotopy/continuation methods

2. **Transient Timestep Control**
   - Newton iteration-based timestep adjustment
   - LTE estimation without step-doubling (Richardson extrapolation)
   - Event-driven timestep control (switch transitions)
   - Timestep smoothing (avoid oscillations)

3. **Device Model Improvements**
   - Voltage limiting for diodes (prevent exp() overflow)
   - Voltage limiting for MOSFETs (Vgs, Vds limits)
   - Junction charge continuity (smooth C-V curves)
   - Piecewise linear (PWL) model options

4. **Linear Solver Optimization**
   - Enable KLU by default (already supported, not default)
   - Matrix reuse strategies (symbolic factorization caching)
   - Pivot ordering optimization

5. **Numerical Stability**
   - GMIN floor conductance (prevent floating nodes)
   - Minimum conductance for off-state devices
   - Limiting functions for all nonlinear devices

### Out of Scope

- Parallel/MPI simulation (separate proposal)
- New device models (BSIM, etc.)
- AC analysis improvements
- GUI changes

## Technical Approach

### Phase 1: DC Operating Point (Priority: High)

Implement a multi-strategy DC solver that tries methods in sequence:

```cpp
enum class DCStrategy {
    Newton,           // Standard Newton-Raphson
    GminStepping,     // Add Gmin, reduce exponentially
    SourceStepping,   // Ramp sources from 0
    PseudoTransient,  // Short transient simulation
    Homotopy          // Continuation method (advanced)
};

NewtonResult dc_operating_point_robust(const DCOptions& opts);
```

**GMIN Stepping Algorithm**:
```
for gmin in [1e-3, 1e-4, ..., 1e-12]:
    add_gmin_to_all_nodes(gmin)
    result = newton_solve()
    if converged:
        break
remove_gmin()
final_result = newton_solve()  // Final solve without gmin
```

### Phase 2: Timestep Control (Priority: High)

Replace step-doubling with efficient LTE estimation:

```cpp
struct TimestepController {
    Real compute_lte(const Vector& x_n, const Vector& x_prev,
                     const Vector& x_prev2, Real dt, Real dt_prev);

    Real adjust_timestep(Real dt, Real lte, int newton_iters);

    Real detect_event_crossing(Real t_prev, Real t_curr,
                               const Vector& x_prev, const Vector& x_curr);
};
```

**Timestep Adjustment Rules**:
1. If Newton iterations > threshold: reduce dt by 0.5
2. If Newton iterations < min_threshold: increase dt by 1.5
3. If LTE > tolerance: reduce dt
4. If LTE < tolerance * 0.1: increase dt
5. Near events: shrink dt to hit event precisely

### Phase 3: Device Models (Priority: Medium)

Add voltage limiting to all nonlinear devices:

```cpp
// Diode voltage limiting
Real limit_diode_voltage(Real V_new, Real V_old, Real Vt, Real n) {
    Real max_change = 4.0 * n * Vt;  // Limit to ~4 thermal voltages
    Real dV = V_new - V_old;
    if (std::abs(dV) > max_change) {
        return V_old + std::copysign(max_change, dV);
    }
    return V_new;
}
```

### Phase 4: Linear Solver (Priority: Medium)

Enable KLU by default and add reuse strategies:

```cpp
struct LinearSolverOptions {
    Backend backend = Backend::KLU;  // Change default from Eigen
    bool reuse_symbolic = true;
    int max_reuses = 50;
    Real refactor_threshold = 0.1;
};
```

## Dependencies

- **Eigen 3.4+**: Already included
- **SuiteSparse/KLU**: Already optional, make recommended
- **SUNDIALS**: Optional, for IDA fallback

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Regression in simple circuits | Medium | High | Extensive test suite |
| Increased complexity | High | Medium | Clear API, documentation |
| Performance overhead | Low | Medium | Optional features |

## Success Criteria

1. All existing tests pass
2. New convergence test suite passes (100 circuits)
3. Benchmark shows no regression (within 10%)
4. 95%+ DC convergence on ngspice test suite
5. 95%+ transient convergence on power electronics benchmarks

## Timeline Estimate

- Phase 1 (DC): Core functionality
- Phase 2 (Timestep): Core functionality
- Phase 3 (Devices): Enhancement
- Phase 4 (Linear): Enhancement

## References

1. [ngspice Manual - Convergence](https://ngspice.sourceforge.io/docs/ngspice-html-manual/manual.xhtml)
2. [Xyce Theory Manual](https://xyce.sandia.gov/documentation/XyceTheoryGuide.pdf)
3. [KLU: A Direct Sparse Solver](https://dl.acm.org/doi/10.1145/1824801.1824814)
4. [Homotopy Methods for DC Analysis](https://jaijeet.github.io/research/PDFs/2006-01-TCAD-Roychowdhury-Melville-MOS-homotopy.pdf)
