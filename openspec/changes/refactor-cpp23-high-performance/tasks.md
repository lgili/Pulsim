## Gates & Definition of Done

- [ ] G.1 Accuracy: analytical RC/RL/RLC <0.1% and SPICE parity where available
- [ ] G.2 Robustness: nonlinear switching circuits converge with deterministic fallback order
- [ ] G.3 Performance: >=2x speedup or equivalent iteration reduction on benchmark suite
- [ ] G.4 Determinism: repeatable results with fixed solver order and seeds
- [ ] G.5 Compatibility: v1 Python/CLI flows remain unchanged
- [ ] G.6 Docs: updated solver tuning + YAML schema examples

## Phase 0: Baseline Alignment (v1 Core)
- [x] 0.1 Confirm v1 runtime is the only core engine (unify-v1-core)
- [x] 0.2 Remove JSON paths and keep YAML-only netlists
- [x] 0.3 Archive legacy core from build targets

## Phase 1: Linear Solver Stack
- [x] 1.1 Add runtime linear solver interface and selection policy
- [x] 1.2 Implement GMRES/BiCGSTAB/CG iterative solvers
- [x] 1.3 Implement preconditioners (ILU0/Jacobi) and scaling
- [x] 1.4 Auto-select solver based on size/conditioning; deterministic fallback order
- [x] 1.5 Emit linear solver telemetry (iterations, residuals, fallbacks)

## Phase 2: Nonlinear Solver Upgrades
- [x] 2.1 Add Anderson acceleration strategy
- [x] 2.2 Add Broyden update option
- [x] 2.3 Add Newton–Krylov path (matrix-free or Jacobian reuse)
- [x] 2.4 Strengthen line search / trust-region policies
- [x] 2.5 Record nonlinear solver telemetry and fallback reason codes

## Phase 3: Stiffness & Event Handling
- [ ] 3.1 Detect stiffness indicators (rejection streaks, conditioning shifts)
- [ ] 3.2 Apply BDF order caps and dt backoff under stiffness
- [ ] 3.3 Event-aligned step splitting for hard switching edges

## Phase 4: Performance & Memory
- [ ] 4.1 Reuse Jacobian structure and symbolic factorization where possible
- [ ] 4.2 Ensure hot-path allocation-free execution (arena reuse)
- [ ] 4.3 SIMD/SoA optimizations for runtime device evaluation

## Phase 5: YAML + Python/CLI Configuration
- [ ] 5.1 Extend YAML schema with `simulation.solver` options
- [ ] 5.2 Map YAML options to v1 simulation config
- [ ] 5.3 Expose solver configuration in Python API
- [ ] 5.4 Update CLI to surface solver selection flags

## Phase 6: Validation & Benchmarks
- [ ] 6.1 Add solver-selection regression tests
- [ ] 6.2 Add stress tests for stiff switching circuits
- [ ] 6.3 Update benchmark suite to report solver paths + telemetry

## Phase 7: Documentation
- [ ] 7.1 Update solver tuning guide
- [ ] 7.2 Update netlist format docs with solver options
- [ ] 7.3 Update README and notebooks with new solver configuration examples
