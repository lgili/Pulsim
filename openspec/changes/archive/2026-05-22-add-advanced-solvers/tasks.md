## Gates & Definition of Done

- [ ] G.1 Accuracy: analytical RC/RL/RLC <0.1% and SPICE parity where applicable
- [ ] G.2 Robustness: stiff switching circuits converge with clear fallback traces
- [ ] G.3 Performance: >=2x speedup or equivalent iteration reduction
- [ ] G.4 Determinism: fixed solver order, reproducible results per hardware class
- [ ] G.5 Compatibility: v1 Python/CLI flows remain unchanged

## Phase 1: Solver Order + Safety
- [x] 1.1 Separate primary vs fallback solver order in config and YAML
- [x] 1.2 Enforce SPD checks for CG and disable CG on non‑SPD matrices
- [x] 1.3 Add tests for solver order and CG gating

## Phase 2: JFNK + Telemetry
- [x] 2.1 Implement Jacobian‑vector product (finite‑difference J·v)
- [x] 2.2 Add Newton–Krylov path that avoids full Jacobian assembly
- [x] 2.3 Extend telemetry for JFNK iterations and fallback paths

## Phase 3: Advanced Preconditioning
- [x] 3.1 Add ILUT preconditioner for iterative solvers
- [x] 3.2 Add optional AMG preconditioner (feature‑flagged)
- [x] 3.3 Add preconditioner selection tests and metrics

## Phase 4: Stiff Integrators
- [x] 4.1 Implement TR‑BDF2 integrator with LTE
- [x] 4.2 Implement Rosenbrock‑W/SDIRK integrator
- [x] 4.3 Add stiffness detection hooks for order/dt control

## Phase 5: Periodic Steady‑State
- [x] 5.1 Add shooting method for periodic steady‑state
- [x] 5.2 Add harmonic balance (HB) option for switching converters
- [x] 5.3 Add periodic validation cases

## Phase 6: YAML + Python/CLI
- [x] 6.1 Extend YAML schema for new solvers and integrators
- [x] 6.2 Expose configuration in Python API
- [x] 6.3 Add CLI flags for solver selection (no CLI target in repo; documented YAML options)

## Phase 7: Docs
- [x] 7.1 Update solver tuning guide with new methods
- [x] 7.2 Update netlist format docs with new solver fields
