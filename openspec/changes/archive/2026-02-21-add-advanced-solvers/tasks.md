## Gates & Definition of Done

- [x] G.1 Accuracy: RC/RL/RLC benchmark validations and mapped ngspice parity checks pass within configured thresholds.
- [x] G.2 Robustness: stiff/transient solver and stress test suites pass with fallback telemetry available.
- [x] G.3 Performance: benchmark telemetry shows >2x runtime improvement or strong iteration reduction on advanced solver paths.
- [x] G.4 Determinism: repeated benchmark runs match on scenario status, steps, and error metrics with fixed solver order.
- [x] G.5 Compatibility: v1 Python flows remain stable across benchmark runner, matrix runner, and ngspice comparator workflows.

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
