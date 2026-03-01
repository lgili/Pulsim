## Context
Current kernel evolution delivered major capability gains, but architecture remains concentrated in large compilation units and headers:
- `core/include/pulsim/v1/runtime_circuit.hpp` (~2954 LOC)
- `core/include/pulsim/v1/high_performance.hpp` (~2530 LOC)
- `core/include/pulsim/v1/integration.hpp` (~1862 LOC)
- `core/src/v1/simulation.cpp` (~2153 LOC)
- `core/src/v1/yaml_parser.cpp` (~1870 LOC)

This size and coupling profile makes refactors expensive, increases blast radius for regressions, and slows addition of new devices/solvers/features.

## Goals
- Deliver a professional, mature C++ core architecture with clear boundaries.
- Increase feature velocity by making extension points explicit and stable.
- Strengthen runtime safety and diagnosability under invalid inputs and hard nonlinear cases.
- Preserve or improve numerical robustness and runtime KPIs.
- Guarantee no KPI regression through versioned baseline gates.

## Non-Goals
- Rewriting physical models from scratch.
- Dropping backward compatibility in one step for Python/YAML users.
- Changing numerical semantics without benchmark and parity evidence.

## Baseline Problems
- Monolithic headers and source files mix orchestration, policy, algorithms, and data transforms.
- Extension points for new features are implicit and require touching core orchestration.
- Safety policy is not codified as a project-wide contract (sanitizers/static analysis/gates).
- Performance policy exists in code but not as enforceable KPI governance.
- Compatibility policy for YAML/Python growth is not formalized as a strict lifecycle.

## C++20/23 Engineering Policy
### Ownership and lifetime
- RAII-first: no owning raw `new/delete` in kernel application paths.
- Prefer value semantics where practical; use `std::unique_ptr` for ownership transfer and `std::shared_ptr` only for justified shared lifetimes.
- Use explicit non-owning views (`std::span`, `std::string_view`) on hot interfaces.

### Interface and compile-time safety
- Constrain extension templates with concepts where compile-time requirements are meaningful.
- Keep const-correctness and `noexcept` policy explicit for low-level services.
- Separate policy objects from mutable runtime state to reduce hidden coupling.

### Error and diagnostic model
- Standardize typed deterministic failure taxonomy (`reason_code` + structured context payload).
- Avoid text-based control flow in recovery paths.
- Ensure parser/runtime/bindings preserve the same structured error semantics.

### Performance policy
- Prefer contiguous storage and predictable memory layouts in hot loops.
- Reserve/reuse buffers and caches; allocate during setup/rebuild boundaries, not per-step steady-state loops.
- Require measured benchmark evidence for algorithm/data-structure substitutions.

### Toolchain gates
- Enforce strict warnings (`-Wall -Wextra -Wpedantic`) for managed core targets.
- Enforce sanitizer/static-analysis runs on touched core modules.
- Keep CMake target-based compile options and avoid global mutable flag drift.

## Target Architecture
### Layer model
1. `domain-model`: device/state/time primitives, pure data and invariants.
2. `equation-services`: assembly, Jacobian, residual, topology signature.
3. `solve-services`: nonlinear, linear, recovery, step policy, event scheduler.
4. `runtime-orchestrator`: simulation loop, commit semantics, telemetry.
5. `adapters`: YAML parser, Python bindings, benchmark adapters.

Rules:
- Dependencies are one-way (higher layer depends only on lower layers).
- Adapters SHALL not back-inject logic into lower layers.
- New feature categories (device/solver/control) SHALL be plugged through registries/contracts.

### Extensibility contracts
- Device registration contract: stamp/residual/state hooks + metadata + validation.
- Solver registration contract: capabilities + constraints + telemetry schema.
- Integrator registration contract: stability class + error estimate contract.
- Contract tests SHALL validate extension compatibility without patching orchestrator internals.

### Safety contracts
- Deterministic error model with typed failure reasons.
- Finite-value and dimensional consistency guards at service boundaries.
- Sanitizer suite (ASan/UBSan) and static analysis in CI for changed core modules.
- No owning raw `new/delete` in application path.

### Performance contracts
- Allocation-free steady-state stepping in hot loops (except configured setup/rebuild boundaries).
- Symbolic/factorization/preconditioner reuse keyed by topology signature with deterministic invalidation.
- KPI gates by scenario class:
  - convergence success rate
  - parity RMS/final error
  - event-time error
  - runtime p50/p95
  - retry/fallback rates

### Compatibility contracts
- YAML: versioned schema evolution with strict diagnostics and migration hints.
- Python: backward-compatible procedural surface during migration windows.
- Deprecation lifecycle SHALL be explicit and test-backed.

## KPI Governance
### Baseline policy
- Freeze baseline artifacts for benchmark matrix (converter, linear, stress, electrothermal).
- Record environment fingerprint (compiler, flags, machine class).

### Gate policy
- CI blocks merge when any KPI exceeds allowed regression threshold.
- Suggested default limits:
  - success-rate: >= baseline - 0.5 pp
  - parity RMS: <= baseline + 5%
  - runtime p95: <= baseline + 5%
  - event-time error: <= baseline + 10%
  - fallback-rate: <= baseline + 10%

## Rollout Strategy
1. Baseline freeze and gate wiring.
2. Layer decomposition and dependency enforcement.
3. Extension contracts + registry scaffolding.
4. Safety hardening and diagnostics unification.
5. Performance cache/allocation policy hardening.
6. YAML/Python compatibility and deprecation governance.
7. Final parity + benchmark + stress gate pass.

## Risks and Mitigations
- Risk: temporary churn from file/module split.
  - Mitigation: phase-by-phase refactor with interface compatibility shims.
- Risk: performance dip during abstraction extraction.
  - Mitigation: require each phase to pass runtime gates before proceeding.
- Risk: API drift for Python/YAML users.
  - Mitigation: enforce compatibility scenarios and migration diagnostics.
- Risk: parallel overlap with active solver/backend changes.
  - Mitigation: explicit integration checkpoints and shared KPI corpus.

## Open Questions
- Final module granularity for `runtime_circuit` split (by domain vs by service).
- Whether extension contracts are compile-time-only (concepts) or include runtime plugin ABI.
- Exact deprecation windows for YAML/Python compatibility aliases.
