## 0. Baseline and KPI gate setup
- [x] 0.1 Freeze benchmark/parity/stress baseline artifacts with environment fingerprint.
- [x] 0.2 Define versioned KPI thresholds for convergence, accuracy, runtime, event timing, and fallback rates.
- [x] 0.3 Add CI non-regression gate jobs that block merge on KPI threshold violations.

## 1. Architecture decomposition
- [x] 1.1 Define target module map (`domain-model`, `equation-services`, `solve-services`, `runtime-orchestrator`, `adapters`).
- [x] 1.2 Split monolithic headers/sources into module-owned units without behavior changes.
- [x] 1.3 Add dependency-boundary checks to prevent forbidden layer coupling.
- [x] 1.4 Add architecture tests ensuring fixed/variable modes share the same solve-service contracts.

## 2. Modern C++20/23 hardening
- [x] 2.1 Replace unsafe ownership patterns in touched modules with RAII/value-semantics contracts.
- [x] 2.2 Standardize non-owning interface views (`std::span`, `std::string_view`) and remove avoidable copies on hot paths.
- [x] 2.3 Add concepts/compile-time constraints for extension contracts where interfaces are template-based.
- [x] 2.4 Consolidate typed deterministic error model and remove text-only control flow in recovery logic.
- [x] 2.5 Enforce target-level warnings (`-Wall -Wextra -Wpedantic`) and keep compile options target-scoped in CMake.

## 3. Extensibility framework
- [x] 3.1 Define device/solver/integrator extension contracts with required metadata and telemetry fields.
- [x] 3.2 Implement registry and validation hooks for extension loading and compatibility checks.
- [x] 3.3 Add contract tests proving new feature classes can be added without orchestrator edits.

## 4. Safety hardening
- [x] 4.1 Add finite-value, bounds, and dimensional guards at service boundaries.
- [x] 4.2 Enable sanitizer/static-analysis CI jobs for touched core modules.
- [x] 4.3 Add regression tests for malformed netlists/options and hard nonlinear failure containment.

## 5. Performance hardening
- [x] 5.1 Enforce allocation discipline in hot paths (zero dynamic allocations in steady-state stepping loops).
- [x] 5.2 Implement deterministic cache reuse/invalidation for symbolic/factorization/preconditioner state.
- [x] 5.3 Add performance telemetry fields and cache hit/miss observability in results.
- [ ] 5.4 Run matrix benchmarks and confirm runtime/convergence KPIs remain non-regressive.

## 6. Compatibility and migration
- [ ] 6.1 Add YAML schema-evolution policy checks and strict migration diagnostics.
- [ ] 6.2 Maintain Python procedural compatibility while exposing canonical class/runtime surfaces.
- [ ] 6.3 Add compatibility tests for previous YAML/Python usage patterns.
- [ ] 6.4 Publish migration guidance with before/after examples.

## 7. Final validation
- [ ] 7.1 Run full benchmark + parity + stress suites and publish KPI delta report.
- [ ] 7.2 Verify no KPI regression against frozen baseline under approved thresholds.
- [ ] 7.3 Mark task completion only after all gates pass.
