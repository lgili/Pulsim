## 1. Contract and Schema
- [x] 1.1 Define canonical frequency-analysis contract in YAML (`simulation.frequency_analysis`) with modes, anchoring, sweep, perturbation, injection, and measurement fields.
- [x] 1.2 Add strict validation and deterministic diagnostics for invalid ranges, unsupported mode combinations, and malformed measurement bindings.
- [x] 1.3 Define canonical kernel-side option/result structures for AC sweep and publish field-level documentation.

## 2. Kernel Runtime Implementation
- [x] 2.1 Implement analysis orchestration for open-loop, closed-loop, and impedance sweeps.
- [x] 2.2 Implement deterministic anchoring pipeline (`dc`, `periodic`, `averaged`, `auto`) with explicit fallback policy and typed failure reasons.
- [x] 2.3 Implement sweep execution with deterministic grid generation, perturbation application, response extraction, and complex transfer computation.
- [x] 2.4 Add derived metrics computation (crossovers, phase/gain margins) with clear "undefined" semantics.

## 3. Result Surface and Telemetry
- [x] 3.1 Expose structured per-run results including frequency vector, complex response arrays, magnitude/phase arrays, and derived stability metrics.
- [x] 3.2 Add canonical metadata for response channels/quantities to avoid frontend heuristics.
- [x] 3.3 Add deterministic telemetry and failure taxonomy (anchor failure, singular linearization, invalid probe mapping, sweep-domain errors).

## 4. Python Bindings and API
- [x] 4.1 Expose class-based and procedural Python APIs for AC sweep with typed request/response structures.
- [x] 4.2 Preserve backward compatibility for existing transient/DC APIs.
- [x] 4.3 Ensure Python exceptions carry structured diagnostics and reason codes.

## 5. Benchmark and Regression Coverage
- [x] 5.1 Add analytical/reference AC sweep scenarios (at least one linear circuit and one converter/control scenario).
- [x] 5.2 Add determinism regression tests (repeat-run stability on same machine class).
- [x] 5.3 Add KPI gates for accuracy and runtime (`ac_sweep_mag_error`, `ac_sweep_phase_error`, runtime percentile).
- [x] 5.4 Add parser contract tests for strict-mode failures and migration-safe defaults.

## 6. Documentation and UX Boundary
- [x] 6.1 Document backend AC sweep contract (inputs, outputs, diagnostics, limits).
- [x] 6.2 Document explicit GUI responsibilities (setup UX, plotting, report composition) and non-responsibilities (no synthetic physics).
- [x] 6.3 Publish usage examples for YAML and Python.

## 7. Quality Gates (Mandatory Before Merge)
- [x] 7.1 `openspec validate add-frequency-domain-ac-sweep-analysis --strict` passes.
- [x] 7.2 Targeted tests pass (kernel + python + parser + benchmark AC sweep subset).
- [x] 7.3 Benchmark non-regression gates for AC sweep KPIs pass in CI.
- [x] 7.4 Acceptance evidence (artifacts + threshold values + known limitations) is recorded in change notes.
