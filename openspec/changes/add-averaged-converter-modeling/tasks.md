## 1. Contract and Schema
- [x] 1.1 Define canonical YAML contract for `simulation.averaged_converter` including mode enablement, topology, mapped elements, duty-input source, and envelope policy fields.
- [x] 1.2 Add strict parser validation and deterministic diagnostics for malformed or incomplete averaged-model definitions.
- [x] 1.3 Define kernel runtime option/result structures for averaged-mode execution and mapping diagnostics.

## 2. Kernel Runtime Implementation
- [x] 2.1 Implement averaged-state equations for supported topologies (MVP topology set explicitly documented).
- [x] 2.2 Implement deterministic switching-to-averaged mapping pipeline with explicit required-field checks.
- [x] 2.3 Implement operating-envelope checks (CCM-focused in MVP) with policy-driven behavior (`strict` fail vs `warn` continue).
- [x] 2.4 Ensure no silent mode substitution: averaged-mode failures return typed diagnostics instead of implicit fallback to switched mode.

## 3. Result Surface and Telemetry
- [x] 3.1 Expose averaged-state channels as structured outputs with canonical metadata (domain/unit/source).
- [x] 3.2 Expose mapping/envelope telemetry fields for CI and frontend explainability.
- [x] 3.3 Keep deterministic ordering and allocation-bounded behavior for averaged-mode result channels.

## 4. Python Bindings and API
- [x] 4.1 Expose typed Python surface for averaged-converter options/enums/diagnostics.
- [x] 4.2 Support class and procedural workflows without breaking existing transient APIs.
- [x] 4.3 Provide structured exception/error context for averaged-mode configuration/execution failures.

## 5. Benchmark and Regression Coverage
- [x] 5.1 Add paired switching-vs-averaged benchmark cases for supported topologies.
- [x] 5.2 Add determinism regression checks for repeated averaged-mode runs.
- [x] 5.3 Add KPI gates for fidelity and runtime value (error metrics + speedup floor).
- [x] 5.4 Add strict parser contract tests for invalid averaged-model configurations.

## 6. Documentation and UX Boundary
- [ ] 6.1 Document averaged-model contract, supported topologies, and explicit validity envelope.
- [ ] 6.2 Document frontend responsibilities (mode setup, envelope warnings, plotting/selection) and non-responsibilities.
- [ ] 6.3 Publish YAML + Python examples and migration guidance from switched transient workflows.

## 7. Quality Gates (Mandatory Before Merge)
- [x] 7.1 `openspec validate add-averaged-converter-modeling --strict` passes.
- [x] 7.2 Targeted kernel/python/parser tests for averaged mode pass locally and in CI.
- [ ] 7.3 Benchmark KPI gate for averaged modeling (fidelity + speedup) passes in CI.
- [x] 7.4 Acceptance evidence (artifacts, thresholds, known limitations) is recorded in change notes.
