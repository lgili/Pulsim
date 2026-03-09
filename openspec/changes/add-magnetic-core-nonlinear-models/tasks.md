## 1. Contract and Schema
- [x] 1.1 Define canonical YAML contract for nonlinear magnetic-core configuration (`component.magnetic_core`), including model family, parameter sets, initialization, and loss policy fields.
- [x] 1.2 Add strict parser validation and deterministic diagnostics for missing fields, inconsistent dimensions, invalid ranges, and unsupported model-family/type combinations.
- [x] 1.3 Define canonical kernel-side option/result structures for nonlinear magnetic-core state and loss telemetry.

## 2. Kernel Runtime Implementation
- [x] 2.1 Implement nonlinear saturation behavior for supported magnetic components with deterministic state semantics.
- [ ] 2.2 Implement hysteresis behavior with explicit memory-state initialization and update ordering.
- [x] 2.3 Implement frequency-dependent core-loss evaluation and coupling into loss summaries.
- [ ] 2.4 Integrate magnetic-core losses with electrothermal pipeline when enabled, without silent fallback.
- [x] 2.5 Ensure fail-fast typed diagnostics for unsupported magnetic models or invalid runtime conditions.

## 3. Result Surface and Telemetry
- [x] 3.1 Expose canonical magnetic channels and metadata (state/loss quantities, units, source component).
- [ ] 3.2 Expose deterministic summary reductions consistent with time-series channels.
- [ ] 3.3 Keep deterministic channel ordering and allocation-bounded behavior in repeated runs.

## 4. Python Bindings and API
- [ ] 4.1 Expose typed Python configuration surfaces for nonlinear magnetic-core models.
- [ ] 4.2 Expose magnetic-core channels/metadata and summary fields through existing result APIs.
- [x] 4.3 Propagate structured parser/runtime diagnostics with stable reason codes and field context.

## 5. Benchmark and Regression Coverage
- [ ] 5.1 Add canonical fixture circuits for saturation validation against analytical/reference expectations.
- [ ] 5.2 Add canonical fixture circuits for hysteresis-loop behavior and cycle-energy consistency checks.
- [x] 5.3 Add canonical fixture circuits for frequency-dependent core-loss trend validation.
- [x] 5.4 Add repeat-run determinism tests for magnetic-core channels and summaries.
- [ ] 5.5 Add KPI gates for fidelity, determinism, runtime, and allocation stability.

## 6. Documentation and UX Boundary
- [x] 6.1 Document backend nonlinear magnetic-core contract (model families, parameters, outputs, diagnostics, limits).
- [x] 6.2 Document frontend responsibilities and non-responsibilities for magnetic-core setup and visualization.
- [x] 6.3 Publish YAML + Python examples and at least one tutorial notebook.
- [x] 6.4 Add migration guidance from legacy/simple magnetic approximations to canonical nonlinear core models.

## 7. Quality Gates (Mandatory Before Merge)
- [x] 7.1 `openspec validate add-magnetic-core-nonlinear-models --strict` passes.
- [x] 7.2 Targeted parser/kernel/python tests for nonlinear magnetic-core flows pass locally and in CI.
- [ ] 7.3 Benchmark KPI gate for magnetic-core fidelity/determinism/performance passes in CI.
- [x] 7.4 Acceptance evidence (artifacts, thresholds, known limitations) is recorded in change notes.
