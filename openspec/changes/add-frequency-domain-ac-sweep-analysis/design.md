## Context
Pulsim needs a first-class frequency-domain capability aligned with modern control-design workflows. The implementation must stay backend-owned, deterministic, and compatible with existing YAML/Python runtime contracts.

## Goals / Non-Goals
- Goals:
  - Provide deterministic open-loop, closed-loop, and impedance sweeps.
  - Support switching converters without manual averaged-model derivation through explicit anchoring modes.
  - Expose structured results and diagnostics suitable for CI and frontend integration.
- Non-Goals:
  - GUI plotting/rendering logic.
  - Automatic controller synthesis (separate change).
  - Broad multi-domain co-simulation orchestration (separate change).

## Decisions
- Decision: Canonical analysis contract
  - Introduce a canonical `frequency_analysis` contract in YAML/runtime with explicit mode, anchoring, sweep, injection, and measurement definitions.
  - Rationale: avoids ad-hoc scripting and ensures reproducible CI behavior.

- Decision: Explicit anchoring strategy
  - Support `dc`, `periodic`, `averaged`, and `auto` anchoring modes.
  - `auto` uses deterministic selection policy and emits selected anchor in telemetry.
  - Rationale: covers both linear/non-switching and switching converter workflows.

- Decision: Structured output, no text parsing
  - Return frequency vector, complex response, magnitude/phase arrays, and margin metrics in typed result structures.
  - Rationale: stable machine-consumable contract for Python tools and GUI adapters.

- Decision: Deterministic error taxonomy
  - Standardize typed errors for invalid schema, unsupported mode/topology, singular linearization, and measurement mapping failures.
  - Rationale: predictable CI behavior and better user diagnostics.

## Risks / Trade-offs
- Numerical fragility near resonances/high-Q dynamics.
  - Mitigation: deterministic perturbation scaling constraints, bounded fallback policy, and strict diagnostics when response is unreliable.
- Runtime cost for wide/high-resolution sweeps.
  - Mitigation: deterministic cache reuse and explicit benchmark performance gates.
- Ambiguous margin calculations in exotic cases.
  - Mitigation: explicit undefined semantics and reason-tagged output fields.

## Migration Plan
1. Add contract surfaces (YAML + runtime + Python types) with strict validation.
2. Implement kernel execution path with anchor selection and structured outputs.
3. Add analytical and converter benchmark cases, then enforce KPI/determinism gates.
4. Publish docs and examples; keep backward compatibility for existing simulation entrypoints.

## Open Questions
- Should first release include MIMO/multi-port responses or remain SISO + impedance-only?
- Should periodic anchoring rely only on existing periodic solvers or include dedicated FRA injection cycles?
- Which default sweep density balances accuracy vs runtime for CI-grade benchmark gates?
