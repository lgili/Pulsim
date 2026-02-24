## Context
The current SUNDIALS backend in v1 is robust enough to converge many hard circuits, but it is built around a projected-wrapper callback that embeds native Newton inside SUNDIALS callbacks. This duplicates work and introduces model distortion for switched converter DAEs, producing lower fidelity and slower execution versus native for benchmark scenarios.

## Goals / Non-Goals
- Goals:
  - Provide a true direct SUNDIALS formulation for IDA/CVODE/ARKODE.
  - Improve parity with native/LTspice on switched converter outputs.
  - Preserve deterministic fallback and existing backward-compatible behavior.
- Non-Goals:
  - Replacing native solver as default in this change.
  - Full symbolic Jacobian infrastructure.
  - Altering device physics models.

## Decisions
- Decision: add `SundialsFormulationMode` with `ProjectedWrapper` and `Direct`.
  - Rationale: allows incremental rollout and A/B comparison without breaking existing users.
- Decision: implement IDA direct path first and treat it as preferred direct family for MNA DAE systems.
  - Rationale: IDA naturally matches residual-based DAE formulation.
- Decision: retain projected-wrapper as explicit fallback for edge cases during migration.
  - Rationale: minimizes convergence regressions while direct path matures.
- Decision: require telemetry to label backend + formulation + key SUNDIALS counters.
  - Rationale: enables benchmark-driven tuning and CI guardrails.

## Risks / Trade-offs
- Risk: direct Jacobian assembly may increase implementation complexity.
  - Mitigation: start with sparse finite-difference Jacobian fallback and add optimized path incrementally.
- Risk: different SUNDIALS families may diverge in behavior.
  - Mitigation: establish family-specific parity thresholds and deterministic fallback order.
- Risk: event/reinit logic may break consistency in direct mode.
  - Mitigation: add dedicated switched-event regression tests for cold/warm starts.

## Migration Plan
1. Introduce formulation enum + config plumbing (YAML/Python).
2. Implement direct IDA callbacks and keep wrapper fallback.
3. Implement direct CVODE/ARKODE callbacks.
4. Add telemetry + tests + benchmark gates.
5. Update docs/notebooks and set operational guidance.

## Open Questions
- Should `Auto` prefer `Direct` SUNDIALS before wrapper when native fails?
- What parity threshold is acceptable per topology (RMSE, final-value error, runtime overhead)?
- Whether to add mass-matrix explicit support as a separate milestone.
