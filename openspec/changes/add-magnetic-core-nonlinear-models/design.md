## Context
Pulsim already supports magnetic-related components (for example saturable/coupled inductor variants), but there is no unified backend contract that covers nonlinear core physics end-to-end across YAML, kernel runtime, Python bindings, and benchmark gates.

For professional power-electronics workflows, the missing pieces are:
- explicit model-family contracts for saturation + hysteresis + frequency-dependent core loss,
- deterministic state/loss telemetry channels,
- strict diagnostics for invalid model setups,
- CI-grade fidelity and determinism benchmarks.

## Goals / Non-Goals
- Goals:
  - Provide deterministic nonlinear magnetic-core behavior for supported inductor/transformer workflows.
  - Provide canonical YAML/Python configuration surfaces with strict validation.
  - Expose structured outputs (channels + metadata + summaries) suitable for frontend and benchmark automation.
  - Enforce benchmark gates for fidelity, determinism, runtime, and allocation stability.
- Non-Goals:
  - Spatial/FEM magnetic simulation.
  - Automatic magnetic design/synthesis.
  - GUI-only physics calculations outside backend.

## Decisions
- Decision: Canonical `component.magnetic_core` configuration surface
  - Use a canonical nested block for nonlinear core configuration instead of fragmented ad-hoc parameters.
  - Rationale: explicit schema evolution, strict validation, and portable backend/frontend workflows.

- Decision: Model family strategy with deterministic validation
  - Introduce explicit model families with required parameter sets and declared validity semantics.
  - Invalid family/parameter combinations fail with typed diagnostics.
  - Rationale: avoids silent behavior drift and ambiguous setups.

- Decision: Explicit hysteresis memory-state semantics
  - Define initialization and update ordering for hysteresis internal states.
  - Deterministic replay of state trajectories is required under identical inputs.
  - Rationale: hysteresis is path-dependent; explicit state semantics are mandatory for reproducibility.

- Decision: Backend-owned magnetic telemetry
  - Export canonical magnetic state/loss channels and structured metadata from kernel result surfaces.
  - Rationale: frontend should not reconstruct magnetic physics heuristically.

- Decision: KPI-gated rollout
  - Require benchmark fixtures and KPI gates for fidelity, determinism, and performance before merge.
  - Rationale: nonlinear magnetic regressions are subtle and easy to miss without objective gates.

## Proposed Runtime Architecture
1. Magnetic Core Config Resolver
   - Parses canonical model family and normalized parameter sets from validated component descriptors.
2. Magnetic State Engine
   - Updates nonlinear saturation/hysteresis internal state per accepted integration step with deterministic ordering.
3. Core-Loss Evaluator
   - Computes frequency-dependent and hysteresis-related loss terms per accepted step.
4. Electrothermal Coupler
   - Routes core losses to electrothermal pipeline when enabled; enforces consistency with summary reductions.
5. Result Assembler
   - Emits canonical channels/metadata and per-component summaries in deterministic order.

## Diagnostics Taxonomy (Planned)
- `MagneticCoreConfigInvalid`
- `MagneticCoreModelUnsupported`
- `MagneticCoreParameterOutOfRange`
- `MagneticCoreTableInvalid`
- `MagneticCoreStateInvalid`
- `MagneticCoreNumericalFailure`
- `MagneticCoreLossUndefined`

## Performance and Determinism Strategy
- Use explicit channel registration order and stable tie-breaking rules.
- Reuse scratch buffers for magnetic state/loss computation in repeated runs.
- Keep floating-point tolerance contracts explicit in tests and KPI gates.
- Prohibit silent fallback to linear/legacy magnetic paths when nonlinear core is requested.

## Migration Plan
1. Add schema + parser validation + diagnostics for canonical magnetic-core block.
2. Implement kernel nonlinear magnetic state/loss path with typed telemetry.
3. Expose Python typed surfaces and result mappings.
4. Add benchmark fixtures/KPI gates and determinism regressions.
5. Publish docs/examples and migration guidance for existing magnetic workflows.

## Open Questions
- Which model families are mandatory in MVP versus phase-2 expansion?
- How strict should default out-of-range table handling be (`fail`, `clamp`, `warn`) per policy mode?
- Which canonical magnetic telemetry channels are required for MVP frontend workflows?
