## Context
Pulsim needs a deterministic averaged-converter workflow for fast control-design iteration while preserving trust against switching-reference behavior. The feature must be backend-owned, CI-verifiable, and explicit about approximation limits.

## Goals / Non-Goals
- Goals:
  - Add first-class averaged-converter runtime mode with typed contract and diagnostics.
  - Keep switching-to-averaged mapping deterministic and auditable.
  - Provide measurable value: faster runtime with bounded error in supported regimes.
- Non-Goals:
  - Generic symbolic reduction for arbitrary nonlinear topologies.
  - Automatic controller synthesis.
  - Real-time scheduler/HIL semantics.

## Decisions
- Decision: Canonical averaged configuration block
  - Introduce `simulation.averaged_converter` as the only canonical entrypoint for averaged modeling configuration.
  - Rationale: avoid fragmented ad-hoc flags and keep parser/runtime behavior stable.

- Decision: Deterministic topology-scoped MVP
  - MVP supports an explicit topology whitelist with deterministic required mapping fields per topology.
  - Rationale: practical implementation boundary that can be validated and benchmarked.

- Decision: Explicit envelope policy
  - Add envelope policy with deterministic behavior:
    - `strict`: fail fast on out-of-envelope conditions.
    - `warn`: continue with explicit warnings/telemetry tags.
  - Rationale: prevents silent misuse while supporting exploratory runs.

- Decision: No implicit fallback to switched mode
  - Averaged-mode invalidity/errors return typed diagnostics; runtime does not silently switch execution mode.
  - Rationale: preserves reproducibility and avoids hidden behavior changes.

- Decision: Paired KPI gating
  - CI must gate both fidelity (error bounds vs switching references) and runtime value (minimum speedup floor).
  - Rationale: feature is useful only if both correctness and performance value hold.

## Risks / Trade-offs
- Approximation misuse outside valid operating regions.
  - Mitigation: envelope checks, typed diagnostics, and docs with explicit limits.
- Topology coverage pressure may outpace validation depth.
  - Mitigation: topology whitelist with explicit expansion plan and per-topology tests.
- Added complexity in parser/runtime contracts.
  - Mitigation: strict schema, explicit defaults, and stable typed API surfaces.

## Migration Plan
1. Define and validate YAML/runtime/Python contracts.
2. Implement averaged runtime path for MVP topology set with envelope checks.
3. Add paired switching-reference benchmark cases and KPI gates.
4. Publish docs/notebook examples and frontend integration guidance.

## Open Questions
- Which topology set should be mandatory in MVP (`buck`, `boost`, `buck_boost`, etc.)?
- Should envelope checks rely purely on configured regime metadata, dynamic detection, or both?
- Which KPI thresholds should be fixed globally vs topology-specific in CI policy?
