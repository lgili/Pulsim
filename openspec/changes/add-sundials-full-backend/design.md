## Context
v1 runtime currently uses an internal Newton-based transient engine with adaptive timestep and fallback aids. Build-time detection for SUNDIALS exists, but there is no runtime SUNDIALS backend execution path.

## Goals / Non-Goals
- Goals:
  - Integrate SUNDIALS as a first-class runtime backend for stiff transient simulation.
  - Keep deterministic behavior and telemetry comparable to existing fallback traces.
  - Preserve compatibility when SUNDIALS is not compiled.
- Non-Goals:
  - Replacing all native integrators.
  - Changing DC operating point strategy in this change.
  - Introducing new external APIs beyond simulation configuration and telemetry.

## Decisions
- Decision: Introduce backend mode in transient runtime (`Native`, `SundialsOnly`, `Auto`).
  - Rationale: preserves existing default while enabling explicit user control and deterministic fallback.
- Decision: Prefer IDA for MNA DAE systems in SUNDIALS path and keep CVODE/ARKODE selectable.
  - Rationale: MNA naturally yields DAE structure; IDA is the robust default.
- Decision: Map fallback reason codes into existing trace stream with backend markers.
  - Rationale: avoids parallel diagnostics channels and keeps existing tooling stable.
- Decision: Keep SUNDIALS optional at compile time; runtime requests degrade with explicit diagnostics if unavailable.
  - Rationale: cross-platform packaging constraints.

## Risks / Trade-offs
- Risk: Inconsistent behavior across platforms due to differing SUNDIALS builds.
  - Mitigation: enforce one SUNDIALS-enabled CI job and stable tolerances in regression tests.
- Risk: Performance regressions for easy circuits.
  - Mitigation: keep `Native` as default and gate SUNDIALS escalation to repeated failure cases.
- Risk: Added complexity in Jacobian/mass matrix callbacks.
  - Mitigation: isolate in dedicated backend module and keep strict test coverage.

## Migration Plan
1. Add backend interfaces + no-op stubs for builds without SUNDIALS.
2. Implement IDA path first, then CVODE/ARKODE selection.
3. Wire `Auto` fallback after native failure thresholds.
4. Expose config/telemetry in YAML and Python.
5. Add tests and CI coverage.

## Open Questions
- Whether `Auto` should allow retry from SUNDIALS back to Native for very small systems.
- Whether to expose per-step SUNDIALS stats (nfe/nje/nni) in full detail or summarized counters only.
