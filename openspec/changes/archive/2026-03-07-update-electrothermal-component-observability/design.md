## Context
Electrothermal simulation is a core differentiator for Pulsim power-electronics workflows. The runtime already maintains loss and thermal services, but outputs are split and not guaranteed to cover every component consistently. YAML currently accepts component thermal blocks with partial defaults, but explicit strict behavior for missing/invalid parameters and unsupported thermal-port activation needs to be formalized.

## Goals / Non-Goals
- Goals:
  - Provide deterministic per-component electrothermal telemetry (loss + temperature) from one stable contract.
  - Make thermal-port activation rules explicit and validated at parse/runtime boundaries.
  - Keep backward compatibility for existing summary surfaces.
  - Add validation coverage at benchmark/test level for component-level electrothermal correctness.
- Non-Goals:
  - Introduce new physics models beyond existing lumped thermal + scaling policy.
  - Replace existing `loss_summary`/`thermal_summary` payloads.
  - Redesign GUI thermal-scope behavior in this change.

## Decisions
- Decision: Introduce a canonical per-component electrothermal result surface.
  - Why: consumers need one deterministic list keyed by component name.
  - Details:
    - include all non-virtual circuit components in deterministic order.
    - include zero-loss entries instead of dropping components with zero dissipation.
    - include thermal fields for every entry; mark unsupported/non-enabled components with `thermal_enabled=false` and ambient-derived temperatures.
- Decision: Validate thermal-port enablement against component capability.
  - Why: avoid silent ignore behavior and ambiguous results.
  - Details:
    - if `component.thermal.enabled=true` on a non-thermal-capable component, parsing fails with deterministic diagnostics.
- Decision: Define strict vs non-strict missing-parameter behavior.
  - Why: preserve compatibility while raising quality for strict CI runs.
  - Details:
    - strict mode: thermal-enabled component missing `rth` or `cth` fails parsing.
    - non-strict mode: missing values are defaulted from `simulation.thermal.default_rth/default_cth` and warning diagnostics are emitted.
- Decision: Add runtime guardrails for invalid thermal constants.
  - Why: prevent numerically invalid runs from producing misleading telemetry.
  - Details:
    - reject non-finite values, `rth <= 0`, and `cth < 0` for enabled thermal components before stepping.

## Risks / Trade-offs
- Risk: larger result payloads for large circuits.
  - Mitigation: keep compact numeric fields and deterministic ordering; avoid duplicate verbose payloads.
- Risk: stricter diagnostics can fail previously accepted thermal netlists in strict mode.
  - Mitigation: non-strict fallback defaults + explicit warnings; document migration path.
- Risk: combining legacy summary paths and new per-component view could drift.
  - Mitigation: parity tests that cross-check aggregate totals against per-component reductions.

## Migration Plan
1. Add new per-component electrothermal telemetry struct/surface in kernel and Python bindings.
2. Populate it from existing loss/thermal services in one deterministic merge pass over circuit components.
3. Add parser validation rules for thermal-port capability and parameter completeness.
4. Add integration tests, parser tests, and benchmark assertions.
5. Update docs/examples to show recommended thermal-port configuration and result consumption.

## Open Questions
- Should passive components without thermal models always report ambient temperatures, or should they report null/absent thermal metrics? (proposal assumes ambient + `thermal_enabled=false` for deterministic shape)
