## Context
The v1 runtime already uses service registries and extension metadata, but orchestration still carries multiple cross-cutting responsibilities in broad files. This creates high blast radius for changes in losses/thermal/control/channel output logic.

To scale toward PSIM/PLECS-class workflow breadth while keeping kernel performance and community maintainability, the architecture needs strict internal modularity with deterministic module contracts.

## Goals / Non-Goals
- Goals:
  - isolate runtime concerns into independent modules with explicit lifecycle hooks.
  - preserve current external contracts (YAML + Python + canonical channels).
  - reduce coupling so module evolution rarely requires orchestrator edits.
  - keep hot-path performance discipline and deterministic outputs.
- Non-Goals:
  - introducing new user-facing analysis features in the first extraction phase.
  - changing canonical channel naming contracts.
  - rewriting all solver mathematics in this change.

## Decisions

### 1) Introduce Runtime Module Lifecycle Interface
- Decision:
  - add a typed module interface with hooks:
    - `on_run_initialize`
    - `on_step_attempt`
    - `on_step_accepted`
    - `on_sample_emit`
    - `on_finalize`
- Rationale:
  - cleanly separates policy ownership and removes ad-hoc cross-cut code paths.

### 2) Deterministic Module Dependency Graph
- Decision:
  - each module declares required capabilities and produced capabilities.
  - runtime resolves deterministic execution order or fails with typed diagnostics.
- Rationale:
  - avoids hidden order dependencies and makes plugin growth safe.

### 3) Module-Owned Channel/Telemetry Registration
- Decision:
  - channels and metadata are registered through a module channel registry contract.
  - summary reductions remain deterministic and validated centrally.
- Rationale:
  - clear ownership and easier extension of new channel families.

### 4) Non-Breaking Compatibility Envelope
- Decision:
  - preserve existing external outputs and parser behavior during extraction.
- Rationale:
  - architecture refactor should not break user workflows.

### 5) Module-Scoped Quality Gates
- Decision:
  - add tests and KPI gates at module boundaries:
    - deterministic output order,
    - no hot-path unplanned allocations,
    - compatibility of summaries vs channels.
- Rationale:
  - catches regressions where they originate.

## Proposed Module Topology

1. `events_topology_module`
- calendar boundaries, event refinement, topology signature transitions.

2. `control_mixed_domain_module`
- virtual block execution and sampled-data scheduler semantics.

3. `loss_module`
- switching/conduction/recovery accounting and loss channels.

4. `thermal_module`
- thermal network stepping, coupling, and thermal channels.

5. `telemetry_channels_module`
- channel metadata registration and summary consistency checks.

6. `analysis_module` (follow-up extraction target)
- periodic/small-signal/frequency-analysis kernels.

## Risks / Trade-offs
- Risk: temporary complexity increase while migrating existing logic.
  - Mitigation: phase extraction with behavior-locked tests at every phase.
- Risk: module interfaces add call overhead in hot loops.
  - Mitigation: pre-bound vectors/spans, reserve patterns, and benchmark gates.
- Risk: hidden ordering assumptions break after modularization.
  - Mitigation: explicit dependency declarations + deterministic resolver diagnostics.

## Migration Plan
1. Extract module interfaces and dependency resolver.
2. Wrap existing behavior in adapter modules without changing outputs.
3. Gradually relocate implementation from orchestrator into module packages.
4. Add module-focused tests and KPI thresholds.
5. Keep compatibility checks for existing channel and summary contracts.

## Open Questions
- Should module loading be compile-time only at first, or include runtime dynamic loading in a later phase?
- Should extension registry add a dedicated runtime-module category now or in follow-up?
