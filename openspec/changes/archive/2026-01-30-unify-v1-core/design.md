## Context

We need a single, robust simulation core that is fast, correct, and Python-friendly. Today the runtime core (`core/`) provides events and netlist parsing but uses a simpler solver, while `pulsim/v1` contains the advanced numerical algorithms and is already used by Python bindings. The duplication prevents consistent validation and evolution.

## Goals / Non-Goals

- Goals:
  - Single authoritative kernel based on `pulsim/v1` runtime circuit.
  - Full simulator pipeline (DC + transient + adaptive timestep + events + losses).
  - YAML-only netlist with schema versioning.
  - Deterministic results with fixed ordering.
- Non-Goals:
  - New device models (BSIM, etc.).
  - Distributed/parallel simulation.
  - GUI or gRPC changes beyond required API compatibility.

## Decisions

- Decision: Use `pulsim/v1::Circuit` runtime (variant-based) as the sole runtime IR.
  - Why: Already used by Python and supports runtime construction.
- Decision: Build `v1::Simulator` in C++ that mirrors the feature set of `core::Simulator`.
  - Why: Consolidates events, losses, and adaptive stepping into the single kernel.
- Decision: Adopt YAML as the only netlist format and require `version` field.
  - Why: Compact syntax and explicit schema evolution.
- Decision: Require explicit `schema` identifier for YAML netlists.
  - Why: Avoid ambiguity across future schema versions and tools.
- Decision: Support model inheritance with local overrides.
  - Why: Enables flexible reuse while keeping component definitions concise.
- Decision: Integrate `yaml-cpp` via FetchContent.
  - Why: Avoid maintaining a custom YAML subset parser.

## Risks / Trade-offs

- Risk: Breaking change for JSON netlists.
  - Mitigation: Provide a migration guide and examples in YAML.
- Risk: Regressions during kernel unification.
  - Mitigation: Use existing validation suite (71 tests) as gates.
- Risk: YAML schema drift.
  - Mitigation: Require `version` and validate fields strictly.

## Migration Plan

1. Implement `v1::Simulator` with DC + transient + events + losses.
2. Add YAML parser that builds `v1::Circuit` and `SimulationOptions`.
3. Port event/loss logic from `core::Simulator` into v1.
4. Update examples and docs to YAML.
5. Update Python bindings to use `v1::Simulator` path (simplified API).
6. Deprecate/remove JSON and legacy simulator paths.

## Open Questions

- Exact YAML schema details for models, waveforms, and aliases.
- Whether to keep a lightweight JSON-to-YAML conversion helper.
