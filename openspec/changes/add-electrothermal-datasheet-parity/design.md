## Context
Pulsim already exports canonical thermal traces (`T(component)`) and component electrothermal summaries. That solved observability and baseline contract stability, but it does not yet fully cover professional semiconductor-loss workflows:

- limited switching-loss event coverage for some closed-loop forced semiconductor paths;
- scalar loss data not sufficient for datasheet-grade dependency on current/voltage/temperature and gate condition;
- simplified thermal network semantics for advanced junction/case/sink use cases;
- frontend still needs a clear boundary to avoid implementing physics logic.

This design defines a backend-first electrothermal architecture where all numerical physics and validation stay in core, and GUI remains a presentation/input layer.

## Goals / Non-Goals
- Goals:
  - Backend computes professional-grade conduction + switching losses from structured characterization data.
  - Backend computes thermal trajectories from explicit thermal networks with deterministic behavior.
  - Backend exports complete, time-aligned channels for loss and temperature, with metadata for UI routing.
  - Backend contract is complete enough for headless and scripted operation without GUI heuristics.
  - Existing users keep backward compatibility with current scalar fields and summaries.
- Non-Goals:
  - Recreating proprietary internal algorithms of third-party tools verbatim.
  - Building GUI editors/wizards in this change.
  - Introducing electromagnetic FEA or package-level CFD.

## Professional Parity Targets
The capability is considered production-ready when all targets below are met:

1. Switching losses are non-zero and physically coherent in closed-loop switching examples where events exist.
2. Conduction/switching/recovery are separately observable per component over time.
3. Junction temperature reacts continuously to dissipated power using selected thermal network model.
4. Summary values are exact reductions of exported time-series under deterministic reduction rules.
5. Runtime remains allocation-disciplined in hot loops and gated by regression thresholds.

## Architecture Decisions

### 1) Backend-Owned Electrothermal Physics
- Decision: all loss and thermal computation resides in kernel services.
- Rationale: deterministic physics and reproducibility cannot depend on GUI implementation choices.
- Consequence: GUI may assist data entry, but cannot invent or post-process physical quantities.

### 2) Two-Tier Loss Modeling
- Decision: support both `scalar` and `datasheet` loss modes.
- `scalar` mode preserves current behavior (`eon/eoff/err` constants + analytic conduction).
- `datasheet` mode adds interpolated surfaces and explicit operating-condition mappings.
- Rationale: backward compatibility + progressive precision.

### 3) Unified Event Coverage for Switching Loss
- Decision: event detector and switching-loss commit path must cover:
  - native `VoltageControlledSwitch` transitions;
  - forced-state `MOSFET`/`IGBT` transitions (for example, via `pwm_generator.target_component`);
  - diode reverse-recovery transitions where configured.
- Rationale: avoid under-reporting switching loss in realistic closed-loop topologies.

### 4) Thermal Network Families as Runtime Primitives
- Decision: thermal solver supports `single_rc`, `foster`, and `cauer` as first-class per-device network types.
- Decision: optional shared thermal coupling (for example shared sink node) is supported with deterministic topology.
- Rationale: professional workflows need more than single-lumped RC for package/heatsink interactions.

### 5) Canonical Electrothermal Waveform Contract
- Decision: export canonical per-component time-series channels with metadata:
  - `T(<name>)` (junction temperature canonical alias)
  - `Pcond(<name>)`
  - `Psw_on(<name>)`
  - `Psw_off(<name>)`
  - `Prr(<name>)`
  - `Ploss(<name>)`
- All channels are aligned to `result.time` and covered by metadata domain/unit/source fields.
- Rationale: frontends and tooling should plot directly without heuristics.

### 6) Deterministic Reduction Rules
- Decision: summary fields are deterministic reductions of emitted samples.
- Example:
  - `final_temperature == last(T(...))`
  - `peak_temperature == max(T(...))`
  - `average_temperature == mean(T(...))`
  - loss summaries equal integrals/averages of exported `P*` channels under documented rules.

### 7) Hot-Path Performance Discipline
- Decision: no unplanned per-step dynamic allocation after channel/service warm-up.
- Decision: interpolation structures are prevalidated and preindexed before stepping.
- Rationale: richer physics cannot regress simulator responsiveness.

## Data Model (Conceptual)

### Loss Characterization
- `mode`: `scalar | datasheet`
- `conduction_model`:
  - analytic coefficients or LUT (`I`, `T`) for equivalent `Rds_on`, `Vce_sat`, or `Vf`
- `switching_model`:
  - LUTs for `Eon`, `Eoff`, `Err` over axes (`I`, `V`, `T`)
  - optional gate-condition scaling (`rg_ref`, `rg_on`, `rg_off`, `k_rg_*`)
  - deterministic interpolation/extrapolation policy (`clamp` by default)

### Thermal Characterization
- `network_kind`: `single_rc | foster | cauer`
- `stages`: ordered thermal stages
- optional coupling descriptor (`shared_sink_id`, thermal resistance/capacitance coupling terms)
- initial conditions (`temp_init`) and references (`temp_ref`, `alpha`)

## Numerical Strategy

### Loss Evaluation Loop
1. Determine instantaneous operating point per component (`I`, `V`, state, `Tj`).
2. Compute conduction power from conduction model.
3. On switching events, sample switching-energy model and commit impulse energy.
4. Build per-step power channels (`Pcond`, `Psw_*`, `Ploss`) in deterministic order.

### Thermal Update Loop
1. Aggregate dissipated power into thermal source nodes.
2. Advance thermal state for configured network kind using stable discrete update.
3. Update temperature-dependent electrical scaling where policy enables it.
4. Emit thermal channels and summary accumulators from the same accepted-step timeline.

## Backend vs GUI Responsibilities (Explicit)

### Backend MUST own
- Physics models for conduction/switching/recovery and thermal evolution.
- Validation of parameter ranges, table dimensions, and schema consistency.
- Event detection, energy accounting, and deterministic sample/reduction contracts.
- Canonical channels and metadata for plotting/telemetry.
- Backward-compatible migration behavior for old netlists.

### GUI SHOULD own
- User input UX (forms/wizards for model parameters).
- Optional curve digitization/import assistant UX (CSV/PDF workflow support at UI level).
- Unit helper UI, presets, and validation messaging presentation.
- Plot organization, dashboards, and interaction controls.

### GUI MUST NOT own
- Computing synthetic loss/thermal curves.
- Reconstructing switching loss from electrical channels on its own.
- Overriding backend physical results with heuristic post-processing.

## Backward Compatibility and Migration
- Existing `loss` scalar fields remain valid.
- Existing `thermal` single RC fields remain valid.
- Existing `loss_summary`, `thermal_summary`, and `component_electrothermal` remain available.
- New richer blocks/channels are additive.
- Strict mode provides deterministic migration diagnostics where data is incomplete/invalid.

## Validation and Benchmarks
- Add closed-loop buck parity validation with non-zero switching loss in semiconductor targets.
- Add per-component minimum circuits with theoretical thermal expected traces.
- Add integration tests for channel existence, length alignment, and summary consistency.
- Add performance KPI gates for electrothermal runs with datasheet tables.

## Risks / Trade-offs
- Risk: richer models increase parser and runtime complexity.
  - Mitigation: keep scalar mode unchanged; isolate datasheet mode paths.
- Risk: larger result payloads due to additional channels.
  - Mitigation: predictable channel naming, optional channel filtering policy, preallocation.
- Risk: out-of-range interpolation can hide data-quality issues.
  - Mitigation: strict diagnostics and explicit clamp/extrapolation policy options.

## Rollout Plan
1. Add YAML + Python surfaces for new characterization structures.
2. Implement kernel loss-event coverage and datasheet interpolation path.
3. Implement thermal-network extensions and coupling.
4. Export canonical loss channels with metadata and consistency checks.
5. Gate with converter + analytic + performance tests.
6. Publish user/contributor docs including GUI responsibility boundaries.

## Open Questions
- Whether shared thermal coupling should launch with full network matrix or staged pairwise coupling.
- Whether optional channel filtering should be runtime-configurable in first release or follow-up.
- Whether to provide built-in curve-fit helpers in backend or keep fitting as offline tooling.
