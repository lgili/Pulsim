## Why
Professional users expect nonlinear magnetic-core modeling as a first-class capability for converter and magnetics design. PSIM and PLECS expose this workflow directly (saturation, hysteresis, and core-loss studies), while Pulsim currently has only partial magnetic behavior coverage and no backend-complete contract for nonlinear core physics across YAML/kernel/Python/benchmark surfaces.

Without a canonical contract, teams rely on ad-hoc approximations and external scripts, which weakens reproducibility, CI gating, and frontend consistency.

Reference baseline from vendor feature pages:
- PSIM: https://altair.com/psim/
- PSIM AC Sweep / analysis workflows: https://altair.com/resource/frequency-analysis-ac-sweeps-with-psim
- PLECS product and analysis tools: https://www.plexim.com/products/plecs and https://www.plexim.com/products/plecs/analysis_tools

## What Changes
- Add a canonical nonlinear magnetic-core modeling contract for inductor/transformer workflows, including:
  - saturation behavior (flux/current dependent)
  - hysteresis behavior with explicit loop-state semantics
  - frequency-dependent core-loss modeling
- Add strict YAML validation and deterministic diagnostics for invalid/unsupported magnetic-core configurations.
- Add kernel runtime support with deterministic telemetry channels and metadata for magnetic-core states/losses.
- Add Python typed surfaces to configure/read nonlinear magnetic-core models and diagnostics.
- Add benchmark/KPI gates for physical fidelity, determinism, and runtime overhead.
- Add documentation and examples describing backend/frontend boundaries and user workflows.

## Non-Goals (This Change)
- Full FEM-grade spatial magnetic simulation.
- Automatic magnetic component synthesis/optimization.
- GUI-only magnetic calculations detached from backend contract.

## Implementation Gates (Definition of Done)
- Gate G1: Contract completeness
  - YAML, kernel, Python, and benchmark contracts are fully specified and validated.
- Gate G2: Physical fidelity envelope
  - Saturation/hysteresis/core-loss outputs pass declared error envelopes on canonical fixtures.
- Gate G3: Determinism
  - Repeated runs with identical setup produce stable magnetic-core channels, summaries, and diagnostics.
- Gate G4: Performance safety
  - Runtime/allocation overhead for nonlinear magnetic workflows remains within benchmark thresholds.
- Gate G5: Integration readiness
  - Structured outputs and docs are sufficient for frontend/reporting without name-heuristic reconstruction.

## Impact
- Affected specs:
  - `device-models`
  - `netlist-yaml`
  - `kernel-v1-core`
  - `python-bindings`
  - `benchmark-suite`
- Affected code (planned):
  - `core/include/pulsim/v1/components/*`
  - `core/src/v1/yaml_parser.cpp`
  - `core/src/v1/simulation*.cpp`
  - `core/include/pulsim/v1/simulation.hpp`
  - `python/bindings.cpp`
  - `python/pulsim/*`
  - `python/tests/*`
  - `benchmarks/*`
  - `docs/*`
  - `examples/*`

## Risks and Mitigations
- Risk: users run outside model validity envelope and trust wrong results.
  - Mitigation: explicit model-family validity semantics, strict diagnostics, and benchmark fidelity gates.
- Risk: hysteresis state updates introduce non-determinism.
  - Mitigation: deterministic state-update ordering, explicit initialization semantics, and repeat-run determinism gates.
- Risk: nonlinear magnetic models slow down transient simulation.
  - Mitigation: bounded-state implementation, reuse of scratch buffers, and KPI runtime/allocation gates.
- Risk: frontend reconstructs magnetic metrics inconsistently.
  - Mitigation: backend-owned canonical channels + metadata + documented frontend non-responsibilities.

## Acceptance Evidence
Execution evidence (fixtures, tolerance thresholds, KPI reports, determinism checks, and known limits) is recorded in:

- `openspec/changes/add-magnetic-core-nonlinear-models/change-notes.md`
