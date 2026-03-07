## Why
PSIM and PLECS provide built-in waveform analysis as a standard workflow layer (FFT, THD, RMS, power/efficiency, loop/response measurements). Pulsim currently depends on ad-hoc external scripts, which creates drift in metric definitions, weak reproducibility, and inconsistent frontend behavior.

To be competitive for professional power-electronics workflows, Pulsim needs a first-class, backend-owned waveform post-processing contract with deterministic diagnostics, benchmark gates, and explicit frontend consumption rules.

Reference baseline from vendor feature pages:
- PSIM: https://altair.com/psim/
- PSIM AC Sweep / analysis workflows: https://altair.com/resource/frequency-analysis-ac-sweeps-with-psim
- PLECS product and analysis tools: https://www.plexim.com/products/plecs and https://www.plexim.com/products/plecs/analysis_tools

## What Changes
- Add a canonical waveform post-processing pipeline for transient/frequency results:
  - time-domain metrics (RMS, mean, min/max, peak-to-peak, crest factor, ripple, settling/overshoot where applicable)
  - spectral metrics (FFT bins, harmonic table, THD)
  - power metrics (average power, apparent/reactive power, power factor, efficiency)
  - loop/response metrics from response arrays where applicable
- Define strict configuration and diagnostic contract (invalid windows, sampling preconditions, undefined metrics, missing signals).
- Define structured result surface and metadata for Python/frontend consumption (no regex parsing / no UI-side metric fabrication).
- Define benchmark and KPI gates for metric correctness, determinism, and runtime overhead bounds.
- Document backend/frontend responsibilities and migration path from script-based post-processing.

## Non-Goals (This Change)
- Symbolic model identification or automatic control synthesis.
- Replacing AC-sweep solver logic; this change focuses on post-processing of available signals/results.
- GUI-only calculations detached from backend contract.

## Implementation Gates (Definition of Done)
- Gate G1: Contract completeness
  - YAML/runtime/Python configuration and result schema are fully specified with strict validation semantics.
- Gate G2: Metric correctness
  - FFT/THD/time-domain/power metrics meet analytical/reference tolerances on canonical fixtures.
- Gate G3: Determinism
  - Repeated identical runs produce stable metric outputs and diagnostic behavior.
- Gate G4: Performance value
  - Post-processing runtime overhead remains within declared KPI thresholds.
- Gate G5: Integration readiness
  - Frontend/docs/benchmark artifacts consume structured outputs without heuristic reconstruction.

## Impact
- Affected specs:
  - `kernel-v1-core`
  - `netlist-yaml`
  - `python-bindings`
  - `benchmark-suite`
- Affected code (planned):
  - `core/include/pulsim/v1/simulation.hpp`
  - `core/src/v1/*post_processing*`
  - `core/src/v1/yaml_parser.cpp`
  - `python/bindings.cpp`
  - `python/pulsim/*`
  - `benchmarks/*`
  - `python/tests/*`
  - `docs/*`
  - `examples/*`

## Risks and Mitigations
- Risk: metric inconsistency across teams/tools.
  - Mitigation: backend-owned canonical definitions and typed outputs.
- Risk: false trust from invalid FFT/sampling setup.
  - Mitigation: strict prerequisite diagnostics and undefined-metric reason codes.
- Risk: excessive runtime overhead in CI/interactive workflows.
  - Mitigation: KPI overhead gate and allocation-aware implementation requirements.
- Risk: frontend recreates metrics differently.
  - Mitigation: explicit frontend boundary: consume backend metric outputs directly.

## Acceptance Evidence
Execution evidence (fixtures, tolerance thresholds, KPI reports, determinism checks, and known limits) is recorded in:

- `openspec/changes/add-waveform-post-processing-tools/change-notes.md`
