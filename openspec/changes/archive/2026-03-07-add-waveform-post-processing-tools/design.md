## Context
Pulsim requires a deterministic, backend-owned waveform post-processing layer so users do not depend on external notebooks/scripts for core metrics (FFT/THD/RMS/power/efficiency/loop measurements). The layer must be modular, benchmark-gated, and frontend-consumable without heuristic reconstruction.

## Goals / Non-Goals
- Goals:
  - Add canonical post-processing contracts for transient and frequency-domain result surfaces.
  - Standardize metric definitions and typed diagnostics.
  - Ensure deterministic behavior and bounded runtime overhead.
  - Provide structured outputs for Python and frontend workflows.
- Non-Goals:
  - New simulation solver families.
  - Automatic control design/tuning.
  - GUI-only proprietary metric paths not reproducible in backend/headless mode.

## Decisions
- Decision: Backend-owned post-processing service
  - Introduce a dedicated kernel service/module for waveform post-processing, fed by canonical result channels.
  - Rationale: single source of truth for metric computation and diagnostics.

- Decision: Explicit analysis jobs with deterministic ordering
  - Configuration is expressed as explicit post-processing jobs (FFT, THD, metric-set, efficiency, loop-metric).
  - Jobs execute in stable insertion order with canonical job identifiers.
  - Rationale: reproducibility and predictable artifact diffs.

- Decision: Strict sampling/window precondition checks
  - Define deterministic prerequisite checks for window bounds, minimum samples, and spectral conditions.
  - Undefined metrics are represented via typed reason codes, not silent NaN-only behavior.
  - Rationale: avoid silent misuse and inconsistent external script behavior.

- Decision: Unified typed result surface
  - Expose scalar metrics, spectra, harmonic tables, and diagnostic fields in structured objects.
  - Include metadata (units/domain/source) and machine-stable keys.
  - Rationale: frontend and benchmark tooling must not parse free-form logs.

- Decision: KPI gate inclusion from day 1
  - Require benchmark gates for correctness, determinism, and overhead.
  - Rationale: post-processing regressions are often subtle and must be caught in CI.

## Proposed Runtime Architecture
1. Signal Resolver
   - Resolves requested signal sources from transient/frequency result channels.
   - Verifies existence, unit compatibility, and sample-domain constraints.
2. Window Planner
   - Converts config (`time`, `index`, `cycle`) into deterministic sample ranges.
   - Emits typed failures on empty/invalid windows.
3. Metric Engines
   - Time metrics engine (RMS/mean/min/max/p2p/crest/ripple/settling).
   - Spectral engine (FFT bins/harmonics/THD, window functions, leakage-aware definitions).
   - Power/efficiency engine (Pin/Pout aggregates and derived factors).
   - Loop-metric engine (crossovers/margins from response arrays when present).
4. Result Assembler
   - Produces canonical post-processing result object with ordered outputs and metadata.
5. Telemetry and Diagnostics
   - Emits deterministic diagnostic codes and performance counters for benchmark gating.

## Diagnostics Taxonomy (Planned)
- `PostProcessingInvalidConfiguration`
- `PostProcessingSignalNotFound`
- `PostProcessingInvalidWindow`
- `PostProcessingInsufficientSamples`
- `PostProcessingSamplingMismatch`
- `PostProcessingUndefinedMetric`
- `PostProcessingNumericalFailure`

## Performance / Determinism Strategy
- Reuse scratch buffers between jobs in a run.
- Use stable job ordering and deterministic tie-breaking for spectral peak/fundamental selection.
- Avoid locale/timezone/system-clock dependence in metric computations.
- Keep floating-point tolerances explicit and tested in CI.

## Migration Plan
1. Define schema and runtime contracts (YAML + Python + kernel outputs).
2. Implement core metric engines with deterministic diagnostics.
3. Add benchmark fixtures and KPI thresholds.
4. Publish docs and tutorial notebook with frontend integration boundaries.
5. Phase out script-only reference workflows where canonical backend jobs exist.

## Open Questions
- Initial mandatory metric subset for MVP vs phase-2 expansion (ex: settling/overshoot variants).
- Cycle-window semantics for variable-frequency switching cases.
- Whether loop metrics should only consume frequency-analysis outputs or also time-domain derived estimators.
