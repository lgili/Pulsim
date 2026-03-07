## Why
Frequency-domain analysis is a core workflow for professional power-electronics development (loop design, stability margins, input/output impedance shaping). Today Pulsim has strong transient and electrothermal capabilities, but it still lacks a first-class AC/FRA contract that users expect from PSIM/PLECS-class tools.

Without this capability, users must export waveforms and build custom scripts/manual perturbation setups, which is slow, error-prone, and hard to reproduce in CI.

Reference baseline from vendor feature pages:
- PSIM: https://altair.com/psim/
- PSIM AC Sweep / analysis workflows: https://altair.com/resource/frequency-analysis-ac-sweeps-with-psim
- PLECS product and analysis tools: https://www.plexim.com/products/plecs and https://www.plexim.com/products/plecs/analysis_tools

## What Changes
- Add a backend-owned frequency-domain analysis engine for:
  - open-loop transfer functions
  - closed-loop transfer functions
  - impedance sweeps (at least input and output)
- Add deterministic operating-point anchoring (`dc`, `periodic`, `averaged`, `auto`) so switching converters can be analyzed without manual averaged-model derivation.
- Add canonical sweep contract (linear/log grid, frequency limits, point count, perturbation amplitude, injection and measurement definitions).
- Add strict validation and deterministic diagnostics for unsupported topologies/configurations and malformed sweep definitions.
- Add canonical result contract with complex response data and derived metrics (magnitude/phase and crossover/margin metrics where defined).
- Expose complete YAML + Python surfaces and benchmark/CI gates for accuracy, determinism, and performance.

## Non-Goals (This Change)
- GUI plotting/layout behavior (front-end responsibility).
- Full symbolic control-design synthesis (covered by `add-controller-auto-design-suite`).
- EMI/EMC compliance analysis beyond standard control/impedance frequency-response workflows.

## Implementation Gates (Definition of Done)
- Gate G1: Contract completeness
  - YAML schema, kernel API contract, and Python result structures are finalized and documented.
- Gate G2: Numerical correctness
  - Analytical/reference circuits pass configured magnitude/phase error thresholds.
- Gate G3: Determinism
  - Repeated runs on same machine class and same seed/config produce stable frequency grids and bounded numeric drift.
- Gate G4: Performance safety
  - Runtime and allocation KPIs for AC sweep scenarios stay within approved benchmark thresholds.
- Gate G5: Integration readiness
  - Benchmark artifacts and docs expose enough structured data for frontend/reporting without text parsing or heuristics.

## Impact
- Affected specs:
  - `kernel-v1-core`
  - `netlist-yaml`
  - `python-bindings`
  - `benchmark-suite`
- Affected code:
  - `core/include/pulsim/v1/simulation.hpp`
  - `core/src/v1/simulation*.cpp`
  - `core/src/v1/transient_services.cpp` (shared anchoring/linearization hooks where applicable)
  - `core/src/v1/yaml_parser.cpp`
  - `python/bindings.cpp`
  - `python/pulsim/*`
  - `benchmarks/*`
  - `python/tests/*`
  - `docs/*`

## Risks and Mitigations
- Risk: Incorrect small-signal extraction for strongly switched systems.
  - Mitigation: explicit anchor modes + benchmark parity against known references + strict unsupported-case diagnostics.
- Risk: Feature becomes non-deterministic due to adaptive internals.
  - Mitigation: deterministic sweep grid contract, bounded interpolation policy, and CI determinism gates.
- Risk: Runtime overhead on large sweep campaigns.
  - Mitigation: caching/reuse strategy and KPI performance gates in benchmark CI.

## Acceptance Evidence
Execution evidence, KPI gate outputs, thresholds, and known limitations are recorded in:

- `openspec/changes/add-frequency-domain-ac-sweep-analysis/change-notes.md`
