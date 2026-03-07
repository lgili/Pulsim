## Why
PSIM/PLECS expose averaged converter modeling as a core workflow for fast control-loop design and plant iteration. Pulsim currently requires fully switched transient setups for those studies, which increases runtime, makes loop iteration slower, and complicates deterministic benchmark gating for controller development.

To be competitive for professional power-electronics teams, Pulsim needs a first-class averaged-model contract with explicit validity envelope, deterministic diagnostics, and benchmarked fidelity against switching references.

Reference baseline from vendor feature pages:
- PSIM: https://altair.com/psim/
- PSIM AC Sweep / analysis workflows: https://altair.com/resource/frequency-analysis-ac-sweeps-with-psim
- PLECS product and analysis tools: https://www.plexim.com/products/plecs and https://www.plexim.com/products/plecs/analysis_tools

## What Changes
- Add a canonical backend-owned averaged-converter mode for transient/control workflows.
- Define deterministic switching-to-averaged parameter mapping contract (topology + mapped elements + duty input source).
- Define explicit operating-envelope policy (first release: CCM-focused) with typed out-of-envelope diagnostics.
- Define structured result/telemetry contract for averaged states and mapping summary.
- Expose YAML + Python surfaces and benchmark gates for fidelity, determinism, and runtime speedup.

## Non-Goals (This Change)
- Full arbitrary symbolic model-order reduction for any topology.
- Automated controller synthesis/tuning (covered by `add-controller-auto-design-suite`).
- Multi-rate FPGA/HIL real-time scheduling semantics (covered by `add-real-time-hil-execution-mode`).

## Implementation Gates (Definition of Done)
- Gate G1: Contract completeness
  - YAML schema, runtime options, and Python API contract are fully specified and validated.
- Gate G2: Physical fidelity envelope
  - Averaged-mode outputs stay within configured error envelopes versus switching-reference cases in supported operating regimes.
- Gate G3: Determinism
  - Repeated runs with the same inputs produce deterministic averaged-state trajectories and diagnostics.
- Gate G4: Performance value
  - Averaged-mode benchmark scenarios meet minimum runtime improvement thresholds versus switching references.
- Gate G5: Integration readiness
  - Telemetry and docs are sufficient for frontend/report workflows without inference heuristics.

## Impact
- Affected specs:
  - `kernel-v1-core`
  - `netlist-yaml`
  - `python-bindings`
  - `benchmark-suite`
- Affected code:
  - `core/include/pulsim/v1/simulation.hpp`
  - `core/src/v1/simulation*.cpp`
  - `core/src/v1/yaml_parser.cpp`
  - `python/bindings.cpp`
  - `python/pulsim/*`
  - `benchmarks/*`
  - `python/tests/*`
  - `docs/*`

## Risks and Mitigations
- Risk: Averaged model used outside validity envelope and silently misleads control design.
  - Mitigation: explicit envelope policy + typed diagnostics + mandatory benchmark fidelity gates.
- Risk: Runtime mode divergence from switching reference semantics.
  - Mitigation: deterministic mapping contract + paired benchmark scenarios with bounded error KPIs.
- Risk: Feature introduces hidden complexity in frontend integration.
  - Mitigation: backend-owned structured telemetry/metadata and explicit frontend responsibility boundaries in docs.

## Acceptance Evidence
Execution evidence (fidelity KPIs, runtime speedup KPIs, determinism checks, known limitations, and envelope boundaries) is recorded in:

- `openspec/changes/add-averaged-converter-modeling/change-notes.md`
