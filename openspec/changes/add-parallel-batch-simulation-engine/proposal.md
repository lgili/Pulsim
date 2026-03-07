## Why
PSIM/PLECS expose this capability as part of their standard professional workflow, while Pulsim currently lacks a first-class contract for it. Closing this gap is required to position Pulsim as a serious simulator for industrial power-electronics development.

Reference baseline from vendor feature pages:
- PSIM: https://altair.com/psim/
- PSIM AC Sweep / analysis workflows: https://altair.com/resource/frequency-analysis-ac-sweeps-with-psim
- PLECS product and analysis tools: https://www.plexim.com/products/plecs and https://www.plexim.com/products/plecs/analysis_tools

## What Changes
- Introduce a formal capability contract for: parallel execution of independent simulations/analyses with deterministic aggregation and reproducible ordering
- Define deterministic runtime behavior and diagnostics for this feature.
- Define outputs/telemetry suitable for backend, Python, and benchmarking integration.

## Impact
- Affected specs: benchmark-suite, python-bindings, kernel-v1-core
- Affected code: benchmarks/*, python/pulsim/*, runtime scheduling modules
