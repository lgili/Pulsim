## Why
PSIM/PLECS expose this capability as part of their standard professional workflow, while Pulsim currently lacks a first-class contract for it. Closing this gap is required to position Pulsim as a serious simulator for industrial power-electronics development.

Reference baseline from vendor feature pages:
- PSIM: https://altair.com/psim/
- PSIM AC Sweep / analysis workflows: https://altair.com/resource/frequency-analysis-ac-sweeps-with-psim
- PLECS product and analysis tools: https://www.plexim.com/products/plecs and https://www.plexim.com/products/plecs/analysis_tools

## What Changes
- Introduce a formal capability contract for: fault-injection scenarios (short/open/sensor/control faults) with deterministic event timing and outcome classification
- Define deterministic runtime behavior and diagnostics for this feature.
- Define outputs/telemetry suitable for backend, Python, and benchmarking integration.

## Impact
- Affected specs: benchmark-suite, kernel-v1-core, netlist-yaml
- Affected code: core/src/v1/simulation.cpp, core/src/v1/runtime_circuit.hpp, core/src/v1/yaml_parser.cpp, benchmarks/*
