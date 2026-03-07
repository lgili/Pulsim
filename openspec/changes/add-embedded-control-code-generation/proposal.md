## Why
PSIM/PLECS expose this capability as part of their standard professional workflow, while Pulsim currently lacks a first-class contract for it. Closing this gap is required to position Pulsim as a serious simulator for industrial power-electronics development.

Reference baseline from vendor feature pages:
- PSIM: https://altair.com/psim/
- PSIM AC Sweep / analysis workflows: https://altair.com/resource/frequency-analysis-ac-sweeps-with-psim
- PLECS product and analysis tools: https://www.plexim.com/products/plecs and https://www.plexim.com/products/plecs/analysis_tools

## What Changes
- Introduce a formal capability contract for: automatic generation of deterministic C code from control diagrams for embedded targets and rapid control prototyping
- Define deterministic runtime behavior and diagnostics for this feature.
- Define outputs/telemetry suitable for backend, Python, and benchmarking integration.

## Impact
- Affected specs: python-bindings, kernel-v1-core, benchmark-suite
- Affected code: python/pulsim/*, core/include/pulsim/v1/runtime_circuit.hpp, core/src/v1/runtime_module_adapters.cpp, tooling/codegen/*
