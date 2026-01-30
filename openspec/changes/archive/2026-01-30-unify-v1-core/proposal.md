## Why

PulsimCore currently has two divergent kernels: the runtime core in `core/` (CLI/JSON/event pipeline) and the high-performance kernel in `pulsim/v1` (used by Python). This duplication creates inconsistent behavior, split validation, and blocks robust convergence + performance improvements from becoming the single source of truth.

## What Changes

- Unify the simulator around `pulsim/v1` as the only core engine (single pipeline).
- Implement a full `v1::Simulator` with DC, transient, adaptive timestep, events, and loss tracking.
- Replace JSON netlists with **versioned YAML** (YAML required).
- Integrate `yaml-cpp` via CMake FetchContent.
- Keep a simplified Python API for initial validation (expand later).

## Impact

- **BREAKING**: JSON netlist support removed; YAML becomes mandatory.
- Affected specs: `kernel-v1-core`, `netlist-yaml`.
- Affected code: `core/include/pulsim/v1/`, `core/src/`, parser, Python bindings, docs/examples.
