# Contributing to PulsimCore

Thank you for contributing.

This repository is a Python-first power-electronics simulator with a C++ runtime kernel. Contributions should preserve three invariants:

1. Deterministic results.
2. Stable runtime contracts (Python + YAML + `SimulationResult`).
3. High-performance transient execution.

## Development Setup

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DPULSIM_BUILD_PYTHON=ON
cmake --build build -j
```

## Minimum Validation Before PR

```bash
# Python runtime contract tests
PYTHONPATH=build/python pytest python/tests -v --ignore=python/tests/validation

# Core C++ tests
ctest --test-dir build --output-on-failure
```

For backend changes in transient flow, also run:

```bash
ctest --test-dir build --output-on-failure -R "^v1 "
```

## Backend Coding Guidelines

- Keep hot-path code allocation-free when possible.
- Avoid string construction and map growth inside the per-step simulation loop.
- Prefer precomputation/caching outside `while (t < tstop)` in transient runtime.
- Keep diagnostic and fallback behavior deterministic.
- Use explicit finite-value guards for numeric inputs and runtime state.
- Keep public contracts backward compatible unless a migration note is included.

## Performance Rules of Thumb

- Measure first, optimize second.
- Prioritize algorithmic and memory-layout improvements before micro-optimizations.
- Preserve readability in critical kernels with short, precise comments.
- Do not claim speedups without reproducible benchmark evidence.

## Thermal and Control Contract

Frontend and API consumers rely on canonical channels and metadata.

- Thermal traces: `T(<component_name>)` in `SimulationResult.virtual_channels`.
- Channel length must match `result.time`.
- Keep consistency with `thermal_summary` and `component_electrothermal`.
- Use `virtual_channel_metadata` to classify `domain`, `unit`, and `source_component`.

See:

- `docs/frontend-control-signals.md`
- `docs/electrothermal-workflow.md`
- `docs/backend-cpp-contributor-guide.md`

## Doxygen Documentation

Backend `.cpp` files are part of the reference docs and must stay documented.

Generate local Doxygen output with:

```bash
cd docs
doxygen Doxyfile
```

When changing backend runtime behavior, update function/file Doxygen blocks in `core/src`.

## Pull Request Checklist

- Tests pass locally.
- Docs updated for behavior/contract changes.
- New fields/channels have runtime and binding coverage.
- Performance-sensitive changes include benchmark or telemetry evidence.
- Commit messages explain behavior impact, not only refactor intent.
