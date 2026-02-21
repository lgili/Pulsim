## Why

Pulsim still has mixed "old vs new" surfaces (legacy source trees, stale docs, and duplicate API narratives), which makes the project harder to maintain and reduces user confidence in results.

The target product direction is now explicit:
- Python is the only supported user-facing runtime surface.
- The v1 kernel is the only simulation core.
- Converter simulation must be robust, fast, and trustworthy, including thermal behavior.
- Numerical results must be validated against external SPICE references (LTspice priority) to protect user trust.

## What Changes

- Define a Python-only product contract and remove stale user-facing CLI/C++ usage paths from docs and workflows.
- Execute a full legacy retirement plan: inventory, migrate useful features to v1, then delete legacy code and build hooks.
- Establish a converter component support matrix (electrical + thermal + loss) and close migration gaps from legacy paths.
- Add electro-thermal simulation requirements for power converter workflows.
- Upgrade validation and benchmarking to include LTspice parity runs with explicit tolerances and reproducible artifacts.
- Add stress suites from light to very large circuits, with deterministic fallback telemetry and convergence gating.
- Clean build and packaging configuration so Python build does not depend on legacy include paths or obsolete options.

## Impact

- Affected specs:
  - `kernel-v1-core`
  - `python-bindings`
  - `benchmark-suite`
  - `netlist-yaml`
  - `device-models`
- Affected code:
  - `core/legacy/**` (migration and removal)
  - `core/include/pulsim/v1/**`, `core/src/v1/**`
  - `python/bindings.cpp`, `python/CMakeLists.txt`, `python/pulsim/**`
  - `benchmarks/**`, `python/tests/validation/**`
  - `README.md`, `docs/**`

### Breaking / Behavioral Changes

- **BREAKING**: legacy code paths and deprecated APIs can be removed after migration gates pass.
- **BREAKING**: unsupported/stale interfaces documented today (CLI/grpc/JSON flows not in supported target) are removed or clearly marked unsupported.
- Benchmark parity workflow will require explicit external simulator configuration (LTspice path and mapping metadata).

## Success Criteria

1. Python package is the only supported runtime interface for end users.
2. No legacy source/include/build dependency remains in the default build and Python wheel path.
3. Converter component matrix (including thermal/loss paths) is complete for the declared support set.
4. Stress suite converges deterministically across declared benchmark tiers.
5. LTspice parity suite runs for the declared catalog and meets configured error thresholds.
6. Documentation matches actual supported workflows with no stale CLI/legacy guidance.
