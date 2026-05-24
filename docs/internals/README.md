# Pulsim — internal architecture

The Pulsim kernel is structured as **seven strictly-layered C++
modules** under `core/include/pulsim/`. Each module:

- Lives in its own subfolder (`numeric/`, `sparse/`, `topology/`,
  `models/`, `stamping/`, `pwl/`, `solver/`, `builder/`, `yaml/`).
- Depends only on layers strictly below it — compile-time enforced
  through `#include` discipline.
- Has its own Catch2 test binary (`pulsim_layerN_tests`) that links
  only `pulsim::core` + Catch2.
- Can be replaced wholesale without touching anything else.

This page is the entry point for **understanding how the kernel is
built**. End users don't need it — `docs/getting-started.md` and
`docs/mental-model.md` are the public-facing intros. Read on if you
want to extend the device library, port a new analysis, or audit
the solver.

## Layered architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ Layer 8: YAML loader        (yaml/loader.hpp)                    │
├──────────────────────────────────────────────────────────────────┤
│ Layer 7: pybind11 binding   (python/bindings.cpp, outside core/) │
├──────────────────────────────────────────────────────────────────┤
│ Layer 6: Builder API        (builder/circuit_builder.hpp)        │
├──────────────────────────────────────────────────────────────────┤
│ Layer 5: Solver             (solver/run_transient.hpp + events)  │
├──────────────────────────────────────────────────────────────────┤
│ Layer 4: PWL state-space    (pwl/cache.hpp — the PLECS killer)   │
├──────────────────────────────────────────────────────────────────┤
│ Layer 3: Stamping pipeline  (stamping/stamp_device.hpp)          │
├──────────────────────────────────────────────────────────────────┤
│ Layer 2: Device models      (models/*.hpp, AD-driven)            │
│          + motors, blockchain, analysis, switchgear, thermal     │
├──────────────────────────────────────────────────────────────────┤
│ Layer 1: Topology           (topology/graph.hpp + switch_state)  │
├──────────────────────────────────────────────────────────────────┤
│ Layer 0: Numeric + sparse   (numeric/types.hpp, sparse/*.hpp)    │
└──────────────────────────────────────────────────────────────────┘
```

## Per-layer design docs

| Layer | Subfolder       | Design doc                                                                            |
|-------|------------------|---------------------------------------------------------------------------------------|
| 0     | `numeric/`, `sparse/` | [layer0-numeric-and-sparse.md](layer0-numeric-and-sparse.md)                    |
| 1     | `topology/`     | [layer1-topology-and-enumeration.md](layer1-topology-and-enumeration.md)              |
| 2     | `models/`       | [layer2-ad-and-device-models.md](layer2-ad-and-device-models.md) — concept + first 3 devices |
|       |                 | [layer2-v1-mosfet-igbt.md](layer2-v1-mosfet-igbt.md) — MOSFET + IGBT builders          |
|       |                 | [layer2-v2-transformer.md](layer2-v2-transformer.md) — two-winding transformer         |
| 3     | `stamping/`     | [layer3-stamping-pipeline.md](layer3-stamping-pipeline.md)                            |
| 4     | `pwl/`          | [layer4-pwl-state-space-cache.md](layer4-pwl-state-space-cache.md) — base cache       |
|       |                 | [layer4-v1-trapezoidal-companion.md](layer4-v1-trapezoidal-companion.md) — dynamic devices |
|       |                 | [layer4-v2-dc-operating-point.md](layer4-v2-dc-operating-point.md)                    |
|       |                 | [layer4-v3-nonlinear-newton.md](layer4-v3-nonlinear-newton.md)                        |
|       |                 | [layer4-v4-newton-globalization.md](layer4-v4-newton-globalization.md) — line search  |
|       |                 | [layer4-v5-lm-newton.md](layer4-v5-lm-newton.md) — Levenberg-Marquardt                |
|       |                 | [layer4-v6-lazy-cache.md](layer4-v6-lazy-cache.md)                                    |
|       |                 | [layer4-v7-multi-dt-cache.md](layer4-v7-multi-dt-cache.md)                            |
|       |                 | [layer4-v8-continuation.md](layer4-v8-continuation.md) — homotopy                     |
|       |                 | [layer4-v9-vf0-continuation.md](layer4-v9-vf0-continuation.md)                        |
|       |                 | [layer4-v10-warm-start.md](layer4-v10-warm-start.md) — pseudo-transient               |
| 5     | `solver/`       | [layer5-solver-and-events.md](layer5-solver-and-events.md) — fixed-dt + bisection     |
|       |                 | [layer5-v1.5-buck-validation.md](layer5-v1.5-buck-validation.md) — buck validation (covers dynamic-device history) |
|       |                 | [layer5-v2.1-event-detection.md](layer5-v2.1-event-detection.md) — diode auto-commutation |
|       |                 | [layer5-v3-substep-correction.md](layer5-v3-substep-correction.md)                    |
| 6     | `builder/`      | [layer6-builder-api.md](layer6-builder-api.md)                                        |
| 8     | `yaml/`         | [layer8-yaml-parser.md](layer8-yaml-parser.md)                                        |
| 9     | n/a (showcases) | [layer9-smps-showcase.md](layer9-smps-showcase.md) — end-to-end converter benchmarks  |
| 10    | n/a (benchmarks)| [layer10-benchmarks.md](layer10-benchmarks.md)                                        |

The "V<n>" milestones under a single layer (e.g. Layer 4 V1..V10)
are the chronological build-out — each landed an OpenSpec change
proposal under `openspec/changes/archive/`.

## Five non-negotiable invariants

The kernel is built around five constraints that ripple through
every layer:

1. **Header-only.** No `.cpp` translation units in the kernel
   itself. The pybind11 binding (`python/bindings.cpp`) is the only
   compiled boundary.
2. **C++23.** Concepts, ranges, `mdspan`-like buffers,
   `if consteval`, and `std::expected`-style error returns.
3. **AD-driven Jacobians.** A single forward-mode AD scalar drives
   every nonlinear device's stamp; no hand-written Jacobians.
4. **PLECS-style PWL cache.** Every reachable switch configuration
   is pre-factored into a (A, B, C, D) state-space tuple; the
   transient loop is one sparse solve per step in the linear case.
5. **No globals, no singletons.** Every entity (Graph, DevicePool,
   PwlStateSpaceCache, SimulationOptions) is value-owned by the
   caller.

## Build + test

```bash
# Configure
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build everything (all layer test binaries + Python extension)
cmake --build build -j

# Run only Layer N's test binary
build/core/pulsim_layerN_tests

# Or run the whole ctest harness
ctest --test-dir build --output-on-failure
```

Each layer's test binary is independent — Layer 4 tests don't link
Layer 5 sources, etc. This is the practical proof that the
dependency graph really is layered.

## See also

- [`../mental-model.md`](../mental-model.md) — the end-user-facing
  one-page summary (Graph + DevicePool + PwlStateSpaceCache + Newton).
- [`../api-reference.md`](../api-reference.md) — the full Python
  surface in one page.
- `openspec/changes/archive/` — every OpenSpec proposal that
  shipped a layer or a layer milestone, with the design rationale
  and test coverage table.
