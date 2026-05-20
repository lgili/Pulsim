# `pulsim::v2` — clean-slate kernel rebuild

This directory hosts the **`pulsim::v2`** kernel, built from scratch
to fix seven structural problems in `pulsim::v1` that incremental
refactors could not address economically. v1 (in
`core/include/pulsim/v1/`) stays in production; v2 grows in parallel,
layer by layer, with zero coupling and its own test binaries.

See `docs/architecture-review-v1.md` for the diagnosis and
`docs/pulsim-v2/README.md` for the layered-architecture rationale.

## Layered architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ Layer 6: Frontend (Python bindings, YAML loader, schematic in)   │
├──────────────────────────────────────────────────────────────────┤
│ Layer 5: Solver (per-step segment dispatch, Newton fallback)     │
├──────────────────────────────────────────────────────────────────┤
│ Layer 4: PWL State-Space Cache (the PLECS-killer layer)          │
├──────────────────────────────────────────────────────────────────┤
│ Layer 3: Stamping (one generic stamper, AD-driven)               │
├──────────────────────────────────────────────────────────────────┤
│ Layer 2: Device Models (math only, single source of truth, AD)   │
├──────────────────────────────────────────────────────────────────┤
│ Layer 1: Topology (graph + switch combinatorics + enumeration)   │
├──────────────────────────────────────────────────────────────────┤
│ Layer 0: Numeric Primitives + Sparse Linear Algebra              │
└──────────────────────────────────────────────────────────────────┘
```

Each layer:
- Is one subfolder under `pulsim/v2/`.
- Has its own test binary (`pulsim_v2_layerN_tests`).
- Depends ONLY on layers strictly below it (compiler-enforceable).
- Has a clean public API surface that the layer above consumes.
- Can be ripped out and replaced without touching anything else.

## Currently landed

| Layer | Subfolder                  | OpenSpec change-id                          | Status |
|-------|----------------------------|---------------------------------------------|--------|
| 0     | `numeric/` + `sparse/`     | `bootstrap-pulsim-v2-kernel`                | ✅     |
| 1     | `topology/`                | `pulsim-v2-topology-and-switch-enumeration` | pending |
| 2     | `models/`                  | `pulsim-v2-device-models-ad-driven`         | pending |
| 3     | `stamping/`                | `pulsim-v2-generic-stamping-pipeline`       | pending |
| 4     | `pwl_state_space/`         | `pulsim-v2-pwl-state-space-cache`           | pending |
| 5     | `solver/`                  | `pulsim-v2-solver-and-events`               | pending |
| 6     | `runtime/` + `frontend/`   | `pulsim-v2-circuit-builder-api` (+2 more)   | pending |

## Why parallel-namespace, not refactor-in-place

The May 2026 architecture review concluded that v1's seven
structural problems compound on each other:

1. Quadruple-duplicated device stamps (4 sites, drift between paths)
2. `std::variant<22>` AoS storage (cache-hostile, alignment bugs)
3. No PWL state-space topology caching (10-50× slower than PLECS)
4. Mixed-order integrators (effective order-1)
5. No DSL / equation layer (6-file edit per new device)
6. Single-threaded everywhere
7. Convergence band-aids piled on as default-on options

Removing any one of these in place breaks dozens of downstream
assumptions. The cumulative cost of incremental refactor exceeds
the cost of building the replacement cleanly in parallel and
switching the runtime at the API boundary when v2 reaches
feature-parity.

v1 stays production through the entire v2 build-out (estimated
~12 weeks single-developer pace). Cutover happens when v2
catches up at Layer 6, gated behind a `use_v2=True` Python flag.

## How to use v2 (today)

```cpp
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/sparse/matrix.hpp"
#include "pulsim/v2/sparse/solver.hpp"

namespace v2 = pulsim::v2;

// Build a small SPD system
v2::sparse::Matrix M(3, 3);
std::vector<v2::sparse::Triplet> t = {
    {0, 0, 4.0}, {0, 1, -1.0},
    {1, 0, -1.0}, {1, 1, 4.0}, {1, 2, -1.0},
    {2, 1, -1.0}, {2, 2, 4.0},
};
M.setFromTriplets(t.begin(), t.end());
v2::sparse::compress_in_place(M);

// Solve
auto solver = v2::sparse::make_default_solver();
solver->analyze(M);
solver->factorize(M);
v2::Vector b(3); b << 2, 4, 2;
v2::Vector x;
solver->solve(b, x);
// x = [6/7, 10/7, 6/7]
```

That's the entire surface of Layer 0 today. Layers 1-6 grow on top.
