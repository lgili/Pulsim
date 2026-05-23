# `pulsim` — header-only C++23 kernel

This directory is the entire Pulsim simulation engine. Every layer
(numeric primitives, sparse linear algebra, topology, AD-based
device models, the PWL state-space cache, the time-stepping solver,
the high-level builder, the YAML loader) is implemented in headers
under `pulsim/<subdir>/...` and exposed as a single INTERFACE library
target `pulsim::core`.

See `docs/internals/README.md` for the layered-architecture
rationale.

## Layered architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ Layer 8  yaml/                YAML loader → CircuitBuilder       │
├──────────────────────────────────────────────────────────────────┤
│ Layer 7  pybind11 (python/bindings.cpp, lives outside this tree) │
├──────────────────────────────────────────────────────────────────┤
│ Layer 6  builder/             CircuitBuilder (string node names) │
├──────────────────────────────────────────────────────────────────┤
│ Layer 5  solver/              run_transient + event detection    │
├──────────────────────────────────────────────────────────────────┤
│ Layer 4  pwl/                 PWL state-space cache + Newton     │
├──────────────────────────────────────────────────────────────────┤
│ Layer 3  stamping/            Templated MNA stamp pipeline       │
├──────────────────────────────────────────────────────────────────┤
│ Layer 2  models/              Device models (R, L, C, MOSFET, …) │
│          motors/              DC, BLDC, PMSM, induction          │
│          analysis/, blockchain/, switchgear/, thermal/, sources/ │
├──────────────────────────────────────────────────────────────────┤
│ Layer 1  topology/            Graph + switch combinatorics       │
├──────────────────────────────────────────────────────────────────┤
│ Layer 0  numeric/, sparse/    Types + sparse linear algebra      │
└──────────────────────────────────────────────────────────────────┘
```

Each layer:

- Is one subfolder under `pulsim/`.
- Has its own test binary (`pulsim_core_layerN_tests` under
  `core/tests/layerN/`).
- Includes only from the layers below it (compile-time enforced
  through `#include` discipline).

## Design constraints

The kernel is built around five non-negotiable invariants:

1. **Header-only** — no `.cpp` translation units in the kernel
   itself. The pybind11 binding is the only compiled boundary.
2. **C++23** — concepts, ranges, `mdspan`-like buffers,
   `if consteval`, and `std::expected`-style error returns.
3. **AD-driven Jacobians** — one forward-mode AD scalar drives
   every nonlinear device's stamp; no hand-written Jacobians.
4. **PLECS-style PWL cache** — every reachable switch configuration
   is pre-factored into a (A, B, C, D) state-space tuple; the
   transient loop is one sparse solve per step.
5. **No globals, no singletons** — every entity (Graph, DevicePool,
   PwlStateSpaceCache, SimulationOptions) is value-owned by the
   caller.

## How to use it from C++

```cpp
#include "pulsim/numeric/types.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/sparse/solver.hpp"

namespace p = pulsim;
// … build a Graph + DevicePool, then call pwl::Cache::solve(...)
```

In CMake:

```cmake
target_link_libraries(my_target PRIVATE pulsim::core)
```

The header tree resolves to `core/include/pulsim/<layer>/<file>.hpp`
and the target carries the C++23 requirement plus Eigen / yaml-cpp /
KLU / (optional) HYPRE link lines as an INTERFACE inheritance.
