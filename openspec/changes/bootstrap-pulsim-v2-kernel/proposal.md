## Why

The May 2026 architecture review (`docs/architecture-review-v1.md`)
diagnosed seven structural problems in `pulsim::v1` that are not bugs
but consequences of incremental decisions accumulated over the
codebase's life:

1. **Quadruple-duplicated device stamps.** Every nonlinear device
   (MOSFET, IGBT, IdealDiode, switches) has the same math implemented
   in 4 separate places: device-class behavioral stamp, AD stamp,
   PWL Ideal stamp, AND hand-rolled runtime stamp in
   `runtime_circuit.hpp`. Every change has ~30 % chance of introducing
   sign-convention drift between paths (the May 2026 diode-fails-
   after-reverse-bias bug came from exactly this drift).

2. **`std::variant<22 devices>` AoS storage.** Each element occupies
   the size of the largest member (~512 B). A 16 B Resistor wastes
   32× cache. Bus errors in `analyze_circuit_robustness` come from
   variant alignment ops. Per-device-type SIMD vectorization is
   impossible.

3. **No PWL state-space topology caching — the SINGLE biggest
   architectural gap vs PLECS.** For N switches, there are up to 2^N
   topology combinations. PLECS computes `(A, B, C, D)` and the
   pre-factorized Tustin LU for each combination once at init; per-
   step is a triangular solve. Pulsim runs full Newton-Raphson every
   step (5-30 factorizations per step) on circuits that are
   topologically piecewise-linear. Expected speedup with PWL caching:
   **10-50× on power-electronics workloads** — this is the
   architectural reason PLECS dominates.

4. **Mixed-order integrators.** MNA uses trapezoidal companion, motor
   internal state uses forward Euler, mechanical state uses semi-
   implicit Euler. The composed system is effectively order-1 even
   when half is trapezoidal. PLECS uses Tustin uniformly.

5. **No equation / DSL layer.** Adding a new device requires editing 6
   files (header, runtime stamp, traits, YAML parser, Python bindings,
   tests). Modelica describes the same device in ONE `.mo` file and
   generates everything else.

6. **Single-threaded everywhere.** No subcircuit partitioning, no
   parallel sparse solve, no vectorized stamping. PLECS uses Intel MKL
   parallel direct solver.

7. **Convergence aids piled on as band-aids.** GMIN stepping, source
   stepping, pseudo-transient, homotopy, model regularization, line
   search, random restart, fallback policy — each was added for a
   specific bug. None would be necessary if PWL state-space were the
   default path (linear-per-segment → always converges first try).

Patching v1 in place is no longer cost-effective. Each "minor" change
breaks something downstream because the architecture has no separation
of concerns. This proposal starts a **clean-slate `pulsim::v2`** in
parallel with v1, built bottom-up from a minimal foundation, with
the seven structural improvements baked in from the start. v1 stays
untouched and shipping; v2 is opt-in via the new namespace until
feature-parity is reached.

## What Changes

**Strategy: parallel namespace, layer-by-layer, test-driven.**

`pulsim::v1` continues to live in `core/include/pulsim/v1/` and stays
the production runtime through the entire v2 build-out. The existing
test suite (`pulsim_tests`, `pulsim_simulation_tests`) keeps
running against v1 with zero regression. A new namespace `pulsim::v2`
lives in `core/include/pulsim/v2/`, built layer by layer with
isolated tests per layer. Users can convert circuits between v1 and
v2 at the API boundary; nothing in v1 is touched.

The v2 architecture is **seven layers, strictly separated**:

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
- Is one folder under `core/include/pulsim/v2/`
- Has its own test binary (`pulsim_v2_layerN_tests`)
- Depends only on layers strictly below it
- Has a clean public-API surface that the layer above consumes
- Can be ripped out and replaced without touching anything else

**This proposal (Phase 0 of v2) lands only Layer 0** — numeric types
+ sparse linear algebra. It's the foundation everything else builds
on. Choosing it correctly means future layers compose; choosing it
poorly means v2 inherits v1's mistakes.

Subsequent OpenSpecs will land the higher layers:
- `pulsim-v2-topology-and-switch-enumeration` (Layer 1)
- `pulsim-v2-device-models-ad-driven` (Layer 2)
- `pulsim-v2-generic-stamping-pipeline` (Layer 3)
- `pulsim-v2-pwl-state-space-cache` (Layer 4 — the big win)
- `pulsim-v2-solver-and-events` (Layer 5)
- `pulsim-v2-circuit-builder-api` (Layer 6, partial)
- `pulsim-v2-python-yaml-frontend` (Layer 6, full)

Total estimated effort to feature-parity with v1: **6-9 months**.
Each layer is independently testable and shippable. v1 keeps
production until v2 catches up at Layer 6.

## Impact

- **Affected specs**:
  - NEW capability `kernel-v2-numeric` (Layer 0 surface).
  - NEW capability `kernel-v2-sparse-la` (sparse LA contract).
  - NO modification to existing `kernel-v1-core`, `device-models`,
    `dc-operating-point`, etc. — v1 untouched.
- **Affected code** (this proposal, Layer 0 only — estimated 600-900
  LOC added, 0 LOC modified):
  - NEW directory `core/include/pulsim/v2/` (header-only, like v1).
  - NEW directory `core/include/pulsim/v2/numeric/` — Real / Index /
    Vector / DenseMatrix typedefs and free functions.
  - NEW directory `core/include/pulsim/v2/sparse/` — SparseMatrix
    wrapper + Solver interface + at least SparseLU implementation.
  - NEW directory `core/tests/v2/layer0/` — Layer 0 unit tests.
  - NEW CMake target `pulsim::v2` (interface library, header-only).
  - NEW CMake test target `pulsim_v2_layer0_tests`.
  - NEW `docs/architecture-review-v1.md` documenting WHY v2 exists
    and `docs/pulsim-v2/README.md` explaining the layered design.
- **Migration**: none required for users — v1 is completely untouched.
  v2 is opt-in via `#include "pulsim/v2/..."` headers and the
  `pulsim::v2::` namespace. No symbol clashes possible.
- **Risk**: low for this proposal (Layer 0 is foundation only — no
  features yet, no risk of behaviour change). Risk grows
  proportionally with later layers; the architecture isolates that
  risk to one layer at a time.
- **What this proposal explicitly does NOT do**:
  - No device models (Layer 2). Layer 0 is pure linear algebra.
  - No topology analyzer (Layer 1).
  - No solver, no Newton, no integrator (Layer 5).
  - No PWL state-space cache (Layer 4) — the big architectural win
    lands in its own dedicated proposal after Layers 1-3 are in.
  - No Python bindings, no YAML, no Circuit API.
  - No v1 code is touched. Period.
