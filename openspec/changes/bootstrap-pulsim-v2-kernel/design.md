# Design — `bootstrap-pulsim-v2-kernel` (Layer 0)

## Why parallel namespace, not refactor-in-place

The May 2026 architecture review identified seven structural
problems with `pulsim::v1` that compound on each other:

1. Quadruple-duplicated device stamps (4 places, drift between paths)
2. `std::variant<22>` AoS storage (cache-hostile, alignment bugs)
3. No PWL state-space caching (10-50× slower than PLECS)
4. Mixed-order integrators (effective order-1)
5. No DSL / equation layer (6-file edit per new device)
6. Single-threaded everywhere
7. Convergence band-aids piled on as default-on options

These are not bugs to fix one-by-one. Each one is a CONSEQUENCE of an
early architectural choice that the next 12 months of development
built on top of. Removing any one in place breaks dozens of downstream
assumptions. The cumulative cost of incremental refactor is HIGHER
than building the replacement cleanly in parallel and switching the
runtime at the API boundary when v2 reaches feature-parity.

The parallel-namespace strategy:

- v1 continues to live in `core/include/pulsim/v1/`. Production-stable.
  Existing test suites (`pulsim_tests`, `pulsim_simulation_tests`)
  keep running with zero touch. Existing users see no change.
- v2 lives in `core/include/pulsim/v2/`. Built bottom-up, layer by
  layer. Each layer has its own test binary and is independently
  verifiable.
- Final cutover (months from now): the Python frontend grows a
  `use_v2=True` flag. Users opt in. Once v2 catches up across the
  feature matrix, v1 enters maintenance mode, then archive.

This is the same strategy GCC used for the LLVM-style modernisation,
that Vue used for v3, that Eigen used for 3.4 → 4.0. It is well-
tested and low-risk for this kind of compounding-debt cleanup.

## Layer 0 scope — why JUST numeric + sparse LA

The seven-layer architecture is:

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

Layer 0 is the foundation everything else uses. Choosing it once,
correctly, means every layer above composes naturally. Choosing it
poorly (e.g. wrong index width, wrong matrix row-major-ity, no
factor caching) means every layer above pays the cost forever.

This proposal lands ONLY Layer 0. No features, no device models, no
solver. The follow-up OpenSpec list:

| Layer | OpenSpec ID                                  | Rough effort |
|-------|----------------------------------------------|--------------|
| 1     | `pulsim-v2-topology-and-switch-enumeration`  | ~1 week      |
| 2     | `pulsim-v2-device-models-ad-driven`          | ~2 weeks     |
| 3     | `pulsim-v2-generic-stamping-pipeline`        | ~1 week      |
| 4     | `pulsim-v2-pwl-state-space-cache`            | ~3-4 weeks   |
| 5     | `pulsim-v2-solver-and-events`                | ~2 weeks     |
| 6a    | `pulsim-v2-circuit-builder-api`              | ~1 week      |
| 6b    | `pulsim-v2-python-yaml-frontend`             | ~2 weeks     |

Total: ~12-13 weeks to feature parity on a single-developer pace.
Each layer is independently mergeable; v1 stays production
throughout.

## Layer 0 design decisions

### `Real = double`, parameterizable but not parameter-default

Future-proofing for single-precision builds (FPGA / embedded HIL
targets) without making the default API depend on a template
parameter. The `Real` alias resolves to `double` by default; a
build with `-DPULSIM_V2_REAL_TYPE=float` flips it. Layer 2 + Layer 3
templates will accept any floating-point type (so AD scalars and
float can be used in the same template path), but Layer 4 + Layer 5
work in `Real` (double) for numeric stability.

### `Index = std::int32_t`

v1 uses `int` (typically 32 b but implementation-defined). v2 fixes
it at 4 bytes signed:

- 4 B index arrays fit twice as much per cache line as 8 B
- 2^31 ≈ 2 G nodes is plenty for any conceivable power-electronics
  circuit
- Sparse matrix libraries (KLU, UMFPACK, MKL Sparse) expect int32 by
  default — no extra wrapper needed
- Signed allows `-1` as sentinel for ground / "not such device"
  without burning bit-31 (we get -1 ≡ all-bits-set ≡ trivially
  detectable in masks)

### Sparse matrix layout — ColMajor, int32 index

`Eigen::SparseMatrix<Real, ColMajor, Index>` matches every direct
sparse solver's expected input format. RowMajor would force a
transpose-and-copy at every solver call.

ColMajor also matches the natural MNA stamping pattern: for each
device, you stamp a few entries in a few columns (the device's
nodes); ColMajor means those entries live next to each other in
memory.

### Solver lifecycle — `analyze → factorize → solve` separation

PLECS's secret sauce is: for a given switch combination, `analyze`
runs ONCE, `factorize` runs ONCE (the matrix is constant within the
segment), and `solve` runs every step. The current Pulsim v1
`linear_factor_cache` partially captures this but doesn't separate
`analyze` from `factorize` — every cache miss redoes both.

v2's `DirectSolver` API enforces the separation at the type level.
Layer 4's PWL state-space cache will store the result of `analyze`
ONCE per topology combination, the result of `factorize` ONCE per
(combination, dt), and only `solve` runs in the hot loop.

This single separation gives a 5-10× speedup on multi-segment PWM
workloads vs v1's current cache.

### Factory pattern for solver backends

Layer 0 ships ONE concrete solver: `SparseLuSolver` (Eigen's
`Eigen::SparseLU`). The factory returns a `unique_ptr<DirectSolver>`
of this type. Future layers can register alternative backends
(KLU, MKL Pardiso, UMFPACK, HYPRE) through the same factory —
adding them does NOT touch any consumer that depends only on the
`DirectSolver` interface.

This avoids v1's pattern where solver choice leaks into
`SimulationOptions::linear_solver` with custom enums and per-backend
fallback policies. v2's solver choice is a runtime polymorphism;
the dispatcher chooses based on matrix properties + user hint.

## What this layer does NOT do

- No device models. A `Real` and a `SparseMatrix` are not a device.
- No Newton-Raphson. Newton lives in Layer 5.
- No integrator. ODE integration lives in Layer 5.
- No graph / topology. Sparse matrix knows nothing about KCL / KVL.
- No event detection.
- No state space.

Future layers compose by ADDING capability without modifying Layer 0.
If a future layer needs something not in Layer 0 (e.g. a banded
linear solver for tridiagonal systems), Layer 0 grows by ADDING a
new class — no rewrite of consumers.

## Risks

1. **Eigen lock-in.** v2's Layer 0 is a thin wrapper over Eigen.
   That's a deliberate trade — Eigen is mature, well-optimized,
   and already a Pulsim dependency. Risk: if Eigen ever needs to
   be replaced (e.g. for GPU offload via custom kernels), Layer 0
   becomes a refactor surface. Mitigation: keep the public API
   surface MINIMAL (just typedefs + a few utilities + solver
   interface). The smaller the surface, the cheaper the future
   swap.

2. **Solver interface might miss something.** PLECS-style state-space
   caching benefits from knowing the matrix's CHANGES, not just
   its current values. Layer 0's `analyze / factorize / solve` API
   captures the common case but might force inefficiency for the
   "matrix entries changed slightly but pattern is identical" path.
   Mitigation: future layers can add a `refactorize_with_same_pattern`
   helper to `DirectSolver` interface without breaking existing
   consumers.

3. **`Real = double` may be wrong for some users.** Embedded HIL
   wants float. Mitigation: build-time switch
   `-DPULSIM_V2_REAL_TYPE=float`. Tests in Layer 0 add a CMake
   matrix entry for float build (deferred to a later phase — Layer 0
   tests pass at double; float build verification is a follow-up).

## Validation strategy

- Layer 0 tests live in `core/tests/v2/layer0/` (parallel to v1
  tests). Own binary `pulsim_v2_layer0_tests`.
- Coverage target: every public Layer 0 function is exercised by
  at least one test. Initial target: 15 assertions / 6 test cases.
- v1 tests MUST remain green. Layer 0 cannot regress v1; it
  doesn't even share a header.
- `openspec validate --strict` MUST pass on this change before
  archive.
