## Context

`replace-klu-with-pulsim-sparse-lu` (merging now as v1.3.0) replaced
SuiteSparse KLU with the in-house `PulsimSparseLuSolver` for the PWL
state-space cache / real-scalar MNA path. After that lands, **one
remaining call site** still uses a third-party LU factorisation in
production: `core/include/pulsim/analysis/mna_sweep.hpp:230` —
`Eigen::SparseLU<std::complex<Real>>` per frequency for the AC small-
signal sweep.

This proposal closes the gap by templating `PulsimSparseLuSolver` on
`Scalar` and switching the AC sweep call site to the in-house
implementation. After this lands, the only references to
`Eigen::SparseLU` in the kernel are:

1. `SparseLuSolver` class (`Backend::Eigen`) — kept as the benchmark
   baseline for the TPEL paper's 3-backend / 2-backend comparison
   tables.
2. `eigen_reference_fill()` test helper — used to assert
   `pulsim_fill ≤ 3 × eigen_fill` (sanity guard).
3. New `test_bench_ac_sweep.cpp` — benchmark fixture exercising the
   Eigen baseline as the comparison.

All three are bounded to tests / benchmarks. Production code paths
have zero third-party LU factorisation.

## Goals / Non-Goals

**Goals**
- Zero third-party LU in production code paths (PWL transient + AC
  sweep + DC OP, all served by `PulsimSparseLuSolver`).
- Bit-identical AC sweep output (within 1e-10 complex tolerance) vs
  the v1.3.0 Eigen-backed implementation on the 10 reference
  converter projects.
- Per-frequency latency within ~1.5× of Eigen (acceptable — the
  paper claim is "we don't depend on third parties", not "we beat
  them at every workload").
- Source-compatible: all existing real-scalar call sites compile
  unchanged thanks to default-template-arg `Scalar = Real`.

**Non-Goals**
- Outperform Eigen on the complex sparse LU. The TPEL paper's
  algorithmic novelty is the *path-based partial refactor for the
  real-scalar PWL switching case*; AC sweep is a derived workload
  where the in-house solver matches Eigen, not beats it.
- BTF block-triangular decomposition (deferred for both real and
  complex)
- `Scalar = float` or `std::complex<float>` (no call site)
- Per-frequency factorisation reuse / sliding-solver in AC sweep
  (separate follow-up proposal)

## Decisions

### Decision 1: Template on `Scalar` rather than parallel `ComplexPulsimSparseLuSolver`

**Decision**: Template the existing class on `Scalar`, with a
default `Scalar = Real` for backward compat. Add concrete
`using PulsimComplexSparseLuSolver = PulsimSparseLuSolver<std::complex<Real>>;`
typedef for the AC sweep call site.

**Alternatives considered**:

- *Separate `PulsimComplexSparseLuSolver` class*: would duplicate
  ~900 lines of header-only code (Gilbert-Peierls, RCM, etree,
  path-based partial refactor) with `Real → Complex` substitutions.
  Rejected: maintenance burden + algorithmic-divergence risk.
- *Type-erased Scalar (runtime dispatch)*: would push `std::variant`
  or `std::any` into the hot loop. Rejected: 5-10× per-call
  overhead from the dispatch, defeating the purpose of an in-house
  implementation.
- *C-style code generation*: generate the complex variant via macro
  preprocessing or codegen tool. Rejected: violates the C++23
  "boring, proven" principle from the original proposal's design.

The template approach is the canonical C++23 path. Eigen itself
templates `SparseLU` on Scalar; we're following the same proven
pattern.

### Decision 2: Pivot threshold check `std::abs(x[i])` works as-is

**Decision**: The existing pivot-threshold check
`std::abs(x[i_max]) >= PIVOT_THRESH * column_infinity_norm` operates
unchanged on `std::complex<Real>` because `std::abs` is overloaded
for complex types to return the natural magnitude
`std::sqrt(re² + im²)`.

**Rationale**: This is the standard treatment for complex partial
pivoting per Demmel 1997 §3.4 / LAPACK ZGETRF. The magnitude metric
generalises correctly; the algorithm itself is identical to the
real-scalar case.

### Decision 3: Keep `SparseLuSolver` (Eigen fallback) untouched as `Backend::Eigen`

**Decision**: After this proposal, `Backend::Eigen` continues to
return `SparseLuSolver` (Eigen-backed). The class is also templated
on Scalar so it can serve both real and complex AC sweep benchmarks.

**Rationale**: The TPEL paper §VI tables require a baseline. The
3-backend microbench for the real-scalar PWL rank-1 cache, and the
2-backend microbench for the complex AC sweep, both need
`Backend::Eigen` as the reference. Removing it would force readers
to dig into git history to compare; keeping it bounded to a single
file (`solver.hpp`) preserves the comparison fixture cheaply.

### Decision 4: Defer per-frequency factorisation reuse to a follow-up

**Decision**: This proposal does the minimum needed to eliminate
the production Eigen LU dependency. It does NOT add the AC sweep
analogue of the PWL sliding-solver pattern (where the symbolic
factorisation is reused across frequencies because only `j·ω` shifts
the values).

**Rationale**: That optimisation would be ~3-5× wall-clock faster
per sweep (one analyze + N numeric factorises) but is independent of
the "drop Eigen LU" work. Split per the OpenSpec principle of one
proposal = one focused capability.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Per-frequency latency regresses vs Eigen on real workloads | Run benchmark capture on buck + NPC + MMC; if any converter shows > 2× regression, defer the proposal and fold in per-frequency reuse first. |
| Complex partial pivoting hits more numerical edge cases than real (e.g. cancellation in `re·im` cross-terms) | Test suite §5.1.2 includes asymmetric complex MNA. If a real converter sweep triggers a pivot failure, the existing pivot-fault → invalidate + return-false path lets the AC sweep caller fall back to a full re-factorise. No silent breakage. |
| Template instantiation increases compile time | Header-only design + two explicit instantiations (Real, complex<Real>) keeps the cost bounded. Measure before/after; if compile time grows > 10 % on a clean build, move the implementation to a `.tpp` file and explicit-instantiate only the two scalar types. |
| Symbol bloat in the kernel `.so` | The two instantiations contribute ~2× the per-class size. Acceptable — current kernel is ~3 MB stripped; +500 KB is within noise. |

## Migration Plan

This is **additive** at the public API level (the new template
parameter has a default value matching the current behaviour). No
downstream caller, Python user, or notebook needs to change. The
only observable difference:

1. AC sweep telemetry now reports
   `pulsim_sparse_lu` as the solver under the hood, not
   `eigen_sparselu`. Test suites that pin the telemetry string
   need updating (TBD — search `eigen_sparselu` references in
   tests as part of task 4).
2. Per-frequency timing on AC sweeps shifts by some amount
   (expected ±50 % vs v1.3.0 on the captured benchmark).
   Notebook-level smoke tests should tolerate this since they
   compare Bode plots within ±0.1 dB / 1°, not within ±0 ns.

Rollback: revert the single-line switch in `mna_sweep.hpp` from
`PulsimComplexSparseLuSolver` back to `Eigen::SparseLU<Complex>`.
The template addition stays — it's backward-compat.

## Open Questions

1. Should we make `Backend::Eigen` available from the Python API
   for the AC sweep path (parallel to the existing
   `cache.set_rank1_backend(...)`)? Currently the AC sweep path
   doesn't expose a backend selector. **Tentative answer**: no —
   it would be one more knob for benchmark replication, but
   notebook users don't need it. Add only if a reproducer notebook
   for the paper requires it.

2. Should the `eigen_reference_fill()` helper get a complex
   companion (`eigen_reference_fill_complex()`)? **Tentative
   answer**: yes — same template-on-Scalar treatment, same
   fill-bloat sanity check. Add as part of §5 test coverage.

3. Does the path-based partial refactor have any value in the AC
   sweep workload? Single-frequency sweeps don't benefit (matrix
   changes across frequencies are not single-column). Multi-input /
   multi-output sweeps that perturb one source at a time MIGHT
   benefit. **Tentative answer**: deferred to a follow-up — out of
   scope here.
