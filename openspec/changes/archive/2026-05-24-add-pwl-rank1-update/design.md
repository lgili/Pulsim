## Context

Pulsim v2's PWL state-space cache (`pulsim::pwl::PwlStateSpaceCache` in
`core/include/pulsim/pwl/cache.hpp`) pre-builds one sparse LU
factorization per switch-state combination. The per-step hot path is
hash-lookup + triangular-solve (O(nnz)) — already excellent.

The bottleneck is in the COLD path: when the simulator encounters a
switch state for the first time (lazy mode) or revisits many states
across a long simulation. Each cache miss costs an `analyze +
factorize` cycle — `O(nnz·log n)` for circuit matrices per Davis,
*Direct Methods for Sparse Linear Systems*, SIAM 2006.

For an MMC with N=12 sub-modules per arm (48 switches total), the
cache has 2⁴⁸ possible states. Lazy enumeration is mandatory; partial
refactorization on Gray-code-adjacent flips reduces the per-flip cost
from `O(nnz·log n)` to `O(path_length)`, where `path_length` is the
depth of the affected column in the elimination tree — typically
O(√n) for circuit matrices.

Expected wall-time per flip at n=200 (per Q6 of the 2026-05-24
library audit): full `factorize()` ≈ 0.3-1 ms; partial refactor ≈
30-100 µs → **~10× speedup**, material at the MMC scale that drives
the planned IEEE TPEL paper.

## Goals / Non-Goals

**Goals**

- Add KLU as a fast-path sparse direct solver backend, gated behind
  CMake detection of SuiteSparse.
- Add a single-bit Gray-code rank-1 partial-refactor path to
  `PwlStateSpaceCache` that falls back transparently to the current
  full-rebuild path on any unsupported condition.
- Provide telemetry sufficient to attribute speedup in the IEEE TPEL
  benchmark table (`artigos/02_tpel_methods/benchmarks/`).

**Non-goals**

- Replacing `Eigen::SparseLU` as the default solver. Eigen remains
  the only required dependency; KLU is optional.
- Sherman-Morrison-Woodbury implementation. We adopt partial
  refactorization instead — exact, no floating-point error
  accumulation, matches the 2024 IEEE TPEL paper's approach.
- Multi-bit rank-k updates. Out of scope — the cost approaches full
  refactorization, and Gray-code enumeration already guarantees
  single-bit transitions for the common case.
- Reconciling the legacy `linear-solver` spec with the v2 code. That
  spec describes an `AdvancedLinearSolver` that doesn't exist in v2;
  separate cleanup deferred to a dedicated change.

## Decisions

### D1. KLU as the rank-1 backend (vs LUSOL or PARDISO)

KLU is purpose-built for circuit MNA matrices (Davis & Natarajan,
*ACM TOMS* 37(3), 2010, Algorithm 907), already listed as a
documented project dependency in `openspec/project.md`, LGPL+ license
compatible with Pulsim's MIT.

LUSOL is Fortran 77/90 optimised for LP simplex (different problem
structure; 2-3 weeks of integration vs KLU's 1-2). PARDISO is
closed-source — clashes with the open-source narrative of the
planned TPEL paper and adds a non-trivial license barrier for
downstream users.

### D2. Partial refactorization (vs Sherman-Morrison-Woodbury)

SMW updates the inverse via outer-product correction — mathematically
elegant but (a) accumulates floating-point error across many
sequential updates, (b) requires special handling of the rank-1 spike,
(c) doesn't naturally fit KLU's elimination-tree data structures.

Partial refactorization — re-eliminate only the columns on the path
from the changed column to the root of the elimination tree — is
**exact** (no error accumulation), matches the KLU primitive
operations, and follows Chen et al. 2024 ("Partial Refactorization
Techniques for EMT Simulations"). That paper established the
approach for the adjacent power-system EMT simulation domain;
adapting it to converter-level PWL caches is a clean port.

### D3. Auto-fallback over user-controlled mode flag

Failures (multi-bit flip, KLU unavailable, partial refactor
singularity) MUST NOT surface as errors to the simulator user. The
cache transparently falls back to the current full-rebuild path.

Rationale: the rank-1 path is a *performance optimisation*, not a
behavioural change. Telemetry counters let the benchmark suite
observe attribution; user code is unaffected.

### D4. `Backend::Auto` crossover at n=100

For n<100, `Eigen::SparseLU::factorize()` is fast enough that KLU's
analyze-once-then-partial-refactor amortisation doesn't win (Q6 of
library audit: ~30-80 µs full refactor for n=50 vs ~10-25 µs rank-1
update — marginal speedup ~3×). The crossover heuristic is encoded
once in `make_default_solver(n, hint)` and tuneable via a build-time
constant `PULSIM_KLU_AUTO_THRESHOLD` (default 100).

### D5. Telemetry as monotonic atomic counters

`CacheMetrics` uses `std::atomic<uint64_t>` so the counters can be
read from a background telemetry thread without locking. They are
strictly monotonic and read-only from outside the cache. The fields
are typed (not a free-form string map) so future schema evolution is
explicit and the benchmark CSVs stay machine-parseable.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| KLU `klu_refactor` does FULL refactor, not partial. We need a custom path-based variant. | The path-based variant is a ~200-line patch atop `klu_l_factor.c` per Chen et al. 2024 §III. Encapsulated in `KluSolver::partial_refactor`; no upstream KLU patches needed (we drive KLU at the symbolic-factor level). |
| Build complexity from optional KLU dep. | Gate cleanly with `PULSIM_HAVE_KLU`; CI tests both with and without KLU installed. Failed `find_package` is non-fatal. |
| Numerical drift between full-refactor and partial-refactor paths after many sequential updates. | Integration test 5.1 enforces bit-identical output within 1e-12 over a 100-step simulation; any drift signals a bug, not a feature. |
| Pulsim users who patched their own `DirectSolver` impl might break. | `DirectSolver` gains TWO new virtual methods (`supports_partial_refactor`, `partial_refactor`) with default implementations returning `false`. Subclasses don't need to override — fallback engages automatically. |
| `Eigen::SparseLU` and `KluSolver` may produce numerically different solutions due to different pivoting strategies. | Acceptance criterion is 1e-12, not bit-equality. Documented in unit test 2.7.1 docstring. |

## Migration Plan

Pure addition. No data format change, no API breakage, no migration
needed by existing callers. Existing simulations produce identical
output (within 1e-12) and gain speedup automatically when n ≥ 100
and KLU is available.

**Rollback:** revert the patch series. The DirectSolver virtual
default-`false` methods can stay (harmless).

## Open Questions

- Q: Should `Backend::Auto`'s n=100 crossover be tuneable at runtime
  (env var, config) or only at build time?
  **Tentative answer:** build-time constant `PULSIM_KLU_AUTO_THRESHOLD`;
  revisit if benchmark data points us elsewhere.
- Q: When KLU is unavailable at build time, should `Backend::KLU`
  requested explicitly throw or silently fall back to Eigen?
  **Tentative answer:** throw — loud failure honours the explicit
  user request rather than silently masking it.
- Q: Should the rank-1 fast-path be exposed at the Python binding
  level as a distinct method, or transparently engaged behind
  `solve()`?
  **Tentative answer:** transparent — Python users don't need to
  care about the optimisation; benchmarks read `metrics()`.
