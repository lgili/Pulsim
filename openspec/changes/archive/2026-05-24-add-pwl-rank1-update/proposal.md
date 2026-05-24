## Why

The Pulsim PWL state-space cache currently rebuilds its sparse LU
factorization from scratch on every cache miss
(`PwlStateSpaceCache::lookup` → `make_segment` → `analyze + factorize`).
For circuits with many switches (MMC N=12 has 24 switches per arm; NPC
3-level has 30 switches), each cache miss costs O(nnz·log n). The
existing Gray-code enumeration (`enumerate_switch_states`) guarantees
that consecutive states differ by exactly one bit — a textbook setup
for rank-1 LU update via path-based partial refactorization (Chen
et al., *IEEE Trans. Power Electron.*, 2024, for power-system EMT;
Davis & Natarajan, *ACM TOMS* 37(3), 2010, for KLU itself).

Eigen::SparseLU exposes no rank-1 update API. KLU — already listed
as a documented project dependency in `openspec/project.md` — is the
natural backend choice: it was designed by Tim Davis specifically
for circuit-MNA factorization and exposes the elimination-tree
machinery needed for path-based partial refactorization.

This change unlocks the algorithmic novelty path identified in the
2026-05-24 PWL audit for the planned IEEE TPEL methods paper on
Pulsim's PWL cache (see `artigos/02_tpel_methods/`). Without it,
Pulsim is a clean re-implementation of Pejovic-Maksimović 1995 /
Allmeling 1999 — useful but not publishable.

## What Changes

- Add a new `KluSolver` implementation of the
  `pulsim::sparse::DirectSolver` interface, wrapping SuiteSparse KLU.
- Add `Backend` enum and overload `make_default_solver(n, hint)`;
  `Backend::Auto` returns `KluSolver` when n ≥ 100 and KLU is built,
  `SparseLuSolver` otherwise.
- Add `DirectSolver::partial_refactor(new_M, changed_cols)` virtual
  method with default `return false;` — unsupported on
  `SparseLuSolver`, supported on `KluSolver`.
- Add a `PwlStateSpaceCache::solve_rank1(mask, b_extra, x)` fast-path
  that, on single-bit Gray-code flip AND a partial-refactor-capable
  backend, calls `partial_refactor` on the affected columns instead
  of rebuilding the segment. **Falls back transparently** to the
  existing full-rebuild path on multi-bit flips, unsupported
  backends, or numerical singularities encountered during partial
  refactorization.
- Add `PwlStateSpaceCache::metrics()` returning `CacheMetrics
  { rank1_hits, full_refactor_hits, fallbacks }` — read-only
  monotonic counters for benchmark attribution.
- CMake: gate KLU compilation behind
  `find_package(SuiteSparse COMPONENTS KLU)`; if absent,
  `KluSolver` is not built and `Backend::Auto` always returns
  Eigen.
- Extend the TPEL benchmark suite
  (`artigos/02_tpel_methods/benchmarks/`) to capture rank-1 hit rate
  per topology, producing the headline §VI table for the paper.

**No BREAKING changes.** Every existing caller and every reference
project in `projects/` produces bit-identical output (within 1e-12);
the rank-1 path is purely additive performance.

## Impact

- **Affected specs (NEW capability):** `pwl-rank1-update`
- **Affected code:**
  - `core/include/pulsim/sparse/solver.hpp` — extend factory + interface
  - `core/include/pulsim/sparse/klu_solver.hpp` — **NEW**
  - `core/src/sparse/klu_solver.cpp` — **NEW**
  - `core/include/pulsim/pwl/cache.hpp` — add `solve_rank1` + metrics
  - `core/CMakeLists.txt` — add KLU detection
  - `core/tests/layer0/test_klu_solver.cpp` — **NEW**
  - `core/tests/layer4/test_pwl_cache_rank1.cpp` — **NEW**
  - `artigos/02_tpel_methods/benchmarks/*` — extend orchestrator
- **Affected build:** new optional dependency on `SuiteSparse-KLU`
  (LGPL+, permissively compatible with Pulsim's MIT per
  `project.md`).
- **Not affected:** Python bindings (transparent backend swap),
  public `CircuitBuilder` API, all 8 reference projects in `projects/`.
