## Why

The shipped `pwl-rank1-update` capability (v1.2.0, merged 2026-05-24)
ties Pulsim's rank-1 cache fast-path to SuiteSparse KLU as the
required backend for `partial_refactor`. While V0's `klu_refactor`
delegation gave a measured 3.15× speedup at n=14, two structural
limitations remain:

1. **The published path-based partial refactor algorithm
   (Chan/Brandwajn/Tinney 1986, Dinkelbach et al. 2021) is not
   exposed by upstream Davis KLU.** Only a third-party fork
   (dpsim-simulator/SuiteSparse) implements it as a C patch with
   added `klu_numeric` fields and new public functions.

2. **The TPEL methods paper's algorithmic novelty must be ours.**
   Vendoring a third-party C patch and wrapping it in C++ is a thin
   contribution. Re-implementing the underlying sparse direct LU stack
   in-house — including the path-based partial refactor — gives Pulsim
   a fully self-contained algorithmic kernel and a publishable
   contribution unblocked from any external solver's roadmap.

This change **replaces KLU entirely** with an in-house C++23 sparse
LU implementation built on Eigen sparse-matrix containers (Pulsim's
existing `Eigen::SparseMatrix<Real, ColMajor, int32>` type). The new
solver implements every layer from symbolic analysis through
path-based partial refactorization, citing the published references
(Davis 2006; Gilbert & Peierls 1988; Dinkelbach et al. 2021) but
contributing no third-party C dependencies.

**Trade-off explicitly accepted by the project owner (2026-05-24):**
this is a 4-6+ week effort (likely more once numerical-stability
debugging is factored in). The TPEL paper submission target shifts
from Oct 2026 to Q1 2027.

## What Changes

### Drop KLU entirely

- **Remove** `KluSolver` (`core/include/pulsim/sparse/klu_solver.hpp`,
  `core/tests/layer0/test_klu_solver.cpp`).
- **Remove** the `Backend::KLU` enum value from `sparse/solver.hpp`
  (replaced by `Backend::Pulsim` — see below).
- **Remove** the `find_package(KLU CONFIG)` block from root
  `CMakeLists.txt` and the `PULSIM_HAVE_KLU` / `PULSIM_ENABLE_KLU`
  options.
- **Remove** `libsuitesparse-dev` / `suite-sparse` from CI installs.
- **Remove** the SuiteSparse KLU row from the README "Build
  prerequisites" section.

### Add `pulsim::sparse::PulsimSparseLuSolver` — in-house C++ sparse LU

A new `DirectSolver` implementation that owns the full
analyze + factor + solve + partial_refactor pipeline, operating on
Pulsim's existing `Eigen::SparseMatrix<Real, ColMajor, int32>` type
as both input and internal storage. Eigen is used **only as a
container** — none of Eigen's own LU machinery is invoked.

Components:

| Component | Algorithm | Reference | Lines (est.) |
|---|---|---|--:|
| Column ordering (fill-reducing) | Reverse Cuthill-McKee (MVP V0); AMD or COLAMD as follow-up | George 1971; Davis 2006 §7 | 80-150 |
| Symbolic analysis | Compute fill-in pattern + elimination tree | Davis 2006 §4.10; Liu 1986 | 100-150 |
| Numeric factorization | Gilbert-Peierls left-looking with partial pivoting | Gilbert & Peierls, *SIAM J. Sci. Stat. Comput.* 9, 1988 | 200-300 |
| Triangular solve | Forward/back substitution on sparse L+U | Davis 2006 §3 | 50-80 |
| Path-based partial refactor | Path detection via etree + re-elimination on path columns | Chan/Brandwajn/Tinney *IEEE TPS* 1986; Dinkelbach *Energies* 2021 §3 | 150-200 |
| Pivot-fault detection + recovery | Pivot magnitude vs threshold check + automatic full-refactor fallback | Dinkelbach §3.2 | 30-50 |

**Total estimated new C++**: ~600-900 lines.

### Public API

`PulsimSparseLuSolver` implements the existing `DirectSolver`
interface. No changes to `partial_refactor(M, changed_cols)`
signature. `Backend::Pulsim` replaces `Backend::KLU` in the factory
hint enum; `Backend::Auto` picks Pulsim's LU for any n (no more
KLU/Eigen crossover threshold — we are the only non-Eigen path).

### Tests + bench

- **Replace** `test_klu_solver.cpp` with `test_pulsim_lu_solver.cpp`
  — same scenarios (analyze/factor/solve parity vs SparseLuSolver,
  partial_refactor parity, lifecycle errors) plus 2 new (numerical
  conditioning sweep, ill-conditioned circuit fallback).
- **Update** `test_pwl_cache_rank1.cpp` integration tests to use the
  Pulsim backend (drops the "metrics fallbacks include KLU-disabled
  path" branch in test 5.2 — the only backend now is Pulsim).
- **Re-run** rank-1 microbench. Update RANK1_RESULTS.md with three
  columns: baseline `solve` (per-mask cache), `solve_rank1` with
  Eigen::SparseLU (V0-equivalent path), `solve_rank1` with
  PulsimSparseLuSolver (V8.1).

### Versioning

Bump to **v1.3.0**. This is a backwards-compatible release at the
public Python API level — `pp.simulate(...)` keeps working — but it
is a substantial internal architecture change documented in detail
in the CHANGELOG.

## Impact

- **Affected specs:**
  - `pwl-rank1-update` — MODIFIED requirements to drop KLU-specific
    text (REMOVE "KLU Backend for Sparse Direct Solvers"; MODIFY
    "Single-Bit Gray-Code Partial Refactorization" to reference
    Pulsim's own LU; MODIFY "Transparent Fallback…" similarly)
  - `pulsim-sparse-lu` — NEW capability for the in-house solver itself
- **Affected code:**
  - `core/include/pulsim/sparse/klu_solver.hpp` — **DELETED**
  - `core/include/pulsim/sparse/pulsim_lu_solver.hpp` — **NEW**
    (~600-900 lines)
  - `core/include/pulsim/sparse/solver.hpp` — drop `Backend::KLU`,
    add `Backend::Pulsim`; drop the KLU-gated factory branch; drop
    `PULSIM_KLU_AUTO_THRESHOLD`
  - `core/CMakeLists.txt` — drop `SuiteSparse::KLU` linkage from
    `pulsim_core`; drop the KLU-related sources from test targets
  - `CMakeLists.txt` (root) — drop the `find_package(KLU)` block
  - `.github/workflows/ci.yml` — drop `libsuitesparse-dev` /
    `suite-sparse` from all install commands
  - `README.md` — drop the SuiteSparse row from Build prerequisites
  - `core/tests/layer0/test_klu_solver.cpp` — **DELETED**
  - `core/tests/layer0/test_pulsim_lu_solver.cpp` — **NEW**
  - `core/tests/layer4/test_pwl_cache_rank1.cpp` — small edit (test
    5.2 simplification)
  - `core/tests/benchmarks/test_bench_pwl_rank1.cpp` — extend output
    to capture 3 backends instead of 2
  - `artigos/02_tpel_methods/benchmarks/RANK1_RESULTS.md` — full
    rewrite for 3-backend comparison
  - `CHANGELOG.md` — `[1.3.0]` entry
  - `pyproject.toml`, `python/pulsim/__init__.py`, `CITATION.cff` —
    version bump
- **Affected build:** drops the optional SuiteSparse dependency
  entirely. Pulsim builds with **Eigen + a C++23 compiler** as the
  only required native deps. CI matrix simplifies.
- **Not affected:** public Python API, all 8 reference projects in
  `projects/`, Layer 5 `run_transient`. The `PwlStateSpaceCache`
  consumers see no API change.

**BREAKING change at the C++ kernel-builder level only:** any
out-of-tree code that explicitly constructed `KluSolver` or passed
`Backend::KLU` to `make_default_solver` will need to switch to
`PulsimSparseLuSolver` / `Backend::Pulsim`. The standard
`make_default_solver()` / `make_default_solver(n, Backend::Auto)`
entry points continue to work transparently.

## Scope of work + timeline

Honest estimate: **6-10 weeks** of focused work, broken into
~6 milestones (Sections 1-6 in `tasks.md`). Likely tracks:

1. Symbolic analysis + RCM ordering + etree (~1 week)
2. Numeric factorization with partial pivoting (~2 weeks — the
   numerically delicate piece)
3. Triangular solve + parity tests vs Eigen::SparseLU (~1 week)
4. Path-based partial refactor + pivot-fault recovery (~1-2 weeks)
5. Integration tests, bench, microbench (~1 week)
6. Spec close-out + PR + archive (~few days)

The TPEL paper drafting timeline shifts to Q1 2027 to absorb this
schedule. Updated in `artigos/02_tpel_methods/README.md` once
implementation actually starts.
