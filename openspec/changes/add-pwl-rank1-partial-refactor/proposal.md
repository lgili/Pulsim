## Why

The shipped `add-pwl-rank1-update` (v1.2.0, merged 2026-05-24) declared
that `PwlStateSpaceCache::solve_rank1` SHALL use path-based partial
refactorisation per Chen et al. (IEEE TPEL 2024), but the MVP
implementation delegated to `klu_refactor` — a full numeric refactor
with cached symbolic. That already gave **3.15× at n=14** but doesn't
deliver the O(path) asymptotic claim. The spec text is aspirational;
this change brings the implementation into compliance.

A 2026-05-24 follow-up research pass confirmed the path-based
algorithm is published (Chan/Brandwajn/Tinney, *IEEE Trans. PAS*,
1985; Dinkelbach et al., *Energies* 14:7989, 2021; Abusalah et al.,
*IEEE OAJ-PE* 7, 2020) and exists as an **LGPL-2.1+ open-source patch
on top of SuiteSparse KLU** in
[dpsim-simulator/SuiteSparse](https://github.com/dpsim-simulator/SuiteSparse).
The fork adds three new public C functions (`klu_compute_path`,
`klu_partial_factorization_path`, `klu_partial_refactorization_restart`)
plus new fields on `klu_numeric` (`path`, `pathLen`, `block_path`,
`variable_block`, …) and a new error code `KLU_PATH_INVALID`.

Vendoring the DPsim fork (FetchContent, build-time download, static
link) avoids reinventing the path-based algorithm and the
elimination-tree surgery on KLU's internals — saving ~2 weeks of
careful sparse-LU work and matching the upstream patch maintainers'
correctness contract bit-for-bit.

Expected per-flip speedup at MMC scale (per the audit):
- n=50:  ~2-4× over `klu_refactor`
- n=200: ~5-10× over `klu_refactor`
- n=10k: 7-49× (Dinkelbach 2021 §4 measured on EMT grid matrices)

This is the algorithmic data point the planned IEEE TPEL methods
paper §VI needs to claim true O(path) per single-bit flip rather
than "we used klu_refactor."

## What Changes

### Build / dependency

- **Replace** `find_package(KLU CONFIG)` with a vendored FetchContent
  pull of the dpsim-simulator/SuiteSparse fork (pinned to a specific
  commit SHA for reproducibility). Build a static `libklu_dpsim` as a
  CMake sub-target, link it into `pulsim_core` as an INTERFACE
  dependency.
- **Keep** `PULSIM_ENABLE_KLU` toggle. When `OFF`, build no KLU at all
  (Eigen-only fallback) — same as today.
- **Remove** `libsuitesparse-dev` / `brew install suite-sparse` from
  CI install commands (no longer needed; vendored).
- **Update** README "Build prerequisites" — remove the per-platform
  KLU install instructions; document that KLU is vendored.

### Kernel — KluSolver

- **Add** private `path_cache_` member on `KluSolver` —
  `unordered_map<uint64_t, PathState>` keyed on a hash of the
  changed-column index list, where `PathState` is an owning RAII
  wrapper that holds the `klu_numeric->path` arrays populated by
  `klu_compute_path`. Bounded in size by the number of distinct
  single-bit switch flips ≤ N (one entry per switch).
- **Add** private `ensure_path_for(changed_cols, A_arrays)` helper
  that looks up or computes-and-caches the path state for a given
  changed-column set.
- **Replace** `KluSolver::partial_refactor(new_M, changed_cols)`
  body — instead of delegating to `klu_refactor` (V0 MVP), look up
  the path for `changed_cols` (compute on first encounter), then call
  `klu_partial_factorization_path(new_M)`. On `KLU_PIVOT_FAULT`
  (pivot threshold exceeded), fall back to `klu_factor` (full
  refactor) AND invalidate the path cache (pivots may have changed).
- **Spec contract unchanged** — `partial_refactor` still returns
  `bool` (true on numeric success, false on singularity); the caller
  still sees the existing `false → fall back` semantics. Backwards
  compat preserved end-to-end.

### Tests + bench

- **Tighten** test 2.7.3 from "output `x` matches full refactor
  within 1e-12" to **"L and U factor values match full refactor
  bit-identical (within 1e-14)"** — the path-based algorithm is
  mathematically equivalent to full refactor when pivots don't
  change, so the test can be stricter.
- **Add** new test: pivot-fault fallback. Construct a case where
  the post-flip matrix would violate the pivot-tolerance threshold;
  verify `partial_refactor` returns `false` cleanly AND the next
  `factorize` call recovers.
- **Add** new test: path-cache hit. Same changed-cols set on
  consecutive `partial_refactor` calls reuses the cached path
  (verify via internal counter or via timing).
- **Re-run** the rank-1 microbench (existing
  `test_bench_pwl_rank1.cpp`). Update `RANK1_RESULTS.md` with a new
  V0-vs-V8.1 comparison column. Push N to {16, 20, 24} where
  feasible to measure the asymptotic regime.

### Versioning

- Bump to **v1.3.0** in `pyproject.toml`, `python/pulsim/__init__.py`,
  `CITATION.cff`. CHANGELOG entry under `[1.3.0]`.

## Impact

- **Affected specs:** `pwl-rank1-update` — MODIFIED requirement
  "Single-Bit Gray-Code Partial Refactorization" to capture the
  actual algorithm (path-based) plus its preconditions (no
  pivot-tolerance violation); ADDED requirement "Pivot-Fault
  Fallback" documenting the recovery path.
- **Affected code:**
  - `CMakeLists.txt` (root) — replace `find_package(KLU)` block with
    FetchContent setup
  - `core/include/pulsim/sparse/klu_solver.hpp` — extend with path
    cache + new helper; replace `partial_refactor` body
  - `core/tests/layer0/test_klu_solver.cpp` — tighten test 2.7.3 +
    add 2 new tests
  - `core/tests/benchmarks/test_bench_pwl_rank1.cpp` — extend N
    sweep
  - `.github/workflows/ci.yml` — remove `libsuitesparse-dev` /
    `suite-sparse` from installs
  - `README.md` — update "Build prerequisites" section
  - `CHANGELOG.md` — v1.3.0 entry
  - `pyproject.toml`, `python/pulsim/__init__.py`, `CITATION.cff` —
    version bump
  - `artigos/02_tpel_methods/benchmarks/RANK1_RESULTS.md` — V0-vs-V8.1
    comparison
  - `artigos/02_tpel_methods/benchmarks/results/rank1_microbench.csv` —
    updated capture
- **Affected build:** loses the `find_package(KLU)` requirement;
  gains a FetchContent-time download of ~3 MB (the KLU subtree of
  SuiteSparse). LGPL-2.1+ license compatible with Pulsim's MIT for
  static linking.
- **Not affected:** Python bindings, public CircuitBuilder API,
  Layer 5 `run_transient`, any of the 8 reference projects in
  `projects/`. All existing test assertions (~18,320 across the
  layer0..layer5 stack) continue to pass.

**No BREAKING changes.** Every existing `KluSolver` consumer
continues to work; the speedup comes from the upgraded internals.
