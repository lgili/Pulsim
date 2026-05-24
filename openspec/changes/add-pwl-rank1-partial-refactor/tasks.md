## 1. Build / dependency wiring (vendor DPsim KLU fork)

- [x] 1.1 SHA pinned: `6cf768091962336466808e7f02d476842e4c5281`
      (master @ 2023-03-08, last commit on the fork's master branch).
      Verified the path-based files exist in this commit
      (`klu_compute_path.c`, `klu_partial_factorization_path.c`,
      `klu_partial_refactorization_restart.c`). Documented in
      `LICENSES/SuiteSparse-DPsim-fork.md`.
- [x] 1.2 Replaced the `find_package(KLU CONFIG)` block in root
      `CMakeLists.txt` with `FetchContent_Declare(klu_dpsim ...
      GIT_TAG 6cf76809)` + `FetchContent_MakeAvailable`. Sets
      `WITH_GPL=OFF` to skip any GPL-gated bits; fork's CMake only
      builds 5 targets (amd, colamd, btf, suitesparseconfig, klu) so
      no CHOLMOD/UMFPACK contamination. `klu` static lib aliased as
      `SuiteSparse::KLU` so the V8 link line in `core/CMakeLists.txt`
      keeps working unchanged.
- [x] 1.3 `LICENSES/` directory created:
  - `LICENSES/README.md` — index of bundled third-party licenses +
    redistribution requirements (LGPL-2.1+ relinking + attribution
    + ship-the-LICENSES-dir)
  - `LICENSES/LGPL-2.1.txt` — full 501-line GNU LGPL 2.1 text
    (fetched from www.gnu.org)
  - `LICENSES/SuiteSparse-DPsim-fork.md` — provenance record: source
    URL, pinned commit SHA, capture date, list of new API surface
    + new `klu_numeric` fields + new error codes vs upstream, plus
    re-verification recipe
- [x] 1.4 No code change needed — the alias `SuiteSparse::KLU` keeps
      the existing `target_link_libraries(... SuiteSparse::KLU)` line
      in `core/CMakeLists.txt` working without edits. `PULSIM_HAVE_KLU=1`
      compile def still propagated.
- [x] 1.5 CI workflow updates (`.github/workflows/ci.yml`):
  - [x] 1.5.1 Removed `libsuitesparse-dev` from every Linux apt install
        command (4 matrix entries + coverage job)
  - [x] 1.5.2 Removed `suite-sparse` from the macOS brew install (2 jobs)
  - [ ] 1.5.3 No explicit "verify FetchContent succeeded" step —
        skipped because a successful build already implies success.
- [x] 1.6 README "Build prerequisites" — removed per-platform
      `libsuitesparse-dev` / `suite-sparse` install instructions;
      added a row in the dependency table marking KLU as
      "bundled (vendored)" with the SHA + fork URL + license summary.

## 2. KluSolver upgrade (path-based partial refactor)

- [ ] 2.1 Add private `struct PathState` to `klu_solver.hpp` — RAII
      wrapper holding the malloc'd `path` / `block_path` /
      `variable_block` / `variable_offdiag_*` arrays from
      `klu_compute_path`. Destructor frees them.
- [ ] 2.2 Add private `path_cache_` member — `unordered_map<uint64_t,
      PathState>` keyed on `hash(sorted(changed_cols))`. Mutable so
      `partial_refactor` stays const-safe.
- [ ] 2.3 Add private helper `ensure_path_for(changed_cols, A_arrays)`:
  - [ ] 2.3.1 Compute hash of `changed_cols` (sorted, deduplicated)
  - [ ] 2.3.2 If hit in cache → return path pointer
  - [ ] 2.3.3 Else call `klu_compute_path(symbolic_, numeric_, common_,
        Ap, Ai, varying_cols, varying_rows, n_varying_entries)` to
        populate; clone the populated arrays out into a new `PathState`
        for caching (the function over-allocates internally — we
        deep-copy to avoid surprise free)
- [ ] 2.4 Replace `KluSolver::partial_refactor(new_M, changed_cols)`
      body:
  - [ ] 2.4.1 If `changed_cols.empty()` → fast-path return true (no
        change to refactor)
  - [ ] 2.4.2 Call `ensure_path_for(changed_cols, new_M-arrays)`
  - [ ] 2.4.3 Temporarily install the cached path on `numeric_->path`
        etc. (or use the same numeric we already mutated — confirm
        from the C source whether `klu_compute_path` mutates `numeric_`)
  - [ ] 2.4.4 Call `klu_partial_factorization_path(Ap, Ai, Ax,
        symbolic_, numeric_, common_)`
  - [ ] 2.4.5 Map return value:
        - return `TRUE` + `common_.status == KLU_OK` → return `true`
        - return `FALSE` + `common_.status == KLU_PIVOT_FAULT` →
          invalidate `path_cache_` (pivots may have shifted!) + return `false`
        - return `FALSE` + `common_.status == KLU_SINGULAR` → return
          `false` (caller falls back to `factorize`)
- [ ] 2.5 Confirm `supports_partial_refactor()` still returns `true`
      (no signature change).

## 3. Tests

- [ ] 3.1 **Tighten** existing test 2.7.3 — assertion changes from
      "x_partial matches x_full within 1e-12" to "L and U factor
      values match full refactor within 1e-14" (extract via
      `klu_extract`). Documents the path-based algorithm's bit-exact
      equivalence under unchanged pivots.
- [ ] 3.2 **Add** new test
      `test_klu_solver.cpp::partial_refactor_falls_back_on_pivot_fault`:
      construct a 2-mask sequence where the perturbation forces a
      pivot below `common_.pivot_tol_fail`; verify `partial_refactor`
      returns `false` AND the next `factorize` succeeds AND the path
      cache is invalidated.
- [ ] 3.3 **Add** new test
      `test_klu_solver.cpp::partial_refactor_reuses_path_cache`: call
      `partial_refactor` 3 times with identical `changed_cols`;
      verify the path is computed only once (probe via a
      `path_cache_size()` introspection accessor or via timing).

## 4. Bench + close-out

- [ ] 4.1 Re-run `pulsim_benchmarks "[rank1][microbench]"` with the
      V8.1 path-based KluSolver. Capture wall-time CSV.
- [ ] 4.2 Extend the microbench's N sweep upward: target N ∈
      {12, 16, 20, 24} to push toward the MMC-scale regime where the
      audit predicts 5-10× speedup over V0 MVP.
- [ ] 4.3 Add V8.1 columns to
      `artigos/02_tpel_methods/benchmarks/results/rank1_microbench.csv`:
      `wall_v8_1_ms`, `speedup_v8_1_over_v0`.
- [ ] 4.4 Update `artigos/02_tpel_methods/benchmarks/RANK1_RESULTS.md`
      with a 3-column table (baseline / V0 MVP / V8.1 path-based) +
      interpretation paragraph.
- [ ] 4.5 CHANGELOG `[1.3.0]` entry — captured speedup table,
      reference to DPsim fork SHA, license attribution to KLU
      maintainers.
- [ ] 4.6 Bump version 1.2.0 → 1.3.0 in `pyproject.toml`,
      `python/pulsim/__init__.py`, `CITATION.cff`.
- [ ] 4.7 Run `openspec validate add-pwl-rank1-partial-refactor --strict`;
      resolve any issues. Open PR feat → main.
- [ ] 4.8 Post-merge: archive the change per `openspec/AGENTS.md` Stage 3.
