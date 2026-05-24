## 1. Build / dependency wiring (vendor DPsim KLU fork)

- [ ] 1.1 Pick a pinned commit SHA of dpsim-simulator/SuiteSparse to
      vendor (must contain `KLU/Source/klu_compute_path.c` +
      `klu_partial_factorization_path.c`). Record SHA + date in design.md.
- [ ] 1.2 Replace the `find_package(KLU CONFIG)` block in root
      `CMakeLists.txt` with `FetchContent_Declare(klu_dpsim GIT_REPOSITORY ...
      GIT_TAG <sha>)` + `FetchContent_MakeAvailable`. Configure
      SuiteSparse's CMake to build ONLY the KLU + BTF + AMD + COLAMD
      sub-projects (skip CHOLMOD, UMFPACK, etc.). License audit on the
      4 vendored sub-projects (KLU: LGPL-2.1+; BTF: LGPL-2.1+; AMD:
      BSD-3; COLAMD: BSD-3).
- [ ] 1.3 Add a `LICENSES/` entry recording the LGPL-2.1+ text + the
      SHA we vendored. JOSS-style provenance: which file came from where.
- [ ] 1.4 CMake: link `KLU` (the new vendored static lib) into
      `pulsim_core` as INTERFACE, replacing the previous
      `SuiteSparse::KLU` imported target. Keep `PULSIM_HAVE_KLU=1`
      semantics intact (still `#ifdef`-gated).
- [ ] 1.5 CI workflow updates:
  - [ ] 1.5.1 Remove `libsuitesparse-dev` from Linux apt install commands
  - [ ] 1.5.2 Remove `suite-sparse` from macOS brew install commands
  - [ ] 1.5.3 Add CI step that verifies the FetchContent download
        succeeded (cache-friendly)
- [ ] 1.6 README "Build prerequisites" — remove per-platform KLU install
      instruction; add a 1-paragraph note that KLU is vendored from the
      DPsim fork at commit `<sha>` and explain how to override via
      `-DPULSIM_KLU_FETCH_SOURCE=local` for air-gapped builds.

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
