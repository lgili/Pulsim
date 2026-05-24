## 1. CMake + dependency wiring

- [x] 1.1 Add `find_package(KLU CONFIG)` (with Homebrew fallback + legacy
      `find_library(klu)` last resort) to root `CMakeLists.txt`; gate the
      whole block behind `option(PULSIM_ENABLE_KLU ON)`.
- [x] 1.2 Add `PULSIM_HAVE_KLU=1` compile definition + `SuiteSparse::KLU`
      INTERFACE link to `pulsim_core` in `core/CMakeLists.txt`, conditional
      on `PULSIM_HAVE_KLU` being set by step 1.1.
- [x] 1.3 Update CI matrix to install `libsuitesparse-dev` on Linux + `suite-sparse` via brew on macOS (`.github/workflows/ci.yml` patched across all 5 matrix entries + coverage job)
- [x] 1.4 Document new optional dependency in top-level `README.md` "Build prerequisites" section, with install commands for macOS / Debian / Fedora + the `-DPULSIM_ENABLE_KLU=OFF` opt-out

## 2. KluSolver implementation

- [x] 2.1 Create `core/include/pulsim/sparse/klu_solver.hpp` — header-only, gated on `#ifdef PULSIM_HAVE_KLU`
- [x] 2.2 Implement `analyze()` via `klu_analyze` (Eigen ColMajor CSC pointers pass through directly)
- [x] 2.3 Implement `factorize()` via `klu_factor`; frees stale numeric factor first
- [x] 2.4 Implement `solve()` via `klu_solve` (single RHS, in-place; copies `b → x` first)
- [x] 2.5 Implement `supports_partial_refactor()` returning `true`
- [x] 2.6 Implement `partial_refactor` **MVP V0** — delegates to `klu_refactor` (full numeric refactor reusing the cached symbolic). Wins over Eigen's `factorize()` by reusing the COLAMD ordering. Path-based re-elimination per Chen et al. 2024 §III deferred to a V8.1 follow-up commit; the MVP unblocks the rest of the proposal.
- [x] 2.7 Unit tests under `core/tests/layer0/test_klu_solver.cpp`:
  - [x] 2.7.1 Parity vs `SparseLuSolver` on SPD 3x3 within 1e-12
  - [x] 2.7.2 Parity on representative buck-cache-like 8x8 asymmetric MNA within 1e-12
  - [x] 2.7.3 `partial_refactor` parity: same output as fresh full factor on a perturbed system within 1e-12
  - [x] Plus: `supports_partial_refactor()` advertises correctly; lifecycle errors throw `std::logic_error`

## 3. Factory + Backend hint

- [x] 3.1 Add `Backend { Auto, Eigen, KLU }` enum in `sparse/solver.hpp`
- [x] 3.2 Add `DirectSolver::supports_partial_refactor()` virtual default `false`
- [x] 3.3 Add `DirectSolver::partial_refactor(...)` virtual default `false`
- [x] 3.4 Add overload `make_default_solver(Size n, Backend hint = Backend::Auto)` — declaration in `solver.hpp`, KLU-aware impl in `klu_solver.hpp` (ODR-safe via `#ifdef`)
- [x] 3.5 Implement `Backend::Auto` heuristic: KLU when `n >= PULSIM_KLU_AUTO_THRESHOLD` (default 100) AND `PULSIM_HAVE_KLU`
- [x] 3.6 Implement `Backend::KLU` explicit request: throws `std::runtime_error` if `!PULSIM_HAVE_KLU` (fallback impl in `solver.hpp`); succeeds at any n when KLU is built
- [x] 3.7 Unit tests for factory behaviour: Auto crossover at threshold, Eigen always honoured, KLU always honoured (when built)

## 4. PWL cache rank-1 fast-path

- [ ] 4.1 Add `pulsim::pwl::CacheMetrics` struct (3 atomic uint64 counters)
- [ ] 4.2 Add `previous_mask_` + `previous_segment_*` members to `PwlStateSpaceCache`
- [ ] 4.3 Implement bit-difference helper `popcount(mask_a XOR mask_b)`
- [ ] 4.4 Implement `compute_changed_columns(mask_prev, mask_curr)` — maps toggled-switch branch_id to MNA column indices via DevicePool
- [ ] 4.5 Implement `solve_rank1(mask, b_extra, x)`:
  - [ ] 4.5.1 If `previous_mask_` unset OR `popcount != 1`: dispatch to existing `solve(mask, b_extra, x)`, increment `full_refactor_hits`
  - [ ] 4.5.2 Else if `!solver->supports_partial_refactor()`: dispatch to `solve`, increment `fallbacks`
  - [ ] 4.5.3 Else: assemble updated J, call `partial_refactor`, increment `rank1_hits` on success
  - [ ] 4.5.4 On `partial_refactor` returning false: dispatch to `solve`, increment `fallbacks`
- [ ] 4.6 Add `metrics() const noexcept` accessor

## 5. Integration tests

- [ ] 5.1 `core/tests/layer4/test_pwl_cache_rank1.cpp` — single-bit flip on MMC N=3 cache: 100 alternating mask switches; verify `metrics().rank1_hits == 99` AND output bit-identical to baseline
- [ ] 5.2 Multi-bit flip falls back: 50 random masks (≥ 2-bit diff each), verify `metrics().full_refactor_hits == 50`
- [ ] 5.3 `SparseLuSolver` backend: verify `metrics().fallbacks` accumulates correctly
- [ ] 5.4 Numerical-singularity poisoning test: construct a switch state whose partial refactor would fail (rank-deficient); verify fallback engages cleanly

## 6. Benchmark suite extension

- [ ] 6.1 Extend `artigos/02_tpel_methods/benchmarks/buck/run_buck_benchmark.py` to report `metrics().rank1_hits / total_solves` per run
- [ ] 6.2 Add new columns to `<topology>_summary.csv`: `rank1_hit_rate`, `wall_s_full_refactor`, `wall_s_rank1`
- [ ] 6.3 Author benchmark runners for the remaining 9 converters (boost, buck-boost, forward, flyback, half-bridge LLC, boost PFC, VSI 3φ, NPC 3-level, MMC N=3)
- [ ] 6.4 Re-run all 10; commit `<topology>_summary.csv` rows to `artigos/02_tpel_methods/benchmarks/results/`
- [ ] 6.5 Write 1-page report `artigos/02_tpel_methods/benchmarks/RANK1_RESULTS.md` — table of speedups, narrative on crossover at n≈100

## 7. Validation + spec close-out

- [ ] 7.1 Re-run NPC and MMC validation notebooks (`projects/inverters/{npc_3phase,mmc}/00_*.ipynb`); verify output bit-identical to pre-rank1 baseline
- [ ] 7.2 Run `openspec validate add-pwl-rank1-update --strict`; resolve any issues
- [ ] 7.3 Update `CHANGELOG.md` with v1.2.0 entry summarising the rank-1 path
- [ ] 7.4 Bump version: `pyproject.toml`, `python/pulsim/__init__.py`, `CITATION.cff` → 1.2.0
- [ ] 7.5 Archive the change: move to `openspec/changes/archive/YYYY-MM-DD-add-pwl-rank1-update/` per `openspec/AGENTS.md` Stage 3
