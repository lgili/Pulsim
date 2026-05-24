## 1. CMake + dependency wiring

- [ ] 1.1 Add `find_package(SuiteSparse COMPONENTS KLU)` to `core/CMakeLists.txt`
- [ ] 1.2 Add `PULSIM_HAVE_KLU` compile definition gated on KLU presence
- [ ] 1.3 Update CI matrix to install `libsuitesparse-dev` on Linux + `suite-sparse` via brew on macOS
- [ ] 1.4 Document new optional dependency in top-level `README.md` "Build prerequisites" section

## 2. KluSolver implementation

- [ ] 2.1 Create `core/include/pulsim/sparse/klu_solver.hpp` skeleton (header-only or paired .cpp)
- [ ] 2.2 Implement `analyze()` via `klu_analyze` (Aᵀ pattern — KLU is CSC)
- [ ] 2.3 Implement `factorize()` via `klu_factor`
- [ ] 2.4 Implement `solve()` via `klu_solve` (single RHS, in-place)
- [ ] 2.5 Implement `supports_partial_refactor()` returning `true`
- [ ] 2.6 Implement `partial_refactor(new_M, changed_cols)` via path-based re-elimination (Chen et al. 2024 §III)
- [ ] 2.7 Unit tests under `core/tests/layer0/test_klu_solver.cpp`:
  - [ ] 2.7.1 `analyze + factorize + solve` parity with `SparseLuSolver` on a 50-node random sparse SPD matrix
  - [ ] 2.7.2 Same on a real Pulsim buck cache segment (n=8)
  - [ ] 2.7.3 `partial_refactor` parity: full refactor vs partial refactor of same single-column change produce identical output (within 1e-12)

## 3. Factory + Backend hint

- [ ] 3.1 Add `Backend { Auto, Eigen, KLU }` enum in `sparse/solver.hpp`
- [ ] 3.2 Add `DirectSolver::supports_partial_refactor()` virtual default `false`
- [ ] 3.3 Add `DirectSolver::partial_refactor(...)` virtual default `false`
- [ ] 3.4 Add overload `make_default_solver(Size n, Backend hint = Backend::Auto)`
- [ ] 3.5 Implement `Backend::Auto` heuristic: KLU when `n >= PULSIM_KLU_AUTO_THRESHOLD` (default 100) AND `PULSIM_HAVE_KLU`
- [ ] 3.6 Implement `Backend::KLU` explicit request: throws `std::runtime_error` if `!PULSIM_HAVE_KLU`
- [ ] 3.7 Unit tests for factory behaviour with/without `PULSIM_HAVE_KLU`

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
