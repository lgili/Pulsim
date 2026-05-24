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

- [x] 4.1 Add `pulsim::pwl::CacheMetrics` struct with 3 monotonic counters (`rank1_hits`, `full_refactor_hits`, `fallbacks`). Atomic uint64 for thread-safe sampling.
- [x] 4.2 Add `rank1_solver_` + `rank1_mask_` + `rank1_b_constant_` + `rank1_initialized_` members to `PwlStateSpaceCache` (all `mutable` so `solve_rank1` stays const-correct).
- [x] 4.3 Bit-difference computed inline via `std::popcount(mask_curr.bits() ^ mask_prev.bits())`.
- [x] 4.4 Implement `compute_changed_columns_(prev_mask, curr_mask)` — walks the graph's BranchKind::Switch branches in order, maps each flipped bit to its switch's (from, to) MNA columns. Returns `std::vector<Index>` for the V8.1 path-based-partial-refactor follow-up; the V8 MVP `KluSolver::partial_refactor` accepts but ignores the hint.
- [x] 4.5 Implement `solve_rank1(mask, b_extra, x)` with 4 branches (first call / same mask / single-bit + supported / multi-bit OR unsupported), each updating the right counter. Sliding solver pattern: `analyze` runs once on first call, every subsequent call only `factorize` or `partial_refactor` (the sparsity pattern is invariant across switch states).
- [x] 4.6 Add `metrics() const noexcept` accessor returning a `CacheMetrics` snapshot via `std::memory_order_relaxed` loads.

## 5. Integration tests

- [x] 5.1 `core/tests/layer4/test_pwl_cache_rank1.cpp` — Gray-code sweep over all 16 masks of a 4-switch fixture; verify `solve_rank1` output matches `solve` (per-mask path) within 1e-12 at every step.
- [x] 5.2 Multi-bit + first-encounter + single-bit mix: assert exact counter partitioning. The single-bit-diff case bumps `rank1_hits` on KLU build / `fallbacks` on Eigen build — test accepts either via `(rank1_hits + fallbacks) == 1` invariant.
- [x] 5.3 Orthogonality test: 10 `solve_rank1` calls on a `build_lazy(dt)`-only cache leave `num_built_segments() == 0` — proves `solve_rank1` is independent of the per-mask `segments_` map.
- [x] 5.4 Same-mask repeats: `solve_rank1` called 3× with identical mask increments `rank1_hits` only on the 2 repeats; first call hits `full_refactor_hits`. Refreshes `b_constant` without refactor.
- [x] 5.5 `metrics()` on a fresh cache returns `{0, 0, 0}`.

## 6. Benchmark suite — re-scoped to C++ microbenchmark

> **Scope shift recorded 2026-05-24:** the originally-planned Python
> per-converter benchmark extension requires `solve_rank1` to be wired
> into Layer 5's `run_transient` and exposed through the Python bindings
> — neither of which lives in this OpenSpec change. The honest fast
> path was to re-scope Section 6 to a kernel-level C++ microbenchmark
> that exercises `cache.solve_rank1` directly. The original Python
> work moves to a follow-up proposal `add-pwl-rank1-runtime-integration`
> (TBD). Detailed in `artigos/02_tpel_methods/benchmarks/RANK1_RESULTS.md`
> "Honest limitations" section.

- [x] 6.1 Author `core/tests/benchmarks/test_bench_pwl_rank1.cpp` —
      Catch2 binary inside the opt-in `pulsim_benchmarks` target. Uses
      `bench_helpers::Stopwatch` for clean wall-clock measurement,
      writes per-N CSV rows.
- [x] 6.2 Add `set_rank1_backend(Backend)` setter to `PwlStateSpaceCache`
      so the benchmark can force `Backend::KLU` regardless of the
      synthetic fixture's tiny `state_size` (avoids needing a build-time
      flag override).
- [x] 6.3 Sweep N ∈ {4, 6, 8, 10, 12} switches on a synthetic N-switch
      chain fixture. Capture baseline vs rank-1 wall-time, per-call cost,
      speedup ratio, and counter partition (rank1_hits / full_refactor_hits
      / fallbacks).
- [x] 6.4 Write captured CSV to
      `artigos/02_tpel_methods/benchmarks/results/rank1_microbench.csv`
      (path override via `PULSIM_BENCH_RESULTS_DIR` env var).
- [x] 6.5 Write `artigos/02_tpel_methods/benchmarks/RANK1_RESULTS.md` —
      reproduction recipe, captured table, interpretation, and honest
      limitations (synthetic fixture, n_state capped at 14, microbench
      forces backend hint).

> **DEFERRED** to follow-up proposal `add-pwl-rank1-runtime-integration`:
> - Per-converter benchmarks on the 10 real reference projects (buck →
>   MMC). Requires Layer 5 + Python wire-up first.

## 7. Validation + spec close-out

- [ ] 7.1 Re-run NPC and MMC validation notebooks (`projects/inverters/{npc_3phase,mmc}/00_*.ipynb`); verify output bit-identical to pre-rank1 baseline
- [ ] 7.2 Run `openspec validate add-pwl-rank1-update --strict`; resolve any issues
- [ ] 7.3 Update `CHANGELOG.md` with v1.2.0 entry summarising the rank-1 path
- [ ] 7.4 Bump version: `pyproject.toml`, `python/pulsim/__init__.py`, `CITATION.cff` → 1.2.0
- [ ] 7.5 Archive the change: move to `openspec/changes/archive/YYYY-MM-DD-add-pwl-rank1-update/` per `openspec/AGENTS.md` Stage 3
