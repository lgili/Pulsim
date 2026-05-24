## 1. Drop KLU + add `Backend::Pulsim` enum slot

- [ ] 1.1 Delete `core/include/pulsim/sparse/klu_solver.hpp`
- [ ] 1.2 Delete `core/tests/layer0/test_klu_solver.cpp`
- [ ] 1.3 Remove the `find_package(KLU CONFIG)` block from root
      `CMakeLists.txt` (also remove `PULSIM_HAVE_KLU` /
      `PULSIM_ENABLE_KLU` options + the SuiteSparse-related summary line)
- [ ] 1.4 Remove `SuiteSparse::KLU` link from `core/CMakeLists.txt`
      and the `tests/layer0/test_klu_solver.cpp` source entry
- [ ] 1.5 In `core/include/pulsim/sparse/solver.hpp`:
      - Replace `Backend::KLU` with `Backend::Pulsim` in the enum
      - Drop the `#ifdef PULSIM_HAVE_KLU` conditional includes + factory
      - Forward-declare `PulsimSparseLuSolver` instead of `KluSolver`
      - Update factory: `Backend::Auto` picks Pulsim for any n
- [ ] 1.6 Remove `libsuitesparse-dev` / `suite-sparse` from
      `.github/workflows/ci.yml` (5 matrix entries + coverage + python
      Linux/macOS install commands)
- [ ] 1.7 Update README "Build prerequisites": drop the SuiteSparse KLU
      row; drop the `brew install suite-sparse` /
      `apt install libsuitesparse-dev` lines; drop the
      `-DPULSIM_ENABLE_KLU=OFF` opt-out note
- [ ] 1.8 Verify Eigen-only build still passes: layer0 + layer4 tests
      run unchanged after KLU removal (this is the V0 fallback path
      that always worked, now becomes the only path until Section 5
      lands)

## 2. `PulsimSparseLuSolver` — symbolic analysis layer

Implements `DirectSolver::analyze(const Matrix& M)`.

- [ ] 2.1 Create `core/include/pulsim/sparse/pulsim_lu_solver.hpp`
      skeleton (header-only, `class PulsimSparseLuSolver final :
      public DirectSolver`, RAII)
- [ ] 2.2 Add private `ColumnOrdering` struct holding the permutation
      vectors `Pcol` (column permutation from RCM/COLAMD) and `Pinv`
      (its inverse)
- [ ] 2.3 Implement Reverse Cuthill-McKee column ordering
      (`compute_rcm_ordering_(M)`) — George 1971 algorithm, simple
      starting point. Bandwidth-reducing; not as good as COLAMD but
      ~80 lines and well-understood. Returns the permutation vector.
- [ ] 2.4 Add private `EliminationTree` struct holding the `parent`
      array (parent of each column in the elimination tree) plus a
      `postorder` permutation.
- [ ] 2.5 Implement `compute_etree_(M_perm)` — Liu 1986 algorithm
      operating on the column-permuted matrix's pattern (Davis 2006
      §4.10, ~30 lines).
- [ ] 2.6 Implement `symbolic_factorize_(M_perm)` — given the etree
      and the row pattern of each column, compute the fill-in for L
      and U. Output: row-index arrays + column pointers for L and U
      (separately).
- [ ] 2.7 `analyze(M)` orchestrates: validate dims → compute_rcm →
      apply permutation → compute_etree → symbolic_factorize → set
      `analyzed_ = true`. Allocate L_, U_ structures (Eigen::SparseMatrix
      members) with the symbolic non-zero pattern.
- [ ] 2.8 Unit tests in new `test_pulsim_lu_solver.cpp`:
      - 2.8.1 analyze() on canonical SPD 3x3 succeeds, computes
        plausible etree
      - 2.8.2 analyze() on buck-like 8x8 succeeds, fill-in matches
        Eigen::SparseLU's count within +/- 50% (RCM is less optimal
        than COLAMD; the +/- 50% slack reflects that)
      - 2.8.3 analyze() on a 0x0 matrix returns false cleanly

## 3. `PulsimSparseLuSolver` — numeric factorization layer

Implements `DirectSolver::factorize(const Matrix& M)`.

- [ ] 3.1 Implement Gilbert-Peierls left-looking column step:
      `gp_column_eliminate_(k, M_perm, L, U)` — for column k, the
      sparse triangular solve `L[1:k-1, :]^{-1} a_k` reusing already-
      stored L columns. Davis 2006 §3 has the reference implementation.
- [ ] 3.2 Implement partial pivoting within each column: search for
      the largest-magnitude entry in the current column's lower-half,
      swap row with current k. Update permutation Prow + L's
      already-built rows.
- [ ] 3.3 `factorize(M)` orchestrates: column-by-column loop, calls
      gp_column_eliminate + partial_pivot per column. On zero pivot,
      set numeric_singular_ flag, return false.
- [ ] 3.4 Add private `Prow` row-permutation vector member (updated
      by pivoting).
- [ ] 3.5 Unit tests:
      - 3.5.1 factorize() on SPD 3x3 → output L*U == P_row * M *
        P_col within 1e-12 (verify via Eigen::SparseMatrix multiply)
      - 3.5.2 factorize() on buck-like 8x8 (asymmetric MNA) → same
        identity within 1e-12
      - 3.5.3 factorize() on a deliberately-singular matrix → returns
        false, numeric_singular_ flag set
      - 3.5.4 Pivoting test: matrix that would zero-pivot without
        partial pivoting → factorize() succeeds via row swap

## 4. `PulsimSparseLuSolver` — triangular solve

Implements `DirectSolver::solve(const Vector& b, Vector& x) const`.

- [ ] 4.1 Implement forward substitution `solve_lower_(L, b, y)` —
      solves Ly = P_row · b, where y is the intermediate vector.
- [ ] 4.2 Implement back substitution `solve_upper_(U, y, x)` —
      solves Ux = y, then permutes via P_col to produce final x.
- [ ] 4.3 `solve(b, x)` orchestrates both with the stored permutations.
- [ ] 4.4 Unit tests:
      - 4.4.1 solve() on SPD 3x3 with reference b → x matches
        Eigen::SparseLU's solution within 1e-12
      - 4.4.2 solve() on buck-like 8x8 → same parity
      - 4.4.3 solve() before factorize() → throws std::logic_error
      - 4.4.4 solve() after factorize() of multiple different M (same
        symbolic) → each result correct (factor is overwritten by
        each new factorize call)

## 5. `PulsimSparseLuSolver` — path-based partial refactor

Implements `DirectSolver::partial_refactor(new_M, changed_cols)`.

- [ ] 5.1 Add private `path_state_` — { union of changed cols seen,
      cached path (vector<int>), path_valid_ bool, path_compute_count_
      diagnostic }. Mirrors KluSolver's lazy-union design from V8.1
      (option B's old plan).
- [ ] 5.2 Implement `compute_path_(varying_columns)` — for each
      varying column, traverse the etree from that column up to the
      root, marking nodes. The union of all traversed nodes is the
      path. Dinkelbach §3.1.
- [ ] 5.3 Implement `re_eliminate_path_(new_M_perm, path, L, U)` —
      re-run gp_column_eliminate (from Section 3) only for the columns
      in `path`. The numerical re-elimination uses the current
      (cached) factor for columns NOT in the path.
- [ ] 5.4 Add pivot-validity check: after each path-column
      re-elimination, verify the new diagonal pivot magnitude ≥
      `pivot_tol_fail` (default 1e-3 of the column's largest entry).
      On violation, set pivot_fault_ flag.
- [ ] 5.5 `partial_refactor(new_M, changed_cols)` orchestrates:
      - Empty changed_cols → return true (no-op)
      - Compute new union; if grew, invalidate cached path
      - If !path_valid_, call compute_path_; if it fails, return false
      - Apply column permutation to new_M
      - Call re_eliminate_path_
      - If pivot_fault_: clear path_state_, return false (caller
        falls back to full factorize)
      - Else: return true
- [ ] 5.6 Override `supports_partial_refactor()` → returns true.
- [ ] 5.7 Unit tests:
      - 5.7.1 partial_refactor after value perturbation produces
        L, U that match a fresh full factorize within 1e-14 (bit-exact
        per Dinkelbach §3.2)
      - 5.7.2 Repeated identical changed_cols hit the path cache
        (verify via `path_compute_count()` diagnostic)
      - 5.7.3 Pivot-fault case (constructed perturbation) returns
        false + invalidates path cache + a subsequent full factorize
        succeeds
      - 5.7.4 changed_cols that introduces a previously-unseen column
        forces path recompute (verify count incremented)
      - 5.7.5 `analyze()` call clears the path cache

## 6. Integration + bench + close-out

- [ ] 6.1 Update `core/tests/layer4/test_pwl_cache_rank1.cpp` test
      5.2 — simplify the "either rank1_hits or fallbacks" branch
      since there's only one backend with partial_refactor support
      now.
- [ ] 6.2 Extend
      `core/tests/benchmarks/test_bench_pwl_rank1.cpp` to capture
      three columns: baseline `solve` (per-mask cache), `solve_rank1`
      with Eigen::SparseLU forced (this falls back to full factorize
      every flip since Eigen doesn't support partial_refactor), and
      `solve_rank1` with PulsimSparseLuSolver (the V8.1 path-based win).
- [ ] 6.3 Re-run microbench across N ∈ {4, 6, 8, 10, 12, 16, 20, 24}.
      Capture CSV.
- [ ] 6.4 Rewrite
      `artigos/02_tpel_methods/benchmarks/RANK1_RESULTS.md` —
      3-column comparison table + interpretation paragraph + honest
      limitations + updated reproduction recipe.
- [ ] 6.5 Verify NPC + MMC validation notebooks under `projects/`
      still produce bit-identical output (use the Python pp.simulate
      path; the kernel changes don't touch run_transient's solve()
      call site, so this should be unchanged).
- [ ] 6.6 CHANGELOG `[1.3.0]` entry — explain the architectural pivot
      (KLU dropped, PulsimSparseLuSolver added, RCM-based ordering as
      MVP). Reference RANK1_RESULTS.md for the captured speedups.
- [ ] 6.7 Bump version 1.2.0 → 1.3.0 in pyproject.toml,
      python/pulsim/__init__.py, CITATION.cff.
- [ ] 6.8 Run `openspec validate replace-klu-with-pulsim-sparse-lu
      --strict`; resolve issues.
- [ ] 6.9 Open PR feat/replace-klu-with-pulsim-sparse-lu → main.
- [ ] 6.10 Post-merge: archive the change.

## Out of scope (future proposals)

- COLAMD or AMD fill-reducing ordering (RCM MVP first; replace once
  numerical validation is solid)
- BTF block-triangular decomposition (the upstream KLU's secret
  sauce for circuit MNA; add only if benchmarks show RCM-only is
  unacceptably slow)
- Multi-bit partial refactor (single-bit Gray-code is the common
  case; multi-bit goes to full factorize per current `solve_rank1`
  contract)
- GPU offload, parallelism within factorization
- `klu_analyze_partial`-style varying-set-aware ordering (V8.2+)
