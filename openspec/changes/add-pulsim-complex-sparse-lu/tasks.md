## 1. Template `PulsimSparseLuSolver` on `Scalar`

- [x] 1.1 Renamed `class PulsimSparseLuSolver` to
      `template <typename Scalar> class PulsimSparseLuSolverT` in
      `core/include/pulsim/sparse/pulsim_lu_solver.hpp`. Backward-
      compat alias (matching the `MatrixT`/`Matrix` and
      `DirectSolverT`/`DirectSolver` pattern):
      ```cpp
      using PulsimSparseLuSolver        = PulsimSparseLuSolverT<Real>;
      using PulsimComplexSparseLuSolver = PulsimSparseLuSolverT<std::complex<Real>>;
      ```
      Every Layer 1-9 consumer that writes `PulsimSparseLuSolver
      solver;` keeps compiling unchanged.
- [x] 1.2 Replaced `Matrix` (which is `Eigen::SparseMatrix<Real,
      ColMajor, Index>`) with `MatrixType = typename
      DirectSolverT<Scalar>::MatrixType = MatrixT<Scalar>` in the
      class template's signatures + workspace types. The
      `MatrixT<Scalar>` alias was added to `pulsim/sparse/matrix.hpp`
      alongside the legacy `Matrix = MatrixT<Real>` alias.
      `l_values_` / `u_values_` switched from `std::vector<Real>` to
      `std::vector<Scalar>`.
- [x] 1.3 Replaced `Vector` with `VectorType = typename
      DirectSolverT<Scalar>::VectorType = pulsim::VectorT<Scalar>`
      in `solve()` and `partial_refactor()` workspace types.
      `VectorT<Scalar>` lives in `pulsim/numeric/dense.hpp` next
      to the legacy `Vector = VectorT<Real>` alias.
- [x] 1.4 Updated `core/include/pulsim/sparse/solver.hpp`:
      `DirectSolverT<Scalar=Real>` and `SparseLuSolverT<Scalar=Real>`
      are now class templates, with `DirectSolver = DirectSolverT<Real>`
      and `SparseLuSolver = SparseLuSolverT<Real>` shims for source
      compatibility. Forward decl is
      `template <typename Scalar = Real> class PulsimSparseLuSolverT;`
      so the default lives on the declaration only (C++ forbids
      restating defaults on the definition).
- [x] 1.5 Verified the elimination-tree / RCM-ordering / symbolic-
      pattern / path-computation code is structural and Scalar-
      agnostic — those routines only touch `outerIndexPtr()` and
      `innerIndexPtr()`. The two Scalar-touching points are: (a)
      `std::abs(x[i])` for pivot magnitude (returns `Real` for both
      `double` and `std::complex<double>`); (b) the L-update inner
      loop's `x[i] -= L * x[j]` (overloaded `*` / `-=` work for
      both). Real-scalar regression suite: 478/478 pass post-
      refactor (no algorithmic change).

## 2. Complex specialisation of the pivot check

- [x] 2.1 Confirmed: `std::abs(std::complex<Real>)` returns
      `std::sqrt(re² + im²)` (the LAPACK ZGETRF magnitude). No
      behavioural change vs the real case — the existing
      partial-pivoting argmax keeps semantics. Implementation
      retains the `max_abs = std::abs(x[k])` initialiser typed
      as `Real`; `std::abs(Scalar)` always returns `Real`.
- [x] 2.2 `PIVOT_THRESH = 1e-3` and `PIVOT_TOL = 1e-14` are
      `constexpr Real`. They land on relative magnitudes — both
      sides of the comparison are `Real` regardless of whether
      `Scalar` is `Real` or `std::complex<Real>`. No change
      needed.
- [ ] 2.3 `numeric_singular_threshold()` accessor — DEFERRED.
      None of the v1.4.0 fixtures regress on the default
      threshold; revisit only if AC sweep at low-condition-number
      frequencies starts producing spurious fallbacks.

## 3. Update factory + builder for the template

- [x] 3.1 Added `template <typename Scalar>
      [[nodiscard]] std::unique_ptr<DirectSolverT<Scalar>>
      make_default_solver_t(Size n, Backend hint = Backend::Auto);`
      Legacy non-template `make_default_solver(n, hint)` is a
      backward-compat shim that dispatches to
      `make_default_solver_t<Real>(n, hint)`.
- [x] 3.2 `Backend::Pulsim` (and `Backend::Auto`) → returns
      `std::make_unique<PulsimSparseLuSolverT<Scalar>>()`.
      `Backend::Eigen` → returns
      `std::make_unique<SparseLuSolverT<Scalar>>()`. New complex
      test `make_default_solver_t<Complex>: factory returns
      expected backend` exercises both branches and confirms
      output parity within 1e-10.
- [x] 3.3 `SparseLuSolverT<Scalar=Real>` wraps
      `Eigen::SparseLU<MatrixT<Scalar>, COLAMDOrdering<Index>>`.
      Backward-compat alias `SparseLuSolver = SparseLuSolverT<Real>`.
      `Backend::Eigen` retained as the IEEE TPEL §VI.B AC-sweep
      paper-comparison baseline.

## 4. Switch `mna_sweep.hpp` to the in-house complex solver

- [x] 4.1 Replaced the inline `Eigen::SparseLU<ComplexSparseMatrix,
      Eigen::COLAMDOrdering<Index>> solver;` with
      `sparse::PulsimComplexSparseLuSolver solver;`.
- [x] 4.2 Replaced `analyzePattern(M)` + `factorize(M)` with
      `analyze(M)` + `factorize(M)` returning `bool`. Both
      checked with explicit `if (!...) throw std::runtime_error`.
- [x] 4.3 Replaced `solver.info() != Eigen::Success` with the
      `bool` return of `factorize` + `numeric_singular()` for
      diagnostics. Runtime-error message now reads "complex
      numeric factorisation (Pulsim) failed at f=… — matrix is
      numerically singular" when the pivot tolerance fires.
- [x] 4.4 Switched `X = solver.solve(B)` to the in-place
      `solver.solve(B, X)` signature matching the
      `DirectSolverT::solve` contract.
- [x] 4.5 Dropped `#include <Eigen/SparseLU>`. Pulled in
      `pulsim/sparse/matrix.hpp` + `pulsim/sparse/solver.hpp`
      instead. Also changed `ComplexSparseMatrix` from RowMajor
      to ColMajor (`sparse::MatrixT<Complex>`) so the in-house
      solver consumes it without transpose-and-copy.

## 5. Test coverage — synthetic complex fixtures

- [x] 5.1 New file `core/tests/layer0/test_pulsim_lu_solver_complex.cpp`
      (5 test cases, 31 assertions, all green):
      - [x] 5.1.1 Hermitian PD 3×3: analyze succeeds, factorize
        identity `(L+I)·U == P_row·M·P_col` within `1e-12`
        complex-magnitude tolerance; solve against M·x_true=b
        recovers x_true within `1e-12`.
      - [x] 5.1.2 Asymmetric complex MNA 8×8 (= real buck-like
        fixture with `+j·ω·C` on the R2 cap pair at f=1 MHz):
        same identity at `1e-12`; residual `|M·x_hat − b|` at
        `1e-10`. Forces partial pivoting at the voltage-source
        row (zero diagonal).
      - [x] 5.1.3 partial_refactor on a single-column perturbation
        (col 3, imaginary part scaled 1.1× → mimics ω change in
        an AC sweep): post-refactor solve vs fresh-factorise
        within `1e-10`; perturbation is non-trivial vs the
        pre-refactor solve.
      - [x] 5.1.4 Bonus: lifecycle (solve-before-factorize throws
        `std::logic_error` on the complex specialisation, matches
        real contract).
      - [x] 5.1.5 Bonus: `make_default_solver_t<Complex>` factory
        returns `Backend::Pulsim` vs `Backend::Eigen` with
        agreement within `1e-10`.
- [x] 5.2 Integration test through `mna_sweep` —
      `core/tests/analysis/test_mna_sweep.cpp` (2 test cases,
      6 assertions, all green):
      - [x] 5.2.1 RC low-pass tank: Bode magnitude within `0.1 dB`
        and phase within `1°` of analytic `1/(1 + jωRC)` across
        50 log-spaced frequencies from 1 Hz to 1 MHz. ✓
      - [x] 5.2.2 Series RLC: peak frequency within `1.5 %` of
        `1/(2π√(LC))` (relaxed from `0.5 %` to absorb the 401-bin
        grid quantisation; analytic theory gives <1 % deviation
        from ω₀ for Q≈5) and `|H(peak)|` within `5 %` of Q. ✓
      - 5.2.3 Buck SISO open-loop bit-identical Bode vs the
        v1.3.0 Eigen-backed run — DEFERRED to the v1.4.0 PR
        validation step (requires the legacy Eigen-path
        ground-truth dataset capture from the showcase suite,
        which lives in `artigos/02_tpel_methods/benchmarks/`).

## 6. AC sweep benchmark capture + paper artefacts

- [x] 6.1 `core/tests/benchmarks/test_bench_ac_sweep.cpp` —
      2-backend AC-sweep microbench. Synthetic MNA-shaped fixture
      (tri-diagonal bandwidth-1 + ω-dependent diagonal capacitance)
      across `n ∈ {8, 16, 32, 64, 128}`. 100 log-spaced
      frequencies from 1 Hz to 1 MHz per row. Drives the
      `analyze + factorize + solve` lifecycle on each frequency
      for both `Backend::Eigen` and `Backend::Pulsim`. Includes
      a parity gate (1e-10) between the two backends at the mid
      frequency.
      `(The realistic per-converter buck/NPC/MMC bench is)`
      `(deferred to add-ac-sweep-per-converter-bench; this v1.4.0)`
      `(bench characterises the solver kernel, not the stamping)`
      `(pipeline.)`
- [x] 6.2 Captured CSV at
      `artigos/02_tpel_methods/benchmarks/results/ac_sweep_microbench.csv`
      on **macOS 26.5 / Apple Silicon / AppleClang 17.0.0 /
      Release -O3 -DNDEBUG**. Numbers (µs/freq, Pulsim ÷ Eigen):
      n=8 → 0.61× (Pulsim wins), n=16 → 0.94×, n=32 → 1.04×,
      n=64 → 1.17×, n=128 → 1.98× (Eigen wins on the large end).
      Parity verified at every n: Δ ≤ 4.34×10⁻²¹.
- [x] 6.3 `artigos/02_tpel_methods/benchmarks/AC_SWEEP_RESULTS.md`
      written — parallel structure to RANK1_RESULTS.md.
      Captures the headline table, 3 interpretation paragraphs
      ("rough parity at SMPS sizes; Eigen pulls ahead at n ≥ 64
      due to asymptotic constants; v1.4.0 reachability-based fast
      path would close the gap"), and the 4-bullet limitations
      section (synthetic fixture, no symbolic reuse, n caps at
      128, single-threaded). The v1.4.0 win is correctly framed as
      a software-supply-chain argument, not a perf one — exactly
      as called out by §6 of the proposal.

## 7. Update spec deltas + finalize

- [ ] 7.1 Update `specs/pulsim-sparse-lu/spec.md` in `openspec/specs/`
      (post-archive of `replace-klu-with-pulsim-sparse-lu`) so the
      Complex Scalar Specialization requirement lands.
- [ ] 7.2 Update `specs/ac-analysis/spec.md` so the new
      "in-house complex solver" requirement lands.
- [x] 7.3 `openspec validate add-pulsim-complex-sparse-lu --strict`
      — passes (`Change 'add-pulsim-complex-sparse-lu' is valid`).
- [ ] 7.4 Open PR `feat/pulsim-complex-sparse-lu` → main.
- [x] 7.5 v1.4.0 release: version bumped to 1.4.0 in
      `pyproject.toml`, `python/pulsim/__init__.py`, `CITATION.cff`.
      Full CHANGELOG entry added covering: AC-sweep migration to
      in-house complex solver, ComplexSparseMatrix RowMajor →
      ColMajor switch, `Eigen::SparseLU<Complex>` no longer
      compiled into the production path (`Backend::Eigen` remains
      explicitly as the paper baseline), 5 new complex unit tests
      + 2 integration tests + 1 AC-sweep microbench, regression
      485/485 C++ tests pass.
- [ ] 7.6 Post-merge: archive the change under
      `openspec/changes/archive/YYYY-MM-DD-add-pulsim-complex-sparse-lu/`.

## Out of scope (future proposals)

- `Scalar = float` / `Scalar = std::complex<float>` (no current
  call site)
- BTF block-triangular decomposition (deferred from
  `replace-klu-with-pulsim-sparse-lu`; reusable for both real and
  complex when added)
- Per-frequency factorisation reuse via sliding-solver / rank-1
  pattern across AC sweep points (the symbolic pattern is
  identical across frequencies — only `j·ω` changes — so the
  amortised-symbolic win from V8 PWL state-space cache is reusable
  here; a follow-up `add-ac-sweep-symbolic-reuse` proposal would
  wire it in)
