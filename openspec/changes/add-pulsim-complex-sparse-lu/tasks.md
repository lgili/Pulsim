## 1. Template `PulsimSparseLuSolver` on `Scalar`

- [ ] 1.1 Rename `class PulsimSparseLuSolver` to
      `template <typename Scalar> class PulsimSparseLuSolver` in
      `core/include/pulsim/sparse/pulsim_lu_solver.hpp`. Default
      `Scalar = Real` via a separate concrete `using` alias at
      the bottom of the header for backward compat:
      ```cpp
      using PulsimRealSparseLuSolver  = PulsimSparseLuSolver<Real>;
      using PulsimComplexSparseLuSolver
          = PulsimSparseLuSolver<std::complex<Real>>;
      ```
- [ ] 1.2 Replace `Matrix` (which is `Eigen::SparseMatrix<Real,
      ColMajor, Index>`) with a generic
      `MatrixT = Eigen::SparseMatrix<Scalar, ColMajor, Index>`
      typedef inside the class template. Same for the value-array
      members (`l_values_`, `u_values_`).
- [ ] 1.3 Replace `Vector = Eigen::Matrix<Real, Dynamic, 1>` with
      `VectorT = Eigen::Matrix<Scalar, Dynamic, 1>` in `solve()` and
      `partial_refactor()` workspace types.
- [ ] 1.4 Update DirectSolver base class `core/include/pulsim/sparse/solver.hpp`
      to a `template <typename Scalar>` form. Existing real-scalar
      consumers untouched (the default-template-arg keeps them
      source-compatible).
- [ ] 1.5 Verify zero changes to the elimination tree, RCM
      ordering, symbolic pattern computation, and path
      computation — those are structural and Scalar-agnostic.
      The only touch points are: (a) the pivot-magnitude check
      (`|x[i]|` becomes `std::abs(x[i])` which already works for
      both Real and complex), (b) the L-update inner loop's
      multiply/subtract (already generic via overloaded `*` / `-=`).

## 2. Complex specialisation of the pivot check

- [ ] 2.1 Confirm `std::abs(std::complex<Real>)` returns
      `std::sqrt(re² + im²)` — this is the natural complex
      magnitude and the correct threshold for partial pivoting
      per Bunch 1971 + Demmel 1997 §3.4. No behavioural change vs
      the real case beyond the magnitude metric.
- [ ] 2.2 Adjust the `PIVOT_THRESH = 1e-3` default to remain valid
      under complex magnitudes (it does — the threshold is
      dimensionless, applied to relative magnitudes only).
- [ ] 2.3 Add a `numeric_singular_threshold()` accessor that lets
      AC sweep tighten the threshold per frequency if needed
      (deferred — only add if a sweep regresses; complex MNA is
      usually well-conditioned at production frequencies).

## 3. Update factory + builder for the template

- [ ] 3.1 Make `make_default_solver<Scalar>(n, hint)` a function
      template. Default `Scalar = Real` keeps every existing call
      site source-compatible.
- [ ] 3.2 `Backend::Pulsim` (and `Backend::Auto` → Pulsim) returns
      `std::make_unique<PulsimSparseLuSolver<Scalar>>()`.
      `Backend::Eigen` returns `std::make_unique<SparseLuSolver<Scalar>>()`.
- [ ] 3.3 `SparseLuSolver` (the Eigen fallback) also becomes a
      template wrapping `Eigen::SparseLU<MatrixT, COLAMDOrdering<Index>>`.
      Kept intentionally — `Backend::Eigen` remains the benchmark
      baseline for the TPEL §VI.B AC sweep table.

## 4. Switch `mna_sweep.hpp` to the in-house complex solver

- [ ] 4.1 In `core/include/pulsim/analysis/mna_sweep.hpp`, replace
      the inline `Eigen::SparseLU<ComplexSparseMatrix,
      Eigen::COLAMDOrdering<Index>> solver;` instantiation with
      a `PulsimComplexSparseLuSolver solver;`.
- [ ] 4.2 Replace `solver.analyzePattern(M)` + `solver.factorize(M)`
      with `solver.analyze(M)` + `solver.factorize(M)` (our API
      uses the same names as Eigen for the call site; only the
      underlying type changes).
- [ ] 4.3 Replace `solver.info() != Eigen::Success` with
      `solver.numeric_singular()` for the failure check; preserve
      the existing `throw std::runtime_error(...)` message but
      mention the Pulsim solver in the diagnostic text.
- [ ] 4.4 Switch the solve call from
      `X = solver.solve(B)` to the in-place `solver.solve(B, X)`
      signature.
- [ ] 4.5 Remove the `#include <Eigen/SparseLU>` from
      `mna_sweep.hpp` (no longer needed — `Eigen/Sparse` stays for
      `ComplexSparseMatrix` typedef).

## 5. Test coverage — synthetic complex fixtures

- [ ] 5.1 New file `core/tests/layer0/test_pulsim_lu_solver_complex.cpp`:
      - 5.1.1 SPD complex 3×3 (Hermitian positive-definite):
        analyze succeeds, factorize identity `(L+I)·U == P_row·M·P_col`
        within `1e-12 + 1e-12·i` element-wise complex tolerance. ✓
      - 5.1.2 Asymmetric complex MNA 8×8 (buck-like fixture
        translated to complex via `j·ω·E + J` with ω = 2π·1000 Hz,
        E and J real): same identity, same tolerance. ✓
      - 5.1.3 Partial-refactor on the asymmetric fixture with a
        single column perturbation (varying the source magnitude
        at a single frequency): solve parity vs fresh-factorise
        within `1e-10`. ✓
- [ ] 5.2 Integration test through `mna_sweep`:
      - 5.2.1 RC low-pass tank: Bode magnitude within `0.1 dB` and
        phase within `1°` of the analytic `1/(1 + jωRC)` form
        across 100 frequencies from 1 Hz to 1 MHz. ✓
      - 5.2.2 RLC bandpass: peak frequency within `0.5 %` of
        `1/(2π√(LC))` and Q within `5 %` of `(1/R)·√(L/C)`. ✓
      - 5.2.3 Buck SISO open-loop: bit-identical (within solver
        tolerance `1e-10`) Bode plot vs the v1.3.0 Eigen-backed
        run on the same fixture. Confirms the in-house complex
        solver produces equivalent answers.

## 6. AC sweep benchmark capture + paper artefacts

- [ ] 6.1 New file `core/tests/benchmarks/test_bench_ac_sweep.cpp`:
      mirror the 3-backend pattern from the real-scalar microbench
      — same fixture (a representative converter MNA: buck + NPC
      + MMC arm), sweep 100 frequencies decade-log from 1 Hz to
      1 MHz, time per-frequency factorize + solve for both
      `Backend::Pulsim` (in-house complex) and `Backend::Eigen`
      (`Eigen::SparseLU<complex>` baseline).
- [ ] 6.2 Run the benchmark on macOS / Apple Silicon. Capture CSV
      under `artigos/02_tpel_methods/benchmarks/results/ac_sweep_microbench.csv`.
- [ ] 6.3 Write `artigos/02_tpel_methods/benchmarks/AC_SWEEP_RESULTS.md`
      with the 2-backend comparison (parallel structure to
      `RANK1_RESULTS.md`): captured numbers + interpretation +
      honest limitations (per-frequency cost, condition-number
      dependence, single-threaded). The expected result is **rough
      parity** (within 1.5× either way) — the complex sparse LU
      isn't a place where Pulsim has a structural algorithmic edge
      vs Eigen; the win is "no third-party LU in production",
      which is a software-supply-chain argument, not a perf one.

## 7. Update spec deltas + finalize

- [ ] 7.1 Update `specs/pulsim-sparse-lu/spec.md` in `openspec/specs/`
      (post-archive of `replace-klu-with-pulsim-sparse-lu`) so the
      Complex Scalar Specialization requirement lands.
- [ ] 7.2 Update `specs/ac-analysis/spec.md` so the new
      "in-house complex solver" requirement lands.
- [ ] 7.3 `openspec validate add-pulsim-complex-sparse-lu --strict`.
- [ ] 7.4 Open PR `feat/pulsim-complex-sparse-lu` → main.
- [ ] 7.5 v1.4.0 release: bump version in `pyproject.toml`,
      `python/pulsim/__init__.py`, `CITATION.cff`. CHANGELOG entry
      explaining the AC sweep migration + that Eigen LU is now
      strictly a benchmark baseline.
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
