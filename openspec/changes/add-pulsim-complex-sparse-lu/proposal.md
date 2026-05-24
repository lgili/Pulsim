## Why

After `replace-klu-with-pulsim-sparse-lu` (v1.3.0) lands, the Pulsim
C++ kernel still uses `Eigen::SparseLU<std::complex<Real>>` in **one
remaining place**: `core/include/pulsim/analysis/mna_sweep.hpp:230`,
the AC small-signal sweep path. That's the last spot where a
third-party library actively factorises matrices on Pulsim's behalf;
everything else uses `pulsim::sparse::PulsimSparseLuSolver`.

For the IEEE TPEL methods paper, the "all algorithms are ours"
narrative requires zero third-party LU factorisation in production
code paths. The project owner's directive (2026-05-24) is explicit:
**vendors stay in the codebase only as benchmark baselines for the
paper, not as production fallbacks**. The remaining
`Eigen::SparseLU<complex>` call violates that contract.

Adding complex-scalar support to `PulsimSparseLuSolver` closes the
gap. The Gilbert-Peierls algorithm, partial pivoting (now using
`|x| = sqrt(re² + im²)` for the threshold check), elimination tree,
and path-based partial refactor all generalise to `std::complex<Real>`
with no algorithmic changes — only the scalar type changes.

## What Changes

- **NEW**: Template `PulsimSparseLuSolver` on `Scalar` (defaulting to
  `Real` for backward compat). Add an explicit
  `PulsimComplexSparseLuSolver` typedef
  (`= PulsimSparseLuSolver<std::complex<Real>>`) for the AC sweep
  call site.
- **MODIFIED**: `analysis/mna_sweep.hpp` switches from
  `Eigen::SparseLU<ComplexSparseMatrix, COLAMDOrdering<Index>>` to
  `PulsimComplexSparseLuSolver` for the per-frequency
  factorize + solve.
- **NEW**: New benchmark target
  `core/tests/benchmarks/test_bench_ac_sweep.cpp` capturing
  per-frequency latency on a representative buck + boost + NPC AC
  sweep, comparing the new in-house complex solver against the
  Eigen reference (kept as `Backend::Eigen` for the benchmark, same
  pattern as the real-scalar 3-backend microbench).
- **NEW**: Test coverage for complex factorize + solve + roundtrip
  identity (M·x = b within `1e-10` complex tolerance) on synthetic
  fixtures (Hermitian SPD complex matrix, asymmetric complex MNA
  with a complex-valued source) and on a real MNA sweep fixture
  (RC tank, RLC bandpass, buck SISO open-loop) checked against
  reference Bode magnitude/phase within `0.1 dB / 1°`.
- **MODIFIED**: `make_default_solver(n, hint)` factory acquires a
  `Scalar` template parameter. Backward-compat default is `Real`.
- **REMOVED** (BREAKING at kernel-builder level): zero direct
  references to `Eigen::SparseLU` outside of:
  - `core/include/pulsim/sparse/solver.hpp` (`SparseLuSolver`
    fallback, intentionally kept as `Backend::Eigen` for the
    benchmark baseline)
  - `core/tests/layer0/test_pulsim_lu_solver.cpp`
    (`eigen_reference_fill` — used to compare fill against COLAMD)
  - `core/tests/benchmarks/test_bench_ac_sweep.cpp` (the new AC
    sweep benchmark, exercising `Backend::Eigen` as comparison)
- **NOT changed**: Python API, builder API,
  `pp.simulate(...)` / `pp.run_ac_sweep(...)`. The change is
  transparent to all downstream callers — at most they see
  different telemetry counters under the AC sweep path.

## Impact

- **Affected specs**:
  - `pulsim-sparse-lu` — ADDED requirement (Complex Scalar
    Specialization)
  - `ac-analysis` — ADDED requirement (AC Sweep Uses In-House
    Complex Solver)
- **Affected code**:
  - `core/include/pulsim/sparse/pulsim_lu_solver.hpp` (template
    Scalar, add complex specialisation tests for the pivot check)
  - `core/include/pulsim/sparse/solver.hpp` (factory template)
  - `core/include/pulsim/analysis/mna_sweep.hpp` (switch call site)
  - `core/tests/layer0/test_pulsim_lu_solver_complex.cpp` (NEW —
    complex-scalar coverage)
  - `core/tests/benchmarks/test_bench_ac_sweep.cpp` (NEW)
  - `artigos/02_tpel_methods/benchmarks/AC_SWEEP_RESULTS.md` (NEW —
    per-frequency latency table feeding paper §VI.B)
- **Target release**: v1.4.0 (post-TPEL submission). Best as a
  separate PR after `replace-klu-with-pulsim-sparse-lu` archives.
- **Scope estimate**: ~2-3 weeks. The algorithm is reusable; the
  work is in template-param plumbing + complex-scalar test
  coverage + benchmark capture.

## Out of scope

- BTF block-triangular decomposition for the complex case
  (`replace-klu-with-pulsim-sparse-lu`'s out-of-scope list defers
  BTF for the real case; complex inherits that deferral)
- GPU / multi-threaded complex factorisation
- `Scalar = float` (32-bit real) specialisation — not needed for
  any current call site
- `Scalar = std::complex<float>` — same reason
