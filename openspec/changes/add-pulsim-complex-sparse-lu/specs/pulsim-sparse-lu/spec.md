## ADDED Requirements

### Requirement: Complex Scalar Specialization

The `PulsimSparseLuSolver` class SHALL be templated on a `Scalar`
parameter that defaults to `Real` (= `double`) and SHALL be
explicitly instantiable for `std::complex<Real>`. The full
DirectSolver lifecycle — `analyze`, `factorize`, `solve`,
`partial_refactor`, `numeric_singular`, `supports_partial_refactor`
— SHALL operate correctly on complex matrices using the same
Gilbert-Peierls left-looking algorithm, RCM column ordering,
elimination tree, and path-based partial-refactor structures used
for the real-scalar case.

The pivot-threshold check SHALL use `std::abs(x[i])` (the natural
complex magnitude `sqrt(re² + im²)`) when `Scalar == std::complex<Real>`,
consistent with Demmel 1997 §3.4 / LAPACK ZGETRF threshold
pivoting conventions.

A convenience typedef
`using PulsimComplexSparseLuSolver = PulsimSparseLuSolver<std::complex<Real>>;`
SHALL be exposed at the public header level for AC-sweep call sites.

#### Scenario: Complex factorize-solve roundtrip
- **GIVEN** a Hermitian-positive-definite complex 3×3 matrix `M`
  and a complex right-hand side `b`
- **WHEN** `PulsimComplexSparseLuSolver` runs
  `analyze(M)` → `factorize(M)` → `solve(b, x)`
- **THEN** `||M·x - b||_∞ ≤ 1e-12` over both real and imaginary
  parts
- **AND** `numeric_singular()` returns `false`

#### Scenario: Asymmetric complex MNA factorise identity
- **GIVEN** an asymmetric complex matrix derived from the buck-like
  8×8 fixture as `M(jω) = j·ω·E + J` with `ω = 2π·1000 Hz`
  (`E` and `J` real-valued matrices populated by the existing
  test fixture)
- **WHEN** `analyze(M) + factorize(M)` runs
- **THEN** the identity `(L + I) · U == P_row · M · P_col` holds
  element-wise within `1e-12` complex magnitude tolerance
- **AND** the row + column permutations are valid bijections
  on `[0, n)`

#### Scenario: Real-scalar backward compatibility
- **GIVEN** existing code that uses the bare
  `PulsimSparseLuSolver` class name without a template argument
- **WHEN** the code compiles under the post-change header
- **THEN** the default `Scalar = Real` instantiation is selected
  and the call site behaves identically to the v1.3.0
  pre-template implementation
- **AND** no caller-facing API change is required

#### Scenario: Complex partial_refactor parity vs fresh-factorise
- **GIVEN** a complex MNA matrix `M(jω₁)` already factorised,
  then perturbed at a single column to form `M(jω₂)` (e.g.
  a single source-magnitude change)
- **WHEN** `partial_refactor(M(jω₂), {changed_col})` runs followed
  by `solve(b, x_partial)`
- **AND** a fresh `analyze + factorize` of `M(jω₂)` followed by
  `solve(b, x_fresh)` produces a reference
- **THEN** `||x_partial - x_fresh||_∞ ≤ 1e-10` (complex tolerance)
- **AND** the path-based update incurs no full re-factorisation
  (`rank1_hits == 1`, `full_refactor_hits == 0`)

### Requirement: No Third-Party LU in Production Paths

After this capability lands, the Pulsim kernel SHALL contain zero
calls to `Eigen::SparseLU::factorize` (or any equivalent third-
party LU factorisation entry point) in production code paths.
References to `Eigen::SparseLU` SHALL be confined to:

1. The `SparseLuSolver` class (selectable via `Backend::Eigen`),
   intentionally retained as the benchmark baseline for the IEEE
   TPEL methods paper's microbenchmark tables.
2. Test helpers (e.g. `eigen_reference_fill`) used for fill-pattern
   sanity comparison.
3. Microbenchmark fixtures (`test_bench_pwl_rank1.cpp`,
   `test_bench_ac_sweep.cpp`) that exercise `Backend::Eigen` as
   the comparison column.

Production code paths — the PWL state-space cache, the MNA AC
sweep, the DC operating-point solver — SHALL all use
`PulsimSparseLuSolver` (real or complex Scalar) by default.

#### Scenario: Repo grep confirms zero production Eigen-LU calls
- **WHEN** `rg "Eigen::SparseLU\b" core/include/pulsim core/src`
  is run from the repo root, excluding `core/include/pulsim/sparse/solver.hpp`
  (the `Backend::Eigen` fallback), `core/tests/`, and
  `core/include/pulsim/analysis/mna_sweep.hpp` comments
- **THEN** zero hits are produced
- **AND** the kernel build links no third-party LU runtime

#### Scenario: AC sweep call site uses in-house solver
- **GIVEN** `core/include/pulsim/analysis/mna_sweep.hpp`'s
  per-frequency factorisation block
- **WHEN** the source is grepped for the solver type
- **THEN** `PulsimComplexSparseLuSolver` appears as the
  declaration type
- **AND** `Eigen::SparseLU<` does NOT appear in any
  factorisation-call line
