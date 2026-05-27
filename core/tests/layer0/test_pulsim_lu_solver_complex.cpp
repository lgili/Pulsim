// =============================================================================
// Layer 0 — PulsimSparseLuSolverT<std::complex<Real>> coverage
// =============================================================================
//
// `openspec/changes/add-pulsim-complex-sparse-lu` Section 5.1 tests.
//
// The real-scalar specialisation lives in `test_pulsim_lu_solver.cpp` and
// passes 100% of its identity / pivot / refactor scenarios as of v1.3.0.
// This file mirrors three of those scenarios on the complex specialisation
// so the v1.4.0 templatisation is exercised end-to-end:
//
//   5.1.1  Hermitian positive-definite 3×3:
//          analyze succeeds, factorize succeeds, the LU identity
//                  `(L+I) · U == P_row · M · P_col`
//          holds with `max |·|` ≤ 1e-12 in the complex magnitude.
//
//   5.1.2  Asymmetric MNA 8×8 = J_real + j·ω·E_real (the v1.3.0
//          buck-like fixture plus a single capacitor pair on nodes
//          2-3 that supplies the imaginary part). Same identity,
//          same tolerance. Forces partial pivoting at the voltage-
//          source row.
//
//   5.1.3  partial_refactor on the asymmetric fixture with a single
//          column perturbation (varying the source magnitude at one
//          frequency): the post-refactor solve matches a fresh
//          factorise within 1e-10.
//
// Tolerances chosen to mirror the real-scalar tests' bounds (1e-12 for
// the direct LU identity, 1e-10 for the partial-refactor parity).
// Complex arithmetic introduces roughly 2× the round-off vs real, but
// the matrices here are small enough that 1e-12 still holds.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/numeric/types.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/sparse/pulsim_lu_solver.hpp"
#include "pulsim/sparse/solver.hpp"

#include <Eigen/Dense>

#include <array>
#include <complex>
#include <span>
#include <vector>

using namespace pulsim;
using namespace pulsim::sparse;

using Complex          = std::complex<Real>;
using ComplexMatrix    = MatrixT<Complex>;
using ComplexVector    = VectorT<Complex>;
using ComplexTriplet   = TripletT<Complex>;

namespace {

// -----------------------------------------------------------------------------
// Fixture builders
// -----------------------------------------------------------------------------

/// Hermitian positive-definite 3×3 complex matrix. Real diagonal 4,
/// off-diagonal (-1 - 0.2 j) on (0,1)/(1,0) and (-1 + 0.3 j) /
/// (-1 - 0.3 j) on (1,2)/(2,1) to keep the matrix Hermitian (M[i,j] =
/// conj(M[j,i])). Diagonally dominant — no need for partial pivoting,
/// so this isolates "does the template compile + run" from "does
/// pivoting still work".
ComplexMatrix make_hermitian_pd_3x3() {
    ComplexMatrix M(3, 3);
    std::vector<ComplexTriplet> t = {
        {0, 0, Complex{4.0, 0.0}}, {0, 1, Complex{-1.0, -0.2}},
        {1, 0, Complex{-1.0,  0.2}}, {1, 1, Complex{4.0, 0.0}}, {1, 2, Complex{-1.0,  0.3}},
        {2, 1, Complex{-1.0, -0.3}}, {2, 2, Complex{4.0, 0.0}},
    };
    M.setFromTriplets(t.begin(), t.end());
    M.makeCompressed();
    return M;
}

/// Asymmetric MNA-style 8×8 with imaginary contributions on the
/// capacitor diagonal. Pattern matches `make_buck_like_8x8()` from
/// `test_pulsim_lu_solver.cpp` plus the `j·ω·C` term that an AC sweep
/// would inject for a 1 nF cap between nodes 2-3 at f = 1 MHz
/// (ω ≈ 6.28e6, j·ω·C ≈ +6.28e-3 j). The added entries land on the
/// existing pattern slots so the symbolic structure of the real and
/// complex matrices is identical — exactly the case AC sweep produces.
ComplexMatrix make_complex_mna_8x8() {
    constexpr Index N = 8;
    ComplexMatrix M(N, N);
    std::vector<ComplexTriplet> t;

    // Anchor node 0
    t.emplace_back(0, 0, Complex{1.0e6, 0.0});

    // R1 between nodes 1-2 (real-only)
    t.emplace_back(1, 1, Complex{1.0, 0.0});
    t.emplace_back(1, 2, Complex{-1.0, 0.0});
    t.emplace_back(2, 1, Complex{-1.0, 0.0});
    t.emplace_back(2, 2, Complex{1.0, 0.0});

    // R2 + C between nodes 2-3:
    //   R2 = 2 Ω → conductance 0.5 contributes to diag(2)/(3) and off-diag (-0.5)
    //   C  = 1 nF at f = 1 MHz → j·ω·C ≈ +6.283e-3 j contributes the SAME pattern
    const Complex y_RC{0.5, 6.283e-3};
    t.emplace_back(2, 2, y_RC);
    t.emplace_back(2, 3, -y_RC);
    t.emplace_back(3, 2, -y_RC);
    t.emplace_back(3, 3, y_RC);

    // Voltage source: asymmetric contributions on row/col 7 ↔ row/col 0,1
    t.emplace_back(1, 7, Complex{1.0, 0.0});
    t.emplace_back(0, 7, Complex{-1.0, 0.0});
    t.emplace_back(7, 1, Complex{1.0, 0.0});
    t.emplace_back(7, 0, Complex{-1.0, 0.0});

    // Anchored nodes 4, 5, 6 (small ground resistors)
    for (Index i = 4; i <= 6; ++i) {
        t.emplace_back(i, i, Complex{1.0, 0.0});
    }

    M.setFromTriplets(t.begin(), t.end());
    M.makeCompressed();
    return M;
}

// -----------------------------------------------------------------------------
// Reusable helpers (mirror the real-scalar test file)
// -----------------------------------------------------------------------------

/// Apply row + column permutations to M, returning `P_row · M · P_col`.
ComplexMatrix permuted_matrix(const ComplexMatrix& M,
                              std::span<const Index> Prow,
                              std::span<const Index> Pcol) {
    const Index n = static_cast<Index>(M.rows());
    std::vector<Index> Pinv_row(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i) {
        Pinv_row[static_cast<std::size_t>(Prow[i])] = i;
    }
    ComplexMatrix Mp(n, n);
    std::vector<ComplexTriplet> trips;
    const Index*   Ap = M.outerIndexPtr();
    const Index*   Ai = M.innerIndexPtr();
    const Complex* Ax = M.valuePtr();
    for (Index k = 0; k < n; ++k) {
        const Index orig_col = Pcol[k];
        for (Index p = Ap[orig_col]; p < Ap[orig_col + 1]; ++p) {
            const Index orig_row = Ai[p];
            trips.emplace_back(Pinv_row[static_cast<std::size_t>(orig_row)],
                                k, Ax[p]);
        }
    }
    Mp.setFromTriplets(trips.begin(), trips.end());
    Mp.makeCompressed();
    return Mp;
}

/// `max | (L+I)·U - P_row·M·P_col |` element-wise (using the complex
/// magnitude). The real-scalar test takes `cwiseAbs().maxCoeff()` on the
/// dense representation; for complex the equivalent is
/// `cwiseAbs()` (= |re² + im²|^½) → `maxCoeff()`. We need to convert to
/// a real magnitude matrix because Eigen's `maxCoeff()` on a complex
/// dense matrix is not well-ordered.
Real lu_identity_max_error_complex(
        const PulsimSparseLuSolverT<Complex>& solver,
        const ComplexMatrix& M) {
    ComplexMatrix L = solver.extract_L_matrix();
    ComplexMatrix U = solver.extract_U_matrix();
    ComplexMatrix I(static_cast<Index>(solver.n()),
                    static_cast<Index>(solver.n()));
    I.setIdentity();
    ComplexMatrix LU     = (L + I) * U;
    ComplexMatrix M_perm = permuted_matrix(M, solver.row_permutation(),
                                            solver.column_permutation());

    Eigen::MatrixXcd LU_dense(LU);
    Eigen::MatrixXcd Mp_dense(M_perm);
    Eigen::MatrixXcd diff = LU_dense - Mp_dense;
    // .cwiseAbs() on a complex matrix returns a real magnitude matrix
    // (Eigen::MatrixXd) on which maxCoeff() is well-defined.
    return diff.cwiseAbs().maxCoeff();
}

/// `max | x_a - x_b |` element-wise complex magnitude.
Real complex_vec_max_diff(const ComplexVector& a, const ComplexVector& b) {
    REQUIRE(a.size() == b.size());
    Real err = 0;
    for (Index i = 0; i < a.size(); ++i) {
        const Real e = std::abs(a[i] - b[i]);
        err = std::max(err, e);
    }
    return err;
}

}  // namespace

// -----------------------------------------------------------------------------
// 5.1.1 — Hermitian PD 3×3
// -----------------------------------------------------------------------------
TEST_CASE("PulsimSparseLuSolverT<Complex>: factorize identity on Hermitian PD 3×3",
          "[v2][layer0][sparse][pulsim_lu][complex][factorize]") {
    ComplexMatrix M = make_hermitian_pd_3x3();
    PulsimComplexSparseLuSolver solver;
    REQUIRE(solver.analyze(M));
    REQUIRE(solver.factorize(M));
    REQUIRE(solver.is_factorized());
    REQUIRE_FALSE(solver.numeric_singular());

    const Real err = lu_identity_max_error_complex(solver, M);
    INFO("max |(L+I)·U - P_row·M·P_col| (complex) = " << err);
    REQUIRE(err < 1e-12);

    // Sanity-check a solve against a known right-hand side. b = M · x_true
    // for x_true = (1+j, 2-j, -1+0.5j), then solve(b) must recover x_true.
    ComplexVector x_true(3);
    x_true << Complex{1.0, 1.0}, Complex{2.0, -1.0}, Complex{-1.0, 0.5};
    ComplexVector b = M * x_true;
    ComplexVector x_hat;
    solver.solve(b, x_hat);
    REQUIRE(complex_vec_max_diff(x_hat, x_true) < 1e-12);
}

// -----------------------------------------------------------------------------
// 5.1.2 — Asymmetric 8×8 complex MNA (forces partial pivoting at the
// voltage-source row whose diagonal is structurally zero)
// -----------------------------------------------------------------------------
TEST_CASE("PulsimSparseLuSolverT<Complex>: factorize identity on asymmetric 8×8 MNA",
          "[v2][layer0][sparse][pulsim_lu][complex][factorize]") {
    ComplexMatrix M = make_complex_mna_8x8();
    PulsimComplexSparseLuSolver solver;
    REQUIRE(solver.analyze(M));
    REQUIRE(solver.factorize(M));
    REQUIRE(solver.is_factorized());
    REQUIRE_FALSE(solver.numeric_singular());

    const Real err = lu_identity_max_error_complex(solver, M);
    INFO("max |(L+I)·U - P_row·M·P_col| (8×8 complex) = " << err);
    REQUIRE(err < 1e-12);

    // Solve parity: pick an arbitrary complex RHS and confirm M · x_hat = b
    // within tolerance. We use 1e-10 here (not 1e-12) because the dense
    // residual product also accumulates round-off.
    ComplexVector b(8);
    for (Index i = 0; i < 8; ++i) {
        b[i] = Complex{static_cast<Real>(i + 1), static_cast<Real>(-i)};
    }
    ComplexVector x_hat;
    solver.solve(b, x_hat);

    Eigen::MatrixXcd M_dense(M);
    ComplexVector residual = M_dense * x_hat - b;
    Real rmax = 0;
    for (Index i = 0; i < residual.size(); ++i) {
        rmax = std::max(rmax, std::abs(residual[i]));
    }
    INFO("max |M · x_hat - b| = " << rmax);
    REQUIRE(rmax < 1e-10);
}

// -----------------------------------------------------------------------------
// 5.1.3 — partial_refactor on a single column perturbation
// -----------------------------------------------------------------------------
TEST_CASE("PulsimSparseLuSolverT<Complex>: partial_refactor matches fresh factorise",
          "[v2][layer0][sparse][pulsim_lu][complex][partial_refactor]") {
    ComplexMatrix M1 = make_complex_mna_8x8();

    PulsimComplexSparseLuSolver solver;
    REQUIRE(solver.analyze(M1));
    REQUIRE(solver.factorize(M1));

    // Reference solve on M1
    ComplexVector b(8);
    for (Index i = 0; i < 8; ++i) {
        b[i] = Complex{static_cast<Real>(i + 1), static_cast<Real>(-i)};
    }
    ComplexVector x_ref_M1;
    solver.solve(b, x_ref_M1);

    // Perturb column 3 (the R2/C output node): scale its imaginary
    // contributions by 1.1 (mimicking ω increasing from 1 MHz to 1.1 MHz
    // in an AC sweep). The structural pattern is unchanged — only the
    // numerical values at the pre-existing slots shift, which is exactly
    // the use-case partial_refactor was designed for.
    ComplexMatrix M2 = M1;
    {
        Complex* vals = M2.valuePtr();
        const Index* outer = M2.outerIndexPtr();
        for (Index p = outer[3]; p < outer[4]; ++p) {
            // Only touch the imaginary part — the resistor's real part
            // stays put. This isolates the change to numeric-only.
            vals[p] = Complex{vals[p].real(), vals[p].imag() * 1.1};
        }
    }

    REQUIRE(solver.supports_partial_refactor());
    std::array<Index, 1> changed_cols = {3};
    REQUIRE(solver.partial_refactor(M2, std::span<const Index>(changed_cols)));

    ComplexVector x_partial;
    solver.solve(b, x_partial);

    // Reference: fresh factorise of M2 from scratch
    PulsimComplexSparseLuSolver fresh;
    REQUIRE(fresh.analyze(M2));
    REQUIRE(fresh.factorize(M2));
    ComplexVector x_fresh;
    fresh.solve(b, x_fresh);

    const Real err = complex_vec_max_diff(x_partial, x_fresh);
    INFO("max |x_partial - x_fresh| = " << err);
    REQUIRE(err < 1e-10);

    // x_partial MUST also differ from x_ref_M1 by a non-trivial amount —
    // otherwise the perturbation didn't actually take effect.
    const Real diff_from_M1 = complex_vec_max_diff(x_partial, x_ref_M1);
    INFO("|x_partial - x_ref_M1| = " << diff_from_M1
         << " (must be > 0)");
    REQUIRE(diff_from_M1 > 1e-12);
}

// -----------------------------------------------------------------------------
// Lifecycle sanity: calling solve before factorize on the complex
// specialisation throws (matches the real specialisation's contract).
// -----------------------------------------------------------------------------
TEST_CASE("PulsimSparseLuSolverT<Complex>: solve before factorize throws",
          "[v2][layer0][sparse][pulsim_lu][complex][lifecycle]") {
    ComplexMatrix M = make_hermitian_pd_3x3();
    ComplexVector b(3);
    b << Complex{1, 0}, Complex{1, 0}, Complex{1, 0};
    ComplexVector x(3);
    PulsimComplexSparseLuSolver solver;
    REQUIRE(solver.analyze(M));
    REQUIRE_THROWS_AS(solver.solve(b, x), std::logic_error);
}

// -----------------------------------------------------------------------------
// Factory test: make_default_solver_t<Complex> dispatches correctly.
// -----------------------------------------------------------------------------
TEST_CASE("make_default_solver_t<Complex>: factory returns expected backend",
          "[v2][layer0][sparse][pulsim_lu][complex][factory]") {
    // Backend::Pulsim (and Backend::Auto) should produce a working
    // complex Pulsim solver; Backend::Eigen should produce a working
    // SparseLuSolverT<Complex>. Both must accept the same matrix and
    // produce solves that agree within 1e-10.
    ComplexMatrix M = make_hermitian_pd_3x3();
    ComplexVector b(3);
    b << Complex{1, 0}, Complex{1, 0}, Complex{1, 0};

    auto pulsim = make_default_solver_t<Complex>(3, Backend::Pulsim);
    REQUIRE(pulsim->analyze(M));
    REQUIRE(pulsim->factorize(M));
    ComplexVector x_pulsim;
    pulsim->solve(b, x_pulsim);

    auto eigen_baseline = make_default_solver_t<Complex>(3, Backend::Eigen);
    REQUIRE(eigen_baseline->analyze(M));
    REQUIRE(eigen_baseline->factorize(M));
    ComplexVector x_eigen;
    eigen_baseline->solve(b, x_eigen);

    REQUIRE(complex_vec_max_diff(x_pulsim, x_eigen) < 1e-10);
}
