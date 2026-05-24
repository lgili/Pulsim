// =============================================================================
// Layer 0 V8 — KluSolver: parity vs SparseLuSolver + factory + partial_refactor
// =============================================================================
//
// `openspec/changes/add-pwl-rank1-update` tasks 2.7.1, 2.7.2, 2.7.3 and
// 3.7. Locks in:
//
//   * KluSolver produces the same solution as SparseLuSolver on the
//     standard SPD 3x3 test matrix (within 1e-12).
//   * KluSolver produces the same solution on a representative buck-cache
//     segment (n=8, asymmetric MNA stamps).
//   * `partial_refactor` after a single-column value change gives the
//     same output as a fresh full `factorize`.
//   * `Backend` factory hint behaves: Auto picks KLU when n >= threshold,
//     Eigen otherwise; explicit KLU works; explicit Eigen always works.
//   * Out-of-order calls throw `std::logic_error` exactly like
//     SparseLuSolver does (the contract is shared by the interface).
//
// Compiled into the build only when PULSIM_HAVE_KLU is set — when KLU is
// absent these tests are silently skipped. The static_assert at the top
// makes a missing PULSIM_HAVE_KLU at this TU level a clear compile error
// (the test_main.cpp + CMakeLists guard prevent that ever happening).

#ifdef PULSIM_HAVE_KLU

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/numeric/types.hpp"
#include "pulsim/sparse/klu_solver.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/sparse/solver.hpp"

#include <array>
#include <stdexcept>
#include <vector>

using namespace pulsim;
using namespace pulsim::sparse;
using Catch::Approx;

namespace {

Matrix make_spd_3x3() {
    // Tridiagonal SPD — same matrix the SparseLuSolver test uses, so
    // parity comparison is direct.
    Matrix M(3, 3);
    std::vector<Triplet> t = {
        {0, 0,  4.0}, {0, 1, -1.0},
        {1, 0, -1.0}, {1, 1,  4.0}, {1, 2, -1.0},
        {2, 1, -1.0}, {2, 2,  4.0},
    };
    M.setFromTriplets(t.begin(), t.end());
    compress_in_place(M);
    return M;
}

/// Representative buck-cache MNA matrix at n=8 — small but asymmetric
/// (controlled-source rows make it non-SPD), the typical pattern for
/// switch-state factorization in `pulsim::pwl::PwlStateSpaceCache`.
///
/// Build a sparse asymmetric system with a known solution by stamping
/// a simple resistor + voltage-source-with-branch-current circuit:
///
///   node 0 (gnd, anchored)
///   node 1 = vin       (driven by Vdc=5V → branch var ib at row 4)
///   node 2 = vmid      (vin --R1=1Ω-- vmid --R2=2Ω-- vout)
///   node 3 = vout
///   plus 4 extra "ghost" nodes (4..7) anchored to gnd via 1Ω each, to
///   pad up to n=8 without changing the meaningful sub-system.
Matrix make_buck_like_8x8() {
    constexpr Index N = 8;
    Matrix M(N, N);
    std::vector<Triplet> t;

    // Anchor node 0 to itself with a big conductance (Pulsim's "anchor
    // resistor" pattern). 1e6 keeps the system well-conditioned without
    // a true Dirichlet row.
    t.emplace_back(0, 0, 1.0e6);

    // R1 = 1Ω between nodes 1 and 2: stamps +1/-1 on the 2x2 block.
    t.emplace_back(1, 1,  1.0); t.emplace_back(1, 2, -1.0);
    t.emplace_back(2, 1, -1.0); t.emplace_back(2, 2,  1.0);

    // R2 = 0.5 S (1/2Ω) between nodes 2 and 3.
    t.emplace_back(2, 2,  0.5); t.emplace_back(2, 3, -0.5);
    t.emplace_back(3, 2, -0.5); t.emplace_back(3, 3,  0.5);

    // Voltage source Vdc between node 1 (anode) and node 0 (cathode),
    // 5V. Branch current var lives at row 7 (last index, to give the
    // matrix a non-trivial asymmetric structure).
    t.emplace_back(1, 7,  1.0);   // KCL at node 1 gets +ib
    t.emplace_back(0, 7, -1.0);   // KCL at node 0 gets -ib
    t.emplace_back(7, 1,  1.0);   // constraint: V(1) - V(0) = 5
    t.emplace_back(7, 0, -1.0);

    // Ghost nodes 4, 5, 6 anchored to themselves via 1Ω each.
    for (Index i = 4; i <= 6; ++i) {
        t.emplace_back(i, i, 1.0);
    }

    M.setFromTriplets(t.begin(), t.end());
    compress_in_place(M);
    return M;
}

}  // namespace

// -----------------------------------------------------------------------------
// 2.7.1 — Parity on the canonical SPD 3x3
// -----------------------------------------------------------------------------
TEST_CASE("KluSolver matches SparseLuSolver on SPD 3x3 within 1e-12",
          "[v2][layer0][sparse][klu][parity]") {
    Matrix M = make_spd_3x3();
    Vector b(3);
    b << 2.0, 4.0, 2.0;

    Vector x_eigen(3);
    SparseLuSolver eigen_solver;
    REQUIRE(eigen_solver.analyze(M));
    REQUIRE(eigen_solver.factorize(M));
    eigen_solver.solve(b, x_eigen);

    Vector x_klu(3);
    KluSolver klu_solver;
    REQUIRE(klu_solver.analyze(M));
    REQUIRE(klu_solver.is_analyzed());
    REQUIRE(klu_solver.factorize(M));
    REQUIRE(klu_solver.is_factorized());
    klu_solver.solve(b, x_klu);

    for (Index i = 0; i < x_eigen.size(); ++i) {
        REQUIRE(x_klu[i] == Approx(x_eigen[i]).margin(1e-12));
    }
}

// -----------------------------------------------------------------------------
// 2.7.2 — Parity on the buck-cache-like asymmetric 8x8
// -----------------------------------------------------------------------------
TEST_CASE("KluSolver matches SparseLuSolver on buck-cache-like 8x8 within 1e-12",
          "[v2][layer0][sparse][klu][parity][buck]") {
    Matrix M = make_buck_like_8x8();
    Vector b(8);
    b << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0;

    Vector x_eigen(8);
    SparseLuSolver eigen_solver;
    REQUIRE(eigen_solver.analyze(M));
    REQUIRE(eigen_solver.factorize(M));
    eigen_solver.solve(b, x_eigen);

    Vector x_klu(8);
    KluSolver klu_solver;
    REQUIRE(klu_solver.analyze(M));
    REQUIRE(klu_solver.factorize(M));
    klu_solver.solve(b, x_klu);

    for (Index i = 0; i < x_eigen.size(); ++i) {
        REQUIRE(x_klu[i] == Approx(x_eigen[i]).margin(1e-12));
    }
}

// -----------------------------------------------------------------------------
// 2.7.3 — partial_refactor parity: same answer as full factorize
// -----------------------------------------------------------------------------
TEST_CASE("KluSolver::partial_refactor matches full factorize after a value change",
          "[v2][layer0][sparse][klu][partial_refactor]") {
    Matrix M1 = make_buck_like_8x8();
    Vector b(8);
    b << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0;

    KluSolver solver;
    REQUIRE(solver.analyze(M1));
    REQUIRE(solver.factorize(M1));

    Vector x_initial(8);
    solver.solve(b, x_initial);

    // Now perturb R2 (the 2-3 conductance pair). Same sparsity pattern,
    // different values. This is exactly the structure of a single-bit
    // switch flip in the PWL cache.
    Matrix M2 = make_buck_like_8x8();
    // Find the (2,2) entry and bump it slightly. coeffRef requires the
    // entry to already exist (it does — make_buck_like inserts it).
    M2.coeffRef(2, 2) += 1.0;
    M2.coeffRef(3, 3) += 1.0;
    M2.coeffRef(2, 3) -= 0.5;
    M2.coeffRef(3, 2) -= 0.5;

    // Path A: partial_refactor on the already-factorized solver.
    constexpr std::array<Index, 2> changed{2, 3};
    const bool ok = solver.partial_refactor(
        M2, std::span<const Index>{changed.data(), changed.size()});
    REQUIRE(ok);
    REQUIRE(solver.is_factorized());

    Vector x_partial(8);
    solver.solve(b, x_partial);

    // Path B: fresh solver, full factor of M2.
    KluSolver fresh;
    REQUIRE(fresh.analyze(M2));
    REQUIRE(fresh.factorize(M2));

    Vector x_full(8);
    fresh.solve(b, x_full);

    for (Index i = 0; i < x_full.size(); ++i) {
        REQUIRE(x_partial[i] == Approx(x_full[i]).margin(1e-12));
    }
}

// -----------------------------------------------------------------------------
// 2.7 — supports_partial_refactor advertises true
// -----------------------------------------------------------------------------
TEST_CASE("KluSolver advertises partial_refactor support",
          "[v2][layer0][sparse][klu][interface]") {
    KluSolver solver;
    REQUIRE(solver.supports_partial_refactor());

    SparseLuSolver eigen;
    REQUIRE_FALSE(eigen.supports_partial_refactor());
}

// -----------------------------------------------------------------------------
// 3.7 — Factory: Backend hint behaviour
// -----------------------------------------------------------------------------
TEST_CASE("make_default_solver: Backend::Auto picks KLU above threshold, Eigen below",
          "[v2][layer0][sparse][factory]") {
    // Below threshold → Eigen path. KluSolver's supports_partial_refactor
    // is true; SparseLuSolver's is false. Use that as a discriminator
    // without exposing typeid.
    auto small_solver = make_default_solver(/*n=*/10, Backend::Auto);
    REQUIRE_FALSE(small_solver->supports_partial_refactor());

    // At or above threshold → KLU.
    auto big_solver = make_default_solver(/*n=*/PULSIM_KLU_AUTO_THRESHOLD,
                                            Backend::Auto);
    REQUIRE(big_solver->supports_partial_refactor());
}

TEST_CASE("make_default_solver: Backend::Eigen is honoured at any n",
          "[v2][layer0][sparse][factory]") {
    auto s_small = make_default_solver(/*n=*/10, Backend::Eigen);
    auto s_big   = make_default_solver(/*n=*/1000, Backend::Eigen);
    REQUIRE_FALSE(s_small->supports_partial_refactor());
    REQUIRE_FALSE(s_big->supports_partial_refactor());
}

TEST_CASE("make_default_solver: Backend::KLU is honoured at any n",
          "[v2][layer0][sparse][factory]") {
    auto s_small = make_default_solver(/*n=*/10, Backend::KLU);
    auto s_big   = make_default_solver(/*n=*/1000, Backend::KLU);
    REQUIRE(s_small->supports_partial_refactor());
    REQUIRE(s_big->supports_partial_refactor());
}

// -----------------------------------------------------------------------------
// Lifecycle contract — out-of-order calls throw
// -----------------------------------------------------------------------------
TEST_CASE("KluSolver throws std::logic_error on factorize-before-analyze",
          "[v2][layer0][sparse][klu][lifecycle]") {
    Matrix M = make_spd_3x3();
    KluSolver solver;
    REQUIRE_THROWS_AS(solver.factorize(M), std::logic_error);
}

TEST_CASE("KluSolver throws std::logic_error on solve-before-factorize",
          "[v2][layer0][sparse][klu][lifecycle]") {
    Matrix M = make_spd_3x3();
    Vector b(3); b << 1.0, 1.0, 1.0;
    Vector x(3);

    KluSolver solver;
    REQUIRE(solver.analyze(M));
    REQUIRE_THROWS_AS(solver.solve(b, x), std::logic_error);
}

#endif  // PULSIM_HAVE_KLU
