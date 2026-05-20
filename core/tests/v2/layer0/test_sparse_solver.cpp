// =============================================================================
// Layer 0 — direct sparse solver (analyze / factorize / solve lifecycle)
// =============================================================================
//
// Locks in the lifecycle contract: out-of-order calls throw, factor
// caching across multiple `factorize` calls works, and the SPD
// reference system solves to high precision.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/sparse/matrix.hpp"
#include "pulsim/v2/sparse/solver.hpp"

#include <stdexcept>
#include <string>
#include <vector>

using namespace pulsim::v2;
using namespace pulsim::v2::sparse;
using Catch::Approx;

namespace {

Matrix make_spd_3x3() {
    // Tridiagonal SPD: [[4,-1,0],[-1,4,-1],[0,-1,4]]
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

}  // namespace

TEST_CASE("SparseLuSolver solves an SPD 3x3 system to high precision",
          "[v2][layer0][sparse][solver]") {
    Matrix M = make_spd_3x3();
    Vector b(3);
    b << 2.0, 4.0, 2.0;

    SparseLuSolver solver;
    REQUIRE(solver.analyze(M));
    REQUIRE(solver.is_analyzed());

    REQUIRE(solver.factorize(M));
    REQUIRE(solver.is_factorized());

    Vector x;
    solver.solve(b, x);

    // Analytic solution computed by Gaussian elimination on the
    // tridiagonal:
    //   x0 = 6/7 ≈ 0.857
    //   x1 = 10/7 ≈ 1.429
    //   x2 = 6/7 ≈ 0.857
    REQUIRE(x.size() == 3);
    REQUIRE(x[0] == Approx(6.0 / 7.0).margin(1e-12));
    REQUIRE(x[1] == Approx(10.0 / 7.0).margin(1e-12));
    REQUIRE(x[2] == Approx(6.0 / 7.0).margin(1e-12));
}

TEST_CASE("SparseLuSolver throws when solve is called before factorize",
          "[v2][layer0][sparse][solver]") {
    SparseLuSolver solver;
    Vector b(3), x;
    b << 1.0, 2.0, 3.0;

    REQUIRE_FALSE(solver.is_factorized());

    try {
        solver.solve(b, x);
        FAIL("solve before factorize did NOT throw");
    } catch (const std::logic_error& e) {
        const std::string what = e.what();
        // The diagnostic must mention 'factorize' so the user knows
        // what step they missed.
        REQUIRE(what.find("factorize") != std::string::npos);
    }
}

TEST_CASE("SparseLuSolver throws when factorize is called before analyze",
          "[v2][layer0][sparse][solver]") {
    Matrix M = make_spd_3x3();
    SparseLuSolver solver;

    REQUIRE_FALSE(solver.is_analyzed());

    try {
        (void)solver.factorize(M);
        FAIL("factorize before analyze did NOT throw");
    } catch (const std::logic_error& e) {
        const std::string what = e.what();
        REQUIRE(what.find("analyze") != std::string::npos);
    }
}

TEST_CASE("SparseLuSolver caches the symbolic factor across reanalyze-free "
          "refactorize calls",
          "[v2][layer0][sparse][solver]") {
    // The whole point of separating analyze from factorize: a Layer 4
    // PWL state-space cache fixes the sparsity pattern per switch
    // combination, then re-factorizes when matrix values drift
    // (e.g. dt changed). analyze MUST run once; factorize MUST run
    // multiple times against the SAME pattern.
    Matrix M1 = make_spd_3x3();

    // Same structure, different numeric values.
    Matrix M2(3, 3);
    std::vector<Triplet> t = {
        {0, 0, 10.0}, {0, 1, -2.0},
        {1, 0, -2.0}, {1, 1, 10.0}, {1, 2, -2.0},
        {2, 1, -2.0}, {2, 2, 10.0},
    };
    M2.setFromTriplets(t.begin(), t.end());
    compress_in_place(M2);

    SparseLuSolver solver;
    REQUIRE(solver.analyze(M1));
    REQUIRE(solver.factorize(M1));

    // Second factorize against M2 (same sparsity, different values).
    // analyze NOT re-called.
    REQUIRE(solver.factorize(M2));

    // Solve uses the M2 factor.
    Vector b(3);
    b << 10.0, 8.0, 10.0;
    Vector x;
    solver.solve(b, x);

    // For M2, the solution to Mx = b can be verified by recomputing
    // Mx and comparing to b.
    Vector residual = M2 * x - b;
    for (Index i = 0; i < 3; ++i) {
        REQUIRE(residual[i] == Approx(Real{0}).margin(1e-10));
    }
}

TEST_CASE("make_default_solver returns a non-null SparseLuSolver",
          "[v2][layer0][sparse][solver]") {
    auto solver = make_default_solver();
    REQUIRE(solver != nullptr);
    REQUIRE(dynamic_cast<SparseLuSolver*>(solver.get()) != nullptr);
}

TEST_CASE("Solver via abstract interface dispatches polymorphically",
          "[v2][layer0][sparse][solver]") {
    // Consumers depend on DirectSolver, not SparseLuSolver. Verify
    // the polymorphic path works end-to-end.
    std::unique_ptr<DirectSolver> solver = make_default_solver();
    Matrix M = make_spd_3x3();
    Vector b(3);
    b << 2.0, 4.0, 2.0;
    Vector x;

    REQUIRE(solver->analyze(M));
    REQUIRE(solver->factorize(M));
    solver->solve(b, x);

    REQUIRE(x[0] == Approx(6.0 / 7.0).margin(1e-12));
}
