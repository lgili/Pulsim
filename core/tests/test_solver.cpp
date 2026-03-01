#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "pulsim/solver.hpp"

using namespace pulsim;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("Linear solver", "[solver]") {
    LinearSolver solver;

    SECTION("Simple 2x2 system") {
        // Solve: [2 1; 1 3] * x = [5; 6]
        // Solution: x = [1.8; 1.4]
        SparseMatrix A(2, 2);
        A.insert(0, 0) = 2.0;
        A.insert(0, 1) = 1.0;
        A.insert(1, 0) = 1.0;
        A.insert(1, 1) = 3.0;
        A.makeCompressed();

        Vector b(2);
        b << 5.0, 6.0;

        auto result = solver.solve(A, b);
        REQUIRE(result.status == SolverStatus::Success);

        CHECK_THAT(result.x(0), WithinRel(1.8, 1e-10));
        CHECK_THAT(result.x(1), WithinRel(1.4, 1e-10));
    }

    SECTION("Reuse factorization") {
        SparseMatrix A(2, 2);
        A.insert(0, 0) = 4.0;
        A.insert(0, 1) = 1.0;
        A.insert(1, 0) = 1.0;
        A.insert(1, 1) = 3.0;
        A.makeCompressed();

        solver.analyze_pattern(A);
        REQUIRE(solver.factorize(A));

        Vector b1(2);
        b1 << 5.0, 4.0;
        auto result1 = solver.solve(b1);
        REQUIRE(result1.status == SolverStatus::Success);

        Vector b2(2);
        b2 << 10.0, 8.0;
        auto result2 = solver.solve(b2);
        REQUIRE(result2.status == SolverStatus::Success);

        // Second solution should be 2x the first (linear system)
        CHECK_THAT(result2.x(0), WithinRel(2.0 * result1.x(0), 1e-10));
        CHECK_THAT(result2.x(1), WithinRel(2.0 * result1.x(1), 1e-10));
    }
}

TEST_CASE("Newton solver", "[solver]") {
    NewtonSolver::Options opts;
    opts.abstol = 1e-10;
    opts.reltol = 1e-8;
    opts.max_iterations = 50;

    NewtonSolver solver(opts);

    SECTION("Linear system (converges in 1 iteration)") {
        // Solve Ax - b = 0 where A = [2 0; 0 3], b = [4; 9]
        // Solution: x = [2; 3]
        auto system = [](const Vector& x, Vector& f, SparseMatrix& J) {
            f.resize(2);
            f(0) = 2.0 * x(0) - 4.0;
            f(1) = 3.0 * x(1) - 9.0;

            J.resize(2, 2);
            J.insert(0, 0) = 2.0;
            J.insert(1, 1) = 3.0;
            J.makeCompressed();
        };

        Vector x0 = Vector::Zero(2);
        auto result = solver.solve(x0, system);

        REQUIRE(result.status == SolverStatus::Success);
        CHECK_THAT(result.x(0), WithinAbs(2.0, 1e-9));
        CHECK_THAT(result.x(1), WithinAbs(3.0, 1e-9));
        CHECK(result.iterations <= 2);
    }

    SECTION("Nonlinear system: x^2 = 4") {
        // Solve x^2 - 4 = 0
        // Solutions: x = 2 or x = -2
        auto system = [](const Vector& x, Vector& f, SparseMatrix& J) {
            f.resize(1);
            f(0) = x(0) * x(0) - 4.0;

            J.resize(1, 1);
            J.insert(0, 0) = 2.0 * x(0);
            J.makeCompressed();
        };

        Vector x0(1);
        x0(0) = 1.0;  // Start at x=1
        auto result = solver.solve(x0, system);

        REQUIRE(result.status == SolverStatus::Success);
        CHECK_THAT(result.x(0), WithinAbs(2.0, 1e-9));
    }

    SECTION("Nonlinear system: exp(x) = 2") {
        // Solve exp(x) - 2 = 0
        // Solution: x = ln(2) ≈ 0.693
        auto system = [](const Vector& x, Vector& f, SparseMatrix& J) {
            f.resize(1);
            f(0) = std::exp(x(0)) - 2.0;

            J.resize(1, 1);
            J.insert(0, 0) = std::exp(x(0));
            J.makeCompressed();
        };

        Vector x0(1);
        x0(0) = 0.0;
        auto result = solver.solve(x0, system);

        REQUIRE(result.status == SolverStatus::Success);
        CHECK_THAT(result.x(0), WithinAbs(std::log(2.0), 1e-9));
    }

    SECTION("2D nonlinear system") {
        // Solve: x^2 + y^2 = 1, x = y
        // Solution: x = y = 1/sqrt(2) ≈ 0.707
        auto system = [](const Vector& x, Vector& f, SparseMatrix& J) {
            f.resize(2);
            f(0) = x(0) * x(0) + x(1) * x(1) - 1.0;
            f(1) = x(0) - x(1);

            J.resize(2, 2);
            J.insert(0, 0) = 2.0 * x(0);
            J.insert(0, 1) = 2.0 * x(1);
            J.insert(1, 0) = 1.0;
            J.insert(1, 1) = -1.0;
            J.makeCompressed();
        };

        Vector x0(2);
        x0 << 0.5, 0.5;
        auto result = solver.solve(x0, system);

        REQUIRE(result.status == SolverStatus::Success);
        Real expected = 1.0 / std::sqrt(2.0);
        CHECK_THAT(result.x(0), WithinAbs(expected, 1e-8));
        CHECK_THAT(result.x(1), WithinAbs(expected, 1e-8));
    }
}
