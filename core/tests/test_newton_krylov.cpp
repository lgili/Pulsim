#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/solver.hpp"

#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;

TEST_CASE("Newton-Krylov solves scalar nonlinear system", "[newton][jfnk]") {
    NewtonOptions opts;
    opts.enable_newton_krylov = true;
    opts.max_iterations = 20;
    opts.krylov_max_iterations = 20;
    opts.krylov_restart = 5;
    opts.krylov_tolerance = 1e-12;
    opts.tolerances.residual_tol = 1e-12;
    opts.num_nodes = 1;
    opts.num_branches = 0;

    NewtonRaphsonSolver<> solver(opts);

    auto system_func = [](const Vector& x, Vector& f, SparseMatrix& J) {
        f.resize(1);
        f[0] = x[0] * x[0] - 2.0;
        J.resize(1, 1);
        J.setZero();
        J.coeffRef(0, 0) = 2.0 * x[0];
    };

    auto residual_func = [](const Vector& x, Vector& f) {
        f.resize(1);
        f[0] = x[0] * x[0] - 2.0;
    };

    Vector x0(1);
    x0[0] = 1.0;

    auto result = solver.solve(x0, system_func, residual_func);
    REQUIRE(result.success());
    CHECK(result.solution[0] == Approx(std::sqrt(2.0)).margin(1e-6));
}
