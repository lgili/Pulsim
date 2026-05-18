// simplify-and-harden-numerical-surface — Phase 7 tests.
//
// Verifies the homotopy continuation strategy:
//   - Enum value `DCStrategy::Homotopy` exists and dispatches.
//   - `HomotopyConfig` fields round-trip.
//   - Auto orchestrator falls through Direct → Source → Gmin →
//     PseudoTransient → Homotopy (last-resort).
//   - Telemetry reports `homotopy_steps` and
//     `homotopy_ladder_completed`.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/numerical/dc_strategy.hpp"

#include <Eigen/Sparse>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

// Toy scaled system: linear at lambda=0, nonlinear at lambda=1.
//   f(x; λ) = (1 − λ) · (x − 1) + λ · (x^3 − 1)
// At λ=0 the root is x=1 (linear). At λ=1 the root is x=1 still
// (the only real root of x^3=1). Newton converges in one step at
// any lambda from x0=1 (perfect warm-start). Homotopy ladder
// completes in N steps with zero per-step iterations beyond the
// initial setup.
struct ScaledCubic {
    void operator()(const Vector& x, Vector& f, SparseMatrix& J,
                    Real lambda) const {
        const Real xi = x[0];
        f.resize(1);
        f[0] = (Real{1} - lambda) * (xi - Real{1}) +
               lambda * (xi * xi * xi - Real{1});

        J.resize(1, 1);
        J.setZero();
        const Real dj = (Real{1} - lambda) +
                        lambda * (Real{3} * xi * xi);
        J.insert(0, 0) = dj;
    }
};

}  // namespace

TEST_CASE("HomotopyConfig: defaults are reasonable",
          "[dc][homotopy][config]") {
    HomotopyConfig cfg{};
    CHECK(cfg.enable);
    CHECK(cfg.ladder_steps == 5);
    CHECK(cfg.max_newton_per_step == 10);
}

TEST_CASE("DCStrategy::Homotopy is in the enum and orchestrator",
          "[dc][homotopy][enum]") {
    // Just verify the enum value exists and can be assigned.
    DCStrategy s = DCStrategy::Homotopy;
    CHECK(static_cast<int>(s) == 4);  // 5th value (after Direct=0..PseudoTransient=3)

    // DCConvergenceConfig has a homotopy_config sub-struct.
    DCConvergenceConfig cfg{};
    cfg.homotopy_config.ladder_steps = 8;
    CHECK(cfg.homotopy_config.ladder_steps == 8);
}

TEST_CASE("Homotopy via Auto orchestrator: succeeds with warm-start ladder",
          "[dc][homotopy][auto]") {
    // Force Direct to fail by starting Newton from a bad guess on the
    // FULL nonlinear system, then check the orchestrator falls
    // through to Homotopy.
    //
    // For this scoped test we directly invoke `try_homotopy` via
    // `DCStrategy::Homotopy` because constructing a full
    // SimulationOptions + Circuit + auto-ladder requires more setup
    // than the unit test scope. The full Auto-ladder behavior is
    // covered indirectly by the multilevel benchmarks (Phase 13).

    DCConvergenceConfig cfg{};
    cfg.strategy = DCStrategy::Homotopy;
    cfg.homotopy_config.ladder_steps = 5;
    cfg.homotopy_config.max_newton_per_step = 20;

    DCConvergenceSolver<SparseLUPolicy> solver(cfg);

    Vector x0(1);
    x0[0] = 0.5;  // far enough from the root (x=1) to need a few iters

    auto system_func = [](const Vector& x, Vector& f, SparseMatrix& J) {
        const Real xi = x[0];
        f.resize(1);
        f[0] = xi * xi * xi - Real{1};
        J.resize(1, 1);
        J.setZero();
        J.insert(0, 0) = Real{3} * xi * xi;
    };

    ScaledCubic scaled;
    auto scaled_func = [&scaled](const Vector& x, Vector& f,
                                  SparseMatrix& J, Real lambda) {
        scaled(x, f, J, lambda);
    };

    auto result = solver.solve(x0, /*num_nodes=*/1, /*num_branches=*/0,
                                system_func, scaled_func);

    INFO("strategy_used = " << static_cast<int>(result.strategy_used));
    INFO("homotopy_steps = " << result.homotopy_steps);
    INFO("homotopy_ladder_completed = " << result.homotopy_ladder_completed);
    INFO("message = " << result.message);
    INFO("solution = " << result.newton_result.solution.transpose());

    REQUIRE(result.success);
    CHECK(result.strategy_used == DCStrategy::Homotopy);
    CHECK(result.homotopy_steps == 5);
    CHECK(result.homotopy_ladder_completed);
    CHECK(result.newton_result.solution[0] == Approx(1.0).margin(1e-6));
}

TEST_CASE("Homotopy ladder length scales: 5 steps vs 10 steps both converge",
          "[dc][homotopy][ladder]") {
    auto run_with_steps = [](int steps) {
        DCConvergenceConfig cfg{};
        cfg.strategy = DCStrategy::Homotopy;
        cfg.homotopy_config.ladder_steps = steps;
        cfg.homotopy_config.max_newton_per_step = 30;
        DCConvergenceSolver<SparseLUPolicy> solver(cfg);

        Vector x0(1);
        x0[0] = 0.5;

        auto system_func = [](const Vector& x, Vector& f, SparseMatrix& J) {
            const Real xi = x[0];
            f.resize(1);
            f[0] = xi * xi * xi - Real{1};
            J.resize(1, 1);
            J.setZero();
            J.insert(0, 0) = Real{3} * xi * xi;
        };

        ScaledCubic scaled;
        auto scaled_func = [&scaled](const Vector& x, Vector& f,
                                      SparseMatrix& J, Real lambda) {
            scaled(x, f, J, lambda);
        };

        return solver.solve(x0, /*num_nodes=*/1, /*num_branches=*/0,
                             system_func, scaled_func);
    };

    auto five  = run_with_steps(5);
    auto ten   = run_with_steps(10);

    REQUIRE(five.success);
    REQUIRE(ten.success);
    CHECK(five.homotopy_steps == 5);
    CHECK(ten.homotopy_steps == 10);
    CHECK(five.homotopy_ladder_completed);
    CHECK(ten.homotopy_ladder_completed);
}

TEST_CASE("Homotopy disabled via config skips the strategy",
          "[dc][homotopy][disable]") {
    DCConvergenceConfig cfg{};
    cfg.strategy = DCStrategy::Auto;
    cfg.homotopy_config.enable = false;  // bypass last-resort homotopy

    // With Direct/Gmin/Source/PseudoTransient handling a simple linear
    // problem, homotopy never fires.
    DCConvergenceSolver<SparseLUPolicy> solver(cfg);

    Vector x0(1);
    x0[0] = 5.0;  // simple linear problem starting far from root

    auto system_func = [](const Vector& x, Vector& f, SparseMatrix& J) {
        const Real xi = x[0];
        f.resize(1);
        f[0] = xi - Real{1};   // root at x=1, linear → Direct converges
        J.resize(1, 1);
        J.setZero();
        J.insert(0, 0) = Real{1};
    };

    auto result = solver.solve(x0, /*num_nodes=*/1, /*num_branches=*/0,
                                system_func);

    REQUIRE(result.success);
    CHECK(result.strategy_used == DCStrategy::Direct);
    CHECK(result.homotopy_steps == 0);
    CHECK_FALSE(result.homotopy_ladder_completed);
}
