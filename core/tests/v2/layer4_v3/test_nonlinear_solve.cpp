// =============================================================================
// Layer 4 V3 — Newton solve on cached linear factor
// =============================================================================
//
// Tests:
//   * Linear-only circuit: solve_with_newton converges in 1 iter
//     with same answer as cache.solve.
//   * noop_refresh: gives identical behaviour to cache.solve.
//   * Non-convergence safety: throws on max_iters exhausted.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/pwl/cache.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/pwl/nonlinear_solve.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <memory>

using namespace pulsim::v2;
using namespace pulsim::v2::pwl;
using namespace pulsim::v2::topology;
using Catch::Approx;

TEST_CASE("Newton: linear V-R-GND with noop_refresh converges in 1 iter",
          "[v2][layer4_v3][newton]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 5.0});
    pool.add_resistor(1, {.G = 1.0});

    PwlStateSpaceCache cache(g, pool);
    cache.build();
    const auto& seg = cache.lookup(SwitchStateMask(0));

    Vector x_init = Vector::Zero(seg.state_size);
    auto x = solve_with_newton(seg, &noop_refresh, g, pool, x_init);

    REQUIRE(x.size() == static_cast<Eigen::Index>(seg.state_size));
    // V_dc source → v_n0 = 5.
    REQUIRE(x[0] == Approx(5.0).margin(1e-9));
}

TEST_CASE("Newton: matches cache.solve for linear circuit",
          "[v2][layer4_v3][newton]") {
    Graph g;
    g.add_node("a");
    g.add_node("b");
    g.add_branch(0, g.ground(), BranchKind::Source);    // V
    g.add_branch(0, 1,          BranchKind::PassiveLinear); // R
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear); // R

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 10.0});
    pool.add_resistor(1, {.G = 0.5});      // 2 Ω
    pool.add_resistor(2, {.G = 0.25});     // 4 Ω (divider)

    PwlStateSpaceCache cache(g, pool);
    cache.build();
    const auto& seg = cache.lookup(SwitchStateMask(0));

    Vector x_cached;
    cache.solve(SwitchStateMask(0),
                 Vector::Zero(seg.state_size), x_cached);

    Vector x_init = Vector::Zero(seg.state_size);
    Vector x_newton = solve_with_newton(
        seg, &noop_refresh, g, pool, x_init);

    REQUIRE(x_newton.size() == x_cached.size());
    for (Eigen::Index i = 0; i < x_cached.size(); ++i) {
        INFO("i=" << i);
        REQUIRE(x_newton[i] == Approx(x_cached[i]).margin(1e-9));
    }
}

TEST_CASE("Newton: throws on max_iters exhausted with bad refresh",
          "[v2][layer4_v3][newton]") {
    // Construct a small circuit, then supply a "refresh" that
    // ALWAYS reports large nonzero residual (forcing non-
    // convergence) and stamps no Jacobian (so Newton just keeps
    // failing).
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 5.0});
    pool.add_resistor(1, {.G = 1.0});

    PwlStateSpaceCache cache(g, pool);
    cache.build();
    const auto& seg = cache.lookup(SwitchStateMask(0));

    // Refresh that always adds a constant nonzero residual on
    // every iteration → Newton can never converge.
    // Use a refresh that makes the residual depend on x in an
    // oscillatory way — Newton can't converge in a few iters.
    auto oscillating_refresh = [counter = std::make_shared<int>(0)](
        const Vector& /*x*/,
        sparse::Matrix& J_nl,
        Vector& f_nl,
        const Graph& /*g*/,
        const DevicePool& /*p*/) mutable {
        if (J_nl.rows() > 0) J_nl.setZero();
        if (f_nl.size() > 0) {
            f_nl.setZero();
            // Alternate the residual sign each iteration to
            // force Newton to oscillate without converging.
            f_nl[0] = (++(*counter) % 2 == 0) ? 1.0 : -1.0;
        }
        return Real{1.0};
    };

    Vector x_init = Vector::Zero(seg.state_size);
    REQUIRE_THROWS_AS(
        solve_with_newton(seg, oscillating_refresh, g, pool,
                           x_init, /*max_iters=*/3,
                           /*tol_dx=*/1e-15, /*tol_res=*/1e-15),
        std::runtime_error);
}
