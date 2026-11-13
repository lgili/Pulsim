// =============================================================================
// Newton roundoff acceptance (add-newton-roundoff-acceptance): the net-
// contraction criterion, and the DC call sites that must never use it.
// =============================================================================

#include <catch2/catch_test_macros.hpp>

#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/nonlinear_solve.hpp"
#include "pulsim/topology/graph.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::topology;

namespace {
// A 1 GV source on a 1 S resistor (i = -1e9 A) and a "device" on the KCL
// row that walks a roundoff limit cycle in whole ulps u = 2^-23 of i:
// n (ulps of i below -1e9) goes 0 -> 6 -> -4 -> 0, so ||dx|| = 6, 10, 4 u
// (7.2e-7, 1.2e-6, 4.8e-7; the documented stall ended in 3.2e-7, 5.0e-7,
// 1.9e-7). The residual stays at 12 u on a row scale of ~4e9 (omega
// 1.6 eps) and never grows, so no line search is promoted; the device's
// d(f)/d(i) sets the step: dx = -f/(1 + a).
struct Cycle {
    Real operator()(const Vector& x, sparse::Matrix& J_nl, Vector& f_nl,
                    const Graph&, const DevicePool&) const {
        const Real u = std::ldexp(1.0, -23);
        const long n = std::lround((-x[1] - 1e9) / u);
        Real f = 0, a = 0;   // residual (ulps) and slope wanted at n
        if (n == 0) { f = 12; a = 1.0; }          // step +6
        else if (n == 6) { f = -12; a = 0.2; }    // step -10
        else if (n == -4) { f = 12; a = 2.0; }    // step +4
        J_nl.setZero();
        f_nl.setZero();
        if (f != 0) {
            f_nl[0] = (f + static_cast<Real>(n)) * u;
            J_nl.coeffRef(0, 1) = a;
        }
        return 0.0;
    }
};
// The same row, but ||dx|| DOUBLES each iteration (1, 2, 4, ... u):
// n = 2^k - 1 steps by 2^k ulps at a residual of 12 u. Every residual
// is at roundoff against the row scale and ||dx|| never contracts, so
// only the two-sided test can see that the iterate is running away.
struct Grow {
    Real operator()(const Vector& x, sparse::Matrix& J_nl, Vector& f_nl,
                    const Graph&, const DevicePool&) const {
        const Real u = std::ldexp(1.0, -23);
        const long n = std::lround((-x[1] - 1e9) / u);
        J_nl.setZero();
        f_nl.setZero();
        const Real p2 = static_cast<Real>(n + 1);
        if (n >= 0 && (n & (n + 1)) == 0) {   // n = 2^k - 1
            f_nl[0] = (12.0 + static_cast<Real>(n)) * u;
            J_nl.coeffRef(0, 1) = 12.0 / p2 - 1.0;
        }
        return 0.0;
    }
};
// A transient step starts from the previous state, here the solution
// itself (v = 1e9 V, i = -1e9 A): the acceptance takes its row scale
// from that start state, so a cold zero start has none on the KCL row.
Vector x_start(const PwlSegment& seg) {
    Vector x = Vector::Zero(seg.state_size);
    x[0] = 1e9;
    x[1] = -1e9;
    return x;
}
}  // namespace

TEST_CASE("Roundoff acceptance: a growing ||dx|| is never accepted",
          "[v2][newton][roundoff]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);
    DevicePool pool;
    pool.add_voltage_source(0, {.V = 1e9});
    pool.add_resistor(1, {.G = 1.0});
    PwlStateSpaceCache cache(g, pool);
    cache.build();
    const auto& seg = cache.lookup(SwitchStateMask(0));
    NewtonRoundoffAcceptance ro;
    ro.enabled = true;
    const Grow grow;
    try {
        (void)solve_with_newton_b_extra(
            seg, grow, g, pool, x_start(seg),
            Vector::Zero(seg.state_size), 12, 1e-9, 1e-9, false, false,
            nullptr, &ro);
        FAIL("a growing sequence was accepted");
    } catch (const std::runtime_error& e) {
        REQUIRE(std::string(e.what()).find("growing") != std::string::npos);
    }
    REQUIRE_FALSE(ro.accepted);
    REQUIRE(ro.total == 0);
}

TEST_CASE("Roundoff acceptance: the run budget refuses a run of accepts "
          "by name",
          "[v2][newton][roundoff]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);
    DevicePool pool;
    pool.add_voltage_source(0, {.V = 1e9});
    pool.add_resistor(1, {.G = 1.0});
    PwlStateSpaceCache cache(g, pool);
    cache.build();
    const auto& seg = cache.lookup(SwitchStateMask(0));
    NewtonRoundoffAcceptance ro;   // one object per run, as the engines keep it
    ro.enabled = true;
    ro.max_consecutive = 2;
    for (int k = 0; k < 2; ++k) {
        const Cycle cyc;
        (void)solve_with_newton_b_extra(
            seg, cyc, g, pool, x_start(seg),
            Vector::Zero(seg.state_size), 10, 1e-9, 1e-9, false, false,
            nullptr, &ro);
        REQUIRE(ro.accepted);
    }
    REQUIRE(ro.consecutive == 2);
    const Cycle cyc;
    try {
        (void)solve_with_newton_b_extra(
            seg, cyc, g, pool, x_start(seg),
            Vector::Zero(seg.state_size), 10, 1e-9, 1e-9, false, false,
            nullptr, &ro);
        FAIL("the budget did not throw");
    } catch (const NewtonRoundoffBudgetExceeded& e) {
        const std::string w = e.what();
        REQUIRE(w.find("3 consecutive") != std::string::npos);
        REQUIRE(w.find("2 accepted in this run") != std::string::npos);
    }
    REQUIRE(ro.total == 2);
}

TEST_CASE("Roundoff acceptance: a limit cycle without net contraction "
          "is accepted at every budget",
          "[v2][newton][roundoff]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);
    DevicePool pool;
    pool.add_voltage_source(0, {.V = 1e9});
    pool.add_resistor(1, {.G = 1.0});
    PwlStateSpaceCache cache(g, pool);
    cache.build();
    const auto& seg = cache.lookup(SwitchStateMask(0));
    // Every phase of the cycle ends the budget once. With "each of the last
    // two steps at least half the previous" two of the three endings read
    // as contraction and the solve threw.
    for (const Size iters : {Size{9}, Size{10}, Size{11}}) {
        NewtonRoundoffAcceptance ro;
        ro.enabled = true;
        const Cycle cyc;
        const Vector x = solve_with_newton_b_extra(
            seg, cyc, g, pool, x_start(seg),
            Vector::Zero(seg.state_size), iters, 1e-9, 1e-9, false, false, nullptr,
            &ro);
        REQUIRE(ro.accepted);
        REQUIRE(ro.omega <= ro.omega_max);
        REQUIRE(std::abs(x[0] - 1e9) < 1.0);
        // Without the acceptance (the DC callers pass none) it throws.
        const Cycle cyc2;
        REQUIRE_THROWS(solve_with_newton_b_extra(
            seg, cyc2, g, pool, x_start(seg),
            Vector::Zero(seg.state_size), iters, 1e-9, 1e-9));
    }
}

TEST_CASE("Roundoff acceptance: the DC call sites pass none",
          "[v2][newton][roundoff]") {
    // The DC cascade depends on the absolute step test FAILING on a stiff
    // diode chain (it falls through to gmin stepping; a relaxed test stops
    // at a root 10.67 V off). No DC source file may name the acceptance.
    const std::filesystem::path inc{PULSIM_PWL_INCLUDE_DIR};
    for (const char* f : {"continuation.hpp", "dc_strategy.hpp",
                          "dc_assemble.hpp", "dc_operating_point.hpp",
                          "gmin.hpp", "pseudo_transient.hpp"}) {
        std::ifstream in(inc / f);
        REQUIRE(in.good());
        std::stringstream ss;
        ss << in.rdbuf();
        INFO(f);
        REQUIRE(ss.str().find("NewtonRoundoffAcceptance") == std::string::npos);
        REQUIRE(ss.str().find("roundoff") == std::string::npos);
    }
}
