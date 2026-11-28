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
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

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
// A row that walks a whole-ulp path given as a list of steps (s_k ulps at
// a residual of 12 u: f = 12·sign(s), slope 12/|s| - 1). A closed list
// (steps summing to 0) repeats as a limit cycle; an open one ends.
struct Path {
    std::vector<long> steps;
    std::vector<long> at;   // n before each step
    explicit Path(std::vector<long> s, int repeat = 1) {
        long n = 0;
        for (int r = 0; r < repeat; ++r) {
            for (const long d : s) {
                if (r == 0) { steps.push_back(d); at.push_back(n); }
                n += d;
            }
        }
    }
    Real operator()(const Vector& x, sparse::Matrix& J_nl, Vector& f_nl,
                    const Graph&, const DevicePool&) const {
        const Real u = std::ldexp(1.0, -23);
        const long n = std::lround((-x[1] - 1e9) / u);
        J_nl.setZero();
        f_nl.setZero();
        for (std::size_t k = 0; k < at.size(); ++k) {
            if (at[k] == n) {
                const Real s = static_cast<Real>(steps[k]);
                f_nl[0] = ((s > 0 ? 12.0 : -12.0) + static_cast<Real>(n)) * u;
                J_nl.coeffRef(0, 1) = 12.0 / std::abs(s) - 1.0;
                break;
            }
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

TEST_CASE("Roundoff acceptance: a period-7 cycle is accepted at every phase",
          "[v2][newton][roundoff]") {
    // F-S-5: Newton cycling at the companion's floor with a period near 7.
    // ||dx|| = 48, 20, 10, 4, 2, 12, 4 u. Ending after 48, 20, 10 (or 20,
    // 10, 4) the last two steps contract by more than 4, so the two-step
    // rule of DECISIONS 8 declined it ("still contracting"); the window
    // minimum (2 u) is the same in both windows (15-16 steps here).
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
    const Path cyc({48, -20, -10, -4, 2, -12, -4});
    for (Size iters = 30; iters <= 36; ++iters) {
        NewtonRoundoffAcceptance ro;
        ro.enabled = true;
        INFO(iters);
        const Vector x = solve_with_newton_b_extra(
            seg, cyc, g, pool, x_start(seg),
            Vector::Zero(seg.state_size), iters, 1e-9, 1e-9, false, false,
            nullptr, &ro);
        REQUIRE(ro.accepted);
        REQUIRE(ro.omega <= ro.omega_max);
    }
}

TEST_CASE("Roundoff acceptance: a solve that converged and then sat at the "
          "floor is accepted",
          "[v2][newton][roundoff]") {
    // ll13_active_clamp_forward on trbdf2 (DECISIONS 11 amendment): Newton
    // converged (||dx|| 3e-2 -> 1.8e-5) and then sat at 1.19e-8 for 15
    // iterations at omega 0.31 eps. Compared as two windows of 16, the older
    // window held the convergence, so the floor read as "still contracting";
    // the decline sent the run into an h_min abort. Here: 17 halving steps
    // 2^19 .. 2^3 u, then +-1 u. The windows of kw = 13..16 decline it at
    // budgets 26..33; a split of 8 sees the floor in both windows.
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
    std::vector<long> s;
    for (int k = 0; k < 17; ++k) s.push_back((k % 2 == 0 ? 1L : -1L) << (19 - k));
    for (int k = 0; k < 40; ++k) s.push_back(k % 2 == 0 ? 1 : -1);
    const Path settle(s);
    for (Size iters = 26; iters <= 33; ++iters) {
        NewtonRoundoffAcceptance ro;
        ro.enabled = true;
        INFO(iters);
        const Vector x = solve_with_newton_b_extra(
            seg, settle, g, pool, x_start(seg),
            Vector::Zero(seg.state_size), iters, 1e-9, 1e-9, false, false,
            nullptr, &ro);
        REQUIRE(ro.accepted);
        REQUIRE(ro.omega <= ro.omega_max);
    }
}

TEST_CASE("Roundoff acceptance: one small step inside a cycle does not read "
          "as contraction",
          "[v2][newton][roundoff]") {
    // ll13 again: a cycle at 1.2-1.4e-6 with ONE step of 5.4e-8 among the
    // last 16 made the newer window's minimum 23x below the older's. Here:
    // steps 21, 19 u with one step of 1 u (step 17 of the path); at budgets
    // 25..32 it lies 8..15 iterations back, inside the newer window of kw
    // and inside the OLDER window of the split 8.
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
    std::vector<long> s;
    for (int k = 0; k < 8; ++k) { s.push_back(21); s.push_back(-19); }
    s.push_back(1);
    s.push_back(20);
    for (int k = 0; k < 20; ++k) { s.push_back(-19); s.push_back(21); }
    const Path dip(s);
    for (Size iters = 25; iters <= 32; ++iters) {
        NewtonRoundoffAcceptance ro;
        ro.enabled = true;
        INFO(iters);
        (void)solve_with_newton_b_extra(
            seg, dip, g, pool, x_start(seg),
            Vector::Zero(seg.state_size), iters, 1e-9, 1e-9, false, false,
            nullptr, &ro);
        REQUIRE(ro.accepted);
        REQUIRE(ro.omega <= ro.omega_max);
    }
}

TEST_CASE("Roundoff acceptance: slow convergence near the floor is declined",
          "[v2][newton][roundoff]") {
    // The protected case: the residual is at roundoff (12 u, omega 1.6 eps)
    // but ||dx|| still falls geometrically at rate 0.7, so the iterate is
    // still moving toward its root. The two-step rule accepted it (0.49 of
    // ||dx_{k-2}|| is "no net contraction by 4"); the window minimum falls
    // by 0.7^k for every split k = 8..15 compared here (0.7^8 = 0.058 < 1/4)
    // and it is declined while it contracts.
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
    std::vector<long> s;
    for (int k = 0; k < 40; ++k) {
        s.push_back(std::lround(65536.0 * std::pow(0.7, k)) + 1);
    }
    const Path slow(s);
    for (const Size iters : {Size{20}, Size{25}, Size{30}}) {
        NewtonRoundoffAcceptance ro;
        ro.enabled = true;
        INFO(iters);
        try {
            (void)solve_with_newton_b_extra(
                seg, slow, g, pool, x_start(seg),
                Vector::Zero(seg.state_size), iters, 1e-9, 1e-9, false,
                false, nullptr, &ro);
            FAIL("a converging iteration was accepted");
        } catch (const std::runtime_error& e) {
            REQUIRE(std::string(e.what()).find("still contracting")
                    != std::string::npos);
        }
        REQUIRE_FALSE(ro.accepted);
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

TEST_CASE("Roundoff acceptance: a floating-point fixed point is judged at its "
          "own magnitudes only when judge_own_scale is set",
          "[v2][newton][roundoff]") {
    // DECISIONS 12 (ll13 on trbdf2): the start state is cold (x = 0, so the
    // KCL row has no magnitude at the start's scale: omega = inf there),
    // the first step lands at 1e9 and Newton then cycles +2, -1, -1 u
    // (u = ulp(1e9), every step within 4 eps of x_k). At x_k's own scale the
    // row's residual (12 u) is ~3 eps. trbdf2/esdirk43 set the flag and
    // accept; the fixed-step engine leaves it off and keeps 2.1.0's outcome:
    // the throw its caller turns into a halved-dt retry.
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
    const Path cyc({2, -1, -1});
    const Vector x0 = Vector::Zero(seg.state_size);
    {
        NewtonRoundoffAcceptance ro;
        ro.enabled = true;
        ro.judge_own_scale = true;
        (void)solve_with_newton_b_extra(seg, cyc, g, pool, x0,
                                        Vector::Zero(seg.state_size), 40, 1e-9,
                                        1e-9, false, false, nullptr, &ro);
        REQUIRE(ro.accepted);
        REQUIRE(ro.fixed_point);
        REQUIRE(ro.omega <= ro.omega_max);
        REQUIRE(ro.total == 1);
    }
    {
        NewtonRoundoffAcceptance ro;   // the default: as the fixed-step engine
        ro.enabled = true;
        REQUIRE_FALSE(ro.judge_own_scale);
        try {
            (void)solve_with_newton_b_extra(seg, cyc, g, pool, x0,
                                            Vector::Zero(seg.state_size), 40,
                                            1e-9, 1e-9, false, false, nullptr,
                                            &ro);
            FAIL("the fixed point was accepted with judge_own_scale off");
        } catch (const NewtonRoundoffBudgetExceeded&) {
            FAIL("a decline must not be the budget error");
        } catch (const std::runtime_error& e) {
            REQUIRE(std::string(e.what()).find("Roundoff acceptance declined")
                    != std::string::npos);
        }
        REQUIRE_FALSE(ro.accepted);
        REQUIRE_FALSE(ro.fixed_point);
        REQUIRE(ro.total == 0);
        REQUIRE(ro.consecutive == 0);
    }
}

namespace {
// V1 = 1 GV at n0, 1 S n0-n1, 1 S n1-ground, V2 = 0.5 GV at n1: V2's current
// is the n1 KCL row's cancellation of 5e8 A terms, ~0 A. A device on that
// row walks V2's current in steps of u = 2^-24 A (one ulp of the 5e8 terms
// that set it) at a residual of 12 u (1.6 eps of the row's terms), from the
// moment the first step lands (off at the cold start, so the start state's
// scale has nothing on the row). ll13's Vin branch at the 1.5 us turn-off:
// 13 A moving 3.7e-9 A, two million of its own ulps.
struct PropagatedPath {
    Path p;
    Real u;
    Real r = 12;   // residual, in u
    Real operator()(const Vector& x, sparse::Matrix& J_nl, Vector& f_nl,
                    const Graph&, const DevicePool&) const {
        J_nl.setZero();
        f_nl.setZero();
        if (std::abs(x[0]) < 1) return 0.0;
        const long n = std::lround(-x[3] / u);
        for (std::size_t k = 0; k < p.at.size(); ++k) {
            if (p.at[k] == n) {
                const Real s = static_cast<Real>(p.steps[k]);
                f_nl[1] = ((s > 0 ? r : -r) + static_cast<Real>(n)) * u;
                J_nl.coeffRef(1, 3) = r / std::abs(s) - 1.0;
                break;
            }
        }
        return 0.0;
    }
};
struct TwoSourceCircuit {
    Graph g;
    DevicePool pool;
    std::unique_ptr<PwlStateSpaceCache> cache;
    TwoSourceCircuit() {
        g.add_node("n0");
        g.add_node("n1");
        g.add_branch(0, g.ground(), BranchKind::Source);
        g.add_branch(0, 1, BranchKind::PassiveLinear);
        g.add_branch(1, g.ground(), BranchKind::PassiveLinear);
        g.add_branch(1, g.ground(), BranchKind::Source);
        pool.add_voltage_source(0, {.V = 1e9});
        pool.add_resistor(1, {.G = 1.0});
        pool.add_resistor(2, {.G = 1.0});
        pool.add_voltage_source(3, {.V = 5e8});
        cache = std::make_unique<PwlStateSpaceCache>(g, pool);
        cache->build();
    }
};
}  // namespace

TEST_CASE("Roundoff acceptance: a fixed point at the rounding propagated to a "
          "component is accepted, a larger step is not",
          "[v2][newton][roundoff]") {
    TwoSourceCircuit c;
    const auto& seg = c.cache->lookup(SwitchStateMask(0));
    REQUIRE(seg.state_size == 4);
    const Real u = std::ldexp(1.0, -24);
    const Real eps = std::numeric_limits<Real>::epsilon();
    {
        const PropagatedPath dev{Path({2, -1, -1}), u};
        NewtonRoundoffAcceptance ro;
        ro.enabled = true;
        ro.judge_own_scale = true;
        const Vector x = solve_with_newton_b_extra(
            seg, dev, c.g, c.pool, Vector::Zero(4), Vector::Zero(4), 40, 1e-9,
            1e-9, false, false, nullptr, &ro);
        REQUIRE(ro.accepted);
        REQUIRE(ro.fixed_point);
        // the step it stalled at is above the absolute tolerance and far
        // above V2's current's own ulps: only the propagated bound takes it
        REQUIRE(u > 1e-9);
        REQUIRE(u > 1e6 * eps * std::abs(x[3]));
        REQUIRE(std::abs(x[1] - 5e8) <= 1e-6 * 5e8);
        // the accepted iterate solves the unmodified equations to the
        // round-off of each row's own terms
        sparse::Matrix J_nl(4, 4);
        Vector f_nl = Vector::Zero(4);
        (void)dev(x, J_nl, f_nl, c.g, c.pool);
        const Vector res = seg.J * x + seg.b_constant + f_nl;
        const Vector sk = Vector(seg.J.cwiseAbs() * x.cwiseAbs())
                          + seg.b_constant.cwiseAbs() + f_nl.cwiseAbs();
        for (Index i = 0; i < 4; ++i) {
            INFO(i);
            REQUIRE(std::abs(res[i]) <= ro.omega_max * sk[i]);
        }
    }
    {
        // Not converged: the same cycle scaled by 2^14 (steps of 1e-3 A, the
        // residual 2^14 x 12 u = 1.2e-2 A, 2.6e4 eps of the row's terms) on
        // an equally well-conditioned row (slope 5): stagnated and inside
        // the dx cap (|x_0| = 1e4), 1e3 x the propagated bound. Declined.
        const PropagatedPath dev{Path({1L << 15, -(1L << 14), -(1L << 14)}), u,
                                 12.0 * (1L << 14)};
        NewtonRoundoffAcceptance ro;
        ro.enabled = true;
        ro.judge_own_scale = true;
        Vector x0 = Vector::Zero(4);
        x0[2] = 1e4;
        try {
            (void)solve_with_newton_b_extra(seg, dev, c.g, c.pool, x0,
                                            Vector::Zero(4), 40, 1e-9, 1e-9,
                                            false, false, nullptr, &ro);
            FAIL("a step far above the propagated rounding was accepted: fp="
                 << ro.fixed_point << " omega/eps=" << ro.omega / eps
                 << " row=" << ro.row);
        } catch (const NewtonRoundoffBudgetExceeded&) {
            FAIL("a decline must not be the budget error");
        } catch (const std::runtime_error& e) {
            INFO(e.what());
            REQUIRE(std::string(e.what()).find("Roundoff acceptance declined")
                    != std::string::npos);
            REQUIRE(std::string(e.what()).find("stagnated") != std::string::npos);
            REQUIRE(std::string(e.what()).find("within 1e-6") != std::string::npos);
        }
        REQUIRE_FALSE(ro.accepted);
    }
}
