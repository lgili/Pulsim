// =============================================================================
// DSED — the period map and its closed-form integrals
// =============================================================================
//
// add-design-optimization-workflow, kernel item 3
// (pulsim/dsed/period_integrals.hpp): ∫y, ∫y², ∫ya·yb, the extrema and
// M = dP/dx0 against analytic answers — across a gate edge, with a
// repeated eigenvalue, and the refusal on a defective basis.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/dsed/builder_adapter.hpp"
#include "pulsim/dsed/period_integrals.hpp"
#include "pulsim/pwl/cache.hpp"

#include <cmath>

using namespace pulsim;

namespace {

using SSM = topology::SwitchStateMask;

bool rel(Real a, Real b, Real tol = 1e-12) {
    return std::abs(a - b) <= tol * std::max(std::abs(a), std::abs(b));
}

/// The switch on for t mod T < D·T, with an edge oracle.
struct Gate {
    Real T, D;
    std::size_t n_sw;
    SSM operator()(Real t) const {
        SSM m(n_sw);
        const Real ph = t - T * std::floor(t / T);
        if (n_sw) m.set(0, ph < D * T);
        return m;
    }
    Real next_edge_after(Real t) const {
        const Real k = std::floor(t / T);
        for (Real c : {k * T + D * T, (k + 1) * T, (k + 1) * T + D * T}) {
            if (c > t * (1 + 1e-15) + 1e-300) return c;
        }
        return (k + 2) * T;
    }
};

template <class Ad>
dsed::PeriodMapResult<SSM> one_period(
    Ad& ad, const Gate& g, const Vector& x0,
    const std::vector<dsed::PeriodObservable>& obs,
    const std::vector<std::pair<int, int>>& pairs) {
    dsed::EventPredictor pred;
    auto diodes = dsed::add_diode_predicates(ad, pred);
    dsed::PEDSimulator<Ad, Gate> sim(ad, g, dsed::PIController(1e-11, 1e-14),
                                     std::move(pred), g.T * 1e-6, g.T, 1);
    sim.set_event_projection(false);
    const auto res = sim.simulate(x0, g.T);
    return dsed::period_map_from_run(ad, res, diodes, obs, pairs, true);
}

dsed::PeriodObservable node(Index k, bool extrema = true) {
    dsed::PeriodObservable o;
    o.w = {{k, 1.0}};
    o.extrema = extrema;
    return o;
}

}  // namespace

TEST_CASE("Period map: RC charge, integrals and M in closed form",
          "[v2][dsed][period_integrals]") {
    const Real V = 10.0, R = 1e3, C = 1e-6, tau = R * C, T = 1e-3;
    builder::CircuitBuilder b;
    b.add_voltage_source("V1", "in", "gnd", V);
    b.add_resistor("R1", "in", "n1", R);
    b.add_capacitor("C1", "n1", "gnd", C);
    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    dsed::NativeCircuitBuilderAdapter<> ad{b.graph(), b.pool(), cache};
    const Index k = b.node_id_of("n1");
    const auto r = one_period(ad, Gate{T, 0.5, 0}, Vector::Zero(1),
                              {node(k), node(k)}, {{0, 1}});
    REQUIRE(r.integrals_ok);
    const Real e1 = std::exp(-T / tau), e2 = std::exp(-2 * T / tau);
    const Real I1 = V * (T - tau * (1 - e1));
    const Real I2 = V * V * (T - 2 * tau * (1 - e1) + tau / 2 * (1 - e2));
    CHECK(rel(r.stats[0].int1, I1));
    CHECK(rel(r.stats[0].int2, I2));
    CHECK(rel(r.cross[0], I2));
    CHECK(rel(r.stats[0].peak, V * (1 - e1)));
    CHECK(r.stats[0].t_peak == T);
    CHECK(r.stats[0].valley == 0.0);
    REQUIRE(r.monodromy.rows() == 1);
    CHECK(rel(r.monodromy(0, 0), e1));
}

TEST_CASE("Period map: a gate edge splits the period, M composes",
          "[v2][dsed][period_integrals]") {
    const Real V = 10.0, R = 100.0, C = 1e-6, T = 4e-4, D = 0.3;
    const Real g_on = 1e3, g_off = 1e-6;
    builder::CircuitBuilder b;
    b.add_voltage_source("V1", "in", "gnd", V);
    b.add_switch("S", "in", "a", g_on, g_off);
    b.add_resistor("R1", "a", "n1", R);
    b.add_capacitor("C1", "n1", "gnd", C);
    b.add_resistor("Rd", "n1", "gnd", 300.0);
    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    dsed::NativeCircuitBuilderAdapter<> ad{b.graph(), b.pool(), cache};
    const Index k = b.node_id_of("n1");
    Vector x0(1);
    x0 << 1.5;
    const auto r = one_period(ad, Gate{T, D, 1}, x0, {node(k)}, {});
    REQUIRE(r.integrals_ok);
    // v' = (v_s - v)/tau per phase: Thevenin of (V, R + 1/g) against Rd
    auto phase = [&](Real g, Real v0, Real h, Real& I1) {
        const Real Rs = R + 1 / g, Rd = 300.0;
        const Real vs = V * Rd / (Rs + Rd), tau = C * Rs * Rd / (Rs + Rd);
        const Real e = std::exp(-h / tau);
        I1 += vs * h + (v0 - vs) * tau * (1 - e);
        return std::pair{vs + (v0 - vs) * e, e};
    };
    Real I1 = 0;
    const auto [v1, m1] = phase(g_on, 1.5, D * T, I1);
    const auto [v2, m2] = phase(g_off, v1, (1 - D) * T, I1);
    CHECK(rel(r.x_end[0], v2, 1e-11));
    CHECK(rel(r.stats[0].int1, I1));
    CHECK(rel(r.monodromy(0, 0), m1 * m2, 1e-11));
    CHECK(rel(r.stats[0].peak, v1, 1e-11));
    CHECK(std::abs(r.stats[0].t_peak - D * T) <= 1e-12 * T);
    REQUIRE(r.events.size() >= 1);
    // the switch current jumps at the edge: left and right limits
    dsed::PeriodObservable is = node(b.node_id_of("in"));
    is.w.push_back({b.node_id_of("a"), -1.0});
    is.bit = 0;
    is.g_on = g_on;
    is.g_off = g_off;
    const auto r2 = one_period(ad, Gate{T, D, 1}, x0, {is}, {});
    REQUIRE(r2.y_left.rows() >= 1);
    const Index ev = r2.y_left.rows() - 1;   // the turn-off at D·T
    CHECK(r2.y_left(ev, 0) > 1e-3);           // closed: V − v_a over 1/g_on
    CHECK(std::abs(r2.y_right(ev, 0)) < 1e-4);  // open: g_off·(V − v_a)
}

TEST_CASE("Period map: a repeated eigenvalue with a sound basis is exact",
          "[v2][dsed][period_integrals]") {
    // two identical RC branches: A = −1/τ·I, double eigenvalue, V = I
    const Real V = 5.0, R = 2e3, C = 0.5e-6, tau = R * C, T = 2e-3;
    builder::CircuitBuilder b;
    b.add_voltage_source("V1", "in", "gnd", V);
    b.add_resistor("R1", "in", "n1", R);
    b.add_capacitor("C1", "n1", "gnd", C);
    b.add_resistor("R2", "in", "n2", R);
    b.add_capacitor("C2", "n2", "gnd", C);
    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    dsed::NativeCircuitBuilderAdapter<> ad{b.graph(), b.pool(), cache};
    Vector x0(2);
    x0 << 1.0, 3.0;
    const auto r = one_period(ad, Gate{T, 0.5, 0}, x0,
                              {node(b.node_id_of("n1")),
                               node(b.node_id_of("n2"))}, {{0, 1}});
    REQUIRE(r.integrals_ok);
    const Real e = std::exp(-T / tau), e2 = std::exp(-2 * T / tau);
    auto I1 = [&](Real v0) { return V * T + (v0 - V) * tau * (1 - e); };
    // ∫ (V + a e^{-t/τ})(V + c e^{-t/τ})
    auto I12 = [&](Real a, Real c) {
        return V * V * T + V * (a + c) * tau * (1 - e)
               + a * c * tau / 2 * (1 - e2);
    };
    // the state order is the adapter's; match by initial value
    const int i1 = std::abs(r.y_start[0] - 1.0) < 1e-12 ? 0 : 1;
    CHECK(rel(r.stats[static_cast<std::size_t>(i1)].int1, I1(1.0)));
    CHECK(rel(r.stats[static_cast<std::size_t>(1 - i1)].int1, I1(3.0)));
    CHECK(rel(r.cross[0], I12(1.0 - V, 3.0 - V)));
    CHECK(rel(r.monodromy(0, 0), e));
    CHECK(std::abs(r.monodromy(0, 1)) <= 1e-15);
}

TEST_CASE("Period map: a defective basis declines the integrals by name",
          "[v2][dsed][period_integrals]") {
    // critically damped series RLC: R = 2·sqrt(L/C), a double eigenvalue
    // with ONE eigenvector
    const Real L = 1e-3, C = 1e-6, R = 2 * std::sqrt(L / C), T = 5e-4;
    builder::CircuitBuilder b;
    b.add_voltage_source("V1", "in", "gnd", 1.0);
    b.add_resistor("R1", "in", "a", R);
    b.add_inductor("L1", "a", "n1", L);
    b.add_capacitor("C1", "n1", "gnd", C);
    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    dsed::NativeCircuitBuilderAdapter<> ad{b.graph(), b.pool(), cache};
    const auto r = one_period(ad, Gate{T, 0.5, 0}, Vector::Zero(2),
                              {node(b.node_id_of("n1"))}, {});
    CHECK_FALSE(r.integrals_ok);
    CHECK_FALSE(r.integrals_reason.empty());
    // the map itself is still there: x(T) of the critically damped step
    // response v = 1 − (1 + t/τ)e^{−t/τ}, τ = sqrt(LC)
    const Real tau = std::sqrt(L * C);
    const Real v = 1 - (1 + T / tau) * std::exp(-T / tau);
    const Index k = b.node_id_of("n1");
    ad.set_mask(SSM(0));
    const Vector xf = ad.recover_from_state() * r.x_end + ad.recover_const();
    CHECK(std::abs(xf[k] - v) <= 1e-8);
    CHECK(r.monodromy.rows() == 2);
}
