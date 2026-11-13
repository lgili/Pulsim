// =============================================================================
// add-power-mosfet-datasheet-device — ChargeTable and the charge-law variant
// of NonlinearCapacitor (tasks 1.1–1.5)
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "pulsim/models/charge_table.hpp"
#include "pulsim/models/nonlinear_capacitor.hpp"

#include <cmath>
#include <memory>
#include <vector>

using namespace pulsim;
using models::ChargeTable;
using models::NonlinearCapacitor;

namespace {
const auto kJunction = NonlinearCapacitor::junction(2e-9, 25.0, 0.5);

/// The junction law at 64 knots on [0, 400] V (quadratic spacing) with
/// exact slopes.
ChargeTable junction_table() {
    std::vector<Real> v, q, c;
    for (int k = 0; k < 64; ++k) {
        const Real u = k / 63.0;
        const Real vk = 400.0 * u * u;
        v.push_back(vk);
        q.push_back(NonlinearCapacitor::charge(kJunction, vk));
        c.push_back(NonlinearCapacitor::capacitance(kJunction, vk));
    }
    return ChargeTable(v, q, c, "junction table");
}

/// Largest |C(v) − (Q(v+δ) − Q(v−δ))/2δ| / C(v) over a sweep.
template <class Q, class C>
Real max_fd_error(Q&& charge, C&& cap, Real lo, Real hi, int n) {
    Real worst = 0;
    for (int k = 0; k <= n; ++k) {
        const Real v = lo + (hi - lo) * k / n;
        const Real d = 1e-6 * std::max(Real{1}, std::abs(v));
        const Real fd = (charge(v + d) - charge(v - d)) / (2 * d);
        worst = std::max(worst, std::abs(cap(v) - fd) / cap(v));
    }
    return worst;
}

/// Q strictly increasing at 1 pV spacing over ±1000 pV around v0.
template <class Q>
bool monotone_at_picovolts(Q&& charge, Real v0) {
    Real prev = charge(v0 - 1000e-12);
    for (int k = -999; k <= 1000; ++k) {
        const Real q = charge(v0 + k * 1e-12);
        if (!(q > prev)) return false;
        prev = q;
    }
    return true;
}
}  // namespace

TEST_CASE("ChargeTable: reproduces the junction law (task 1.2)",
          "[mosfet][charge_table][unit]") {
    const auto tab = junction_table();
    // Spec scenario: Q(400 V) = 312.31 nC within 1e-4, C = 485.07 pF within 1e-3.
    CHECK(std::abs(tab.charge(400.0) - 312.31e-9) / 312.31e-9 < 1e-4);
    CHECK(std::abs(tab.capacitance(400.0) - 485.07e-12) / 485.07e-12 < 1e-3);
    // Between knots, against the closed form (the knots are exact by
    // construction, so the meaningful check is off-knot).
    Real eq = 0, ec = 0;
    for (int k = 1; k < 4000; ++k) {
        const Real v = 400.0 * k / 4000.0;
        const Real qa = NonlinearCapacitor::charge(kJunction, v);
        const Real ca = NonlinearCapacitor::capacitance(kJunction, v);
        eq = std::max(eq, std::abs(tab.charge(v) - qa) / qa);
        ec = std::max(ec, std::abs(tab.capacitance(v) - ca) / ca);
    }
    INFO("max rel err off-knot: Q " << eq << "  C " << ec);
    CHECK(eq < 1e-4);
    CHECK(ec < 1e-3);
    CHECK(tab.clamped_tangents() == 0);
}

TEST_CASE("ChargeTable: C(v) is the exact derivative of Q(v) (task 1.2)",
          "[mosfet][charge_table][unit]") {
    const auto tab = junction_table();
    const Real e = max_fd_error([&](Real v) { return tab.charge(v); },
                                [&](Real v) { return tab.capacitance(v); },
                                -50.0, 450.0, 5003);
    INFO("max |C - dQ/dv| / C = " << e);
    CHECK(e < 1e-7);
}

TEST_CASE("ChargeTable: continuous Q and C across knots and tails",
          "[mosfet][charge_table][unit]") {
    const auto tab = junction_table();
    for (Real vk : tab.v_knots()) {
        const Real eps = 1e-9 * std::max(Real{1}, vk);
        CHECK(std::abs(tab.charge(vk + eps) - tab.charge(vk - eps))
              < 2.0 * eps * tab.capacitance(vk) * 1.0001);
        CHECK(std::abs(tab.capacitance(vk + eps) - tab.capacitance(vk - eps))
              / tab.capacitance(vk) < 1e-6);
    }
}

TEST_CASE("ChargeTable: Q strictly increasing at 1 pV spacing (task 1.3)",
          "[mosfet][charge_table][unit]") {
    const auto tab = junction_table();
    for (Real v0 : {0.0, 1.0, 400.0}) {
        INFO("around " << v0 << " V");
        CHECK(monotone_at_picovolts([&](Real v) { return tab.charge(v); }, v0));
    }
}

TEST_CASE("ChargeTable: refusals name the table and the knot (task 1.1)",
          "[mosfet][charge_table][unit]") {
    using Catch::Matchers::ContainsSubstring;
    CHECK_THROWS_WITH(ChargeTable({0.0, 1.0}, {0.0, 1e-9}, {1e-9, 1e-9}, "M1.cgd"),
                      ContainsSubstring("M1.cgd") && ContainsSubstring("at least 3"));
    CHECK_THROWS_WITH(ChargeTable({1.0, 2.0, 3.0}, {1e-9, 2e-9, 3e-9},
                                  {1e-9, 1e-9, 1e-9}, "M1.cgd"),
                      ContainsSubstring("v = 0"));
    CHECK_THROWS_WITH(ChargeTable({0.0, 2.0, 1.0}, {0.0, 2e-9, 3e-9},
                                  {1e-9, 1e-9, 1e-9}, "M1.cgd"),
                      ContainsSubstring("strictly increasing"));
    CHECK_THROWS_WITH(ChargeTable({0.0, 1.0, 2.0}, {0.0, 1e-9, 2e-9},
                                  {1e-9, 0.0, 1e-9}, "M1.cgd"),
                      ContainsSubstring("M1.cgd") && ContainsSubstring("knot 1"));
    CHECK_THROWS_WITH(ChargeTable::from_capacitance({0.0, 1.0, 2.0},
                                                    {1e-9, -1e-12, 1e-9}, "M1.cds"),
                      ContainsSubstring("M1.cds") && ContainsSubstring("knot 1"));
}

TEST_CASE("ChargeTable: from (v, C) pairs integrates the curve and is not odd",
          "[mosfet][charge_table][unit]") {
    // C falls two decades from 0 to 100 V; also covers negative v.
    std::vector<Real> v, c;
    for (int k = -4; k <= 40; ++k) {
        const Real vk = 2.5 * k;
        v.push_back(vk);
        c.push_back(1e-9 / (1.0 + std::max(vk, -8.0) / 1.0) + 1e-11);
    }
    const auto tab = ChargeTable::from_capacitance(v, c, "pairs");
    CHECK(tab.charge(0.0) == 0.0);
    CHECK(tab.charge(-5.0) < 0.0);
    CHECK(tab.charge(5.0) > 0.0);
    CHECK(std::abs(tab.charge(5.0) + tab.charge(-5.0)) > 1e-12);  // not odd
    for (std::size_t k = 0; k < v.size(); ++k) {
        CHECK(std::abs(tab.capacitance(v[k]) - tab.c_knots()[k]) <= 1e-24);
    }
    const Real e = max_fd_error([&](Real x) { return tab.charge(x); },
                                [&](Real x) { return tab.capacitance(x); },
                                -12.0, 110.0, 3001);
    CHECK(e < 1e-6);
}

// ---- vdmos_cgd (task 1.4) ---------------------------------------------------

TEST_CASE("vdmos_cgd: matches ngspice 45.2 VDMOS at ten bias points",
          "[mosfet][vdmos_cgd][unit]") {
    // probes/cgd.cir with `set numdgt=12`: CGDMAX = 1n, CGDMIN = 50p,
    // A = 0.5, Vg = 0, Vd swept; C_gd = -imag(i(Vg))/(2π·1 kHz), so
    // v_gd = −V_d.
    const auto p = NonlinearCapacitor::vdmos_cgd(1e-9, 50e-12, 0.5);
    const Real vd[] = {-5, -2, -1, 0, 1, 2, 5, 20, 100, 400};
    const Real ng[] = {9.95053510609e-10, 9.11900624145e-10, 8.01233300640e-10,
                       6.30464696834e-10, 4.59130537080e-10, 3.40232348417e-10,
                       1.90610539423e-10, 8.68310857145e-11, 5.73897208723e-11,
                       5.18476611188e-11};
    for (int k = 0; k < 10; ++k) {
        const Real c = NonlinearCapacitor::capacitance(p, -vd[k]);
        INFO("v_gd = " << -vd[k] << "  C = " << c << "  ngspice " << ng[k]);
        CHECK(std::abs(c - ng[k]) / ng[k] < 1e-6);
    }
}

TEST_CASE("vdmos_cgd: C = dQ/dv, Q monotone at picovolts, limits",
          "[mosfet][vdmos_cgd][unit]") {
    const auto p = NonlinearCapacitor::vdmos_cgd(1e-9, 50e-12, 0.5);
    auto Q = [&](Real u) { return NonlinearCapacitor::charge(p, u); };
    auto C = [&](Real u) { return NonlinearCapacitor::capacitance(p, u); };
    const Real e = max_fd_error(Q, C, -400.0, 40.0, 4401);
    INFO("max |C - dQ/dv| / C = " << e);
    CHECK(e < 1e-9);
    CHECK(Q(0.0) == 0.0);
    for (Real u0 : {-400.0, -1.0, 0.0, 1.0, 400.0}) {
        INFO("around " << u0 << " V");
        CHECK(monotone_at_picovolts(Q, u0));
    }
    // C¹ at 0 and the two asymptotes; ln cosh holds at large |a·u|.
    CHECK(std::abs(C(1e-12) - C(-1e-12)) / C(0.0) < 1e-12);
    CHECK(std::abs(C(1e4) - 1e-9) / 1e-9 < 1e-12);
    CHECK(std::abs(C(-1e9) - 50e-12) / 50e-12 < 1e-6);
    CHECK(std::isfinite(Q(5e3)));
    CHECK(std::abs((Q(2000.0) - Q(1000.0)) / 1000.0 - 1e-9) / 1e-9 < 1e-12);
}

TEST_CASE("NonlinearCapacitor: junction law unchanged, refusals by law (1.5)",
          "[mosfet][nonlinear_capacitor][unit]") {
    // The default-constructed Params is the junction law, and the junction
    // branch is the pre-variant arithmetic: compare bit-for-bit.
    NonlinearCapacitor::Params legacy{.C0 = 2e-9, .V0 = 25.0, .m = 0.5, .v_floor = -0.9};
    for (Real v : {-30.0, -1.0, 0.0, 3.3, 400.0}) {
        const Real u = std::max(v / 25.0, -0.9);
        CHECK(NonlinearCapacitor::capacitance(legacy, v) == 2e-9 / std::pow(1.0 + u, 0.5));
        CHECK(NonlinearCapacitor::charge(legacy, v)
              == 2e-9 * 25.0 / 0.5 * (std::pow(1.0 + u, 0.5) - 1.0));
    }
    CHECK_THROWS(NonlinearCapacitor::validate(NonlinearCapacitor::vdmos_cgd(1e-9, -1e-12, 0.5)));
    CHECK_NOTHROW(NonlinearCapacitor::validate(NonlinearCapacitor::vdmos_cgd(1e-9, 0.0, 0.5)));
    CHECK(NonlinearCapacitor::capacitance(NonlinearCapacitor::vdmos_cgd(1e-9, 0.0, 0.5), -1e3) > 0.0);
    CHECK_THROWS(NonlinearCapacitor::validate(NonlinearCapacitor::vdmos_cgd(1e-12, 1e-9, 0.5)));
    CHECK_THROWS(NonlinearCapacitor::validate(NonlinearCapacitor::vdmos_cgd(1e-9, 5e-11, 0.0)));
    CHECK_THROWS(NonlinearCapacitor::validate(NonlinearCapacitor::from_table(nullptr)));
    const auto t = NonlinearCapacitor::from_table(
        std::make_shared<const ChargeTable>(junction_table()));
    CHECK_NOTHROW(NonlinearCapacitor::validate(t));
    CHECK(NonlinearCapacitor::charge(t, 400.0) == t.table->charge(400.0));
    CHECK(std::string{NonlinearCapacitor::law_name(t)} == "table");
}
