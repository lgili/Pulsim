// =============================================================================
// Layer 5 V2 — SwitchedDiode PSIM-style forward drop V_f, end to end
// =============================================================================
//
// ON is g_on in series with V_f (v = V_f + i/g_on), OFF is plain g_off.
// The drop is a per-mask constant in b: J and C are untouched, the ON
// mask's b moves by ∓g_on·V_f at anode / cathode, the OFF mask's b is
// bitwise unchanged, and dc_assemble's source_scale leaves it alone.
// Then every engine (fixed pwl, TR-BDF2, DC operating point) must land
// 10 A through g_on = 20 S, V_f = 0.7 V at exactly 1.2 V, and TR-BDF2
// must localise a half-wave's turn-on at asin(V_f/V̂)/ω and its turn-off
// at the current zero (v = V_f), not at v = 0.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/pwl/assemble.hpp"
#include "pulsim/pwl/dc_assemble.hpp"
#include "pulsim/pwl/dc_operating_point.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/solver/trbdf2_transient.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

using namespace pulsim;
using Catch::Approx;

namespace {

constexpr Real kGon = 20.0;
constexpr Real kGoff = 1e-9;
constexpr Real kVf = 0.7;

bool bitwise_equal(const Vector& a, const Vector& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(),
                       sizeof(Real) * static_cast<std::size_t>(a.size())) == 0;
}

bool dense_equal(const sparse::Matrix& a, const sparse::Matrix& b) {
    return Eigen::MatrixXd(a) == Eigen::MatrixXd(b);
}

// V source → a ─D→ k ─R→ gnd, with either diode terminal optionally
// grounded instead.
enum class Wiring { Floating, CathodeGrounded, AnodeGrounded };

builder::CircuitBuilder make(Wiring w, Real V_f) {
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "s", "gnd", 5.0);
    b.add_resistor("Rs", "s", "a", 1.0);
    b.add_resistor("Rk", "k", "gnd", 10.0);
    const char* an = w == Wiring::AnodeGrounded ? "gnd" : "a";
    const char* ca = w == Wiring::CathodeGrounded ? "gnd" : "k";
    b.add_diode("D", an, ca, kGon, kGoff, 0.0, V_f);
    return b;
}

topology::SwitchStateMask diode_mask(const builder::CircuitBuilder& b,
                                     bool on) {
    topology::SwitchStateMask m(b.graph().num_switches());
    m.set(static_cast<Size>(b.switch_index_of("D")), on);
    return m;
}

}  // namespace

TEST_CASE("Forward drop: J and C unchanged, b moves by ∓g_on·V_f on the "
          "ON mask only",
          "[v2][layer5_v2][forward_drop][stamp]") {
    for (const Wiring w : {Wiring::Floating, Wiring::CathodeGrounded,
                           Wiring::AnodeGrounded}) {
        INFO("wiring=" << static_cast<int>(w));
        auto b0 = make(w, 0.0);
        auto b1 = make(w, kVf);
        const Index a = w == Wiring::AnodeGrounded ? Index{-1}
                                                   : b1.node_id_of("a");
        const Index k = w == Wiring::CathodeGrounded ? Index{-1}
                                                     : b1.node_id_of("k");
        for (const bool on : {false, true}) {
            INFO("on=" << on);
            sparse::Matrix G0, C0, G1, C1;
            Vector bb0, bb1;
            pwl::assemble_segment_split(b0.graph(), b0.pool(),
                                        diode_mask(b0, on), G0, C0, bb0);
            pwl::assemble_segment_split(b1.graph(), b1.pool(),
                                        diode_mask(b1, on), G1, C1, bb1);
            REQUIRE(dense_equal(G0, G1));
            REQUIRE(dense_equal(C0, C1));
            if (!on) {
                REQUIRE(bitwise_equal(bb0, bb1));
                continue;
            }
            Vector expect = bb0;
            if (a >= 0) expect[a] -= kGon * kVf;
            if (k >= 0) expect[k] += kGon * kVf;
            REQUIRE(bitwise_equal(expect, bb1));
        }
    }
}

TEST_CASE("Forward drop: dc_assemble does not source-scale the drop",
          "[v2][layer5_v2][forward_drop][stamp][dc_op]") {
    auto b0 = make(Wiring::Floating, 0.0);
    auto b1 = make(Wiring::Floating, kVf);
    const Index a = b1.node_id_of("a");
    const Index k = b1.node_id_of("k");
    for (const Real scale : {1.0, 0.5, 0.0}) {
        INFO("source_scale=" << scale);
        sparse::Matrix J0, J1;
        Vector bb0, bb1;
        pwl::dc_assemble(b0.graph(), b0.pool(), diode_mask(b0, true), J0,
                         bb0, 0.0, 0.0, scale);
        pwl::dc_assemble(b1.graph(), b1.pool(), diode_mask(b1, true), J1,
                         bb1, 0.0, 0.0, scale);
        REQUIRE(dense_equal(J0, J1));
        Vector expect = bb0;
        expect[a] -= kGon * kVf;
        expect[k] += kGon * kVf;
        REQUIRE(bitwise_equal(expect, bb1));
        // OFF mask: identical.
        pwl::dc_assemble(b0.graph(), b0.pool(), diode_mask(b0, false), J0,
                         bb0, 0.0, 0.0, scale);
        pwl::dc_assemble(b1.graph(), b1.pool(), diode_mask(b1, false), J1,
                         bb1, 0.0, 0.0, scale);
        REQUIRE(bitwise_equal(bb0, bb1));
    }
}

TEST_CASE("Forward drop: add_diode refuses a negative or non-finite V_f "
          "by name",
          "[v2][layer5_v2][forward_drop][builder]") {
    for (const Real bad : {-0.1, std::nan(""), std::numeric_limits<Real>::infinity()}) {
        builder::CircuitBuilder b;
        try {
            b.add_diode("Dbad", "a", "gnd", kGon, kGoff, 0.0, bad);
            FAIL("accepted V_f = " << bad);
        } catch (const std::invalid_argument& e) {
            REQUIRE(std::string(e.what()).find("Dbad") != std::string::npos);
            REQUIRE(std::string(e.what()).find("V_f") != std::string::npos);
        }
        // Nothing half-registered.
        REQUIRE(b.graph().num_branches() == 0);
    }
}

// 10 A forced into the anode of a grounded-cathode diode: v = V_f + i/g_on.
TEST_CASE("Forward drop: 10 A through g_on = 20 S, V_f = 0.7 V sits at "
          "1.2 V on pwl, TR-BDF2 and the DC operating point",
          "[v2][layer5_v2][forward_drop][engines]") {
    builder::CircuitBuilder b;
    b.add_current_source("I", "a", "gnd", 10.0);
    b.add_diode("D", "a", "gnd", kGon, kGoff, 0.0, kVf);
    const Index a = b.node_id_of("a");
    const Size n_sw = b.graph().num_switches();
    const auto open = [n_sw](Real) { return topology::SwitchStateMask(n_sw); };
    constexpr Real v_ref = 1.2;

    SECTION("DC operating point") {
        pwl::DiodeEventState diodes(b.graph(), b.pool());
        const auto op = pwl::compute_dc_operating_point(
            b.graph(), b.pool(), topology::SwitchStateMask(n_sw), {}, {},
            &diodes);
        REQUIRE(op.x[a] == Approx(v_ref).epsilon(1e-12));
        REQUIRE(diodes.current_diode_mask().get(0));
    }
    SECTION("fixed pwl") {
        constexpr Real dt = 1e-6;
        pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
        cache.build(dt);
        solver::SimulationOptions o{.t_start = 0.0, .t_end = 2e-5, .dt = dt};
        const auto r = solver::run_transient(cache, b.graph(), b.pool(), o,
                                             open);
        for (Size k = 1; k < r.times.size(); ++k) {
            INFO("k=" << k);
            REQUIRE(r.states[k][a] == Approx(v_ref).epsilon(1e-12));
        }
    }
    SECTION("TR-BDF2") {
        pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
        cache.build_lazy(1e-6);
        solver::TrBdf2Options o;
        o.t_end = 2e-5;
        const auto r = solver::run_transient_trbdf2(cache, b.graph(),
                                                    b.pool(), o, open);
        REQUIRE(r.states.back()[a] == Approx(v_ref).epsilon(1e-12));
    }
}

// Resistive half-wave: 10 V, 50 Hz, R = 10 Ω, g_on = 20 S, V_f = 0.7 V.
// ON: v_out = (v_s − V_f)·R/(R + 1/g_on); events at v_s = V_f on the way
// up (asin(V_f/V̂)/ω) AND on the way down (π − asin)/ω — the old
// V_th-only law would turn off at v_s = 0 (10 ms).
TEST_CASE("Forward drop: TR-BDF2 half-wave commutates at v_s = V_f both "
          "ways",
          "[v2][layer5_v2][forward_drop][engines][trbdf2]") {
    constexpr Real Vhat = 10.0, f = 50.0, R = 10.0;
    const Real w = 2.0 * std::numbers::pi * f;
    builder::CircuitBuilder b;
    b.add_sine_voltage_source("Vac", "s", "gnd", 0.0, Vhat, f, 0.0);
    b.add_diode("D", "s", "out", kGon, kGoff, 0.0, kVf);
    b.add_resistor("R", "out", "gnd", R);
    const Index out = b.node_id_of("out");
    const Size bit = static_cast<Size>(b.switch_index_of("D"));
    const Size n_sw = b.graph().num_switches();

    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build_lazy(1e-6);
    solver::TrBdf2Options o;
    o.t_end = 15e-3;
    o.rtol = 1e-7;
    o.atol = 1e-10;
    o.h_max = 1e-5;
    const auto r = solver::run_transient_trbdf2(
        cache, b.graph(), b.pool(), o,
        [n_sw](Real) { return topology::SwitchStateMask(n_sw); });

    // A landing is recorded under the PRE-commutation mask, so the event
    // time is the last sample before the recorded bit changes.
    std::vector<std::pair<Real, bool>> events;
    for (Size k = 0; k + 1 < r.times.size(); ++k) {
        const bool now = r.switch_state(k, bit);
        const bool next = r.switch_state(k + 1, bit);
        if (now != next) events.emplace_back(r.times[k], next);
    }
    REQUIRE(events.size() == 2);
    const Real t_on = std::asin(kVf / Vhat) / w;            // 222.999 µs
    const Real t_off = (std::numbers::pi - std::asin(kVf / Vhat)) / w;
    CHECK(events[0].second);
    CHECK(events[0].first == Approx(t_on).margin(1e-8));
    CHECK_FALSE(events[1].second);
    CHECK(events[1].first == Approx(t_off).margin(1e-8));

    Real i_min = 0.0, worst = 0.0;
    for (Size k = 0; k < r.times.size(); ++k) {
        const Real v_s = Vhat * std::sin(w * r.times[k]);
        const Real v_o = r.states[k][out];
        const bool on = r.switch_state(k, bit);
        const Real i_d = on ? kGon * (v_s - v_o - kVf) : kGoff * (v_s - v_o);
        i_min = std::min(i_min, i_d);
        if (on) {
            worst = std::max(worst, std::abs(v_o - (v_s - kVf) * R /
                                                       (R + 1.0 / kGon)));
        }
    }
    // The turn-off landing sits a localisation residual (~1e-12 s) past
    // the zero, under the pre-commutation (ON) mask: g_on·V̂ω·1e-12 is
    // tens of nA. A decision that ignored V_f would conduct backwards
    // up to g_on·V_f = 14 A.
    REQUIRE(i_min >= -1e-6);
    REQUIRE(worst < 1e-9);
}
