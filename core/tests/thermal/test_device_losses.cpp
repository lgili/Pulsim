// add-in-kernel-event-losses group 2 — the loss port on the fixed-step
// engine (solver/loss_tracker.hpp). The physical checks (open / closed
// loop, conduction, VSI parity) are in python/tests/test_device_losses_
// kernel.py; here: the reading rule itself, the adiabatic deposit, the
// ledger, a dt retry that must not count an event twice, and the
// refusals.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/solver/trbdf2_transient.hpp"
#include "pulsim/thermal/device_losses.hpp"

#include <cmath>
#include <limits>
#include <memory>

using namespace pulsim;

namespace {

bool near(double a, double b, double rel) {
    return std::abs(a - b) <= rel * std::max(std::abs(a), std::abs(b));
}

std::shared_ptr<const thermal::LossTable3D> constant(double e) {
    return std::make_shared<const thermal::LossTable3D>(
        std::vector<double>{100.0}, std::vector<double>{1.0},
        std::vector<double>{25.0}, std::vector<double>{e});
}

// S1 chops 10 V into 10 Ω at 10 kHz (duty 0.5); junction node "tj" with
// C_th = 0.01 J/K and R_th = 1e9 K/W to a 25 V ambient.
builder::CircuitBuilder chopper() {
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", 10.0);
    b.add_switch("S1", "in", "o", 1e6, 1e-12);
    b.add_resistor("RL", "o", "gnd", 10.0);
    b.add_voltage_source("Vamb", "amb", "gnd", 25.0);
    b.add_resistor("Rth", "tj", "amb", 1e9);
    b.add_capacitor("Cth", "tj", "amb", 0.01);
    return b;
}

topology::SwitchStateMask gate(Real t) {
    topology::SwitchStateMask m(1);
    const Real ph = std::fmod(t, 1e-4);
    m.set(0, ph < 5e-5);
    return m;
}

Index node_of(const builder::CircuitBuilder& b, std::string_view n) {
    for (Index i = 0; i < b.graph().num_nodes(); ++i) {
        if (b.graph().node(i).name == n) return i;
    }
    return -1;
}

}  // namespace

namespace {
// the 12.8 leg's switch (1 mOhm on, 1 nS off) and a diode
constexpr thermal::EdgeLaw kSw{true, 1e3, 1e-9};
constexpr thermal::EdgeLaw kDiode{false, 1e3, 1e-9};
}  // namespace

TEST_CASE("reading rule: turn-on v before / i after, turn-off the reverse",
          "[thermal][device_losses]") {
    const auto on = thermal::commutation_point(true, -600.0, 0.1, 1.0, -40.0, kSw, false);
    CHECK(on.v == 600.0);
    CHECK(on.i == 40.0);
    CHECK_FALSE(on.soft);
    const auto off = thermal::commutation_point(false, 1.0, -40.0, -600.0, 0.1, kSw, false);
    CHECK(off.v == 600.0);
    CHECK(off.i == 40.0);
    CHECK_FALSE(off.soft);
}

TEST_CASE("reading rule: a switch's edge with opposite-sign readings is soft",
          "[thermal][device_losses]") {
    // the low switch of a zero-dead-time leg: blocking +600 V, then
    // carrying the load current in reverse (its diode direction)
    const auto on = thermal::commutation_point(true, 600.0, 1e-7, -1e-3, -40.0, kSw, false);
    CHECK(on.soft);
    CHECK(on.v == 0.0);
    CHECK(on.i == 40.0);
    const auto off = thermal::commutation_point(false, -1e-3, -40.0, 600.0, 1e-7, kSw, false);
    CHECK(off.soft);
    CHECK(off.v == 0.0);
    // with a dead time its diode clamps it at -V_f: signs agree, charged
    const auto dt = thermal::commutation_point(true, -0.8, -1e-9, -1e-3, -40.0, kSw, false);
    CHECK_FALSE(dt.soft);
    CHECK(dt.v == 0.8);
    // a diode's recovery (forward i, then reverse v) is never soft
    const auto rr = thermal::commutation_point(false, 0.7, 40.0, -600.0, -1e-7, kDiode, false);
    CHECK_FALSE(rr.soft);
    CHECK(rr.v == 600.0);
    CHECK(rr.i == 40.0);
}

TEST_CASE("reading rule: a reverse current within the device's leakage is not resolved",
          "[thermal][device_losses]") {
    // g_off |v| = 6e-7 A at 600 V; eps g_on |v| = 1.3e-10 A
    const double floor = (1e-9 + std::numeric_limits<double>::epsilon() * 1e3) * 600.0;
    CHECK_FALSE(thermal::commutation_point(true, 600.0, 6e-7, 0.0, -floor, kSw, false).soft);
    CHECK(thermal::commutation_point(true, 600.0, 6e-7, 0.0, -2.0 * floor, kSw, false).soft);
    CHECK_FALSE(thermal::commutation_point(true, 600.0, 6e-7, 0.0, -1e-15, kSw, false).soft);
    CHECK(thermal::commutation_point(true, 600.0, 6e-7, 0.0, -1e-15, kSw, false).v == 600.0);
    // g_off = 0: the rounding of the closed law alone
    constexpr thermal::EdgeLaw ideal{true, 1e3, 0.0};
    CHECK_FALSE(thermal::commutation_point(true, 600.0, 0.0, 0.0, -1e-10, ideal, false).soft);
    CHECK(thermal::commutation_point(true, 600.0, 0.0, 0.0, -1e-9, ideal, false).soft);
    // a branch-form switch (g_on = inf) reads its current as an unknown
    constexpr thermal::EdgeLaw branch{true, std::numeric_limits<double>::infinity(), 1e-9};
    CHECK(thermal::commutation_point(true, 600.0, 0.0, 0.0, -1e-6, branch, false).soft);
    CHECK_FALSE(thermal::commutation_point(true, 600.0, 0.0, 0.0, -5e-7, branch, false).soft);
}

TEST_CASE("reading rule: an impulsive edge and a non-finite reading are never soft",
          "[thermal][device_losses]") {
    // a switch closing a capacitor loop: 8 V before, the jump's -8397 A after
    const auto imp = thermal::commutation_point(true, 8.0, 0.0, 0.0, -8397.0, kSw, true);
    CHECK_FALSE(imp.soft);
    CHECK(imp.v == 8.0);
    const double inf = std::numeric_limits<double>::infinity();
    const auto a = thermal::commutation_point(true, inf, 0.0, 0.0, -40.0, kSw, false);
    CHECK_FALSE(a.soft);
    CHECK(std::isinf(a.v));
    const auto b = thermal::commutation_point(false, 0.0, 40.0, std::nan(""), 0.0, kSw, false);
    CHECK_FALSE(b.soft);
    CHECK(std::isnan(b.v));
}

TEST_CASE("a non-finite reading at an edge is refused by the device's name",
          "[thermal][device_losses]") {
    auto b = chopper();
    thermal::DeviceLossSpec s;
    s.device = "S1";
    s.tj = 25.0;
    s.e_on = constant(1e-3);
    b.set_device_losses(s);
    solver::LossTracker lt(b.graph(), b.pool(), "dsed");
    const Vector x_l = Vector::Zero(b.graph().num_nodes() + 4);
    Vector x_r = x_l;
    x_r[node_of(b, "o")] = std::numeric_limits<Real>::infinity();
    topology::SwitchStateMask m_l(1), m_r(1);
    m_r.set(0, true);
    lt.begin(x_l, m_l);
    // the closed side reads -inf against the 0 V it blocked: never soft
    CHECK_THROWS_WITH(lt.event(1e-6, x_l, m_l, x_r, m_r),
                      Catch::Matchers::ContainsSubstring("'S1'")
                          && Catch::Matchers::ContainsSubstring("non-finite"));
}

TEST_CASE("adiabatic junction: ten 1 mJ events raise it 1.000 K",
          "[thermal][device_losses]") {
    auto b = chopper();
    thermal::DeviceLossSpec s;
    s.device = "S1";
    s.junction = "tj";
    s.e_on = constant(1e-3);
    s.e_off = constant(1e-3);
    b.set_device_losses(s);
    const Real dt = 1e-6;
    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build(dt);
    solver::SimulationOptions opts{.t_start = 0.0, .t_end = 0.52e-3, .dt = dt};
    const auto r = solver::run_transient(cache, b.graph(), b.pool(), opts, gate);
    REQUIRE(r.device_losses.size() == 1);
    const auto& rec = r.device_losses[0];
    REQUIRE(rec.events.size() == 10);
    CHECK(near(rec.E_on + rec.E_off, 10e-3, 1e-12));
    CHECK(near(r.heat_injected + r.heat_pending,
               rec.E_on + rec.E_off + rec.E_cond_electrical, 1e-12));
    const Index tj = node_of(b, "tj");
    const Real rise = r.states.back()[tj] - r.states[1][tj];
    CHECK(near(rise, 1.0, 1e-6));
    // the turn-on reads the 10 V the open switch blocked and the 1 A it
    // then carries (v·g_on is 1 µV); the turn-off the reverse
    for (const auto& e : rec.events) {
        CHECK(near(e.v, 10.0, 1e-5));
        CHECK(near(e.i, 1.0, 1e-5));
        CHECK(e.e == 1e-3);
    }
}

TEST_CASE("a circuit without a loss port carries no record",
          "[thermal][device_losses]") {
    auto b = chopper();
    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build(1e-6);
    solver::SimulationOptions opts{.t_start = 0.0, .t_end = 1e-4, .dt = 1e-6};
    const auto r = solver::run_transient(cache, b.graph(), b.pool(), opts, gate);
    CHECK(r.device_losses.empty());
    CHECK(r.heat_injected == 0.0);
    CHECK(r.heat_pending == 0.0);
}

TEST_CASE("set_device_losses refuses by name; trbdf2 honours the port",
          "[thermal][device_losses]") {
    auto b = chopper();
    thermal::DeviceLossSpec s;
    s.device = "Q9";
    s.tj = 25.0;
    CHECK_THROWS_WITH(b.set_device_losses(s),
                      Catch::Matchers::ContainsSubstring("Q9"));
    s.device = "RL";
    CHECK_THROWS_WITH(b.set_device_losses(s),
                      Catch::Matchers::ContainsSubstring("Resistor"));
    s.device = "S1";
    s.junction = "tj";
    CHECK_THROWS_WITH(b.set_device_losses(s),
                      Catch::Matchers::ContainsSubstring("both junction and Tj"));
    s.tj = std::nan("");
    s.conduction = thermal::ConductionHeat::Table;
    CHECK_THROWS_WITH(b.set_device_losses(s),
                      Catch::Matchers::ContainsSubstring("V_on"));
    s.conduction = thermal::ConductionHeat::Electrical;
    b.set_device_losses(s);
    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build_lazy(1e-6);
    solver::TrBdf2Options o;
    o.t_start = 0.0;
    o.t_end = 1e-4;
    // trbdf2 honours the port (task 3.1): one record, its events booked
    const auto r = solver::run_transient_trbdf2(cache, b.graph(), b.pool(), o, gate);
    REQUIRE(r.device_losses.size() == 1);
    CHECK(r.device_losses[0].device == "S1");
    CHECK(r.device_losses[0].n_on + r.device_losses[0].n_off >= 1);
    CHECK(near(r.heat_injected + r.heat_pending, r.device_losses[0].E_cond_electrical, 1e-12));
}

TEST_CASE("a junction without capacitance is refused when the run starts",
          "[thermal][device_losses]") {
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", 10.0);
    b.add_switch("S1", "in", "o", 1e6, 1e-12);
    b.add_resistor("RL", "o", "gnd", 10.0);
    b.add_capacitor("Co", "o", "gnd", 1e-6);
    b.add_voltage_source("Vamb", "amb", "gnd", 25.0);
    b.add_resistor("Rth", "tj", "amb", 1.0);
    thermal::DeviceLossSpec s;
    s.device = "S1";
    s.junction = "tj";
    b.set_device_losses(s);
    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build(1e-6);
    solver::SimulationOptions opts{.t_start = 0.0, .t_end = 1e-5, .dt = 1e-6};
    CHECK_THROWS_WITH(
        solver::run_transient(cache, b.graph(), b.pool(), opts, gate),
        Catch::Matchers::ContainsSubstring("'S1'") &&
            Catch::Matchers::ContainsSubstring("'tj'") &&
            Catch::Matchers::ContainsSubstring("capacitance"));
}

TEST_CASE("DeviceLossRecord exposes has_e_on and has_e_off for switching tables",
          "[thermal][device_losses]") {
    auto run_case = [](bool with_on, bool with_off, double e_val, bool use_trbdf2) {
        auto b = chopper();
        thermal::DeviceLossSpec s;
        s.device = "S1";
        s.tj = 25.0;
        if (with_on) s.e_on = constant(e_val);
        if (with_off) s.e_off = constant(e_val);
        b.set_device_losses(s);
        pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
        if (use_trbdf2) {
            cache.build_lazy(1e-6);
            solver::TrBdf2Options o;
            o.t_start = 0.0;
            o.t_end = 1.2e-4;
            return solver::run_transient_trbdf2(cache, b.graph(), b.pool(), o, gate);
        }
        cache.build(1e-6);
        solver::SimulationOptions opts{.t_start = 0.0, .t_end = 1.2e-4, .dt = 1e-6};
        return solver::run_transient(cache, b.graph(), b.pool(), opts, gate);
    };

    for (const bool use_trbdf2 : {false, true}) {
        // Zero-valued switching tables still mark has_e_on / has_e_off true.
        const auto zero_both = run_case(true, true, 0.0, use_trbdf2);
        REQUIRE(zero_both.device_losses.size() == 1);
        CHECK(zero_both.device_losses[0].has_e_on);
        CHECK(zero_both.device_losses[0].has_e_off);
        CHECK(zero_both.device_losses[0].E_on == 0.0);
        CHECK(zero_both.device_losses[0].E_off == 0.0);

        const auto on_only = run_case(true, false, 1e-3, use_trbdf2);
        REQUIRE(on_only.device_losses.size() == 1);
        CHECK(on_only.device_losses[0].has_e_on);
        CHECK_FALSE(on_only.device_losses[0].has_e_off);

        const auto off_only = run_case(false, true, 1e-3, use_trbdf2);
        REQUIRE(off_only.device_losses.size() == 1);
        CHECK_FALSE(off_only.device_losses[0].has_e_on);
        CHECK(off_only.device_losses[0].has_e_off);

        // Conduction-only port has both flags false.
        const auto cond_only = run_case(false, false, 0.0, use_trbdf2);
        REQUIRE(cond_only.device_losses.size() == 1);
        CHECK_FALSE(cond_only.device_losses[0].has_e_on);
        CHECK_FALSE(cond_only.device_losses[0].has_e_off);
    }
}
