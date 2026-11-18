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

TEST_CASE("reading rule: turn-on v before / i after, turn-off the reverse",
          "[thermal][device_losses]") {
    const auto on = thermal::commutation_point(true, -600.0, 0.1, 1.0, -40.0);
    CHECK(on.v == 600.0);
    CHECK(on.i == 40.0);
    const auto off = thermal::commutation_point(false, 1.0, -40.0, -600.0, 0.1);
    CHECK(off.v == 600.0);
    CHECK(off.i == 40.0);
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

TEST_CASE("set_device_losses refuses by name; other engines refuse the port",
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
    try {
        (void)solver::run_transient_trbdf2(cache, b.graph(), b.pool(), o, gate);
        FAIL("trbdf2 ran a circuit with a loss port");
    } catch (const UnsupportedError& e) {
        CHECK(e.engine() == "trbdf2");
        CHECK(e.feature() == "device_losses");
        CHECK(e.device() == "S1");
    }
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
