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

    // Resume preserves has_e_on / has_e_off on the restored LossTracker.
    auto b = chopper();
    thermal::DeviceLossSpec s;
    s.device = "S1";
    s.tj = 25.0;
    s.e_on = constant(0.0);
    b.set_device_losses(s);
    solver::LossTracker t1(b.graph(), b.pool(), "pwl");
    const auto saved = t1.save();
    solver::LossTracker t2(b.graph(), b.pool(), "pwl");
    t2.restore(saved);
    solver::SimulationResult out;
    t2.write(out);
    REQUIRE(out.device_losses.size() == 1);
    CHECK(out.device_losses[0].has_e_on);
    CHECK_FALSE(out.device_losses[0].has_e_off);
}

TEST_CASE("fit_conduction_linear recovers ohmic and affine V_on(i, Tj) laws",
          "[thermal][device_losses][electrical_feedback]") {
    // Ohmic positive-TC table: V_on(i, T) = 0.1 * (1 + 0.01 * (T - 25)) * i
    const thermal::ConductionTable ohmic(
        {0.0, 10.0, 20.0}, {25.0, 125.0},
        {0.0, 0.0, 1.0, 2.0, 2.0, 4.0});
    const auto sw_75 = thermal::fit_conduction_linear(ohmic, 75.0, false);
    CHECK(sw_75.v_f == 0.0);
    CHECK(near(sw_75.r_on, 0.15, 1e-12));
    CHECK(near(sw_75.g_on, 1.0 / 0.15, 1e-12));

    // PLECS-style diode table: V(0) = 0, V(i) = V_f(T) + R(T)*i on positive knots
    // T=25: V_f=0.8, R=0.05 -> V(20)=1.8, V(40)=2.8
    // T=125: V_f=0.6, R=0.03 -> V(20)=1.2, V(40)=1.8
    const thermal::ConductionTable diode_tab(
        {0.0, 20.0, 40.0}, {25.0, 125.0},
        {0.0, 0.0, 1.8, 1.2, 2.8, 1.8});
    const auto d_25 = thermal::fit_conduction_linear(diode_tab, 25.0, true);
    CHECK(near(d_25.v_f, 0.8, 1e-12));
    CHECK(near(d_25.r_on, 0.05, 1e-12));
    const auto d_125 = thermal::fit_conduction_linear(diode_tab, 125.0, true);
    CHECK(near(d_125.v_f, 0.6, 1e-12));
    CHECK(near(d_125.r_on, 0.03, 1e-12));
}

TEST_CASE("electrical feedback shifts current sharing of paralleled switches with unequal R_th",
          "[thermal][device_losses][electrical_feedback]") {
    auto build_parallel = [](bool feedback, double dT_refresh) {
        builder::CircuitBuilder b;
        b.add_current_source("I", "a", "gnd", 20.0);
        // Tiny sense resistors (1 mOhm) read exact solved branch currents
        b.add_resistor("Rs1", "a", "n1", 1e-3);
        b.add_switch("S1", "n1", "gnd", 10.0, 1e-9);
        b.add_resistor("Rs2", "a", "n2", 1e-3);
        b.add_switch("S2", "n2", "gnd", 10.0, 1e-9);
        b.add_voltage_source("Vamb", "amb", "gnd", 25.0);
        b.add_resistor("Rth1", "tj1", "amb", 1.0);
        b.add_capacitor("Cth1", "tj1", "amb", 5e-3);
        b.add_resistor("Rth2", "tj2", "amb", 5.0);
        b.add_capacitor("Cth2", "tj2", "amb", 1e-3);

        // Positive TC: R_on(T) = 0.1 * (1 + 0.01 * (T - 25)) -> doubles by 125 C
        auto v_on = std::make_shared<const thermal::ConductionTable>(
            std::vector<double>{0.0, 10.0, 20.0},
            std::vector<double>{25.0, 125.0},
            std::vector<double>{0.0, 0.0, 1.0, 2.0, 2.0, 4.0});

        for (const auto& [dev, junc] :
             {std::pair{"S1", "tj1"}, std::pair{"S2", "tj2"}}) {
            thermal::DeviceLossSpec s;
            s.device = dev;
            s.junction = junc;
            s.v_on = v_on;
            s.conduction = thermal::ConductionHeat::Electrical;
            s.electrical_feedback = feedback;
            s.dT_refresh = dT_refresh;
            b.set_device_losses(std::move(s));
        }
        return b;
    };

    auto all_on = [](Real) {
        topology::SwitchStateMask m(2);
        m.set(0, true);
        m.set(1, true);
        return m;
    };

    const Real dt = 1e-4;
    const solver::SimulationOptions opts{.t_start = 0.0, .t_end = 0.03, .dt = dt};

    // Baseline: electrical_feedback = false -> identical currents, 0 refreshes.
    auto b_off = build_parallel(false, 5.0);
    pwl::PwlStateSpaceCache c_off{b_off.graph(), b_off.pool()};
    c_off.build_lazy(dt);
    const auto r_off = solver::run_transient(c_off, b_off.graph(), b_off.pool(), opts, all_on);
    const Index na = node_of(b_off, "a");
    const Index n1 = node_of(b_off, "n1");
    const Index n2 = node_of(b_off, "n2");
    const Index tj1 = node_of(b_off, "tj1");
    const Index tj2 = node_of(b_off, "tj2");
    const Real i1_off = (r_off.states.back()[na] - r_off.states.back()[n1]) / 1e-3;
    const Real i2_off = (r_off.states.back()[na] - r_off.states.back()[n2]) / 1e-3;
    CHECK(std::abs(i1_off - i2_off) < 1e-12);
    REQUIRE(r_off.device_losses.size() == 2);
    CHECK(r_off.device_losses[0].electrical_refreshes == 0);
    CHECK(r_off.device_losses[1].electrical_refreshes == 0);

    // Feedback ON: hotter S2 (higher R_th) shifts to higher R_on -> takes LESS current.
    auto b_on = build_parallel(true, 5.0);
    const Index s1_bid = b_on.branch_id_of("S1");
    const Index s2_bid = b_on.branch_id_of("S2");
    pwl::PwlStateSpaceCache c_on{b_on.graph(), b_on.pool()};
    c_on.build_lazy(dt);
    const auto r_on = solver::run_transient(c_on, b_on.graph(), b_on.pool(), opts, all_on);
    const Real i1_on = (r_on.states.back()[na] - r_on.states.back()[n1]) / 1e-3;
    const Real i2_on = (r_on.states.back()[na] - r_on.states.back()[n2]) / 1e-3;
    CHECK(near(i1_on + i2_on, 20.0, 1e-9));
    CHECK(r_on.states.back()[tj2] > r_on.states.back()[tj1] + 10.0);
    CHECK(i1_on > i2_on + 1.0);
    REQUIRE(r_on.device_losses.size() == 2);
    CHECK(r_on.device_losses[0].electrical_refreshes > 0);
    CHECK(r_on.device_losses[1].electrical_refreshes > 0);
    CHECK(r_on.device_losses[1].electrical_refreshes >= r_on.device_losses[0].electrical_refreshes);
    // Energy conservation still closes to 1e-12
    const Real e_total = r_on.device_losses[0].E_cond_electrical +
                         r_on.device_losses[1].E_cond_electrical;
    CHECK(near(r_on.heat_injected + r_on.heat_pending, e_total, 1e-12));
    // Caller's builder pool is NEVER mutated by run_transient
    CHECK(b_on.pool().switch_g_on(s1_bid) == 10.0);
    CHECK(b_on.pool().switch_g_on(s2_bid) == 10.0);

    // Snapshot / resume preserves electrical feedback state and matches uninterrupted run
    const solver::SimulationOptions o1{.t_start = 0.0, .t_end = 0.015, .dt = dt};
    const solver::SimulationOptions o2{.t_start = 0.015, .t_end = 0.03, .dt = dt};
    const auto seg1 = solver::run_transient(c_on, b_on.graph(), b_on.pool(), o1, all_on);
    const auto seg2 = solver::run_transient(
        c_on, b_on.graph(), b_on.pool(), o2, all_on, {}, false, {}, {}, nullptr,
        {}, &seg1.final_snapshot);
    CHECK(near(seg2.states.back()[tj1], r_on.states.back()[tj1], 1e-11));
    CHECK(near(seg2.states.back()[tj2], r_on.states.back()[tj2], 1e-11));
    CHECK(seg2.device_losses[0].electrical_refreshes == r_on.device_losses[0].electrical_refreshes);
    CHECK(seg2.device_losses[1].electrical_refreshes == r_on.device_losses[1].electrical_refreshes);
}

TEST_CASE("electrical feedback validation and non-pwl engine refusals",
          "[thermal][device_losses][electrical_feedback]") {
    auto b = chopper();
    auto v_on = std::make_shared<const thermal::ConductionTable>(
        std::vector<double>{0.0, 10.0}, std::vector<double>{25.0, 125.0},
        std::vector<double>{0.0, 0.0, 1.0, 2.0});

    // Missing V_on refused
    {
        thermal::DeviceLossSpec s;
        s.device = "S1";
        s.junction = "tj";
        s.electrical_feedback = true;
        CHECK_THROWS_WITH(b.set_device_losses(s),
                          Catch::Matchers::ContainsSubstring("S1") &&
                              Catch::Matchers::ContainsSubstring("V_on"));
    }
    // Fixed Tj refused
    {
        thermal::DeviceLossSpec s;
        s.device = "S1";
        s.tj = 25.0;
        s.v_on = v_on;
        s.electrical_feedback = true;
        CHECK_THROWS_WITH(b.set_device_losses(s),
                          Catch::Matchers::ContainsSubstring("S1") &&
                              Catch::Matchers::ContainsSubstring("junction"));
    }
    // Non-positive dT_refresh refused
    {
        thermal::DeviceLossSpec s;
        s.device = "S1";
        s.junction = "tj";
        s.v_on = v_on;
        s.electrical_feedback = true;
        s.dT_refresh = 0.0;
        CHECK_THROWS_WITH(b.set_device_losses(s),
                          Catch::Matchers::ContainsSubstring("dT_refresh"));
    }
    // trbdf2 refuses electrical_feedback=true with typed UnsupportedError
    {
        thermal::DeviceLossSpec s;
        s.device = "S1";
        s.junction = "tj";
        s.v_on = v_on;
        s.electrical_feedback = true;
        b.set_device_losses(s);
        pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
        cache.build_lazy(1e-6);
        solver::TrBdf2Options o;
        o.t_start = 0.0;
        o.t_end = 1e-4;
        try {
            (void)solver::run_transient_trbdf2(cache, b.graph(), b.pool(), o, gate);
            FAIL("expected UnsupportedError on trbdf2");
        } catch (const UnsupportedError& e) {
            CHECK(e.engine() == "trbdf2");
            CHECK(e.feature() == "electrical_feedback");
            CHECK(e.device() == "S1");
        }
    }
}
