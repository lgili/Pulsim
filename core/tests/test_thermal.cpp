#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "pulsim/thermal.hpp"
#include <cmath>

using namespace pulsim;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("Simple thermal model steady-state", "[thermal]") {
    ThermalSimulator sim;

    ThermalModel model;
    model.device_name = "M1";
    model.type = ThermalNetworkType::Simple;
    model.rth_jc = 1.0;   // 1 K/W
    model.rth_cs = 0.5;   // 0.5 K/W
    model.rth_sa = 1.5;   // 1.5 K/W
    // Total Rth_ja = 3.0 K/W

    sim.add_model(model);
    sim.set_ambient(25.0);
    sim.initialize();

    // Apply 10W power
    std::unordered_map<std::string, Real> powers;
    powers["M1"] = 10.0;

    // Step for a long time (steady state)
    for (int i = 0; i < 100; ++i) {
        sim.step(0.1, powers);
    }

    // Tj = Tamb + P * Rth_ja = 25 + 10 * 3 = 55C
    CHECK_THAT(sim.junction_temp("M1"), WithinAbs(55.0, 0.5));
}

TEST_CASE("Foster network transient response", "[thermal]") {
    ThermalSimulator sim;

    // Create 2-stage Foster network
    ThermalModel model;
    model.device_name = "M1";
    model.type = ThermalNetworkType::Foster;
    model.rth_cs = 0.0;
    model.rth_sa = 0.0;

    // Stage 1: Rth=1, tau=0.1s
    ThermalRCStage stage1;
    stage1.rth = 1.0;
    stage1.cth = 0.1;  // tau = 0.1s
    model.foster.stages.push_back(stage1);

    // Stage 2: Rth=2, tau=1.0s
    ThermalRCStage stage2;
    stage2.rth = 2.0;
    stage2.cth = 0.5;  // tau = 1.0s
    model.foster.stages.push_back(stage2);

    sim.add_model(model);
    sim.set_ambient(25.0);
    sim.initialize();

    // Apply step power
    std::unordered_map<std::string, Real> powers;
    powers["M1"] = 10.0;

    // At t=0, Tj should be ambient
    CHECK_THAT(sim.junction_temp("M1"), WithinAbs(25.0, 0.1));

    // After 0.1s (1 tau for stage 1), stage 1 should be ~63% of final
    Real dt = 0.001;
    for (int i = 0; i < 100; ++i) {
        sim.step(dt, powers);
    }
    // After 0.1s: stage1 at ~63% of 10W*1K/W = 6.3C, stage2 at ~10% of 10W*2K/W = 2C
    // Expected Tj ~ 25 + 6.3 + 2 = 33.3C (approximately)
    Real tj_100ms = sim.junction_temp("M1");
    CHECK(tj_100ms > 30.0);
    CHECK(tj_100ms < 40.0);

    // After 10s (10 tau for both stages), should reach steady state
    for (int i = 0; i < 10000; ++i) {
        sim.step(dt, powers);
    }
    // Steady state: Tj = 25 + 10*(1+2) = 55C
    CHECK_THAT(sim.junction_temp("M1"), WithinAbs(55.0, 1.0));
}

TEST_CASE("Thermal impedance calculation", "[thermal]") {
    FosterNetwork network;

    ThermalRCStage stage1{1.0, 1.0};  // Rth=1, Cth=1, tau=1
    ThermalRCStage stage2{2.0, 1.0};  // Rth=2, Cth=1, tau=2
    network.stages.push_back(stage1);
    network.stages.push_back(stage2);

    // Total Rth = 3 K/W
    CHECK_THAT(network.rth_total(), WithinAbs(3.0, 0.01));

    // Zth at t=0 should be 0
    CHECK_THAT(network.zth(0.0), WithinAbs(0.0, 0.01));

    // Zth at t=infinity should equal Rth_total
    CHECK_THAT(network.zth(100.0), WithinAbs(3.0, 0.01));

    // Zth formula: sum(Ri * (1 - exp(-t/taui)))
    // At t=1s: Z1 = 1*(1-exp(-1)) = 0.632, Z2 = 2*(1-exp(-0.5)) = 0.787
    Real zth_1s = network.zth(1.0);
    Real expected = 1.0 * (1.0 - std::exp(-1.0)) + 2.0 * (1.0 - std::exp(-0.5));
    CHECK_THAT(zth_1s, WithinAbs(expected, 0.01));
}

TEST_CASE("Temperature-adjusted parameters", "[thermal]") {
    ThermalSimulator sim;

    // Rds_on typically increases with temperature
    Real rds_25c = 0.010;  // 10 mOhm at 25C
    Real tc_rds = 0.004;   // +0.4%/C typical

    // At 125C (100C above 25C)
    Real rds_125c = sim.adjust_rds_on(rds_25c, 125.0, tc_rds);
    // Expected: 0.010 * (1 + 0.004 * 100) = 0.010 * 1.4 = 0.014
    CHECK_THAT(rds_125c, WithinAbs(0.014, 0.0001));

    // Vth typically decreases with temperature
    Real vth_25c = 3.0;    // 3V at 25C
    Real tc_vth = -0.003;  // -3mV/C typical

    // At 125C
    Real vth_125c = sim.adjust_vth(vth_25c, 125.0, tc_vth);
    // Expected: 3.0 - 0.003 * 100 = 2.7V
    CHECK_THAT(vth_125c, WithinAbs(2.7, 0.01));
}

TEST_CASE("Thermal warning detection", "[thermal]") {
    ThermalSimulator sim;

    ThermalModel model;
    model.device_name = "M1";
    model.type = ThermalNetworkType::Simple;
    model.rth_jc = 5.0;
    model.tj_warn = 100.0;
    model.tj_max = 150.0;

    sim.add_model(model);
    sim.set_ambient(25.0);
    sim.initialize();

    std::unordered_map<std::string, Real> powers;
    powers["M1"] = 20.0;  // 20W -> Tj = 25 + 20*5 = 125C (above warning)

    sim.step(0.001, powers);

    const auto& warnings = sim.warnings();
    REQUIRE(warnings.size() >= 1);
    CHECK(warnings[0].device_name == "M1");
    CHECK(warnings[0].temperature > 100.0);
    CHECK(!warnings[0].is_failure);

    // Increase power to exceed max
    powers["M1"] = 30.0;  // 30W -> Tj = 25 + 30*5 = 175C (above max)
    sim.step(0.001, powers);

    REQUIRE(sim.warnings().size() >= 2);
    bool found_failure = false;
    for (const auto& w : sim.warnings()) {
        if (w.is_failure) found_failure = true;
    }
    CHECK(found_failure);
}

TEST_CASE("Create MOSFET thermal model helper", "[thermal]") {
    auto model = create_mosfet_thermal("Q1", 1.0, 0.5, 2.0);

    CHECK(model.device_name == "Q1");
    CHECK(model.type == ThermalNetworkType::Foster);
    CHECK(model.foster.stages.size() == 4);
    CHECK_THAT(model.foster.rth_total(), WithinAbs(1.0, 0.01));
    CHECK_THAT(model.rth_cs, WithinAbs(0.5, 0.01));
    CHECK_THAT(model.rth_sa, WithinAbs(2.0, 0.01));
    CHECK_THAT(model.rth_ja(), WithinAbs(3.5, 0.01));
}

TEST_CASE("Peak temperature tracking", "[thermal]") {
    ThermalSimulator sim;

    ThermalModel model;
    model.device_name = "M1";
    model.type = ThermalNetworkType::Simple;
    model.rth_jc = 1.0;

    sim.add_model(model);
    sim.set_ambient(25.0);
    sim.initialize();

    std::unordered_map<std::string, Real> powers;

    // Apply 100W pulse for 10ms
    powers["M1"] = 100.0;
    for (int i = 0; i < 10; ++i) {
        sim.step(0.001, powers);
    }

    Real tj_peak_during = sim.states()[0].tj_peak;

    // Remove power
    powers["M1"] = 0.0;
    for (int i = 0; i < 100; ++i) {
        sim.step(0.001, powers);
    }

    // Peak should be recorded from when power was applied
    CHECK(sim.states()[0].tj_peak == tj_peak_during);
    CHECK(sim.states()[0].tj_peak > 100.0);  // Should have gotten hot
}
