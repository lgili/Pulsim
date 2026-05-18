// HysteresisInductorDevice — Catch2 smoke for Item 1 of the
// deferred-items follow-up. Confirms the device registers in
// `Circuit::DeviceVariant`, stamps as a linear inductor with
// `L_eff` derived from geometry, and advances Jiles-Atherton
// hysteresis state per accepted step (telemetry-only — the
// nonlinear feedback is a Phase-2 follow-up).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/components/hysteresis_inductor_device.hpp"
#include "pulsim/v1/simulation.hpp"

#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;

TEST_CASE("HysteresisInductorDevice: default L_eff derived from geometry",
          "[item1][hysteresis][device]") {
    HysteresisInductorDevice::Params params{};
    params.geom.turns = 100.0;
    params.geom.area = 1e-4;
    params.geom.path_length = 5e-2;
    params.geom.path_length = 5e-2;
    // Leave L_eff at 0 → device derives a default from geometry.

    HysteresisInductorDevice dev(params, "H1");
    // L = N² · μ_0 · μ_r_init · A_e / l_e
    //   = 1e4 · (4π·1e-7) · 1000 · 1e-4 / 5e-2 = 0.02513... H
    const Real expected = 1e4 * (4.0 * std::numbers::pi_v<Real> * 1e-7) *
                          1000.0 * 1e-4 / 5e-2;
    CHECK(dev.L_eff() == Approx(expected).margin(1e-6));
    // No flux drawn yet → magnetization is at the unmagnetized origin.
    CHECK(dev.flux() == Approx(0.0));
    CHECK(dev.magnetization() == Approx(0.0));
}

TEST_CASE("HysteresisInductorDevice: explicit L_eff overrides the geometry default",
          "[item1][hysteresis][device]") {
    HysteresisInductorDevice::Params params{};
    params.geom.turns = 50.0;
    params.geom.area = 1e-4;
    params.geom.path_length = 5e-2;
    params.L_eff = 1e-3;   // explicitly 1 mH

    HysteresisInductorDevice dev(params);
    CHECK(dev.L_eff() == Approx(1e-3));
}

TEST_CASE("HysteresisInductorDevice: Circuit::add_hysteresis_inductor integrates "
          "the device into the runtime",
          "[item1][hysteresis][circuit]") {
    Circuit ckt;
    auto vin = ckt.add_node("vin");
    auto out = ckt.add_node("out");
    ckt.add_voltage_source("V1", vin, Circuit::ground(), 1.0);

    HysteresisInductorDevice::Params params{};
    params.geom.turns = 100.0;
    params.geom.area = 1e-4;
    params.geom.path_length = 5e-2;
    params.L_eff = 1e-3;
    ckt.add_hysteresis_inductor("H1", vin, out, params);
    ckt.add_resistor("R_load", out, Circuit::ground(), 10.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;
    opts.dt = 1e-5;
    opts.dt_min = 1e-9;
    opts.dt_max = 1e-5;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.newton_options.num_nodes = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);
    const auto run = sim.run_transient();
    REQUIRE(run.success);

    // After the transient, the flux λ ≈ L_eff · i_branch should be
    // non-zero (the linear inductor is being charged through R_load).
    INFO("HysteresisInductor flux after transient: " << ckt.hysteresis_flux("H1"));
    CHECK(std::abs(ckt.hysteresis_flux("H1")) > 0.0);
    // L_eff accessor reads through.
    CHECK(ckt.hysteresis_L_eff("H1") == Approx(1e-3));
}
