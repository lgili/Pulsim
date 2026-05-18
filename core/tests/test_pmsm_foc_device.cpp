// PmsmFocDevice — Catch2 tests for the signal-domain PMSM current-loop
// controller (consolidate-motors-and-three-phase, Phase B.2b).
//
// PmsmFocDevice wraps motors::PmsmFocCurrentLoop (two PI compensators tuned
// against L_d/L_q/R_s for a target bandwidth). The device has no electrical
// pins; the user wires id_ref/iq_ref and id_meas/iq_meas each step, and
// reads back Vd_ref/Vq_ref after the variant walker invokes
// `update_history()`.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/components/pmsm_foc_device.hpp"
#include "pulsim/v1/simulation.hpp"

#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

PmsmFocDevice::Params make_default_foc_params() {
    PmsmFocDevice::Params p{};
    p.motor.Rs = 0.5;
    p.motor.Ld = 1.5e-3;
    p.motor.Lq = 1.5e-3;
    p.motor.psi_pm = 0.05;
    p.motor.pole_pairs = 4;
    p.motor.J = 1e-3;
    p.foc.bandwidth_hz = 1000.0;
    p.foc.Vd_min = -100.0;
    p.foc.Vd_max =  100.0;
    p.foc.Vq_min = -100.0;
    p.foc.Vq_max =  100.0;
    return p;
}

}  // namespace

TEST_CASE("PmsmFocDevice: zero-error reference yields zero V_d/V_q",
          "[pmsm_foc][device][b2b]") {
    PmsmFocDevice dev(make_default_foc_params(), "Ctrl1");
    dev.set_references(0.0, 0.0);
    dev.set_measurements(0.0, 0.0);
    dev.set_timestep(1e-4);
    dev.update_history();
    CHECK(dev.vd_ref() == Approx(0.0).margin(1e-9));
    CHECK(dev.vq_ref() == Approx(0.0).margin(1e-9));
}

TEST_CASE("PmsmFocDevice: positive iq error drives V_q in the right direction",
          "[pmsm_foc][device][b2b]") {
    // iq_ref = 1 A, iq_meas = 0 → positive error → V_q rises (PI sign).
    PmsmFocDevice dev(make_default_foc_params(), "Ctrl1");
    dev.set_references(0.0, 1.0);
    dev.set_measurements(0.0, 0.0);
    dev.set_timestep(1e-4);

    Real vq_prev = 0.0;
    for (int i = 0; i < 10; ++i) {
        dev.update_history();
        // Each step should yield a non-negative Vq (positive error, PI
        // integrator accumulates positive correction).
        REQUIRE(dev.vq_ref() >= vq_prev - 1e-9);
        vq_prev = dev.vq_ref();
    }
    CHECK(dev.vq_ref() > 0.0);
    // V_d still zero because id error is zero.
    CHECK(dev.vd_ref() == Approx(0.0).margin(1e-6));
}

TEST_CASE("PmsmFocDevice: output respects Vd/Vq clamps",
          "[pmsm_foc][device][b2b]") {
    auto params = make_default_foc_params();
    params.foc.Vq_max = 5.0;        // tight clamp
    PmsmFocDevice dev(params, "Ctrl1");
    dev.set_references(0.0, 100.0);  // huge demand → integrator saturates
    dev.set_measurements(0.0, 0.0);
    dev.set_timestep(1e-3);
    for (int i = 0; i < 200; ++i) {
        dev.update_history();
    }
    CHECK(dev.vq_ref() == Approx(5.0).margin(1e-6));
}

TEST_CASE("PmsmFocDevice: registers in Circuit and advances on transient",
          "[pmsm_foc][device][b2b][circuit]") {
    Circuit ckt;
    auto vin = ckt.add_node("vin");
    auto out = ckt.add_node("out");
    ckt.add_voltage_source("V1", vin, Circuit::ground(), 1.0);
    ckt.add_resistor("R1", vin, out, 100.0);
    ckt.add_capacitor("C1", out, Circuit::ground(), 1e-6, 0.0);

    ckt.add_pmsm_foc("Ctrl1", make_default_foc_params());
    ckt.set_pmsm_foc_references("Ctrl1", 0.0, 0.5);
    ckt.set_pmsm_foc_measurements("Ctrl1", 0.0, 0.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 0.001;       // 1 ms
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

    // After ~100 steps of positive iq error, the PI integrator should have
    // pushed Vq above zero (and below clamp).
    const Real vq = ckt.pmsm_foc_vq_ref("Ctrl1");
    INFO("Vq_ref after transient = " << vq << " V");
    CHECK(vq > 0.0);
    CHECK(vq <= 100.0);
}
