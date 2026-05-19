// Regression test for the `run_transient(x0)` consistent-initialization bug.
//
// Before the fix (commit ?), passing `x0 = zeros` to `run_transient(...)` on
// a circuit containing a voltage source that enforced `V(node) = V_src ≠ 0`
// would cause the Tustin discretization to ping-pong the state between
// `0` and `2·V_src` at successive steps. With an even step count the
// last sample would read `0` (instead of `V_src`); with odd, `2·V_src`.
// Both wrong.
//
// Root cause: Tustin's discretization
//   (M + dt/2·N)·x_{n+1} = (M − dt/2·N)·x_n + dt/2·(b_n + b_{n+1})
// gives the correct fixed point for algebraic constraints (`N·x = b`) IFF
// x_n already satisfies them. The Newton-DAE path (FormulationMode::Direct)
// re-enforces the constraint at every step and is unaffected, but the
// segment-primary (Tustin) path used by `FormulationMode::ProjectedWrapper`
// (the default) was vulnerable.
//
// Fix: snap each voltage source's enforced node voltage in x0 to its
// source value at `tstart` before starting the integration. Lives in
// `simulation.cpp::run_transient_native_impl` at the
// "Consistent initialization for voltage-source constraints" block.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/simulation.hpp"
#include "pulsim/v1/runtime_circuit.hpp"

using namespace pulsim::v1;
using Catch::Approx;

namespace {

SimulationOptions make_opts(const Circuit& ckt) {
    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-5;
    opts.dt = 1e-6;
    opts.dt_min = opts.dt;
    opts.dt_max = opts.dt;
    opts.adaptive_timestep = false;
    opts.integrator = Integrator::BDF1;
    opts.newton_options.num_nodes = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();
    return opts;
}

}  // namespace

TEST_CASE("run_transient(zeros) snaps voltage sources to their enforced values",
          "[v1][consistent-init][regression]") {
    SECTION("grounded reference voltage source (the canonical buck PI case)") {
        Circuit ckt;
        const auto gnd = Circuit::ground();
        const auto n_vref = ckt.add_node("vref");
        ckt.add_voltage_source("Vref", n_vref, gnd, 12.0);

        SimulationOptions opts = make_opts(ckt);
        Simulator sim(ckt, opts);

        Vector x0 = Vector::Zero(static_cast<Index>(ckt.system_size()));
        auto result = sim.run_transient(x0);
        REQUIRE(result.success);
        REQUIRE_FALSE(result.states.empty());

        // The first sample (recorded right after the snap) must already
        // have V(vref) = 12 V.
        CHECK(result.states.front()[n_vref] == Approx(12.0).margin(1e-9));
        // Every subsequent sample stays at 12 V (no ping-pong).
        for (const auto& state : result.states) {
            CHECK(state[n_vref] == Approx(12.0).margin(1e-6));
        }
    }

    SECTION("multiple grounded voltage sources") {
        Circuit ckt;
        const auto gnd = Circuit::ground();
        const auto n_a = ckt.add_node("a");
        const auto n_b = ckt.add_node("b");
        ckt.add_voltage_source("Va", n_a, gnd, 5.0);
        ckt.add_voltage_source("Vb", n_b, gnd, -3.0);

        SimulationOptions opts = make_opts(ckt);
        Simulator sim(ckt, opts);

        Vector x0 = Vector::Zero(static_cast<Index>(ckt.system_size()));
        auto result = sim.run_transient(x0);
        REQUIRE(result.success);

        CHECK(result.states.back()[n_a] == Approx(5.0).margin(1e-6));
        CHECK(result.states.back()[n_b] == Approx(-3.0).margin(1e-6));
    }

    SECTION("PWM source snaps to voltage_at(tstart)") {
        Circuit ckt;
        const auto gnd = Circuit::ground();
        const auto n_sw = ckt.add_node("sw");
        PWMParams pwm;
        pwm.v_high = 48.0;
        pwm.v_low = 0.0;
        pwm.frequency = 100e3;
        pwm.duty = 0.25;
        pwm.phase = 0.0;
        ckt.add_pwm_voltage_source("Vpwm", n_sw, gnd, pwm);
        ckt.add_resistor("R", n_sw, gnd, 100.0);

        SimulationOptions opts = make_opts(ckt);
        // Short horizon — just check the snap at t=0
        opts.tstop = 1e-9;
        opts.dt = 1e-10;
        opts.dt_min = opts.dt; opts.dt_max = opts.dt;
        Simulator sim(ckt, opts);

        Vector x0 = Vector::Zero(static_cast<Index>(ckt.system_size()));
        auto result = sim.run_transient(x0);
        REQUIRE(result.success);

        // At t = 0 with duty = 0.25 and phase = 0, voltage_at(0) = v_high = 48.
        CHECK(result.states.front()[n_sw] == Approx(48.0).margin(1e-6));
    }

    SECTION("voltage source consistent with the supplied x0 — no change") {
        // If x0 already satisfies the constraint, the snap is a no-op
        // and the integration proceeds exactly as before. This pins the
        // back-compat contract: existing code paths that supply DC OP
        // solutions don't see any behavior change.
        Circuit ckt;
        const auto gnd = Circuit::ground();
        const auto n_vref = ckt.add_node("vref");
        ckt.add_voltage_source("Vref", n_vref, gnd, 12.0);

        SimulationOptions opts = make_opts(ckt);
        Simulator sim(ckt, opts);

        Vector x0 = Vector::Zero(static_cast<Index>(ckt.system_size()));
        x0[n_vref] = 12.0;  // already consistent
        auto result = sim.run_transient(x0);
        REQUIRE(result.success);

        CHECK(result.states.front()[n_vref] == Approx(12.0).margin(1e-9));
        CHECK(result.states.back()[n_vref] == Approx(12.0).margin(1e-6));
    }
}
