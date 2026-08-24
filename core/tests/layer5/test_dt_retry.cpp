// =============================================================================
// Layer 5 — local time-step reduction on a failed step
// =============================================================================
//
// v2.0 Phase 2 (B.4).
//
// A step whose inner solve fails used to end the run and discard
// everything computed before it. A smaller step is the standard
// answer and — unlike the dead fallback rungs Phase 2 B.2 removed
// from the DC cascade — a genuinely DIFFERENT problem: the
// trapezoidal companion's 2C/dt grows as dt shrinks, which both
// improves the Jacobian's diagonal dominance and puts the previous
// state closer to the answer.
//
// The circuit here is a 170 V mains half-wave rectifier, which is
// about as ordinary as a power circuit gets. It fails at
// dt = 1e-4 and needs exactly one halving.
//
// THE INVARIANT THAT MATTERS MOST is that the output grid does not
// move. Sub-steps are internal; `times[k]` stays exactly
// `t_start + k·dt`, because Phase 1e made decimation a pure stride
// so that an FFT of the result stays valid.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/models/ideal_diode.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/nonlinear_refresh_mosfet_level1.hpp"
#include "pulsim/solver/run_transient.hpp"

#include <algorithm>
#include <string>

using namespace pulsim;
using Catch::Approx;

namespace {

builder::CircuitBuilder mains_rectifier(Real v_peak = 170.0,
                                          Real kappa = 20.0) {
    builder::CircuitBuilder b;
    b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, v_peak, 60.0);
    models::IdealDiode::Params d;
    d.kappa = kappa;
    b.add_nonlinear_diode("D", "ac", "vout", d);
    b.add_resistor("R", "vout", "gnd", 50.0);
    b.add_capacitor("C", "vout", "gnd", 100e-6);
    return b;
}

solver::SimulationOptions rect_options(Real dt) {
    solver::SimulationOptions o;
    o.t_start = 0.0;
    o.t_end   = 1.7e-2;      // just over one mains half-cycle
    o.dt      = dt;
    return o;
}

solver::SimulationResult run(builder::CircuitBuilder& b,
                              const solver::SimulationOptions& o) {
    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build_lazy(o.dt);
    const Size n_sw = static_cast<Size>(b.graph().num_switches());
    solver::SwitchScheduleFn sw = [n_sw](Real) {
        return topology::SwitchStateMask(n_sw);
    };
    return solver::run_transient(
        cache, b.graph(), b.pool(), o, sw,
        /*b_extra_fn=*/{}, /*start_from_dc_op=*/false,
        pwl::make_combined_diode_mosfet_refresh());
}

}  // namespace

TEST_CASE("A step that fails at dt succeeds at dt/2",
          "[v2][layer5][dt_retry]") {
    auto b_off = mains_rectifier();
    auto opts_off = rect_options(1e-4);
    opts_off.max_dt_halvings = 0;          // pre-v2.0 behaviour
    bool threw = false;
    try {
        (void)run(b_off, opts_off);
    } catch (const std::exception& e) {
        threw = true;
        INFO(e.what());
    }
    REQUIRE(threw);   // the premise: this run really does die

    auto b_on = mains_rectifier();
    const auto res = run(b_on, rect_options(1e-4));

    REQUIRE(res.num_steps() > 0);
    REQUIRE(res.dt_retries.size() == 1);
    const auto& r = res.dt_retries.front();
    REQUIRE(r.halvings == 1);              // dt/2 was enough
    REQUIRE(r.t > 0.0);
    // The record says what the nominal attempt reported, so the user
    // can tell a recovered convergence failure from a recovered
    // singularity.
    REQUIRE(r.reason.find("converge") != std::string::npos);
}

TEST_CASE("A retried run keeps the sampling grid exactly",
          "[v2][layer5][dt_retry]") {
    // Sub-steps are internal. If a retry could shift a sample, an
    // FFT of the result would be silently wrong — which is the whole
    // reason Phase 1e made decimation a pure stride.
    constexpr Real dt = 1e-4;
    auto b = mains_rectifier();
    const auto res = run(b, rect_options(dt));
    REQUIRE_FALSE(res.dt_retries.empty());

    const auto opts = rect_options(dt);
    REQUIRE(res.num_steps() == opts.expected_sample_count());
    for (Size k = 0; k < res.num_steps(); ++k) {
        INFO("sample " << k);
        REQUIRE(res.times[k] ==
                 Approx(static_cast<Real>(k) * dt).margin(1e-15));
    }
}

TEST_CASE("The recovered answer is the right one",
          "[v2][layer5][dt_retry][integration]") {
    // Recovering is worth nothing if the number is wrong. A
    // half-wave peak rectifier charges its cap to one diode drop
    // below the source peak, and the run that needed a retry must
    // land there just as the one that did not.
    auto b_coarse = mains_rectifier();
    const auto coarse = run(b_coarse, rect_options(1e-4));
    REQUIRE_FALSE(coarse.dt_retries.empty());

    auto b_fine = mains_rectifier();
    const auto fine = run(b_fine, rect_options(1e-5));
    REQUIRE(fine.dt_retries.empty());      // no retry needed here

    // Resolve "vout" by scanning the graph's node names.
    Index vout = kInvalidIndex;
    for (Index i = 0; i < b_coarse.graph().num_nodes(); ++i) {
        if (b_coarse.graph().node(i).name == "vout") {
            vout = i;
            break;
        }
    }
    REQUIRE(vout != kInvalidIndex);
    Real peak_coarse = 0.0, peak_fine = 0.0;
    for (Size k = 0; k < coarse.num_steps(); ++k) {
        peak_coarse = std::max(peak_coarse, coarse.states[k][vout]);
    }
    for (Size k = 0; k < fine.num_steps(); ++k) {
        peak_fine = std::max(peak_fine, fine.states[k][vout]);
    }
    REQUIRE(peak_fine == Approx(169.3).margin(0.5));
    REQUIRE(peak_coarse == Approx(peak_fine).margin(0.5));
}

TEST_CASE("A run that needs no retry pays nothing and records none",
          "[v2][layer5][dt_retry]") {
    // The guard against making every user pay for a rare recovery:
    // with the retry enabled and disabled the easy run must produce
    // bit-identical output.
    auto b_on = mains_rectifier();
    const auto with_retry = run(b_on, rect_options(1e-5));

    auto b_off = mains_rectifier();
    auto opts_off = rect_options(1e-5);
    opts_off.max_dt_halvings = 0;
    const auto without = run(b_off, opts_off);

    REQUIRE(with_retry.dt_retries.empty());
    REQUIRE(with_retry.num_steps() == without.num_steps());
    const Index n = static_cast<Index>(
        b_on.pool().state_size(b_on.graph()));
    for (Size k = 0; k < with_retry.num_steps(); ++k) {
        for (Index i = 0; i < n; ++i) {
            INFO("sample " << k << " row " << i);
            REQUIRE(with_retry.states[k][i] ==
                     without.states[k][i]);      // exactly equal
        }
    }
}

TEST_CASE("Exhausting the ladder says so, with the sub-dt it reached",
          "[v2][layer5][dt_retry]") {
    // One halving is enough for this circuit, so cap the ladder at
    // zero and confirm the message is the honest one rather than a
    // silent stall.
    auto b = mains_rectifier();
    auto opts = rect_options(1e-4);
    opts.max_dt_halvings = 0;
    bool threw = false;
    try {
        (void)run(b, opts);
    } catch (const std::exception& e) {
        threw = true;
        const std::string msg = e.what();
        INFO(msg);
        // With the ladder disabled the ORIGINAL failure propagates,
        // untouched — no wrapper claiming sub-steps were tried.
        REQUIRE(msg.find("sub-steps") == std::string::npos);
        REQUIRE(msg.find("converge") != std::string::npos);
    }
    REQUIRE(threw);
}

TEST_CASE("A topology defect fails fast, not after six futile halvings",
          "[v2][layer5][dt_retry][diagnostics]") {
    // Reachability to ground is a property of the GRAPH, not of dt,
    // so an isolated subnet is singular at every step size. Retrying
    // one burns the whole ladder and then buries the named
    // diagnostic under a "could not be taken, even split into 64
    // sub-steps" wrapper — the dead-rung defect Phase 2 B.2 removed
    // from the DC cascade, reintroduced in the transient. Caught by
    // this project's own hostile-circuit suite; pinned here.
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", 12.0);
    b.add_resistor("Rp", "vin", "p1", 0.1);
    b.add_transformer("T1", "p1", "gnd", "s1", "s_gnd",
                       1e-3, 4e-3, 0.98);
    b.add_resistor("Rs", "s1", "s_gnd", 10.0);
    // NOTE: no run_preflight() — this is the auto_regularize=False
    // path, where the user asked for the error rather than the fix.

    solver::SimulationOptions o;
    o.t_start = 0.0;
    o.t_end   = 1e-5;
    o.dt      = 1e-6;
    REQUIRE(o.max_dt_halvings > 0);        // the ladder IS enabled

    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build_lazy(o.dt);
    solver::SwitchScheduleFn sw = [](Real) {
        return topology::SwitchStateMask(0);
    };

    bool threw = false;
    try {
        (void)solver::run_transient(cache, b.graph(), b.pool(), o, sw);
    } catch (const std::exception& e) {
        threw = true;
        const std::string msg = e.what();
        INFO(msg);
        // The Phase-1 diagnostic survives, intact and on top.
        REQUIRE((msg.find("s1") != std::string::npos ||
                  msg.find("s_gnd") != std::string::npos));
        // And no wrapper claiming sub-steps were tried.
        REQUIRE(msg.find("sub-steps") == std::string::npos);
    }
    REQUIRE(threw);
}
