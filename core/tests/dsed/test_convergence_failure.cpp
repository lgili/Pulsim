// =============================================================================
// DSED — the engine's own non-convergence failures are typed
// =============================================================================
//
// Each give-up site of PEDSimulator throws pulsim::ConvergenceFailure
// (a std::runtime_error, message text unchanged) with its kind, the
// simulated time and the device when one is known. One case per site.

#include <catch2/catch_test_macros.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/dsed/builder_adapter.hpp"
#include "pulsim/dsed/scheduler.hpp"
#include "pulsim/errors/convergence.hpp"
#include "pulsim/models/switched_diode.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/diode_event_state.hpp"
#include "pulsim/pwl/row_names.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

using namespace pulsim;
using namespace pulsim::dsed;

namespace {

// One state, x' = f(t, x); a single constant mask.
struct ToySystem {
    std::function<Real(Real, Real)> f;
    [[nodiscard]] bool current_mask() const noexcept { return false; }
    void set_mask(bool) noexcept {}
    [[nodiscard]] Vector rhs(Real t, const Vector& x) const {
        Vector d(1);
        d << f(t, x[0]);
        return d;
    }
};
struct NoSwitch {
    [[nodiscard]] bool operator()(Real) const noexcept { return false; }
};

ConvergenceFailure run_toy(std::function<Real(Real, Real)> f, Real dt_max, Real t_end) {
    ToySystem sys{std::move(f)};
    PEDSimulator sim{sys, NoSwitch{}, PIController{1e-6, 1e-9}, EventPredictor{}, Real{1e-9}, dt_max};
    Vector x0 = Vector::Zero(1);
    try {
        (void)sim.simulate(x0, t_end);
    } catch (const ConvergenceFailure& e) {
        return e;
    }
    FAIL("no ConvergenceFailure thrown");
    return ConvergenceFailure(ConvergenceKind::NaN, 0, "", "");
}

struct ZeroMask {
    Size n;
    [[nodiscard]] topology::SwitchStateMask operator()(Real) const {
        return topology::SwitchStateMask(n);
    }
};

// The binding's wiring (python/dsed_bindings.cpp, run_ped_native_builder):
// two predicates per PWL diode, diode commutation on, optionally the LCP.
ConvergenceFailure run_builder(builder::CircuitBuilder& b, Real t_end, Real lcp_h_e) {
    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    using Adapter = NativeCircuitBuilderAdapter<>;
    Adapter ad{b.graph(), b.pool(), cache};
    EventPredictor pred;
    std::vector<SchedulerDiode> diodes;
    pwl::DiodeEventState census(ad.graph(), ad.pool());
    const topology::SwitchStateMask owned = census.diode_owned_bits();
    for (const auto& d : census.entries()) {
        std::string label = pwl::branch_label(ad.graph(), d.branch_id);
        const Adapter* a = &ad;
        const Index from = d.from, to = d.to;
        const auto prm = d.params();
        for (const bool on : {false, true}) {
            const auto type = on ? PredicateType::DiodeOff : PredicateType::DiodeOn;
            pred.add_predicate(EventPredicate{
                .name = label + (on ? " turns OFF" : " turns ON"),
                .fn = [a, from, to, prm, on](Real t, const Vector& x) {
                    return models::SwitchedDiode::crossing_signal(
                        on, a->node_pair_voltage(from, to, t, x), prm);
                },
                .type = type,
                .priority = default_priority(type),
                .aux_index = static_cast<int>(d.switch_idx),
                .required_bit = on ? 1 : 0,
            });
        }
        diodes.push_back(SchedulerDiode{d.switch_idx, d.from, d.to, d.g_on, d.g_off,
                                        d.V_th, d.V_f, std::move(label)});
    }
    const auto n_sw = static_cast<Size>(ad.graph().num_switches());
    PEDSimulator<Adapter, ZeroMask> sim(ad, ZeroMask{n_sw}, PIController{1e-6, 1e-9},
                                        std::move(pred), Real{1e-9}, Real{1e-5}, 1);   // simulate()'s defaults
    sim.enable_diode_commutation(owned, topology::SwitchStateMask(n_sw), std::move(diodes));
    sim.set_event_projection(true);
    if (lcp_h_e > 0) sim.enable_mode_lcp(lcp_h_e, ad.passivity());
    ad.set_mask(topology::SwitchStateMask(n_sw));
    const Vector x0 = Vector::Zero(ad.n_state());
    try {
        (void)sim.simulate(x0, t_end);
    } catch (const ConvergenceFailure& e) {
        return e;
    }
    FAIL("no ConvergenceFailure thrown");
    return ConvergenceFailure(ConvergenceKind::NaN, 0, "", "");
}

// Bridges fed by 100 V 50 Hz sines through 0.2 mH, 6 degrees apart, with
// PWL diodes (g_on 1e3, g_off 1e-9): each on its own 200 uF || 50 ohm,
// or all paralleled on one 2 mF || 5 ohm bus.
void bridge_bank(builder::CircuitBuilder& b, int n, Real vth, bool shared) {
    const char* ph[3] = {"a", "b", "c"};
    const Real ang[3] = {0.0, -120.0, 120.0};
    for (int k = 0; k < n; ++k) {
        const std::string pk = shared ? "p" : "p" + std::to_string(k);
        const std::string nk = shared ? "n" : "n" + std::to_string(k);
        for (int i = 0; i < 3; ++i) {
            const std::string s = "s" + std::to_string(k) + ph[i];
            const std::string x = "x" + std::to_string(k) + ph[i];
            b.add_sine_voltage_source("V" + std::to_string(k) + ph[i], s, "gnd", 0.0, 100.0, 50.0,
                                      (ang[i] + 6.0 * k) * std::numbers::pi / 180.0);
            b.add_inductor("L" + std::to_string(k) + ph[i], s, x, 0.2e-3);
            b.add_diode("Dt" + std::to_string(k) + ph[i], x, pk, 1e3, 1e-9, vth);
            b.add_diode("Db" + std::to_string(k) + ph[i], nk, x, 1e3, 1e-9, vth);
        }
        if (!shared) {
            b.add_capacitor("C" + std::to_string(k), pk, nk, 200e-6);
            b.add_resistor("R" + std::to_string(k), pk, nk, 50.0);
            b.add_resistor("Rg" + std::to_string(k), nk, "gnd", 1e6);
        }
    }
    if (shared) {
        b.add_capacitor("C", "p", "n", 2e-3);
        b.add_resistor("R", "p", "n", 5.0);
        b.add_resistor("Rg", "n", "gnd", 1e6);
    }
}

}  // namespace

TEST_CASE("ConvergenceFailure: a non-finite RK45 step is kind nan",
          "[v2][dsed][convergence]") {
    const auto e = run_toy([](Real, Real) { return std::numeric_limits<Real>::quiet_NaN(); },
                           Real{1e-6}, Real{1e-3});
    REQUIRE(e.kind() == ConvergenceKind::NaN);
    REQUIRE(std::string{e.kind_name()} == "nan");
    REQUIRE(std::string{e.what()}.find("NaN/Inf") != std::string::npos);
}

TEST_CASE("ConvergenceFailure: an integrator grinding on a 1e-12 s mode is kind no_progress",
          "[v2][dsed][convergence]") {
    const auto e = run_toy([](Real, Real x) { return Real{-1e12} * (x - Real{1}); },
                           Real{1e-4}, Real{1.0});
    REQUIRE(e.kind() == ConvergenceKind::NoProgress);
    REQUIRE(e.time() > 0.0);
    REQUIRE(std::string{e.what()}.find("no meaningful progress") != std::string::npos);
}

TEST_CASE("ConvergenceFailure: the step cap is kind max_steps",
          "[v2][dsed][convergence]") {
    // dt_max = 1 ns over 1 s: every 100k steps advance 1e-4 s (>= 100*dt_max,
    // so no stall), and the 10M-step cap is reached at t ~ 1e-2 s.
    const auto e = run_toy([](Real, Real) { return Real{0}; }, Real{1e-9}, Real{1.0});
    REQUIRE(e.kind() == ConvergenceKind::MaxSteps);
    REQUIRE(std::string{e.what()} == "PEDSimulator: exceeded max-step cap");
}

TEST_CASE("ConvergenceFailure: a settle that hits its round cap is kind settle",
          "[v2][dsed][convergence]") {
    // Five bridges paralleled on one bus, no guard: the pairs fight at rest.
    builder::CircuitBuilder b;
    bridge_bank(b, 5, 0.0, true);
    const auto e = run_builder(b, 0.02, 0.0);
    REQUIRE(e.kind() == ConvergenceKind::Settle);
    REQUIRE(e.time() < 1e-9);
    REQUIRE(std::string{e.what()}.find("did not settle after 64 rounds") != std::string::npos);
}

TEST_CASE("ConvergenceFailure: a chattering diode is kind chatter and named",
          "[v2][dsed][convergence]") {
    // Ten bridges behind 0.2 mH, V_th = 1e-2, LCP at h_e = 2 ns: a diode of
    // the first bridge chatters at t ~ 1e-19 s (bank60_cost.out).
    builder::CircuitBuilder b;
    bridge_bank(b, 10, 1e-2, false);
    const auto e = run_builder(b, 0.04, 2e-9);
    REQUIRE(e.kind() == ConvergenceKind::Chatter);
    REQUIRE(!e.device().empty());
    REQUIRE(std::string{e.what()}.find(e.device() + " is chattering") != std::string::npos);
}
