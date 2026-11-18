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

ConvergenceFailure run_toy(std::function<Real(Real, Real)> f, Real dt_max, Real t_end,
                           std::size_t max_steps = 10'000'000) {
    ToySystem sys{std::move(f)};
    PEDSimulator sim{sys, NoSwitch{}, PIController{1e-6, 1e-9}, EventPredictor{}, Real{1e-9}, dt_max};
    sim.set_max_steps(max_steps);
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
ConvergenceFailure run_builder(builder::CircuitBuilder& b, Real t_end, Real lcp_h_e,
                               int settle_cap = 64) {
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
    sim.set_settle_round_cap(settle_cap);
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
    // so no stall). The cap is the test hook's 50k (the 10M default took
    // 120 s on the unoptimised coverage build): reached at t ~ 5e-5 s.
    const auto e = run_toy([](Real, Real) { return Real{0}; }, Real{1e-9}, Real{1.0}, 50'000);
    REQUIRE(e.kind() == ConvergenceKind::MaxSteps);
    REQUIRE((e.time() > 4.9e-5 && e.time() < 5.1e-5));
    REQUIRE(std::string{e.what()} == "PEDSimulator: exceeded max-step cap");
}

TEST_CASE("ConvergenceFailure: a settle that hits its round cap is kind settle",
          "[v2][dsed][convergence]") {
    // Five bridges paralleled on one bus, no guard: the pairs fight at rest.
    // Whether that fight settles inside 64 rounds is a rounding outcome (it
    // stopped hitting the cap when the exact propagator's eigenpairs changed),
    // so the cap is the test hook's 1 round: any event where a diode flips.
    builder::CircuitBuilder b;
    bridge_bank(b, 5, 0.0, true);
    const auto e = run_builder(b, 0.02, 0.0, 1);
    REQUIRE(e.kind() == ConvergenceKind::Settle);
    REQUIRE(e.time() < 1e-9);
    REQUIRE(std::string{e.what()}.find("did not settle after 1 rounds") != std::string::npos);
}

TEST_CASE("ConvergenceFailure: a chattering diode is kind chatter and named",
          "[v2][dsed][convergence]") {
    // The guard counts > 32 fires of one diode inside dt_max / 100 = 100 ns.
    // A half-wave rectifier at 500 MHz (tau = 1 ps, so the diode really
    // commutes twice per 2 ns period) fires ~100 times per window: a
    // deterministic burst. (The ten-bridge bank at rest chattered on Eigen 5
    // but not on Eigen 3.4.0: rounding-dependent, so not used here.)
    builder::CircuitBuilder b;
    b.add_sine_voltage_source("V", "s", "gnd", 0.0, 10.0, 5e8, 0.0);
    b.add_diode("D1", "s", "o", 1e3, 1e-9, 0.0);
    b.add_resistor("R", "o", "gnd", 1e3);
    b.add_capacitor("C", "o", "gnd", 1e-15);
    const auto e = run_builder(b, 1e-6, 0.0);
    REQUIRE(e.kind() == ConvergenceKind::Chatter);
    REQUIRE(e.device().rfind("D1 ", 0) == 0);   // "D1 (branch #N)"
    REQUIRE(e.time() < 2e-7);
    REQUIRE(std::string{e.what()}.find(e.device() + " is chattering") != std::string::npos);
}

// The end-of-run diode-law check every simulate() runs (set_diode_law_check(0) to skip): a stored sample whose
// mask has D ON while the circuit drives it backwards is refused, typed.
TEST_CASE("A stored sample breaking the diode law throws DiodeLawViolation",
          "[v2][dsed][diode_law]") {
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "s", "gnd", -10.0);
    b.add_resistor("R", "s", "a", 1.0);
    b.add_diode("D", "a", "k", 1e3, 1e-9, 0.0);
    b.add_capacitor("C", "k", "gnd", 1e-6);
    b.add_resistor("RL", "k", "gnd", 100.0);
    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    using Adapter = NativeCircuitBuilderAdapter<>;
    Adapter ad{b.graph(), b.pool(), cache};
    std::vector<SchedulerDiode> diodes;
    pwl::DiodeEventState census(ad.graph(), ad.pool());
    for (const auto& d : census.entries())
        diodes.push_back(SchedulerDiode{d.switch_idx, d.from, d.to, d.g_on, d.g_off, d.V_th,
                                        d.V_f, pwl::branch_label(ad.graph(), d.branch_id)});
    const Size di = diodes.front().switch_idx;
    const auto n_sw = static_cast<Size>(ad.graph().num_switches());
    PEDSimulator<Adapter, ZeroMask> sim(ad, ZeroMask{n_sw}, PIController{1e-6, 1e-9},
                                        EventPredictor{}, Real{1e-9}, Real{1e-5}, 1);
    sim.enable_diode_commutation(census.diode_owned_bits(), topology::SwitchStateMask(n_sw),
                                 std::move(diodes));
    ad.set_mask(topology::SwitchStateMask(n_sw));
    auto res = sim.simulate(Vector::Zero(ad.n_state()), 1e-5);   // D stays OFF: no refusal
    REQUIRE_FALSE(res.sample_masks.back().get(di));
    res.sample_masks.back().set(di, true);                        // D ON against -10 V
    try {
        sim.check_diode_law(res);
        FAIL("no DiodeLawViolation thrown");
    } catch (const DiodeLawViolation& e) {
        REQUIRE(e.diode().rfind("D", 0) == 0);
        REQUIRE(e.time() == res.times.back());
    }
    sim.set_diode_law_check(0.0);
    REQUIRE_NOTHROW(sim.check_diode_law(res));
    // a left limit at an event is checked under the mask that produced it
    sim.set_diode_law_check(1e-3);
    res.sample_masks.back().set(di, false);
    REQUIRE_NOTHROW(sim.check_diode_law(res));
    res.pre_event_index.push_back(res.times.size() - 1);
    res.pre_event_states.push_back(res.states.back());
    res.step_masks.back().set(di, true);
    REQUIRE_THROWS_AS(sim.check_diode_law(res), DiodeLawViolation);
}
