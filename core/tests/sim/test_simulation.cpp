// add-kernel-simulation-object: sim::RunOptions, the option x engine table
// and the sim::Simulation facade, from C++ with no Python.

#include <catch2/catch_test_macros.hpp>

#include "pulsim/analysis/cancellation.hpp"
#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/errors/convergence.hpp"
#include "pulsim/ir/circuit.hpp"
#include "pulsim/sim/options.hpp"
#include "pulsim/sim/simulation.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <type_traits>

using namespace pulsim;

namespace {

// A buck with a Shockley freewheel diode: a Newton device the facade must
// stamp without being handed a refresh. dt = 2^-20 s so that a split at a
// grid point lands on the same binary instants as the continuous run.
constexpr Real kDt = 1.0 / 1048576.0;

builder::CircuitBuilder shockley_buck() {
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "in", "gnd", 24.0);
    b.add_switch("S", "in", "sw", 1e3, 1e-9);
    b.add_shockley_diode("D", "gnd", "sw");
    b.add_inductor("L", "sw", "out", 100e-6);
    b.add_capacitor("C", "out", "gnd", 10e-6);
    b.add_resistor("R", "out", "gnd", 5.0);
    return b;
}

topology::SwitchStateMask gate(Real t) {
    topology::SwitchStateMask m(1);
    const Real period = 64 * kDt;
    m.set(0, std::fmod(t, period) < 0.4 * period);
    return m;
}

}  // namespace

TEST_CASE("the option table covers every option once", "[sim][options]") {
    std::set<std::string_view> seen;
    for (const auto& o : sim::kRunOptionTable) {
        REQUIRE(seen.insert(o.name).second);
        REQUIRE(o.engines != 0);
        // An option some engine refuses names the mechanism it lacks.
        if (o.engines != sim::kEvery) REQUIRE(!o.why.empty());
    }
    REQUIRE(sim::find_option("store_every")->honoured_by(sim::Engine::FixedTrap));
    REQUIRE(!sim::find_option("store_every")->honoured_by(sim::Engine::TrBdf2));
    REQUIRE(!sim::find_option("rtol")->honoured_by(sim::Engine::FixedTrap));
}

TEST_CASE("RunOptions resolves to the drivers' defaults", "[sim][options]") {
    sim::RunOptions o;
    o.t_end = 1e-3;
    o.dt = 1e-6;
    const auto s = sim::to_simulation_options(o);
    const solver::SimulationOptions d;
    REQUIRE(s.max_newton_iterations == d.max_newton_iterations);
    REQUIRE(s.max_event_iterations == d.max_event_iterations);
    REQUIRE(s.tol_newton_dx == d.tol_newton_dx);
    REQUIRE(s.max_dt_halvings == d.max_dt_halvings);
    REQUIRE(s.newton_roundoff_budget == d.newton_roundoff_budget);
    REQUIRE(s.store_every == 1);
    const auto t = sim::to_trbdf2_options(o);
    REQUIRE(t.rtol == sim::Defaults::rtol_variable);
    REQUIRE(t.atol == sim::Defaults::atol_variable);
    REQUIRE(t.h_max == 1e-6);
    // Zero is a value, not "default".
    o.max_newton_iterations = 0;
    REQUIRE_THROWS_AS(sim::to_simulation_options(o), sim::OptionsError);
    o.max_newton_iterations.reset();
    o.max_event_iterations = 0;
    REQUIRE(sim::to_simulation_options(o).max_event_iterations == 0);
    REQUIRE_THROWS_AS(sim::to_trbdf2_options(o), UnsupportedError);
}

TEST_CASE("Simulation matches the fixed-step driver bit for bit with no "
          "refresh argument", "[sim][facade]") {
    auto b = shockley_buck();
    const auto circuit = ir::Circuit::from_builder(b);
    sim::RunOptions o;
    o.engine = sim::EngineChoice::Pwl;
    o.t_end = 512 * kDt;
    o.dt = kDt;
    sim::Simulation s{circuit, o};
    s.set_switch_fn(gate);
    const auto r = s.run();
    REQUIRE(s.plan().engine == sim::Engine::FixedTrap);

    // The driver called directly, as pulsim.simulate() calls it.
    auto rb = circuit.to_builder();
    pwl::PwlStateSpaceCache cache(rb.graph(), rb.pool());
    cache.share_symbolic(circuit.symbolic_plan());
    cache.build_lazy(kDt);
    const auto ref = solver::run_transient(cache, rb.graph(), rb.pool(),
                                           sim::to_simulation_options(o), gate);
    REQUIRE(r.times == ref.times);
    REQUIRE(r.states.size() == ref.states.size());
    for (std::size_t k = 0; k < r.states.size(); ++k) {
        REQUIRE((r.states[k].array() == ref.states[k].array()).all());
    }
    // The diode conducted (a missing Newton stamp would leave it open).
    const Index sw = rb.node_id_of("sw");
    Real vmin = 0;
    for (const auto& x : r.states) vmin = std::min(vmin, x[sw]);
    REQUIRE(vmin < -0.3);
}

TEST_CASE("run_until + snapshot + restore reproduce a continuous fixed-step "
          "run exactly", "[sim][facade]") {
    const auto circuit = ir::Circuit::from_builder(shockley_buck());
    sim::RunOptions o;
    o.engine = sim::EngineChoice::Pwl;
    o.t_end = 512 * kDt;
    o.dt = kDt;
    sim::Simulation whole{circuit, o};
    whole.set_switch_fn(gate);
    const auto full = whole.run();

    sim::Simulation first{circuit, o};
    first.set_switch_fn(gate);
    const auto a = first.run_until(200 * kDt);
    sim::Simulation second{circuit, o};
    second.set_switch_fn(gate);
    second.restore(first.snapshot());
    const auto b = second.run();
    // Row 0 of the resumed run is the snapshot state (the last row of a).
    REQUIRE(a.states.size() + b.states.size() - 1 == full.states.size());
    for (std::size_t k = 0; k < full.states.size(); ++k) {
        const auto& x = k < a.states.size() ? a.states[k]
                                            : b.states[k - a.states.size() + 1];
        REQUIRE((x.array() == full.states[k].array()).all());
    }
}

TEST_CASE("Simulation refuses by name before running", "[sim][facade]") {
    const auto circuit = ir::Circuit::from_builder(shockley_buck());
    sim::RunOptions o;
    o.t_end = 1e-4;
    o.engine = sim::EngineChoice::TrBdf2;
    o.store_every = 2;
    o.max_dt_halvings = 3;
    try {
        sim::Simulation s{circuit, o};
        FAIL("no refusal");
    } catch (const UnsupportedError& e) {
        REQUIRE(e.engine() == "trbdf2");
        REQUIRE(e.feature() == "max_dt_halvings");  // the first, in table order
        const std::string what = e.what();
        REQUIRE(what.find("store_every") != std::string::npos);  // every one
    }
    o.engine = sim::EngineChoice::Exact;
    REQUIRE_THROWS_AS((sim::Simulation{circuit, o}), UnsupportedError);
    o.engine = sim::EngineChoice::Pwl;  // pwl needs a step
    o.store_every.reset();
    o.max_dt_halvings.reset();
    REQUIRE_THROWS_AS((sim::Simulation{circuit, o}), sim::OptionsError);
    // 'auto' with a dt is the fixed step, and says why.
    o.engine = sim::EngineChoice::Auto;
    o.dt = kDt;
    sim::Simulation a{circuit, o};
    REQUIRE(a.plan().engine == sim::Engine::FixedTrap);
    REQUIRE(!a.plan().route_reasons.empty());
    // 'auto' without one and an option only the fixed step honours.
    o.dt.reset();
    o.store_every = 4;
    REQUIRE_THROWS_AS((sim::Simulation{circuit, o}), UnsupportedError);
}

TEST_CASE("the caller's circuit is never changed by the preflight",
          "[sim][facade]") {
    // A floating node behind a capacitor: the preflight ties it.
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", 1.0);
    b.add_resistor("R", "in", "a", 1e3);
    b.add_capacitor("C", "a", "f", 1e-9);
    b.add_capacitor("C2", "f", "f2", 1e-9);
    b.add_resistor("R2", "f2", "f3", 1e3);
    const auto circuit = ir::Circuit::from_builder(b);
    const auto n0 = circuit.devices().size();
    sim::RunOptions o;
    o.t_end = 1e-6;
    o.dt = 1e-8;
    sim::Simulation s{circuit, o};
    REQUIRE(circuit.devices().size() == n0);
    REQUIRE(s.circuit().devices().size() == n0 + s.plan().ties.size());
}

TEST_CASE("every pulsim error is a PulsimError and keeps its std base",
          "[sim][errors]") {
    const auto circuit = ir::Circuit::from_builder(shockley_buck());
    sim::RunOptions o;
    o.t_end = 1e-4;
    o.engine = sim::EngineChoice::Pwl;  // no dt
    try {
        sim::Simulation s{circuit, o};
        FAIL("no refusal");
    } catch (const PulsimError& e) {
        REQUIRE(dynamic_cast<const std::invalid_argument*>(&e) != nullptr);
        REQUIRE(std::string(e.message()).find("dt") != std::string::npos);
    }
    o.engine = sim::EngineChoice::Exact;
    REQUIRE_THROWS_AS((sim::Simulation{circuit, o}), PulsimError);
    STATIC_REQUIRE(std::is_base_of_v<PulsimError, solver::SimulationAborted>);
    STATIC_REQUIRE(std::is_base_of_v<PulsimError, ConvergenceFailure>);
    STATIC_REQUIRE(std::is_base_of_v<PulsimError, analysis::Cancelled>);
    STATIC_REQUIRE(std::is_base_of_v<std::runtime_error, analysis::Cancelled>);
}
