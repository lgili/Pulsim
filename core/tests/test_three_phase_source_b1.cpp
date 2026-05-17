// consolidate-motors-and-three-phase, Phase B.1 — Catch2 tests for the new
// programmable + harmonic three-phase source overloads of
// `Circuit::add_three_phase_source`. Pins the leg count and the
// per-leg amplitude/frequency the decomposition produces so the
// expansion does not silently drift.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/simulation.hpp"
#include "pulsim/v1/grid/three_phase_source.hpp"

#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

// Hunt a SineVoltageSource by name suffix in a circuit's device list.
const SineVoltageSource* find_sine(const Circuit& ckt, const std::string& name) {
    const auto& conns = ckt.connections();
    const auto& devs = ckt.devices();
    for (std::size_t i = 0; i < conns.size(); ++i) {
        if (conns[i].name == name) {
            return std::get_if<SineVoltageSource>(&devs[i]);
        }
    }
    return nullptr;
}

}  // namespace

TEST_CASE("Phase B.1: programmable 3φ source decomposes into 3 scaled sine legs",
          "[consolidation][three-phase][programmable]") {
    grid::ThreePhaseSourceProgrammable prog;
    prog.base.v_rms = 100.0;
    prog.base.frequency = 60.0;
    prog.g_a = 0.5;     // sag on A
    prog.g_b = 1.0;
    prog.g_c = 1.2;     // swell on C

    Circuit ckt;
    auto a = ckt.add_node("a");
    auto b = ckt.add_node("b");
    auto c = ckt.add_node("c");
    auto n = ckt.add_node("n");
    ckt.add_three_phase_source("S", a, b, c, n, prog);

    // Three sine legs reserved.
    const auto* sa = find_sine(ckt, "S__A");
    const auto* sb = find_sine(ckt, "S__B");
    const auto* sc = find_sine(ckt, "S__C");
    REQUIRE(sa != nullptr);
    REQUIRE(sb != nullptr);
    REQUIRE(sc != nullptr);

    // Per-leg amplitude is `v_peak · g_k` with v_peak = v_rms · √2.
    const Real v_peak = 100.0 * std::numbers::sqrt2_v<Real>;
    CHECK(sa->params().amplitude == Approx(v_peak * 0.5).margin(1e-9));
    CHECK(sb->params().amplitude == Approx(v_peak * 1.0).margin(1e-9));
    CHECK(sc->params().amplitude == Approx(v_peak * 1.2).margin(1e-9));
    CHECK(sa->params().frequency == Approx(60.0));
}

TEST_CASE("Phase B.1: harmonic 3φ source decomposes into 3+3·N sine legs",
          "[consolidation][three-phase][harmonic]") {
    grid::ThreePhaseHarmonicSource harm;
    harm.fundamental.v_rms = 230.0;
    harm.fundamental.frequency = 50.0;
    harm.harmonics.push_back({5, 0.05, 0.0});   // 5% of fundamental at 250 Hz
    harm.harmonics.push_back({7, 0.02, 0.0});   // 2% of fundamental at 350 Hz

    Circuit ckt;
    auto a = ckt.add_node("a");
    auto b = ckt.add_node("b");
    auto c = ckt.add_node("c");
    auto n = ckt.add_node("n");
    ckt.add_three_phase_source("H", a, b, c, n, harm);

    // 3 (fundamental) + 3·2 (two harmonics) = 9 sine legs.
    int sine_count = 0;
    for (const auto& dev : ckt.devices()) {
        if (std::holds_alternative<SineVoltageSource>(dev)) ++sine_count;
    }
    CHECK(sine_count == 9);

    // Fundamental phase A: amplitude = v_rms · √2.
    const auto* h0a = find_sine(ckt, "H__H0_A");
    REQUIRE(h0a != nullptr);
    CHECK(h0a->params().frequency == Approx(50.0));
    CHECK(h0a->params().amplitude ==
          Approx(230.0 * std::numbers::sqrt2_v<Real>).margin(1e-9));

    // First harmonic (5th): amplitude = 0.05 · v_peak, frequency = 250 Hz.
    const auto* h1a = find_sine(ckt, "H__H1_A");
    REQUIRE(h1a != nullptr);
    CHECK(h1a->params().frequency == Approx(250.0));
    CHECK(h1a->params().amplitude ==
          Approx(0.05 * 230.0 * std::numbers::sqrt2_v<Real>).margin(1e-9));

    // Second harmonic (7th): amplitude = 0.02 · v_peak, frequency = 350 Hz.
    const auto* h2a = find_sine(ckt, "H__H2_A");
    REQUIRE(h2a != nullptr);
    CHECK(h2a->params().frequency == Approx(350.0));
    CHECK(h2a->params().amplitude ==
          Approx(0.02 * 230.0 * std::numbers::sqrt2_v<Real>).margin(1e-9));
}
