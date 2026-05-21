// =============================================================================
// Layer 5 — SimulationResult tests
// =============================================================================
//
// Plain-container tests. The struct is intentionally tiny —
// `run_transient` is the one that populates it. These tests
// guard the small surface (reserve doesn't change size, default
// is empty, push gives the expected num_steps).

#include <catch2/catch_test_macros.hpp>

#include "pulsim/v2/solver/result.hpp"

using namespace pulsim::v2;
using namespace pulsim::v2::solver;

TEST_CASE("SimulationResult: default is empty",
          "[v2][layer5][result]") {
    SimulationResult r{};
    REQUIRE(r.empty());
    REQUIRE(r.num_steps() == 0);
    REQUIRE(r.times.empty());
    REQUIRE(r.states.empty());
}

TEST_CASE("SimulationResult: reserve does not change size",
          "[v2][layer5][result]") {
    SimulationResult r{};
    r.reserve(100);
    REQUIRE(r.empty());
    REQUIRE(r.num_steps() == 0);
    // Capacity is now >= 100, but size is still 0.
    REQUIRE(r.times.capacity() >= 100);
    REQUIRE(r.states.capacity() >= 100);
}

TEST_CASE("SimulationResult: push 3 samples → num_steps == 3",
          "[v2][layer5][result]") {
    SimulationResult r{};
    r.times.push_back(Real{0});
    r.states.push_back(Vector::Zero(2));
    r.times.push_back(Real{0.1});
    r.states.push_back(Vector::Zero(2));
    r.times.push_back(Real{0.2});
    r.states.push_back(Vector::Zero(2));

    REQUIRE_FALSE(r.empty());
    REQUIRE(r.num_steps() == 3);
    REQUIRE(r.times.size() == 3);
    REQUIRE(r.states.size() == 3);
}
