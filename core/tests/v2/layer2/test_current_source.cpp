// =============================================================================
// Layer 2 V3 — CurrentSource device model tests
// =============================================================================
//
// Validates the new CurrentSource:
//   * Model `current(v, p)` returns the configured I.
//   * DevicePool stores and retrieves CurrentSource params.
//   * `add_current_source` adds a Source-kind branch
//     with no branch-var (unlike VoltageSource).
//   * Stamped current source produces the expected node
//     voltage on a simple resistive divider (current-source
//     load test).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/builder/circuit_builder.hpp"
#include "pulsim/v2/models/current_source.hpp"
#include "pulsim/v2/pwl/cache.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/topology/graph.hpp"

using namespace pulsim::v2;
using namespace pulsim::v2::builder;
using namespace pulsim::v2::models;
using namespace pulsim::v2::pwl;
using namespace pulsim::v2::topology;
using Catch::Approx;

TEST_CASE("CurrentSource::current returns the parameter I",
          "[v2][layer2_v3][current_source][unit]") {
    CurrentSource::Params p{.I = 2.5};
    Real v[2] = {0.0, 0.0};
    REQUIRE(CurrentSource::current(v, p) == Approx(2.5));

    // Independent of v.
    v[0] = 100.0;
    v[1] = -50.0;
    REQUIRE(CurrentSource::current(v, p) == Approx(2.5));
}

TEST_CASE("DevicePool stores and retrieves CurrentSource params",
          "[v2][layer2_v3][current_source][unit]") {
    DevicePool pool;
    pool.add_current_source(
        /*branch_id=*/3, CurrentSource::Params{.I = 1.5});

    REQUIRE(pool.kind_of(3) ==
              DevicePool::StoredKind::CurrentSource);
    REQUIRE(pool.current_source_params(3).I ==
              Approx(1.5));
}

TEST_CASE("CircuitBuilder.add_current_source adds a Source branch",
          "[v2][layer2_v3][current_source][builder]") {
    CircuitBuilder b;
    b.add_current_source("Ibias", "n0", "gnd", 0.01);

    REQUIRE(b.num_branches() == 1);
    REQUIRE(b.pool().kind_of(0) ==
              DevicePool::StoredKind::CurrentSource);
    REQUIRE(b.pool().current_source_params(0).I ==
              Approx(0.01));

    // The Graph's branch should be of Source kind.
    REQUIRE(b.graph().branch(0).kind ==
              BranchKind::Source);
}

TEST_CASE("CurrentSource into a resistor — Ohm's law verification",
          "[v2][layer2_v3][current_source][integration]") {
    // I_source (10 mA) → R (1 kΩ) → GND.
    // Expected: v_n0 = I · R = 10 mA · 1 kΩ = 10 V.
    CircuitBuilder b;
    b.add_current_source("Ibias", "n0", "gnd", 10e-3);
    b.add_resistor      ("R1",    "n0", "gnd", 1000.0);

    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build();   // static path
    const auto& seg = cache.lookup(SwitchStateMask(0));

    Vector x = Vector::Zero(seg.state_size);
    Vector b_extra = Vector::Zero(seg.state_size);
    cache.solve(SwitchStateMask(0), b_extra, x);

    INFO("v_n0 = " << x[0] << " V (expected 10 V)");
    REQUIRE(x[0] == Approx(10.0).margin(1e-9));
}

TEST_CASE("CurrentSource has NO branch-current unknown",
          "[v2][layer2_v3][current_source][structure]") {
    // VoltageSource adds 1 to state_size (the branch
    // current unknown). CurrentSource does NOT.
    CircuitBuilder b1;
    b1.add_voltage_source("V", "n0", "gnd", 5.0);
    b1.add_resistor("R", "n0", "gnd", 1.0);
    const Size state_v =
        b1.pool().state_size(b1.graph());
    // state_v = num_nodes(1) + num_voltage_sources(1)
    //         + num_inductors(0) = 2.
    REQUIRE(state_v == 2);

    CircuitBuilder b2;
    b2.add_current_source("I", "n0", "gnd", 1.0);
    b2.add_resistor("R", "n0", "gnd", 1.0);
    const Size state_i =
        b2.pool().state_size(b2.graph());
    // CurrentSource does NOT contribute a branch-current
    // unknown, so state_i = num_nodes(1) + 0 + 0 = 1.
    REQUIRE(state_i == 1);
}

TEST_CASE("CurrentSource sign convention: I from `from` to `to`",
          "[v2][layer2_v3][current_source][signs]") {
    // I_source (5 mA) flows FROM gnd TO n0.
    // KCL at n0: -I_source (entering) + v_n0/R (leaving) = 0
    //          → v_n0 = -I · R = -5 mA · 1 kΩ = -5 V.
    CircuitBuilder b;
    b.add_current_source("Iup", "gnd", "n0", 5e-3);
    b.add_resistor      ("R1",  "n0",  "gnd", 1000.0);

    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build();

    Vector x = Vector::Zero(
        b.pool().state_size(b.graph()));
    Vector b_extra = Vector::Zero(x.size());
    cache.solve(SwitchStateMask(0), b_extra, x);

    INFO("v_n0 (with I from gnd to n0) = " << x[0]
         << " V (expected -5 V)");
    REQUIRE(x[0] == Approx(-5.0).margin(1e-9));
}
