// =============================================================================
// Layer 4 — topology preflight + auto-regularization
// =============================================================================
//
// v2.0 Phase 2 (B.1), audit finding `no-topology-preflight-or-auto-
// shunt` (CRITICAL).
//
// The bar: a circuit whose only defect is a node nobody referenced
// must simulate, with the fix reported rather than applied in
// silence — and a circuit that was already fine must be left
// bit-for-bit alone.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/preflight.hpp"
#include "pulsim/solver/run_transient.hpp"

#include <string>

using namespace pulsim;
using namespace pulsim::pwl;
using Catch::Approx;

namespace {

bool has(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

/// Isolated transformer secondary — the canonical failure the
/// docs have been telling users to patch by hand.
builder::CircuitBuilder isolated_secondary() {
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", 12.0);
    b.add_resistor("Rp", "vin", "p1", 0.1);
    b.add_transformer("T1", "p1", "gnd", "s1", "s_gnd",
                       1e-3, 4e-3, 0.98);
    b.add_resistor("Rs", "s1", "s_gnd", 10.0);
    return b;
}

}  // namespace

TEST_CASE("Preflight leaves a well-posed circuit completely alone",
          "[v2][layer4][preflight]") {
    // The common case must cost nothing and change nothing.
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", 12.0);
    b.add_resistor("R1", "vin", "vout", 1.0);
    b.add_capacitor("C1", "vout", "gnd", 1e-6);
    b.add_inductor("L1", "vout", "gnd", 1e-3);
    const Index branches_before = b.graph().num_branches();

    const auto report = b.run_preflight();
    REQUIRE(report.empty());
    REQUIRE(report.num_fixed() == 0);
    REQUIRE(report.summary().empty());
    REQUIRE(b.graph().num_branches() == branches_before);
}

TEST_CASE("Preflight finds and ties an isolated subnet",
          "[v2][layer4][preflight]") {
    auto b = isolated_secondary();
    const Index before = b.graph().num_branches();

    const auto report = b.run_preflight();
    REQUIRE(report.findings.size() == 1);
    const auto& f = report.findings.front();
    REQUIRE(f.issue == PreflightIssue::IsolatedSubnet);
    REQUIRE(f.component.size() == 2);        // s1 and s_gnd
    REQUIRE(f.was_fixed());
    REQUIRE(f.inserted_resistance == Approx(1e9));

    // The message must name the node and say what was done.
    REQUIRE(has(f.detail, "s1"));
    REQUIRE(has(f.detail, "isolated"));
    REQUIRE(has(f.detail, "Pulsim inserted"));

    // Exactly one branch was added, and it is named after the node.
    REQUIRE(b.graph().num_branches() == before + 1);
    REQUIRE(has(std::string{b.graph().branch_name(before)},
                 "R_auto_iso_"));
}

TEST_CASE("Preflight is idempotent",
          "[v2][layer4][preflight]") {
    auto b = isolated_secondary();
    REQUIRE(b.run_preflight().findings.size() == 1);
    const Index after_first = b.graph().num_branches();
    // The subnet now reaches ground, so there is nothing left to do.
    REQUIRE(b.run_preflight().empty());
    REQUIRE(b.graph().num_branches() == after_first);
}

TEST_CASE("Preflight can report without touching the circuit",
          "[v2][layer4][preflight]") {
    auto b = isolated_secondary();
    const Index before = b.graph().num_branches();
    PreflightOptions opts;
    opts.auto_regularize = false;

    const auto report = b.run_preflight(opts);
    REQUIRE(report.findings.size() == 1);
    REQUIRE_FALSE(report.findings.front().was_fixed());
    REQUIRE(report.num_fixed() == 0);
    REQUIRE(b.graph().num_branches() == before);   // untouched
}

TEST_CASE("Preflight finds a node with no DC path to ground",
          "[v2][layer4][preflight]") {
    // `vfloat` hangs off a capacitor only. Galvanically it IS
    // connected, so the isolated-subnet pass misses it — the DC pass
    // is what catches it, and the two must not double-report.
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", 12.0);
    b.add_resistor("R1", "vin", "gnd", 10.0);
    b.add_capacitor("Cfloat", "vin", "vfloat", 1e-6);

    const auto report = b.run_preflight();
    REQUIRE(report.findings.size() == 1);
    const auto& f = report.findings.front();
    REQUIRE(f.issue == PreflightIssue::NoDcPathToGround);
    REQUIRE(has(f.detail, "vfloat"));
    REQUIRE(has(f.detail, "no DC path"));
    REQUIRE(f.was_fixed());
}

TEST_CASE("Preflight handles several independent subnets",
          "[v2][layer4][preflight]") {
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", 12.0);
    b.add_resistor("R1", "vin", "gnd", 10.0);
    // Two separate floating islands, each internally connected.
    b.add_resistor("Ra", "isl_a1", "isl_a2", 1.0);
    b.add_resistor("Rb", "isl_b1", "isl_b2", 1.0);

    const auto report = b.run_preflight();
    REQUIRE(report.findings.size() == 2);
    REQUIRE(report.num_fixed() == 2);
    for (const auto& f : report.findings) {
        REQUIRE(f.issue == PreflightIssue::IsolatedSubnet);
        REQUIRE(f.component.size() == 2);
    }
    // One tie each — not one per node.
    REQUIRE(b.graph().num_branches() == 4 + 2);
}

TEST_CASE("An auto-tied circuit simulates, and matches a hand-tied one",
          "[v2][layer4][preflight][integration]") {
    // THE point of the feature. Same circuit twice: once relying on
    // the pass, once with the resistor the docs told users to type.
    // The waveforms must agree — a 1 GΩ reference is electrically
    // invisible, which is exactly why the tie must be LARGE. (A
    // small tie would be a galvanic bond and these would diverge.)
    auto b_auto = isolated_secondary();
    const auto report = b_auto.run_preflight();
    REQUIRE(report.num_fixed() == 1);

    builder::CircuitBuilder b_manual;
    b_manual.add_voltage_source("Vin", "vin", "gnd", 12.0);
    b_manual.add_resistor("Rp", "vin", "p1", 0.1);
    b_manual.add_transformer("T1", "p1", "gnd", "s1", "s_gnd",
                              1e-3, 4e-3, 0.98);
    b_manual.add_resistor("Rs", "s1", "s_gnd", 10.0);
    b_manual.add_resistor("R_iso", "s1", "gnd", 1e9);

    solver::SimulationOptions opts;
    opts.t_start = 0.0;
    opts.t_end   = 2e-4;
    opts.dt      = 1e-6;
    solver::SwitchScheduleFn sw = [](Real) {
        return topology::SwitchStateMask(0);
    };

    PwlStateSpaceCache c_auto{b_auto.graph(), b_auto.pool()};
    c_auto.build_lazy(opts.dt);
    auto r_auto = solver::run_transient(c_auto, b_auto.graph(),
                                          b_auto.pool(), opts, sw);

    PwlStateSpaceCache c_man{b_manual.graph(), b_manual.pool()};
    c_man.build_lazy(opts.dt);
    auto r_man = solver::run_transient(c_man, b_manual.graph(),
                                        b_manual.pool(), opts, sw);

    REQUIRE(r_auto.num_steps() == r_man.num_steps());
    // Node ids line up: both circuits declare the same nodes in the
    // same order; only the tie's branch differs.
    const auto n = b_auto.graph().num_nodes();
    for (Size k = 0; k < r_auto.num_steps(); ++k) {
        for (Index i = 0; i < n; ++i) {
            INFO("sample " << k << " node " << i);
            REQUIRE(r_auto.states[k][i] ==
                    Approx(r_man.states[k][i]).margin(1e-9));
        }
    }
}

TEST_CASE("The tie is high-impedance: it must not load the node",
          "[v2][layer4][preflight]") {
    // A 12 V node tied through 1 GΩ draws 12 nA. Assert the
    // conductance actually stamped, so a future edit that "rounds"
    // the default down to something convenient fails here.
    auto b = isolated_secondary();
    const Index before = b.graph().num_branches();
    (void)b.run_preflight();
    const auto& p = b.pool().resistor_params(before);
    REQUIRE(p.G == Approx(1e-9));
    REQUIRE(b.pool().kind_of(before) ==
             DevicePool::StoredKind::Resistor);
}
