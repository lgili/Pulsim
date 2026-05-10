// =============================================================================
// Phase 4 of `add-frequency-domain-analysis`: multi-input transfer-function
// matrix
// =============================================================================
//
// `AcSweepOptions::perturbation_sources` (vector) replaces the single
// `perturbation_source` for MIMO sweeps. The result carries one
// `AcMeasurement` per `(source, node)` pair, with both labels populated.
//
// Test setup: two voltage sources `Va` and `Vb` driving a passive RC
// network with one common output node `out`. Linear superposition says
// `H_out_a = H_out_b` because both sources see the same R-to-out path
// (we wire them in parallel). Phase 4 contract is that the two input
// columns produce two `AcMeasurement` entries each carrying the right
// `perturbation_source` label and matching Bode data within numerical
// noise.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/core.hpp"

#include <cmath>
#include <numbers>

using namespace pulsim::v1;
using Catch::Approx;

TEST_CASE("Phase 4: multi-input AC sweep returns one AcMeasurement per (source, node)",
          "[v1][frequency_analysis][phase4][mimo]") {
    Circuit ckt;
    auto a   = ckt.add_node("a");
    auto b   = ckt.add_node("b");
    auto out = ckt.add_node("out");

    // Two parallel voltage sources, each through its own resistor, summing
    // into the same output capacitor. Linear superposition makes the two
    // input-to-output transfer functions identical in this symmetric
    // setup.
    ckt.add_voltage_source("Va", a, Circuit::ground(), 1.0);
    ckt.add_voltage_source("Vb", b, Circuit::ground(), 1.0);
    ckt.add_resistor("Ra", a, out, 1e3);
    ckt.add_resistor("Rb", b, out, 1e3);
    ckt.add_capacitor("C1", out, Circuit::ground(), 1e-6, 0.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop  = 1e-6;
    opts.dt     = 1e-7;
    opts.dt_min = 1e-12;
    opts.dt_max = 1e-7;
    opts.adaptive_timestep = false;
    opts.newton_options.num_nodes    = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);

    AcSweepOptions ac;
    ac.f_start = 1.0;
    ac.f_stop  = 1e5;
    ac.points_per_decade = 5;
    ac.perturbation_sources = {"Va", "Vb"};   // ← Phase 4: two inputs
    ac.measurement_nodes    = {"out"};

    const auto result = sim.run_ac_sweep(ac);
    REQUIRE(result.success);
    REQUIRE(result.measurements.size() == 2);   // 1 output × 2 inputs

    // Each measurement carries both labels.
    const auto& m0 = result.measurements[0];
    const auto& m1 = result.measurements[1];
    CHECK(m0.node == "out");
    CHECK(m1.node == "out");
    CHECK(m0.perturbation_source == "Va");
    CHECK(m1.perturbation_source == "Vb");

    // By symmetry the two transfer functions must agree at every
    // frequency within tight numerical tolerance.
    REQUIRE(m0.magnitude_db.size() == m1.magnitude_db.size());
    REQUIRE(!m0.magnitude_db.empty());
    for (std::size_t i = 0; i < m0.magnitude_db.size(); ++i) {
        CHECK(m0.magnitude_db[i] == Approx(m1.magnitude_db[i]).margin(1e-9));
        CHECK(m0.phase_deg[i]    == Approx(m1.phase_deg[i]).margin(1e-9));
    }
}

// -----------------------------------------------------------------------------
// 4.3: 2-input × 2-output structure produces 4 measurements
// -----------------------------------------------------------------------------

TEST_CASE("Phase 4: 2-input × 2-output gives 4 H[i,j] entries",
          "[v1][frequency_analysis][phase4][mimo][shape]") {
    Circuit ckt;
    auto a   = ckt.add_node("a");
    auto b   = ckt.add_node("b");
    auto m1  = ckt.add_node("m1");
    auto m2  = ckt.add_node("m2");
    ckt.add_voltage_source("Va", a, Circuit::ground(), 1.0);
    ckt.add_voltage_source("Vb", b, Circuit::ground(), 1.0);
    ckt.add_resistor("Ra1", a, m1, 1e3);
    ckt.add_resistor("Rb2", b, m2, 1e3);
    ckt.add_capacitor("C1", m1, Circuit::ground(), 1e-6, 0.0);
    ckt.add_capacitor("C2", m2, Circuit::ground(), 1e-6, 0.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop  = 1e-6;
    opts.dt     = 1e-7;
    opts.dt_min = 1e-12;
    opts.dt_max = 1e-7;
    opts.adaptive_timestep = false;
    opts.newton_options.num_nodes    = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);

    AcSweepOptions ac;
    ac.f_start = 10.0;
    ac.f_stop  = 1e3;
    ac.points_per_decade = 3;
    ac.perturbation_sources = {"Va", "Vb"};
    ac.measurement_nodes    = {"m1", "m2"};

    const auto result = sim.run_ac_sweep(ac);
    REQUIRE(result.success);
    REQUIRE(result.measurements.size() == 4);  // 2 outputs × 2 inputs

    // Order is (output_slot, input_source) — see `run_ac_sweep` above.
    // m1 × {Va, Vb}, then m2 × {Va, Vb}.
    CHECK(result.measurements[0].node == "m1");
    CHECK(result.measurements[0].perturbation_source == "Va");
    CHECK(result.measurements[1].node == "m1");
    CHECK(result.measurements[1].perturbation_source == "Vb");
    CHECK(result.measurements[2].node == "m2");
    CHECK(result.measurements[2].perturbation_source == "Va");
    CHECK(result.measurements[3].node == "m2");
    CHECK(result.measurements[3].perturbation_source == "Vb");

    // Cross-coupling sanity: with the orthogonal R-C topology above,
    // Va does not drive m2 directly (no path) — H_m2_Va should be near 0.
    // Same for Vb on m1.
    const auto& m1_va = result.measurements[0];
    const auto& m1_vb = result.measurements[1];
    const auto& m2_va = result.measurements[2];
    const auto& m2_vb = result.measurements[3];

    // m1 sees Va but not Vb; magnitude ratio at low f is much greater
    // than 1 in dB (i.e. |m1_va| >> |m1_vb|).
    CHECK(m1_va.magnitude_db.front() > m1_vb.magnitude_db.front() + 50.0);
    CHECK(m2_vb.magnitude_db.front() > m2_va.magnitude_db.front() + 50.0);
}
