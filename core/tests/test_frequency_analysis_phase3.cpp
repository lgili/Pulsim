// =============================================================================
// Phase 3 of `add-frequency-domain-analysis`: FRA contract tests
// =============================================================================
//
// FRA = Frequency Response Analysis: empirical Bode measurement via
// transient + perturbation injection + Goertzel DFT. Phase 3.5's central
// contract: on linear circuits, FRA must agree with the analytical
// `run_ac_sweep` within ≤ 1 dB / ≤ 5° at every sweep point. Nonlinear /
// PWM-loop comparisons are downstream Phase 8.x deliverables.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/core.hpp"

#include <cmath>
#include <numbers>

using namespace pulsim::v1;
using Catch::Approx;

// -----------------------------------------------------------------------------
// FRA on RC low-pass: agrees with AC sweep within 1 dB / 5°
// -----------------------------------------------------------------------------

TEST_CASE("Phase 3.5: FRA on RC low-pass agrees with AC sweep within tolerance",
          "[v1][frequency_analysis][phase3][fra][parity]") {
    constexpr Real R = 1e3;
    constexpr Real C_val = 1e-6;
    const Real f_corner = Real{1} / (Real{2} * std::numbers::pi_v<Real> * R * C_val);

    Circuit ckt;
    auto in  = ckt.add_node("in");
    auto out = ckt.add_node("out");
    ckt.add_voltage_source("V1", in, Circuit::ground(), /*V=*/1.0);
    ckt.add_resistor("R1", in, out, R);
    ckt.add_capacitor("C1", out, Circuit::ground(), C_val, 0.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop  = 1e-6;
    opts.dt     = 1e-7;
    opts.dt_min = 1e-15;
    opts.dt_max = 1e-7;
    opts.adaptive_timestep = false;
    opts.newton_options.num_nodes    = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);

    // 1) AC sweep — analytical reference. Three frequencies straddling the
    //    corner: a decade below, at the corner, a decade above. FRA needs
    //    to land on the same Bode points.
    AcSweepOptions ac;
    ac.f_start = f_corner / Real{10};
    ac.f_stop  = f_corner * Real{10};
    ac.points_per_decade = 4;
    ac.scale = AcSweepScale::Logarithmic;
    ac.perturbation_source = "V1";
    ac.measurement_nodes = {"out"};

    const auto ac_result = sim.run_ac_sweep(ac);
    REQUIRE(ac_result.success);
    REQUIRE(ac_result.frequencies.size() >= 5);

    // 2) FRA at the same frequency grid. Use the AC sweep options we just
    //    built as the FRA grid template.
    FraOptions fra;
    fra.f_start = ac.f_start;
    fra.f_stop  = ac.f_stop;
    fra.points_per_decade = ac.points_per_decade;
    fra.scale = AcSweepScale::Logarithmic;
    fra.perturbation_source    = "V1";
    fra.perturbation_amplitude = 1e-2;
    fra.perturbation_phase     = 0.0;
    fra.measurement_nodes = {"out"};
    fra.n_cycles          = 6;
    fra.discard_cycles    = 2;
    fra.samples_per_cycle = 64;

    const auto fra_result = sim.run_fra(fra);
    REQUIRE(fra_result.success);
    REQUIRE(fra_result.frequencies.size() == ac_result.frequencies.size());
    REQUIRE(fra_result.measurements.size() == 1);

    const auto& ac_meas  = ac_result.measurements[0];
    const auto& fra_meas = fra_result.measurements[0];

    // 3) Phase 3.5 contract: FRA agrees with AC sweep within 1 dB / 5° at
    //    every frequency on linear circuits. Tolerance is chosen so that
    //    Goertzel quantization (samples_per_cycle = 64 → ≈ 5° resolution)
    //    sits well inside the budget but real numerical noise still has
    //    margin.
    REQUIRE(ac_meas.magnitude_db.size() == fra_meas.magnitude_db.size());
    for (std::size_t i = 0; i < ac_meas.magnitude_db.size(); ++i) {
        const Real f = ac_result.frequencies[i];
        INFO("Frequency point " << i << " = " << f << " Hz");
        INFO("  AC  mag = " << ac_meas.magnitude_db[i]
             << " dB  phase = " << ac_meas.phase_deg[i] << "°");
        INFO("  FRA mag = " << fra_meas.magnitude_db[i]
             << " dB  phase = " << fra_meas.phase_deg[i] << "°");
        CHECK(fra_meas.magnitude_db[i]
              == Approx(ac_meas.magnitude_db[i]).margin(1.0));
        CHECK(fra_meas.phase_deg[i]
              == Approx(ac_meas.phase_deg[i]).margin(5.0));
    }
}

// -----------------------------------------------------------------------------
// FRA fails gracefully when DC OP can't establish
// -----------------------------------------------------------------------------

TEST_CASE("Phase 3: FRA reports typed failure on unknown perturbation source",
          "[v1][frequency_analysis][phase3][fra][failure]") {
    Circuit ckt;
    auto in  = ckt.add_node("in");
    auto out = ckt.add_node("out");
    ckt.add_voltage_source("V1", in, Circuit::ground(), 1.0);
    ckt.add_resistor("R1", in, out, 1e3);
    ckt.add_capacitor("C1", out, Circuit::ground(), 1e-6, 0.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop  = 1e-6;
    opts.dt     = 1e-7;
    opts.dt_min = 1e-15;
    opts.adaptive_timestep = false;
    opts.newton_options.num_nodes    = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);

    FraOptions fra;
    fra.f_start = Real{10};
    fra.f_stop  = Real{1e3};
    fra.points_per_decade = 2;
    fra.perturbation_source = "Vphantom";
    fra.measurement_nodes   = {"out"};

    const auto result = sim.run_fra(fra);
    CHECK_FALSE(result.success);
    CHECK(result.failure_reason.find("not_found") != std::string::npos);
}

// -----------------------------------------------------------------------------
// Goertzel sanity: detects fundamental at a single frequency
// -----------------------------------------------------------------------------

TEST_CASE("Phase 3: FRA captures DC + perturbation, returns small-signal H(jω)",
          "[v1][frequency_analysis][phase3][fra][dc_offset]") {
    constexpr Real R = 1e3;
    constexpr Real C_val = 1e-6;
    constexpr Real V_dc = 5.0;       // non-trivial DC offset

    Circuit ckt;
    auto in  = ckt.add_node("in");
    auto out = ckt.add_node("out");
    ckt.add_voltage_source("V1", in, Circuit::ground(), V_dc);
    ckt.add_resistor("R1", in, out, R);
    ckt.add_capacitor("C1", out, Circuit::ground(), C_val, 0.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop  = 1e-6;
    opts.dt     = 1e-7;
    opts.dt_min = 1e-15;
    opts.adaptive_timestep = false;
    opts.newton_options.num_nodes    = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);

    // Run FRA at exactly the corner — should report ≈ -3 dB / -45°
    // regardless of the DC offset on V1 because the linearization is
    // shift-invariant about the operating point.
    const Real f_corner = Real{1} / (Real{2} * std::numbers::pi_v<Real> * R * C_val);
    FraOptions fra;
    fra.f_start = f_corner;
    fra.f_stop  = f_corner;
    fra.points_per_decade = 1;
    fra.scale = AcSweepScale::Logarithmic;
    fra.perturbation_source    = "V1";
    fra.perturbation_amplitude = 1e-2;
    fra.measurement_nodes      = {"out"};
    fra.n_cycles          = 8;
    fra.discard_cycles    = 3;
    fra.samples_per_cycle = 128;

    const auto fra_result = sim.run_fra(fra);
    REQUIRE(fra_result.success);
    REQUIRE(fra_result.measurements.size() == 1);
    REQUIRE(fra_result.frequencies.size() == 2);  // generator emits min 2 points

    // Take the first (or any) frequency — they're all at the corner.
    const auto& m = fra_result.measurements[0];
    CHECK(m.magnitude_db.front() == Approx(-3.0103).margin(1.0));
    CHECK(m.phase_deg.front()    == Approx(-45.0).margin(5.0));
}
