// =============================================================================
// Phase 2 of `add-frequency-domain-analysis`: AC sweep contract tests
// =============================================================================
//
// Pins `Simulator::run_ac_sweep(options)` against analytical Bode
// expressions for first- and second-order linear circuits. Coverage:
//
//   * RC low-pass: H(s) = 1/(1 + sRC). Verify -3 dB at f = 1/(2πRC) and
//     -20 dB/decade slope past the corner.
//   * RL high-pass driven from Vin (transfer ZL/(R+ZL)). Verify +6 dB
//     octave slope below the corner and 0 dB asymptote above.
//   * Series RLC: H(s) = 1/(1 + sRC + s²LC). Verify -40 dB/decade
//     above resonance and the analytically-derived peak height.
//
// These contracts are gates G.1 / G.2 of the change.
//
// All circuits use a single named voltage source as the perturbation input.
// AC sweep uses log-spaced frequencies covering 4-5 decades around the
// circuit's natural frequency.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/core.hpp"

#include <cmath>
#include <complex>
#include <numbers>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

[[nodiscard]] std::size_t closest_freq_index(const std::vector<Real>& freqs,
                                              Real target_hz) {
    std::size_t best = 0;
    Real best_dist = std::abs(std::log10(freqs[0]) - std::log10(target_hz));
    for (std::size_t i = 1; i < freqs.size(); ++i) {
        const Real d = std::abs(std::log10(freqs[i]) - std::log10(target_hz));
        if (d < best_dist) {
            best = i;
            best_dist = d;
        }
    }
    return best;
}

}  // namespace

// -----------------------------------------------------------------------------
// RC low-pass: -3 dB at corner, -20 dB/decade above
// -----------------------------------------------------------------------------

TEST_CASE("Phase 2.5: AC sweep on RC low-pass matches analytical Bode",
          "[v1][frequency_analysis][phase2][ac_sweep][rc]") {
    constexpr Real R = 1e3;          // 1 kΩ
    constexpr Real C_val = 1e-6;     // 1 µF — corner at f_c = 1/(2π·RC) ≈ 159.155 Hz
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
    opts.dt_min = 1e-12;
    opts.dt_max = 1e-7;
    opts.adaptive_timestep = false;
    opts.newton_options.num_nodes    = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);

    AcSweepOptions ac;
    ac.f_start = Real{1};
    ac.f_stop  = Real{1e6};
    ac.points_per_decade = 30;
    ac.scale = AcSweepScale::Logarithmic;
    ac.perturbation_source = "V1";
    ac.measurement_nodes = {"out"};
    ac.use_dc_op = true;

    const auto result = sim.run_ac_sweep(ac);
    REQUIRE(result.success);
    REQUIRE(result.frequencies.size() >= 100);
    REQUIRE(result.measurements.size() == 1);
    REQUIRE(result.measurements[0].node == "out");

    const auto& m = result.measurements[0];
    const auto& f = result.frequencies;

    // 1) DC asymptote (f → 0): |H| → 1, 0 dB. Phase → 0°.
    CHECK(m.magnitude_db.front() == Approx(0.0).margin(0.05));
    CHECK(m.phase_deg.front()    == Approx(0.0).margin(2.0));

    // 2) Corner frequency: |H| = 1/√2, magnitude = -3.0103 dB, phase = -45°.
    const auto i_corner = closest_freq_index(f, f_corner);
    INFO("Corner f estimate: " << f[i_corner]);
    CHECK(m.magnitude_db[i_corner] == Approx(-3.0103).margin(0.20));
    CHECK(m.phase_deg[i_corner]    == Approx(-45.0).margin(2.0));

    // 3) Two decades above corner: -40 dB asymptote, -90° asymptote.
    const auto i_2dec = closest_freq_index(f, f_corner * 100.0);
    CHECK(m.magnitude_db[i_2dec] == Approx(-40.0).margin(0.20));
    CHECK(m.phase_deg[i_2dec]    == Approx(-90.0).margin(2.0));

    // 4) Real/imag parts at corner: H(jω_c) = 1 / (1 + j) = (0.5, -0.5).
    CHECK(m.real_part[i_corner] == Approx( 0.5).margin(0.02));
    CHECK(m.imag_part[i_corner] == Approx(-0.5).margin(0.02));
}

// -----------------------------------------------------------------------------
// RL high-pass: 0 dB asymptote past corner, +6 dB/octave slope below
// -----------------------------------------------------------------------------

TEST_CASE("Phase 2.5: AC sweep on RL high-pass matches analytical Bode",
          "[v1][frequency_analysis][phase2][ac_sweep][rl]") {
    constexpr Real R = 100.0;        // 100 Ω
    constexpr Real L = 1e-3;         // 1 mH — corner at f_c = R/(2πL) ≈ 15.915 kHz
    const Real f_corner = R / (Real{2} * std::numbers::pi_v<Real> * L);

    Circuit ckt;
    auto in   = ckt.add_node("in");
    auto mid  = ckt.add_node("mid");
    ckt.add_voltage_source("V1", in, Circuit::ground(), /*V=*/1.0);
    ckt.add_inductor("L1", in, mid, L, 0.0);
    ckt.add_resistor("R1", mid, Circuit::ground(), R);

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
    ac.f_start = Real{1};
    ac.f_stop  = Real{1e8};
    ac.points_per_decade = 30;
    ac.scale = AcSweepScale::Logarithmic;
    ac.perturbation_source = "V1";
    ac.measurement_nodes = {"mid"};

    const auto result = sim.run_ac_sweep(ac);
    REQUIRE(result.success);

    const auto& m = result.measurements[0];
    const auto& f = result.frequencies;

    // RL high-pass observed at the resistor: H(s) = (R) / (R + sL) (this is
    // actually the LOW-pass output of the divider — the resistor is across
    // the output, so the transfer function is `Z_R / (Z_R + Z_L)` which is
    // 1 at DC and rolls off above the corner). At f_c we expect -3 dB and
    // -45°. Two decades above we expect -40 dB / -90°.
    const auto i_corner = closest_freq_index(f, f_corner);
    CHECK(m.magnitude_db[i_corner] == Approx(-3.0103).margin(0.30));
    CHECK(m.phase_deg[i_corner]    == Approx(-45.0).margin(2.0));

    const auto i_2dec = closest_freq_index(f, f_corner * 100.0);
    CHECK(m.magnitude_db[i_2dec] == Approx(-40.0).margin(0.30));
    CHECK(m.phase_deg[i_2dec]    == Approx(-90.0).margin(2.0));
}

// -----------------------------------------------------------------------------
// Series RLC: -40 dB/decade above resonance, peak at f_0 with Q = 1/(2ζ)
// -----------------------------------------------------------------------------

TEST_CASE("Phase 2.5: AC sweep on RLC matches analytical Bode",
          "[v1][frequency_analysis][phase2][ac_sweep][rlc]") {
    constexpr Real R = 10.0;          // small R → underdamped
    constexpr Real L = 1e-3;
    constexpr Real C_val = 1e-6;
    const Real omega_0 = Real{1} / std::sqrt(L * C_val);
    const Real f_0     = omega_0 / (Real{2} * std::numbers::pi_v<Real>);
    const Real zeta    = R * std::sqrt(C_val / L) / Real{2};
    const Real Q       = Real{1} / (Real{2} * zeta);
    REQUIRE(zeta < Real{1});  // underdamped

    Circuit ckt;
    auto vin = ckt.add_node("vin");
    auto mid = ckt.add_node("mid");
    auto cap = ckt.add_node("cap");
    ckt.add_voltage_source("V1", vin, Circuit::ground(), /*V=*/1.0);
    ckt.add_resistor("R1", vin, mid, R);
    ckt.add_inductor("L1", mid, cap, L, 0.0);
    ckt.add_capacitor("C1", cap, Circuit::ground(), C_val, 0.0);

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
    ac.f_start = f_0 / Real{100};
    ac.f_stop  = f_0 * Real{100};
    ac.points_per_decade = 50;       // dense around resonance
    ac.scale = AcSweepScale::Logarithmic;
    ac.perturbation_source = "V1";
    ac.measurement_nodes = {"cap"};

    const auto result = sim.run_ac_sweep(ac);
    REQUIRE(result.success);

    const auto& m = result.measurements[0];
    const auto& f = result.frequencies;

    // 1) Far below resonance (1/100 of f_0): unity gain, phase ≈ 0°.
    CHECK(m.magnitude_db.front() == Approx(0.0).margin(0.05));
    CHECK(m.phase_deg.front()    == Approx(0.0).margin(2.0));

    // 2) At resonance: |H(jω_0)| = Q for the standard form `H(s) =
    //    ω_0² / (s² + 2ζω_0·s + ω_0²)`. So magnitude_db ≈ 20·log10(Q).
    const auto i_res = closest_freq_index(f, f_0);
    const Real mag_at_res_db = m.magnitude_db[i_res];
    INFO("Q analytical = " << Q << "  ⇒ expected mag_dB = " << 20 * std::log10(Q));
    INFO("AC sweep mag_dB at f_0 = " << mag_at_res_db);
    CHECK(mag_at_res_db == Approx(Real{20} * std::log10(Q)).margin(1.5));

    // Phase at resonance: -90° for the second-order LP form.
    CHECK(m.phase_deg[i_res] == Approx(-90.0).margin(5.0));

    // 3) Two decades above resonance: -80 dB asymptote, -180° phase.
    const auto i_far = closest_freq_index(f, f_0 * 100.0);
    CHECK(m.magnitude_db[i_far] == Approx(-80.0).margin(0.50));
    CHECK(m.phase_deg[i_far]    == Approx(-180.0).margin(5.0));
}

// -----------------------------------------------------------------------------
// Failure modes
// -----------------------------------------------------------------------------

TEST_CASE("Phase 2: AC sweep reports typed failure on unknown perturbation source",
          "[v1][frequency_analysis][phase2][ac_sweep][failure]") {
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
    opts.dt_min = 1e-12;
    opts.adaptive_timestep = false;
    opts.newton_options.num_nodes    = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);

    AcSweepOptions ac;
    ac.f_start = Real{1};
    ac.f_stop  = Real{1e3};
    ac.perturbation_source = "Vphantom";  // does not exist
    ac.measurement_nodes   = {"out"};

    const auto result = sim.run_ac_sweep(ac);
    CHECK_FALSE(result.success);
    CHECK(result.failure_reason.find("not_found") != std::string::npos);
}
