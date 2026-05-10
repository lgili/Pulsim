// =============================================================================
// Phase 8 of `add-frequency-domain-analysis`: validation suite
// =============================================================================
//
// 8.1 RC / RL / RLC analytical Bode parity is already pinned by Phase 2.5
// (`test_frequency_analysis_phase2.cpp`).
//
// 8.2 Buck open-loop transfer function matches the textbook second-order
// LC-filter response: `V_out(s) / V_in(s) = 1 / (1 + s·L/R_load + s²·L·C)`.
// We exercise it via the buck *output stage* (LC filter into resistive
// load) — that's the linearization the small-signal averaged buck
// reduces to, identical in form to the published `H(s)`.
//
// 8.3 Buck closed-loop loop gain is gated on AD-driven Behavioral
// linearization (Phase 1.2), which is deferred. The PWL state-space
// path linearizes only Ideal-mode switches, where the duty cycle is a
// PWM source — perturbing that source's frequency / amplitude doesn't
// directly correspond to a duty perturbation. Once Phase 1.2 lands an
// AD residual-Jacobian path for Behavioral devices, this test extends
// to the standard buck control-to-output transfer function.
//
// 8.4 Three-phase grid impedance analyzer is post-grid-library landing
// (`add-three-phase-grid-library`); deferred here.

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

// Analytical second-order LC response with parallel resistive load:
//   H(s) = R / (R + s·L + s²·R·L·C)  -- normalized by output voltage
// Equivalent transfer function. Compute magnitude_db / phase_deg at f.
struct LcAnalytical {
    Real L;
    Real C;
    Real R;
    [[nodiscard]] std::pair<Real, Real> at(Real f) const {
        const Real omega = Real{2} * std::numbers::pi_v<Real> * f;
        // Z_C = 1/(jωC), Z_L = jωL.
        // Output is V_C; series chain V_in → L → V_out, then R || C to gnd.
        // V_out / V_in = (R || (1/jωC)) / (jωL + R || (1/jωC))
        const std::complex<Real> jw{0, omega};
        const std::complex<Real> Zc = Real{1} / (jw * C);
        const std::complex<Real> Zr{R, 0};
        const std::complex<Real> Zl = jw * L;
        const std::complex<Real> Zload = (Zr * Zc) / (Zr + Zc);
        const std::complex<Real> H = Zload / (Zl + Zload);
        const Real mag_db = Real{20} * std::log10(std::abs(H));
        const Real phase_deg =
            std::arg(H) * Real{180} / std::numbers::pi_v<Real>;
        return {mag_db, phase_deg};
    }
};

}  // namespace

// -----------------------------------------------------------------------------
// 8.2: Buck output-stage LC filter matches analytical H(s)
// -----------------------------------------------------------------------------

TEST_CASE("Phase 8.2: buck output-stage LC filter matches analytical Bode",
          "[v1][frequency_analysis][phase8][validation][buck_lc]") {
    constexpr Real L_val = 47e-6;     // 47 µH inductor
    constexpr Real C_val = 100e-6;    // 100 µF output cap
    constexpr Real R_val = 5.0;       // 5 Ω load
    const LcAnalytical lc{L_val, C_val, R_val};
    const Real f_resonance =
        Real{1} / (Real{2} * std::numbers::pi_v<Real> * std::sqrt(L_val * C_val));
    // Resonance ≈ 2.32 kHz for this LC.

    Circuit ckt;
    auto in   = ckt.add_node("in");
    auto sw   = ckt.add_node("sw");
    auto out  = ckt.add_node("out");
    ckt.add_voltage_source("Vin", in, Circuit::ground(), 12.0);
    ckt.add_inductor("L1", in, sw, L_val, 0.0);
    ckt.add_capacitor("C1", sw, Circuit::ground(), 1e-12, 0.0);
    // Tiny "input" cap to ground at sw is just for numerical regularity;
    // real switching node would be the duty-modulated rail. For the
    // LINEAR LC analysis we feed Vin directly through L into the output
    // capacitor + load — see below for the simpler topology.
    (void)sw;
    (void)out;

    // Simpler topology: Vin → L → out, with C and R from out to ground.
    Circuit lc_ckt;
    auto vin = lc_ckt.add_node("vin");
    auto vout = lc_ckt.add_node("vout");
    lc_ckt.add_voltage_source("Vin", vin, lc_ckt.ground(), 12.0);
    lc_ckt.add_inductor("L1", vin, vout, L_val, 0.0);
    lc_ckt.add_capacitor("C1", vout, lc_ckt.ground(), C_val, 0.0);
    lc_ckt.add_resistor("Rload", vout, lc_ckt.ground(), R_val);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop  = 1e-6;
    opts.dt     = 1e-7;
    opts.dt_min = 1e-12;
    opts.dt_max = 1e-7;
    opts.adaptive_timestep = false;
    opts.newton_options.num_nodes    = lc_ckt.num_nodes();
    opts.newton_options.num_branches = lc_ckt.num_branches();

    Simulator sim(lc_ckt, opts);

    AcSweepOptions ac;
    ac.f_start = f_resonance / Real{100};
    ac.f_stop  = f_resonance * Real{100};
    ac.points_per_decade = 30;
    ac.scale = AcSweepScale::Logarithmic;
    ac.perturbation_source = "Vin";
    ac.measurement_nodes   = {"vout"};

    const auto result = sim.run_ac_sweep(ac);
    REQUIRE(result.success);
    REQUIRE(result.measurements.size() == 1);

    const auto& m = result.measurements[0];

    // Three reference points: well below resonance (DC ≈ 0 dB), at
    // resonance (peak determined by Q = R·sqrt(C/L)), and well above
    // (-40 dB / decade roll-off).
    const auto i_lo  = closest_freq_index(result.frequencies, f_resonance / 100.0);
    const auto i_res = closest_freq_index(result.frequencies, f_resonance);
    const auto i_hi  = closest_freq_index(result.frequencies, f_resonance * 100.0);

    {
        const auto [mag_a, phase_a] = lc.at(result.frequencies[i_lo]);
        INFO("low: AC sweep mag=" << m.magnitude_db[i_lo]
             << " analytical=" << mag_a);
        CHECK(m.magnitude_db[i_lo] == Approx(mag_a).margin(0.30));
        CHECK(m.phase_deg[i_lo]    == Approx(phase_a).margin(2.0));
    }
    {
        const auto [mag_a, phase_a] = lc.at(result.frequencies[i_res]);
        INFO("resonance: AC sweep mag=" << m.magnitude_db[i_res]
             << " analytical=" << mag_a);
        CHECK(m.magnitude_db[i_res] == Approx(mag_a).margin(0.50));
        CHECK(m.phase_deg[i_res]    == Approx(phase_a).margin(5.0));
    }
    {
        const auto [mag_a, phase_a] = lc.at(result.frequencies[i_hi]);
        INFO("high: AC sweep mag=" << m.magnitude_db[i_hi]
             << " analytical=" << mag_a);
        CHECK(m.magnitude_db[i_hi] == Approx(mag_a).margin(0.50));
        CHECK(m.phase_deg[i_hi]    == Approx(phase_a).margin(5.0));
    }
}
