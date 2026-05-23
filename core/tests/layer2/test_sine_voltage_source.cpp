// =============================================================================
// Layer 2 V11 — SineVoltageSource device model tests
// =============================================================================
//
// Validates the new SineVoltageSource:
//   * `value_at(p, t)` produces v_dc + v_amp·sin(2π·f·t + φ).
//   * DevicePool stores params + branch-current row id.
//   * `add_sine_voltage_source` adds a Source-kind branch
//     with a branch-current unknown (same as VoltageSource).
//   * `compute_sine_b_extra` overlays the correct value.
//   * Integration: a sine source driving an RC load through
//     run_transient produces the expected first-order
//     response (high-pass response at f >> 1/RC, low-pass at
//     f << 1/RC).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/models/sine_voltage_source.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/sources/sine_b_extra.hpp"
#include "pulsim/topology/graph.hpp"

#include <cmath>
#include <numbers>

using namespace pulsim;
using namespace pulsim::builder;
using namespace pulsim::models;
using namespace pulsim::pwl;
using namespace pulsim::solver;
using namespace pulsim::sources;
using namespace pulsim::topology;
using Catch::Approx;

TEST_CASE("SineVoltageSource::value_at produces v_dc + amp·sin(ωt + φ)",
          "[v2][layer2_v11][sine_voltage_source][unit]") {
    SineVoltageSource::Params p{
        .v_dc = 0.0, .v_amplitude = 5.0,
        .frequency = 1.0, .phase = 0.0};
    // sin(0)=0  → 0 V
    REQUIRE(SineVoltageSource::value_at(p, 0.0) ==
            Approx(0.0).margin(1e-12));
    // sin(π/2)=1 at t = 1/(4f) = 0.25 s
    REQUIRE(SineVoltageSource::value_at(p, 0.25) ==
            Approx(5.0));
    // sin(π)=0 at t = 0.5
    REQUIRE(SineVoltageSource::value_at(p, 0.5) ==
            Approx(0.0).margin(1e-12));
    // sin(3π/2)=-1 at t = 0.75
    REQUIRE(SineVoltageSource::value_at(p, 0.75) ==
            Approx(-5.0));
    // sin(2π)=0 at t = 1.0 (full cycle)
    REQUIRE(SineVoltageSource::value_at(p, 1.0) ==
            Approx(0.0).margin(1e-12));
}

TEST_CASE("SineVoltageSource respects DC offset",
          "[v2][layer2_v11][sine_voltage_source][unit]") {
    SineVoltageSource::Params p{
        .v_dc = 12.0, .v_amplitude = 1.0,
        .frequency = 60.0, .phase = 0.0};
    // Average over an integer number of cycles is v_dc.
    Real sum = 0;
    constexpr int N = 1000;
    constexpr Real T = 1.0 / 60.0;
    for (int k = 0; k < N; ++k) {
        const Real t = (k + 0.5) * T / N;
        sum += SineVoltageSource::value_at(p, t);
    }
    const Real mean = sum / N;
    REQUIRE(mean == Approx(12.0).margin(0.01));
}

TEST_CASE("SineVoltageSource phase shift offsets waveform",
          "[v2][layer2_v11][sine_voltage_source][unit]") {
    // phase = π/2 → sine becomes cosine: sin(ωt + π/2) = cos(ωt).
    SineVoltageSource::Params p{
        .v_dc = 0.0, .v_amplitude = 1.0,
        .frequency = 1.0,
        .phase = std::numbers::pi_v<Real> * Real{0.5}};
    // cos(0) = 1
    REQUIRE(SineVoltageSource::value_at(p, 0.0) ==
            Approx(1.0));
    // cos(π/2) = 0 at t = 0.25
    REQUIRE(SineVoltageSource::value_at(p, 0.25) ==
            Approx(0.0).margin(1e-12));
    // cos(π) = -1 at t = 0.5
    REQUIRE(SineVoltageSource::value_at(p, 0.5) ==
            Approx(-1.0));
}

TEST_CASE("SineVoltageSource handles frequency=0 safely (returns v_dc)",
          "[v2][layer2_v11][sine_voltage_source][unit]") {
    SineVoltageSource::Params p{
        .v_dc = 3.3, .v_amplitude = 1.0,
        .frequency = 0.0, .phase = 0.0};
    REQUIRE(SineVoltageSource::value_at(p, 0.0) ==
            Approx(3.3));
    REQUIRE(SineVoltageSource::value_at(p, 100.0) ==
            Approx(3.3));
}

TEST_CASE("DevicePool stores SineVoltageSource params + branch-var",
          "[v2][layer2_v11][sine_voltage_source][unit]") {
    CircuitBuilder b;
    b.add_sine_voltage_source(
        "Vac", "n0", "gnd",
        /*v_dc=*/0.0, /*v_amplitude=*/220.0,
        /*frequency=*/50.0);

    REQUIRE(b.num_branches() == 1);
    REQUIRE(b.pool().kind_of(0) ==
            DevicePool::StoredKind::SineVoltageSource);
    REQUIRE(b.pool().sine_voltage_source_params(0)
                .v_amplitude == Approx(220.0));
    REQUIRE(b.pool().sine_voltage_source_params(0)
                .frequency == Approx(50.0));
    // state_size = 1 node + 1 source branch-current = 2.
    REQUIRE(b.pool().state_size(b.graph()) == 2);
}

TEST_CASE("compute_sine_b_extra produces correct overlay",
          "[v2][layer2_v11][sine_voltage_source][unit]") {
    CircuitBuilder b;
    b.add_sine_voltage_source(
        "Vac", "n0", "gnd",
        0.0, 10.0, /*frequency=*/1.0);

    // At t=0.25 (= π/2 phase) v_sine = +10 V →
    // b_extra[src_var] = -10.
    Vector b_extra =
        compute_sine_b_extra(b.pool(), b.graph(), 0.25);
    // src_var = num_nodes (1) + relative_offset (0) = 1.
    REQUIRE(b_extra[1] == Approx(-10.0));

    // At t=0.75 (= 3π/2 phase) v_sine = -10 →
    // b_extra[src_var] = +10.
    b_extra = compute_sine_b_extra(b.pool(), b.graph(), 0.75);
    REQUIRE(b_extra[1] == Approx(+10.0));
}

// -----------------------------------------------------------------------------
// Integration: sine source driving a resistor — output
// tracks the input sine.
// -----------------------------------------------------------------------------

TEST_CASE("Sine source drives a resistor — node tracks sine",
          "[v2][layer2_v11][sine_voltage_source][integration]") {
    constexpr Real V_amp = 10.0;
    constexpr Real f_ac  = 60.0;      // 60 Hz
    constexpr Real T_ac  = 1.0 / f_ac;

    CircuitBuilder b;
    b.add_sine_voltage_source(
        "Vac", "n0", "gnd",
        /*v_dc=*/0.0, V_amp, f_ac);
    b.add_resistor("R", "n0", "gnd", 100.0);

    constexpr Real dt   = 1e-5;
    constexpr Real tend = 3.0 * T_ac;

    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build(dt);

    SimulationOptions opts{
        .t_start = 0.0, .t_end = tend, .dt = dt};
    auto switch_fn = [](Real) {
        return SwitchStateMask(0);
    };

    auto result = run_transient(
        cache, b.graph(), b.pool(), opts, switch_fn);
    REQUIRE(result.num_steps() > 100);

    // Compare result.states[k][0] (v_n0) to the analytical
    // sine at each t. RMS error should be tiny because the
    // load is purely resistive — no LC dynamics.
    Real ssq = 0;
    Size n_compared = 0;
    for (Size k = 0; k < result.num_steps(); ++k) {
        const Real t = result.times[k];
        const Real v_sim = result.states[k][0];
        const Real v_exp = V_amp *
            std::sin(Real{2} * std::numbers::pi_v<Real>
                     * f_ac * t);
        ssq += (v_sim - v_exp) * (v_sim - v_exp);
        ++n_compared;
    }
    const Real rms = std::sqrt(ssq /
        static_cast<Real>(n_compared));
    INFO("RMS error: " << rms << " V (V_amp = " << V_amp
         << ", expect << " << V_amp << ")");
    // RMS error should be << 0.1% of the amplitude.
    REQUIRE(rms < V_amp * 0.01);
}

TEST_CASE("Sine source through RC — first-order low-pass response",
          "[v2][layer2_v11][sine_voltage_source][integration]") {
    // RC time constant: τ = 1ms. Drive at f=60Hz where
    // ωτ = 2π·60·1e-3 ≈ 0.377 (well below cutoff).
    // Output gain ≈ 1/√(1 + (ωτ)²) ≈ 0.936.
    // Phase lag ≈ -atan(ωτ) ≈ -0.36 rad ≈ -20.7°.
    constexpr Real V_amp = 10.0;
    constexpr Real f_ac  = 60.0;
    constexpr Real T_ac  = 1.0 / f_ac;
    constexpr Real R = 1000.0;
    constexpr Real C = 1e-6;
    const Real omega = Real{2} * std::numbers::pi_v<Real> * f_ac;
    const Real omega_tau = omega * R * C;
    const Real expected_gain =
        Real{1} / std::sqrt(Real{1} + omega_tau * omega_tau);

    CircuitBuilder b;
    b.add_sine_voltage_source(
        "Vac", "n0", "gnd", 0.0, V_amp, f_ac);
    b.add_resistor("R", "n0", "vout", R);
    b.add_capacitor("Cf", "vout", "gnd", C);

    constexpr Real dt   = 1e-6;
    constexpr Real tend = 10.0 * T_ac;   // settle the C
    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build(dt);

    SimulationOptions opts{
        .t_start = 0.0, .t_end = tend, .dt = dt};
    auto switch_fn = [](Real) {
        return SwitchStateMask(0);
    };
    auto result = run_transient(
        cache, b.graph(), b.pool(), opts, switch_fn);

    // Take amplitude of vout over the LAST 2 cycles.
    const Index vout_idx = b.node_id_of("vout");
    REQUIRE(vout_idx >= 0);
    const Size k_start = result.num_steps() -
        static_cast<Size>(2.0 * T_ac / dt);
    Real vmin = 1e9, vmax = -1e9;
    for (Size k = k_start; k < result.num_steps(); ++k) {
        const Real v = result.states[k][vout_idx];
        vmin = std::min(vmin, v);
        vmax = std::max(vmax, v);
    }
    const Real vout_amp = (vmax - vmin) * Real{0.5};
    const Real measured_gain = vout_amp / V_amp;
    INFO("Measured gain: " << measured_gain
         << " (expected " << expected_gain << ")");
    REQUIRE(measured_gain == Approx(expected_gain)
                                  .margin(0.02));
}
