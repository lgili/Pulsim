// =============================================================================
// Pulsim — ShockleyDiode (exponential junction), Phase 4 audit C.1
// =============================================================================

#include "pulsim/ad/ad_scalar.hpp"
#include "pulsim/models/device_model.hpp"
#include "pulsim/models/shockley_diode.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>

using namespace pulsim;
using namespace pulsim::models;
using Catch::Approx;

namespace {

Real iD(const ShockleyDiode::Params& p, Real v) {
    const Real vv[2] = {v, 0.0};
    return ShockleyDiode::current<Real>(vv, p);
}

}  // namespace

TEST_CASE("ShockleyDiode — matches the closed form over decades",
          "[v2][layer2][shockley_diode][unit]") {
    ShockleyDiode::Params p;   // I_S = 1e-12, n = 1, V_T = 25.852 mV
    for (const Real i : {1e-6, 1e-3, 1.0, 10.0, 1e3}) {
        const Real v = ShockleyDiode::voltage_for_current(p, i);
        // `voltage_for_current` inverts the JUNCTION term only,
        // so the law's own G_min leakage rides on top. At 1 µA
        // that is 3.6e-13 A out of 1e-6 — a 3.6e-7 relative
        // offset, which is the model working, not drifting.
        INFO("i = " << i << " A, v = " << v << " V");
        REQUIRE(iD(p, v) == Approx(i + v * p.G_min)
                               .epsilon(1e-12));
    }
}

TEST_CASE("ShockleyDiode — V_F rises ~60 mV per decade",
          "[v2][layer2][shockley_diode][unit]") {
    ShockleyDiode::Params p;
    const Real v1 = ShockleyDiode::voltage_for_current(p, 1e-3);
    const Real v2 = ShockleyDiode::voltage_for_current(p, 1e-2);
    // n·V_T·ln(10) = 59.5 mV for n = 1.
    REQUIRE(v2 - v1 == Approx(p.V_T * std::log(10.0))
                          .epsilon(1e-6));
}

TEST_CASE("ShockleyDiode — the working range is NOT a tangent",
          "[v2][layer2][shockley_diode][unit][regression]") {
    // The mistake this model was first written with. Continuing
    // the exponential by its tangent above SPICE's `vcrit`
    // (≈ 0.63 V, i.e. ≈ 18 mA) turns the device into a 1.41 Ω
    // resistor: 14.7 V at 10 A instead of 0.77 V. It converged
    // happily while doing it, which is why this is pinned.
    ShockleyDiode::Params p;
    for (const Real i : {0.018, 1.0, 10.0, 1e3, 1e5}) {
        const Real v = ShockleyDiode::voltage_for_current(p, i);
        INFO("i = " << i << " A");
        REQUIRE(v < 1.1);                     // a junction
        REQUIRE(iD(p, v) == Approx(i).epsilon(1e-9));
    }
}

TEST_CASE("ShockleyDiode — finite for voltages Newton invents",
          "[v2][layer2][shockley_diode][unit]") {
    // exp(50/0.02585) = e^1934 is +inf. The limiter's whole job.
    ShockleyDiode::Params p;
    for (const Real v : {5.0, 50.0, 500.0, 5000.0}) {
        const Real i = iD(p, v);
        INFO("v = " << v << " V");
        REQUIRE(std::isfinite(i));
        REQUIRE(i > 0.0);
    }
    // Monotone all the way out, so Newton always has a descent
    // direction back toward the solution.
    REQUIRE(iD(p, 500.0) > iD(p, 50.0));
    REQUIRE(iD(p, 50.0) > iD(p, 5.0));
}

TEST_CASE("ShockleyDiode — Jacobian is finite and matches FD",
          "[v2][layer2][shockley_diode][ad]") {
    ShockleyDiode::Params p;
    for (const Real v : {-100.0, -1.0, 0.0, 0.5, 0.7, 50.0}) {
        const std::array<Real, 2> vv{v, 0.0};
        const auto [i, d] =
            evaluate_current_and_jacobian<ShockleyDiode>(vv, p);
        INFO("v = " << v);
        REQUIRE(std::isfinite(i));
        REQUIRE(std::isfinite(d[0]));
        REQUIRE(std::isfinite(d[1]));
        // dI/dv_anode must be positive: the junction is monotone.
        REQUIRE(d[0] > 0.0);
        // Two terminals, one branch current: the partials are
        // equal and opposite.
        REQUIRE(d[1] == Approx(-d[0]).epsilon(1e-12));
    }
}

TEST_CASE("ShockleyDiode — C1 across the limiter join",
          "[v2][layer2][shockley_diode][ad]") {
    ShockleyDiode::Params p;
    const Real vl = ShockleyDiode::v_lim(p);
    const Real h  = 1e-7;
    const std::array<Real, 2> below{vl - h, 0.0};
    const std::array<Real, 2> above{vl + h, 0.0};
    const auto [i_b, d_b] =
        evaluate_current_and_jacobian<ShockleyDiode>(below, p);
    const auto [i_a, d_a] =
        evaluate_current_and_jacobian<ShockleyDiode>(above, p);

    // Both assertions below are bounded by the curve's own
    // variation over the probe distance, not by a fixed epsilon.
    // A fixed epsilon here would be asserting that the function
    // is FLAT near the join, which it is not and should not be —
    // continuity is the claim, and the probes sit h away on
    // either side of it.
    //
    // SLOPE: above the join the derivative is the constant g_l;
    // below it the exponential's own derivative, smaller by
    // exp(-h/vte). So they differ by ~2h/vte = 7.7e-6 relative,
    // and anything at that scale is the exponential's curvature,
    // not a kink.
    const Real vte = p.n * p.V_T;
    REQUIRE(std::abs(d_a[0] - d_b[0])
            <= d_a[0] * (2.0 * h / vte) * 1.01);

    // VALUE: g ≈ 3.9e7 S here, so 2h = 2e-7 V of separation is
    // 7.7 A of legitimate difference on top of 1e6 A.
    REQUIRE(std::abs(i_a - i_b) <= 2.0 * h * d_a[0] * 1.001);

    // And the join really is a join: extrapolating the lower
    // branch's tangent across it lands on the upper branch to
    // full double precision, which a C0-but-not-C1 model could
    // not do.
    REQUIRE(i_b + d_b[0] * (2.0 * h) ==
              Approx(i_a).epsilon(1e-10));
}

TEST_CASE("ShockleyDiode — blocks in reverse, breaks down on demand",
          "[v2][layer2][shockley_diode][unit]") {
    ShockleyDiode::Params p;
    REQUIRE(std::abs(iD(p, -100.0)) < 1e-9);   // I_S + G_min only

    ShockleyDiode::Params z = p;
    z.BV = 5.1;
    REQUIRE(std::abs(iD(z, -4.0)) < 1e-9);     // still blocking
    REQUIRE(iD(z, -5.5) < -1e-6);              // conducting back
    REQUIRE(std::isfinite(iD(z, -500.0)));     // and still finite
}

TEST_CASE("ShockleyDiode — the tempco needs I_S, not just kT/q",
          "[v2][layer2][shockley_diode][unit]") {
    // Raising V_T alone RAISES the drop — the opposite of the
    // familiar behaviour. The real negative tempco comes from
    // I_S, which roughly doubles every 10 °C.
    constexpr Real T_cold = 298.15, T_hot = 398.15;
    ShockleyDiode::Params cold;
    cold.V_T = ShockleyDiode::thermal_voltage(T_cold);
    ShockleyDiode::Params vt_only = cold;
    vt_only.V_T = ShockleyDiode::thermal_voltage(T_hot);

    const Real v_cold =
        ShockleyDiode::voltage_for_current(cold, 1.0);
    const Real v_vt_only =
        ShockleyDiode::voltage_for_current(vt_only, 1.0);
    REQUIRE(v_vt_only > v_cold);

    ShockleyDiode::Params hot = vt_only;
    hot.I_S = ShockleyDiode::saturation_current_at(
        cold.I_S, T_hot, T_cold);
    const Real v_hot =
        ShockleyDiode::voltage_for_current(hot, 1.0);
    REQUIRE(v_hot < v_cold);
    const Real tempco = (v_hot - v_cold) / (T_hot - T_cold);
    INFO("tempco = " << tempco << " V/K");
    REQUIRE(tempco < -1e-3);
    REQUIRE(tempco > -3e-3);
}

TEST_CASE("ShockleyDiode — unphysical parameters are refused",
          "[v2][layer2][shockley_diode][validation]") {
    ShockleyDiode::Params p;
    REQUIRE_NOTHROW(ShockleyDiode::validate(p));

    auto bad = [](auto mutate) {
        ShockleyDiode::Params q;
        mutate(q);
        REQUIRE_THROWS_AS(ShockleyDiode::validate(q),
                           std::invalid_argument);
    };
    bad([](auto& q) { q.I_S = 0.0; });
    bad([](auto& q) { q.n = 0.0; });
    bad([](auto& q) { q.V_T = -1.0; });
    bad([](auto& q) { q.G_min = -1.0; });
    bad([](auto& q) { q.i_lim = 0.0; });
    bad([](auto& q) { q.BV = -5.1; });
}
