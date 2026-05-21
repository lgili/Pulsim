// =============================================================================
// Layer 2 — IdealDiode device model (the AD killer test)
// =============================================================================
//
// The smooth-blend behavioral diode is the canary for the AD-only
// stamping pattern. If AD partials match central-difference partials
// here, AD works for every device in Pulsim's catalogue. If they
// don't, NOTHING above Layer 2 can work.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/models/device_model.hpp"
#include "pulsim/v2/models/ideal_diode.hpp"

#include <cmath>

using namespace pulsim::v2;
using namespace pulsim::v2::models;
using Catch::Approx;

namespace {

IdealDiode::Params standard_diode() {
    IdealDiode::Params p;
    p.V_F0  = 0.7;
    p.R_d   = 0.01;
    p.G_off = 1e-9;
    p.kappa = 20.0;
    return p;
}

// Central-difference partial: (i(v + h·e_k) - i(v - h·e_k)) / (2h).
Real fd_partial(const IdealDiode::Params& p,
                 ModelInputs<IdealDiode> v,
                 Size k, Real h) {
    auto v_plus  = v;
    auto v_minus = v;
    v_plus[k]  += h;
    v_minus[k] -= h;
    const Real i_plus  = evaluate_current<IdealDiode>(v_plus,  p);
    const Real i_minus = evaluate_current<IdealDiode>(v_minus, p);
    return (i_plus - i_minus) / (Real{2} * h);
}

}  // namespace

TEST_CASE("IdealDiode: reverse-biased current is essentially zero",
          "[v2][layer2][diode]") {
    auto p = standard_diode();
    ModelInputs<IdealDiode> v = {Real{0}, Real{5}};   // v_diode = -5
    const Real i = evaluate_current<IdealDiode>(v, p);
    // Off-state leakage only — should be small and negative.
    REQUIRE(std::abs(i) < Real{1e-7});
}

TEST_CASE("IdealDiode: forward-biased current scales linearly past V_F0",
          "[v2][layer2][diode]") {
    auto p = standard_diode();
    ModelInputs<IdealDiode> v = {Real{1.0}, Real{0}};   // v_diode = 1.0
    const Real i = evaluate_current<IdealDiode>(v, p);
    // Asymptotic on-state: (v_diode - V_F0) / R_d = 0.3/0.01 = 30 A.
    // The smooth-blend gets close at v_diode = V_F0 + 0.3, kappa=20:
    //   alpha ≈ 1 / (1 + exp(-20·0.3)) ≈ 0.9975
    // → i ≈ 0.9975 · 30 = 29.92, well within 1 % of 30.
    REQUIRE(i == Approx(Real{30}).epsilon(0.01));
}

TEST_CASE("IdealDiode: at V_F0 current is half-way (alpha ≈ 0.5)",
          "[v2][layer2][diode]") {
    auto p = standard_diode();
    ModelInputs<IdealDiode> v = {Real{0.7}, Real{0}};
    const Real i = evaluate_current<IdealDiode>(v, p);
    // At v_diode = V_F0: alpha = 0.5 → i_on = 0 (numerator is zero);
    //   i_off = 0.5 · 0.7 · G_off ≈ tiny.
    // So total i ≈ 0.5 · G_off · V_F0 ≈ 3.5e-10.
    REQUIRE(std::abs(i) < Real{1e-8});
}

TEST_CASE("IdealDiode: AD partials match central-difference (sub-threshold)",
          "[v2][layer2][diode]") {
    auto p = standard_diode();
    ModelInputs<IdealDiode> v = {Real{0.5}, Real{0}};
    const auto [i, J] = evaluate_current_and_jacobian<IdealDiode>(v, p);

    const Real fd0 = fd_partial(p, v, /*k=*/0, /*h=*/Real{1e-6});
    const Real fd1 = fd_partial(p, v, /*k=*/1, /*h=*/Real{1e-6});

    INFO("sub-threshold: i=" << i);
    INFO("AD ∂/∂v[0]=" << J[0] << "  FD=" << fd0);
    INFO("AD ∂/∂v[1]=" << J[1] << "  FD=" << fd1);
    REQUIRE(J[0] == Approx(fd0).margin(1e-6));
    REQUIRE(J[1] == Approx(fd1).margin(1e-6));
}

TEST_CASE("IdealDiode: AD partials match central-difference (at threshold)",
          "[v2][layer2][diode]") {
    auto p = standard_diode();
    ModelInputs<IdealDiode> v = {Real{0.7}, Real{0}};
    const auto [i, J] = evaluate_current_and_jacobian<IdealDiode>(v, p);

    const Real fd0 = fd_partial(p, v, 0, Real{1e-6});
    const Real fd1 = fd_partial(p, v, 1, Real{1e-6});

    INFO("at-threshold: i=" << i);
    INFO("AD ∂/∂v[0]=" << J[0] << "  FD=" << fd0);
    INFO("AD ∂/∂v[1]=" << J[1] << "  FD=" << fd1);
    // Threshold is where the sigmoid changes fastest — tolerance
    // loosens to 1e-4 because central difference at h=1e-6 truncates
    // higher-order terms of the sharp transition.
    REQUIRE(J[0] == Approx(fd0).margin(1e-4));
    REQUIRE(J[1] == Approx(fd1).margin(1e-4));
}

TEST_CASE("IdealDiode: AD partials match central-difference (forward-biased)",
          "[v2][layer2][diode]") {
    auto p = standard_diode();
    ModelInputs<IdealDiode> v = {Real{1.0}, Real{0}};
    const auto [i, J] = evaluate_current_and_jacobian<IdealDiode>(v, p);

    const Real fd0 = fd_partial(p, v, 0, Real{1e-6});
    const Real fd1 = fd_partial(p, v, 1, Real{1e-6});

    INFO("forward-biased: i=" << i);
    INFO("AD ∂/∂v[0]=" << J[0] << "  FD=" << fd0);
    INFO("AD ∂/∂v[1]=" << J[1] << "  FD=" << fd1);
    // In the on-asymptote slope is ~1/R_d = 100, so 1e-4 absolute
    // tolerance is a 1e-6 relative error.
    REQUIRE(J[0] == Approx(fd0).margin(1e-4));
    REQUIRE(J[1] == Approx(fd1).margin(1e-4));
}

TEST_CASE("IdealDiode: partials sum to zero (current depends on Δv only)",
          "[v2][layer2][diode]") {
    auto p = standard_diode();
    for (Real v_a : {Real{-1.0}, Real{0.0}, Real{0.5}, Real{0.7}, Real{1.0}, Real{5.0}}) {
        ModelInputs<IdealDiode> v = {v_a, Real{0}};
        const auto [i, J] = evaluate_current_and_jacobian<IdealDiode>(v, p);
        INFO("v[0] = " << v_a << ", J[0] = " << J[0] << ", J[1] = " << J[1]);
        REQUIRE(J[0] + J[1] == Approx(Real{0}).margin(1e-9));
    }
}
