// =============================================================================
// DSED — exact LTI stepping
// =============================================================================
//
// v2.0 Phase 3, item 2. The closed form must match the analytic
// solution at ANY h — the entire value proposition is that there is
// no stability region and no truncation error to manage.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/dsed/exact_lti.hpp"

#include <cstring>

#include <cmath>

using namespace pulsim;
using namespace pulsim::dsed;
using Catch::Approx;

TEST_CASE("Exact: RC decay at a tenth, ten, and a thousand taus",
          "[v2][dsed][exact_lti]") {
    // dx/dt = -(x - V)/tau, analytic x(h) = V + (x0 - V)·e^{-h/tau}.
    const Real tau = 1e-6, V = 5.0, x0 = 1.0;
    DenseMatrix A(1, 1);
    A(0, 0) = -1.0 / tau;
    Vector b(1);
    b << V / tau;
    Vector x(1);
    x << x0;

    const auto e = make_exact_lti(A);
    REQUIRE(e.valid);
    for (const Real h : {0.1 * tau, 10.0 * tau, 1000.0 * tau}) {
        const Vector out = exact_advance(e, x, b, h);
        const Real ref = V + (x0 - V) * std::exp(-h / tau);
        INFO("h/tau = " << h / tau);
        REQUIRE(out[0] == Approx(ref).epsilon(1e-12).margin(1e-12));
    }
}

TEST_CASE("Exact: a pure integrator mode integrates exactly",
          "[v2][dsed][exact_lti]") {
    // lambda = 0 — the phi1 series branch. x(h) = x0 + b·h, exactly,
    // where the naive (e^s - 1)/s form would divide zero by zero.
    DenseMatrix A = DenseMatrix::Zero(1, 1);
    Vector b(1);
    b << 3.0;
    Vector x(1);
    x << 7.0;

    const auto e = make_exact_lti(A);
    REQUIRE(e.valid);
    const Vector out = exact_advance(e, x, b, 2.5);
    REQUIRE(out[0] == Approx(7.0 + 3.0 * 2.5).margin(1e-12));
}

TEST_CASE("Exact: a 5e9-rad/s mode crossed in ONE 10 µs step",
          "[v2][dsed][exact_lti]") {
    // THE case that kills explicit integrators: lambda = -5e9 (the
    // DCM idle mode's L·g_off). DOPRI5's stability region pins
    // h below ~6.6e-10 s forever; the exact step takes h = 1e-5 —
    // fifteen thousand stability limits at once — and lands exactly
    // on the equilibrium.
    DenseMatrix A(1, 1);
    A(0, 0) = -5e9;
    Vector b(1);
    b << 5e9 * 0.25;      // equilibrium at -b/lambda = 0.25
    Vector x(1);
    x << 100.0;

    const auto e = make_exact_lti(A);
    REQUIRE(e.valid);
    const Vector out = exact_advance(e, x, b, 1e-5);
    REQUIRE(out[0] == Approx(0.25).margin(1e-12));
}

TEST_CASE("Exact: an LC ring keeps amplitude and phase",
          "[v2][dsed][exact_lti]") {
    // Undamped rotation at omega = 1e7: after an arbitrary h the
    // state must be the initial vector rotated by omega·h, energy
    // preserved to machine precision — where an explicit method
    // spirals out and an implicit one spirals in.
    const Real w = 1e7;
    DenseMatrix A(2, 2);
    A << 0.0,   w,
         -w,  0.0;
    const Vector b = Vector::Zero(2);
    Vector x(2);
    x << 1.0, 0.0;

    const auto e = make_exact_lti(A);
    REQUIRE(e.valid);
    const Real h = 3.7e-6;                 // ~5.9 full turns
    const Vector out = exact_advance(e, x, b, h);
    REQUIRE(out[0] == Approx(std::cos(w * h)).margin(1e-9));
    REQUIRE(out[1] == Approx(-std::sin(w * h)).margin(1e-9));
    REQUIRE(out.squaredNorm() == Approx(1.0).margin(1e-12));
}

TEST_CASE("Exact: h = 0 is the identity",
          "[v2][dsed][exact_lti]") {
    DenseMatrix A(2, 2);
    A << -100.0, 20.0, 5.0, -300.0;
    Vector b(2);
    b << 1.0, -2.0;
    Vector x(2);
    x << 4.0, 9.0;
    const auto e = make_exact_lti(A);
    REQUIRE(e.valid);
    const Vector out = exact_advance(e, x, b, 0.0);
    REQUIRE(out[0] == Approx(4.0).margin(1e-13));
    REQUIRE(out[1] == Approx(9.0).margin(1e-13));
}

// -----------------------------------------------------------------
// v2.0 Phase 3 item 3 — the SINE-augmented stepper, end to end
// through the builder adapter. A series RC driven by a sine has a
// closed-form response; one exact step across many time constants
// must land on it, DC offset and phase included.
// -----------------------------------------------------------------

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/dsed/builder_adapter.hpp"
#include "pulsim/pwl/cache.hpp"

TEST_CASE("Exact: sine-driven RC steps to the analytic response",
          "[v2][dsed][exact_lti][augmented]") {
    const Real R = 1e3, C = 1e-6;                 // tau = 1 ms
    const Real A_amp = 5.0, f = 60.0, phase = 0.7, v_dc = 2.0;
    builder::CircuitBuilder b;
    b.add_sine_voltage_source("Vac", "in", "gnd",
                                v_dc, A_amp, f, phase);
    b.add_resistor("R1", "in", "n1", R);
    b.add_capacitor("C1", "n1", "gnd", C);

    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    dsed::NativeCircuitBuilderAdapter<> ad{b.graph(), b.pool(),
                                             cache};
    topology::SwitchStateMask m(0);
    ad.set_mask(m);
    REQUIRE(ad.has_exact_step());       // sine-only drive qualifies

    // Analytic v_C for v_C(0) = 0 under
    // V(t) = v_dc + A·sin(wt + phase):
    //   steady state: v_dc + Vp·sin(wt + phase − theta),
    //   Vp = A/sqrt(1+(wRC)^2), theta = atan(wRC);
    //   transient sized to cancel it at t = 0.
    const Real w = 2.0 * std::numbers::pi_v<Real> * f;
    const Real tau = R * C;
    const Real theta = std::atan(w * tau);
    const Real Vp = A_amp / std::sqrt(1.0 + w * tau * w * tau);
    auto v_ref = [&](Real t) {
        const Real ss = v_dc + Vp * std::sin(w * t + phase - theta);
        const Real ss0 = v_dc + Vp * std::sin(phase - theta);
        return ss - ss0 * std::exp(-t / tau);
    };

    Vector x0(1);
    x0 << 0.0;
    // ONE step of 5 ms (five taus, a third of a mains cycle), then
    // a second from there — exactness must compose across steps.
    const Vector x1 = ad.exact_advance_state(0.0, x0, 5e-3);
    REQUIRE(x1[0] == Approx(v_ref(5e-3)).margin(1e-9));
    const Vector x2 = ad.exact_advance_state(5e-3, x1, 7e-3);
    REQUIRE(x2[0] == Approx(v_ref(12e-3)).margin(1e-9));
}

TEST_CASE("Exact: a PWM source steps exactly inside a piece and names its corners",
          "[v2][dsed][exact_lti][augmented]") {
    // add-exact-propagator-strategy D14: between corners the source is
    // a level, stepped in closed form; each corner is a time event the
    // scheduler lands on (it never steps across one).
    builder::CircuitBuilder b;
    b.add_pwm_voltage_source("Vg", "in", "gnd", 5.0, 0.0, 1e5, 0.5);
    b.add_resistor("R1", "in", "n1", 1e3);
    b.add_capacitor("C1", "n1", "gnd", 1e-6);

    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    dsed::NativeCircuitBuilderAdapter<> ad{b.graph(), b.pool(),
                                             cache};
    topology::SwitchStateMask m(0);
    ad.set_mask(m);
    REQUIRE(ad.has_exact_step());
    REQUIRE(ad.next_source_corner_after(0.0) == Approx(5e-6).epsilon(1e-14));
    REQUIRE(ad.next_source_corner_after(5e-6) == Approx(1e-5).epsilon(1e-14));
    Vector x0(1);
    x0 << 0.0;
    const Vector x1 = ad.exact_advance_state(0.0, x0, 5e-6);   // the 5 V piece
    REQUIRE(x1[0] == Approx(-5.0 * std::expm1(-5e-6 / 1e-3)).epsilon(1e-13));
}

// -----------------------------------------------------------------
// add-exact-propagator-strategy — the kernel states its accuracy.
// -----------------------------------------------------------------

TEST_CASE("Exact: phi1 is accurate to 1e-14 at every |s|",
          "[v2][dsed][exact_lti][phi]") {
    // Reference: the Taylor series below |s| = 1 (40 terms, no
    // cancellation), the direct form above (no cancellation there).
    const auto ref = [](std::complex<Real> s) {
        if (std::abs(s) < 1.0) {
            std::complex<Real> term = 1.0, sum = 1.0;
            for (int k = 2; k < 40; ++k) {
                term *= s / Real(k);
                sum += term;
            }
            return sum;
        }
        return (std::exp(s) - 1.0) / s;
    };
    const std::complex<Real> dirs[] = {
        {1, 0}, {-1, 0}, {0, 1}, {0.70710678118654752, 0.70710678118654752},
        {-0.6, 0.8}};
    for (const Real mag : {1e-12, 1e-9, 0.99e-6, 1e-6, 1.01e-6, 3e-6,
                           1e-4, 1e-2, 0.3, 0.999, 1.0, 3.0, 10.0, 50.0}) {
        for (const auto dir : dirs) {
            const std::complex<Real> s = mag * dir;
            const Real err = std::abs(phi1(s) - ref(s)) / std::abs(ref(s));
            INFO("s = " << s);
            REQUIRE(err <= 1e-14);
        }
    }
    REQUIRE(phi1({0, 0}) == std::complex<Real>{1, 0});
}

TEST_CASE("Exact: phi2 is accurate to 1e-14 at every |s|",
          "[v2][dsed][exact_lti][phi]") {
    // Reference: the Taylor series below |s| = 2 (60 terms), the
    // direct form above (e^s - 1 - s does not cancel there).
    const auto ref = [](std::complex<Real> s) {
        if (std::abs(s) < 2.0) {
            std::complex<Real> term = 0.5, sum = 0.5;
            for (int k = 3; k < 60; ++k) {
                term *= s / Real(k);
                sum += term;
            }
            return sum;
        }
        return (std::exp(s) - 1.0 - s) / (s * s);
    };
    const std::complex<Real> dirs[] = {
        {1, 0}, {-1, 0}, {0, 1}, {0.70710678118654752, 0.70710678118654752},
        {-0.6, 0.8}};
    for (const Real mag : {1e-12, 1e-6, 1e-3, 0.3, 0.4999, 0.5, 0.5001, 1.0,
                           1.99, 3.0, 10.0, 50.0}) {
        for (const auto dir : dirs) {
            const std::complex<Real> s = mag * dir;
            const Real err = std::abs(phi2(s) - ref(s)) / std::abs(ref(s));
            INFO("s = " << s);
            REQUIRE(err <= 1e-14);
        }
    }
    REQUIRE(phi2({0, 0}) == std::complex<Real>{0.5, 0});
}

TEST_CASE("Exact: a level + ramp source is exact on tiers 1 and 2",
          "[v2][dsed][exact_lti][phi]") {
    // x' = A x + c0 + c1 t from 0 over h = 3. Defective A = [[-1, 1],
    // [0, -1]] (no eigenbasis: tier 2) with c0 = c1 = (0, 1):
    // x2 = t, x1 = t - 1 + e^{-t}. Diagonal A = diag(-1, -2) (tier 1)
    // with c0 = (1, 0), c1 = (0, 2): x1 = 1 - e^{-t},
    // x2 = t - (1 - e^{-2t})/2.
    const Real h = 3.0;
    Vector x0 = Vector::Zero(2), out(2);

    DenseMatrix Ad(2, 2);
    Ad << -1.0, 1.0, 0.0, -1.0;
    const auto ed = make_exact_lti(Ad);
    REQUIRE(ed.valid);
    REQUIRE(ed.tier == 2);
    Vector c0(2), c1(2);
    c0 << 0.0, 1.0;
    c1 << 0.0, 1.0;
    exact_advance_affine_into(ed, x0, h, c0, c1, out);
    REQUIRE(out[0] == Approx(h - 1.0 + std::exp(-h)).epsilon(1e-13));
    REQUIRE(out[1] == Approx(h).epsilon(1e-13));

    DenseMatrix Ag = DenseMatrix::Zero(2, 2);
    Ag(0, 0) = -1.0;
    Ag(1, 1) = -2.0;
    const auto eg = make_exact_lti(Ag);
    REQUIRE(eg.valid);
    REQUIRE(eg.tier == 1);
    c0 << 1.0, 0.0;
    c1 << 0.0, 2.0;
    exact_advance_affine_into(eg, x0, h, c0, c1, out);
    REQUIRE(out[0] == Approx(1.0 - std::exp(-h)).epsilon(1e-14));
    REQUIRE(out[1] == Approx(h - 0.5 * (1.0 - std::exp(-2.0 * h))).epsilon(1e-14));
}

namespace {
// Series RLC, x = [i_L, v_C], 10 V step, x0 = [0.3 A, 2 V], advanced
// by one natural period h = 2π·√(LC); references from 50-digit mpmath
// on the same doubles (probes/dsed_baseline.py's circuit).
struct RlcCase { Real R, i_ref, v_ref; };
Real energy_rel_err(const Vector& x, Real i_ref, Real v_ref,
                    Real L, Real C) {
    const Real di = std::sqrt(L) * (x[0] - i_ref);
    const Real dv = std::sqrt(C) * (x[1] - v_ref);
    return std::hypot(di, dv)
           / std::hypot(std::sqrt(L) * i_ref, std::sqrt(C) * v_ref);
}
}  // namespace

TEST_CASE("Exact: critically damped RLC within 1e-10 through the tiers",
          "[v2][dsed][exact_lti][tier]") {
    const Real L = 100e-6, C = 100e-9, h = 1.98691765315922e-05;
    const RlcCase cases[] = {
        {63.30879875657095, 1.984050586612746999e-5, 10.002034882998897815},
        {63.24555326661315, 8.550154551382151842e-6, 10.002506195702407315},
        {63.24555320337391, 8.5501432690247946968e-6, 10.002506196172187995},
        {63.24555320336759, 8.5501432678978433567e-6, 10.00250619617223492},
    };
    Vector d(2);
    d << std::sqrt(L), std::sqrt(C);
    Vector x0(2);
    x0 << 0.3, 2.0;
    for (const auto& c : cases) {
        DenseMatrix A(2, 2);
        A << -c.R / L, -1.0 / L, 1.0 / C, 0.0;
        Vector b(2);
        b << 10.0 / L, 0.0;
        const auto e = make_exact_lti(A, b, d, h);
        INFO("R = " << c.R << " tier " << e.tier << " kappa " << e.kappa);
        REQUIRE(e.valid);
        // The stated bound holds for whichever tier was taken.
        if (e.tier == 1) {
            REQUIRE(kExactRoundoff * e.kappa <= kExactTierBound);
        } else {
            REQUIRE(e.pade_defect <= kExactTierBound);
        }
        const Vector x = exact_advance(e, x0, h);
        REQUIRE(energy_rel_err(x, c.i_ref, c.v_ref, L, C) <= 1e-10);
    }
    // R = R_crit exactly: the basis is "invertible" but kappa ~ 1e8+;
    // it must not be accepted on tier 1 (the modal form was 1.1e-6
    // off component-wise on the untouched kernel).
    DenseMatrix A(2, 2);
    A << -cases[3].R / L, -1.0 / L, 1.0 / C, 0.0;
    Vector b(2);
    b << 10.0 / L, 0.0;
    const auto e = make_exact_lti(A, b, d, h);
    REQUIRE(e.tier == 2);
    REQUIRE_FALSE(e.reason.empty());
    // Tier 2 caches the exponential per h.
    (void)exact_advance(e, x0, h);
    (void)exact_advance(e, x0, h);
    REQUIRE(e.n_pade == 1);
    REQUIRE(e.n_pade_hits == 1);
}

TEST_CASE("Exact: stiff buck idle mode stays on tier 1 to 1e-14",
          "[v2][dsed][exact_lti][tier]") {
    // Switch and diode both open (g_off = 1e-9 each): λ ≈ −5e12,
    // ‖A·h‖₁ ≈ 1e7 at h = 2 µs. Padé would lose 8e-11 here; the
    // eigenbasis in energy coordinates is near-orthogonal.
    const Real L = 100e-6, C = 47e-6, R = 50.0;
    const Real k = 1.0 / (1e-9 + 1e-9);
    DenseMatrix A(2, 2);
    A << -k / L, -1.0 / L, 1.0 / C, -1.0 / (R * C);
    Vector d(2);
    d << std::sqrt(L), std::sqrt(C);
    const auto e = make_exact_lti(A, Vector::Zero(2), d, 2e-6);
    REQUIRE(e.tier == 1);
    Vector x(2);
    x << 0.0, 1.0;
    const Vector y = exact_advance(e, x, 2e-6);
    REQUIRE(std::abs(y[1] - 0.99914929813728287694) <= 1e-14);
}

TEST_CASE("Exact: stiff and near-defective takes the Pade tier",
          "[v2][dsed][exact_lti][tier]") {
    // A Jordan block scaled to 1e12 (off-diagonal 1e12 too: with 1.0
    // it is normal to rounding, kappa2 = 4.5e3, a legitimate tier 1).
    // kappa2 = 3e15: no tier 1. ||A*h||_1 = 2e7, but e^{Ah} = 0 to
    // e^{-1e7}, and the squared form states that (pade_defect ~ u).
    DenseMatrix A(2, 2);
    A << -1e12, 1e12, 0.0, -1e12;
    const auto e = make_exact_lti(A, Vector::Zero(2), Vector{}, 1e-5);
    REQUIRE(e.valid);
    REQUIRE(e.tier == 2);
    REQUIRE(e.pade_defect <= kExactTierBound);
    Vector x(2);
    x << 1.0, 1.0;
    REQUIRE(exact_advance(e, x, 1e-5).cwiseAbs().maxCoeff() <= 1e-15);
}

TEST_CASE("Exact: the squared form keeps the slow part of a stiff matrix",
          "[v2][dsed][exact_lti][pade]") {
    // A = [[-g, c], [0, -a]] (g = c = 1e12): e^{Ah} has the closed form
    // [[e^{-gh}, c(e^{-ah} - e^{-gh})/(g - a)], [0, e^{-ah}]]. Squaring
    // e^X (Eigen's .exp()) lost 2.2e-10 (a = 1, h = 1e-5) and 2.4e-8
    // (h = 1e-3) of it; squaring e^X - I keeps it to a few u.
    const Real g = 1e12, c = 1e12;
    for (const Real a : {1.0, 1e2, 1e4}) {
        for (const Real h : {1e-5, 1e-3}) {
            DenseMatrix A(2, 2);
            A << -g, c, 0.0, -a;
            DenseMatrix ref(2, 2);
            ref << std::exp(-g * h), c * (std::exp(-a * h) - std::exp(-g * h)) / (g - a),
                0.0, std::exp(-a * h);
            const DenseMatrix Ah = A * h;
            INFO("a = " << a << ", h = " << h);
            REQUIRE((expm_aug(Ah) - ref).norm() <= 1e-14);
            REQUIRE(pade_defect(A, Vector::Zero(2), h) <= 1e-14);
        }
    }
}

TEST_CASE("Exact: stiff, near-defective and ill-conditioned is tier 3, with the reason",
          "[v2][dsed][exact_lti][tier]") {
    // Rows of ±1e12 that cancel (a near-conserved mode the stored matrix
    // fixes only to u·1e12) beside a 1e12 Jordan block: no tier 1
    // (kappa2 ~ 1e16), and e^{Ah} at h = 1e-5 is uncertain at ~1e-9
    // whatever the method (the 3-state block alone: squared form 1.5e-9
    // off a 60-digit exponential of the stored matrix, estimate 3.6e-9; macOS).
    const Real g = 1e12;
    DenseMatrix A = DenseMatrix::Zero(5, 5);
    A.topLeftCorner(3, 3) << -g, g, 0.0, g, -g - 1.0, 1e3, 0.0, -1e3, -1.0;
    A.bottomRightCorner(2, 2) << -g, g, 0.0, -g;
    const auto e = make_exact_lti(A, Vector::Zero(5), Vector{}, 1e-5);
    REQUIRE_FALSE(e.valid);
    REQUIRE(e.tier == 3);
    REQUIRE(e.pade_defect > kExactTierBound);
    REQUIRE(e.reason.find("Pade defect") != std::string::npos);
    REQUIRE(e.reason.find("no exact step") != std::string::npos);
}

TEST_CASE("Exact: the Pade cache keeps the main step through partial steps",
          "[v2][dsed][exact_lti][pade]") {
    DenseMatrix A(2, 2);
    A << -1e12, 1e12, 0.0, -1e12;
    const auto e = make_exact_lti(A, Vector::Zero(2), Vector{}, 1e-5);
    REQUIRE(e.tier == 2);
    Vector x(2), out(2);
    x << 1.0, 0.0;
    exact_advance_into(e, x, 1e-5, out);
    for (int k = 1; k <= 40; ++k) {   // more partial lengths than the cache holds
        exact_advance_into(e, x, 1e-5 / (k + 1), out);
        exact_advance_into(e, x, 1e-5, out);
    }
    REQUIRE(e.n_pade == 41);
    REQUIRE(e.n_pade_hits == 40);
}

// -----------------------------------------------------------------
// add-complementarity-mode-selection task 5.1 — passivity classes the
// mode selection must know (D3), read at build time.
// -----------------------------------------------------------------

TEST_CASE("Passivity: coupled inductors, ideal transformer, VCVS",
          "[v2][dsed][modes][passivity]") {
    const auto cls = [](builder::CircuitBuilder& b) {
        pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
        dsed::NativeCircuitBuilderAdapter<> ad{b.graph(), b.pool(), cache};
        return ad.passivity();
    };
    const auto rect = [](builder::CircuitBuilder& b) {
        b.add_diode("D1", "s", "o", 1e3, 1e-9);
        b.add_capacitor("C", "o", "gnd", 1e-6);
        b.add_resistor("R", "o", "gnd", 100.0);
    };
    SECTION("a coupled pair is passive and reciprocal") {
        builder::CircuitBuilder b;
        b.add_sine_voltage_source("V", "a", "gnd", 0.0, 10.0, 50.0);
        b.add_inductor("L1", "a", "gnd", 1e-3);
        b.add_inductor("L2", "s", "gnd", 1e-3);
        b.add_inductor_coupling("L1", "L2", 0.99);
        rect(b);
        const auto c = cls(b);
        REQUIRE(c.passive());
        REQUIRE(std::string(c.name()) == "passive_reciprocal");
    }
    SECTION("pairwise couplings that make an indefinite inductance matrix") {
        // k12 = k13 = 1, k23 = 0: eigenvalues of [[1,1,1],[1,1,0],[1,0,1]]·mH
        // are 1 ± sqrt(2) and 1 (one negative): the stored energy can be
        // negative, so the ports are not certified passive.
        builder::CircuitBuilder b;
        b.add_sine_voltage_source("V", "a", "gnd", 0.0, 10.0, 50.0);
        b.add_inductor("L1", "a", "gnd", 1e-3);
        b.add_inductor("L2", "s", "gnd", 1e-3);
        b.add_inductor("L3", "t", "gnd", 1e-3);
        b.add_resistor("R3", "t", "gnd", 10.0);
        b.add_inductor_coupling("L1", "L2", 1.0);
        b.add_inductor_coupling("L1", "L3", 1.0);
        b.add_inductor_coupling("L2", "L3", 0.0);
        rect(b);
        const auto c = cls(b);
        REQUIRE_FALSE(c.passive());
        REQUIRE(c.device == "L1+L2+L3");
        REQUIRE(c.kind == "InductorCoupling");
        REQUIRE(std::string(c.name()) == "non_passive");
    }
    SECTION("two couplings on one pair add, as the assembly stamps them") {
        // k = 0.8 twice on L1-L2 simulates M = 1.6·sqrt(L1·L2): eigenvalue
        // 1 - 1.6 < 0 mH. Each coupling alone is PSD; the check must sum.
        builder::CircuitBuilder b;
        b.add_sine_voltage_source("V", "a", "gnd", 0.0, 10.0, 50.0);
        b.add_inductor("L1", "a", "gnd", 1e-3);
        b.add_inductor("L2", "s", "gnd", 1e-3);
        b.add_inductor_coupling("L1", "L2", 0.8);
        b.add_inductor_coupling("L1", "L2", 0.8);
        rect(b);
        const auto c = cls(b);
        REQUIRE_FALSE(c.passive());
        REQUIRE(c.device == "L1+L2");
        REQUIRE(c.kind == "InductorCoupling");
    }
    SECTION("an ideal transformer is passive, not reciprocal") {
        builder::CircuitBuilder b;
        b.add_sine_voltage_source("V", "a", "gnd", 0.0, 10.0, 50.0);
        b.add_ideal_transformer("T1", "a", "gnd", "s", "gnd", 2.0);
        rect(b);
        const auto c = cls(b);
        REQUIRE(c.passive());
        REQUIRE(c.nonreciprocal_device == "T1");
        REQUIRE(c.nonreciprocal_kind == "IdealTransformer");
        REQUIRE(std::string(c.name()) == "passive_nonreciprocal");
    }
    SECTION("a VCVS is non-passive and wins over a transformer") {
        builder::CircuitBuilder b;
        b.add_sine_voltage_source("V", "a", "gnd", 0.0, 10.0, 50.0);
        b.add_ideal_transformer("T1", "a", "gnd", "m", "gnd", 2.0);
        b.add_resistor("Rm", "m", "gnd", 10.0);
        b.add_vcvs("E1", "m", "gnd", "s", "gnd", 3.0);
        rect(b);
        const auto c = cls(b);
        REQUIRE_FALSE(c.passive());
        REQUIRE(c.device == "E1");
    }
}

TEST_CASE("Exact: a stiff winding row does not leak into the slow modes (balancing)",
          "[v2][dsed][exact_lti]") {
    // A flyback mode (examples/flyback.yaml, Q1 ON, D1 / D_cl OFF): states
    // (v_Cout, v_Ccl, i_p, i_s). The secondary current is limited by D1's
    // g_off, a −4.1e14 /s row. Without balancing, the eigensolver's backward
    // error (ε·‖A‖ in every entry) drove v_Cout to +1.3e-3 V at 1 ms; the
    // exact solution of this matrix (mpmath, 50 digits) is −9.234e-8 V.
    DenseMatrix A(4, 4);
    A << -2000.0, 0.0, 0.0, 10000.0,
         0.0, -500.00100000000003, 9.9999999998e-06, 0.0,
         194871.7948717948, -1.0256410256205122e-06, -1025.6410256205122, 194871794871794.78,
         -410256.4102564101, 1.948717948678973e-06, 1948.717948678973, -410256410256410.06;
    Vector b(4);
    b << 0.0, -0.048000000000275767, 4923076.92302769, -9353846.15375261;
    const Vector ref = (Vector(4) << -9.2340255851271658e-8, -3.5796794290654677e-5,
                        456.78039343209462, -2.0630293038677716e-8).finished();
    // the adapter's energy coordinates (√C, √L); this test pins that path
    const Vector scale = (Vector(4) << 1e-2, 1e-3, 1e-2, 5e-3).finished();
    {
        const auto e = make_exact_lti(A, b, scale, 1e-3);
        REQUIRE(e.valid);
        Vector x = Vector::Zero(4), out(4);
        for (int k = 0; k < 1000; ++k) {   // the output grid of 1 µs
            exact_advance_into(e, x, 1e-6, out);
            x = out;
        }
        // within 1e-10 of the circuit's scale (48 V, 457 A): the tier-1 claim
        const Real margin[4] = {48e-10, 48e-10, 457e-10, 457e-10};
        for (int i = 0; i < 4; ++i)
            REQUIRE(x[i] == Approx(ref[i]).epsilon(1e-12).margin(margin[i]));
    }
}

TEST_CASE("Exact: a near-conserved bridge mode is refined to the matrix's exponential",
          "[v2][dsed][exact_lti]") {
    // examples/phase_shift_full_bridge.yaml at phi = pi/2, the doc run's mode
    // with LS_B and D_LS_B ON (adapter A and energy scale). Its slowest mode is
    // conserved (mpmath: lambda = 1.2e-13); Eigen's pairs put it at -0.0985 /s
    // and the balanced pairs at +0.0019 /s, a propagator 6.3e-4 / 1.3e-5 off per
    // ms. The refined pairs are within 1e-10 of mpmath's expm of this matrix.
    DenseMatrix A(5, 5);
    A << -1000.0, 0.0, 0.0, 0.0, 10000.0,
         0.0, -139198218262805.42, -278.39643652533255, 1102449888641419.0, -0.19713340198494406,
         0.0, -139198218262805.45, -278.3964365253325, 1102449888641418.8, -0.197133401984944,
         0.0, 275612472160354.72, 551.2249443201582, -2282850779510009.0, 0.4082055293627629,
         -10000.0, 0.0, 0.0, 0.0, -9999999999999.994;
    const Vector scale = (Vector(5) << 0.01, 0.001, 0.006324555320336759,
                          0.0031622776601683794, 0.01).finished();
    const Vector x0 = (Vector(5) << 44.721359549995796, 447.2135954999579, 70.71067811865476,
                       141.4213562373095, 44.721359549995796).finished();   // unit energy
    const Vector ref = (Vector(5) << 16.452068611610981, 7.5300583476036379e-10,
                        -376.50291738055537, -2.2105071865475379e-24,
                        -1.6452068613256197e-8).finished();
    const auto e = make_exact_lti(A, Vector::Zero(5), scale, 1e-6, 1e-3);
    REQUIRE(e.tier == 1);
    REQUIRE_FALSE(e.balanced);
    REQUIRE(e.refined >= 1);
    REQUIRE(e.defect <= kExactTierBound);
    Real lmin = std::numeric_limits<Real>::infinity();
    for (Eigen::Index i = 0; i < e.lambda.size(); ++i) lmin = std::min(lmin, std::abs(e.lambda[i]));
    REQUIRE(lmin < 1e-6);
    Vector x = x0, out(5);
    for (int k = 0; k < 1000; ++k) {
        exact_advance_into(e, x, 1e-6, out);
        x = out;
    }
    for (int i = 0; i < 5; ++i) REQUIRE(std::abs(x[i] - ref[i]) * scale[i] <= 1e-10);
}

TEST_CASE("Exact: pair_quality states the propagator error of a wrong eigenvalue",
          "[v2][dsed][exact_lti]") {
    // A = diag(0, -1). Pairs with the conserved eigenvalue at delta instead of 0
    // propagate e^{delta H} for 1: the first-order defect is delta·H.
    DenseMatrix A = DenseMatrix::Zero(2, 2);
    A(1, 1) = -1.0;
    const Eigen::MatrixXcd V = Eigen::MatrixXcd::Identity(2, 2);
    Eigen::VectorXcd lam(2);
    lam << std::complex<Real>{1e-3, 0}, std::complex<Real>{-1, 0};
    const auto bad = pair_quality(A, V, lam, 2.0);
    REQUIRE(bad.defect == Approx(2e-3).epsilon(0.01));
    REQUIRE(bad.score() > 1.0);
    lam[0] = 0.0;
    const auto good = pair_quality(A, V, lam, 2.0);
    REQUIRE(good.defect == 0.0);
    REQUIRE(good.score() <= 1.0);
    // a fast pair's error is damped at H = 2 but not over the segments the
    // propagator is used on: 1e-3 on lambda = -1e6 peaks at t = 1 us at
    // 1e-3·1e-6/e = 3.68e-10 (review of slice 5: judged at H alone it read 0)
    lam[0] = 0.0;
    DenseMatrix F = DenseMatrix::Zero(2, 2);
    F(1, 1) = -1e6;
    lam[1] = std::complex<Real>{-1e6 + 1e-3, 0};
    const auto fast = pair_quality(F, V, lam, 2.0);
    REQUIRE(fast.defect == Approx(1e-9 / std::exp(1.0)).epsilon(0.05));
    REQUIRE(fast.score() > 1.0);
}

TEST_CASE("Exact: balancing leaves a balanced matrix untouched",
          "[v2][dsed][exact_lti]") {
    DenseMatrix M(2, 2);
    M << -1.0, 2.0, -2.0, -1.0;
    DenseMatrix B = M;
    const Vector s = balance_in_place(B);
    REQUIRE((s.array() == 1.0).all());
    REQUIRE((B.array() == M.array()).all());
}

TEST_CASE("Exact: a degenerate-cluster bridge mode to a 60-digit exponential",
          "[v2][dsed][exact_lti][pade]") {
    // Energy-coordinate state matrix of a mode of three identical paralleled
    // three-phase bridges (python/tests/test_mode_lcp.py::_paralleled_bank(3)),
    // eigenvalues clustered near -5 /s beside -2.5e10 /s; reference e^{A h}
    // at h = 1e-5 from mpmath at 60 digits (exact engine 7). Measured 9e-12
    // relative (macOS arm64, Eigen 5), estimate 2.7e-11.
    DenseMatrix A(10, 10);
    A << -100.00000349987205, -790.56941504209487, -790.56866109634836, -0.00037697287323097938, -790.56941504209487, -1581.1395840299363, -0.00056545930984646905, -790.56979201496813, -1581.139207057063, 0.00018848643661548969,
        790.5699543049036, -2505000003418.231, -5000003418.2312603, -5000003418.2312603, -5000003418.2312603, -5000003418.2312603, -5000003418.2312603, -5000003418.2312603, -5000003418.2312603, -5000003418.2312603,
        790.5699543049036, -5000003418.2312603, -2505000003418.2305, -5000003418.2312613, -5000003418.2312613, -5000003418.2312603, -5000003418.2312603, -5000003418.2312613, -5000003418.2312613, -5000003418.2312603,
        0.00053926438974162701, -5000003418.2312593, -5000003418.2312603, -5000003423.2312622, -5000003418.2312622, -5000003418.2312613, -5000003418.2312622, -5000003418.2312622, -5000003418.2312622, -5000003418.2312593,
        790.5699543049036, -5000003418.2312603, -5000003418.2312622, -5000003418.2312632, -2505000003418.2324, -5000003418.2312603, -5000003418.2312632, -5000003418.2312632, -5000003418.2312632, -5000003418.2312622,
        1581.139369345417, -5000003418.2312603, -5000003418.2312613, -5000003418.2312622, -5000003418.2312622, -5000003423.2312613, -5000003418.2312622, -5000003418.2312622, -5000003418.2312622, -5000003418.2312613,
        0.00053926438995272269, -5000003418.2312603, -5000003418.2312603, -5000003418.2312622, -5000003418.2312622, -5000003418.2312603, -5000003423.2312622, -5000003418.2312622, -5000003418.2312622, -5000003418.2312603,
        790.5699543049036, -5000003418.2312603, -5000003418.2312622, -5000003418.2312622, -5000003418.2312622, -5000003418.2312622, -5000003418.2312622, -2505000003418.231, -5000003418.2312622, -5000003418.2312622,
        1581.139369345417, -5000003418.2312603, -5000003418.2312622, -5000003418.2312622, -5000003418.2312622, -5000003418.2312622, -5000003418.2312622, -5000003418.2312622, -5000003423.2312632, -5000003418.2312613,
        0.00053926438997847217, -5000003418.2312603, -5000003418.2312622, -5000003418.2312622, -5000003418.2312622, -5000003418.2312622, -5000003418.2312622, -5000003418.2312622, -5000003418.2312622, -5000003423.2312613;
    DenseMatrix ref(10, 10);
    ref << 0.99885060517798463, -6.3172707218830934e-11, -6.3172405983547952e-11, 0.0063208947595502201, -6.3172707221242036e-11, -0.0094814089534409826, 0.0063208928757692703, -6.3172857837075116e-11, -0.0094814051858790813, 0.0063209004109835059,
        6.317285783574298e-11, -3.9953927359565451e-21, -3.9953736841648033e-21, 3.9977501752976793e-13, -3.9953927361090389e-21, -5.9966663755887602e-13, 3.9977489838710304e-13, -3.9954022618905332e-21, -5.9966601782289263e-13, 3.9977476464243671e-13,
        6.3172857834537352e-11, -3.9953927358802933e-21, -3.9953736840885523e-21, 3.9977478865556259e-13, -3.9953927360327871e-21, -5.9966648497289467e-13, 3.9977505096355127e-13, -3.9954022618142814e-21, -5.9966624668756479e-13, 3.9977491721888488e-13,
        -0.006320894759591536, 3.9977406440545936e-13, 3.9977223443953922e-13, 0.79994000914699848, 3.9977375925256194e-13, -0.19996001206631031, -0.20000999209702264, 3.9977471237486769e-13, -0.1999600120782265, -0.20000999210750431,
        6.3172857844182174e-11, -3.995392736490294e-21, -3.9953736846985492e-21, 3.9977471239593636e-13, -3.995392736642787e-21, -5.9966579840747568e-13, 3.9977459325327153e-13, -3.9954022624242836e-21, -5.9966632302345294e-13, 3.9977484095925876e-13,
        0.0094814051859116698, -5.9966489328558725e-13, -5.9966195759236652e-13, -0.19996001207631928, -5.9966481700689482e-13, 0.79991501812599974, -0.19996001208525652, -5.9966624669986022e-13, -0.20003498310610496, -0.19996001205522895,
        -0.0063208947595161684, 3.99773682950039e-13, 3.9977223443477243e-13, -0.20000999210298054, 3.9977375924779515e-13, -0.19996001205677427, 0.79994000915295671, 3.997747123701009e-13, -0.19996001207822675, -0.20000999211704043,
        6.3172857835742877e-11, -3.9953927359565391e-21, -3.9953736841647973e-21, 3.9977501752976732e-13, -3.9953927361090321e-21, -5.9966663755887511e-13, 3.9977489838710244e-13, -3.9954022618905272e-21, -5.9966601782289172e-13, 3.997747646424361e-13,
        0.0094814051858212317, -5.996647406996058e-13, -5.9966218645703868e-13, -0.19996001207250497, -5.9966466442091337e-13, -0.20003498312970089, -0.19996001208144221, -5.9966609411387877e-13, 0.79991501813815269, -0.19996001205141464,
        -0.0063208947595764613, 3.9977391182424449e-13, 3.9977170040767079e-13, -0.20000999209725895, 3.9977398812200064e-13, -0.1999600120701249, -0.20000999209130085, 3.997749412443064e-13, -0.19996001207250483, 0.79994000912912411;
    const Real h = 1e-5;
    const DenseMatrix Ah = A * h;
    const Real err = (expm_aug(Ah) - ref).norm() / ref.norm();
    const Real est = pade_defect(A, Vector::Zero(10), h);
    INFO("err " << err << " est " << est);
    REQUIRE(err <= 1e-10);
    REQUIRE(est <= kExactTierBound);
    REQUIRE(est >= Real{0.5} * err);   // an estimate (1.3-10x above on probes), not a bound
}

namespace {
// max over τ ∈ [0, δ] (fine grid) of |x_j(τ) − chord_j(τ)| / tol_j
Real chord_ratio(const ExactLTI& e, const Vector& x, const Vector& b, Real d, const Vector& tol) {
    const Vector xd = exact_advance(e, x, b, d);
    Real worst = 0.0;
    for (int i = 1; i < 400; ++i) {
        const Real f = i / 400.0;
        const Vector xt = exact_advance(e, x, b, f * d);
        for (Eigen::Index j = 0; j < x.size(); ++j) {
            worst = std::max(worst, std::abs(xt[j] - ((1 - f) * x[j] + f * xd[j])) / tol[j]);
        }
    }
    return worst;
}
}  // namespace

TEST_CASE("Exact: the output rule's step is a chord bound for every mode structure",
          "[v2][dsed][exact_lti][output_rule]") {
    // a slow RC charging a state that a 1e12 /s mode slaves to it (a parasitic mode;
    // from x2 = 1e-14 its amplitude is the lag ẋ1/1e12 = 5e-9, 1e-9 of the scale)
    DenseMatrix A1(2, 2);
    A1 << -1e3, 0.0, 1e12, -1e12;
    Vector b1(2);
    b1 << 5e3, 0.0;
    // a damped resonance (complex pair), and a growing mode next to a decaying one
    DenseMatrix A2(2, 2);
    A2 << 0.0, 1.0, -4e8, -2e3;
    DenseMatrix A3(2, 2);
    A3 << 5e2, 0.0, 1e3, -1e4;
    const Vector zero = Vector::Zero(2);
    struct Case { DenseMatrix A; Vector b; Vector x; Real h; };
    Vector x1(2), x1_alive(2), x2(2), x3(2);
    x1 << 0.0, 1e-14;
    x1_alive << 0.0, 1.0;   // the fast mode right after an event: alive
    x2 << 1.0, 0.0;
    x3 << 1.0, 1.0;
    const Case cases[] = {{A1, b1, x1, 1e-4}, {A1, b1, x1_alive, 1e-4},
                          {A2, zero, x2, 1e-4}, {A3, zero, x3, 1e-3}};
    ChordWork w;
    for (const auto& c : cases) {
        const auto e = make_exact_lti(c.A, c.b);
        REQUIRE(e.valid);
        REQUIRE(e.tier == 1);
        const Vector tol = Vector::Constant(2, 1e-5 * 5.0);
        Vector x = c.x;
        for (int s = 0; s < 20; ++s) {   // twenty consecutive cut steps from the start
            bool bounded = false;
            const Real d = exact_chord_step(e, x, nullptr, nullptr, tol, c.h, bounded, w);
            REQUIRE(bounded);
            REQUIRE(d > 0.0);
            REQUIRE(d <= c.h);
            INFO("A(1,0) = " << c.A(1, 0) << " step " << s << " d = " << d);
            REQUIRE(chord_ratio(e, x, c.b, d, tol) <= 1.0);
            x = exact_advance(e, x, c.b, d);
        }
    }
    // the parasitic mode at rounding does not set the step: the slow curvature does
    // (5e6 V/s² at the start: δ = (8·5e-5/5e6)^½ ≈ 8.9e-6 s; λ²·5e-9 in the bound gave 3e-10)
    const auto e1 = make_exact_lti(A1, b1);
    bool bounded = false;
    const Real d1 = exact_chord_step(e1, x1, nullptr, nullptr, Vector::Constant(2, 5e-5), 1e-4,
                                     bounded, w);
    REQUIRE(d1 > 4e-6);
    // alive (amplitude 1) it does
    const Real d1a = exact_chord_step(e1, x1_alive, nullptr, nullptr, Vector::Constant(2, 5e-5),
                                      1e-4, bounded, w);
    REQUIRE(d1a < 1e-10);
}

TEST_CASE("Exact: the output rule bounds the rows of an output map (a high-impedance node)",
          "[v2][dsed][exact_lti][output_rule]") {
    // group F session 4: the samples are the recovered vector y = O·x. A node read as
    // 5000 Ω × (x1 − x2) holds x's errors times 5000; bounding x alone let its chord
    // error reach 596 × its tolerance (ll06_buck_vcswitch's V(sw)).
    DenseMatrix A(2, 2);
    // a fast ring: ω = 2e6 rad/s, lightly damped; x2 is scaled by 1/ω so the eigenbasis is
    // well conditioned (the unscaled [0 1; -4e12 -2e5] is refused: tier 3)
    A << 0.0, 2e6, -2e6, -2e5;
    const Vector b = Vector::Zero(2);
    Vector x(2);
    x << 1.0, 0.0;
    DenseMatrix O(1, 2);
    O << 5000.0, -1000.0;   // y = 5000·x1 − 5e-4·(2e6·x2) (a node behind 5 kΩ)
    const auto e = make_exact_lti(A, b);
    INFO("reason: " << e.reason);
    REQUIRE(e.tier == 1);
    DenseMatrix P(3, 2);
    P.topRows(2).setIdentity();
    P.bottomRows(1) = O;
    e.chord_P_re = P * e.V_re;
    e.chord_P_im = P * e.V_im;
    Vector tol(3);
    tol << 1e-5 * 2.0, 1e-5 * 2.0, 1e-5 * 2.0;   // the node's range is that of x1's, not 5000×
    ChordWork w;
    for (int s = 0; s < 20; ++s) {
        bool bounded = false;
        const Real d = exact_chord_step(e, x, nullptr, nullptr, tol, 1e-5, bounded, w);
        REQUIRE(bounded);
        const Vector xd = exact_advance(e, x, b, d);
        Real worst = 0.0;
        for (int i = 1; i < 400; ++i) {
            const Real f = i / 400.0;
            const Vector xt = exact_advance(e, x, b, f * d);
            const Real yt = (O * xt)(0), y0 = (O * x)(0), y1 = (O * xd)(0);
            worst = std::max(worst, std::abs(yt - ((1 - f) * y0 + f * y1)) / tol[2]);
        }
        INFO("step " << s << " d = " << d);
        REQUIRE(worst <= 1.0);
        REQUIRE(worst > 0.5);   // and the bound is not idle: the node sets the step
        x = xd;
    }
}

TEST_CASE("Exact: a row is promised down to its rounding floor, never hiding a variation above it",
          "[v2][dsed][exact_lti][output_rule]") {
    // group F session 7 (D-2 / D-3): no row is promised below its own rounding floor,
    // 16·u·Σ|O||V||z|. y = x2 − x1 with x1 = 1e6 held and x2 relaxing to it at 1e9 /s: the
    // floor is ~7e-9 (the 1e6 terms), the row's real variation is the fast mode's a.
    DenseMatrix A(2, 2);
    A << 0.0, 0.0, 1e9, -1e9;
    const Vector b = Vector::Zero(2);
    const auto e = make_exact_lti(A, b);
    REQUIRE(e.tier == 1);
    DenseMatrix O(3, 2);
    O.topRows(2).setIdentity();
    O.row(2) << -1.0, 1.0;
    e.chord_P_re = O * e.V_re;
    e.chord_P_im = O * e.V_im;
    e.chord_M = O.cwiseAbs() * e.V_re.binaryExpr(e.V_im, [](Real p, Real q) { return std::hypot(p, q); });
    Vector tol(3);
    tol << 1.0, 1.0, 1e-12;   // the states never bind; the row is held to 1e-12
    ChordWork w;
    auto step = [&](Real a, bool& bounded) {
        Vector x(2);
        x << 1e6, 1e6 + a;
        return exact_chord_step(e, x, nullptr, nullptr, tol, 1e-6, bounded, w);
    };
    bool bounded = false;
    (void)step(0.0, bounded);
    const Real floor = w.te[2];
    INFO("floor " << floor);
    REQUIRE(floor > 1e-9);
    REQUIRE(floor < 1e-7);
    // a real variation just above the floor is refined: the 1 ns mode is resolved
    const Real d_hi = step(1.2 * floor, bounded);
    REQUIRE(bounded);
    REQUIRE(d_hi < 1e-8);   // h = 1e-6: cut to the 1 ns mode's scale
    {
        Vector x0(2);
        x0 << 1e6, 1e6 + 1.2 * floor;
        const Vector x1 = exact_advance(e, x0, b, d_hi);
        Real worst = 0.0;
        for (int i = 1; i < 200; ++i) {
            const Real f = i / 200.0;
            const Vector xt = exact_advance(e, x0, b, f * d_hi);
            const Real line = (1 - f) * (x0[1] - x0[0]) + f * (x1[1] - x1[0]);
            worst = std::max(worst, std::abs((xt[1] - xt[0]) - line));
        }
        REQUIRE(worst <= floor);   // the line it stores is within the promise
    }
    // below it (the bound's 2|γ| = half the floor) the row is constant up to rounding: the
    // step is not cut, and the certification is reported (w.fu: the floor it needed)
    const Real d_lo = step(0.25 * floor, bounded);
    REQUIRE(bounded);
    REQUIRE(d_lo == 1e-6);
    REQUIRE(std::abs(w.fu[2] - floor) <= 1e-6 * floor);
    REQUIRE(w.fu[0] == 0.0);
}

// add-exact-propagator-strategy 10.1 / 10.4 (D8): the fixed-step pair
// x <- Phi x + Gamma of (mask, h) against the modal step it replaces; its
// counters; the byte budget.
TEST_CASE("Exact: the fixed-step pair equals the modal step (tier 1, DC and sine)",
          "[v2][dsed][exact_lti][fixed_step]") {
    for (const bool sine : {false, true}) {
        builder::CircuitBuilder b;
        if (sine) b.add_sine_voltage_source("V", "in", "gnd", 2.0, 5.0, 60.0, 0.7);
        else b.add_voltage_source("V", "in", "gnd", 10.0);
        b.add_resistor("R", "in", "a", 5.0);
        b.add_inductor("L", "a", "o", 100e-6);
        b.add_capacitor("C", "o", "gnd", 10e-6);
        b.add_resistor("RL", "o", "gnd", 20.0);
        pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
        dsed::NativeCircuitBuilderAdapter<> ad{b.graph(), b.pool(), cache};
        ad.set_mask(topology::SwitchStateMask(0));
        REQUIRE(ad.has_exact_step());
        const Real h = 1e-6;
        Vector x = Vector::Zero(ad.n_state()), y(x.size()), z(x.size());
        const std::size_t eig0 = ad.n_eig();
        Real t = 0.0;
        for (int k = 0; k < 200; ++k, t += h) {
            REQUIRE(ad.exact_fixed_step_into(t, x, h, y));
            ad.exact_advance_state_into(t, x, h, z);
            REQUIRE((y - z).cwiseAbs().maxCoeff() <= 1e-13 * std::max(1.0, z.cwiseAbs().maxCoeff()));
            x = z;
        }
        REQUIRE(x.cwiseAbs().maxCoeff() > 0.1);            // the circuit moved
        const std::size_t N = static_cast<std::size_t>(x.size() + (sine ? 2 : 0));
        REQUIRE(ad.n_fixed_steps() == 200);
        REQUIRE(ad.n_phi_gamma_builds() == 1);
        REQUIRE(ad.fixed_step_flops() == 200 * (N * N + N));
        REQUIRE(ad.phi_gamma_bytes() == (N * N + N) * sizeof(Real));
        REQUIRE(ad.n_eig() == eig0);                         // no factorization per step
        // a second length is a second pair; with the budget spent it is refused
        ad.set_phi_gamma_budget(ad.phi_gamma_bytes());
        REQUIRE_FALSE(ad.exact_fixed_step_into(t, x, h / 2, y));
        REQUIRE(ad.n_phi_gamma_refused() == 1);
        REQUIRE(ad.exact_fixed_step_into(t, x, h, y));       // the cached one still steps
    }
}

// Reviewer item (phase 3): the pair cache's eviction under a byte budget of four pairs, on an
// RC whose step has a closed form x(t+h) = x e^{-h/tau} + V (1 - e^{-h/tau}).
TEST_CASE("Exact: the fixed-step pair cache evicts the oldest length and stays exact",
          "[v2][dsed][exact_lti][fixed_step]") {
    const Real V = 10.0, R = 1e3, C = 1e-6, tau = R * C;
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", V);
    b.add_resistor("R", "in", "o", R);
    b.add_capacitor("C", "o", "gnd", C);
    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    dsed::NativeCircuitBuilderAdapter<> ad{b.graph(), b.pool(), cache};
    ad.set_mask(topology::SwitchStateMask(0));
    REQUIRE(ad.has_exact_step());
    REQUIRE(ad.n_state() == 1);
    const std::size_t pair = 2 * sizeof(Real);            // N = 1: Phi 1x1, Gamma 1
    ad.set_phi_gamma_budget(4 * pair);
    Vector x = Vector::Zero(1), y(1);
    Real t = 0, x_ref = 0, err = 0;
    // lengths 1..5, then 1 again: the fifth evicts the first, the sixth (1) evicts the second
    for (const int k : {1, 2, 3, 4, 5, 1}) {
        const Real h = k * 1e-4;
        REQUIRE(ad.exact_fixed_step_into(t, x, h, y));
        x_ref = x_ref * std::exp(-h / tau) + V * (1 - std::exp(-h / tau));
        err = std::max(err, std::abs(y[0] - x_ref));
        x = y;
        t += h;
        REQUIRE(ad.phi_gamma_bytes() <= 4 * pair);
    }
    INFO("max |x - closed form| " << err);
    REQUIRE(err <= 1e-13 * V);
    REQUIRE(ad.n_phi_gamma_builds() == 6);
    REQUIRE(ad.phi_gamma_bytes() == 4 * pair);
    REQUIRE(ad.n_phi_gamma_refused() == 0);
    // a budget below one pair refuses (nothing evicted, nothing written) and the caller steps
    ad.set_phi_gamma_budget(pair - 1);
    REQUIRE_FALSE(ad.exact_fixed_step_into(t, x, 7e-4, y));
    REQUIRE(ad.n_phi_gamma_refused() == 1);
    REQUIRE(ad.phi_gamma_bytes() == 4 * pair);
    REQUIRE(ad.exact_fixed_step_into(t, x, 1e-4, y));       // still cached (no eviction on refusal)
}

TEST_CASE("Exact: the per-h modal coefficients cached by the step are the ones it would compute",
          "[v2][dsed][exact_lti]") {
    // series RLC (complex pair) driven by b: a step on a repeated h reuses e^{λh} and
    // h·φ₁(λh)·w; every step must equal, bit for bit, the step of a fresh propagator
    DenseMatrix A(2, 2);
    A << -0.5, -1.0, 1.0, -1e-3;
    Vector b(2);
    b << 3.0, 0.25;
    const auto cached = make_exact_lti(A, b);
    REQUIRE(cached.valid);
    REQUIRE(cached.tier == 1);
    Vector x(2), y(2), ref(2);
    x << 1.0, -2.0;
    for (const Real h : {0.1, 0.1, 0.37, 0.1, 0.37, 0.37, 0.0, 0.1}) {
        const auto fresh = make_exact_lti(A, b);
        exact_advance_into(fresh, x, h, ref);
        exact_advance_into(cached, x, h, y);
        INFO("h = " << h);
        REQUIRE(std::memcmp(y.data(), ref.data(), sizeof(Real) * 2) == 0);
        x = y;
    }
}
