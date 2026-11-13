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
            REQUIRE(kExactRoundoff * e.norm_A1 * h <= kExactTierBound);
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

TEST_CASE("Exact: stiff and near-defective is tier 3, with the reason",
          "[v2][dsed][exact_lti][tier]") {
    // A Jordan block scaled to 1e12 (off-diagonal 1e12 too: with 1.0
    // it is normal to rounding, kappa2 = 4.5e3, a legitimate tier 1).
    // kappa2 = 3e15 and ||A*h||_1 = 2e7: neither bound holds.
    DenseMatrix A(2, 2);
    A << -1e12, 1e12, 0.0, -1e12;
    const auto e = make_exact_lti(A, Vector::Zero(2), Vector{}, 1e-5);
    REQUIRE_FALSE(e.valid);
    REQUIRE(e.tier == 3);
    REQUIRE(e.reason.find("stiff") != std::string::npos);
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
