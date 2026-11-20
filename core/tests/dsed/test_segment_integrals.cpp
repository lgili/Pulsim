// =============================================================================
// DSED — the closed-form segment integrals of the recorders (F-B2-1)
// =============================================================================
//
// pulsim/dsed/segment_integrals.hpp: ∫y and ∫y² of a degree-1 form l·x̄ and a
// degree-2 form x̄ᵀQx̄ over a segment of x̄' = As·x̄ (and over a piece inside
// it), against 50-digit mpmath truths (mp.quad on mp.expm of the same As,
// .scratch/wi/seg_truth.py of the branch fix/core-exact-window-integrals; no
// pulsim). The hard cases: a zero eigenvalue (an integrator: defective with
// the augmented constant), |λ|h = 1e-12, a repeated root, a complex pair, a
// near-defective pair on both sides of the modal bar and a defective one.

#include <catch2/catch_test_macros.hpp>

#include "pulsim/dsed/segment_integrals.hpp"

#include <array>
#include <cmath>
#include <vector>

using namespace pulsim;

namespace {

bool rel(Real a, Real b, Real tol) {
    const Real e = std::abs(a - b) / std::max(std::abs(a), std::abs(b));
    UNSCOPED_INFO("relative error " << e);
    return e <= tol;
}

dsed::detail::PeriodMode mode_of(const DenseMatrix& A, const Vector& b) {
    dsed::detail::PeriodMode e;
    e.A = A;
    e.n = A.rows();
    e.b = b;
    e.d = Vector::Ones(e.n + 1);
    e.As = DenseMatrix::Zero(e.n + 1, e.n + 1);
    e.As.topLeftCorner(e.n, e.n) = A;
    e.As.topRightCorner(e.n, 1) = b;
    dsed::detail::modal_basis(e);
    return e;
}

struct Case {
    DenseMatrix A;
    Vector b, x0;
    Real s0, s1;
    std::array<Real, 4> truth;   // ∫l·x̄, ∫(l·x̄)², ∫x̄ᵀQx̄, ∫(x̄ᵀQx̄)²
};

constexpr Real kH = 1e-5;

std::vector<dsed::SegForm> forms() {
    Vector l(3);
    l << 1.5, -0.7, 0.3;
    DenseMatrix Q(3, 3);
    Q << 1, 0.5, 0.25, 0.5, -2, 0.1, 0.25, 0.1, 3;
    dsed::SegForm q;
    q.deg = 2;
    q.Q = Q;
    return {dsed::seg::linear(l), q};
}

DenseMatrix m2(Real a, Real b, Real c, Real d) {
    DenseMatrix M(2, 2);
    M << a, b, c, d;
    return M;
}
Vector v2(Real a, Real b) {
    Vector v(2);
    v << a, b;
    return v;
}

/// `modal`: the mode has a modal basis; `sq2`: the degree-2 square has a
/// closed form (modal under u·κ⁴ ≤ 1e-10, else the Kronecker Gramian up to
/// n + 1 = 12; else NaN: the recorder's line)
void check(const Case& c, bool modal, bool sq2, Real tol, const char* name) {
    INFO(name);
    dsed::SegmentMode sm(mode_of(c.A, c.b), forms());
    REQUIRE(sm.mode().modal == modal);
    sm.start(c.x0);
    Real out[4], mag[2];
    const char* why = nullptr;
    REQUIRE(sm.integrate(0, c.s0, c.s1 - c.s0, out, mag, &why));
    CHECK(rel(out[0], c.truth[0], tol));
    CHECK(rel(out[1], c.truth[1], tol));
    REQUIRE(sm.integrate(1, c.s0, c.s1 - c.s0, out, mag, &why));
    CHECK(rel(out[0], c.truth[2], tol));
    if (sq2) {
        CHECK(rel(out[1], c.truth[3], tol));
    } else {
        CHECK(std::isnan(out[1]));
    }
}

}  // namespace

TEST_CASE("segment integrals: a zero eigenvalue (an integrator) takes the matrix form",
          "[dsed][segment_integrals]") {
    // x1' = 3e5 (a ramp: λ = 0 twice with the augmented constant, defective)
    const DenseMatrix A = m2(0, 0, 1e5, -2e5);
    check({A, v2(3e5, 0), v2(0.2, -0.1), 0, kH,
           {2.6135520420230483275e-5, 8.0109609847136084682e-5, 7.9798460447170251618e-5,
            7.8903166637553183949e-4}},
          false, true, 1e-12, "zero");
    check({A, v2(3e5, 0), v2(0.2, -0.1), 0.3 * kH, 0.7 * kH,
           {1.0538236756106591161e-5, 2.8508793469835206003e-5, 2.9068843164262229563e-5,
            2.2080743895324168771e-4}},
          false, true, 1e-12, "zero, a piece inside the step");
}

TEST_CASE("segment integrals: |lambda| h = 1e-12 in the phi form loses no digits",
          "[dsed][segment_integrals]") {
    const Real eps = 1e-12 / kH;
    check({m2(-eps, 0, 0, -3e5), v2(0, 0), v2(1.0, 2.0), 0, kH,
           {1.3565672985709199986e-5, 1.9694992158113077526e-5, 3.9301420148731697398e-5,
            1.7146058977800634923e-4}},
          true, true, 1e-13, "tiny");
}

TEST_CASE("segment integrals: a repeated root and a complex pair (modal)",
          "[dsed][segment_integrals]") {
    check({m2(-2e5, 0, 0, -2e5), v2(1e5, 2e5), v2(0.5, -0.25), 0, kH,
           {7.2829081358398199492e-6, 5.7520409175274061223e-6, 3.2164300712063101274e-5,
            1.042235506554714676e-4}},
          true, true, 1e-12, "repeated");
    const DenseMatrix C = m2(-1e4, 3e5, -3e5, -1e4);
    check({C, v2(2e5, 0), v2(0.1, 0.3), 0, kH,
           {1.7107125901686488848e-5, 3.2670717057580246095e-5, 1.5863091535394489942e-5,
            7.4235792967662690846e-5}},
          true, true, 1e-12, "complex");
    check({C, v2(2e5, 0), v2(0.1, 0.3), 0.25 * kH, 0.9 * kH,
           {1.3365378463470541788e-5, 2.7857442165847962982e-5, 9.356003948384854163e-6,
            3.9842385739272038754e-5}},
          true, true, 1e-12, "complex, a piece inside the step");
}

TEST_CASE("segment integrals: a near-defective pair on both sides of the modal bar",
          "[dsed][segment_integrals]") {
    const Real a = 2e5;
    // δ/a = 3e-2: κ₂(V) ~ 1e2: the products modal (u·κ²), the degree-2
    // square not (u·κ⁴ > 1e-10): the Kronecker Gramian
    check({m2(-a, a, 0, -a * 1.03), v2(0, a), v2(0.4, -0.3), 0, kH,
           {5.2625290114519795526e-6, 3.0399910578346514387e-6, 3.0026866488965023909e-5,
            9.0342351812928693572e-5}},
          true, true, 1e-12, "near 3e-2");
    // δ/a = 1e-3: κ₂(V) ~ 1e3, a modal basis, but the products' sums would
    // carry u·κ² (8e-10 measured) and u·κ⁴ (6e-6): the matrix forms
    check({m2(-a, a, 0, -a * (1 + 1e-3)), v2(0, a), v2(0.4, -0.3), 0, kH,
           {5.2521534356381815674e-6, 3.0315828413420404843e-6, 2.9949805828848605458e-5,
            8.9897575415780687294e-5}},
          true, true, 1e-12, "near 1e-3");
    // δ/a = 1e-8: κ₂(V) ~ 1e8, past the bar: the matrix form
    check({m2(-a, a, 0, -a * (1 + 1e-8)), v2(0, a), v2(0.4, -0.3), 0, kH,
           {5.251790274142556785e-6, 3.0312886819232601032e-6, 2.9947089357029583989e-5,
            8.9881937983882369421e-5}},
          false, true, 1e-12, "near 1e-8");
    // defective (a Jordan block)
    check({m2(-a, a, 0, -a), v2(0, a), v2(0.4, -0.3), 0, kH,
           {5.2517902705090913158e-6, 3.031288678980216142e-6, 2.9947089329844537009e-5,
            8.9881937827405194254e-5}},
          false, true, 1e-12, "defective");
    check({m2(-a, a, 0, -a), v2(0, a), v2(0.4, -0.3), 0.2 * kH, 0.6 * kH,
           {1.7292035088364303409e-6, 7.5976784294050602717e-7, 1.220061046086749684e-5,
            3.7253970107649714348e-5}},
          false, true, 1e-12, "defective, a piece inside the step");
}

TEST_CASE("segment integrals: the matrix form on a stiff mode does not overflow",
          "[dsed][segment_integrals]") {
    // λ = −5e12 over 10 µs (|λ|h = 5e7): ∫x1 = x0/a + (b/a)(h − 1/a), e^{−ah} = 0
    const Real a = 5e12, x0 = 0.3;
    DenseMatrix As = DenseMatrix::Zero(3, 3);
    As(0, 0) = -a;
    As(1, 1) = -1e3;
    As(0, 2) = 1;
    As(1, 2) = 1;
    DenseMatrix Phi, I;
    std::vector<DenseMatrix> W;
    const DenseMatrix Q = DenseMatrix::Identity(3, 3);
    const DenseMatrix* qs[1] = {&Q};
    dsed::seg::gram_doubling(As, kH, qs, Phi, I, W);
    REQUIRE(Phi.allFinite());
    REQUIRE(I.allFinite());
    REQUIRE(W[0].allFinite());
    Vector xb(3);
    xb << x0, 0.0, 1.0;
    const Real truth = x0 / a + (1 / a) * (kH - 1 / a);
    CHECK(rel((I * xb)[0], truth, 1e-12));
    CHECK(std::abs(Phi(0, 0)) < 1e-300);
}

TEST_CASE("segment integrals: probe programs compose to degree <= 2 or refuse",
          "[dsed][segment_integrals]") {
    using Op = ir::Expr::Op;
    using I = ir::Expr::Instr;
    Vector l(3), k(3);
    l << 1, 2, 0;
    k << 0, 1, 5;
    const std::vector<dsed::SegForm> at{dsed::seg::linear(l), dsed::seg::linear(k)};
    // 2*(a - b) / 4: degree 1
    auto f = dsed::seg::compose(std::vector<I>{{Op::Const, 2}, {Op::Param, 0, 0}, {Op::Param, 0, 1},
                                               {Op::Sub}, {Op::Mul}, {Op::Const, 4}, {Op::Div}},
                                at, 3);
    REQUIRE(f.deg == 1);
    CHECK((f.l - 0.5 * (l - k)).norm() == 0);
    // a * b: degree 2, x̄ᵀQx̄ = (l·x̄)(k·x̄)
    f = dsed::seg::compose(std::vector<I>{{Op::Param, 0, 0}, {Op::Param, 0, 1}, {Op::Mul}}, at, 3);
    REQUIRE(f.deg == 2);
    Vector x(3);
    x << 0.3, -1.1, 1;
    CHECK(rel(x.dot(f.Q * x), l.dot(x) * k.dot(x), 1e-15));
    // a ^ 2 + b: degree 2 with a linear part
    f = dsed::seg::compose(std::vector<I>{{Op::Param, 0, 0}, {Op::Const, 2}, {Op::Pow},
                                          {Op::Param, 0, 1}, {Op::Add}},
                           at, 3);
    REQUIRE(f.deg == 2);
    CHECK(rel(x.dot(f.Q * x), l.dot(x) * l.dot(x) + k.dot(x), 1e-15));
    // abs(a), a * a * b, a / b: not polynomials of degree <= 2
    CHECK(dsed::seg::compose(std::vector<I>{{Op::Param, 0, 0}, {Op::Abs}}, at, 3).deg == -1);
    CHECK(dsed::seg::compose(std::vector<I>{{Op::Param, 0, 0}, {Op::Param, 0, 0}, {Op::Mul},
                                            {Op::Param, 0, 1}, {Op::Mul}},
                             at, 3).deg == -1);
    CHECK(dsed::seg::compose(std::vector<I>{{Op::Param, 0, 0}, {Op::Param, 0, 1}, {Op::Div}}, at, 3)
              .deg == -1);
}
