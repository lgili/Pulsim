// add-unified-device-interface step 4: the Newton device contract
// (device/device.hpp), the AD functions it needs (ad/ad_scalar.hpp) and the
// stateful-kind registry (device/kind.hpp, pwl/stateful_histories.hpp).

#include "pulsim/ad/ad_scalar.hpp"
#include "pulsim/device/device.hpp"
#include "pulsim/pwl/stateful_histories.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/stamping/stamp_companion.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

using namespace pulsim;
using device::Formula;
using device::Out;
using device::StepCtx;

namespace {

// A linear capacitor and a linear inductor written ONCE as devices: the
// law is a template over the parameter scalar P, the unknown scalar X and
// the output scalar O.
struct LinCap {
    static constexpr std::size_t n_pins = 2, n_internal = 0, n_params = 1;
    template <class S> struct Params {
        S C;
        template <class F> auto map(F&& f) const {
            return Params<decltype(f(C))>{f(C)};
        }
    };
    template <class P, class X, class O>
    static void eval(const Params<P>& p, const std::array<X, 2>& x,
                     Out<O, 2>& o) {
        const O q = p.C * (x[0] - x[1]);
        o.f = {O{0}, O{0}};
        o.q = {q, -q};
    }
};

struct LinInd {
    static constexpr std::size_t n_pins = 2, n_internal = 1, n_params = 1;
    template <class S> struct Params {
        S L;
        template <class F> auto map(F&& f) const {
            return Params<decltype(f(L))>{f(L)};
        }
    };
    // x = (v_a, v_b, i_L): KCL ±i on the pins, the branch row
    // v_a − v_b + d/dt(−L·i) = 0 (conductance form: diagonal −a·L).
    template <class P, class X, class O>
    static void eval(const Params<P>& p, const std::array<X, 3>& x,
                     Out<O, 3>& o) {
        o.f = {O{x[2]}, O{-x[2]}, O{x[0] - x[1]}};
        o.q = {O{0}, O{0}, O{-(p.L * x[2])}};
    }
};

static_assert(device::Device<LinCap>);
static_assert(device::Device<LinInd>);

StepCtx ctx(Formula f, Real h) { return StepCtx{0.0, f, h}; }

Real g_of(Formula f, Real h) {
    if (f == Formula::BackwardEuler) return 1.0 / h;
    if (f == Formula::Bdf2Stage2) return device::bdf2_weights().c1 / h;
    return 2.0 / h;
}

}  // namespace

TEST_CASE("a linear capacitor as a Device reproduces the companion stamp",
          "[device][interface]") {
    const Real C = 3.3e-9, h = 1e-7;
    const LinCap::Params<Real> p{C};
    for (Formula f : {Formula::Trapezoidal, Formula::BackwardEuler,
                      Formula::Bdf2Stage2}) {
        const auto c = ctx(f, h);
        std::array<Real, 2> F{}, x{12.0, 2.0};
        std::array<std::array<Real, 2>, 2> J{};
        const std::array<Real, 2> q_n{C * 9.0, -C * 9.0};
        const std::array<Real, 2> qd_n{0.4, -0.4};
        const std::array<Real, 2> q_g{C * 9.5, -C * 9.5};
        device::stamp_fq<LinCap>(p, x, c, q_n, qd_n, q_g, F, J);

        sparse::Matrix Jc(2, 2);
        Vector b = Vector::Zero(2);
        stamping::stamp_capacitor_companion(Jc, b, {0, 1}, C * g_of(f, h));
        for (int r = 0; r < 2; ++r) {
            for (int k = 0; k < 2; ++k) {
                REQUIRE(J[r][k] == Catch::Approx(Jc.coeff(r, k)).epsilon(1e-14));
            }
        }
        // The companion current from the committed (q_n, i_n).
        const Real v = 10.0, v_n = 9.0, i_n = 0.4;
        Real i_expect = 0.0;
        if (f == Formula::Trapezoidal) i_expect = (2.0 * C / h) * (v - v_n) - i_n;
        if (f == Formula::BackwardEuler) i_expect = (C / h) * (v - v_n);
        if (f == Formula::Bdf2Stage2) {
            const auto w = device::bdf2_weights();
            i_expect = C * (w.c1 * v + w.c2 * 9.5 + w.c3 * v_n) / h;
        }
        REQUIRE(F[0] == Catch::Approx(i_expect).epsilon(1e-12));
        REQUIRE(F[1] == Catch::Approx(-i_expect).epsilon(1e-12));
    }
}

TEST_CASE("a linear inductor as a Device reproduces the companion stamp "
          "in conductance form", "[device][interface]") {
    const Real L = 2e-3, h = 5e-7;
    const LinInd::Params<Real> p{L};
    for (Formula f : {Formula::Trapezoidal, Formula::BackwardEuler,
                      Formula::Bdf2Stage2}) {
        const auto c = ctx(f, h);
        std::array<Real, 3> F{}, x{5.0, 1.0, 0.75};
        std::array<std::array<Real, 3>, 3> J{};
        const Real i_n = 0.7, v_n = 3.0;
        // q = −L·i, q̇ = −v (the branch voltage it balances)
        const std::array<Real, 3> q_n{0, 0, -L * i_n}, qd_n{0, 0, -v_n},
            q_g{0, 0, -L * 0.72};
        device::stamp_fq<LinInd>(p, x, c, q_n, qd_n, q_g, F, J);

        sparse::Matrix Jc(3, 3);
        Vector b = Vector::Zero(3);
        stamping::stamp_inductor_companion(Jc, b, {0, 1}, 2,
                                           1.0 / (L * g_of(f, h)));
        for (int r = 0; r < 3; ++r) {
            for (int k = 0; k < 3; ++k) {
                REQUIRE(J[r][k] == Catch::Approx(Jc.coeff(r, k)).epsilon(1e-14));
            }
        }
        // the flux-row diagonal is −a·L (never rescaled by h)
        REQUIRE(J[2][2] == Catch::Approx(-L * g_of(f, h)).epsilon(1e-15));
        if (f == Formula::Trapezoidal) {
            // v + v_n = (2L/h)(i − i_n)
            REQUIRE(F[2] == Catch::Approx(4.0 + v_n - (2 * L / h) * (0.75 - i_n))
                                .epsilon(1e-12));
        }
    }
}

TEST_CASE("commit_fq stores the physical charge and its derivative",
          "[device][interface]") {
    const Real C = 1e-6, h = 1e-6;
    const LinCap::Params<Real> p{C};
    std::array<Real, 2> q_n{C * 1.0, -C * 1.0}, qd{0.0, 0.0};
    const std::array<Real, 2> qg{};
    device::commit_fq<LinCap>(p, {2.0, 0.0}, ctx(Formula::Trapezoidal, h),
                              q_n, qd, qg);
    REQUIRE(q_n[0] == Catch::Approx(2e-6));
    REQUIRE(qd[0] == Catch::Approx(2.0 * C / h * 1.0));   // i = 2C/h·Δv − 0
}

TEST_CASE("parameters seeded as AD give the parameter derivative of the "
          "same law", "[device][interface][sensitivity]") {
    const LinInd::Params<Real> p{2e-3};
    const auto pa = device::seed_params<LinInd, 1>(p);
    REQUIRE(pa.L.value() == 2e-3);
    REQUIRE(pa.L.deriv(0) == 1.0);
    Out<ad::ADRealN<1>, 3> o;
    LinInd::eval(pa, std::array<Real, 3>{5.0, 1.0, 0.75}, o);
    REQUIRE(o.q[2].value() == Catch::Approx(-2e-3 * 0.75));
    REQUIRE(o.q[2].deriv(0) == Catch::Approx(-0.75));     // ∂q/∂L = −i
    // the Real instantiation is the same arithmetic as the AD value
    Out<Real, 3> r;
    LinInd::eval(p, std::array<Real, 3>{5.0, 1.0, 0.75}, r);
    REQUIRE(r.q[2] == o.q[2].value());
}

namespace {
// Central difference of a scalar function of one AD input.
Real fd(const std::function<Real(Real)>& f, Real x) {
    const Real e = 1e-6 * std::max(1.0, std::abs(x));
    return (f(x + e) - f(x - e)) / (2 * e);
}
ad::ADRealN<1> var(Real v) { return ad::ADRealN<1>{v, {1.0}}; }
}  // namespace

TEST_CASE("the AD functions the device laws need match finite differences",
          "[device][ad]") {
    for (Real x : {-1.7, -0.3, 0.4, 2.2}) {
        REQUIRE(ad::sin(var(x)).deriv(0)
                == Catch::Approx(fd([](Real v) { return std::sin(v); }, x)).epsilon(1e-8));
        REQUIRE(ad::cos(var(x)).deriv(0)
                == Catch::Approx(fd([](Real v) { return std::cos(v); }, x)).epsilon(1e-8));
        REQUIRE(ad::atan(var(x)).deriv(0)
                == Catch::Approx(fd([](Real v) { return std::atan(v); }, x)).epsilon(1e-8));
        REQUIRE(ad::atan(var(x)).value() == std::atan(x));
    }
    for (Real x : {0.3, 1.5, 4.0}) {
        REQUIRE(ad::pow(var(x), 2.5).deriv(0)
                == Catch::Approx(fd([](Real v) { return std::pow(v, 2.5); }, x)).epsilon(1e-8));
        // x^y with both seeded
        const ad::ADRealN<2> X{x, {1.0, 0.0}}, Y{1.3, {0.0, 1.0}};
        const auto z = ad::pow(X, Y);
        REQUIRE(z.deriv(0) == Catch::Approx(fd([](Real v) { return std::pow(v, 1.3); }, x)).epsilon(1e-8));
        REQUIRE(z.deriv(1) == Catch::Approx(fd([x](Real v) { return std::pow(x, v); }, 1.3)).epsilon(1e-8));
    }
    // atan2 in all four quadrants
    for (auto [yv, xv] : std::vector<std::pair<Real, Real>>{
             {0.5, 1.0}, {0.5, -1.0}, {-0.5, -1.0}, {-0.5, 1.0}}) {
        const ad::ADRealN<2> Y{yv, {1.0, 0.0}}, X{xv, {0.0, 1.0}};
        const auto z = ad::atan2(Y, X);
        REQUIRE(z.value() == std::atan2(yv, xv));
        REQUIRE(z.deriv(0) == Catch::Approx(fd([xv](Real v) { return std::atan2(v, xv); }, yv)).epsilon(1e-8));
        REQUIRE(z.deriv(1) == Catch::Approx(fd([yv](Real v) { return std::atan2(yv, v); }, xv)).epsilon(1e-8));
    }
    // chain: value and tangent from elsewhere, one law body for both scalars
    const auto c = ad::chain(var(2.0), 7.0, 3.0);
    REQUIRE(c.value() == 7.0);
    REQUIRE(c.deriv(0) == 3.0);
    REQUIRE(ad::chain(2.0, 7.0, 3.0) == 7.0);
}

TEST_CASE("the stateful-kind registry: one rollback vector carries every "
          "real, stage scratch included", "[device][registry]") {
    static_assert(pwl::StatefulKinds::size == 5);
    static_assert(device::kNumStateBlocks == pwl::StatefulKinds::size + 1);
    using topology::BranchKind;
    using topology::Graph;
    Graph g;
    g.add_node("a");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::Nonlinear);
    pwl::DevicePool pool;
    pool.add_voltage_source(0, {.V = 5.0});
    pool.add_saturable_inductor(
        1, {.L_0 = 1e-3, .I_sat = 5.0, .L_residual = 5e-5});
    pwl::StatefulHistories hs;
    hs.init(g, pool);
    REQUIRE(hs.any());
    auto& sat = hs.get<pwl::SaturableInductorHistory>();
    REQUIRE(sat.entries().size() == 1);

    const Index row = sat.entries()[0].branch_var_id;
    Vector x = Vector::Zero(std::max<Index>(row + 1, 2));
    x[0] = 5.0;
    x[row] = 1.25;
    hs.commit(x, 1e-6, pwl::TrBdf2Stage::Trapezoidal);
    x[row] = 1.5;
    hs.capture_gamma(x);   // stage scratch: lambda_gamma, i_gamma
    pwl::StatefulHistories::Snapshot s;
    hs.snapshot_into(s);
    REQUIRE(s.size() == 13);   // 7 snapshot reals + 6 stage scratch
    const auto before = sat.entries()[0];

    x[row] = 3.0;
    hs.capture_gamma(x);
    hs.commit(x, 1e-6, pwl::TrBdf2Stage::Trapezoidal);
    REQUIRE(sat.entries()[0].i_gamma != before.i_gamma);
    hs.restore(s);
    const auto& after = sat.entries()[0];
    REQUIRE(after.lambda_old == before.lambda_old);
    REQUIRE(after.i_L_old == before.i_L_old);
    REQUIRE(after.V_L_old == before.V_L_old);
    REQUIRE(after.lambda_gamma == before.lambda_gamma);
    REQUIRE(after.i_gamma == before.i_gamma);
    REQUIRE(after.ja_state.n_sub == before.ja_state.n_sub);
    // a rollback point of another size is refused, never read past its end
    const pwl::StatefulHistories::Snapshot short_s(s.begin(), s.end() - 1);
    REQUIRE_THROWS_AS(hs.restore(short_s), std::logic_error);
    REQUIRE_THROWS_AS(hs.restore(pwl::StatefulHistories::Snapshot{}),
                      std::logic_error);
    // and the snapshot layout is the kinds' own blocks
    pwl::HistoryState history(g, pool);
    const auto l = hs.layout(history);
    REQUIRE(l.count[static_cast<std::size_t>(device::StateBlock::SaturableInductor)] == 1);
    REQUIRE(l.size() == 7);
}
