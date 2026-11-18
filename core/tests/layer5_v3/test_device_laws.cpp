// add-unified-device-interface, dual path (DECISIONS #23): the ported
// kinds' laws (device/laws.hpp) against their hand stamps, and their
// parameter derivatives against central differences.
//
// No exact float compares between two different expressions: the hand
// Jacobian (C(v), L(i)) and the law's AD derivative are the same function
// evaluated two ways, equal to a few ulp.

#include "pulsim/device/laws.hpp"
#include "pulsim/models/charge_table.hpp"
#include "pulsim/models/device_model.hpp"
#include "pulsim/models/flux_table.hpp"
#include "pulsim/models/jiles_atherton.hpp"
#include "pulsim/pwl/device_readings.hpp"
#include "pulsim/pwl/device_refresh.hpp"
#include "pulsim/sparse/matrix.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pulsim;
namespace L = device::laws;

namespace {

// Deterministic uniform in [lo, hi) (no <random>: its distributions are
// not portable across standard libraries).
struct Lcg {
    std::uint64_t s;
    Real operator()(Real lo, Real hi) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        const Real u = static_cast<Real>(s >> 11) * 0x1.0p-53;
        return lo + (hi - lo) * u;
    }
};

constexpr Real kEps = std::numeric_limits<Real>::epsilon();

// |a − b| within `ulps` of the larger magnitude, or of `scale` when both
// are small (a sum of terms of size `scale` carries their round-off).
bool close(Real a, Real b, Real scale, Real ulps = 8) {
    const Real m = std::max({std::abs(a), std::abs(b), std::abs(scale)});
    return std::abs(a - b) <= ulps * kEps * m;
}

Real dense(const sparse::Matrix& J, Index r, Index c) {
    return J.coeff(r, c);
}

// ---- ∂/∂p by AD against a central difference of the same law -------------
template <class D>
void check_param_tangent(const device::aux_t<D>& aux,
                         const typename D::template Params<Real>& p,
                         const std::array<Real, device::rows_v<D>>& x) {
    constexpr std::size_t R = device::rows_v<D>;
    constexpr std::size_t NP = D::n_params;
    const auto pa = device::seed_params<D, NP>(p);
    device::Out<ad::ADRealN<NP>, R> o;
    device::eval<D>(aux, pa, x, o);
    // The parameters as an array, to perturb one at a time.
    std::array<Real, NP> pv{};
    std::size_t n = 0;
    (void)p.map([&](Real v) { pv[n++] = v; return v; });
    REQUIRE(n == NP);
    for (std::size_t k = 0; k < NP; ++k) {
        // Fourth-order central difference (Richardson): its truncation
        // error stays below 1e-8 relative where a plain one's does not
        // (the Shockley exponent's curvature).
        const Real d = Real{1e-4} * std::max(std::abs(pv[k]), Real{1e-30});
        auto at = [&](Real sgn) {
            std::size_t j = 0;
            const auto pk = p.map([&](Real v) { return j++ == k ? v + sgn * d : v; });
            device::Out<Real, R> r;
            device::eval<D>(aux, pk, x, r);
            return r;
        };
        const auto up = at(Real{1});
        const auto dn = at(Real{-1});
        const auto up2 = at(Real{2});
        const auto dn2 = at(Real{-2});
        for (std::size_t r = 0; r < R; ++r) {
            const Real cd_f = (Real{8} * (up.f[r] - dn.f[r]) - (up2.f[r] - dn2.f[r]))
                              / (Real{12} * d);
            const Real cd_q = (Real{8} * (up.q[r] - dn.q[r]) - (up2.q[r] - dn2.q[r]))
                              / (Real{12} * d);
            // Round-off of the difference: eps·|value|/d.
            const Real floor_f = 1e3 * kEps * std::abs(o.f[r].value()) / d;
            const Real floor_q = 1e3 * kEps * std::abs(o.q[r].value()) / d;
            INFO("param " << k << " row " << r << " AD f " << o.f[r].deriv(k)
                 << " CD f " << cd_f << " AD q " << o.q[r].deriv(k)
                 << " CD q " << cd_q);
            CHECK(std::abs(o.f[r].deriv(k) - cd_f)
                  <= 1e-8 * std::max(std::abs(cd_f), std::abs(o.f[r].deriv(k))) + floor_f);
            CHECK(std::abs(o.q[r].deriv(k) - cd_q)
                  <= 1e-8 * std::max(std::abs(cd_q), std::abs(o.q[r].deriv(k))) + floor_q);
        }
    }
}

// ---- a static law against `evaluate_current_and_jacobian` ----------------
template <class D, class Model>
void check_static(const typename Model::Params& mp, Real lo, Real hi,
                  std::uint64_t seed) {
    constexpr std::size_t R = device::rows_v<D>;
    Lcg rng{seed};
    for (int t = 0; t < 200; ++t) {
        std::array<Real, R> x{};
        for (auto& v : x) v = rng(lo, hi);
        models::ModelInputs<Model> vin{};
        for (std::size_t k = 0; k < R; ++k) vin[k] = x[k];
        const auto [i, g] = models::evaluate_current_and_jacobian<Model>(vin, mp);
        std::array<Real, R> F{};
        std::array<std::array<Real, R>, R> J{};
        constexpr std::array<Real, R> z{};
        device::stamp_fq<D>(device::aux_t<D>{}, D::from(mp), x,
                            {Real{0}, device::Formula::Dc, Real{0}}, z, z, z, F, J);
        CHECK(close(F[0], i, i, 2));
        CHECK(close(F[1], -i, i, 2));
        for (std::size_t k = 0; k < R; ++k) {
            CHECK(close(J[0][k], g[k], g[k], 2));
            CHECK(close(J[1][k], -g[k], g[k], 2));
        }
    }
}

}  // namespace

TEST_CASE("static laws equal the hand AD stamp", "[device][laws]") {
    models::ShockleyDiode::Params sd;
    sd.BV = 5.0;
    check_static<L::ShockleyDiode, models::ShockleyDiode>(sd, -6.0, 0.9, 1);
    check_static<L::IdealDiode, models::IdealDiode>({}, -5.0, 5.0, 2);
    check_static<L::MosfetLevel1, models::MosfetLevel1>({}, -10.0, 10.0, 3);
    check_static<L::IgbtLevel1, models::IgbtLevel1>({}, -2.0, 20.0, 4);
}

TEST_CASE("the Coss law stamps, commits and captures as the hand stamp",
          "[device][laws]") {
    const auto jn = models::NonlinearCapacitor::junction(1e-9, 0.8, 0.45);
    std::vector<Real> tv, tq, tc;
    // knots at -0.3, -0.15, 0 (above the junction floor; the table is
    // referenced to zero bias: Q(0) = 0), then
    // quadratic spacing to 50 V
    for (int k = -2; k < 24; ++k) {
        const Real u = k < 0 ? 0.0 : k / 23.0;
        tv.push_back(k < 0 ? 0.15 * Real(k) : 50.0 * u * u);
        tq.push_back(models::NonlinearCapacitor::charge(jn, tv.back()));
        tc.push_back(models::NonlinearCapacitor::capacitance(jn, tv.back()));
    }
    const std::array<models::NonlinearCapacitor::Params, 3> laws{
        jn, models::NonlinearCapacitor::vdmos_cgd(2e-9, 1e-10, 0.3),
        models::NonlinearCapacitor::from_table(
            std::make_shared<const models::ChargeTable>(tv, tq, tc, "t"))};
    for (std::size_t law = 0; law < laws.size(); ++law) {
        const auto& mp = laws[law];
        INFO("law " << law);
        Lcg rng{11u + law};
        for (int t = 0; t < 100; ++t) {
            pwl::NonlinearCapacitorHistory hist;
            pwl::NonlinearCapacitorHistory::Entry e{};
            e.from = 0;
            e.to = 1;
            e.params = mp;
            e.q_prev = rng(-1e-8, 1e-8);
            e.i_prev = rng(-1.0, 1.0);
            hist.mutable_entries().push_back(e);
            hist.mutable_q_gamma().assign(1, rng(-1e-8, 1e-8));
            Vector x(2);
            // the junction above its floor (v ≥ −0.5 V); VDMOS on both
            // sides of x = −1 (v/A < −1); the table inside and past it
            x[0] = law == 1 ? rng(-20.0, 40.0) : rng(0.0, 40.0);
            x[1] = rng(-0.5, 0.5);
            const Real h = rng(1e-9, 1e-6);
            for (auto st : {pwl::TrBdf2Stage::Trapezoidal,
                            pwl::TrBdf2Stage::BackwardEuler,
                            pwl::TrBdf2Stage::Bdf2Stage2}) {
                sparse::Matrix Jh(2, 2), Ji(2, 2);
                Vector fh = Vector::Zero(2), fi = Vector::Zero(2);
                pwl::refresh_nonlinear_capacitors(x, Jh, fh, hist, h,
                                                  pwl::coss_stage_of(st),
                                                  &hist.q_gamma());
                pwl::iface::stamp(hist, x, Ji, fi, h, st, false);
                // F = a·q + r: terms of size a·|q|. The junction law takes
                // Q from ONE pow, (1+u)·(1+u)^(−m) − 1, the model from
                // (1+u)^(1−m) − 1: both cancel near v = 0 against the
                // C0·V0/(1−m) prefactor, so they agree to ulps of THAT.
                const Real qterm = law == 0 ? mp.C0 * mp.V0 / (1.0 - mp.m) : 0.0;
                const Real a = Real{4} / h;
                const Real qs = a * std::max({qterm, std::abs(e.q_prev),
                    std::abs(hist.q_gamma()[0]),
                    std::abs(models::NonlinearCapacitor::charge(mp, x[0] - x[1]))});
                for (Index r = 0; r < 2; ++r) {
                    CHECK(close(fi[r], fh[r], qs, 16));
                    for (Index c = 0; c < 2; ++c) {
                        CHECK(close(dense(Ji, r, c), dense(Jh, r, c),
                                    dense(Jh, r, c), 16));
                    }
                }
                // commit and stage capture write the same state
                auto hl = hist, hi = hist;
                hl.commit(x, h, st);
                pwl::iface::commit(hi, x, h, st);
                const auto& el = hl.entries()[0];
                const auto& ei = hi.entries()[0];
                CHECK(close(ei.v_prev, el.v_prev, el.v_prev, 4));
                CHECK(close(ei.q_prev, el.q_prev, std::max(qterm, el.q_prev), 4));
                CHECK(close(ei.i_prev, el.i_prev,
                            std::max(qs, std::abs(e.i_prev)), 16));
                hl.capture_gamma(x);
                pwl::iface::capture_gamma(hi, x);
                CHECK(close(hi.q_gamma()[0], hl.q_gamma()[0],
                            std::max(qterm, hl.q_gamma()[0]), 4));
            }
        }
        if (law < 2) {
            check_param_tangent<L::NonlinearCapacitor>(
                L::NonlinearCapacitor::aux_from(mp), L::NonlinearCapacitor::from(mp),
                {law == 0 ? 12.0 : 3.0, 0.0});
            check_param_tangent<L::NonlinearCapacitor>(
                L::NonlinearCapacitor::aux_from(mp), L::NonlinearCapacitor::from(mp),
                {law == 0 ? -0.2 : -9.0, 0.1});
        }
        // the table law has no parameter tangent, and says so
        CHECK(L::NonlinearCapacitor::param_tangent(L::NonlinearCapacitor::aux_from(mp))
              == (law < 2));
    }
}

TEST_CASE("the saturable-inductor law stamps, commits and captures as the "
          "hand stamp", "[device][laws]") {
    models::SaturableInductor::Params atan;
    atan.L_0 = 100e-6;
    atan.I_sat = 6.0;
    atan.L_residual = 1e-6;
    std::vector<Real> ti, tl;
    for (int k = 0; k < 40; ++k) {
        const Real u = k / 39.0;
        ti.push_back(40.0 * u * u);
        tl.push_back(models::SaturableInductor::flux(ti.back(), atan));
    }
    auto table = atan;
    table.table = std::make_shared<const models::FluxTable>(ti, tl, "t");
    auto ja = atan;
    {
        models::JilesAthertonCore::Params c;
        c.N = 100; c.Ae = 1e-4; c.le = 0.1;
        c.ja = models::JilesAthertonParams{.Ms = 4.0e5, .a = 50.0, .alpha = 5e-5,
                                           .c = 0.20, .k = 30.0};
        ja.ja = std::make_shared<const models::JilesAthertonCore::Params>(c);
    }
    const std::array<models::SaturableInductor::Params, 3> laws{atan, table, ja};
    for (std::size_t law = 0; law < laws.size(); ++law) {
        const auto& mp = laws[law];
        INFO("law " << law);
        const Real imax = law == 2 ? 0.2 : 30.0;  // JA: ~ a few a of field
        Lcg rng{21u + law};
        for (int t = 0; t < 60; ++t) {
            pwl::SaturableInductorHistory hist;
            pwl::SaturableInductorHistory::Entry e{};
            e.from = 0;
            e.to = 1;
            e.branch_var_id = 2;
            e.params = mp;
            e.i_L_old = rng(-imax, imax);
            e.i_gamma = e.i_L_old + rng(-0.03, 0.03) * imax;
            if (mp.ja) {
                e.ja_state.H = 0.0;
                e.ja_state.M = 0.0;
                e.ja_state.n_sub = mp.ja->substeps_min;
                e.ja_gamma = e.ja_state;
            }
            e.lambda_old = pwl::SaturableInductorHistory::flux_and_inductance(
                e, e.i_L_old).first;
            e.lambda_gamma = pwl::SaturableInductorHistory::flux_and_inductance(
                e, e.i_gamma, pwl::TrBdf2Stage::Bdf2Stage2).first;
            e.V_L_old = rng(-50.0, 50.0);
            hist.mutable_entries().push_back(e);
            Vector x(3);
            x[0] = rng(-50.0, 50.0);
            x[1] = rng(-1.0, 1.0);
            x[2] = e.i_L_old + rng(-0.06, 0.06) * imax;
            const Real h = rng(1e-8, 1e-5);
            for (bool inc : {false, true}) {
                for (auto st : {pwl::TrBdf2Stage::Trapezoidal,
                                pwl::TrBdf2Stage::BackwardEuler,
                                pwl::TrBdf2Stage::Bdf2Stage2}) {
                    sparse::Matrix Jh(3, 3), Ji(3, 3);
                    Vector fh = Vector::Zero(3), fi = Vector::Zero(3);
                    pwl::stamp_saturable_inductor(e, x, Jh, fh, h, st, inc);
                    pwl::iface::stamp(hist, x, Ji, fi, h, st, inc);
                    // The branch row sums v, the history and (2/h)·λ: its
                    // round-off scales with the largest of them.
                    const Real rs = std::max({std::abs(x[0] - x[1]),
                        std::abs(e.V_L_old),
                        (Real{4} / h) * std::abs(e.lambda_old),
                        (Real{4} / h) * std::abs(e.lambda_gamma)});
                    for (Index r = 0; r < 3; ++r) {
                        CHECK(close(fi[r], fh[r], r == 2 ? rs : fh[r], 16));
                        for (Index c = 0; c < 3; ++c) {
                            CHECK(close(dense(Ji, r, c), dense(Jh, r, c),
                                        dense(Jh, r, c), 16));
                        }
                    }
                    if (inc) continue;
                    auto hl = hist, hi = hist;
                    hl.commit(x, h, st);
                    pwl::iface::commit(hi, x, h, st);
                    const auto& el = hl.entries()[0];
                    const auto& ei = hi.entries()[0];
                    CHECK(ei.i_L_old == el.i_L_old);
                    CHECK(close(ei.V_L_old, el.V_L_old, el.V_L_old, 4));
                    CHECK(close(ei.lambda_old, el.lambda_old, el.lambda_old, 4));
                    CHECK(close(ei.ja_state.H, el.ja_state.H, el.ja_state.H, 4));
                    CHECK(close(ei.ja_state.M, el.ja_state.M, el.ja_state.M, 4));
                    CHECK(ei.ja_state.n_sub == el.ja_state.n_sub);
                    hl.capture_gamma(x);
                    pwl::iface::capture_gamma(hi, x);
                    CHECK(close(hi.entries()[0].lambda_gamma,
                                hl.entries()[0].lambda_gamma,
                                hl.entries()[0].lambda_gamma, 4));
                }
            }
        }
        pwl::SaturableInductorHistory::Entry pe{};
        pe.params = mp;
        CHECK(L::SaturableInductor::param_tangent(
                  pwl::iface::sat_aux(pe, pwl::TrBdf2Stage::Trapezoidal))
              == (law == 0));
    }
    check_param_tangent<L::SaturableInductor>({}, L::SaturableInductor::from(atan),
                                              {3.0, 1.0, 9.5});
    check_param_tangent<L::SaturableInductor>({}, L::SaturableInductor::from(atan),
                                              {-3.0, 1.0, -0.4});
}

TEST_CASE("the static laws' parameter derivatives are AD, not zero",
          "[device][laws]") {
    models::ShockleyDiode::Params sd;
    sd.BV = 5.0;
    check_param_tangent<L::ShockleyDiode>({}, L::ShockleyDiode::from(sd), {0.62, 0.0});
    check_param_tangent<L::ShockleyDiode>({}, L::ShockleyDiode::from(sd), {-5.2, 0.0});
    check_param_tangent<L::IdealDiode>({}, L::IdealDiode::from({}), {0.75, 0.0});
    check_param_tangent<L::IdealDiode>({}, L::IdealDiode::from({}), {-3.0, 0.0});
    check_param_tangent<L::MosfetLevel1>({}, L::MosfetLevel1::from({}), {4.0, 0.0, 3.0});
    check_param_tangent<L::MosfetLevel1>({}, L::MosfetLevel1::from({}), {0.3, 0.0, 6.0});
    check_param_tangent<L::MosfetLevel1>({}, L::MosfetLevel1::from({}), {-0.3, 0.0, 6.0});
    check_param_tangent<L::IgbtLevel1>({}, L::IgbtLevel1::from({}), {3.0, 0.0, 15.0});
    check_param_tangent<L::IgbtLevel1>({}, L::IgbtLevel1::from({}), {1.4, 0.0, 5.2});
    // A sensitivity that is really there is nonzero (not silently 0).
    const auto pa = device::seed_params<L::ShockleyDiode, 6>(L::ShockleyDiode::from(sd));
    device::Out<ad::ADRealN<6>, 2> o;
    L::ShockleyDiode::eval(pa, std::array<Real, 2>{0.62, 0.0}, o);
    CHECK(o.f[0].deriv(0) > 0.0);  // ∂i/∂I_S
    CHECK(o.f[0].deriv(1) < 0.0);  // ∂i/∂n
}

TEST_CASE("Params<Real> evaluates the legacy law to rounding",
          "[device][laws]") {
    // The static law IS the model body with another parameter struct of the
    // same member names, so the Real instantiation is the same expression
    // (to 1 ulp, not ==: a compiler may contract it differently per
    // instantiation).
    models::MosfetLevel1::Params mp;
    const auto lp = L::MosfetLevel1::from(mp);
    for (Real vds : {-3.0, 0.0, 0.4, 7.0}) {
        const std::array<Real, 3> v{vds, 0.0, 5.0};
        device::Out<Real, 3> o;
        L::MosfetLevel1::eval(lp, v, o);
        const Real ref = models::MosfetLevel1::current<Real>(v.data(), mp);
        CHECK(close(o.f[0], ref, Real{1}, 1));
    }
}

TEST_CASE("below its floor the junction Coss keeps C(floor) on both paths",
          "[device][laws]") {
    // DECISIONS #26 / #29: Q is flat below u = v_floor, and both Jacobians
    // stay a·C(floor) (a zero row would be a singular Newton matrix).
    const auto jn = models::NonlinearCapacitor::junction(1e-9, 0.8, 0.45);
    pwl::NonlinearCapacitorHistory hist;
    pwl::NonlinearCapacitorHistory::Entry e{};
    e.from = 0;
    e.to = 1;
    e.params = jn;
    hist.mutable_entries().push_back(e);
    Vector x(2);
    x << -3.0, 0.0;   // u = −3.75 < −0.9
    const Real h = 1e-7;
    sparse::Matrix Jh(2, 2), Ji(2, 2);
    Vector fh = Vector::Zero(2), fi = Vector::Zero(2);
    pwl::refresh_nonlinear_capacitors(x, Jh, fh, hist, h);
    pwl::iface::stamp(hist, x, Ji, fi, h, pwl::TrBdf2Stage::Trapezoidal, false);
    const Real g = Real{2} / h * models::NonlinearCapacitor::capacitance(jn, x[0]);
    REQUIRE(g > Real{0});
    CHECK(close(dense(Ji, 0, 0), g, g, 4));
    CHECK(close(dense(Jh, 0, 0), g, g, 4));
    CHECK(close(dense(Ji, 0, 1), -g, g, 4));
}

namespace {
// A stored-charge kind (one local row) against its hand stamp: F, J, the
// commit and the stage capture, every stage, random states.
template <class H, class Hand, class Mk, class Tc>
void check_local_kind(Mk&& make, Hand&& hand, Tc&& time_const, std::uint64_t seed, Real vlo,
                      Real vhi) {
    Lcg rng{seed};
    for (int t = 0; t < 100; ++t) {
        H hist;
        hist.mutable_entries().push_back(make(rng));
        const std::size_t n = 3;
        Vector x(n);
        for (std::size_t k = 0; k < n; ++k) x[k] = rng(vlo, vhi);
        const Real h = rng(1e-9, 1e-6);
        for (auto st : {pwl::TrBdf2Stage::Trapezoidal, pwl::TrBdf2Stage::BackwardEuler,
                        pwl::TrBdf2Stage::Bdf2Stage2}) {
            INFO("trial " << t << " stage " << static_cast<int>(st));
            auto hl = hist, hi = hist;
            sparse::Matrix Jh(n, n), Ji(n, n);
            Vector fh = Vector::Zero(n), fi = Vector::Zero(n);
            hand(x, Jh, fh, hl, h, st);
            pwl::iface::stamp(hi, x, Ji, fi, h, st, false);
            // Both paths solve q = K0 + K1·g(v): terms of the size of the
            // carried charges and of h·q̇ (K0's), and F = q/τ − … cancels
            // against them, so ulps are of THOSE terms, not of the result.
            const auto& e0 = hist.entries()[0];
            const Real qterm = std::max({std::abs(pwl::iface::q_of(e0)),
                                         std::abs(pwl::iface::qg_of(e0)),
                                         h * std::abs(e0.f_prev), Real{1e-300}});
            const Real tc = time_const(e0);
            Real fs = qterm / tc, js = 0;
            for (Index r = 0; r < Index(n); ++r) {
                fs = std::max(fs, std::abs(fh[r]));
                for (Index c = 0; c < Index(n); ++c) js = std::max(js, std::abs(dense(Jh, r, c)));
            }
            for (Index r = 0; r < Index(n); ++r) {
                CHECK(close(fi[r], fh[r], fs, 64));
                for (Index c = 0; c < Index(n); ++c) {
                    CHECK(close(dense(Ji, r, c), dense(Jh, r, c), js, 64));
                }
            }
            // the K0/K1 the history's restart helpers read stay current
            CHECK(hi.entries()[0].k0 == hl.entries()[0].k0);
            CHECK(hi.entries()[0].k1 == hl.entries()[0].k1);
            // The engine's order: the stage point is captured after a
            // trapezoidal (stage-1) solve, and the BDF2 stage is restamped
            // with it before its commit; never capture between a BDF2 stamp
            // and its commit (the hand commit reads the K0 of the stamp).
            if (st == pwl::TrBdf2Stage::Trapezoidal) {
                hl.capture_gamma(x);
                pwl::iface::capture_gamma(hi, x);
            }
            hl.commit(x, h, st);
            pwl::iface::commit(hi, x, h, st);
            const auto& el = hl.entries()[0];
            const auto& ei = hi.entries()[0];
            const Real qs = std::max({qterm, std::abs(pwl::iface::q_of(el)),
                                      std::abs(pwl::iface::qg_of(el)), fs * tc});
            CHECK(close(pwl::iface::qg_of(ei), pwl::iface::qg_of(el), qs, 64));
            CHECK(close(pwl::iface::q_of(ei), pwl::iface::q_of(el), qs, 64));
            // q̇: the hand BDF2 commit takes the formula value, the law −f;
            // they differ by the solved local row's residual (rounding of
            // terms of size a·q, and of q/τ in the rate itself).
            CHECK(close(ei.f_prev, el.f_prev, std::max({std::abs(el.f_prev), qs / h, qs / tc}), 64));
        }
    }
}
}  // namespace

TEST_CASE("the Lauritzen law (local q_M row) stamps, commits and captures as the "
          "hand stamp", "[device][laws]") {
    check_local_kind<pwl::LauritzenDiodeHistory>(
        [](Lcg& rng) {
            pwl::LauritzenDiodeHistory::Entry e{};
            e.from = 0;
            e.to = 1;
            e.params.tau = rng(1e-8, 1e-6);
            e.params.T_M = e.params.tau * rng(0.02, 0.3);
            e.q_m_prev = rng(-1e-7, 1e-7);
            e.f_prev = rng(-1.0, 1.0);
            e.q_m_gamma = rng(-1e-7, 1e-7);
            return e;
        },
        [](const Vector& x, sparse::Matrix& J, Vector& f, auto& hst, Real h,
           pwl::TrBdf2Stage st) { pwl::refresh_lauritzen_diodes(x, J, f, hst, h, st); },
        [](const auto& e) { return e.params.T_M; }, 21, -0.3, 0.45);
    const L::LauritzenDiode::Params<Real> lp =
        L::LauritzenDiode::from(models::LauritzenDiode::Params{});
    check_param_tangent<L::LauritzenDiode>({}, lp, {0.62, 0.0, 3e-8});
    check_param_tangent<L::LauritzenDiode>({}, lp, {-2.0, 0.1, -1e-8});
}

TEST_CASE("the IGBT tail law (local Q row) stamps, commits and captures as the "
          "hand stamp", "[device][laws]") {
    check_local_kind<pwl::IgbtTailHistory>(
        [](Lcg& rng) {
            pwl::IgbtTailHistory::Entry e{};
            e.collector = 0;
            e.emitter = 1;
            e.gate = 2;
            e.params.tau_tail = rng(1e-7, 1e-5);
            e.params.k_tail = rng(0.1, 0.9);
            e.q_prev = rng(-1e-5, 1e-5);
            e.f_prev = rng(-10.0, 10.0);
            e.q_gamma = rng(-1e-5, 1e-5);
            return e;
        },
        [](const Vector& x, sparse::Matrix& J, Vector& f, auto& hst, Real h,
           pwl::TrBdf2Stage st) { pwl::refresh_igbt_tails(x, J, f, hst, h, st); },
        [](const auto& e) { return e.params.tau_tail; }, 22, -1.0, 15.0);
    models::IgbtLevel1::Params ip{};
    ip.tau_tail = 1e-6;
    ip.k_tail = 0.4;
    const auto tp = L::IgbtTail::from(ip);
    check_param_tangent<L::IgbtTail>({}, tp, {8.0, 0.5, 12.0, 2e-6});
    check_param_tangent<L::IgbtTail>({}, tp, {3.0, 0.0, 5.2, -1e-7});
}

TEST_CASE("a sensitivity of a law without a parameter tangent is refused by name",
          "[device][laws]") {
    std::vector<Real> tv{0.0, 1.0, 2.0}, tq{0.0, 1e-9, 1.8e-9}, tc{1e-9, 0.9e-9, 0.7e-9};
    const auto table = models::NonlinearCapacitor::from_table(
        std::make_shared<const models::ChargeTable>(tv, tq, tc, "t"));
    const auto aux_tab = L::NonlinearCapacitor::aux_from(table);
    const auto aux_j = L::NonlinearCapacitor::aux_from(
        models::NonlinearCapacitor::junction(1e-9, 0.8, 0.45));
    CHECK_FALSE(L::has_param_tangent<L::NonlinearCapacitor>(aux_tab));
    CHECK(L::has_param_tangent<L::NonlinearCapacitor>(aux_j));
    CHECK(L::has_param_tangent<L::ShockleyDiode>({}));
    CHECK(L::has_param_tangent<L::LauritzenDiode>({}));
    try {
        L::require_param_tangent<L::NonlinearCapacitor>(aux_tab, "Coss1", 0);
        FAIL("the table law reported a parameter derivative");
    } catch (const std::invalid_argument& e) {
        const std::string m = e.what();
        CHECK(m.find("Coss1.C0") != std::string::npos);
    }
    CHECK_NOTHROW(L::require_param_tangent<L::NonlinearCapacitor>(aux_j, "Coss2", 2));
    CHECK_THROWS_AS(L::require_param_tangent<L::NonlinearCapacitor>(aux_j, "Coss2", 6),
                    std::invalid_argument);
    CHECK(L::IgbtTail::kParamNames[5] == "tau_tail");
}

TEST_CASE("step 9: pin currents, power and outputs read from the laws",
          "[device][laws]") {
    pwl::StatefulHistories hs;
    pwl::NonlinearCapacitorHistory::Entry c{};
    c.from = 0; c.to = -1; c.params = models::NonlinearCapacitor::junction(1e-9, 0.8, 0.45);
    c.i_prev = 0.25;
    hs.get<pwl::NonlinearCapacitorHistory>().mutable_entries().push_back(c);
    pwl::SaturableInductorHistory::Entry s{};
    s.from = 0; s.to = 1; s.branch_var_id = 2;
    hs.get<pwl::SaturableInductorHistory>().mutable_entries().push_back(s);
    pwl::LauritzenDiodeHistory::Entry d{};
    d.from = 1; d.to = -1; d.q_m_prev = 2e-8;
    hs.get<pwl::LauritzenDiodeHistory>().mutable_entries().push_back(d);
    Vector x(3);
    x << 10.0, 0.6, 1.5;   // v(0), v(1), i_L
    int seen = 0;
    pwl::for_each_stateful_reading(hs, x, [&](const pwl::DeviceReading& r) {
        ++seen;
        if (r.kind == "nonlinear_capacitor") {
            CHECK(r.i_pin[0] == 0.25);
            CHECK(close(r.power, 10.0 * 0.25, 2.5, 4));
        } else if (r.kind == "saturable_inductor") {
            CHECK(r.i_pin[0] == 1.5);
            CHECK(r.i_pin[1] == -1.5);
            CHECK(close(r.power, (10.0 - 0.6) * 1.5, 14.1, 4));
            CHECK(close(r.f[0], 9.4, 9.4, 4));   // the branch voltage
            const Real lam = models::SaturableInductor::flux(1.5, s.params);
            CHECK(close(r.q[0], -lam, lam, 4));
        } else if (r.kind == "lauritzen_diode") {
            const auto& p = d.params;
            const Real qe = models::LauritzenDiode::junction_charge<Real>(0.6, p);
            const Real i = (qe - 2e-8) / p.T_M + 0.6 * p.G_min;
            CHECK(close(r.i_pin[0], i, i, 8));
            CHECK(r.q[0] == 2e-8);
        }
    });
    CHECK(seen == 3);
}

namespace {
// The model body instantiated with the model's own `Params` (the legacy
// path's) and with the law's `Params<Real>` (the interface path's): the
// same expression over the same scalar type, so the results must be the
// same double. Each is called out of line through a volatile function
// pointer so no call site is constant-folded or contracted differently
// from the other (that, not the parameter struct, would be the only way
// for two instantiations of one expression to round apart).
template <class Model, class P>
Real current_of(const Real* v, const P& p) { return Model::template current<Real>(v, p); }

template <class Model, class Law, std::size_t NT>
void check_params_real_bit_identical(const typename Model::Params& mp, Real lo, Real hi,
                                     std::uint64_t seed) {
    using LP = typename Law::template Params<Real>;
    Real (*volatile f_model)(const Real*, const typename Model::Params&) =
        &current_of<Model, typename Model::Params>;
    Real (*volatile f_law)(const Real*, const LP&) = &current_of<Model, LP>;
    const LP lp = Law::from(mp);
    Lcg rng{seed};
    for (int t = 0; t < 500; ++t) {
        std::array<Real, NT> v{};
        for (auto& e : v) e = rng(lo, hi);
        const Real a = f_model(v.data(), mp);
        const Real b = f_law(v.data(), lp);
        INFO("trial " << t << " model " << a << " law " << b);
        CHECK(std::memcmp(&a, &b, sizeof(Real)) == 0);
        // The AD instantiation's value is the same law to rounding.
        const auto pa = device::seed_params<Law, Law::n_params>(lp);
        device::Out<ad::ADRealN<Law::n_params>, NT> o;
        Law::eval(pa, v, o);
        CHECK(close(o.f[0].value(), a, a, 4));
    }
}
}  // namespace

TEST_CASE("Params<Real> is bit-identical to the model's own Params (legacy body)",
          "[device][laws]") {
    models::ShockleyDiode::Params sd;
    sd.BV = 5.0;
    check_params_real_bit_identical<models::ShockleyDiode, L::ShockleyDiode, 2>(sd, -6.0, 1.0, 31);
    check_params_real_bit_identical<models::IdealDiode, L::IdealDiode, 2>({}, -3.0, 1.5, 32);
    check_params_real_bit_identical<models::MosfetLevel1, L::MosfetLevel1, 3>({}, -8.0, 12.0, 33);
    check_params_real_bit_identical<models::IgbtLevel1, L::IgbtLevel1, 3>({}, -2.0, 16.0, 34);
}
