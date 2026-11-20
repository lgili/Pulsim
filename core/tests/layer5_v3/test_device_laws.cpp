// add-unified-device-interface, dual path (DECISIONS #23): the ported
// kinds' laws (device/laws.hpp) against their hand stamps, and their
// parameter derivatives against central differences.
//
// No exact float compares between two different expressions: the hand
// Jacobian (C(v), L(i)) and the law's AD derivative are the same function
// evaluated two ways, equal to a few ulp.

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/device/laws.hpp"
#include "pulsim/device/machine_laws.hpp"
#include "pulsim/models/charge_table.hpp"
#include "pulsim/models/device_model.hpp"
#include "pulsim/models/flux_table.hpp"
#include "pulsim/models/jiles_atherton.hpp"
#include "pulsim/pwl/device_readings.hpp"
#include "pulsim/pwl/device_refresh.hpp"
#include "pulsim/pwl/nonlinear_refresh_pmsm_mna.hpp"
#include "pulsim/sparse/matrix.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
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
    check_static<L::MosfetLevel1, models::MosfetLevel1>({}, -8.0, 12.0, 3);
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
    // law 4: a permeance at μ_r ≈ 2000 driven to 40·V0 — the tangent must
    // be the model's sech², not AD's 1 − tanh² (reviewer, 2026-11-18)
    const std::array<models::NonlinearCapacitor::Params, 5> laws{
        jn, models::NonlinearCapacitor::vdmos_cgd(2e-9, 1e-10, 0.3),
        models::NonlinearCapacitor::from_table(
            std::make_shared<const models::ChargeTable>(tv, tq, tc, "t")),
        models::NonlinearCapacitor::tanh_law(2e-6, 1e-7, 5.0),
        models::NonlinearCapacitor::tanh_law(2e-6, 1e-9, 1.0)};
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
            x[0] = law == 1 || law >= 3 ? rng(-20.0, 40.0) : rng(0.0, 40.0);
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
        if (law != 2) {
            check_param_tangent<L::NonlinearCapacitor>(
                L::NonlinearCapacitor::aux_from(mp), L::NonlinearCapacitor::from(mp),
                {law == 0 ? 12.0 : law >= 3 ? 4.0 : 3.0, 0.0});
            check_param_tangent<L::NonlinearCapacitor>(
                L::NonlinearCapacitor::aux_from(mp), L::NonlinearCapacitor::from(mp),
                {law == 0 ? -0.2 : -9.0, 0.1});
        }
        // the table law has no parameter tangent, and says so
        CHECK(L::NonlinearCapacitor::param_tangent(L::NonlinearCapacitor::aux_from(mp))
              == (law != 2));
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
                    // 7.4: conductance form — the flux row's diagonal is
                    // −a·L(i) (a = 2/h, 1/h, c1/h), never h-scaled.
                    const Real a = device::formula_a(pwl::iface::ctx_of(h, st));
                    const Real Li = pwl::SaturableInductorHistory::flux_and_inductance(
                        e, x[2], st).second;
                    CHECK(close(dense(Ji, 2, 2), -a * Li, a * Li, 16));
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
        CHECK(close(o.f[0], ref, Real{1}, 8));
    }
}

TEST_CASE("the junction charge is relative to Q near 0 V on both paths (F-S-5)",
          "[device][laws]") {
    // pow(1+u, 1-m) - 1 held Q only to C0·V0·eps absolute: at 1 nF, 25 V a
    // femtosecond companion turned that into a 1e-8 A Newton residual floor.
    for (Real m : {0.0, 0.5}) {
        const auto jn = models::NonlinearCapacitor::junction(1e-9, 25.0, m);
        const auto lp = L::NonlinearCapacitor::from(jn);
        const auto la = L::NonlinearCapacitor::aux_from(jn);
        for (Real v : {1e-15, -3e-12, 1e-9, 2e-6, 0.7}) {
            // Taylor about 0: Q = C0·v·(1 − m·u/2 + m(m+1)·u²/6), u = v/V0
            const Real u = v / 25.0;
            const Real ref = 1e-9 * v * (1 - m * u / 2 + m * (m + 1) * u * u / 6);
            const Real tol = std::abs(ref) * (v < 1e-3 ? 1e-14 : 1e-5);
            CHECK(std::abs(models::NonlinearCapacitor::charge(jn, v) - ref) <= tol);
            CHECK(std::abs(L::NonlinearCapacitor::charge<Real>(la, lp, v) - ref) <= tol);
        }
        // m = 0 is the linear law C0·v to a few ulp
        if (m == 0.0) {
            CHECK(std::abs(models::NonlinearCapacitor::charge(jn, 3e-13) - 3e-22) <= 3e-22 * 1e-15);
        }
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

// ---- the MOSFET law: C¹ through V_DS = 0 (DECISIONS #39) -------------------
namespace {

/// The raw terminal-swap mirror of the forward law (the model before #44):
/// it jumps by 2·f(a, 0) at V_DS = 0.
template <class S>
S raw_mirror(const std::array<S, 3>& v, const models::MosfetLevel1::Params& p) {
    const S vds = v[0] - v[1];
    const S vov = v[2] - v[1] - p.V_T;
    if (vds < Real{0}) return -models::MosfetLevel1::forward_current_<S>(vov - vds, -vds, p);
    return models::MosfetLevel1::forward_current_<S>(vov, vds, p);
}

/// The raw forward law's current at V_DS = 0⁺ (the leak) times e^{−κ|V_DS|},
/// mirrored: raw mirror − law, to rounding.
template <class S>
S mosfet_leak(const std::array<S, 3>& v, const models::MosfetLevel1::Params& p) {
    using std::exp;
    const S vds = v[0] - v[1];
    const S vov = v[2] - v[1] - p.V_T;
    if (vds < Real{0})
        return -models::MosfetLevel1::forward_current_<S>(vov - vds, S{0}, p) * exp(S(p.kappa * vds));
    return models::MosfetLevel1::forward_current_<S>(vov, S{0}, p) * exp(S(-(p.kappa * vds)));
}

/// i and ∂i/∂(v_d, v_s, v_g) of the interface law at (V_DS, V_GS).
struct Iv { Real i; std::array<Real, 3> g; };
Iv law_at(const models::MosfetLevel1::Params& mp, Real vds, Real vgs) {
    const std::array<Real, 3> x{vds, 0.0, vgs};
    std::array<Real, 3> F{};
    std::array<std::array<Real, 3>, 3> J{};
    constexpr std::array<Real, 3> z{};
    device::stamp_fq<L::MosfetLevel1>(device::aux_t<L::MosfetLevel1>{}, L::MosfetLevel1::from(mp),
                                     x, {Real{0}, device::Formula::Dc, Real{0}}, z, z, z, F, J);
    return {F[0], J[0]};
}

models::MosfetLevel1::Params mosfet(Real K, Real V_T, Real kappa) {
    models::MosfetLevel1::Params p;
    p.K = K;
    p.V_T = V_T;
    p.kappa = kappa;
    return p;
}

// default; the equivalence boost's device; a large device
const std::array<models::MosfetLevel1::Params, 3> kMosfets{
    models::MosfetLevel1::Params{}, mosfet(0.5, 3.0, 4.0), mosfet(50.0, 3.0, 15.0)};

}  // namespace

TEST_CASE("the MOSFET law is the raw mirror minus its leak at V_DS = 0",
          "[device][laws][mosfet]") {
    // Value and Jacobian against the model's AD minus the leak's AD, over
    // the whole range (threshold band included). The tolerance is a
    // relative 1e-9 of the terms' scale (two expressions through exp;
    // the leak itself is up to 1.9e-3·K).
    std::uint64_t seed = 61;
    for (const auto& mp : kMosfets) {
        Lcg rng{seed++};
        for (int t = 0; t < 400; ++t) {
            const Real vds = rng(-6.0, 6.0), vgs = rng(-1.0, 9.0);
            using A = ad::ADRealN<3>;
            const std::array<A, 3> xa{A{vds, {1.0, 0.0, 0.0}}, A{0.0, {0.0, 1.0, 0.0}},
                                      A{vgs, {0.0, 0.0, 1.0}}};
            const A ref = raw_mirror<A>(xa, mp) - mosfet_leak<A>(xa, mp);
            const auto [i, g] = law_at(mp, vds, vgs);
            const Real scale = mp.K * (1.0 + vds * vds + vgs * vgs);
            INFO("vds " << vds << " vgs " << vgs << " law " << i << " ref " << ref.value());
            CHECK(std::abs(i - ref.value()) <= 1e-9 * scale);
            for (std::size_t k = 0; k < 3; ++k)
                CHECK(std::abs(g[k] - ref.deriv(k)) <= 1e-9 * scale);
        }
    }
}

TEST_CASE("the MOSFET law is C1 through V_DS = 0 and the raw mirror is not",
          "[device][laws][mosfet]") {
    for (const auto& mp : kMosfets) {
        // overdrives across the threshold band, in units of 1/κ
        for (Real xk : {-12.0, -4.0, -2.0, -1.0, -0.3, 0.0, 0.5, 1.0, 2.4, 4.0, 12.0, 60.0}) {
            const Real vgs = mp.V_T + xk / mp.kappa;
            const auto z0 = law_at(mp, 0.0, vgs);
            CHECK(z0.i == 0.0);   // no current at zero voltage
            const Real gscale = std::abs(z0.g[0]) + mp.K / mp.kappa;
            // |i(±e) ∓ g0·e| and |∂i(+e) − ∂i(−e)| shrink with e: as e², e
            // (a bound on i'' of 50·K·(1+κ)², generous), to rounding.
            const Real c2 = 50.0 * mp.K * (1.0 + mp.kappa) * (1.0 + mp.kappa);
            std::array<Real, 3> dg_prev{};
            for (Real e : {1e-5, 1e-8, 1e-11}) {
                const auto p = law_at(mp, e, vgs);
                const auto m = law_at(mp, -e, vgs);
                INFO("K " << mp.K << " kappa " << mp.kappa << " xk " << xk << " e " << e);
                CHECK(std::abs(p.i - z0.g[0] * e) <= c2 * e * e + 1e-13 * gscale * e);
                CHECK(std::abs(m.i + z0.g[0] * e) <= c2 * e * e + 1e-13 * gscale * e);
                for (std::size_t k = 0; k < 3; ++k) {
                    const Real dg = std::abs(p.g[k] - m.g[k]);
                    CHECK(dg <= 2.0 * c2 * e + 1e-12 * gscale);
                    // continuous: the jump falls with e (1000× per step here)
                    if (e < 1e-5) CHECK(dg <= 2e-3 * dg_prev[k] + 1e-12 * gscale);
                    dg_prev[k] = dg;
                }
            }
        }
        // Control: the raw mirror jumps by 2·f(a, 0) at a = 2.4/κ, the leak's
        // peak 0.439·K/κ² (DECISIONS #39) — the probe sees a discontinuity.
        const Real vgs = mp.V_T + 2.4 / mp.kappa;
        const std::array<Real, 3> vp{1e-12, 0.0, vgs}, vm{-1e-12, 0.0, vgs};
        const Real jump = raw_mirror<Real>(vp, mp) - raw_mirror<Real>(vm, mp);
        CHECK(jump > 2.0 * 0.43 * mp.K / (mp.kappa * mp.kappa));
    }
}

TEST_CASE("the MOSFET law is monotone in V_DS around zero, to 1 pA",
          "[device][laws][mosfet]") {
    for (const auto& mp : kMosfets) {
        // on, near-on and off; NOT the band a ∈ (≈ −0.75/κ, 0) (below)
        for (Real xk : {0.5, 2.4, 8.0, 60.0, -4.0, -8.0}) {
            const Real vgs = mp.V_T + xk / mp.kappa;
            // a symmetric grid: log-spaced from 1e-13 V, then linear to
            // ±0.5/κ (beyond, at small |a|, lies the band's mirror image:
            // V_DS past V_OV, or V_GD − V_T in the band on the reverse side)
            std::vector<Real> vds;
            for (int k = -13; k <= -4; ++k) vds.push_back(std::pow(10.0, k));
            for (int k = 1; k <= 20; ++k) vds.push_back(k * 0.025 / mp.kappa);
            std::vector<Real> grid{0.0};
            for (Real d : vds) { grid.push_back(d); grid.push_back(-d); }
            std::sort(grid.begin(), grid.end());
            grid.erase(std::unique(grid.begin(), grid.end()), grid.end());
            Real prev = -std::numeric_limits<Real>::infinity();
            for (Real d : grid) {
                const auto s = law_at(mp, d, vgs);
                INFO("K " << mp.K << " kappa " << mp.kappa << " xk " << xk << " vds " << d
                          << " i " << s.i << " g " << s.g[0]);
                CHECK(s.g[0] > 0.0);                  // positive slope
                CHECK(s.i > prev);                    // strictly increasing
                CHECK((d == 0.0 ? s.i == 0.0 : s.i * d > 0.0));   // passive here
                prev = s.i;
            }
        }
        // The band: the slope at V_DS = 0 is negative (the triode polynomial
        // at negative overdrive; release list, #44). On the law it is
        // Kσa·[κ(1−σ²)a + 2σ] (λ aside), negative for a ∈ (≈ −0.75/κ, 0) —
        // the subtracted leak raises it, so at a = −1/κ it is +0.105·K/κ —
        // and on the raw forward law (≈ −1.3/κ, 0) (control).
        const auto s = law_at(mp, 0.0, mp.V_T - 0.5 / mp.kappa);
        CHECK(s.g[0] < 0.0);
        CHECK(law_at(mp, 0.0, mp.V_T - 1.0 / mp.kappa).g[0] > 0.0);
        const Real vgs = mp.V_T - 1.0 / mp.kappa;
        const Real h = 1e-6 / mp.kappa;
        const std::array<Real, 3> vp{h, 0.0, vgs}, v0{2.0 * h, 0.0, vgs};
        CHECK(raw_mirror<Real>(v0, mp) < raw_mirror<Real>(vp, mp));
    }
}

// add-permeance-magnetic-domain 4.2: the excess-loss law F = c·u/(u²+ε²)^¼.
TEST_CASE("the excess-loss law: both paths, slope, monotone to 1e-12, regularisation",
          "[device][laws][excess]") {
    const models::ExcessLoss::Params mp{.c = 0.01, .eps = 0.01};
    check_static<L::ExcessLoss, models::ExcessLoss>(mp, -12.0, 12.0, 41);
    check_static<L::ExcessLoss, models::ExcessLoss>(mp, -0.05, 0.05, 42);
    check_param_tangent<L::ExcessLoss>({}, L::ExcessLoss::from(mp), {0.3, -0.2});
    check_param_tangent<L::ExcessLoss>({}, L::ExcessLoss::from(mp), {-7.0, 0.0});
    const auto at = [&](Real u) {
        const std::array<Real, 2> v{u, 0.0};
        return models::ExcessLoss::current<Real>(v.data(), mp);
    };
    // dF/du (the AD stamp's) against the closed form and a central
    // difference, from inside the core to 10^3·ε (1e-7 of the slope)
    for (Real u : {0.0, 1e-9, 3e-4, 0.004, 0.01, 0.03, 0.2, 1.0, 10.0, -0.02, -9.0}) {
        models::ModelInputs<models::ExcessLoss> vin{u, 0.0};
        const auto [i, g] = models::evaluate_current_and_jacobian<models::ExcessLoss>(vin, mp);
        const Real d = 1e-6 * std::max(std::abs(u), mp.eps);
        const Real cd = (at(u + d) - at(u - d)) / (2.0 * d);
        INFO("u " << u << " AD " << g[0] << " closed " << models::ExcessLoss::slope(mp, u)
                  << " CD " << cd);
        CHECK(close(g[0], models::ExcessLoss::slope(mp, u), g[0], 16));
        CHECK(std::abs(g[0] - cd) <= 1e-7 * g[0]);
        CHECK(g[0] > 0.0);
        CHECK(close(i, at(u), i, 0));
    }
    // c/√ε at u = 0: the regularised core (the AC sweep's small-signal model)
    CHECK(close(models::ExcessLoss::slope(mp, 0.0), mp.c / std::sqrt(mp.eps), 1.0, 4));
    // strictly increasing at 1e-12 spacing around 0, the knee ε and the rated
    // amplitude 10 V (spacing 1e-12 of the scale), both signs; odd
    for (Real centre : {0.0, 0.01, 10.0, -0.01, -10.0}) {
        const Real step = 1e-12 * std::max(std::abs(centre), Real{1});
        Real prev = at(centre - 2001 * step);
        for (int k = -2000; k <= 2000; ++k) {
            const Real u = centre + k * step;
            const Real f = at(u);
            INFO("centre " << centre << " k " << k);
            REQUIRE(f > prev);
            REQUIRE(at(-u) == -f);
            prev = f;
        }
    }
    // the square-root law outside the core: −(1/4)(ε/u)² relative
    for (Real u : {0.1, 1.0, 10.0}) {
        const Real ideal = mp.c * std::sqrt(u);
        const Real rel = (at(u) - ideal) / ideal;
        CHECK(std::abs(rel + 0.25 * (mp.eps / u) * (mp.eps / u)) <= 0.2 * std::pow(mp.eps / u, 4));
    }
    // both device paths stamp the same numbers (one body; 1e-12 gate)
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "a", "gnd", 1.0);
    b.add_excess_loss("X", "a", "gnd", 0.01, 0.01);
    const auto& g = b.graph();
    const auto& pool = b.pool();
    Vector x = Vector::Zero(g.num_nodes() + g.num_branches());
    const Index e = b.node_id_of("X.e");
    for (Real u : {0.0, 2e-3, 0.7, -5.0}) {
        x.setZero();
        x(e) = u;
        const Index n = x.size();
        sparse::Matrix J1(n, n), J2(n, n);
        J1.insert(e, e) = 0.0; J2.insert(e, e) = 0.0;
        Vector f1 = Vector::Zero(n), f2 = Vector::Zero(n);
        pwl::make_combined_diode_mosfet_refresh()(x, J1, f1, g, pool);
        pwl::make_interface_static_refresh()(x, J2, f2, g, pool);
        INFO("u " << u);
        CHECK(std::abs(f1(e) - f2(e)) <= 1e-12 * std::max(std::abs(f1(e)), Real{1e-300}));
        CHECK(std::abs(dense(J1, e, e) - dense(J2, e, e)) <= 1e-12 * dense(J1, e, e));
        CHECK(close(f1(e), at(u), at(u), 0));
    }
}

TEST_CASE("a static reading describes the law both paths stamp",
          "[device][laws][mosfet]") {
    // V_GS at the leak's peak (a = 2.4/κ), V_DS = 0.1 mV: where the raw
    // mirror and the law differ by f(a, 0)·e^{−κd} ≈ 0.439·K/κ² (#44).
    builder::CircuitBuilder b;
    const Real K = 0.5, V_T = 3.0, lam = 0.02, kappa = 4.0;
    b.add_mosfet_level1("M", "d", "gnd", "g", K, V_T, lam, kappa);
    b.add_shockley_diode("D", "a", "gnd", 1e-12, 1.2, 0.025852);
    Vector x = Vector::Zero(b.graph().num_nodes() + b.graph().num_branches());
    const Real vds = 1e-4, vgs = V_T + 2.4 / kappa, va = 0.62;
    x[b.node_id_of("d")] = vds;
    x[b.node_id_of("g")] = vgs;
    x[b.node_id_of("a")] = va;
    models::MosfetLevel1::Params mp{};
    mp.K = K; mp.V_T = V_T; mp.lambda = lam; mp.kappa = kappa;
    const std::array<Real, 3> v{vds, 0.0, vgs};
    device::Out<Real, 3> lo;
    device::eval<L::MosfetLevel1>({}, L::MosfetLevel1::from(mp), v, lo);
    Real i_m = 0, i_d = 0;
    int n = 0;
    pwl::for_each_static_reading(b.graph(), b.pool(), x, [&](const pwl::DeviceReading& r) {
        ++n;
        if (r.kind == "MosfetLevel1") i_m = r.i_pin[0];
        if (r.kind == "ShockleyDiode") i_d = r.i_pin[0];
    });
    CHECK(n == 2);
    CHECK(i_m == lo.f[0]);
    CHECK(i_m == models::MosfetLevel1::current<Real>(v.data(), mp));   // one body
    CHECK(raw_mirror<Real>(v, mp) - i_m > 0.4 * K / (kappa * kappa));  // control: the leak
    CHECK(i_d > 0.0);
}

// ---- the MNA machines (15.6, DECISIONS #42) --------------------------------
namespace {

// One flux-map cell over [-40, 40]²: W' = ψ i_d + ½L_d i_d² + ½L_q i_q² plus
// a softening, cross-coupling cubic (test_pmsm_flux_map.cpp's `one_cell`).
std::shared_ptr<models::FluxMapDQ> sat_cell() {
    auto f = std::make_shared<models::FluxMapDQ>();
    f->kd = {-40, 40};
    f->kq = {-40, 40};
    f->source = "synthetic: test";
    Real a[4][4] = {};
    a[1][0] = 0.05; a[2][0] = 0.5e-3; a[0][2] = 1.5e-3;
    a[3][0] = -0.5e-6; a[2][1] = 0.75e-6; a[1][2] = -0.25e-6; a[0][3] = -1e-6;
    a[1][1] = 2e-4;
    const Real x0 = -40, y0 = -40;
    const Real bn[4][4] = {{1, 0, 0, 0}, {1, 1, 0, 0}, {1, 2, 1, 0}, {1, 3, 3, 1}};
    f->coef.assign(16, 0);
    for (int m = 0; m < 4; ++m)
        for (int n = 0; n < 4; ++n)
            for (int q = m; q < 4; ++q)
                for (int r = n; r < 4; ++r)
                    f->coef[4 * m + n] += a[q][r] * bn[q][m] * bn[r][n]
                                          * std::pow(x0, q - m) * std::pow(y0, r - n);
    f->validate();
    return f;
}

// Machine 0: salient PMSM with PM harmonics; 1: flux map; 2: induction.
pwl::PmsmMnaHistory::Entry machine_entry(int kind, Lcg& rng) {
    pwl::PmsmMnaHistory::Entry e{};
    e.phase_node = {0, 1, 2};
    e.neutral = 3;
    e.omega_node = 4;
    if (kind == 2) {
        e.induction = true;
        e.rotor_node = {5, 6};
        e.cur_row = {7, 8, 9};
        e.im = {.R_s = 0.5, .L_ls = 2e-3, .R_r = 0.4, .L_lr = 2e-3, .L_m = 60e-3,
                .pole_pairs = 2, .T_load = 3.0};
        e.im_sm = models::InductionMachineMna::stator(e.im);
        e.c_rotor = {1.0, 2.5};
        e.lr_old = {rng(-0.5, 0.5), rng(-0.5, 0.5)};
    } else {
        e.theta_node = 5;
        e.cur_row = {6, 7, 8};
        e.params = {.R_s = 0.5, .L_d = 1e-3, .L_q = 3e-3, .psi_pm = 0.05,
                    .pole_pairs = 3, .T_load = 0.7};
        if (kind == 0) {
            e.params.harmonics = std::make_shared<const std::vector<
                models::PmsmMna::PmHarmonic>>(std::vector<models::PmsmMna::PmHarmonic>{
                {1, 0.05, 0}, {5, 0.004, 0.3}});
        } else {
            e.params.flux_map = sat_cell();
        }
        e.c_theta = 1.7;
        e.theta_old = rng(-3.0, 3.0);
    }
    const Real imax = kind == 1 ? 15.0 : 30.0;
    for (int k = 0; k < 3; ++k) {
        e.i_old[k] = rng(-imax, imax);
        e.dlam_old[k] = rng(-200.0, 200.0);
        e.dlam_gamma[k] = rng(-1e-3, 1e-3);
    }
    e.lambda_old = pwl::PmsmMnaHistory::lambda_of(e, e.i_old, e.theta_old, e.lr_old);
    return e;
}

}  // namespace

TEST_CASE("the machine laws stamp, commit and capture as the hand stamp",
          "[device][laws][machines]") {
    for (int kind = 0; kind < 3; ++kind) {
        INFO("machine " << kind);
        Lcg rng{77u + static_cast<std::uint64_t>(kind)};
        const Index n = kind == 2 ? 10 : 9;
        for (int t = 0; t < 40; ++t) {
            pwl::PmsmMnaHistory hist;
            auto e = machine_entry(kind, rng);
            if (t % 2 == 1) {   // an SDIRK offset between restart and commit
                for (int k = 0; k < 3; ++k) e.lam_off[k] = rng(-1e-4, 1e-4);
            }
            if (t % 4 == 3) e.neutral = -1;   // a grounded star point
            hist.mutable_entries().push_back(e);
            Vector x(n);
            for (Index r = 0; r < 4; ++r) x[r] = rng(-300.0, 300.0);
            x[4] = rng(-200.0, 200.0);
            if (kind == 2) {
                x[5] = e.lr_old[0] + rng(-0.01, 0.01);
                x[6] = e.lr_old[1] + rng(-0.01, 0.01);
            } else {
                x[5] = e.theta_old + rng(-0.01, 0.01);
            }
            for (int k = 0; k < 3; ++k) {
                x[e.cur_row[k]] = e.i_old[k] + rng(-0.5, 0.5);
            }
            const Real h = rng(1e-8, 1e-5);
            for (auto st : {pwl::TrBdf2Stage::Trapezoidal, pwl::TrBdf2Stage::BackwardEuler,
                            pwl::TrBdf2Stage::Bdf2Stage2}) {
                INFO("stage " << static_cast<int>(st));
                sparse::Matrix Jh(n, n), Ji(n, n);
                Vector fh = Vector::Zero(n), fi = Vector::Zero(n);
                const Real mh = pwl::refresh_pmsm_mna(x, Jh, fh, hist, h, st);
                const Real mi = pwl::iface::stamp(hist, x, Ji, fi, h, st, true);
                CHECK(mi == mh);
                // A flux row sums v, R·i, dλ/dt_n and a·Δλ: its round-off
                // scales with the largest.
                Real rs = 0;
                for (Index r = 0; r < n; ++r) rs = std::max(rs, std::abs(fh[r]));
                for (int k = 0; k < 3; ++k) rs = std::max(rs, std::abs(e.dlam_old[k]));
                rs = std::max(rs, Real{600});
                for (Index r = 0; r < n; ++r) {
                    CHECK(close(fi[r], fh[r], rs, 32));
                    for (Index c = 0; c < n; ++c) {
                        INFO("J " << r << "," << c);
                        CHECK(close(dense(Ji, r, c), dense(Jh, r, c), dense(Jh, r, c), 16));
                    }
                }
                // Conductance form (7.4): the stator block is −R·δ − a·∂λ/∂i,
                // a = 2/h, 1/h, c1/h, never h-scaled.
                const Real a = device::formula_a(pwl::iface::ctx_of(h, st));
                const Real R_s = kind == 2 ? e.im.R_s : e.params.R_s;
                const Real eps_i = 1e-6;
                for (int j = 0; j < 3; ++j) {
                    auto ip = e.i_old, im = e.i_old;
                    ip[j] += eps_i;
                    im[j] -= eps_i;
                    const auto lp = pwl::PmsmMnaHistory::lambda_of(e, ip, e.theta_old, e.lr_old);
                    const auto lm = pwl::PmsmMnaHistory::lambda_of(e, im, e.theta_old, e.lr_old);
                    for (int k = 0; k < 3; ++k) {
                        // at x = the reference point the law's ∂λ/∂i is the
                        // model's; compare with a central difference
                        Vector x0 = x;
                        for (int q = 0; q < 3; ++q) x0[e.cur_row[q]] = e.i_old[q];
                        if (kind == 2) { x0[5] = e.lr_old[0]; x0[6] = e.lr_old[1]; }
                        else x0[5] = e.theta_old;
                        sparse::Matrix J0(n, n);
                        Vector f0 = Vector::Zero(n);
                        (void)pwl::iface::stamp(hist, x0, J0, f0, h, st, true);
                        const Real dldi = (lp[k] - lm[k]) / (2 * eps_i);
                        const Real want = -a * dldi - (j == k ? R_s : Real{0});
                        CHECK(std::abs(dense(J0, e.cur_row[k], e.cur_row[j]) - want)
                              <= 1e-6 * std::max(std::abs(want), a * 1e-3));
                    }
                }
                auto hl = hist, hi = hist;
                hl.capture_gamma(x);
                pwl::iface::capture_gamma(hi, x);
                for (int k = 0; k < 3; ++k) {
                    // iface::capture_gamma IS the history's capture_gamma (one
                    // function): exact equality cannot depend on FMA contraction.
                    CHECK(hi.entries()[0].dlam_gamma[k] == hl.entries()[0].dlam_gamma[k]);
                }
                hl.commit(x, h, st);
                pwl::iface::commit(hi, x, h, st);
                const auto& el = hl.entries()[0];
                const auto& ei = hi.entries()[0];
                for (int k = 0; k < 3; ++k) {
                    // λ and v − R·i come from two code paths (the law's eval vs the
                    // hand commit): a compiler may contract one of them to an FMA,
                    // so a few ulp, not bit equality (bitwise on macOS clang).
                    // λ sums L·i terms (+ M·λ_r, λ_pm) that can cancel: scale by
                    // the terms (L ≤ 62 mH in machine_entry → 0.1 H·max|i|).
                    const Real i_max = std::max({std::abs(el.i_old[0]), std::abs(el.i_old[1]),
                                                 std::abs(el.i_old[2])});
                    const Real lam_scale = std::max({std::abs(el.lambda_old[0]),
                                                     std::abs(el.lambda_old[1]),
                                                     std::abs(el.lambda_old[2]),
                                                     Real{0.1} * i_max, Real{1}});
                    CHECK(close(ei.lambda_old[k], el.lambda_old[k], lam_scale, 8));
                    CHECK(ei.i_old[k] == el.i_old[k]);   // copies of x (set_reference)
                    CHECK(ei.v_old[k] == el.v_old[k]);   // one subtraction, no contraction
                    const Real ri = std::abs((kind == 2 ? e.im.R_s : e.params.R_s) * ei.i_old[k]);
                    CHECK(close(ei.dlam_old[k], el.dlam_old[k],
                                std::max(std::abs(el.v_old[k]), ri), 8));
                    CHECK(ei.lam_off[k] == Real{0});
                }
                CHECK(ei.theta_old == el.theta_old);
                CHECK(ei.lr_old == el.lr_old);
            }
        }
    }
    // The law's whole x-Jacobian (∂λ/∂θ with harmonics and the flux map,
    // ∂T, the rotor rows) against central differences of its Real eval —
    // independent of the hand stamp, whose partials chain_n restates.
    {
        Lcg r2{9u};
        for (int kind = 0; kind < 3; ++kind) {
            INFO("FD machine " << kind);
            const auto e = machine_entry(kind, r2);
            auto check = [&](auto law, const auto& aux, const auto& prm) {
                using D = decltype(law);
                constexpr std::size_t R = device::rows_v<D>;
                std::array<Real, R> x{};
                for (std::size_t r = 0; r < R; ++r) x[r] = r2(-50.0, 50.0);
                for (std::size_t k = 0; k < 3; ++k) x[D::n_pins + k] = r2(-10.0, 10.0);
                if (kind == 2) { x[5] = r2(-0.5, 0.5); x[6] = r2(-0.5, 0.5); }
                else x[5] = r2(-3.0, 3.0);
                std::array<ad::ADRealN<R>, R> xs{};
                for (std::size_t k = 0; k < R; ++k) {
                    std::array<Real, R> d{};
                    d[k] = 1;
                    xs[k] = ad::ADRealN<R>{x[k], d};
                }
                device::Out<ad::ADRealN<R>, R> o;
                device::eval<D>(aux, prm, xs, o);
                for (std::size_t k = 0; k < R; ++k) {
                    const Real hk = 1e-6 * std::max(Real{1}, std::abs(x[k]));
                    auto xp = x, xm = x;
                    xp[k] += hk;
                    xm[k] -= hk;
                    device::Out<Real, R> op, om;
                    device::eval<D>(aux, prm, xp, op);
                    device::eval<D>(aux, prm, xm, om);
                    for (std::size_t r = 0; r < R; ++r) {
                        INFO("row " << r << " col " << k);
                        const Real cf = (op.f[r] - om.f[r]) / (2 * hk);
                        const Real cq = (op.q[r] - om.q[r]) / (2 * hk);
                        CHECK(std::abs(o.f[r].deriv(k) - cf)
                              <= 1e-5 * std::max(Real{1}, std::abs(cf)));
                        CHECK(std::abs(o.q[r].deriv(k) - cq)
                              <= 1e-5 * std::max(Real{1e-3}, std::abs(cq)));
                    }
                }
            };
            if (kind == 2) {
                check(L::InductionMachine{}, pwl::iface::im_aux(e), L::InductionMachine::from(e.im));
            } else {
                check(L::PmsmMna{}, pwl::iface::pmsm_aux(e), L::PmsmMna::from(e.params));
            }
        }
    }
    // The induction machine's hand Jacobian (`eval_jac`, DECISIONS #48) and
    // commit rows (`eval_internal`) against its AD eval: the same values.
    {
        using IM = L::InductionMachine;
        Lcg r3{11u};
        const auto e = machine_entry(2, r3);
        const auto aux = pwl::iface::im_aux(e);
        const auto prm = IM::from(e.im);
        for (int trial = 0; trial < 20; ++trial) {
            std::array<Real, 10> x{};
            for (auto& v : x) v = r3(-50.0, 50.0);
            x[5] = r3(-0.5, 0.5);
            x[6] = r3(-0.5, 0.5);
            std::array<ad::ADRealN<10>, 10> xs{};
            for (std::size_t k = 0; k < 10; ++k) {
                std::array<Real, 10> d{};
                d[k] = 1;
                xs[k] = ad::ADRealN<10>{x[k], d};
            }
            device::Out<ad::ADRealN<10>, 10> o;
            device::eval<IM>(aux, prm, xs, o);
            device::Out<Real, 10> oi;
            IM::eval_internal(aux, prm, x, oi);
            for (const Real a : {2.0 / 1e-6, 1.0 / 1e-6, 1.7 / 1e-6}) {
                device::Out<Real, 10> oh;
                std::array<std::array<Real, 10>, 10> J{};
                IM::eval_jac(aux, prm, x, a, oh, J);
                for (std::size_t r = 0; r < 10; ++r) {
                    INFO("trial " << trial << " row " << r << " a " << a);
                    const auto close = [](Real u, Real v) {
                        return std::abs(u - v) <= 1e-14 * std::max(Real{1}, std::abs(v));
                    };
                    CHECK(close(oh.f[r], o.f[r].value()));
                    CHECK(close(oh.q[r], o.q[r].value()));
                    if (r >= IM::n_pins) {
                        CHECK(close(oi.f[r], o.f[r].value()));
                        CHECK(close(oi.q[r], o.q[r].value()));
                    }
                    for (std::size_t k = 0; k < 10; ++k) {
                        INFO("col " << k);
                        CHECK(close(J[r][k], o.f[r].deriv(k) + a * o.q[r].deriv(k)));
                    }
                }
            }
        }
    }
    // The saturable inductor's hand Jacobian (atan law, DECISIONS #50).
    {
        using SL = L::SaturableInductor;
        const SL::Params<Real> sp{1e-3, 2.0, 1e-5};
        for (const Real i : {-7.0, -2.0, -0.3, 0.0, 0.4, 1.9, 6.5}) {
            const std::array<Real, 3> x{1.5, -0.25, i};
            std::array<ad::ADRealN<3>, 3> xs{};
            for (std::size_t k = 0; k < 3; ++k) {
                std::array<Real, 3> d{};
                d[k] = 1;
                xs[k] = ad::ADRealN<3>{x[k], d};
            }
            device::Out<ad::ADRealN<3>, 3> o;
            device::eval<SL>(SL::Aux{}, sp, xs, o);
            for (const Real a : {2.0 / 1e-7, 1.0 / 1e-7, 1.7 / 1e-7}) {
                device::Out<Real, 3> oh;
                std::array<std::array<Real, 3>, 3> J{};
                SL::eval_jac(SL::Aux{}, sp, x, a, oh, J);
                for (std::size_t r = 0; r < 3; ++r) {
                    INFO("i " << i << " row " << r << " a " << a);
                    const auto close = [](Real u, Real v) {
                        return std::abs(u - v) <= 1e-15 * std::abs(v);
                    };
                    CHECK(close(oh.f[r], o.f[r].value()));
                    CHECK(close(oh.q[r], o.q[r].value()));
                    for (std::size_t k = 0; k < 3; ++k)
                        CHECK(close(J[r][k], o.f[r].deriv(k) + a * o.q[r].deriv(k)));
                }
            }
        }
    }
    // ∂/∂(R_s, T_load) by AD against central differences
    Lcg rng{5u};
    for (int kind = 0; kind < 3; ++kind) {
        const auto e = machine_entry(kind, rng);
        if (kind == 2) {
            check_param_tangent<L::InductionMachine>(
                pwl::iface::im_aux(e), L::InductionMachine::from(e.im),
                {100, -50, -40, 3, 150, 0.4, -0.2, 12, -5, -7});
        } else {
            check_param_tangent<L::PmsmMna>(pwl::iface::pmsm_aux(e),
                                            L::PmsmMna::from(e.params),
                                            {100, -50, -40, 3, 150, 0.8, 12, -5, -7});
        }
    }
}
