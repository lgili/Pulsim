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
#include "pulsim/pwl/device_refresh.hpp"
#include "pulsim/sparse/matrix.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
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
                // F = a·q + r: terms of size a·|q|.
                const Real a = Real{4} / h;
                const Real qs = a * std::max({std::abs(e.q_prev),
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
                CHECK(close(ei.q_prev, el.q_prev, el.q_prev, 4));
                CHECK(close(ei.i_prev, el.i_prev,
                            std::max(qs, std::abs(e.i_prev)), 16));
                hl.capture_gamma(x);
                pwl::iface::capture_gamma(hi, x);
                CHECK(close(hi.q_gamma()[0], hl.q_gamma()[0], hl.q_gamma()[0], 4));
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
