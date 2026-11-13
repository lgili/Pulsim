// =============================================================================
// update-newton-persistent-solver, stage A1 — Newton workspace.
// Bit-identity checks: the in-place Jacobian and the reused analysis must
// give the bytes of the per-iteration fresh path.
// =============================================================================

#include <catch2/catch_test_macros.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/lauritzen_diode_history.hpp"
#include "pulsim/pwl/nonlinear_refresh_diode.hpp"
#include "pulsim/pwl/nonlinear_refresh_lauritzen_diode.hpp"
#include "pulsim/pwl/nonlinear_refresh_saturable_inductor.hpp"
#include "pulsim/pwl/saturable_inductor_history.hpp"
#include "pulsim/pwl/nonlinear_solve.hpp"

#include <cmath>
#include <cstring>
#include <random>

using namespace pulsim;
using namespace pulsim::pwl;
using topology::SwitchStateMask;

namespace {

builder::CircuitBuilder smooth_buck() {
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "p", "gnd", 24.0);
    b.add_switch("S", "p", "sw", 1e3, 1e-9);
    b.add_nonlinear_diode("D", "gnd", "sw", models::IdealDiode::Params{});
    b.add_nonlinear_diode("D2", "o", "q", models::IdealDiode::Params{});
    b.add_resistor("Rq", "q", "gnd", 50.0);
    b.add_inductor("L", "sw", "o", 1e-4);
    b.add_capacitor("C", "o", "gnd", 4.7e-5);
    b.add_resistor("R", "o", "gnd", 10.0);
    return b;
}

bool same_bytes(const sparse::Matrix& A, const sparse::Matrix& B) {
    if (A.nonZeros() != B.nonZeros() || A.rows() != B.rows()) return false;
    const auto nnz = static_cast<std::size_t>(A.nonZeros());
    const auto no = static_cast<std::size_t>(A.outerSize() + 1);
    return std::memcmp(A.outerIndexPtr(), B.outerIndexPtr(), no * sizeof(Index)) == 0 &&
           std::memcmp(A.innerIndexPtr(), B.innerIndexPtr(), nnz * sizeof(Index)) == 0 &&
           std::memcmp(A.valuePtr(), B.valuePtr(), nnz * sizeof(Real)) == 0;
}

bool same_bytes(const Vector& a, const Vector& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(),
                       static_cast<std::size_t>(a.size()) * sizeof(Real)) == 0;
}

}  // namespace

TEST_CASE("workspace Jacobian equals seg.J + J_nl byte for byte",
          "[v2][layer4][newton][workspace]") {
    auto b = smooth_buck();
    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build_lazy(1e-7);
    const auto& seg = cache.lookup(SwitchStateMask(1));
    const auto n = static_cast<Index>(seg.state_size);
    NewtonWorkspace w;
    std::mt19937 gen(7);
    std::uniform_real_distribution<Real> u(-2.0, 30.0);
    for (int k = 0; k < 100; ++k) {
        Vector x(n);
        for (Index i = 0; i < n; ++i) x[i] = u(gen);
        sparse::Matrix J_nl(n, n);
        Vector f_nl = Vector::Zero(n);
        (void)refresh_smooth_diodes(x, J_nl, f_nl, b.graph(), b.pool());
        sparse::compress_in_place(J_nl);
        const bool reused = w.combine(seg.J, J_nl);
        if (k > 0) REQUIRE(reused);
        sparse::Matrix ref = seg.J;
        ref += J_nl;
        ref.makeCompressed();
        REQUIRE(same_bytes(w.J, ref));
    }
    // An empty J_nl changes the pattern: rebuilt, values copied.
    sparse::Matrix empty(n, n);
    REQUIRE_FALSE(w.combine(seg.J, empty));
    REQUIRE(same_bytes(w.J, seg.J));
}

TEST_CASE("a run workspace reuses one analysis and gives the bytes of the fresh path",
          "[v2][layer4][newton][workspace]") {
    auto b = smooth_buck();
    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build_lazy(1e-7);
    const auto n = static_cast<Index>(b.pool().state_size(b.graph()));
    NonlinearRefreshFn rf = [](const Vector& x, sparse::Matrix& J, Vector& f,
                               const topology::Graph& g, const DevicePool& p) {
        return refresh_smooth_diodes(x, J, f, g, p);
    };
    NewtonWorkspace w;
    w.backend = sparse::Backend::Eigen;   // A1 bit-identity: the fresh path's LU
    Vector x_ws = Vector::Zero(n), x_ref = Vector::Zero(n);
    std::uint64_t iters = 0;
    for (int step = 0; step < 200; ++step) {
        SwitchStateMask m(1);
        m.set(0, (step / 10) % 2 == 0);
        const auto& seg = cache.lookup(m);
        const Vector be = Vector::Constant(n, 1e-3 * step);
        x_ref = solve_with_newton_b_extra(seg, rf, b.graph(), b.pool(), x_ref, be);
        x_ws = solve_with_newton_b_extra(seg, rf, b.graph(), b.pool(), x_ws, be,
                                          Size{50}, Real{1e-9}, Real{1e-9},
                                          false, false, &w);
        REQUIRE(same_bytes(x_ws, x_ref));
        iters = w.iterations;
    }
    REQUIRE(w.solves == 200);
    REQUIRE(iters > 200);
    REQUIRE(w.factorizations == w.iterations);
    // Both masks share the dynamic pattern: one analysis for the run.
    REQUIRE(w.analyses == 1);
}

TEST_CASE("stage A2: in-house LU keeps the iteration counts, one analysis, in-place refactors",
          "[v2][layer4][newton][workspace]") {
    auto b = smooth_buck();
    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build_lazy(1e-7);
    const auto n = static_cast<Index>(b.pool().state_size(b.graph()));
    NonlinearRefreshFn rf = [](const Vector& x, sparse::Matrix& J, Vector& f,
                               const topology::Graph& g, const DevicePool& p) {
        return refresh_smooth_diodes(x, J, f, g, p);
    };
    NewtonWorkspace eig, pul;   // default backend: Pulsim
    eig.backend = sparse::Backend::Eigen;
    REQUIRE(pul.backend == sparse::Backend::Pulsim);
    Vector xe = Vector::Zero(n), xp = Vector::Zero(n);
    Real worst = 0;
    for (int step = 0; step < 200; ++step) {
        SwitchStateMask m(1);
        m.set(0, (step / 10) % 2 == 0);
        const auto& seg = cache.lookup(m);
        const Vector be = Vector::Constant(n, 1e-3 * step);
        solve_with_newton_in_place(seg, rf, b.graph(), b.pool(), xe, be,
                                   Size{50}, Real{1e-9}, Real{1e-9}, false, false, &eig);
        solve_with_newton_in_place(seg, rf, b.graph(), b.pool(), xp, be,
                                   Size{50}, Real{1e-9}, Real{1e-9}, false, false, &pul);
        for (Index i = 0; i < n; ++i) {
            worst = std::max(worst, std::abs(xe[i] - xp[i]) / std::max(Real{1}, std::abs(xe[i])));
        }
    }
    INFO("largest relative difference Eigen vs in-house: " << worst);
    REQUIRE(worst < 1e-9);
    REQUIRE(pul.iterations_per_solve == eig.iterations_per_solve);
    REQUIRE(pul.iterations_per_solve.size() == 200);
    REQUIRE(pul.analyses == 1);
    REQUIRE(pul.factorizations == pul.iterations);
    const auto c = pul.take_counters();
    REQUIRE(c.iterations == eig.iterations);
    REQUIRE(c.iterations_per_solve.size() == 200);
}

TEST_CASE("two stateful device kinds: a kept J_nl pattern loses no stamp",
          "[v2][layer4][newton][workspace]") {
    // Task 2.6: base refresh + Lauritzen diode + saturable inductor,
    // composed as run_transient's wrapper chain composes them. A J_nl
    // whose pattern is kept between refreshes must hold the bytes of a
    // freshly built one.
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "p", "gnd", 5.0);
    b.add_nonlinear_diode("D", "p", "a", models::IdealDiode::Params{});
    b.add_lauritzen_diode("DL", "a", "k");
    b.add_resistor("R", "k", "gnd", 10.0);
    b.add_saturable_inductor("Ls", "a", "gnd", 1e-3, 2.0, 1e-5);
    const auto& g = b.graph();
    const auto& pool = b.pool();
    LauritzenDiodeHistory laur;
    laur.init(g, pool);
    SaturableInductorHistory sat;
    sat.init(g, pool);
    REQUIRE_FALSE(laur.empty());
    REQUIRE_FALSE(sat.entries().empty());
    const Real h = 1e-7;
    NonlinearRefreshFn chain = [](const Vector& x, sparse::Matrix& J, Vector& f,
                                  const topology::Graph& gg, const DevicePool& pp) {
        return refresh_smooth_diodes(x, J, f, gg, pp);
    };
    chain = [user = chain, &laur, h](const Vector& x, sparse::Matrix& J, Vector& f,
                                     const topology::Graph& gg, const DevicePool& pp) {
        const Real a = user(x, J, f, gg, pp);
        return std::max(a, refresh_lauritzen_diodes(x, J, f, laur, h));
    };
    chain = [user = chain, &sat, h](const Vector& x, sparse::Matrix& J, Vector& f,
                                    const topology::Graph& gg, const DevicePool& pp) {
        Real m = user(x, J, f, gg, pp);
        for (const auto& e : sat.entries()) {
            m = std::max(m, stamp_saturable_inductor(e, x, J, f, h,
                                                     TrBdf2Stage::Trapezoidal));
        }
        return m;
    };
    const auto n = static_cast<Index>(pool.state_size(g));
    std::mt19937 gen(11);
    std::uniform_real_distribution<Real> u(-1.0, 1.5);
    sparse::Matrix kept(n, n);
    Vector fk = Vector::Zero(n);
    for (int k = 0; k < 50; ++k) {
        Vector x(n);
        for (Index i = 0; i < n; ++i) x[i] = u(gen);
        (void)chain(x, kept, fk, g, pool);
        sparse::compress_in_place(kept);
        sparse::Matrix fresh(n, n);
        Vector ff = Vector::Zero(n);
        (void)chain(x, fresh, ff, g, pool);
        sparse::compress_in_place(fresh);
        REQUIRE(same_bytes(kept, fresh));
        REQUIRE(same_bytes(fk, ff));
    }
    // Every kind stamped: the chain's pattern holds each kind's own.
    sparse::Matrix only_sat(n, n);
    Vector fs = Vector::Zero(n);
    const Vector x = Vector::Constant(n, 0.3);
    for (const auto& e : sat.entries()) {
        (void)stamp_saturable_inductor(e, x, only_sat, fs, h, TrBdf2Stage::Trapezoidal);
    }
    sparse::Matrix only_laur(n, n);
    (void)refresh_lauritzen_diodes(x, only_laur, fs, laur, h);
    for (const sparse::Matrix* part : {&only_sat, &only_laur}) {
        REQUIRE(part->nonZeros() > 0);
        for (Index j = 0; j < n; ++j) {
            for (sparse::Matrix::InnerIterator it(*part, j); it; ++it) {
                bool found = false;
                for (sparse::Matrix::InnerIterator jt(kept, j); jt; ++jt) {
                    found |= jt.row() == it.row();
                }
                REQUIRE(found);
            }
        }
    }
}

TEST_CASE("restart stages share the run workspace's analysis",
          "[v2][layer4][newton][workspace][restart]") {
    // run_transient (discontinuity_restart) solves the SDIRK stages on a
    // segment assembled at 2*gamma*dt (`retry_seg`) with the run's
    // workspace; its pattern must be the nominal segment's, or every
    // restart window would re-analyse twice.
    auto b = smooth_buck();
    const Real dt = 1e-7;
    const Real restart_dt = Real{2} * ((Real{1} - Real{0.70710678118654752440}) * dt);
    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build_lazy(dt);
    const auto n = static_cast<Index>(b.pool().state_size(b.graph()));
    NonlinearRefreshFn rf = [](const Vector& x, sparse::Matrix& J, Vector& f,
                               const topology::Graph& g, const DevicePool& p) {
        return refresh_smooth_diodes(x, J, f, g, p);
    };
    NewtonWorkspace w;
    PwlSegment retry_seg;
    Vector x = Vector::Zero(n);
    for (int step = 0; step < 60; ++step) {
        SwitchStateMask m(1);
        m.set(0, (step / 10) % 2 == 0);
        const Vector be = Vector::Constant(n, 1e-3 * step);
        if (step % 10 < 3) {   // the three restart steps after each edge
            assemble_segment(b.graph(), b.pool(), m, restart_dt,
                             retry_seg.J, retry_seg.b_constant);
            sparse::compress_in_place(retry_seg.J);
            retry_seg.state_size = static_cast<Size>(n);
            for (int stage = 0; stage < 2; ++stage) {
                x = solve_with_newton_b_extra(retry_seg, rf, b.graph(), b.pool(), x, be,
                                              Size{50}, Real{1e-9}, Real{1e-9},
                                              false, false, &w);
            }
        } else {
            x = solve_with_newton_b_extra(cache.lookup(m), rf, b.graph(), b.pool(), x, be,
                                          Size{50}, Real{1e-9}, Real{1e-9},
                                          false, false, &w);
        }
    }
    REQUIRE(w.solves == 6 * 3 * 2 + 6 * 7);
    REQUIRE(w.analyses == 1);
}
