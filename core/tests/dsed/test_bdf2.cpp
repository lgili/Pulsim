// =============================================================================
// PED Gate 4 — BDF2 integrator + stiffness detector unit tests
// =============================================================================
//
// Mirrors the Python prototype validation
// (`prototype/dsed/run_stiff_rlc_validation.py`):
//
//   Stiff RLC (L=1µH, C=1µF, R=0.1Ω)
//   Eigenvalues ≈ (-1.01e5, -9.9e6)  →  stiffness ratio ≈ 100×
//   DOPRI5 stable h ≤ 0.3 µs
//   BDF2 takes h = 5 µs (16× past stability) at order-2 accuracy
//
// Slow-mode-only IC (x0 = x_ss + 5·v_slow) → BDF2 RMSE ≤ 0.5 % V_in
// matches the Python prototype to 4 significant figures.

#include <Eigen/Eigenvalues>
#include <chrono>
#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pulsim/dsed/bdf2_integrator.hpp"
#include "pulsim/dsed/rk45_dormand_prince.hpp"
#include "pulsim/dsed/stiffness_detector.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

using pulsim::Real;
using pulsim::Vector;
using pulsim::DenseMatrix;
using namespace pulsim::dsed;

namespace {

// -------------------------------------------------------------------------
// Stiff RLC system: A = [[0, 1/L], [-1/C, -1/(R·C)]], b = [0, V_in/(R·C)]
// -------------------------------------------------------------------------

struct StiffRLC {
    Real L = Real{1e-6};
    Real C = Real{1e-6};
    Real R = Real{0.1};
    Real V_in = Real{5.0};

    [[nodiscard]] DenseMatrix A() const {
        DenseMatrix M(2, 2);
        M << Real{0},          Real{1} / L,
             -Real{1} / C,    -Real{1} / (R * C);
        return M;
    }

    [[nodiscard]] Vector b() const {
        Vector v(2);
        v << Real{0}, V_in / (R * C);
        return v;
    }
};

// -------------------------------------------------------------------------
// Helpers: fixed-step driver for DOPRI5 / BDF2
// -------------------------------------------------------------------------

struct FixedRun {
    std::vector<Real> times;
    std::vector<Vector> states;
    std::size_t n_steps = 0;
    Real cpu_seconds = Real{0};
};

FixedRun run_dopri5_fixed(const DenseMatrix& A, const Vector& b,
                              const Vector& x0, Real t_end, Real h) {
    auto f = [&A, &b](Real /*t*/, const Vector& x) -> Vector {
        return A * x + b;
    };
    const auto n_steps = static_cast<std::size_t>(std::ceil(t_end / h));
    FixedRun r;
    r.times.reserve(n_steps + 1);
    r.states.reserve(n_steps + 1);
    r.times.push_back(Real{0});
    r.states.push_back(x0);
    Vector x = x0;
    RK45State state;
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (std::size_t k = 0; k < n_steps; ++k) {
        const Real t_cur = static_cast<Real>(k) * h;
        auto [x_new, err] = step(f, t_cur, x, h, state);
        x = std::move(x_new);
        r.times.push_back(static_cast<Real>(k + 1) * h);
        r.states.push_back(x);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    r.n_steps = n_steps;
    r.cpu_seconds = std::chrono::duration<Real>(t1 - t0).count();
    return r;
}

FixedRun run_bdf2_fixed(const DenseMatrix& A, const Vector& b,
                            const Vector& x0, Real t_end, Real h) {
    auto b_fn = [&b](Real /*t*/) -> Vector { return b; };
    const auto n_steps = static_cast<std::size_t>(std::ceil(t_end / h));
    FixedRun r;
    r.times.reserve(n_steps + 1);
    r.states.reserve(n_steps + 1);
    r.times.push_back(Real{0});
    r.states.push_back(x0);
    Vector x = x0;
    BDF2State state;
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (std::size_t k = 0; k < n_steps; ++k) {
        const Real t_cur = static_cast<Real>(k) * h;
        auto [x_new, err] = bdf2_step(A, b_fn, t_cur, x, h, state);
        x = std::move(x_new);
        r.times.push_back(static_cast<Real>(k + 1) * h);
        r.states.push_back(x);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    r.n_steps = n_steps;
    r.cpu_seconds = std::chrono::duration<Real>(t1 - t0).count();
    return r;
}

[[nodiscard]] Real linear_interp(const std::vector<Real>& ts,
                                    const std::vector<Vector>& xs,
                                    Real t, Eigen::Index component) {
    auto it = std::lower_bound(ts.begin(), ts.end(), t);
    if (it == ts.begin()) return xs.front()(component);
    if (it == ts.end()) return xs.back()(component);
    const auto i = static_cast<std::size_t>(it - ts.begin()) - 1;
    const Real alpha = (t - ts[i]) / (ts[i + 1] - ts[i]);
    return (Real{1} - alpha) * xs[i](component) + alpha * xs[i + 1](component);
}

}  // namespace

// =============================================================================
// Test 1 — Stiffness detector: eigenvalues + selection
// =============================================================================

TEST_CASE("StiffnessDetector selects BDF2 above threshold and DOPRI5 below",
          "[dsed][gate4][stiffness]") {
    StiffRLC rlc;
    DenseMatrix A = rlc.A();
    StiffnessDetector det(Real{10.0});

    const Real lam_max = det.lambda_max(/*mode_id=*/0, A);
    REQUIRE(lam_max > Real{1e6});       // overdamped fast eigenvalue
    REQUIRE(lam_max < Real{2e7});

    // At h = 50 ns: λ·h ≈ 0.5 → DOPRI5
    auto choice_fast = det.select(0, A, Real{50e-9});
    REQUIRE(choice_fast == IntegratorChoice::DOPRI5);

    // At h = 5 µs: λ·h ≈ 50 → BDF2
    auto choice_slow = det.select(0, A, Real{5e-6});
    REQUIRE(choice_slow == IntegratorChoice::BDF2);

    // explain() returns the same answer + the ratio
    auto info = det.explain(0, A, Real{5e-6});
    REQUIRE(info.stiff);
    REQUIRE(info.lambda_h == Catch::Approx(lam_max * Real{5e-6}));
}

// =============================================================================
// Test 2 — Stiffness detector caches eigenvalues
// =============================================================================

TEST_CASE("StiffnessDetector caches eigenvalues per mode_id",
          "[dsed][gate4][stiffness]") {
    StiffRLC rlc;
    DenseMatrix A = rlc.A();
    StiffnessDetector det;

    REQUIRE(det.cache_size() == 0);
    (void) det.lambda_max(42, A);
    REQUIRE(det.cache_size() == 1);
    (void) det.lambda_max(42, A);       // re-query same mode_id
    REQUIRE(det.cache_size() == 1);     // still 1 — cached
    (void) det.lambda_max(7, A);
    REQUIRE(det.cache_size() == 2);
    det.clear_cache();
    REQUIRE(det.cache_size() == 0);
}

// =============================================================================
// Test 3 — BDF2 reproduces analytical steady state when starting AT steady state
// =============================================================================

TEST_CASE("BDF2 holds y at analytical steady state if x0 = x_ss",
          "[dsed][gate4][bdf2]") {
    StiffRLC rlc;
    DenseMatrix A = rlc.A();
    Vector b = rlc.b();

    // x_ss = -A^{-1} b
    Vector x_ss = -A.fullPivLu().solve(b);

    auto run = run_bdf2_fixed(A, b, x_ss, Real{50e-6}, Real{5e-6});
    REQUIRE(run.states.size() == 11);   // 10 steps + initial

    // Should stay at steady state to machine precision
    const Vector& x_final = run.states.back();
    REQUIRE(std::abs(x_final(0) - x_ss(0)) < Real{1e-9});
    REQUIRE(std::abs(x_final(1) - x_ss(1)) < Real{1e-9});
}

// =============================================================================
// Test 4 — BDF2 convergence order: halving h → err drops ~4×
// =============================================================================

TEST_CASE("BDF2 is order-2: halving h reduces global error 2-8×",
          "[dsed][gate4][bdf2]") {
    StiffRLC rlc;
    DenseMatrix A = rlc.A();
    Vector b = rlc.b();
    Vector x_ss = -A.fullPivLu().solve(b);

    // Slow-mode IC
    Eigen::EigenSolver<DenseMatrix> es(A);
    auto eigs = es.eigenvalues();
    auto eigvecs = es.eigenvectors();
    Eigen::Index slow_idx = 0;
    Real slow_mag = std::abs(eigs(0));
    for (Eigen::Index i = 1; i < eigs.size(); ++i) {
        if (std::abs(eigs(i)) < slow_mag) {
            slow_mag = std::abs(eigs(i));
            slow_idx = i;
        }
    }
    Vector v_slow(2);
    v_slow << eigvecs(0, slow_idx).real(), eigvecs(1, slow_idx).real();
    v_slow.normalize();
    Vector x0 = x_ss + Real{5} * v_slow;

    const Real t_end = Real{50e-6};
    const Real h_truth = Real{50e-9};   // DOPRI5 ground truth

    auto gt = run_dopri5_fixed(A, b, x0, t_end, h_truth);
    auto b1 = run_bdf2_fixed(A, b, x0, t_end, Real{5e-6});
    auto b2 = run_bdf2_fixed(A, b, x0, t_end, Real{2.5e-6});

    auto rmse_against_gt = [&](const FixedRun& r) -> Real {
        Real ss = Real{0};
        std::size_t n = 0;
        for (std::size_t i = 0; i < gt.times.size(); ++i) {
            const Real v_b = linear_interp(r.times, r.states, gt.times[i], 1);
            const Real d = v_b - gt.states[i](1);
            ss += d * d;
            ++n;
        }
        return std::sqrt(ss / static_cast<Real>(n));
    };

    const Real err_h    = rmse_against_gt(b1);
    const Real err_half = rmse_against_gt(b2);

    INFO("err(h=5µs)   = " << err_h);
    INFO("err(h=2.5µs) = " << err_half);
    INFO("ratio        = " << err_h / err_half);

    // Order-2 method → halving h drops error 4× (allow 2-8× band)
    const Real ratio = err_h / err_half;
    REQUIRE(ratio >= Real{2.0});
    REQUIRE(ratio <= Real{8.0});
}

// =============================================================================
// Test 5 — BDF2 vs DOPRI5-stability-limit: correctness + speedup
// =============================================================================

TEST_CASE("BDF2 at h>>stability limit matches DOPRI5 ground truth ≤0.5% V_in",
          "[dsed][gate4][bdf2][validation]") {
    StiffRLC rlc;
    DenseMatrix A = rlc.A();
    Vector b = rlc.b();
    Vector x_ss = -A.fullPivLu().solve(b);

    // Slow-mode IC (same as Python prototype)
    Eigen::EigenSolver<DenseMatrix> es(A);
    auto eigs = es.eigenvalues();
    auto eigvecs = es.eigenvectors();
    Eigen::Index slow_idx = 0;
    Real slow_mag = std::abs(eigs(0));
    for (Eigen::Index i = 1; i < eigs.size(); ++i) {
        if (std::abs(eigs(i)) < slow_mag) {
            slow_mag = std::abs(eigs(i));
            slow_idx = i;
        }
    }
    Vector v_slow(2);
    v_slow << eigvecs(0, slow_idx).real(), eigvecs(1, slow_idx).real();
    v_slow.normalize();
    Vector x0 = x_ss + Real{5} * v_slow;

    const Real t_end = Real{50e-6};
    auto gt = run_dopri5_fixed(A, b, x0, t_end, Real{50e-9});
    auto dopri = run_dopri5_fixed(A, b, x0, t_end, Real{300e-9});
    auto bdf2 = run_bdf2_fixed(A, b, x0, t_end, Real{5e-6});

    // RMSE on v_C (component 1)
    auto rmse_vs_gt = [&](const FixedRun& r) -> Real {
        Real ss = Real{0};
        std::size_t n = 0;
        for (std::size_t i = 0; i < gt.times.size(); ++i) {
            const Real v = linear_interp(r.times, r.states, gt.times[i], 1);
            const Real d = v - gt.states[i](1);
            ss += d * d;
            ++n;
        }
        return std::sqrt(ss / static_cast<Real>(n));
    };

    const Real rmse_bdf2 = rmse_vs_gt(bdf2);
    const Real rmse_pct = Real{100} * rmse_bdf2 / rlc.V_in;
    INFO("BDF2 RMSE = " << rmse_bdf2 * Real{1000} << " mV ("
         << rmse_pct << " % of V_in)");
    INFO("DOPRI5 wall = " << dopri.cpu_seconds * Real{1000} << " ms ("
         << dopri.n_steps << " steps)");
    INFO("BDF2 wall   = " << bdf2.cpu_seconds * Real{1000} << " ms ("
         << bdf2.n_steps << " steps)");
    INFO("Speedup     = " << dopri.cpu_seconds / bdf2.cpu_seconds);

    // Gate 4-correctness: BDF2 RMSE ≤ 0.5 % V_in
    REQUIRE(rmse_pct <= Real{0.5});

    // Gate 4-speedup: BDF2 faster than DOPRI5 at stability limit
    REQUIRE(bdf2.cpu_seconds < dopri.cpu_seconds);
}

// =============================================================================
// Test 6 — BDF2State invalidate() resets history + LU
// =============================================================================

TEST_CASE("BDF2State invalidate clears history + LU cache",
          "[dsed][gate4][bdf2]") {
    StiffRLC rlc;
    DenseMatrix A = rlc.A();
    Vector b = rlc.b();
    auto b_fn = [&b](Real /*t*/) -> Vector { return b; };

    Vector x0 = Vector::Zero(2);
    BDF2State state;
    REQUIRE_FALSE(state.has_history());

    auto [x1, _err1] = bdf2_step(A, b_fn, Real{0}, x0, Real{1e-6}, state);
    REQUIRE(state.has_history());
    REQUIRE(state.lu.has_value());

    state.invalidate();
    REQUIRE_FALSE(state.has_history());
    REQUIRE_FALSE(state.lu.has_value());
    REQUIRE(state.h_cached == Real{0});
}
