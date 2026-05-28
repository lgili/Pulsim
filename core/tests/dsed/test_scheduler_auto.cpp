// =============================================================================
// PED Gate 4.D — Auto-dispatch scheduler (RK45↔BDF2 per mode)
// =============================================================================
//
// Validates that PEDSimulatorAuto:
//   1. Picks BDF2 for stiff modes and DOPRI5 for non-stiff modes
//      based solely on the cached |λ_max|·h product per mode.
//   2. Correctly invalidates both integrator states at every mask
//      change so the new mode starts with the right integrator and
//      no stale FSAL/BDF2 history.
//   3. Records per-segment integrator choice in the event log so
//      downstream consumers can audit the dispatch decisions.
//   4. Tracks a high-resolution DOPRI5 ground truth within tolerance.
//
// Scenario: 2-mode mixed-stiffness LC system switched at 10 kHz.
//   Mode A (stiff)     : R = 0.1 Ω  → |λ_max| ≈ 1e7
//   Mode B (non-stiff) : R = 10  Ω  → |λ_max| ≈ 1e5 (LC ringing only)
//
// At dt_max = 5 µs:
//   Mode A: |λ_max|·h = 50  → BDF2 (over threshold 10)
//   Mode B: |λ_max|·h = 0.5 → DOPRI5

#include <Eigen/Eigenvalues>
#include <chrono>
#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pulsim/dsed/scheduler_auto.hpp"
#include "pulsim/dsed/rk45_dormand_prince.hpp"
#include "pulsim/dsed/stiffness_detector.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

using pulsim::Real;
using pulsim::Vector;
using pulsim::DenseMatrix;
using namespace pulsim::dsed;

namespace {

// -----------------------------------------------------------------------------
// 2-mode mixed-stiffness RLC.
//   Mode A (mask=true)  : R = 0.1 Ω, V_in = 5 V    → stiff
//   Mode B (mask=false) : R = 10  Ω, V_in = 0 V    → non-stiff (LC ringing)
// -----------------------------------------------------------------------------

class MixedStiffnessRLC {
public:
    Real L = Real{1e-6};
    Real C = Real{1e-6};
    Real R_stiff = Real{0.1};
    Real R_nonstiff = Real{10.0};
    Real V_in_stiff = Real{5.0};

    [[nodiscard]] bool current_mask() const noexcept { return mask_a_; }
    void set_mask(bool m) noexcept { mask_a_ = m; }

    [[nodiscard]] DenseMatrix A_matrix() const {
        const Real R = mask_a_ ? R_stiff : R_nonstiff;
        DenseMatrix A(2, 2);
        A << Real{0},          Real{1} / L,
             -Real{1} / C,    -Real{1} / (R * C);
        return A;
    }

    [[nodiscard]] Vector b_vector(Real /*t*/) const {
        const Real R = mask_a_ ? R_stiff : R_nonstiff;
        const Real V = mask_a_ ? V_in_stiff : Real{0};
        Vector b(2);
        b << Real{0}, V / (R * C);
        return b;
    }

    [[nodiscard]] Vector rhs(Real t, const Vector& x) const {
        return A_matrix() * x + b_vector(t);
    }

private:
    bool mask_a_ = true;
};

// 50% duty switch at 10 kHz: mask_a (stiff) for first half of each period.
class MixedSwitchFn {
public:
    MixedSwitchFn(Real T_sw, Real D) : T_{T_sw}, D_{D} {}

    [[nodiscard]] bool operator()(Real t) const {
        const Real phase = std::fmod(t / T_, Real{1});
        return phase < D_;
    }

    [[nodiscard]] Real next_edge_after(Real t) const {
        const auto k = static_cast<std::int64_t>(std::floor(t / T_));
        const Real candidates[3] = {
            static_cast<Real>(k) * T_ + D_ * T_,
            static_cast<Real>(k + 1) * T_,
            static_cast<Real>(k + 1) * T_ + D_ * T_,
        };
        constexpr Real eps = Real{1e-15};
        for (Real c : candidates) {
            if (c > t + eps) return c;
        }
        return static_cast<Real>(k + 2) * T_;
    }

private:
    Real T_;
    Real D_;
};

// -----------------------------------------------------------------------------
// DOPRI5 fixed-step reference with mask switching (mirror of test_bdf2_scheduler.cpp)
// -----------------------------------------------------------------------------

struct FixedRun {
    std::vector<Real> times;
    std::vector<Vector> states;
    std::size_t n_steps = 0;
    Real cpu_seconds = Real{0};
};

template <class System, class SwitchFn>
FixedRun run_dopri5_switching(System& sys, SwitchFn sf,
                                  const Vector& x0, Real t_end, Real h) {
    auto f = [&sys](Real tau, const Vector& x) -> Vector {
        return sys.rhs(tau, x);
    };
    const auto n_steps = static_cast<std::size_t>(std::ceil(t_end / h));
    FixedRun r;
    r.times.reserve(n_steps + 1);
    r.states.reserve(n_steps + 1);
    r.times.push_back(Real{0});
    r.states.push_back(x0);
    Vector x = x0;
    RK45State state;
    sys.set_mask(sf(Real{0}));
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (std::size_t k = 0; k < n_steps; ++k) {
        const Real t_cur = static_cast<Real>(k) * h;
        const bool m = sf(t_cur + Real{0.5} * h);
        if (m != sys.current_mask()) {
            sys.set_mask(m);
            state.invalidate();
        }
        auto [x_new, _err] = step(f, t_cur, x, h, state);
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
// Test 1 — Stiffness detector picks the right integrator per mode
// =============================================================================

TEST_CASE("PEDSimulatorAuto: stiffness detector picks BDF2 for stiff mode, "
          "DOPRI5 for non-stiff mode",
          "[dsed][gate4][auto-dispatch]") {
    MixedStiffnessRLC sys;
    PEDSimulatorAuto<MixedStiffnessRLC, MixedSwitchFn> sim(
        sys, MixedSwitchFn{Real{100e-6}, Real{0.5}},
        PIController{}, StiffnessDetector{Real{10.0}},
        Real{1e-9}, Real{5e-6}, Real{1e-6});

    // Mode A (stiff): R = 0.1, |λ_max| ≈ 1e7. At dt_max = 5µs: λ·h = 50 → BDF2
    sys.set_mask(true);
    const auto choice_A = sim.current_integrator_for(1, sys.A_matrix());
    REQUIRE(choice_A == IntegratorChoice::BDF2);

    // Mode B (non-stiff): R = 10, |λ_max| ≈ 1e5-ish. At dt_max = 5µs:
    // λ·h ≈ 0.5 → DOPRI5
    sys.set_mask(false);
    const auto choice_B = sim.current_integrator_for(0, sys.A_matrix());
    REQUIRE(choice_B == IntegratorChoice::DOPRI5);
}

// =============================================================================
// Test 2 — Auto-dispatcher actually switches integrator per mode
// =============================================================================

TEST_CASE("PEDSimulatorAuto event log records integrator-per-segment",
          "[dsed][gate4][auto-dispatch]") {
    const Real T_sw = Real{100e-6};   // 10 kHz
    const Real D = Real{0.5};
    const Real t_end = Real{500e-6};   // 5 cycles → 10 events

    Vector x0(2);
    x0 << Real{0}, Real{0};

    MixedStiffnessRLC sys;
    PEDSimulatorAuto<MixedStiffnessRLC, MixedSwitchFn> sim(
        sys, MixedSwitchFn{T_sw, D},
        PIController{}, StiffnessDetector{Real{10.0}},
        Real{1e-9}, Real{5e-6}, Real{1e-6});

    const auto r = sim.simulate(x0, t_end);

    // Expect roughly 10 gate edges (2 per cycle × 5 cycles).
    REQUIRE(r.n_events >= 8);
    REQUIRE(r.n_events <= 12);

    // Each event records the integrator that was active leading up to it.
    // Sequence should be: BDF2 (mode A → fires HS-off), DOPRI5 (mode B → HS-on),
    // BDF2 (A → HS-off), DOPRI5 (B → HS-on), …
    int n_bdf2_segments = 0;
    int n_rk45_segments = 0;
    for (const auto& ev : r.event_log) {
        if (ev.integrator_used == IntegratorChoice::BDF2) ++n_bdf2_segments;
        if (ev.integrator_used == IntegratorChoice::DOPRI5) ++n_rk45_segments;
    }
    INFO("BDF2 segments: " << n_bdf2_segments
         << ", DOPRI5 segments: " << n_rk45_segments);
    // Each integrator should have been used at least once.
    REQUIRE(n_bdf2_segments >= 1);
    REQUIRE(n_rk45_segments >= 1);

    // Aggregate step counts: both integrators should have contributed.
    INFO("n_rk45_steps: " << r.n_rk45_steps
         << ", n_bdf2_steps: " << r.n_bdf2_steps);
    REQUIRE(r.n_rk45_steps > 0);
    REQUIRE(r.n_bdf2_steps > 0);
}

// =============================================================================
// Test 3 — Auto-dispatcher tracks DOPRI5 ground truth within tolerance
// =============================================================================

TEST_CASE("PEDSimulatorAuto matches DOPRI5 ground truth within envelope",
          "[dsed][gate4][auto-dispatch][validation]") {
    const Real T_sw = Real{100e-6};
    const Real D = Real{0.5};
    const Real t_end = Real{1e-3};       // 10 cycles
    const Real h_truth = Real{50e-9};

    Vector x0(2);
    x0 << Real{0}, Real{0};

    // Auto-dispatcher
    MixedStiffnessRLC sys_auto;
    PEDSimulatorAuto<MixedStiffnessRLC, MixedSwitchFn> sim_auto(
        sys_auto, MixedSwitchFn{T_sw, D},
        PIController{}, StiffnessDetector{Real{10.0}},
        Real{1e-9}, Real{5e-6}, Real{1e-6});
    const auto r_auto = sim_auto.simulate(x0, t_end);

    // DOPRI5 ground truth
    MixedStiffnessRLC sys_gt;
    const auto r_gt = run_dopri5_switching(
        sys_gt, MixedSwitchFn{T_sw, D}, x0, t_end, h_truth);

    // RMSE over the second half of the window (after the inrush settles).
    Real ss = Real{0};
    std::size_t n_samples = 0;
    for (std::size_t i = 0; i < r_gt.times.size(); ++i) {
        if (r_gt.times[i] < Real{500e-6}) continue;
        const Real v = linear_interp(r_auto.times, r_auto.states,
                                          r_gt.times[i], 1);
        const Real d = v - r_gt.states[i](1);
        ss += d * d;
        ++n_samples;
    }
    const Real rmse = std::sqrt(ss / static_cast<Real>(n_samples));
    const Real rmse_pct = Real{100} * rmse / sys_auto.V_in_stiff;
    INFO("Auto RMSE = " << rmse * Real{1000} << " mV ("
         << rmse_pct << " % of V_in)");
    INFO("n_rk45_steps = " << r_auto.n_rk45_steps
         << ", n_bdf2_steps = " << r_auto.n_bdf2_steps);
    INFO("n_events = " << r_auto.n_events);
    INFO("Auto wall = " << r_auto.cpu_time_seconds * Real{1000} << " ms");

    // 15% envelope: the auto-dispatcher inherits the BDF2 fast-mode
    // smoothing artifact at each stiff→non-stiff transition (mode A
    // contributes the artifact). Bound is slightly looser than the
    // BDF2-only scheduler (10%) because BOTH modes excite the LC and
    // both have to bootstrap.
    REQUIRE(rmse_pct <= Real{15.0});
    // No max-step blow up
    REQUIRE(r_auto.n_accept < std::size_t{200'000});
}

// =============================================================================
// Test 4 — Mask-change invalidates BOTH integrator states (no stale FSAL)
// =============================================================================

TEST_CASE("PEDSimulatorAuto invalidates both RK45 and BDF2 state on every "
          "mask change",
          "[dsed][gate4][auto-dispatch]") {
    // We validate via the EFFECT (no NaN propagation, no diverging
    // trajectories) — the actual state is private.
    const Real T_sw = Real{200e-6};   // 5 kHz
    const Real D = Real{0.3};
    const Real t_end = Real{600e-6};   // 3 cycles, 6 events

    Vector x0(2);
    x0 << Real{0}, Real{0};

    MixedStiffnessRLC sys;
    PEDSimulatorAuto<MixedStiffnessRLC, MixedSwitchFn> sim(
        sys, MixedSwitchFn{T_sw, D},
        PIController{}, StiffnessDetector{},
        Real{1e-9}, Real{5e-6}, Real{1e-6});
    const auto r = sim.simulate(x0, t_end);

    // Event log is monotonically increasing
    for (std::size_t i = 1; i < r.event_log.size(); ++i) {
        REQUIRE(r.event_log[i].t >= r.event_log[i - 1].t);
    }
    // All events are gate edges (no other event types in this scenario)
    for (const auto& e : r.event_log) {
        REQUIRE(e.type == PredicateType::GateEdge);
    }
    // No NaN in trajectory
    for (const auto& xs : r.states) {
        REQUIRE(std::isfinite(xs(0)));
        REQUIRE(std::isfinite(xs(1)));
    }
}
