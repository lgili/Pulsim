// =============================================================================
// PED Gate 2 Phase 2.B — C++ end-to-end validation on buck CCM
// =============================================================================
//
// Reproduces in C++ the Gate 1 Python validation captured in
// `notes/GATE1_RESULTS.md` (RMSE 0.0057% vs trap reference,
// wall-clock 0.61× of trap). Same physical buck:
//
//   24V → 12V sync-buck, 100 kHz, D = 0.5
//   L = 100 µH, C = 100 µF, R_load = 2.4 Ω
//   5 ms simulation window
//
// Gate 2A target: RMSE ≤ 0.1 %
// Gate 2B target: wall-clock ≤ 2× trap reference

#include <chrono>
#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pulsim/dsed/scheduler.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

using pulsim::Real;
using pulsim::Vector;
using namespace pulsim::dsed;

namespace {

// -----------------------------------------------------------------------------
// BuckCCMModel — same physical model as prototype/dsed/buck_model.py
// -----------------------------------------------------------------------------

struct BuckParams {
    Real V_in = Real{24};
    Real L = Real{100e-6};
    Real C = Real{100e-6};
    Real R_load = Real{2.4};
    Real f_sw = Real{100e3};
    Real D = Real{0.5};

    [[nodiscard]] Real V_out_steady() const noexcept { return D * V_in; }
    [[nodiscard]] Real I_L_steady() const noexcept { return V_out_steady() / R_load; }
    [[nodiscard]] Real T_sw() const noexcept { return Real{1} / f_sw; }
};

class BuckCCMModel {
public:
    explicit BuckCCMModel(BuckParams p) : p_{p} {
        A_(0, 0) = Real{0};            A_(0, 1) = -Real{1} / p_.L;
        A_(1, 0) = Real{1} / p_.C;     A_(1, 1) = -Real{1} / (p_.R_load * p_.C);
        b_HSon_(0) = p_.V_in / p_.L;   b_HSon_(1) = Real{0};
        b_HSoff_(0) = Real{0};         b_HSoff_(1) = Real{0};
        b_cur_ = b_HSon_;
    }

    [[nodiscard]] bool current_mask() const noexcept { return mask_; }

    void set_mask(bool m) noexcept {
        mask_ = m;
        b_cur_ = m ? b_HSon_ : b_HSoff_;
    }

    [[nodiscard]] Vector rhs(Real /*t*/, const Vector& x) const {
        return A_ * x + b_cur_;
    }

    [[nodiscard]] const Eigen::Matrix2d& A() const noexcept { return A_; }
    [[nodiscard]] const Eigen::Vector2d& b_HSon() const noexcept { return b_HSon_; }

private:
    BuckParams p_;
    Eigen::Matrix2d A_ = Eigen::Matrix2d::Zero();
    Eigen::Vector2d b_HSon_ = Eigen::Vector2d::Zero();
    Eigen::Vector2d b_HSoff_ = Eigen::Vector2d::Zero();
    Eigen::Vector2d b_cur_ = Eigen::Vector2d::Zero();
    bool mask_ = true;  // HS on by default
};

// -----------------------------------------------------------------------------
// BuckPSCSwitchFn — exposes next_edge_after for the scheduler's fast path
// -----------------------------------------------------------------------------

class BuckPSCSwitchFn {
public:
    BuckPSCSwitchFn(Real T_sw, Real D) : T_sw_{T_sw}, D_{D} {}

    [[nodiscard]] bool operator()(Real t) const noexcept {
        const Real phase = std::fmod(t / T_sw_, Real{1});
        return phase < D_;
    }

    [[nodiscard]] Real next_edge_after(Real t) const noexcept {
        const Real k = std::floor(t / T_sw_);
        const Real candidates[3] = {
            k * T_sw_ + D_ * T_sw_,
            (k + Real{1}) * T_sw_,
            (k + Real{1}) * T_sw_ + D_ * T_sw_,
        };
        constexpr Real eps = Real{1e-15};
        for (Real c : candidates) {
            if (c > t + eps) return c;
        }
        return (k + Real{2}) * T_sw_;
    }

private:
    Real T_sw_;
    Real D_;
};

// -----------------------------------------------------------------------------
// Fixed-step trapezoidal reference (v1.4.0 PWL cache emulation)
// -----------------------------------------------------------------------------

struct TrapResult {
    std::vector<Real> times;
    std::vector<Vector> states;
    Real cpu_seconds = Real{0};
};

TrapResult run_trapezoidal_reference(BuckParams p,
                                       BuckPSCSwitchFn switch_fn,
                                       Vector x0,
                                       Real t_end,
                                       Real dt) {
    const int n_steps = static_cast<int>(std::ceil(t_end / dt));

    // Pre-build trap-companion matrices: x_new = Khat·x + dt_M_inv·b
    BuckCCMModel model{p};
    Eigen::Matrix2d I = Eigen::Matrix2d::Identity();
    Eigen::Matrix2d M_inv = (I - Real{0.5} * dt * model.A()).inverse();
    Eigen::Matrix2d Khat = M_inv * (I + Real{0.5} * dt * model.A());
    Eigen::Matrix2d dt_M_inv = dt * M_inv;

    TrapResult r;
    r.times.reserve(n_steps + 1);
    r.states.reserve(n_steps + 1);
    r.times.push_back(Real{0});
    r.states.push_back(x0);
    Vector x = x0;

    Eigen::Vector2d b_on = model.b_HSon();
    Eigen::Vector2d b_off = Eigen::Vector2d::Zero();

    auto t0_wall = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < n_steps; ++k) {
        const Real t_cur = k * dt;
        // Sample switch_fn at midpoint for trapezoidal averaging
        const bool mask = switch_fn(t_cur + Real{0.5} * dt);
        Eigen::Vector2d b = mask ? b_on : b_off;
        x = Khat * x + dt_M_inv * b;
        r.times.push_back((k + 1) * dt);
        r.states.push_back(x);
    }
    auto t1_wall = std::chrono::high_resolution_clock::now();
    r.cpu_seconds = std::chrono::duration<Real>(t1_wall - t0_wall).count();
    return r;
}

// Hermite interpolation onto a target time grid
Vector interpolate_vout(const std::vector<Real>& times,
                          const std::vector<Vector>& states,
                          const Vector& target_times) {
    Vector vout(target_times.size());
    std::size_t idx = 0;
    for (Eigen::Index i = 0; i < target_times.size(); ++i) {
        const Real t = target_times(i);
        // Linear search (sorted; advance idx)
        while (idx + 1 < times.size() && times[idx + 1] < t) ++idx;
        if (idx + 1 >= times.size()) {
            vout(i) = states.back()(1);
            continue;
        }
        const Real t0 = times[idx];
        const Real t1 = times[idx + 1];
        const Real alpha = (t - t0) / (t1 - t0);
        vout(i) = (Real{1} - alpha) * states[idx](1)
                  + alpha * states[idx + 1](1);
    }
    return vout;
}

}  // namespace

// =============================================================================
// TESTS
// =============================================================================

TEST_CASE("Buck CCM in PED matches fixed-step trapezoidal within 0.1% RMSE",
           "[dsed][buck][gate2A]") {
    BuckParams p;
    BuckPSCSwitchFn switch_fn{p.T_sw(), p.D};
    const Real t_end = Real{5e-3};
    Vector x0(2);
    x0 << p.I_L_steady(), p.V_out_steady();

    // Reference: fixed-step trapezoidal at dt=100ns
    const Real dt_ref = Real{100e-9};
    auto ref = run_trapezoidal_reference(p, switch_fn, x0, t_end, dt_ref);

    // PED simulator
    BuckCCMModel model{p};
    PIController controller{Real{1e-6}, Real{1e-9}};
    PEDSimulator sim{
        model, switch_fn, std::move(controller), EventPredictor{},
        Real{1e-9},        // dt_init
        p.T_sw() / Real{4} // dt_max
    };
    auto result = sim.simulate(x0, t_end);

    // Compute RMSE on V_out over the second half (steady state)
    const Real t_start = Real{2.5e-3};
    std::vector<Real> t_window_vec;
    std::vector<Real> vout_ref;
    for (std::size_t i = 0; i < ref.times.size(); ++i) {
        if (ref.times[i] >= t_start) {
            t_window_vec.push_back(ref.times[i]);
            vout_ref.push_back(ref.states[i](1));
        }
    }
    Vector t_window(t_window_vec.size());
    for (std::size_t i = 0; i < t_window_vec.size(); ++i) {
        t_window(i) = t_window_vec[i];
    }

    Vector vout_ped = interpolate_vout(result.times, result.states, t_window);
    Vector vout_ref_v(vout_ref.size());
    for (std::size_t i = 0; i < vout_ref.size(); ++i) {
        vout_ref_v(i) = vout_ref[i];
    }

    Vector diff = vout_ped - vout_ref_v;
    Real rmse = std::sqrt(diff.squaredNorm() / static_cast<Real>(diff.size()));
    Real rmse_rel_pct = Real{100} * rmse / p.V_out_steady();

    INFO("PED steps     : " << result.n_accept);
    INFO("PED rejects   : " << result.n_reject);
    INFO("PED events    : " << result.n_events);
    INFO("PED wall-clock: " << result.cpu_time_seconds * Real{1000} << " ms");
    INFO("Trap steps    : " << ref.times.size() - 1);
    INFO("Trap wall-clk : " << ref.cpu_seconds * Real{1000} << " ms");
    INFO("RMSE abs      : " << rmse * Real{1000} << " mV");
    INFO("RMSE rel      : " << rmse_rel_pct << " %");

    // Gate 2A: RMSE ≤ 0.1 % (Python prototype achieved 0.0057 %)
    REQUIRE(rmse_rel_pct <= Real{0.1});

    // Sanity: PED should converge to ~12.0 V steady state
    const Real mean_ped = vout_ped.mean();
    REQUIRE(mean_ped == Catch::Approx(p.V_out_steady()).margin(Real{0.01}));

    // PED should fire ~999 events (1000 edges over 500 cycles minus
    // one for the period boundary at t = t_end)
    REQUIRE(result.n_events >= 990);
    REQUIRE(result.n_events <= 1010);

    // PI controller should have zero rejections on this smooth scenario
    REQUIRE(result.n_reject == 0);
}

TEST_CASE("Buck CCM PED wall-clock within 2× of fixed-step trapezoidal",
           "[dsed][buck][gate2B]") {
    BuckParams p;
    BuckPSCSwitchFn switch_fn{p.T_sw(), p.D};
    const Real t_end = Real{5e-3};
    Vector x0(2);
    x0 << p.I_L_steady(), p.V_out_steady();

    // Reference
    const Real dt_ref = Real{100e-9};
    auto ref = run_trapezoidal_reference(p, switch_fn, x0, t_end, dt_ref);

    // PED
    BuckCCMModel model{p};
    PIController controller;
    PEDSimulator sim{
        model, switch_fn, std::move(controller), EventPredictor{},
        Real{1e-9},
        p.T_sw() / Real{4}
    };
    auto result = sim.simulate(x0, t_end);

    const Real ratio = result.cpu_time_seconds / ref.cpu_seconds;
    INFO("Wall-clock ratio PED/trap: " << ratio << "×");
    INFO("Step count ratio PED/trap: "
         << (static_cast<Real>(result.n_accept)
             / static_cast<Real>(ref.times.size() - 1)));

    // Gate 2B: wall-clock ≤ 2× trap (Python prototype hit 0.61×, much better)
    REQUIRE(ratio <= Real{2.0});
}

TEST_CASE("Buck CCM PED takes many fewer steps than fixed-step trap",
           "[dsed][buck]") {
    BuckParams p;
    BuckPSCSwitchFn switch_fn{p.T_sw(), p.D};
    const Real t_end = Real{1e-3};  // 1 ms = 100 cycles
    Vector x0(2);
    x0 << p.I_L_steady(), p.V_out_steady();

    BuckCCMModel model{p};
    PEDSimulator sim{
        model, switch_fn, PIController{}, EventPredictor{},
        Real{1e-9}, p.T_sw() / Real{4}
    };
    auto result = sim.simulate(x0, t_end);

    // Trapezoidal at dt=100ns would take 10000 steps; PED should take
    // < 1000 (variable step amortises across smooth intervals)
    REQUIRE(result.n_accept < 1000);
    INFO("PED accepted steps over 1 ms: " << result.n_accept);
    INFO("Step reduction vs trap (dt=100ns): "
         << (Real{10000} / static_cast<Real>(result.n_accept)) << "×");
}
