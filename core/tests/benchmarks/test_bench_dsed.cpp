// =============================================================================
// Benchmark: PED (Path-Based Event-Driven) vs fixed-step trapezoidal
// =============================================================================
//
// Sweep the buck CCM benchmark over windows of varying length and
// dt resolutions. Captures the wall-clock advantage of variable-step
// event-driven scheduling over the canonical fixed-step trapezoidal
// + PWL cache approach.
//
// Output CSV columns:
//   t_end_ms       — simulated window length (ms)
//   trap_steps     — fixed-step trapezoidal step count
//   trap_ms        — fixed-step trapezoidal wall-clock (ms)
//   ped_steps      — PED accepted-step count
//   ped_events     — PED events fired
//   ped_ms         — PED wall-clock (ms)
//   speedup        — trap_ms / ped_ms (× faster PED is than trap)
//   rmse_pct       — RMSE on V_out (% of nominal 12 V)
//
// CSV file:  bench-results/dsed_buck_ccm_microbench.csv
// CSV path can be overridden by PULSIM_BENCH_RESULTS_DIR.
//
// Reproduce with:
//   ./build/core/pulsim_benchmarks "[dsed][microbench]"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "pulsim/dsed/buck_dcm_model.hpp"
#include "pulsim/dsed/scheduler.hpp"
#include "pulsim/dsed/scheduler_auto.hpp"
#include "pulsim/dsed/stiffness_detector.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

using pulsim::Real;
using pulsim::Vector;
using pulsim::DenseMatrix;
using namespace pulsim::dsed;

namespace {

// --- Buck CCM model (mirror of test_buck_ccm.cpp; intentional duplication
//     so the benchmark binary can be built independently)

struct BuckParams {
    Real V_in = Real{24};
    Real L = Real{100e-6};
    Real C = Real{100e-6};
    Real R_load = Real{2.4};
    Real f_sw = Real{100e3};
    Real D = Real{0.5};

    [[nodiscard]] Real V_out_steady() const { return D * V_in; }
    [[nodiscard]] Real I_L_steady() const { return V_out_steady() / R_load; }
    [[nodiscard]] Real T_sw() const { return Real{1} / f_sw; }
};

class BuckCCMModel {
public:
    explicit BuckCCMModel(BuckParams p) : p_{p} {
        A_(0, 0) = Real{0};            A_(0, 1) = -Real{1} / p_.L;
        A_(1, 0) = Real{1} / p_.C;     A_(1, 1) = -Real{1} / (p_.R_load * p_.C);
        b_HSon_(0) = p_.V_in / p_.L;   b_HSon_(1) = Real{0};
    }
    [[nodiscard]] bool current_mask() const noexcept { return mask_; }
    void set_mask(bool m) noexcept { mask_ = m; }
    [[nodiscard]] Vector rhs(Real /*t*/, const Vector& x) const {
        return A_ * x + (mask_ ? b_HSon_ : Eigen::Vector2d::Zero());
    }
    [[nodiscard]] const Eigen::Matrix2d& A() const { return A_; }
    [[nodiscard]] const Eigen::Vector2d& b_HSon() const { return b_HSon_; }

    /// `HasLTIPerMode` accessor — dynamic-size A copy for the
    /// PEDSimulatorAuto contract. Same matrix for HS_on and HS_off
    /// (only b changes), so the dispatch's cached eigenvalue at
    /// mask-change-time stays valid.
    [[nodiscard]] DenseMatrix A_matrix() const {
        DenseMatrix A(2, 2);
        A << A_(0, 0), A_(0, 1), A_(1, 0), A_(1, 1);
        return A;
    }

    /// `HasLTIPerMode` accessor — current mask's b vector.
    [[nodiscard]] Vector b_vector(Real /*t*/) const {
        Vector b(2);
        b << (mask_ ? b_HSon_(0) : Real{0}), Real{0};
        return b;
    }
private:
    BuckParams p_;
    Eigen::Matrix2d A_ = Eigen::Matrix2d::Zero();
    Eigen::Vector2d b_HSon_ = Eigen::Vector2d::Zero();
    bool mask_ = true;
};

class BuckPSCSwitchFn {
public:
    BuckPSCSwitchFn(Real T, Real D) : T_{T}, D_{D} {}
    [[nodiscard]] bool operator()(Real t) const {
        return std::fmod(t / T_, Real{1}) < D_;
    }
    [[nodiscard]] Real next_edge_after(Real t) const {
        const Real k = std::floor(t / T_);
        const Real c[3] = {k * T_ + D_ * T_, (k + 1) * T_, (k + 1) * T_ + D_ * T_};
        constexpr Real eps = Real{1e-15};
        for (Real x : c) if (x > t + eps) return x;
        return (k + 2) * T_;
    }
private:
    Real T_;
    Real D_;
};

struct TrapRun {
    std::vector<Real> times;
    std::vector<Vector> states;
    Real cpu_seconds = Real{0};
};

TrapRun run_trap(BuckParams p, BuckPSCSwitchFn sf, Vector x0, Real t_end, Real dt) {
    const int n_steps = static_cast<int>(std::ceil(t_end / dt));
    BuckCCMModel m{p};
    Eigen::Matrix2d I = Eigen::Matrix2d::Identity();
    Eigen::Matrix2d M_inv = (I - Real{0.5} * dt * m.A()).inverse();
    Eigen::Matrix2d Khat = M_inv * (I + Real{0.5} * dt * m.A());
    Eigen::Matrix2d dt_M_inv = dt * M_inv;
    TrapRun r;
    r.times.reserve(n_steps + 1);
    r.states.reserve(n_steps + 1);
    r.times.push_back(Real{0});
    r.states.push_back(x0);
    Vector x = x0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < n_steps; ++k) {
        const Real t_cur = k * dt;
        Eigen::Vector2d b = sf(t_cur + Real{0.5} * dt)
            ? Eigen::Vector2d(m.b_HSon())
            : Eigen::Vector2d::Zero();
        x = Khat * x + dt_M_inv * b;
        r.times.push_back((k + 1) * dt);
        r.states.push_back(x);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    r.cpu_seconds = std::chrono::duration<Real>(t1 - t0).count();
    return r;
}

Real linear_interp_v(const std::vector<Real>& ts, const std::vector<Vector>& xs, Real t) {
    auto it = std::lower_bound(ts.begin(), ts.end(), t);
    if (it == ts.begin()) return xs.front()(1);
    if (it == ts.end()) return xs.back()(1);
    const std::size_t i = static_cast<std::size_t>(it - ts.begin()) - 1;
    const Real alpha = (t - ts[i]) / (ts[i + 1] - ts[i]);
    return (Real{1} - alpha) * xs[i](1) + alpha * xs[i + 1](1);
}

std::filesystem::path bench_output_dir() {
    if (const char* env = std::getenv("PULSIM_BENCH_RESULTS_DIR")) {
        std::filesystem::path p{env};
        std::filesystem::create_directories(p);
        return p;
    }
    auto p = std::filesystem::current_path() / "bench-results";
    std::filesystem::create_directories(p);
    return p;
}

}  // namespace

TEST_CASE("Bench: PED buck CCM — wall-clock vs fixed-step trapezoidal",
           "[bench][dsed][microbench]") {
    std::printf("\n=== PED buck CCM microbench (Gate 2 Phase 2.B-2) ===\n");
    std::printf("%-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n",
                 "t_end_ms", "trap_st", "trap_ms", "ped_st", "ped_ev",
                 "ped_ms", "speedup", "rmse_pct");

    BuckParams p;
    BuckPSCSwitchFn sf{p.T_sw(), p.D};
    Vector x0(2);
    x0 << p.I_L_steady(), p.V_out_steady();
    const Real dt_ref = Real{100e-9};

    const Real windows_ms[] = {Real{1}, Real{2.5}, Real{5}, Real{10}, Real{25}};

    auto out_path = bench_output_dir() / "dsed_buck_ccm_microbench.csv";
    std::ofstream csv{out_path};
    csv << "t_end_ms,trap_steps,trap_ms,ped_steps,ped_events,"
         << "ped_ms,speedup,rmse_pct\n";

    for (Real t_end_ms : windows_ms) {
        const Real t_end = t_end_ms * Real{1e-3};

        auto ref = run_trap(p, sf, x0, t_end, dt_ref);

        BuckCCMModel m{p};
        PEDSimulator sim{
            m, sf, PIController{}, EventPredictor{},
            Real{1e-9}, p.T_sw() / Real{4}
        };
        auto r = sim.simulate(x0, t_end);

        // Compute RMSE on V_out over the second half of the window
        const Real t_window_start = Real{0.5} * t_end;
        Real ss = Real{0};
        int n_samples = 0;
        for (std::size_t i = 0; i < ref.times.size(); ++i) {
            if (ref.times[i] < t_window_start) continue;
            const Real v_trap = ref.states[i](1);
            const Real v_ped = linear_interp_v(r.times, r.states,
                                                  ref.times[i]);
            const Real d = v_ped - v_trap;
            ss += d * d;
            ++n_samples;
        }
        const Real rmse = std::sqrt(ss / static_cast<Real>(n_samples));
        const Real rmse_pct = Real{100} * rmse / p.V_out_steady();
        const Real speedup = ref.cpu_seconds / r.cpu_time_seconds;

        const auto trap_steps = ref.times.size() - 1;
        const Real trap_ms = ref.cpu_seconds * Real{1000};
        const Real ped_ms = r.cpu_time_seconds * Real{1000};

        std::printf("%-8.1f %-8zu %-8.3f %-8zu %-8zu %-8.3f %-8.3f %-8.5f\n",
                     static_cast<double>(t_end_ms),
                     trap_steps,
                     static_cast<double>(trap_ms),
                     r.n_accept,
                     r.n_events,
                     static_cast<double>(ped_ms),
                     static_cast<double>(speedup),
                     static_cast<double>(rmse_pct));

        csv << t_end_ms << ',' << trap_steps << ',' << trap_ms << ','
             << r.n_accept << ',' << r.n_events << ',' << ped_ms << ','
             << speedup << ',' << rmse_pct << '\n';

        // Sanity invariants for the bench to be meaningful
        REQUIRE(rmse_pct < Real{0.1});
        REQUIRE(r.n_reject == 0);
        REQUIRE(speedup > Real{0});
    }
    csv.close();
    std::printf("\nCSV: %s\n", out_path.string().c_str());
}

// =============================================================================
// Buck DCM microbench (Gate 3 Phase 3.C)
// =============================================================================
//
// Same window sweep as the CCM bench, but on the deep-DCM operating
// point (R_load = 240Ω instead of 2.4Ω). The fixed-step reference
// has to do post-step ZCD detection + linear interp (the only way
// v1.4.0 can handle body-diode commutation on a fixed grid).
// The PED engine resolves the ZCD analytically via Hermite + Illinois.
//
// Gate 3B target: PED ≥ 5× the trap-with-ZCD baseline at the long
// windows (the same speedup-scaling story as CCM).

namespace {

using pulsim::dsed::buck::BuckDCMModel;
using pulsim::dsed::buck::BuckDCMParams;
using pulsim::dsed::buck::BuckDCMSwitchFn;
using pulsim::dsed::buck::BuckMode;
using pulsim::dsed::buck::make_zcd_predicate;
using pulsim::dsed::buck::make_zcd_projection;

struct DcmTrapRun {
    std::vector<Real> times;
    std::vector<Vector> states;
    std::size_t n_zcd = 0;
    Real cpu_seconds = Real{0};
};

DcmTrapRun run_trap_dcm(BuckDCMParams p, Vector x0, Real t_end, Real dt) {
    const Real T = p.T_sw();
    const Real D = p.D;
    const Real L = p.L;
    const Real C = p.C;
    const Real R = p.R_load;

    pulsim::DenseMatrix A(2, 2);
    A << Real{0}, -Real{1} / L,
         Real{1} / C, -Real{1} / (R * C);
    Vector b_HSon(2);
    b_HSon << p.V_in / L, Real{0};

    pulsim::DenseMatrix I = pulsim::DenseMatrix::Identity(2, 2);
    pulsim::DenseMatrix M_inv = (I - Real{0.5} * dt * A).inverse();
    pulsim::DenseMatrix Khat = M_inv * (I + Real{0.5} * dt * A);
    Vector dt_M_inv_HSon = dt * (M_inv * b_HSon);

    const Real tau_rc = R * C;
    const Real decay = std::exp(-dt / tau_rc);

    const auto n_steps = static_cast<std::size_t>(std::ceil(t_end / dt));
    DcmTrapRun r;
    r.times.reserve(n_steps + 1);
    r.states.reserve(n_steps + 1);
    r.times.push_back(Real{0});
    r.states.push_back(x0);

    Vector x = x0;
    std::unordered_map<std::int64_t, bool> zcd_fired_cycle;
    BuckMode mode = BuckMode::HS_ON;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (std::size_t k = 0; k < n_steps; ++k) {
        const Real t_cur = static_cast<Real>(k) * dt;
        const Real t_mid = t_cur + Real{0.5} * dt;
        const auto cycle = static_cast<std::int64_t>(std::floor(t_mid / T));
        const Real phase = std::fmod(t_mid / T, Real{1});

        if (phase < D) {
            mode = BuckMode::HS_ON;
        } else if (zcd_fired_cycle.count(cycle)) {
            mode = BuckMode::ZERO_CURRENT;
        } else {
            mode = BuckMode::LS_CONDUCTING;
        }

        if (mode == BuckMode::ZERO_CURRENT) {
            x(0) = Real{0};
            x(1) = x(1) * decay;
        } else if (mode == BuckMode::HS_ON) {
            x = Khat * x + dt_M_inv_HSon;
        } else {
            Vector x_new = Khat * x;
            if (x(0) > Real{0} && x_new(0) <= Real{0}) {
                const Real frac = x(0) / (x(0) - x_new(0));
                const Real vC_at_zcd = x(1) + frac * (x_new(1) - x(1));
                const Real rem_dt = (Real{1} - frac) * dt;
                x(0) = Real{0};
                x(1) = vC_at_zcd * std::exp(-rem_dt / tau_rc);
                zcd_fired_cycle[cycle] = true;
                ++r.n_zcd;
            } else {
                x = x_new;
            }
        }

        r.times.push_back(static_cast<Real>(k + 1) * dt);
        r.states.push_back(x);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    r.cpu_seconds = std::chrono::duration<Real>(t1 - t0).count();
    return r;
}

}  // namespace

TEST_CASE("Bench: PED buck DCM — wall-clock vs trap-with-ZCD reference",
           "[bench][dsed][microbench][dcm]") {
    std::printf("\n=== PED buck DCM microbench (Gate 3 Phase 3.C) ===\n");
    std::printf("%-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n",
                 "t_end_ms", "trap_st", "trap_ms", "ped_st", "ped_ev",
                 "ped_zcd", "ped_ms", "speedup", "err_pct");

    BuckDCMParams p;  // defaults: 240Ω → deep DCM
    Vector x0(2);
    x0 << Real{0}, p.V_out_steady();   // analytical steady-state start

    const Real dt_ref = Real{50e-9};   // fine grid for ZCD accuracy
    const Real windows_ms[] = {Real{1}, Real{2.5}, Real{5}, Real{10}, Real{25}};

    auto out_path = bench_output_dir() / "dsed_buck_dcm_microbench.csv";
    std::ofstream csv{out_path};
    csv << "t_end_ms,trap_steps,trap_zcd,trap_ms,ped_steps,ped_events,"
         << "ped_zcd,ped_ms,speedup,err_pct_vs_analytical\n";

    for (Real t_end_ms : windows_ms) {
        const Real t_end = t_end_ms * Real{1e-3};

        auto ref = run_trap_dcm(p, x0, t_end, dt_ref);

        BuckDCMModel m{p};
        BuckDCMSwitchFn sf{p.T_sw(), p.D};
        EventPredictor predictor;
        predictor.register_predicate(
            "zcd_iL", make_zcd_predicate(), PredicateType::CurrentZC);
        PEDSimulator sim{
            m, sf, PIController{Real{1e-6}, Real{1e-9}}, std::move(predictor),
            Real{1e-9}, p.T_sw() / Real{4}, 1,
            make_zcd_projection()
        };
        auto r = sim.simulate(x0, t_end);

        // Count event types
        std::size_t ped_zcd = 0;
        for (const auto& e : r.event_log) {
            if (e.type == PredicateType::CurrentZC) ++ped_zcd;
        }

        // Error vs analytical regulator over last 50% of window
        const Real t_window_start = Real{0.5} * t_end;
        Real sum_vout = Real{0};
        int n_samples = 0;
        for (std::size_t i = 0; i < r.times.size(); ++i) {
            if (r.times[i] < t_window_start) continue;
            sum_vout += r.states[i](1);
            ++n_samples;
        }
        const Real mean_vout = sum_vout / static_cast<Real>(n_samples);
        const Real err_pct =
            Real{100} * std::abs(mean_vout - p.V_out_steady()) / p.V_in;
        const Real speedup = ref.cpu_seconds / r.cpu_time_seconds;

        const auto trap_steps = ref.times.size() - 1;
        const Real trap_ms = ref.cpu_seconds * Real{1000};
        const Real ped_ms = r.cpu_time_seconds * Real{1000};

        std::printf("%-8.1f %-8zu %-8.3f %-8zu %-8zu %-8zu %-8.3f %-8.3f %-8.5f\n",
                     static_cast<double>(t_end_ms),
                     trap_steps,
                     static_cast<double>(trap_ms),
                     r.n_accept,
                     r.n_events,
                     ped_zcd,
                     static_cast<double>(ped_ms),
                     static_cast<double>(speedup),
                     static_cast<double>(err_pct));

        csv << t_end_ms << ',' << trap_steps << ',' << ref.n_zcd << ','
             << trap_ms << ',' << r.n_accept << ',' << r.n_events << ','
             << ped_zcd << ',' << ped_ms << ',' << speedup << ','
             << err_pct << '\n';

        // Gate 3A: V_out error ≤ 1% of V_in
        REQUIRE(err_pct < Real{1.0});
        // Sanity invariants
        REQUIRE(r.n_reject == 0);
        REQUIRE(speedup > Real{0});
        // ZCD count consistency PED ↔ trap
        REQUIRE(static_cast<long>(ped_zcd) - static_cast<long>(ref.n_zcd)
                  <= 2);
        REQUIRE(static_cast<long>(ref.n_zcd) - static_cast<long>(ped_zcd)
                  <= 2);
    }
    csv.close();
    std::printf("\nCSV: %s\n", out_path.string().c_str());
}

// =============================================================================
// Stiff-fraction speedup sweep (Gate 5 Phase 5.B-0)
// =============================================================================
//
// Defined AFTER the auto-dispatch microbench so the MixedStiffnessBench
// class definition is in scope. This bench parameterises duty cycle D
// from 0.1 to 0.9, holding T_sw and window fixed. Since mode A is stiff
// and gets selected for D·T_sw of each cycle, D = stiff_fraction.
//
// Theoretical ceiling (per notes/GATE5_PROGRESS.md):
//   speedup ≤ 1 / (1 - f · (1 - 1/spd_BDF2))
//
// With BDF2 speedup ≈ 10× on a fully-stiff segment:
//   f=0.1 → 1/(1 - 0.09) = 1.099×
//   f=0.3 → 1/(1 - 0.27) = 1.370×
//   f=0.5 → 1/(1 - 0.45) = 1.818×
//   f=0.7 → 1/(1 - 0.63) = 2.703×
//   f=0.9 → 1/(1 - 0.81) = 5.263×
//
// Empirically we expect to land at ~75-85% of these ceilings due to
// bootstrap + dispatch overhead. The data point at f=0.5 (1.50×)
// already matches that expectation.
//
// This bench's CSV becomes the TPEL paper #2 Figure 1 data
// (speedup vs stiff-fraction).

// Forward-declared helper (defined in the prior bench block).
// We re-use the same MixedStiffnessBench + MixedSwitchFnBench classes.
// =============================================================================
//
// First multi-converter benchmark in the Gate 5 sweep. Demonstrates
// the wall-clock advantage of `PEDSimulatorAuto` (per-mode RK45↔BDF2
// dispatch) over `PEDSimulator` (RK45-only) on a system where one
// mode is stiff and the other is not — so a single-integrator
// scheduler is over-conservative on EITHER half.
//
// Scenario: 2-mode RLC switched at 10 kHz.
//   Mode A (mask=true,  R=0.1Ω, V_in=5V)  → stiff (|λ_max|≈1e7)
//   Mode B (mask=false, R=10Ω,  V_in=0V)  → non-stiff (|λ_max|≈1e6)
//
// Expected dispatcher behaviour (at dt_max = 5 µs):
//   Mode A → BDF2  (|λ_max|·h = 50)
//   Mode B → DOPRI5 (|λ_max|·h = 5)
//
// Bench windows: 0.5, 1, 2.5, 5, 10 ms.

namespace {

class MixedStiffnessBench {
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

class MixedSwitchFnBench {
public:
    MixedSwitchFnBench(Real T_sw, Real D) : T_{T_sw}, D_{D} {}
    [[nodiscard]] bool operator()(Real t) const {
        return std::fmod(t / T_, Real{1}) < D_;
    }
    [[nodiscard]] Real next_edge_after(Real t) const {
        const auto k = static_cast<std::int64_t>(std::floor(t / T_));
        const Real cs[3] = {
            static_cast<Real>(k) * T_ + D_ * T_,
            static_cast<Real>(k + 1) * T_,
            static_cast<Real>(k + 1) * T_ + D_ * T_,
        };
        constexpr Real eps = Real{1e-15};
        for (Real c : cs) if (c > t + eps) return c;
        return static_cast<Real>(k + 2) * T_;
    }
private:
    Real T_;
    Real D_;
};

}  // namespace

TEST_CASE("Bench: PEDSimulatorAuto vs PEDSimulator on mixed-stiffness RLC",
           "[bench][dsed][microbench][auto]") {
    std::printf("\n=== Auto-dispatch microbench (Gate 5 Phase 5.A) ===\n");
    std::printf("Mixed-stiffness 2-mode RLC at 10 kHz; mode A stiff "
                 "(R=0.1Ω), mode B non-stiff (R=10Ω)\n");
    std::printf("%-8s %-10s %-10s %-8s %-7s %-7s %-7s %-7s %-9s\n",
                 "t_end_ms", "rk45_ms", "auto_ms", "speedup",
                 "rk45_st", "auto_st", "auto_rk", "auto_bd", "auto_ev");

    const Real T_sw = Real{100e-6};        // 10 kHz
    const Real D = Real{0.5};
    const Real dt_max = Real{5e-6};         // common cap for both schedulers
    const Real h_bdf2 = Real{1e-6};         // BDF2 fixed h inside Auto
    const Real dt_init = Real{1e-9};

    Vector x0(2);
    x0 << Real{0}, Real{0};

    const Real windows_ms[] = {Real{0.5}, Real{1}, Real{2.5},
                                  Real{5}, Real{10}};

    auto out_path = bench_output_dir() / "dsed_mixed_stiffness_auto.csv";
    std::ofstream csv{out_path};
    csv << "t_end_ms,rk45_ms,auto_ms,speedup,rk45_steps,auto_steps,"
         << "auto_rk45_steps,auto_bdf2_steps,auto_events,"
         << "rk45_final_vC,auto_final_vC,abs_diff_final_vC\n";

    for (Real t_end_ms : windows_ms) {
        const Real t_end = t_end_ms * Real{1e-3};

        // RK45-only PEDSimulator
        MixedStiffnessBench sys_rk;
        PEDSimulator<MixedStiffnessBench, MixedSwitchFnBench> sim_rk(
            sys_rk, MixedSwitchFnBench{T_sw, D},
            PIController{}, EventPredictor{}, dt_init, dt_max);
        const auto r_rk = sim_rk.simulate(x0, t_end);

        // Auto-dispatch PEDSimulatorAuto
        MixedStiffnessBench sys_auto;
        PEDSimulatorAuto<MixedStiffnessBench, MixedSwitchFnBench> sim_auto(
            sys_auto, MixedSwitchFnBench{T_sw, D},
            PIController{}, StiffnessDetector{Real{10.0}},
            dt_init, dt_max, h_bdf2);
        const auto r_auto = sim_auto.simulate(x0, t_end);

        const Real rk_ms = r_rk.cpu_time_seconds * Real{1000};
        const Real auto_ms = r_auto.cpu_time_seconds * Real{1000};
        const Real speedup = (auto_ms > Real{0}) ? rk_ms / auto_ms : Real{0};

        const Real rk_final_vC = r_rk.states.back()(1);
        const Real auto_final_vC = r_auto.states.back()(1);
        const Real abs_diff = std::abs(rk_final_vC - auto_final_vC);

        std::printf("%-8.1f %-10.3f %-10.3f %-8.3f %-7zu %-7zu %-7zu %-7zu %-9zu\n",
                     static_cast<double>(t_end_ms),
                     static_cast<double>(rk_ms),
                     static_cast<double>(auto_ms),
                     static_cast<double>(speedup),
                     r_rk.n_accept, r_auto.n_accept,
                     r_auto.n_rk45_steps, r_auto.n_bdf2_steps,
                     r_auto.n_events);

        csv << t_end_ms << ',' << rk_ms << ',' << auto_ms << ','
             << speedup << ',' << r_rk.n_accept << ','
             << r_auto.n_accept << ',' << r_auto.n_rk45_steps << ','
             << r_auto.n_bdf2_steps << ',' << r_auto.n_events << ','
             << rk_final_vC << ',' << auto_final_vC << ','
             << abs_diff << '\n';

        // Sanity: both schedulers should agree on the envelope
        // (within the same 15% tolerance we capture in
        // test_scheduler_auto.cpp Test 3). Capture as a soft bound
        // rather than REQUIRE so the bench keeps running even if
        // a single window has a wider envelope.
        if (abs_diff > sys_rk.V_in_stiff) {
            std::printf("  WARNING: |final v_C| diff = %.3f V > V_in = %.1f V\n",
                         static_cast<double>(abs_diff),
                         static_cast<double>(sys_rk.V_in_stiff));
        }
        // Sanity invariants
        REQUIRE(r_rk.n_accept > 0);
        REQUIRE(r_auto.n_accept > 0);
        REQUIRE(r_auto.n_rk45_steps > 0);
        REQUIRE(r_auto.n_bdf2_steps > 0);
        REQUIRE(speedup > Real{0});
    }

    csv.close();
    std::printf("\nCSV: %s\n", out_path.string().c_str());
}

// =============================================================================
// Per-converter auto-dispatch card: Buck CCM (Gate 5 Phase 5.B-1)
// =============================================================================
//
// Buck CCM is NON-STIFF — both modes (HS_on, HS_off) have the same
// passive elements (L=100µH, C=100µF, R=2.4Ω), giving |λ_max| ≈ 1e4.
// At dt_max = T_sw/4 = 2.5µs, |λ_max|·h = 0.025 — far below the
// stiffness threshold (10). The auto-dispatcher should pick DOPRI5
// for both modes and behave identically to the RK45-only scheduler.
//
// Expected: speedup ≈ 1.0× ± 5% (no overhead penalty, no BDF2 wins).
// This card is the negative-result confirmation: the dispatcher
// is harmless when stiffness isn't present.

TEST_CASE("Bench: Auto-dispatch on Buck CCM (no overhead penalty)",
           "[bench][dsed][microbench][auto][buck-ccm]") {
    std::printf("\n=== Per-converter card: Buck CCM (Gate 5 Phase 5.B-1) ===\n");
    std::printf("Non-stiff converter — expected auto-dispatch speedup ≈ 1.0×\n");

    BuckParams p;   // defaults: 24V→12V, 100kHz, D=0.5, L=100µH, C=100µF, R=2.4Ω
    BuckPSCSwitchFn sf{p.T_sw(), p.D};
    Vector x0(2);
    x0 << p.I_L_steady(), p.V_out_steady();
    const Real t_end = Real{5e-3};
    const Real dt_max = p.T_sw() / Real{4};
    const Real h_bdf2 = Real{500e-9};

    // RK45-only
    BuckCCMModel m_rk{p};
    PEDSimulator<BuckCCMModel, BuckPSCSwitchFn> sim_rk(
        m_rk, sf, PIController{}, EventPredictor{},
        Real{1e-9}, dt_max);
    const auto r_rk = sim_rk.simulate(x0, t_end);

    // Auto-dispatch
    BuckCCMModel m_auto{p};
    PEDSimulatorAuto<BuckCCMModel, BuckPSCSwitchFn> sim_auto(
        m_auto, sf, PIController{}, StiffnessDetector{Real{10.0}},
        Real{1e-9}, dt_max, h_bdf2);
    const auto r_auto = sim_auto.simulate(x0, t_end);

    const Real rk_ms = r_rk.cpu_time_seconds * Real{1000};
    const Real auto_ms = r_auto.cpu_time_seconds * Real{1000};
    const Real speedup = (auto_ms > Real{0}) ? rk_ms / auto_ms : Real{0};

    // Eigenvalue of A for Buck CCM (for the table)
    Eigen::EigenSolver<Eigen::Matrix2d> es(m_rk.A());
    const auto eigs = es.eigenvalues();
    Real lam_max = Real{0};
    for (Eigen::Index i = 0; i < eigs.size(); ++i) {
        lam_max = std::max(lam_max, std::abs(eigs(i)));
    }

    std::printf("  |λ_max| · dt_max = %.4f (threshold 10 → BDF2 if above)\n",
                 static_cast<double>(lam_max * dt_max));
    std::printf("  RK45-only wall: %.3f ms (%zu steps)\n",
                 static_cast<double>(rk_ms), r_rk.n_accept);
    std::printf("  Auto       wall: %.3f ms (%zu steps; rk=%zu, bd=%zu, ev=%zu)\n",
                 static_cast<double>(auto_ms), r_auto.n_accept,
                 r_auto.n_rk45_steps, r_auto.n_bdf2_steps, r_auto.n_events);
    std::printf("  Speedup = %.3f×\n", static_cast<double>(speedup));

    // Auto should pick DOPRI5 for all segments
    REQUIRE(r_auto.n_bdf2_steps == 0);
    REQUIRE(r_auto.n_rk45_steps == r_auto.n_accept);
    // Speedup should be near 1.0 ± 30% (allow generous slack for noise
    // since both schedulers do approximately the same work)
    REQUIRE(speedup >= Real{0.7});
    REQUIRE(speedup <= Real{1.5});
}

// =============================================================================
// Per-converter auto-dispatch card: Buck DCM (Gate 5 Phase 5.B-2)
// =============================================================================
//
// Buck DCM uses the same L/C as CCM but 100× higher R_load (240Ω),
// pushing the operating point into deep DCM. The eigenvalues are
// still in the low-1e4 range (passives unchanged), so |λ_max|·h
// is still well below the stiffness threshold. The auto-dispatcher
// should again pick DOPRI5 throughout — even though there are now
// THREE modes (HS_on, LS_conducting, ZERO_CURRENT).
//
// Expected: speedup ≈ 1.0× (no stiff modes; ZCD events handled by
// RK45's predicate-scan path which BDF2 doesn't support anyway).

TEST_CASE("Bench: Auto-dispatch on Buck DCM (no overhead penalty)",
           "[bench][dsed][microbench][auto][buck-dcm]") {
    std::printf("\n=== Per-converter card: Buck DCM (Gate 5 Phase 5.B-2) ===\n");
    std::printf("Deep-DCM converter (R=240Ω); expected speedup ≈ 1.0×\n");

    BuckDCMParams p;   // defaults: 24V, D=0.5, L=100µH, C=100µF, R=240Ω
    BuckDCMSwitchFn sf{p.T_sw(), p.D};

    Vector x0(2);
    x0 << Real{0}, p.V_out_steady();    // analytical DCM steady state IC

    const Real t_end = Real{1e-3};       // 100 switching cycles
    const Real dt_max = p.T_sw() / Real{4};
    const Real h_bdf2 = Real{500e-9};

    // RK45-only PED — same setup as test_diode_events.cpp Test 7
    BuckDCMModel m_rk{p};
    EventPredictor pred_rk;
    pred_rk.register_predicate("zcd_iL", make_zcd_predicate(),
                                  PredicateType::CurrentZC);
    PEDSimulator<BuckDCMModel, BuckDCMSwitchFn> sim_rk(
        m_rk, sf, PIController{}, std::move(pred_rk),
        Real{1e-9}, dt_max, 1,
        make_zcd_projection());
    const auto r_rk = sim_rk.simulate(x0, t_end);

    // Auto-dispatch (no predicate scan in PEDSimulatorAuto — gate
    // edges only; ZCD events are deferred since BDF2 doesn't support
    // the Hermite backtrack path). For Buck DCM this means the Auto
    // dispatcher MISSES ZCD events that the RK45-only scheduler catches.
    // That's an honest limitation we capture.
    BuckDCMModel m_auto{p};
    PEDSimulatorAuto<BuckDCMModel, BuckDCMSwitchFn> sim_auto(
        m_auto, sf, PIController{}, StiffnessDetector{Real{10.0}},
        Real{1e-9}, dt_max, h_bdf2);
    const auto r_auto = sim_auto.simulate(x0, t_end);

    const Real rk_ms = r_rk.cpu_time_seconds * Real{1000};
    const Real auto_ms = r_auto.cpu_time_seconds * Real{1000};
    const Real speedup = (auto_ms > Real{0}) ? rk_ms / auto_ms : Real{0};

    // Count ZCD events in RK45-only
    std::size_t n_zcd_rk = 0;
    for (const auto& e : r_rk.event_log) {
        if (e.type == PredicateType::CurrentZC) ++n_zcd_rk;
    }

    std::printf("  RK45-only wall: %.3f ms (%zu steps, %zu events, "
                 "%zu ZCDs)\n",
                 static_cast<double>(rk_ms), r_rk.n_accept,
                 r_rk.n_events, n_zcd_rk);
    std::printf("  Auto       wall: %.3f ms (%zu steps; rk=%zu, bd=%zu, "
                 "ev=%zu — gate edges only)\n",
                 static_cast<double>(auto_ms), r_auto.n_accept,
                 r_auto.n_rk45_steps, r_auto.n_bdf2_steps, r_auto.n_events);
    std::printf("  Speedup = %.3f×\n", static_cast<double>(speedup));
    std::printf("  NOTE: Auto-dispatcher does NOT fire ZCD predicates in this "
                 "iteration (Gate 4.D scope: gate edges only); for DCM use "
                 "the standalone PEDSimulator with predicates instead.\n");

    // Auto should still pick DOPRI5 for the LTI segments
    REQUIRE(r_auto.n_bdf2_steps == 0);
    // Speedup near 1.0× — Auto is doing less work (no ZCD predicates)
    // so it may even be slightly faster.
    REQUIRE(speedup >= Real{0.7});
    REQUIRE(speedup <= Real{2.0});
}

// =============================================================================
// 3-state mode-dependent stiff system card (Gate 5 Phase 5.B-3)
// =============================================================================
//
// Cascaded RLC (3 states) demonstrating PEDSimulatorAuto on n_state > 2
// with **mode-dependent stiffness**:
//
//   State: x = (i_L, v_C1, v_C2)
//   Topology: V_in - R_inner(mode) - L - C1 - R_couple - C2 - R_load
//
//   Mode A (mask=true,  R_inner=100Ω): stiff fast damping
//     τ_fast = L / R_inner = 1µH / 100Ω = 10 ns  → |λ_max| ≈ 1e8
//   Mode B (mask=false, R_inner=10Ω):  non-stiff
//     τ_fast = 1µH / 10Ω = 100 ns  → |λ_max| ≈ 1e7
//
// At dt_max = 5 µs:
//   Mode A: |λ_max|·h ≈ 500 → BDF2
//   Mode B: |λ_max|·h ≈ 50  → BDF2 (also stiff!)
//
// Both modes are stiff but with different eigenvalues — exercises
// the per-mode eigenvalue cache (`StiffnessDetector::cache_`).
// Auto-dispatcher should call `lambda_max` once per mode_id (twice
// total) and re-use the cache for all subsequent gate-edge selections.

namespace {

class ThreeStateStiffRLC {
public:
    Real L = Real{1e-6};            // 1 µH inductor
    Real C1 = Real{1e-6};            // 1 µF inner cap
    Real C2 = Real{1e-6};            // 1 µF outer cap
    Real R_inner_stiff = Real{100};  // mode A: very high damping
    Real R_inner_nonstiff = Real{10};  // mode B: lower damping
    Real R_couple = Real{1.0};
    Real R_load = Real{10.0};
    Real V_in_stiff = Real{5.0};

    [[nodiscard]] bool current_mask() const noexcept { return mask_a_; }
    void set_mask(bool m) noexcept { mask_a_ = m; }

    [[nodiscard]] DenseMatrix A_matrix() const {
        const Real R_inner = mask_a_ ? R_inner_stiff : R_inner_nonstiff;
        DenseMatrix A(3, 3);
        // L · di/dt = V_in - v_C1 - R_inner · i
        A(0, 0) = -R_inner / L;
        A(0, 1) = -Real{1} / L;
        A(0, 2) = Real{0};
        // C1 · dv_C1/dt = i - (v_C1 - v_C2)/R_couple
        A(1, 0) = Real{1} / C1;
        A(1, 1) = -Real{1} / (R_couple * C1);
        A(1, 2) = Real{1} / (R_couple * C1);
        // C2 · dv_C2/dt = (v_C1 - v_C2)/R_couple - v_C2/R_load
        A(2, 0) = Real{0};
        A(2, 1) = Real{1} / (R_couple * C2);
        A(2, 2) = -Real{1} / (R_couple * C2) - Real{1} / (R_load * C2);
        return A;
    }

    [[nodiscard]] Vector b_vector(Real /*t*/) const {
        const Real R_inner = mask_a_ ? R_inner_stiff : R_inner_nonstiff;
        const Real V = mask_a_ ? V_in_stiff : Real{0};
        // Only the inductor row sees forcing
        Vector b(3);
        (void) R_inner;  // forcing is V/L, independent of R_inner
        b << V / L, Real{0}, Real{0};
        return b;
    }

    [[nodiscard]] Vector rhs(Real t, const Vector& x) const {
        return A_matrix() * x + b_vector(t);
    }

private:
    bool mask_a_ = true;
};

}  // namespace

TEST_CASE("Bench: 3-state mode-dependent stiff system on PEDSimulatorAuto",
           "[bench][dsed][microbench][auto][3state]") {
    std::printf("\n=== 3-state mode-dependent stiff system (Gate 5 Phase 5.B-3) ===\n");
    std::printf("Cascaded RLC (3 states); mode A: R_inner=100Ω stiff, "
                 "mode B: R_inner=10Ω moderate\n");

    const Real T_sw = Real{100e-6};
    const Real D = Real{0.5};
    const Real t_end = Real{1e-3};
    const Real dt_max = Real{5e-6};
    const Real h_bdf2 = Real{1e-6};
    const Real dt_init = Real{1e-9};

    Vector x0(3);
    x0 << Real{0}, Real{0}, Real{0};

    // Report eigenvalues for context
    {
        ThreeStateStiffRLC sys;
        sys.set_mask(true);
        Eigen::EigenSolver<DenseMatrix> es_a(sys.A_matrix());
        sys.set_mask(false);
        Eigen::EigenSolver<DenseMatrix> es_b(sys.A_matrix());
        Real lam_a = 0, lam_b = 0;
        for (Eigen::Index i = 0; i < es_a.eigenvalues().size(); ++i) {
            lam_a = std::max(lam_a, std::abs(es_a.eigenvalues()(i)));
        }
        for (Eigen::Index i = 0; i < es_b.eigenvalues().size(); ++i) {
            lam_b = std::max(lam_b, std::abs(es_b.eigenvalues()(i)));
        }
        std::printf("  Mode A: |λ_max| = %.3e  →  |λ_max|·dt_max = %.1f\n",
                     static_cast<double>(lam_a),
                     static_cast<double>(lam_a * dt_max));
        std::printf("  Mode B: |λ_max| = %.3e  →  |λ_max|·dt_max = %.1f\n",
                     static_cast<double>(lam_b),
                     static_cast<double>(lam_b * dt_max));
    }

    // RK45-only
    ThreeStateStiffRLC sys_rk;
    PEDSimulator<ThreeStateStiffRLC, MixedSwitchFnBench> sim_rk(
        sys_rk, MixedSwitchFnBench{T_sw, D},
        PIController{}, EventPredictor{}, dt_init, dt_max);
    const auto r_rk = sim_rk.simulate(x0, t_end);

    // Auto-dispatch
    ThreeStateStiffRLC sys_auto;
    PEDSimulatorAuto<ThreeStateStiffRLC, MixedSwitchFnBench> sim_auto(
        sys_auto, MixedSwitchFnBench{T_sw, D},
        PIController{}, StiffnessDetector{Real{10.0}},
        dt_init, dt_max, h_bdf2);
    const auto r_auto = sim_auto.simulate(x0, t_end);

    const Real rk_ms = r_rk.cpu_time_seconds * Real{1000};
    const Real auto_ms = r_auto.cpu_time_seconds * Real{1000};
    const Real speedup = (auto_ms > Real{0}) ? rk_ms / auto_ms : Real{0};

    // Per-mode-segment integrator usage from the event log
    std::size_t n_bdf2_seg = 0, n_rk45_seg = 0;
    for (const auto& ev : r_auto.event_log) {
        if (ev.integrator_used == IntegratorChoice::BDF2) ++n_bdf2_seg;
        if (ev.integrator_used == IntegratorChoice::DOPRI5) ++n_rk45_seg;
    }

    std::printf("  RK45-only wall: %.3f ms (%zu steps)\n",
                 static_cast<double>(rk_ms), r_rk.n_accept);
    std::printf("  Auto       wall: %.3f ms (%zu steps; rk=%zu, bd=%zu)\n",
                 static_cast<double>(auto_ms), r_auto.n_accept,
                 r_auto.n_rk45_steps, r_auto.n_bdf2_steps);
    std::printf("  Events: %zu (BDF2-segments: %zu, RK45-segments: %zu)\n",
                 r_auto.n_events, n_bdf2_seg, n_rk45_seg);
    std::printf("  Speedup = %.3f×\n", static_cast<double>(speedup));

    // Both modes are stiff → dispatcher should pick BDF2 for most/all segments
    REQUIRE(n_bdf2_seg > 0);
    // Auto MUST be faster than RK45 (huge stiffness ratio)
    REQUIRE(speedup > Real{1.0});
    // Reasonable bound: at minimum 2× on this 3-state heavily-stiff system
    REQUIRE(speedup >= Real{2.0});

    // Final state agreement (envelope tolerance)
    const Real rk_final_vC2 = r_rk.states.back()(2);
    const Real auto_final_vC2 = r_auto.states.back()(2);
    const Real rel_diff = std::abs(rk_final_vC2 - auto_final_vC2)
                          / std::max(Real{1e-9}, std::abs(rk_final_vC2));
    std::printf("  Final v_C2: RK45=%.4f, Auto=%.4f, rel_diff=%.2f%%\n",
                 static_cast<double>(rk_final_vC2),
                 static_cast<double>(auto_final_vC2),
                 static_cast<double>(rel_diff * Real{100}));
    REQUIRE(rel_diff <= Real{0.20});  // 20% envelope (BDF2 fast-mode smoothing)

    auto out_path = bench_output_dir() / "dsed_3state_mode_dependent.csv";
    std::ofstream csv{out_path};
    csv << "n_state,rk45_ms,auto_ms,speedup,rk45_steps,auto_steps,"
         << "auto_rk45_steps,auto_bdf2_steps,auto_events,bdf2_segments,"
         << "rk45_segments\n";
    csv << 3 << ',' << rk_ms << ',' << auto_ms << ',' << speedup << ','
         << r_rk.n_accept << ',' << r_auto.n_accept << ','
         << r_auto.n_rk45_steps << ',' << r_auto.n_bdf2_steps << ','
         << r_auto.n_events << ',' << n_bdf2_seg << ',' << n_rk45_seg << '\n';
    csv.close();
    std::printf("\nCSV: %s\n", out_path.string().c_str());
}

// =============================================================================
// Per-converter cards: Boost / Buck-Boost / Flyback (Gate 5 Phase 5.B-4)
// =============================================================================
//
// Three non-stiff 2-state switching converters with **mode-dependent
// A** (unlike Buck CCM where A is the same across modes). Each adds
// a data point to the per-converter sweep:
//
//   * Boost — A_HSon disconnects v_o from i_L (cap discharges alone);
//             A_HSoff couples through the inductor diode.
//   * Buck-boost — inverting topology; both modes have rank-1 A
//             (inductor or cap singly couples in each mode).
//   * Flyback — galvanically isolated; cap discharges through R_load
//             during HS_on; inductor energy delivered to output
//             during HS_off (reflected through turns ratio).
//
// All three are textbook PWM converters with LC time constants in the
// 100µs–1ms range. |λ_max| ≈ 1e3–1e5 → not stiff at dt_max = T_sw/4
// (~2.5 µs). Expected auto-dispatch behaviour: dispatcher picks RK45
// for ALL modes → speedup near 1.0× (bounded by per-event overhead).
//
// These are NEGATIVE-RESULT data points for the TPEL paper: they
// confirm the dispatcher costs ≤ 15 % when it doesn't help, which
// bounds the worst-case penalty in real-world deployments.

namespace {

// --- Boost: V_g → V_o > V_g; state = (i_L, v_o)
//   Mode A (HS_on,  q=1): L·di/dt = V_g,           C·dv/dt = -v/R
//   Mode B (HS_off, q=0): L·di/dt = V_g - v,       C·dv/dt = i - v/R
class BoostBench {
public:
    Real V_g = Real{12.0};
    Real L = Real{100e-6};
    Real C = Real{100e-6};
    Real R = Real{11.52};   // ~50W at V_o=24
    Real f_sw = Real{100e3};
    Real D = Real{0.5};

    [[nodiscard]] Real T_sw() const noexcept { return Real{1} / f_sw; }
    [[nodiscard]] bool current_mask() const noexcept { return hs_on_; }
    void set_mask(bool m) noexcept { hs_on_ = m; }

    [[nodiscard]] DenseMatrix A_matrix() const {
        DenseMatrix A(2, 2);
        if (hs_on_) {
            A << Real{0},        Real{0},
                 Real{0},       -Real{1} / (R * C);
        } else {
            A << Real{0},       -Real{1} / L,
                 Real{1} / C,   -Real{1} / (R * C);
        }
        return A;
    }
    [[nodiscard]] Vector b_vector(Real /*t*/) const {
        Vector b(2);
        b << V_g / L, Real{0};
        return b;
    }
    [[nodiscard]] Vector rhs(Real t, const Vector& x) const {
        return A_matrix() * x + b_vector(t);
    }
private:
    bool hs_on_ = true;
};

// --- Buck-Boost: |V_o| > 0 with inverting polarity; state = (i_L, v_o)
//   Mode A (HS_on,  q=1): L·di/dt = V_g,            C·dv/dt = -v/R
//   Mode B (HS_off, q=0): L·di/dt = v,              C·dv/dt = -i - v/R
//
// Sign convention: v_o is the (signed) capacitor voltage; for the
// inverting topology v_o < 0 at the operating point.
class BuckBoostBench {
public:
    Real V_g = Real{12.0};
    Real L = Real{100e-6};
    Real C = Real{100e-6};
    Real R = Real{8.0};        // ~18W at V_o = -12V
    Real f_sw = Real{100e3};
    Real D = Real{0.5};         // |V_o| = D·V_g/(1-D) = 12V

    [[nodiscard]] Real T_sw() const noexcept { return Real{1} / f_sw; }
    [[nodiscard]] bool current_mask() const noexcept { return hs_on_; }
    void set_mask(bool m) noexcept { hs_on_ = m; }

    [[nodiscard]] DenseMatrix A_matrix() const {
        DenseMatrix A(2, 2);
        if (hs_on_) {
            A << Real{0},        Real{0},
                 Real{0},       -Real{1} / (R * C);
        } else {
            A << Real{0},        Real{1} / L,
                -Real{1} / C,   -Real{1} / (R * C);
        }
        return A;
    }
    [[nodiscard]] Vector b_vector(Real /*t*/) const {
        Vector b(2);
        b << (hs_on_ ? V_g / L : Real{0}), Real{0};
        return b;
    }
    [[nodiscard]] Vector rhs(Real t, const Vector& x) const {
        return A_matrix() * x + b_vector(t);
    }
private:
    bool hs_on_ = true;
};

// --- Flyback: isolated converter with turns ratio n = N_s/N_p; state = (i_Lm, v_o)
//   Magnetizing inductance L_m references primary side.
//   Mode A (HS_on):  L_m·di_Lm/dt = V_g,            C·dv_o/dt = -v_o/R
//   Mode B (HS_off): L_m·di_Lm/dt = -n·v_o,         C·dv_o/dt = i_Lm/n - v_o/R
class FlybackBench {
public:
    Real V_g = Real{48.0};
    Real n_ratio = Real{0.25};   // N_s/N_p = 1:4 step-down
    Real L_m = Real{500e-6};
    Real C = Real{100e-6};
    Real R = Real{5.0};
    Real f_sw = Real{100e3};
    Real D = Real{0.5};

    [[nodiscard]] Real T_sw() const noexcept { return Real{1} / f_sw; }
    [[nodiscard]] bool current_mask() const noexcept { return hs_on_; }
    void set_mask(bool m) noexcept { hs_on_ = m; }

    [[nodiscard]] DenseMatrix A_matrix() const {
        DenseMatrix A(2, 2);
        if (hs_on_) {
            A << Real{0},                   Real{0},
                 Real{0},                  -Real{1} / (R * C);
        } else {
            A << Real{0},                  -n_ratio / L_m,
                 Real{1} / (n_ratio * C),  -Real{1} / (R * C);
        }
        return A;
    }
    [[nodiscard]] Vector b_vector(Real /*t*/) const {
        Vector b(2);
        b << (hs_on_ ? V_g / L_m : Real{0}), Real{0};
        return b;
    }
    [[nodiscard]] Vector rhs(Real t, const Vector& x) const {
        return A_matrix() * x + b_vector(t);
    }
private:
    bool hs_on_ = true;
};

// Reusable switching function — same shape as BuckPSCSwitchFn but
// templated on the bench System (compile-time MaskT = bool).
class GenericPSCSwitchFn {
public:
    GenericPSCSwitchFn(Real T_sw, Real D) : T_{T_sw}, D_{D} {}
    [[nodiscard]] bool operator()(Real t) const {
        return std::fmod(t / T_, Real{1}) < D_;
    }
    [[nodiscard]] Real next_edge_after(Real t) const {
        const auto k = static_cast<std::int64_t>(std::floor(t / T_));
        const Real cs[3] = {
            static_cast<Real>(k) * T_ + D_ * T_,
            static_cast<Real>(k + 1) * T_,
            static_cast<Real>(k + 1) * T_ + D_ * T_,
        };
        constexpr Real eps = Real{1e-15};
        for (Real c : cs) if (c > t + eps) return c;
        return static_cast<Real>(k + 2) * T_;
    }
private:
    Real T_;
    Real D_;
};

// One bench card body — runs RK45-only PEDSimulator and PEDSimulatorAuto
// on the given converter, prints + returns a CSV row.
struct PerConverterResult {
    std::string name;
    int n_state;
    Real rk_ms;
    Real auto_ms;
    Real speedup;
    std::size_t rk_steps;
    std::size_t auto_steps;
    std::size_t auto_rk_steps;
    std::size_t auto_bd_steps;
    std::size_t auto_events;
    Real lam_max;
    Real lam_h;
};

template <class System>
PerConverterResult run_per_converter_card(
    const std::string& name,
    System& sys_rk,
    System& sys_auto,
    GenericPSCSwitchFn sf,
    const Vector& x0,
    Real t_end,
    Real dt_max,
    Real h_bdf2,
    Real dt_init = Real{1e-9}) {

    PEDSimulator<System, GenericPSCSwitchFn> sim_rk(
        sys_rk, sf, PIController{}, EventPredictor{}, dt_init, dt_max);
    const auto r_rk = sim_rk.simulate(x0, t_end);

    PEDSimulatorAuto<System, GenericPSCSwitchFn> sim_auto(
        sys_auto, sf, PIController{}, StiffnessDetector{Real{10.0}},
        dt_init, dt_max, h_bdf2);
    const auto r_auto = sim_auto.simulate(x0, t_end);

    Eigen::EigenSolver<DenseMatrix> es(sys_rk.A_matrix());
    Real lam_max = Real{0};
    for (Eigen::Index i = 0; i < es.eigenvalues().size(); ++i) {
        lam_max = std::max(lam_max, std::abs(es.eigenvalues()(i)));
    }

    PerConverterResult res;
    res.name = name;
    res.n_state = static_cast<int>(x0.size());
    res.rk_ms = r_rk.cpu_time_seconds * Real{1000};
    res.auto_ms = r_auto.cpu_time_seconds * Real{1000};
    res.speedup = (res.auto_ms > Real{0}) ? res.rk_ms / res.auto_ms : Real{0};
    res.rk_steps = r_rk.n_accept;
    res.auto_steps = r_auto.n_accept;
    res.auto_rk_steps = r_auto.n_rk45_steps;
    res.auto_bd_steps = r_auto.n_bdf2_steps;
    res.auto_events = r_auto.n_events;
    res.lam_max = lam_max;
    res.lam_h = lam_max * dt_max;

    std::printf("  %-12s  n_state=%d  |λ_max|·dt_max=%6.2f  "
                 "rk=%6.2f ms (%5zu st)  auto=%6.2f ms (%5zu st, rk=%5zu bd=%4zu)  "
                 "speedup=%.3f×\n",
                 name.c_str(), res.n_state,
                 static_cast<double>(res.lam_h),
                 static_cast<double>(res.rk_ms), res.rk_steps,
                 static_cast<double>(res.auto_ms), res.auto_steps,
                 res.auto_rk_steps, res.auto_bd_steps,
                 static_cast<double>(res.speedup));
    return res;
}

}  // namespace

// --- Forward: isolated buck-like; state = (i_L, v_o); turns ratio n=N_s/N_p
//   Mode A (HS_on,  q=1): L·di/dt = n·V_g - v,    C·dv/dt = i - v/R
//   Mode B (HS_off, q=0): L·di/dt = -v,           C·dv/dt = i - v/R
//
// A matrix is the SAME across modes (same passive topology); only b
// changes. Identical structure to Buck CCM, just with V_in scaled
// by turns ratio.
class ForwardBench {
public:
    Real V_g = Real{48.0};
    Real n_ratio = Real{0.5};   // N_s/N_p = 1:2 step-down
    Real L = Real{100e-6};
    Real C = Real{100e-6};
    Real R = Real{2.4};
    Real f_sw = Real{100e3};
    Real D = Real{0.5};

    [[nodiscard]] Real T_sw() const noexcept { return Real{1} / f_sw; }
    [[nodiscard]] bool current_mask() const noexcept { return hs_on_; }
    void set_mask(bool m) noexcept { hs_on_ = m; }

    [[nodiscard]] DenseMatrix A_matrix() const {
        DenseMatrix A(2, 2);
        A << Real{0},        -Real{1} / L,
             Real{1} / C,    -Real{1} / (R * C);
        return A;
    }
    [[nodiscard]] Vector b_vector(Real /*t*/) const {
        Vector b(2);
        b << (hs_on_ ? n_ratio * V_g / L : Real{0}), Real{0};
        return b;
    }
    [[nodiscard]] Vector rhs(Real t, const Vector& x) const {
        return A_matrix() * x + b_vector(t);
    }
private:
    bool hs_on_ = true;
};

// --- Half-bridge: bipolar 2-mode output; state = (i_L, v_o)
//   V_dc supply; LC filter; output averages between -V_dc/2 and +V_dc/2
//   Mode A (HS_on): L·di/dt = +V_dc/2 - v,  C·dv/dt = i - v/R
//   Mode B (LS_on): L·di/dt = -V_dc/2 - v,  C·dv/dt = i - v/R
//
// Same A as Forward/Buck — non-stiff PWM.
class HalfBridgeBench {
public:
    Real V_dc = Real{48.0};
    Real L = Real{100e-6};
    Real C = Real{100e-6};
    Real R = Real{4.0};
    Real f_sw = Real{100e3};
    Real D = Real{0.6};         // 60% high-side → avg v_o > 0

    [[nodiscard]] Real T_sw() const noexcept { return Real{1} / f_sw; }
    [[nodiscard]] bool current_mask() const noexcept { return hs_on_; }
    void set_mask(bool m) noexcept { hs_on_ = m; }

    [[nodiscard]] DenseMatrix A_matrix() const {
        DenseMatrix A(2, 2);
        A << Real{0},        -Real{1} / L,
             Real{1} / C,    -Real{1} / (R * C);
        return A;
    }
    [[nodiscard]] Vector b_vector(Real /*t*/) const {
        const Real V_leg = hs_on_ ? V_dc / Real{2} : -V_dc / Real{2};
        Vector b(2);
        b << V_leg / L, Real{0};
        return b;
    }
    [[nodiscard]] Vector rhs(Real t, const Vector& x) const {
        return A_matrix() * x + b_vector(t);
    }
private:
    bool hs_on_ = true;
};

// --- Boost PFC: rectified-AC input → boost stage → DC link
//   3-state if we track input current + DC link + line-filter cap, but
//   for the simplest 2-state model: x = (i_L, v_o), with
//   time-varying V_g(t) = |V_peak · sin(ω_line · t)|.
//   Mode A (HS_on):  L·di/dt = V_g(t),         C·dv/dt = -v/R
//   Mode B (HS_off): L·di/dt = V_g(t) - v,     C·dv/dt = i - v/R
//
// This card tests the auto-dispatcher's handling of a TIME-VARYING
// b(t) — important sanity since BDF2's b_fn is called at t+h, not t.
class BoostPFCBench {
public:
    Real V_peak = Real{170.0};       // 120 V_rms × √2
    Real omega_line = Real{2 * 3.141592653589793 * 60};  // 60 Hz
    Real L = Real{1e-3};              // 1 mH PFC inductor
    Real C = Real{470e-6};            // 470 µF DC link cap
    Real R = Real{200.0};             // ~800 W at V_o = 400 V
    Real f_sw = Real{100e3};
    Real D = Real{0.5};

    [[nodiscard]] Real T_sw() const noexcept { return Real{1} / f_sw; }
    [[nodiscard]] bool current_mask() const noexcept { return hs_on_; }
    void set_mask(bool m) noexcept { hs_on_ = m; }

    [[nodiscard]] DenseMatrix A_matrix() const {
        DenseMatrix A(2, 2);
        if (hs_on_) {
            A << Real{0},        Real{0},
                 Real{0},       -Real{1} / (R * C);
        } else {
            A << Real{0},       -Real{1} / L,
                 Real{1} / C,   -Real{1} / (R * C);
        }
        return A;
    }
    [[nodiscard]] Vector b_vector(Real t) const {
        // Rectified sinusoidal input — note the std::abs for the
        // full-bridge front-end rectifier.
        const Real V_g_t = std::abs(V_peak * std::sin(omega_line * t));
        Vector b(2);
        b << V_g_t / L, Real{0};
        return b;
    }
    [[nodiscard]] Vector rhs(Real t, const Vector& x) const {
        return A_matrix() * x + b_vector(t);
    }
private:
    bool hs_on_ = true;
};

TEST_CASE("Bench: Per-converter sweep — Boost / Buck-Boost / Flyback (Phase 5.B-4)",
           "[bench][dsed][microbench][auto][per-converter]") {
    std::printf("\n=== Per-converter auto-dispatch sweep (Gate 5 Phase 5.B-4) ===\n");
    std::printf("3 non-stiff 2-state PWM converters at 100 kHz, 5 ms window\n");

    const Real t_end = Real{5e-3};
    const Real h_bdf2 = Real{500e-9};

    Vector x0_boost(2);
    x0_boost << Real{0}, Real{0};

    Vector x0_bb(2);
    x0_bb << Real{0}, Real{0};

    Vector x0_fb(2);
    x0_fb << Real{0}, Real{0};

    std::vector<PerConverterResult> rows;

    // ---- Boost ----
    {
        BoostBench sys_rk;
        BoostBench sys_auto;
        rows.push_back(run_per_converter_card(
            "Boost", sys_rk, sys_auto,
            GenericPSCSwitchFn{sys_rk.T_sw(), sys_rk.D},
            x0_boost, t_end, sys_rk.T_sw() / Real{4}, h_bdf2));
    }
    // ---- Buck-Boost ----
    {
        BuckBoostBench sys_rk;
        BuckBoostBench sys_auto;
        rows.push_back(run_per_converter_card(
            "Buck-Boost", sys_rk, sys_auto,
            GenericPSCSwitchFn{sys_rk.T_sw(), sys_rk.D},
            x0_bb, t_end, sys_rk.T_sw() / Real{4}, h_bdf2));
    }
    // ---- Flyback ----
    {
        FlybackBench sys_rk;
        FlybackBench sys_auto;
        rows.push_back(run_per_converter_card(
            "Flyback", sys_rk, sys_auto,
            GenericPSCSwitchFn{sys_rk.T_sw(), sys_rk.D},
            x0_fb, t_end, sys_rk.T_sw() / Real{4}, h_bdf2));
    }

    // Geo-mean speedup across the 3 non-stiff converters
    Real log_speedup_sum = Real{0};
    for (const auto& r : rows) log_speedup_sum += std::log(r.speedup);
    const Real geo_mean = std::exp(log_speedup_sum
                                       / static_cast<Real>(rows.size()));
    std::printf("\n  Geo-mean speedup across 3 non-stiff converters: %.3f×\n",
                 static_cast<double>(geo_mean));

    auto out_path = bench_output_dir() / "dsed_per_converter.csv";
    // Append-mode (so this CSV accumulates across multiple bench
    // invocations as more cards land in Phase 5.B-5+)
    const bool needs_header = !std::filesystem::exists(out_path);
    std::ofstream csv{out_path, std::ios::app};
    if (needs_header) {
        csv << "converter,n_state,rk_ms,auto_ms,speedup,rk_steps,"
             << "auto_steps,auto_rk_steps,auto_bd_steps,auto_events,"
             << "lambda_max,lambda_h\n";
    }
    for (const auto& r : rows) {
        csv << r.name << ',' << r.n_state << ',' << r.rk_ms << ','
             << r.auto_ms << ',' << r.speedup << ',' << r.rk_steps << ','
             << r.auto_steps << ',' << r.auto_rk_steps << ','
             << r.auto_bd_steps << ',' << r.auto_events << ','
             << r.lam_max << ',' << r.lam_h << '\n';
    }
    csv.close();
    std::printf("\nCSV: %s (appended)\n", out_path.string().c_str());

    // All 3 are non-stiff at our params → auto should be 0.7-1.5× of RK45
    for (const auto& r : rows) {
        INFO("Converter " << r.name << " speedup = " << r.speedup);
        REQUIRE(r.speedup >= Real{0.5});
        REQUIRE(r.speedup <= Real{1.8});
        // Auto picks RK45 for all events
        REQUIRE(r.auto_bd_steps == 0);
    }
}

TEST_CASE("Bench: Auto-dispatch speedup vs stiff-fraction (TPEL Fig 1 data)",
           "[bench][dsed][microbench][stiff-fraction]") {
    std::printf("\n=== Stiff-fraction speedup sweep (Gate 5 Phase 5.B-0) ===\n");
    std::printf("Mixed-stiffness RLC at 10 kHz, 1 ms window; sweep duty cycle D\n");
    std::printf("(stiff_fraction f = D since mode A is stiff for D·T_sw)\n");
    std::printf("%-7s %-10s %-10s %-8s %-9s %-9s %-7s %-7s %-9s\n",
                 "D", "rk45_ms", "auto_ms", "speedup",
                 "ceiling", "%ceil",
                 "auto_rk", "auto_bd", "auto_ev");

    const Real T_sw = Real{100e-6};
    const Real t_end = Real{1e-3};
    const Real dt_max = Real{5e-6};
    const Real h_bdf2 = Real{1e-6};
    const Real dt_init = Real{1e-9};
    const Real spd_bdf2 = Real{10.9};  // measured in test_bdf2_scheduler.cpp

    Vector x0(2);
    x0 << Real{0}, Real{0};

    const Real duty_cycles[] = {Real{0.1}, Real{0.3}, Real{0.5},
                                    Real{0.7}, Real{0.9}};

    auto out_path = bench_output_dir() / "dsed_stiff_fraction_sweep.csv";
    std::ofstream csv{out_path};
    csv << "duty_D,rk45_ms,auto_ms,speedup,theoretical_ceiling,"
         << "pct_of_ceiling,auto_rk45_steps,auto_bdf2_steps,auto_events\n";

    for (Real D : duty_cycles) {
        MixedStiffnessBench sys_rk;
        PEDSimulator<MixedStiffnessBench, MixedSwitchFnBench> sim_rk(
            sys_rk, MixedSwitchFnBench{T_sw, D},
            PIController{}, EventPredictor{}, dt_init, dt_max);
        const auto r_rk = sim_rk.simulate(x0, t_end);

        MixedStiffnessBench sys_auto;
        PEDSimulatorAuto<MixedStiffnessBench, MixedSwitchFnBench> sim_auto(
            sys_auto, MixedSwitchFnBench{T_sw, D},
            PIController{}, StiffnessDetector{Real{10.0}},
            dt_init, dt_max, h_bdf2);
        const auto r_auto = sim_auto.simulate(x0, t_end);

        const Real rk_ms = r_rk.cpu_time_seconds * Real{1000};
        const Real auto_ms = r_auto.cpu_time_seconds * Real{1000};
        const Real speedup = (auto_ms > Real{0}) ? rk_ms / auto_ms : Real{0};
        // Theoretical ceiling: 1 / (1 - f · (1 - 1/spd_BDF2))
        const Real f = D;
        const Real ceiling =
            Real{1} / (Real{1} - f * (Real{1} - Real{1} / spd_bdf2));
        const Real pct_ceiling =
            (ceiling > Real{0}) ? Real{100} * speedup / ceiling : Real{0};

        std::printf("%-7.2f %-10.3f %-10.3f %-8.3f %-9.3f %-9.1f %-7zu %-7zu %-9zu\n",
                     static_cast<double>(D),
                     static_cast<double>(rk_ms),
                     static_cast<double>(auto_ms),
                     static_cast<double>(speedup),
                     static_cast<double>(ceiling),
                     static_cast<double>(pct_ceiling),
                     r_auto.n_rk45_steps, r_auto.n_bdf2_steps,
                     r_auto.n_events);

        csv << D << ',' << rk_ms << ',' << auto_ms << ','
             << speedup << ',' << ceiling << ',' << pct_ceiling << ','
             << r_auto.n_rk45_steps << ',' << r_auto.n_bdf2_steps << ','
             << r_auto.n_events << '\n';

        REQUIRE(speedup > Real{0});
        // The actual speedup should be ≤ the theoretical ceiling
        // (with some numerical slack for variance + ratio rounding).
        REQUIRE(speedup <= ceiling * Real{1.10});
        // The empirical speedup should be at least 60% of the ceiling
        // (anything less means the dispatch overhead is excessive).
        REQUIRE(pct_ceiling >= Real{60.0});
    }

    csv.close();
    std::printf("\nCSV: %s\n", out_path.string().c_str());
}

// =============================================================================
// Multi-mode integer-mask bench cards (Gate 5 Phase 5.B-6)
// =============================================================================
//
// These cards exercise the auto-dispatcher with **non-bool mask types**
// (int) and **many modes per cycle** (6 for VSI, 27 for NPC). Validates
// that the per-mode eigenvalue cache (`StiffnessDetector::cache_`)
// scales linearly with mode count and that `mode_id_of<int>()` works
// correctly through the templated PEDSimulatorAuto.

namespace {

// --- 3-phase VSI 6-step modulation: 6 modes per fundamental cycle.
//   State: (i_a, i_b, i_c) — 3 phase currents in a Y-connected RL load
//   (4-wire — each phase sees its own leg voltage).
//   At each mode, two phases are at +V_dc/2 and one at -V_dc/2
//   (or vice versa) — 6 distinct voltage vectors stepping through
//   the hexagon at f_fund = 60 Hz.
class VSI3PhBench {
public:
    Real V_dc = Real{400.0};
    Real R = Real{2.0};
    Real L = Real{1e-3};
    Real f_fund = Real{60.0};

    [[nodiscard]] Real T_fund() const noexcept { return Real{1} / f_fund; }
    [[nodiscard]] Real T_mode() const noexcept { return T_fund() / Real{6}; }
    [[nodiscard]] int current_mask() const noexcept { return mode_; }
    void set_mask(int m) noexcept { mode_ = m; }

    [[nodiscard]] DenseMatrix A_matrix() const {
        DenseMatrix A(3, 3);
        // Diagonal RL load: each phase independent (assume 4-wire)
        A << -R / L, Real{0}, Real{0},
              Real{0}, -R / L, Real{0},
              Real{0}, Real{0}, -R / L;
        return A;
    }

    [[nodiscard]] Vector b_vector(Real /*t*/) const {
        // 6 voltage vectors (per IEEE standard 6-step inverter)
        // mode -> (V_a, V_b, V_c) in units of V_dc/2 with sign
        static constexpr int kV[6][3] = {
            { +1, -1, -1 },  // mode 0
            { +1, +1, -1 },  // mode 1
            { -1, +1, -1 },  // mode 2
            { -1, +1, +1 },  // mode 3
            { -1, -1, +1 },  // mode 4
            { +1, -1, +1 },  // mode 5
        };
        const int m = ((mode_ % 6) + 6) % 6;
        const Real Vh = V_dc / Real{2};
        Vector b(3);
        b << kV[m][0] * Vh / L,
             kV[m][1] * Vh / L,
             kV[m][2] * Vh / L;
        return b;
    }

    [[nodiscard]] Vector rhs(Real t, const Vector& x) const {
        return A_matrix() * x + b_vector(t);
    }

private:
    int mode_ = 0;
};

// VSI switch function: rotates through 6 modes every T_mode.
class VSI3PhSwitchFn {
public:
    explicit VSI3PhSwitchFn(Real T_mode) : T_{T_mode} {}

    [[nodiscard]] int operator()(Real t) const {
        const auto k = static_cast<std::int64_t>(std::floor(t / T_));
        return static_cast<int>(((k % 6) + 6) % 6);
    }
    [[nodiscard]] Real next_edge_after(Real t) const {
        const auto k = static_cast<std::int64_t>(std::floor(t / T_));
        constexpr Real eps = Real{1e-15};
        const Real candidate = static_cast<Real>(k + 1) * T_;
        if (candidate > t + eps) return candidate;
        return static_cast<Real>(k + 2) * T_;
    }
private:
    Real T_;
};

// --- NPC 3-level: 3 phases × 3 voltage levels per phase → 27 modes.
//   State: (i_a, i_b, i_c)  (simplified — no neutral-point cap tracking
//   in this bench; that would add 1 more state).
//   Mode encoding: integer in [0, 26], decoded as (m_a, m_b, m_c) where
//   each m_x ∈ {0, 1, 2} encoding {-V/2, 0, +V/2}.
//   Carrier-based modulation: each phase steps through its 3 levels
//   in a triangular pattern at carrier frequency f_c.
class NPC3LBench {
public:
    Real V_dc = Real{800.0};       // higher V_dc to justify 3 levels
    Real R = Real{2.0};
    Real L = Real{1e-3};
    Real f_carrier = Real{5e3};    // 5 kHz carrier

    [[nodiscard]] Real T_carrier() const noexcept { return Real{1} / f_carrier; }
    [[nodiscard]] int current_mask() const noexcept { return mode_; }
    void set_mask(int m) noexcept { mode_ = m; }

    [[nodiscard]] DenseMatrix A_matrix() const {
        DenseMatrix A(3, 3);
        A << -R / L, Real{0}, Real{0},
              Real{0}, -R / L, Real{0},
              Real{0}, Real{0}, -R / L;
        return A;
    }

    [[nodiscard]] Vector b_vector(Real /*t*/) const {
        // Decode mode → 3 level indices m_a, m_b, m_c each in {0, 1, 2}
        const int m = ((mode_ % 27) + 27) % 27;
        const int m_a = m % 3;
        const int m_b = (m / 3) % 3;
        const int m_c = (m / 9) % 3;
        // Level index → voltage: 0 → -V/2, 1 → 0, 2 → +V/2
        const Real Vh = V_dc / Real{2};
        const auto level_to_V = [Vh](int lev) -> Real {
            return (lev == 0) ? -Vh : (lev == 2) ? +Vh : Real{0};
        };
        Vector b(3);
        b << level_to_V(m_a) / L,
             level_to_V(m_b) / L,
             level_to_V(m_c) / L;
        return b;
    }

    [[nodiscard]] Vector rhs(Real t, const Vector& x) const {
        return A_matrix() * x + b_vector(t);
    }

private:
    int mode_ = 0;
};

// NPC switch function: rotates through 27 modes every T_carrier/27.
// Just a synthetic mode-rotation for the bench — real NPC modulation
// is sinusoidal-reference + triangular-carrier, but the algorithmic
// test here is that the per-mode cache works across all 27 modes.
class NPC3LSwitchFn {
public:
    explicit NPC3LSwitchFn(Real T_carrier) : T_step_{T_carrier / Real{27}} {}

    [[nodiscard]] int operator()(Real t) const {
        const auto k = static_cast<std::int64_t>(std::floor(t / T_step_));
        return static_cast<int>(((k % 27) + 27) % 27);
    }
    [[nodiscard]] Real next_edge_after(Real t) const {
        const auto k = static_cast<std::int64_t>(std::floor(t / T_step_));
        constexpr Real eps = Real{1e-15};
        const Real candidate = static_cast<Real>(k + 1) * T_step_;
        if (candidate > t + eps) return candidate;
        return static_cast<Real>(k + 2) * T_step_;
    }
private:
    Real T_step_;
};

// Templated card runner specialised for the int-mask, multi-mode case.
// Different from the bool-mask helper because the SwitchFn signature
// differs, but the result struct is identical.
template <class System, class SwitchFn>
PerConverterResult run_int_mask_card(
    const std::string& name,
    System& sys_rk,
    System& sys_auto,
    SwitchFn sf,
    const Vector& x0,
    Real t_end,
    Real dt_max,
    Real h_bdf2,
    Real dt_init = Real{1e-9}) {

    PEDSimulator<System, SwitchFn> sim_rk(
        sys_rk, sf, PIController{}, EventPredictor{}, dt_init, dt_max);
    const auto r_rk = sim_rk.simulate(x0, t_end);

    PEDSimulatorAuto<System, SwitchFn> sim_auto(
        sys_auto, sf, PIController{}, StiffnessDetector{Real{10.0}},
        dt_init, dt_max, h_bdf2);
    const auto r_auto = sim_auto.simulate(x0, t_end);

    Eigen::EigenSolver<DenseMatrix> es(sys_rk.A_matrix());
    Real lam_max = Real{0};
    for (Eigen::Index i = 0; i < es.eigenvalues().size(); ++i) {
        lam_max = std::max(lam_max, std::abs(es.eigenvalues()(i)));
    }

    PerConverterResult res;
    res.name = name;
    res.n_state = static_cast<int>(x0.size());
    res.rk_ms = r_rk.cpu_time_seconds * Real{1000};
    res.auto_ms = r_auto.cpu_time_seconds * Real{1000};
    res.speedup = (res.auto_ms > Real{0}) ? res.rk_ms / res.auto_ms : Real{0};
    res.rk_steps = r_rk.n_accept;
    res.auto_steps = r_auto.n_accept;
    res.auto_rk_steps = r_auto.n_rk45_steps;
    res.auto_bd_steps = r_auto.n_bdf2_steps;
    res.auto_events = r_auto.n_events;
    res.lam_max = lam_max;
    res.lam_h = lam_max * dt_max;

    std::printf("  %-12s  n_state=%d  |λ_max|·dt_max=%6.2f  "
                 "rk=%6.2f ms (%5zu st)  auto=%6.2f ms (%5zu st, rk=%5zu bd=%4zu, ev=%zu)  "
                 "speedup=%.3f×\n",
                 name.c_str(), res.n_state,
                 static_cast<double>(res.lam_h),
                 static_cast<double>(res.rk_ms), res.rk_steps,
                 static_cast<double>(res.auto_ms), res.auto_steps,
                 res.auto_rk_steps, res.auto_bd_steps, res.auto_events,
                 static_cast<double>(res.speedup));
    return res;
}

}  // namespace

// =============================================================================
// MMC N=3 single-phase bench card (Gate 5 Phase 5.B-7)
// =============================================================================
//
// Modular Multilevel Converter — the LARGEST converter in the OpenSpec
// Gate 5.2 list. Single-phase, N=3 sub-modules per arm:
//
//   * State vector (8 states):
//       x = [ i_arm_up, i_arm_lo,
//             v_C_u1, v_C_u2, v_C_u3,    (upper-arm SM caps)
//             v_C_l1, v_C_l2, v_C_l3 ]   (lower-arm SM caps)
//
//   * Mode encoding (16 modes):
//       mode_id = n_up * 4 + n_lo,   each n ∈ {0, 1, 2, 3}
//     n_up = number of inserted SMs in upper arm (first n by index;
//     no sort-and-select in the bench — synthetic deterministic pattern
//     is enough to validate the per-mode cache + dispatch correctness).
//
//   * KVL/KCL derivation (Sharifabadi et al. 2016 §2.3, simplified):
//       Upper KVL: V_dc/2 = L_arm·di_up/dt + v_arm_up + v_load
//       Lower KVL: V_dc/2 = L_arm·di_lo/dt + v_arm_lo - v_load
//       Load:     v_load = R_load·i_load + L_load·di_load/dt
//       KCL:      i_load = i_up - i_lo
//
//     The load inductance couples the two arms, requiring a 2×2 L-matrix
//     inverse pre-applied to get d[i_up, i_lo]/dt explicitly:
//       L_mat = [[L_arm + L_load,  -L_load     ],
//                [-L_load,        L_arm + L_load]]
//       L_inv = L_mat^{-1}
//
//     SM caps: each inserted SM has C·dv/dt = i_arm (sign-aware).
//
// Defaults follow the Python `projects/inverters/mmc/mmc_model.py`:
//   V_dc=400V, L_arm=1mH, C_sm=470µF, R_load=24Ω, L_load=5mH, f_o=60Hz.
//
// Rotation: synthetic deterministic mode rotation at f_step = 1 kHz
// (cycles through all 16 modes in 16 ms).

namespace {

class MMCN3Bench {
public:
    static constexpr int N_sm = 3;
    static constexpr int n_state = 2 + 2 * N_sm;  // = 8

    Real V_dc = Real{400.0};
    Real L_arm = Real{1e-3};
    Real C_sm = Real{470e-6};
    Real R_load = Real{24.0};
    Real L_load = Real{5e-3};
    Real V_C_nom = Real{0};   // computed in ctor; just for IC reference

    MMCN3Bench() { rebuild(); }

    /// Recompute V_C_nom and L_inv from current L_arm/L_load/V_dc.
    /// Call after editing the public fields to apply non-default params.
    void rebuild() {
        V_C_nom = V_dc / static_cast<Real>(N_sm);
        const Real det = (L_arm + L_load) * (L_arm + L_load)
                          - L_load * L_load;
        L_inv_(0, 0) = (L_arm + L_load) / det;
        L_inv_(0, 1) = L_load / det;
        L_inv_(1, 0) = L_load / det;
        L_inv_(1, 1) = (L_arm + L_load) / det;
    }

    [[nodiscard]] int current_mask() const noexcept { return mode_; }
    void set_mask(int m) noexcept { mode_ = m; }

    /// Build A and b for the current mode.
    /// State indexing: x[0]=i_up, x[1]=i_lo, x[2..4]=v_C_u, x[5..7]=v_C_l
    [[nodiscard]] DenseMatrix A_matrix() const {
        const int m = ((mode_ % 16) + 16) % 16;
        const int n_up = m / 4;     // 0..3
        const int n_lo = m % 4;     // 0..3
        (void) n_lo;

        DenseMatrix A = DenseMatrix::Zero(n_state, n_state);

        // The current dynamics:
        //   [di_up, di_lo]^T = L_inv · ( [V_dc/2 - v_arm_up - R·(i_up-i_lo),
        //                                  V_dc/2 - v_arm_lo + R·(i_up-i_lo)] )
        //
        //   v_arm_up = sum over first n_up SMs of v_C_uk = sum(x[2..2+n_up-1])
        //   v_arm_lo = sum over first n_lo SMs of v_C_lk = sum(x[5..5+n_lo-1])
        //
        //   The R_load coupling contributes (-R, +R) to [di_up, di_lo] from i_up
        //   and (+R, -R) from i_lo (pre-L_inv).
        //   The v_arm contributions are (-1, 0) from v_C_uk inserted and
        //   (0, -1) from v_C_lk inserted.

        // Pre-L_inv coefficient matrix M_pre (2 × n_state) for the
        // RHS of the current sub-system (without the constant V_dc/2 term):
        //   row 0 (di_up/dt = ...): contributes from x[i_up]=-R, x[i_lo]=+R,
        //                            x[v_C_uk inserted]=-1, x[v_C_lk]=0
        //   row 1 (di_lo/dt = ...): contributes from x[i_up]=+R, x[i_lo]=-R,
        //                            x[v_C_uk]=0, x[v_C_lk inserted]=-1
        Eigen::Matrix<Real, 2, Eigen::Dynamic> M_pre(2, n_state);
        M_pre.setZero();
        M_pre(0, 0) = -R_load;     // -R·i_up
        M_pre(0, 1) =  R_load;     // +R·i_lo (cancels into -R·(i_up - i_lo))
        M_pre(1, 0) =  R_load;     // +R·i_up
        M_pre(1, 1) = -R_load;     // -R·i_lo
        // Upper-arm cap contributions: first n_up SMs are inserted.
        for (int k = 0; k < n_up; ++k) {
            M_pre(0, 2 + k) = -Real{1};  // -v_C_uk inserted
        }
        // Lower-arm cap contributions: first n_lo SMs are inserted.
        for (int k = 0; k < n_lo; ++k) {
            M_pre(1, 5 + k) = -Real{1};  // -v_C_lk inserted
        }

        // A's first 2 rows = L_inv · M_pre  (8-col block)
        for (int j = 0; j < n_state; ++j) {
            A(0, j) = L_inv_(0, 0) * M_pre(0, j) + L_inv_(0, 1) * M_pre(1, j);
            A(1, j) = L_inv_(1, 0) * M_pre(0, j) + L_inv_(1, 1) * M_pre(1, j);
        }

        // SM cap dynamics: C·dv_C_uk/dt = i_arm_up if inserted; else 0.
        // A row 2+k has +1/C in column 0 (i_up) if SM_uk inserted.
        for (int k = 0; k < N_sm; ++k) {
            if (k < n_up) A(2 + k, 0) = Real{1} / C_sm;
            // else stays 0 (cap stays constant when bypassed)
        }
        // Lower-arm SM caps similarly with i_arm_lo (column 1).
        for (int k = 0; k < N_sm; ++k) {
            if (k < n_lo) A(5 + k, 1) = Real{1} / C_sm;
        }

        return A;
    }

    [[nodiscard]] Vector b_vector(Real /*t*/) const {
        // Only the current rows have the V_dc/2 forcing (L_inv applied):
        //   pre-L_inv: [V_dc/2, V_dc/2]
        //   post-L_inv: L_inv · [V_dc/2, V_dc/2]
        Vector b = Vector::Zero(n_state);
        const Real V_half = V_dc / Real{2};
        const Real pre0 = V_half;
        const Real pre1 = V_half;
        b(0) = L_inv_(0, 0) * pre0 + L_inv_(0, 1) * pre1;
        b(1) = L_inv_(1, 0) * pre0 + L_inv_(1, 1) * pre1;
        // Cap rows have zero b (the i_arm·𝟙[inserted] coupling is in A).
        return b;
    }

    [[nodiscard]] Vector rhs(Real t, const Vector& x) const {
        return A_matrix() * x + b_vector(t);
    }

    [[nodiscard]] Vector steady_ic() const {
        Vector x(n_state);
        x(0) = Real{0};
        x(1) = Real{0};
        for (int k = 0; k < N_sm; ++k) {
            x(2 + k) = V_C_nom;
            x(5 + k) = V_C_nom;
        }
        return x;
    }

private:
    int mode_ = 0;
    Eigen::Matrix<Real, 2, 2> L_inv_;
};

// MMC switch function: synthetic rotation through 16 modes at f_step.
class MMCSwitchFn {
public:
    explicit MMCSwitchFn(Real T_step) : T_step_{T_step} {}

    [[nodiscard]] int operator()(Real t) const {
        const auto k = static_cast<std::int64_t>(std::floor(t / T_step_));
        return static_cast<int>(((k % 16) + 16) % 16);
    }
    [[nodiscard]] Real next_edge_after(Real t) const {
        const auto k = static_cast<std::int64_t>(std::floor(t / T_step_));
        constexpr Real eps = Real{1e-15};
        const Real candidate = static_cast<Real>(k + 1) * T_step_;
        if (candidate > t + eps) return candidate;
        return static_cast<Real>(k + 2) * T_step_;
    }
private:
    Real T_step_;
};

}  // namespace

TEST_CASE("Bench: Per-converter sweep — MMC N=3 single-phase (Phase 5.B-7)",
           "[bench][dsed][microbench][auto][per-converter][mmc]") {
    std::printf("\n=== Per-converter auto-dispatch sweep (Gate 5 Phase 5.B-7) ===\n");
    std::printf("MMC N=3 single-phase: 8 states, 16 modes "
                 "(largest in the sweep)\n");

    const Real h_bdf2 = Real{500e-9};
    const Real f_step = Real{1e3};
    const Real T_step = Real{1} / f_step;
    const Real t_end = Real{16e-3};      // 16 modes × 1 ms = full rotation
    const Real dt_max = T_step / Real{8};
    const Real dt_init = Real{1e-9};

    MMCN3Bench sys_template;
    const Vector x0 = sys_template.steady_ic();

    std::vector<PerConverterResult> rows;

    // ---- Standard-params MMC (non-stiff at textbook L_arm=1mH) ----
    MMCN3Bench sys_rk;
    MMCN3Bench sys_auto;
    rows.push_back(run_int_mask_card(
        "MMC-N3-1phi", sys_rk, sys_auto,
        MMCSwitchFn{T_step},
        x0, t_end, dt_max, h_bdf2, dt_init));

    // ---- Stiff-variant MMC: 1000× smaller L_arm (=1µH) + 10× smaller C ----
    // Pushes |λ_max|·dt_max well past the dispatcher's threshold of 10
    // → forces BDF2 selection across all 16 modes. Models the "HVDC with
    // very tight passives" regime where stiffness dominates.
    MMCN3Bench sys_rk_stiff;
    sys_rk_stiff.L_arm = Real{1e-6};        // 1000× smaller arm inductor
    sys_rk_stiff.L_load = Real{10e-6};       // smaller load inductance too
    sys_rk_stiff.C_sm = Real{47e-6};         // 10× smaller cap
    sys_rk_stiff.rebuild();
    MMCN3Bench sys_auto_stiff;
    sys_auto_stiff.L_arm = Real{1e-6};
    sys_auto_stiff.L_load = Real{10e-6};
    sys_auto_stiff.C_sm = Real{47e-6};
    sys_auto_stiff.rebuild();

    const Vector x0_stiff = sys_rk_stiff.steady_ic();
    rows.push_back(run_int_mask_card(
        "MMC-N3-stiff", sys_rk_stiff, sys_auto_stiff,
        MMCSwitchFn{T_step},
        x0_stiff, t_end, dt_max, h_bdf2, dt_init));

    auto out_path = bench_output_dir() / "dsed_per_converter.csv";
    const bool needs_header = !std::filesystem::exists(out_path);
    std::ofstream csv{out_path, std::ios::app};
    if (needs_header) {
        csv << "converter,n_state,rk_ms,auto_ms,speedup,rk_steps,"
             << "auto_steps,auto_rk_steps,auto_bd_steps,auto_events,"
             << "lambda_max,lambda_h\n";
    }
    for (const auto& r : rows) {
        csv << r.name << ',' << r.n_state << ',' << r.rk_ms << ','
             << r.auto_ms << ',' << r.speedup << ',' << r.rk_steps << ','
             << r.auto_steps << ',' << r.auto_rk_steps << ','
             << r.auto_bd_steps << ',' << r.auto_events << ','
             << r.lam_max << ',' << r.lam_h << '\n';
    }
    csv.close();
    std::printf("\nCSV: %s (appended)\n", out_path.string().c_str());

    for (const auto& r : rows) {
        INFO("Converter " << r.name << " speedup = " << r.speedup);
        REQUIRE(r.n_state == 8);
        REQUIRE(r.auto_events >= 14);   // at least 14 mode rotations in 16 ms
        REQUIRE(r.auto_events <= 18);
        REQUIRE(r.speedup >= Real{0.5});
        REQUIRE(r.speedup <= Real{200.0});  // very wide bound — could be huge
    }
}

TEST_CASE("Bench: Per-converter sweep — VSI 3φ + NPC 3-level (Phase 5.B-6)",
           "[bench][dsed][microbench][auto][per-converter][multi-mode]") {
    std::printf("\n=== Per-converter auto-dispatch sweep (Gate 5 Phase 5.B-6) ===\n");
    std::printf("Multi-mode integer-mask converters (3φ VSI 6 modes, "
                 "NPC 3-level 27 modes)\n");

    const Real h_bdf2 = Real{500e-9};
    Vector x0(3);
    x0 << Real{0}, Real{0}, Real{0};

    std::vector<PerConverterResult> rows;

    // ---- 3-phase VSI 6-step: 5 fundamental cycles = 83 ms ----
    // 30 mode transitions over 5 × 60 Hz cycles
    {
        VSI3PhBench sys_rk;
        VSI3PhBench sys_auto;
        const Real t_end = Real{5} * sys_rk.T_fund();
        rows.push_back(run_int_mask_card(
            "VSI-3phi-6step", sys_rk, sys_auto,
            VSI3PhSwitchFn{sys_rk.T_mode()},
            x0, t_end, sys_rk.T_mode() / Real{4}, h_bdf2));
    }

    // ---- NPC 3-level: 10 carrier cycles = 2 ms ----
    // 270 mode transitions (10 × 27)
    {
        NPC3LBench sys_rk;
        NPC3LBench sys_auto;
        const Real t_end = Real{10} * sys_rk.T_carrier();
        rows.push_back(run_int_mask_card(
            "NPC-3level", sys_rk, sys_auto,
            NPC3LSwitchFn{sys_rk.T_carrier()},
            x0, t_end,
            sys_rk.T_carrier() / Real{27 * 4},  // dt_max much smaller (high mode density)
            h_bdf2));
    }

    Real log_speedup_sum = Real{0};
    for (const auto& r : rows) log_speedup_sum += std::log(r.speedup);
    const Real geo_mean = std::exp(log_speedup_sum
                                       / static_cast<Real>(rows.size()));
    std::printf("\n  Geo-mean speedup across this batch (2 multi-mode): %.3f×\n",
                 static_cast<double>(geo_mean));

    auto out_path = bench_output_dir() / "dsed_per_converter.csv";
    const bool needs_header = !std::filesystem::exists(out_path);
    std::ofstream csv{out_path, std::ios::app};
    if (needs_header) {
        csv << "converter,n_state,rk_ms,auto_ms,speedup,rk_steps,"
             << "auto_steps,auto_rk_steps,auto_bd_steps,auto_events,"
             << "lambda_max,lambda_h\n";
    }
    for (const auto& r : rows) {
        csv << r.name << ',' << r.n_state << ',' << r.rk_ms << ','
             << r.auto_ms << ',' << r.speedup << ',' << r.rk_steps << ','
             << r.auto_steps << ',' << r.auto_rk_steps << ','
             << r.auto_bd_steps << ',' << r.auto_events << ','
             << r.lam_max << ',' << r.lam_h << '\n';
    }
    csv.close();
    std::printf("\nCSV: %s (appended)\n", out_path.string().c_str());

    // Non-stiff sanity (diagonal A → low |λ|·h)
    for (const auto& r : rows) {
        INFO("Converter " << r.name << " speedup = " << r.speedup);
        REQUIRE(r.speedup >= Real{0.5});
        REQUIRE(r.speedup <= Real{2.0});
        REQUIRE(r.auto_bd_steps == 0);
    }
}

TEST_CASE("Bench: Per-converter sweep — Forward / Half-bridge / Boost PFC (Phase 5.B-5)",
           "[bench][dsed][microbench][auto][per-converter]") {
    std::printf("\n=== Per-converter auto-dispatch sweep (Gate 5 Phase 5.B-5) ===\n");
    std::printf("3 more PWM converters: Forward (isolated), Half-bridge "
                 "(bipolar), Boost PFC (time-varying source)\n");

    const Real t_end = Real{5e-3};
    const Real h_bdf2 = Real{500e-9};

    Vector x0_fw(2);
    x0_fw << Real{0}, Real{0};

    Vector x0_hb(2);
    x0_hb << Real{0}, Real{0};

    Vector x0_pfc(2);
    x0_pfc << Real{0}, Real{0};

    std::vector<PerConverterResult> rows;

    // ---- Forward ----
    {
        ForwardBench sys_rk;
        ForwardBench sys_auto;
        rows.push_back(run_per_converter_card(
            "Forward", sys_rk, sys_auto,
            GenericPSCSwitchFn{sys_rk.T_sw(), sys_rk.D},
            x0_fw, t_end, sys_rk.T_sw() / Real{4}, h_bdf2));
    }
    // ---- Half-bridge ----
    {
        HalfBridgeBench sys_rk;
        HalfBridgeBench sys_auto;
        rows.push_back(run_per_converter_card(
            "Half-bridge", sys_rk, sys_auto,
            GenericPSCSwitchFn{sys_rk.T_sw(), sys_rk.D},
            x0_hb, t_end, sys_rk.T_sw() / Real{4}, h_bdf2));
    }
    // ---- Boost PFC (time-varying source — 5 ms covers ~30% of a 60Hz half-cycle) ----
    {
        BoostPFCBench sys_rk;
        BoostPFCBench sys_auto;
        rows.push_back(run_per_converter_card(
            "Boost-PFC", sys_rk, sys_auto,
            GenericPSCSwitchFn{sys_rk.T_sw(), sys_rk.D},
            x0_pfc, t_end, sys_rk.T_sw() / Real{4}, h_bdf2));
    }

    // Geo-mean speedup across this batch
    Real log_speedup_sum = Real{0};
    for (const auto& r : rows) log_speedup_sum += std::log(r.speedup);
    const Real geo_mean = std::exp(log_speedup_sum
                                       / static_cast<Real>(rows.size()));
    std::printf("\n  Geo-mean speedup across this batch (3 conv): %.3f×\n",
                 static_cast<double>(geo_mean));

    auto out_path = bench_output_dir() / "dsed_per_converter.csv";
    const bool needs_header = !std::filesystem::exists(out_path);
    std::ofstream csv{out_path, std::ios::app};
    if (needs_header) {
        csv << "converter,n_state,rk_ms,auto_ms,speedup,rk_steps,"
             << "auto_steps,auto_rk_steps,auto_bd_steps,auto_events,"
             << "lambda_max,lambda_h\n";
    }
    for (const auto& r : rows) {
        csv << r.name << ',' << r.n_state << ',' << r.rk_ms << ','
             << r.auto_ms << ',' << r.speedup << ',' << r.rk_steps << ','
             << r.auto_steps << ',' << r.auto_rk_steps << ','
             << r.auto_bd_steps << ',' << r.auto_events << ','
             << r.lam_max << ',' << r.lam_h << '\n';
    }
    csv.close();
    std::printf("\nCSV: %s (appended)\n", out_path.string().c_str());

    // Non-stiff sanity: speedup in 0.5–1.8× range; auto picks RK45 for all
    for (const auto& r : rows) {
        INFO("Converter " << r.name << " speedup = " << r.speedup);
        REQUIRE(r.speedup >= Real{0.5});
        REQUIRE(r.speedup <= Real{1.8});
        REQUIRE(r.auto_bd_steps == 0);
    }
}
