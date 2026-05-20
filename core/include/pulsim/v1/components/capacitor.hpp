#pragma once

#include "pulsim/v1/components/base.hpp"

namespace pulsim::v1 {

// =============================================================================
// Capacitor Device (CRTP with dynamics)
// =============================================================================
//
// AD bypass note (`add-automatic-differentiation`, Phase 3):
//   Capacitor stamps the trapezoidal companion model `G_eq = 2C/dt` plus a
//   constant equivalent-current source `I_eq` derived from the previous
//   step's state. The Jacobian is constant per topology / per timestep —
//   no AD needed. `stamp_jacobian_via_ad` is intentionally not provided.

class Capacitor : public DynamicDeviceBase<Capacitor> {
public:
    using Base = DynamicDeviceBase<Capacitor>;
    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::Capacitor);

    /// Parameter structure for Capacitor
    struct Params {
        Scalar capacitance = 1e-6;
        Scalar initial_voltage = 0.0;

        // -------- ESR loss + thermal binding (Phase 3 of
        //          inverter-bridge-losses, Pulsim 0.10.0a9) --------
        // Equivalent-series resistance loss model:
        //     P_loss(t) = I_cap(t)² · ESR(T_j)
        //     ESR(T_j)  = ESR_25 · (1 + ESR_tc · (T_j − T_ref))
        //
        // ESR is NOT included in the trapezoidal stamping itself —
        // the cap's electrical dynamics stay the same as the ideal
        // model. ESR enters only through the loss accumulator (i_cap²·ESR
        // integrated each accepted step). This matches what `powerStage`
        // does and keeps the simulation lean — for moderately small
        // ESR (~tens of mΩ) the impact on bus dynamics is negligible.
        //
        // Backward-compat: defaults R_th_ja = 0 → accumulator disabled,
        // device behaves exactly as before the upgrade.
        Scalar ESR       = 0.0;       // Ω at T_ref
        Scalar ESR_tc    = 0.0;       // 1/K (typically positive for AlElCap)
        Scalar T_ref     = 25.0;      // °C
        Scalar R_th_ja   = 0.0;       // K/W (0 = disabled, legacy mode)
        Scalar T_amb     = 25.0;      // °C
    };

    explicit Capacitor(Scalar capacitance, Scalar initial_voltage = 0.0, std::string name = "")
        : Base(std::move(name))
        , capacitance_(capacitance)
        , v_prev_(initial_voltage)
        , i_prev_(0.0)
        , history_initialized_(false) {}

    /// Construct from a fully-specified Params struct (Phase 3 entry-point
    /// for realistic ESR + thermal models). Backward-compatible when
    /// `params.R_th_ja == 0`.
    explicit Capacitor(const Params& params, std::string name = "")
        : Base(std::move(name))
        , capacitance_(params.capacitance)
        , v_prev_(params.initial_voltage)
        , i_prev_(0.0)
        , history_initialized_(false)
        , ESR_(params.ESR)
        , ESR_tc_(params.ESR_tc)
        , T_ref_(params.T_ref)
        , R_th_ja_(params.R_th_ja)
        , T_amb_(params.T_amb)
        , T_j_(params.T_amb) {}

    /// Stamp implementation using Trapezoidal companion model
    /// Correct formula: I_eq = (2C/dt) * V_n - (2C/dt) * V_{n-1} - I_{n-1}
    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& b, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2) return;

        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];

        // Trapezoidal: G_eq = 2C/dt
        const Scalar g_eq = 2.0 * capacitance_ / dt_;

        // Equivalent current source: I_eq = G_eq * V_{n-1} + I_{n-1}
        const Scalar i_eq = g_eq * v_prev_ + i_prev_;

        // Stamp conductance
        if (n_plus >= 0) {
            G.coeffRef(n_plus, n_plus) += g_eq;
            if (n_minus >= 0) {
                G.coeffRef(n_plus, n_minus) -= g_eq;
            }
        }
        if (n_minus >= 0) {
            G.coeffRef(n_minus, n_minus) += g_eq;
            if (n_plus >= 0) {
                G.coeffRef(n_minus, n_plus) -= g_eq;
            }
        }

        // Stamp equivalent current source
        if (n_plus >= 0) {
            b[n_plus] += i_eq;
        }
        if (n_minus >= 0) {
            b[n_minus] -= i_eq;
        }
    }

    /// Update history after accepted timestep
    void update_history_impl() {
        // Store current values for next step
        // Note: v_current and i_current must be set by the solver after each step
        v_prev_ = v_current_;
        i_prev_ = i_current_;
        history_initialized_ = true;
    }

    /// Whether the capacitor has stored a real previous-step current
    /// (i.e. not just the constructor's default `i_prev_=0`). Used by
    /// the runtime to pick BDF1 for the first step from initial
    /// conditions, avoiding the trapezoidal startup ringing where
    /// `i_n = g_eq·v_n - i_{n-1}` doubles the cap current when
    /// `i_{n-1}` defaults to 0 instead of the analytical t=0+ value.
    [[nodiscard]] bool history_initialized() const { return history_initialized_; }
    void mark_history_initialized() { history_initialized_ = true; }
    void reset_history() { history_initialized_ = false; }

    /// Set current state (called by solver after each Newton iteration)
    void set_current_state(Scalar v, Scalar i) {
        v_current_ = v;
        i_current_ = i;
    }

    static constexpr auto jacobian_pattern_impl() {
        return StaticSparsityPattern<4>{{
            JacobianEntry{0, 0},
            JacobianEntry{0, 1},
            JacobianEntry{1, 0},
            JacobianEntry{1, 1}
        }};
    }

    [[nodiscard]] Scalar capacitance() const { return capacitance_; }
    [[nodiscard]] Scalar voltage_prev() const { return v_prev_; }
    [[nodiscard]] Scalar current_prev() const { return i_prev_; }

    // -------- Realistic-loss + thermal accessors (Phase 3) --------
    [[nodiscard]] Scalar ESR()       const noexcept { return ESR_; }
    [[nodiscard]] Scalar ESR_tc()    const noexcept { return ESR_tc_; }
    [[nodiscard]] Scalar T_ref()     const noexcept { return T_ref_; }
    [[nodiscard]] Scalar R_th_ja()   const noexcept { return R_th_ja_; }
    [[nodiscard]] Scalar T_amb()     const noexcept { return T_amb_; }
    void set_ESR(Scalar r)        noexcept { ESR_ = r; }
    void set_ESR_tc(Scalar tc)    noexcept { ESR_tc_ = tc; }
    void set_R_th_ja(Scalar r)    noexcept { R_th_ja_ = r; }
    void set_T_amb(Scalar t)      noexcept { T_amb_ = t; T_j_ = t; }
    void set_T_j_init(Scalar t)   noexcept { T_j_ = t; }

    /// Temperature-corrected ESR at the device's current T_j.
    [[nodiscard]] Scalar ESR_at_Tj() const noexcept {
        return ESR_ * (Scalar{1} + ESR_tc_ * (T_j_ - T_ref_));
    }

    // Loss accumulator accessors.
    [[nodiscard]] Scalar total_energy()   const noexcept { return e_cond_; }
    [[nodiscard]] Scalar peak_power()     const noexcept { return p_peak_; }
    [[nodiscard]] Scalar last_power()     const noexcept { return p_last_; }
    [[nodiscard]] Scalar last_current()   const noexcept { return i_last_; }
    [[nodiscard]] Scalar conduction_time() const noexcept { return t_sim_; }
    [[nodiscard]] Scalar junction_temperature() const noexcept { return T_j_; }
    [[nodiscard]] Scalar steady_state_junction_temperature() const noexcept {
        return T_amb_ + average_power() * R_th_ja_;
    }
    [[nodiscard]] Scalar average_power() const noexcept {
        return (t_sim_ > Scalar{0}) ? e_cond_ / t_sim_ : Scalar{0};
    }

    /// Zero the loss accumulator. Does NOT touch T_j — same convention
    /// as the diode/MOSFET/IGBT reset_loss().
    void reset_loss() noexcept {
        e_cond_ = 0.0;
        p_peak_ = 0.0;
        p_last_ = 0.0;
        t_sim_ = 0.0;
        i_last_ = 0.0;
    }

    /// Always tracks i_cap² · ESR(T_j) over the past `dt` seconds.
    /// R_th_ja > 0 enables T_j-corrected ESR; otherwise the static
    /// `ESR_` is used so the device's loss accessor and
    /// `SystemLossSummary` stay consistent. With ESR == 0 (default)
    /// the integral is identically zero (ideal cap, legacy behavior).
    void accumulate_loss(Scalar i_cap, Scalar dt) noexcept {
        if (dt < Scalar{0}) return;
        i_last_ = i_cap;
        const Scalar esr = (R_th_ja_ > Scalar{0}) ? ESR_at_Tj() : ESR_;
        const Scalar p = i_cap * i_cap * esr;
        // Trapezoidal time-averaging (see components/mosfet.hpp).
        const Scalar p_avg = (t_sim_ > Scalar{0})
                           ? Scalar{0.5} * (p_last_ + p)
                           : p;
        p_last_ = p;
        if (p_avg > Scalar{0}) {
            e_cond_ += p_avg * dt;
            if (p > p_peak_) p_peak_ = p;
        }
        t_sim_ += dt;
    }

private:
    Scalar capacitance_;
    Scalar v_prev_;      // Voltage at previous timestep
    Scalar i_prev_;      // Current at previous timestep (CRITICAL for Trapezoidal!)
    Scalar v_current_ = 0.0;  // Current voltage (set by solver)
    Scalar i_current_ = 0.0;  // Current current (set by solver)
    bool history_initialized_ = false;  // True once the first accepted
                                         // step's current/voltage are
                                         // stored in i_prev_/v_prev_.

    // ESR loss + thermal binding (Phase 3, default values disable
    // the loss accumulator → legacy behaviour preserved).
    Scalar ESR_      = 0.0;
    Scalar ESR_tc_   = 0.0;
    Scalar T_ref_    = 25.0;
    Scalar R_th_ja_  = 0.0;
    Scalar T_amb_    = 25.0;
    // Loss accumulator state.
    Scalar e_cond_ = 0.0;
    Scalar p_peak_ = 0.0;
    Scalar p_last_ = 0.0;
    Scalar t_sim_  = 0.0;
    Scalar i_last_ = 0.0;
    Scalar T_j_    = 25.0;
};

template<>
struct device_traits<Capacitor> {
    static constexpr DeviceType type = DeviceType::Capacitor;
    static constexpr std::size_t num_pins = 2;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = true;
    static constexpr bool is_dynamic = true;
    // close-electrothermal-loop-and-promote-thermal-traits Phase 2.4:
    // Capacitor exposes ESR + ESR_at_Tj() (ripple-current loss),
    // R_th_ja, T_amb, T_j_, set_T_j_init, junction_temperature(). The
    // accumulator at runtime_circuit.hpp:4786-4788 already integrates
    // I²·ESR_at_Tj() per step — now ESR_at_Tj() sees T_j(t).
    static constexpr bool has_loss_model = true;
    static constexpr bool has_thermal_model = true;
    static constexpr std::size_t jacobian_size = 4;
};

}  // namespace pulsim::v1
