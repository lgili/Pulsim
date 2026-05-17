#pragma once

#include "pulsim/v1/components/base.hpp"

namespace pulsim::v1 {

// =============================================================================
// Inductor Device (CRTP with dynamics)
// =============================================================================
//
// AD bypass note (`add-automatic-differentiation`, Phase 3):
//   Inductor stamps the trapezoidal companion model `G_eq = dt/(2L)` plus a
//   constant equivalent-current source from the previous step's state.
//   Jacobian is constant per topology / per timestep — no AD needed.
//   `stamp_jacobian_via_ad` is intentionally not provided.

class Inductor : public DynamicDeviceBase<Inductor> {
public:
    using Base = DynamicDeviceBase<Inductor>;
    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::Inductor);

    struct Params {
        Scalar inductance = 1e-3;
        Scalar initial_current = 0.0;

        // -------- Loss + thermal binding (Phase 3 of
        //          inverter-bridge-losses, Pulsim 0.10.0a11) --------
        // Copper-loss model: P_cu = I² · DCR(T_j)
        //     DCR(T_j) = DCR_25 · (1 + DCR_tc · (T_j − T_ref))
        //
        // Core loss is not yet modelled here — that requires
        // Steinmetz coefficients (k, α, β) and the core volume + the
        // instantaneous flux density. The `magnetic/bh_curve.hpp`
        // helpers expose Steinmetz, so a future iteration can plug
        // those in as a separate accumulator term.
        //
        // Default R_th_ja = 0 disables the loss accumulator entirely.
        Scalar DCR       = 0.0;       // Ω at T_ref (DC winding resistance)
        Scalar DCR_tc    = 3.9e-3;    // 1/K (copper)
        Scalar T_ref     = 25.0;      // °C
        Scalar R_th_ja   = 0.0;       // K/W (0 = disabled)
        Scalar T_amb     = 25.0;      // °C
    };

    explicit Inductor(Scalar inductance, Scalar initial_current = 0.0, std::string name = "")
        : Base(std::move(name))
        , inductance_(inductance)
        , i_prev_(initial_current)
        , v_prev_(0.0) {}

    /// Construct from a fully-specified Params struct (Pulsim 0.10.0a11
    /// entry-point for DCR + thermal). Backward-compat when R_th_ja=0.
    explicit Inductor(const Params& params, std::string name = "")
        : Base(std::move(name))
        , inductance_(params.inductance)
        , i_prev_(params.initial_current)
        , v_prev_(0.0)
        , DCR_(params.DCR)
        , DCR_tc_(params.DCR_tc)
        , T_ref_(params.T_ref)
        , R_th_ja_(params.R_th_ja)
        , T_amb_(params.T_amb)
        , T_j_(params.T_amb) {}

    /// Stamp implementation using Trapezoidal companion model
    /// For inductor: V = L * dI/dt
    /// Trapezoidal: V_n = (2L/dt) * I_n - (2L/dt) * I_{n-1} - V_{n-1}
    /// Companion model is a resistor R_eq = 2L/dt in series with voltage V_eq
    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& b, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2) return;

        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];

        // Trapezoidal: R_eq = 2L/dt (conductance g_eq = dt/(2L))
        const Scalar g_eq = dt_ / (2.0 * inductance_);

        // Equivalent voltage: V_eq = (2L/dt) * I_{n-1} + V_{n-1}
        const Scalar v_eq = (2.0 * inductance_ / dt_) * i_prev_ + v_prev_;

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

        // Stamp equivalent current source (from V_eq through g_eq)
        const Scalar i_eq = g_eq * v_eq;
        if (n_plus >= 0) {
            b[n_plus] += i_eq;
        }
        if (n_minus >= 0) {
            b[n_minus] -= i_eq;
        }
    }

    void update_history_impl() {
        i_prev_ = i_current_;
        v_prev_ = v_current_;
        history_initialized_ = true;
    }

    void set_current_state(Scalar v, Scalar i) {
        v_current_ = v;
        i_current_ = i;
    }

    /// Whether the inductor has stored a real previous-step voltage
    /// (i.e. not just the constructor's default `v_prev_=0`). Used by
    /// the runtime to pick BDF1 for the first step from initial
    /// conditions, avoiding the trapezoidal startup ringing where
    /// `v_n = g_eq·i_n - v_{n-1}` doubles the inductor voltage when
    /// `v_{n-1}` defaults to 0 instead of the analytical t=0+ value.
    [[nodiscard]] bool history_initialized() const { return history_initialized_; }
    void mark_history_initialized() { history_initialized_ = true; }
    void reset_history() { history_initialized_ = false; }

    static constexpr auto jacobian_pattern_impl() {
        return StaticSparsityPattern<4>{{
            JacobianEntry{0, 0},
            JacobianEntry{0, 1},
            JacobianEntry{1, 0},
            JacobianEntry{1, 1}
        }};
    }

    [[nodiscard]] Scalar inductance() const { return inductance_; }
    [[nodiscard]] Scalar current_prev() const { return i_prev_; }
    [[nodiscard]] Scalar voltage_prev() const { return v_prev_; }

    // -------- Loss + thermal accessors (Phase 3) --------
    [[nodiscard]] Scalar DCR()       const noexcept { return DCR_; }
    [[nodiscard]] Scalar DCR_tc()    const noexcept { return DCR_tc_; }
    [[nodiscard]] Scalar T_ref()     const noexcept { return T_ref_; }
    [[nodiscard]] Scalar R_th_ja()   const noexcept { return R_th_ja_; }
    [[nodiscard]] Scalar T_amb()     const noexcept { return T_amb_; }
    void set_DCR(Scalar r)         noexcept { DCR_ = r; }
    void set_R_th_ja(Scalar r)     noexcept { R_th_ja_ = r; }
    void set_T_amb(Scalar t)       noexcept { T_amb_ = t; T_j_ = t; }
    void set_T_j_init(Scalar t)    noexcept { T_j_ = t; }

    [[nodiscard]] Scalar DCR_at_Tj() const noexcept {
        return DCR_ * (Scalar{1} + DCR_tc_ * (T_j_ - T_ref_));
    }

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

    void reset_loss() noexcept {
        e_cond_ = 0.0;
        p_peak_ = 0.0;
        p_last_ = 0.0;
        t_sim_ = 0.0;
        i_last_ = 0.0;
    }

    /// Sample I² · DCR(T_j) over the past `dt` seconds. Same convention
    /// as the cap/resistor — no-op when R_th_ja == 0 (legacy preserved).
    void accumulate_loss(Scalar i_branch, Scalar dt) noexcept {
        if (dt < Scalar{0}) return;
        i_last_ = i_branch;
        if (R_th_ja_ <= Scalar{0}) {
            p_last_ = Scalar{0};
            return;
        }
        const Scalar DCR_eff = DCR_at_Tj();
        const Scalar p = i_branch * i_branch * DCR_eff;
        p_last_ = p;
        if (p > Scalar{0}) {
            e_cond_ += p * dt;
            if (p > p_peak_) p_peak_ = p;
        }
        t_sim_ += dt;
    }

private:
    Scalar inductance_;
    Scalar i_prev_;
    Scalar v_prev_;
    Scalar i_current_ = 0.0;
    Scalar v_current_ = 0.0;
    bool history_initialized_ = false;  // True once the first accepted
                                         // step's current/voltage are
                                         // stored in i_prev_/v_prev_.

    // DCR + thermal + loss accumulator (Phase 3).
    Scalar DCR_      = 0.0;
    Scalar DCR_tc_   = 3.9e-3;
    Scalar T_ref_    = 25.0;
    Scalar R_th_ja_  = 0.0;
    Scalar T_amb_    = 25.0;
    Scalar e_cond_ = 0.0;
    Scalar p_peak_ = 0.0;
    Scalar p_last_ = 0.0;
    Scalar t_sim_  = 0.0;
    Scalar i_last_ = 0.0;
    Scalar T_j_    = 25.0;
};

template<>
struct device_traits<Inductor> {
    static constexpr DeviceType type = DeviceType::Inductor;
    static constexpr std::size_t num_pins = 2;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = true;
    static constexpr bool is_dynamic = true;
    static constexpr bool has_loss_model = false;
    static constexpr bool has_thermal_model = false;
    static constexpr std::size_t jacobian_size = 4;
};

}  // namespace pulsim::v1
