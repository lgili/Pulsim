#pragma once

#include "pulsim/v1/components/base.hpp"

namespace pulsim::v1 {

// =============================================================================
// Resistor Device (CRTP)
// =============================================================================
//
// AD bypass note (`add-automatic-differentiation`, Phase 3):
//   Resistor's Jacobian contribution is the constant conductance G = 1/R.
//   AD's value here is to derive ∂i/∂x for *non-linear* devices where a
//   manual closed-form would be tedious or error-prone. For a constant-G
//   element, AD evaluation buys nothing while paying the cost of an ADReal
//   pass; we therefore intentionally do **not** expose `stamp_jacobian_via_ad`
//   on linear devices. The `stamp_impl` direct-stamp path is the canonical
//   route, and the build flag `PULSIM_USE_AD_STAMP` only retargets the
//   nonlinear devices.

class Resistor : public LinearDeviceBase<Resistor> {
public:
    using Base = LinearDeviceBase<Resistor>;
    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::Resistor);

    /// Parameter structure for Resistor
    struct Params {
        Scalar resistance = 1000.0;

        // -------- Loss + thermal binding (Phase 3 of
        //          inverter-bridge-losses, Pulsim 0.10.0a10) --------
        // Loss model: P = I² · R(T_j)
        //     R(T_j) = R_25 · (1 + TCR · (T_j − T_ref))
        //
        // Default R_th_ja = 0 disables the loss accumulator entirely —
        // existing resistor uses see no change (legacy mode preserved).
        Scalar TCR       = 0.0;       // 1/K (positive for typical metal-film)
        Scalar T_ref     = 25.0;      // °C
        Scalar R_th_ja   = 0.0;       // K/W (0 = disabled)
        Scalar T_amb     = 25.0;      // °C
    };

    explicit Resistor(Scalar resistance, std::string name = "")
        : Base(std::move(name)), resistance_(resistance) {}

    /// Construct from a full Params struct so callers can configure
    /// the thermal binding (Pulsim 0.10.0a10). Backward-compatible
    /// when `params.R_th_ja == 0`.
    explicit Resistor(const Params& params, std::string name = "")
        : Base(std::move(name))
        , resistance_(params.resistance)
        , TCR_(params.TCR)
        , T_ref_(params.T_ref)
        , R_th_ja_(params.R_th_ja)
        , T_amb_(params.T_amb)
        , T_j_(params.T_amb) {}

    /// Stamp implementation (called via CRTP)
    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& /*b*/, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2) return;

        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];
        const Scalar g = 1.0 / resistance_;

        // Stamp conductance matrix
        if (n_plus >= 0) {
            G.coeffRef(n_plus, n_plus) += g;
            if (n_minus >= 0) {
                G.coeffRef(n_plus, n_minus) -= g;
            }
        }
        if (n_minus >= 0) {
            G.coeffRef(n_minus, n_minus) += g;
            if (n_plus >= 0) {
                G.coeffRef(n_minus, n_plus) -= g;
            }
        }
    }

    /// Jacobian pattern (compile-time)
    static constexpr auto jacobian_pattern_impl() {
        // Resistor contributes to 4 positions: (n+,n+), (n+,n-), (n-,n+), (n-,n-)
        return StaticSparsityPattern<4>{{
            JacobianEntry{0, 0},
            JacobianEntry{0, 1},
            JacobianEntry{1, 0},
            JacobianEntry{1, 1}
        }};
    }

    [[nodiscard]] Scalar resistance() const { return resistance_; }
    void set_resistance(Scalar r) { resistance_ = r; }

    // -------- Loss + thermal accessors (Phase 3) --------
    [[nodiscard]] Scalar TCR()       const noexcept { return TCR_; }
    [[nodiscard]] Scalar T_ref()     const noexcept { return T_ref_; }
    [[nodiscard]] Scalar R_th_ja()   const noexcept { return R_th_ja_; }
    [[nodiscard]] Scalar T_amb()     const noexcept { return T_amb_; }
    void set_TCR(Scalar tcr)       noexcept { TCR_ = tcr; }
    void set_R_th_ja(Scalar r)     noexcept { R_th_ja_ = r; }
    void set_T_amb(Scalar t)       noexcept { T_amb_ = t; T_j_ = t; }
    void set_T_j_init(Scalar t)    noexcept { T_j_ = t; }

    /// Temperature-corrected resistance at the device's current T_j.
    [[nodiscard]] Scalar R_at_Tj() const noexcept {
        return resistance_ * (Scalar{1} + TCR_ * (T_j_ - T_ref_));
    }

    [[nodiscard]] Scalar total_energy()   const noexcept { return e_cond_; }
    [[nodiscard]] Scalar peak_power()     const noexcept { return p_peak_; }
    [[nodiscard]] Scalar last_power()     const noexcept { return p_last_; }
    [[nodiscard]] Scalar last_current()   const noexcept { return i_last_; }
    [[nodiscard]] Scalar last_voltage()   const noexcept { return v_last_; }
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
        v_last_ = 0.0;
    }

    /// Sample I² · R(T_j) over the past `dt` seconds.
    /// No-op when R_th_ja == 0 (legacy preserved).
    void accumulate_loss(Scalar v_resistor, Scalar dt) noexcept {
        if (dt < Scalar{0}) return;
        const Scalar i = (resistance_ > Scalar{0})
            ? v_resistor / resistance_ : Scalar{0};
        v_last_ = v_resistor;
        i_last_ = i;
        if (R_th_ja_ <= Scalar{0}) {
            p_last_ = Scalar{0};
            return;
        }
        const Scalar R_eff = R_at_Tj();
        const Scalar p = i * i * R_eff;
        p_last_ = p;
        if (p > Scalar{0}) {
            e_cond_ += p * dt;
            if (p > p_peak_) p_peak_ = p;
        }
        t_sim_ += dt;
    }

private:
    Scalar resistance_;
    // Thermal + loss accumulator state (Phase 3).
    Scalar TCR_      = 0.0;
    Scalar T_ref_    = 25.0;
    Scalar R_th_ja_  = 0.0;
    Scalar T_amb_    = 25.0;
    Scalar e_cond_ = 0.0;
    Scalar p_peak_ = 0.0;
    Scalar p_last_ = 0.0;
    Scalar t_sim_  = 0.0;
    Scalar i_last_ = 0.0;
    Scalar v_last_ = 0.0;
    Scalar T_j_    = 25.0;
};

// Specialization of device_traits for Resistor
template<>
struct device_traits<Resistor> {
    static constexpr DeviceType type = DeviceType::Resistor;
    static constexpr std::size_t num_pins = 2;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = true;
    static constexpr bool is_dynamic = false;
    static constexpr bool has_loss_model = true;  // Conduction loss = I²R
    static constexpr bool has_thermal_model = false;
    static constexpr std::size_t jacobian_size = 4;  // 2x2 contribution
};

}  // namespace pulsim::v1
