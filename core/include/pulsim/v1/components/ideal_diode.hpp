#pragma once

#include "pulsim/v1/components/base.hpp"
#include "pulsim/v1/ad/ad_scalar.hpp"

#include <cmath>

namespace pulsim::v1 {

// =============================================================================
// Ideal Diode (CRTP - Nonlinear, supports SwitchingMode::Ideal)
// =============================================================================
//
// Two stamping modes (selectable via SwitchingMode):
//
//  - Behavioral: legacy tanh-smoothed conductance, suitable for Newton
//                iteration even very close to the conduction threshold.
//
//  - Ideal     : sharp piecewise-linear stamp using exactly g_on or g_off
//                per the current state. No tanh, no derivative-of-conductance
//                term, no Newton iteration required when topology is stable.
//                The state is mutated only via commit_pwl_state(); the kernel
//                queries should_commute() to detect events.
//
// Device state (`pwl_state_`) is a first-class member, replacing the legacy
// `mutable bool is_on_` smell. Stamping methods are const-correct: they read
// state but do not mutate it.

class IdealDiode : public NonlinearDeviceBase<IdealDiode> {
public:
    using Base = NonlinearDeviceBase<IdealDiode>;
    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::Diode);

    struct Params {
        Scalar g_on = 1e3;          ///< On-state conductance (1/R_on)
        Scalar g_off = 1e-9;        ///< Off-state conductance (leakage)
        Scalar v_threshold = 0.0;   ///< Threshold voltage (0 for ideal)
        Scalar v_smooth = 0.1;      ///< Smoothing voltage for Behavioral mode (0 = sharp)
    };

    explicit IdealDiode(Scalar g_on = 1e3,
                        Scalar g_off = 1e-9,
                        std::string name = "")
        : Base(std::move(name))
        , g_on_(g_on)
        , g_off_(g_off) {}

    // --- Smoothing controls (Behavioral mode only) -----------------------------
    void set_smoothing(Scalar v_smooth) { v_smooth_ = v_smooth; }
    [[nodiscard]] Scalar smoothing() const { return v_smooth_; }
    [[nodiscard]] Scalar g_on() const { return g_on_; }
    [[nodiscard]] Scalar g_off() const { return g_off_; }

    // --- SwitchingMode contract -----------------------------------------------
    [[nodiscard]] SwitchingMode switching_mode() const noexcept { return mode_; }
    void set_switching_mode(SwitchingMode mode) noexcept { mode_ = mode; }

    /// Hysteresis band (volts) that suppresses commute chatter near
    /// the conduction threshold. Default 1e-9 V.
    [[nodiscard]] Scalar event_hysteresis() const noexcept { return event_hysteresis_; }
    void set_event_hysteresis(Scalar h) noexcept { event_hysteresis_ = h; }

    // --- PWL two-state contract -----------------------------------------------
    /// Current PWL on/off state (true = conducting).
    [[nodiscard]] bool pwl_state() const noexcept { return pwl_state_; }
    /// Commit the device to a new PWL state. Mutates state explicitly; never
    /// invoked from a stamping path.
    void commit_pwl_state(bool conducting) noexcept { pwl_state_ = conducting; }

    /// Predicate consulted by the event scheduler. Returns true when the
    /// device should flip its current PWL state given the supplied electrical
    /// observables (with hysteresis applied to suppress chatter).
    ///
    ///  - On  state: commute when current goes negative (i < -hysteresis).
    ///  - Off state: commute when voltage goes positive (v > +hysteresis).
    [[nodiscard]] bool should_commute(const PwlEventContext& ctx) const noexcept {
        const Scalar h = std::max<Scalar>(ctx.event_hysteresis, event_hysteresis_);
        return pwl_state_ ? (ctx.current < -h) : (ctx.voltage > h);
    }

    /// Convenience: matches legacy `is_conducting()` semantics.
    [[nodiscard]] bool is_conducting() const noexcept { return pwl_state_; }

    // --- Stamping --------------------------------------------------------------

    template<typename Matrix, typename Vec>
    void stamp_jacobian_impl(Matrix& J, Vec& f, const Vec& x, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2) {
            return;
        }
        const SwitchingMode active_mode = resolve_switching_mode(mode_);
        if (active_mode == SwitchingMode::Ideal) {
            stamp_jacobian_ideal(J, f, x, nodes);
        } else {
#ifdef PULSIM_USE_AD_STAMP
            // Phase 2.4 of `add-automatic-differentiation`: AD-derived path
            // is the default when the build opts in via the CMake option.
            stamp_jacobian_via_ad(J, f, x, nodes);
#else
            stamp_jacobian_behavioral(J, f, x, nodes);
#endif
        }
    }

    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& /*b*/, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2) {
            return;
        }

        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];
        const Scalar g = pwl_state_ ? g_on_ : g_off_;

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

    static constexpr auto jacobian_pattern_impl() {
        return StaticSparsityPattern<4>{{
            JacobianEntry{0, 0},
            JacobianEntry{0, 1},
            JacobianEntry{1, 0},
            JacobianEntry{1, 1}
        }};
    }

    // ---- Phase 2 of `add-automatic-differentiation` --------------------------
    //
    // Templated current expression for the Behavioral diode model. Both the
    // manual `stamp_jacobian_behavioral` and the AD-driven
    // `stamp_jacobian_via_ad` evaluate the same forward-current formula —
    // this guarantees the two paths cannot drift in math and lets the
    // cross-validation test in `test_ad_diode_stamp.cpp` assert they agree
    // at every operating point.
    //
    // Important: Eigen AutoDiffScalar erases derivatives if you construct
    // `S{constant}`. We keep all numeric coefficients (`v_smooth_`, `g_on_`,
    // `g_off_`, etc.) as `Real` and rely on `Real * S` overloads for the
    // derivative chain — see `add-automatic-differentiation/tasks.md` for
    // the full pegadinha list.
    template <typename S>
    [[nodiscard]] S forward_current_behavioral(S v_anode, S v_cathode) const {
        const S v_diode = v_anode - v_cathode;
        if (v_smooth_ > Real{0.0}) {
            using std::tanh;
            const S alpha = tanh(v_diode / v_smooth_);
            const Real g_avg = (g_on_ + g_off_) * Real{0.5};
            const Real g_diff = (g_on_ - g_off_) * Real{0.5};
            const S g = g_avg + g_diff * alpha;
            return g * v_diode;
        }
        // Sharp transition: g picked from a Real conditional that does not
        // touch S, so AD propagates `g * v_diode` correctly with d(i)/d(v) = g.
        const Real g = (v_anode > v_cathode) ? g_on_ : g_off_;
        return g * v_diode;
    }

    /// AD-derived stamp of the Behavioral residual + Jacobian. Produces the
    /// same `J` entries and `f` contribution as `stamp_jacobian_behavioral`
    /// within floating-point precision; serves as the default path under
    /// the `PULSIM_USE_AD_STAMP` build flag (Phase 2.4).
    /// Non-const so it can mirror the manual stamp's `pwl_state_` mutation —
    /// downstream code (`is_conducting()`, telemetry, segment engine) reads
    /// that bit between stamping calls.
    template <typename Matrix, typename Vec>
    void stamp_jacobian_via_ad(Matrix& J, Vec& f, const Vec& x,
                               std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2) return;
        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];

        const Scalar v_anode = (n_plus >= 0) ? x[n_plus] : Scalar{0.0};
        const Scalar v_cathode = (n_minus >= 0) ? x[n_minus] : Scalar{0.0};

        auto seeded = ad::seed_from_values({v_anode, v_cathode});
        const ad::ADReal i_ad =
            forward_current_behavioral<ad::ADReal>(seeded[0], seeded[1]);

        // Mirror the manual stamp's pwl_state_ side-effect so downstream
        // accessors observe the device-state runtime telemetry consistently.
        const Scalar v_diode = v_anode - v_cathode;
        if (v_smooth_ > Real{0.0}) {
            const Scalar alpha = std::tanh(v_diode / v_smooth_);
            pwl_state_ = (alpha > Real{0.0});
        } else {
            pwl_state_ = (v_diode > Real{0.0});
        }

        const Scalar i_diode = i_ad.value();
        const Scalar di_dva = (i_ad.derivatives().size() > 0)
            ? Scalar{i_ad.derivatives()[0]}
            : Scalar{0.0};
        const Scalar di_dvc = (i_ad.derivatives().size() > 1)
            ? Scalar{i_ad.derivatives()[1]}
            : Scalar{0.0};

        // Stamp Jacobian using AD-derived partials.
        // Anode row: ∂(i_diode)/∂v_anode = di_dva, ∂(i_diode)/∂v_cathode = di_dvc.
        // Cathode row carries the negative (KCL: f[cathode] -= i_diode).
        if (n_plus >= 0) {
            J.coeffRef(n_plus, n_plus) += di_dva;
            if (n_minus >= 0) {
                J.coeffRef(n_plus, n_minus) += di_dvc;
            }
        }
        if (n_minus >= 0) {
            J.coeffRef(n_minus, n_minus) -= di_dvc;
            if (n_plus >= 0) {
                J.coeffRef(n_minus, n_plus) -= di_dva;
            }
        }

        if (n_plus >= 0)  f[n_plus]  += i_diode;
        if (n_minus >= 0) f[n_minus] -= i_diode;
    }

private:
    // --- Behavioral (legacy) Jacobian stamp ----------------------------------
    template<typename Matrix, typename Vec>
    void stamp_jacobian_behavioral(Matrix& J, Vec& f, const Vec& x,
                                   std::span<const NodeIndex> nodes) {
        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];

        const Scalar v_anode = (n_plus >= 0) ? x[n_plus] : Scalar{0.0};
        const Scalar v_cathode = (n_minus >= 0) ? x[n_minus] : Scalar{0.0};
        const Scalar v_diode = v_anode - v_cathode;

        Scalar g;
        Scalar dg_dv;

        if (v_smooth_ > Scalar{0.0}) {
            // Smooth tanh transition: g varies between g_off and g_on
            // as v_diode sweeps across zero.
            const Scalar alpha = std::tanh(v_diode / v_smooth_);
            const Scalar g_avg = (g_on_ + g_off_) * Scalar{0.5};
            const Scalar g_diff = (g_on_ - g_off_) * Scalar{0.5};
            g = g_avg + g_diff * alpha;

            const Scalar dalpha_dv = (Scalar{1.0} - alpha * alpha) / v_smooth_;
            dg_dv = g_diff * dalpha_dv;

            // Update soft state indicator (no PWL semantics here).
            pwl_state_ = (alpha > Scalar{0.0});
        } else {
            // Sharp transition (v_smooth == 0): equivalent to Ideal stamp shape
            // but without the kernel guarantee about no-Newton stable windows.
            pwl_state_ = (v_diode > Scalar{0.0});
            g = pwl_state_ ? g_on_ : g_off_;
            dg_dv = Scalar{0.0};
        }

        const Scalar i_diode = g * v_diode;
        // Jacobian of i = g(v) * v: di/dv = g + v * dg/dv.
        const Scalar j_diag = g + v_diode * dg_dv;

        if (n_plus >= 0) {
            J.coeffRef(n_plus, n_plus) += j_diag;
            if (n_minus >= 0) {
                J.coeffRef(n_plus, n_minus) -= j_diag;
            }
        }
        if (n_minus >= 0) {
            J.coeffRef(n_minus, n_minus) += j_diag;
            if (n_plus >= 0) {
                J.coeffRef(n_minus, n_plus) -= j_diag;
            }
        }

        if (n_plus >= 0) {
            f[n_plus] += i_diode;
        }
        if (n_minus >= 0) {
            f[n_minus] -= i_diode;
        }
    }

    // --- Ideal (PWL) Jacobian stamp ------------------------------------------
    // Sharp two-state stamp keyed off pwl_state_. Conductance is exactly g_on
    // or g_off, with no derivative-of-conductance contribution. The kernel
    // mutates pwl_state_ via commit_pwl_state() at event boundaries; this
    // method never updates the state.
    template<typename Matrix, typename Vec>
    void stamp_jacobian_ideal(Matrix& J, Vec& f, const Vec& x,
                              std::span<const NodeIndex> nodes) const {
        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];

        const Scalar v_anode = (n_plus >= 0) ? x[n_plus] : Scalar{0.0};
        const Scalar v_cathode = (n_minus >= 0) ? x[n_minus] : Scalar{0.0};
        const Scalar v_diode = v_anode - v_cathode;

        const Scalar g = pwl_state_ ? g_on_ : g_off_;
        const Scalar i_diode = g * v_diode;

        if (n_plus >= 0) {
            J.coeffRef(n_plus, n_plus) += g;
            if (n_minus >= 0) {
                J.coeffRef(n_plus, n_minus) -= g;
            }
        }
        if (n_minus >= 0) {
            J.coeffRef(n_minus, n_minus) += g;
            if (n_plus >= 0) {
                J.coeffRef(n_minus, n_plus) -= g;
            }
        }

        if (n_plus >= 0) {
            f[n_plus] += i_diode;
        }
        if (n_minus >= 0) {
            f[n_minus] -= i_diode;
        }
    }

    // --- Members ---------------------------------------------------------------
    Scalar g_on_;
    Scalar g_off_;
    Scalar v_smooth_ = Scalar{0.1};         ///< Behavioral smoothing voltage.
    Scalar event_hysteresis_ = Scalar{1e-9};///< PWL event-hysteresis band.
    SwitchingMode mode_ = SwitchingMode::Auto;
    bool pwl_state_ = false;                ///< on/off state (true = conducting).
};

template<>
struct device_traits<IdealDiode> {
    static constexpr DeviceType type = DeviceType::Diode;
    static constexpr std::size_t num_pins = 2;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = false;  // Nonlinear in Behavioral; PWL otherwise
    static constexpr bool is_dynamic = false;
    static constexpr bool has_loss_model = true;
    static constexpr bool has_thermal_model = false;
    static constexpr bool supports_pwl = true;
    static constexpr std::size_t jacobian_size = 4;
};

}  // namespace pulsim::v1
