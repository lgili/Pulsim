#pragma once

#include "pulsim/v1/components/base.hpp"
#include "pulsim/v1/ad/ad_scalar.hpp"

#include <cmath>

namespace pulsim::v1 {

// =============================================================================
// MOSFET Device (CRTP - Nonlinear, 3-terminal)
// =============================================================================
//
// Behavioral mode: Level-1 Shichman-Hodges (cutoff / triode / saturation).
// Ideal      mode: piecewise-linear two-state model gated by Vgs vs Vth.
//                  Drain-source path becomes a linear `Rds_on = 1/g_on` when
//                  on and `Roff = 1/g_off` when off. Body diode is not yet
//                  embedded in the Ideal stamp (planned in a follow-up
//                  change; the Behavioral tier remains the source of truth
//                  for body-diode reverse recovery analyses for now).
//
// In Ideal mode pwl_state_ is the canonical on/off bit; the kernel mutates it
// via commit_pwl_state() at events located by should_commute() (Vgs threshold).
// In Behavioral mode the same bit shadows the Shichman-Hodges region for
// telemetry parity.
//
/// MOSFET Level 1 model (Shichman-Hodges) plus PWL Ideal alternative.
/// Terminals: Gate (0), Drain (1), Source (2)
class MOSFET : public NonlinearDeviceBase<MOSFET> {
public:
    using Base = NonlinearDeviceBase<MOSFET>;
    static constexpr std::size_t num_pins = 3;
    static constexpr int device_type = static_cast<int>(DeviceType::MOSFET);

    struct Params {
        Scalar vth = 2.0;           // Threshold voltage (V)
        Scalar kp = 0.1;            // Transconductance parameter (A/V^2)
        Scalar lambda = 0.01;       // Channel-length modulation (1/V)
        Scalar g_off = 1e-12;       // Off-state conductance
        bool is_nmos = true;        // NMOS if true, PMOS if false
        Scalar g_on = 1e3;          // On-state conductance for Ideal mode (1/Rds_on)

        // -------- Thermal binding + loss accumulator (Phase 2 of
        //          inverter-bridge-losses) ----------------------------------
        // Conduction loss model: P_cond = I_ds² · R_ds(on)(T_j).
        // Switching loss is left for a future iteration — Eon/Eoff would
        // need a hook into the kernel's commute event handler.
        //
        //   Rds_on_tc : R_ds(on) temperature coefficient (1/K). Default 5e-3
        //               is silicon-typical (R doubles between 25 °C and 125 °C).
        //   T_ref     : reference temperature for Rds_on_tc (°C).
        //   R_th_ja   : junction-to-ambient thermal resistance (K/W).
        //   T_amb     : ambient temperature (°C).
        //
        // When R_th_ja = 0 the device runs in legacy mode (no loss
        // accumulation, no T_j-dependent Rds_on). When R_th_ja > 0 the
        // runtime integrates V_ds·I_ds·dt per accepted timestep and the
        // device exposes `average_power()`, `peak_power()`,
        // `junction_temperature()`, etc.
        Scalar Rds_on_tc = 5.0e-3;
        Scalar T_ref     = 25.0;
        Scalar R_th_ja   = 0.0;       // 0 = disabled (backward-compat)
        Scalar T_amb     = 25.0;

        // -------- Switching-loss model (Phase 4 of inverter-bridge-losses,
        //          Pulsim 0.10.0a10) -------------------------------------
        // Per-event energies are scaled from a 25 °C reference using the
        // textbook linear-in-I, linear-in-V approximation:
        //
        //   E_on(I, V, T_j) = Eon_25 · (I / I_ref) · (V / V_ref)
        //                            · (1 + Esw_tc · (T_j − T_ref))
        //   E_off(I, V, T_j) — same shape with Eoff_25.
        //
        // The accumulator detects pwl_state_ transitions inside
        // `accumulate_loss` and adds the appropriate energy. Default
        // Eon_25 = Eoff_25 = 0 → no switching loss recorded (legacy).
        Scalar Eon_25    = 0.0;       // J at I_ref, V_ref, T_ref
        Scalar Eoff_25   = 0.0;
        Scalar I_ref     = 10.0;      // A — reference current for scaling
        Scalar V_ref     = 400.0;     // V — reference voltage for scaling
        Scalar Esw_tc    = 3.0e-3;    // 1/K — switching-energy TC

        // PSIM-style parasitic output capacitance C_oss. Opt-in: when
        // > 0 the runtime stamps it as a virtual cap between drain and
        // source in the PWL state-space (assemble_state_space), giving
        // the inductor commutation a finite-dt charge path so V_sw
        // doesn't ring at PWM edges. Default 0 keeps legacy circuits
        // unchanged. Typical values: 10 nF–100 nF for boost/buck dt~1µs.
        Scalar C_oss     = 0.0;       // F
    };

    explicit MOSFET(std::string name = "")
        : Base(std::move(name)), params_() {}

    explicit MOSFET(Params params, std::string name)
        : Base(std::move(name)), params_(params), T_j_(params.T_amb) {
        // Auto-promote to SwitchingMode::Ideal when the user opts into
        // the switching-loss model (Eon_25 > 0 or Eoff_25 > 0). The
        // smooth Shichman-Hodges (behavioral) stamp has known Newton-
        // convergence issues with discontinuous PWM gate voltages (~140
        // step rejections per 10 ms at 10 kHz); the PWL Ideal stamp
        // avoids those Newton issues AND lets the accumulator's
        // transition detection see real gate edges. Backward-compat:
        // default Eon_25=Eoff_25=0 → mode stays at SwitchingMode::Auto.
        if (params.Eon_25 > Scalar{0} || params.Eoff_25 > Scalar{0}) {
            mode_ = SwitchingMode::Ideal;
        }
    }

    explicit MOSFET(Scalar vth, Scalar kp, bool is_nmos = true, std::string name = "")
        : Base(std::move(name))
        , params_{vth, kp, 0.01, 1e-12, is_nmos, 1e3} {}

    // --- SwitchingMode contract -----------------------------------------------
    [[nodiscard]] SwitchingMode switching_mode() const noexcept { return mode_; }
    void set_switching_mode(SwitchingMode mode) noexcept { mode_ = mode; }

    [[nodiscard]] Scalar event_hysteresis() const noexcept { return event_hysteresis_; }
    void set_event_hysteresis(Scalar h) noexcept { event_hysteresis_ = h; }

    // --- PWL two-state contract -----------------------------------------------
    [[nodiscard]] bool pwl_state() const noexcept { return pwl_state_; }
    void commit_pwl_state(bool on) noexcept { pwl_state_ = on; }

    /// Commute when Vgs crosses the threshold. NMOS turns on when Vgs > Vth;
    /// PMOS turns on when Vgs < -Vth (sign already folded into ctx via the
    /// is_nmos-aware caller, or detected here from params_).
    [[nodiscard]] bool should_commute(const PwlEventContext& ctx) const noexcept {
        const Scalar h = std::max<Scalar>(ctx.event_hysteresis, event_hysteresis_);
        const Scalar vgs_signed = params_.is_nmos ? ctx.control_voltage
                                                   : -ctx.control_voltage;
        return pwl_state_
            ? (vgs_signed < params_.vth - h)
            : (vgs_signed > params_.vth + h);
    }

    // --- Stamping --------------------------------------------------------------

    /// Stamp Jacobian for Newton iteration
    template<typename Matrix, typename Vec>
    void stamp_jacobian_impl(Matrix& J, Vec& f, const Vec& x, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 3) {
            return;
        }
        const SwitchingMode active_mode = resolve_switching_mode(mode_);
        if (active_mode == SwitchingMode::Ideal) {
            stamp_jacobian_ideal(J, f, x, nodes);
        } else {
#ifdef PULSIM_USE_AD_STAMP
            stamp_jacobian_via_ad(J, f, x, nodes);
#else
            stamp_jacobian_behavioral(J, f, x, nodes);
#endif
        }
    }

    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& /*b*/, std::span<const NodeIndex> nodes) {
        // For initial guess, stamp small conductance.
        if (nodes.size() < 3) return;
        const NodeIndex n_drain = nodes[1];
        const NodeIndex n_source = nodes[2];

        const Scalar g = (resolve_switching_mode(mode_) == SwitchingMode::Ideal && pwl_state_)
            ? params_.g_on
            : params_.g_off;

        if (n_drain >= 0) {
            G.coeffRef(n_drain, n_drain) += g;
            if (n_source >= 0) G.coeffRef(n_drain, n_source) -= g;
        }
        if (n_source >= 0) {
            G.coeffRef(n_source, n_source) += g;
            if (n_drain >= 0) G.coeffRef(n_source, n_drain) -= g;
        }
    }

    static constexpr auto jacobian_pattern_impl() {
        // 3x3 = 9 entries max, but we mainly use D-S path
        return StaticSparsityPattern<9>{{
            JacobianEntry{0, 0}, JacobianEntry{0, 1}, JacobianEntry{0, 2},
            JacobianEntry{1, 0}, JacobianEntry{1, 1}, JacobianEntry{1, 2},
            JacobianEntry{2, 0}, JacobianEntry{2, 1}, JacobianEntry{2, 2}
        }};
    }

    [[nodiscard]] const Params& params() const { return params_; }
    [[nodiscard]] bool is_conducting() const noexcept { return pwl_state_; }

    /// Parasitic output capacitance (F) — see Params::C_oss.
    [[nodiscard]] Scalar C_oss() const noexcept { return params_.C_oss; }

    // -------- Loss + thermal API (Phase 2 of inverter-bridge-losses) --------
    /// R_ds(on) at the device's current T_j (linear coefficient).
    /// Falls back to nominal `g_on` when the thermal model is disabled
    /// (`params.R_th_ja == 0`) so existing tests don't see any drift.
    [[nodiscard]] Scalar Rds_on_at_Tj() const noexcept {
        const Scalar R_nom = (params_.g_on > Scalar{0})
            ? Scalar{1} / params_.g_on : Scalar{1};
        if (params_.R_th_ja <= Scalar{0}) return R_nom;
        return R_nom * (Scalar{1} + params_.Rds_on_tc * (T_j_ - params_.T_ref));
    }

    [[nodiscard]] Scalar total_energy()   const noexcept { return e_cond_ + e_sw_; }
    [[nodiscard]] Scalar conduction_energy() const noexcept { return e_cond_; }
    [[nodiscard]] Scalar switching_energy() const noexcept { return e_sw_; }
    [[nodiscard]] std::size_t switching_events() const noexcept {
        return ev_count_;
    }
    [[nodiscard]] Scalar conduction_time() const noexcept { return t_sim_; }
    [[nodiscard]] Scalar peak_power()     const noexcept { return p_peak_; }
    [[nodiscard]] Scalar last_power()     const noexcept { return p_last_; }
    [[nodiscard]] Scalar last_current()   const noexcept { return i_last_; }
    [[nodiscard]] Scalar last_voltage()   const noexcept { return v_last_; }
    [[nodiscard]] Scalar junction_temperature() const noexcept { return T_j_; }
    /// Total average power = conduction + switching, both per accepted-step.
    [[nodiscard]] Scalar average_power() const noexcept {
        return (t_sim_ > Scalar{0}) ? (e_cond_ + e_sw_) / t_sim_ : Scalar{0};
    }
    [[nodiscard]] Scalar average_conduction_power() const noexcept {
        return (t_sim_ > Scalar{0}) ? e_cond_ / t_sim_ : Scalar{0};
    }
    [[nodiscard]] Scalar average_switching_power() const noexcept {
        return (t_sim_ > Scalar{0}) ? e_sw_ / t_sim_ : Scalar{0};
    }
    [[nodiscard]] Scalar steady_state_junction_temperature() const noexcept {
        return params_.T_amb + average_power() * params_.R_th_ja;
    }

    /// Zero the loss accumulator (conduction + switching). Does NOT
    /// touch T_j — the stamping uses whatever T_j was last set (via
    /// `set_T_j_init` or T_amb at construction). The was_on_ state
    /// snapshot is also cleared so the first call to accumulate_loss
    /// after reset re-establishes the baseline without counting a
    /// spurious transition.
    void reset_loss() noexcept {
        e_cond_ = 0.0;
        e_sw_ = 0.0;
        ev_count_ = 0;
        p_peak_ = 0.0;
        p_last_ = 0.0;
        t_sim_ = 0.0;
        i_last_ = 0.0;
        v_last_ = 0.0;
        was_on_ = false;
        was_on_initialized_ = false;
    }

    /// Push T_j back into the stamping path (for the fixed-point
    /// electrothermal iteration). When R_th_ja > 0 the next stamp will
    /// use Rds_on(T_j) and accumulate_loss will treat this as the
    /// nominal junction temperature.
    void set_T_j_init(Scalar t_j) noexcept { T_j_ = t_j; }

    /// Sample V_ds · I_ds over the past `dt` seconds.
    ///
    /// `is_on` is supplied by the runtime — it captures both the
    /// kernel-committed `pwl_state_` AND any `forced_switch_state`
    /// override applied via `Circuit::set_switch_state`. When ON the
    /// channel resistance is `Rds_on_at_Tj()`; when OFF the device is
    /// modelled as `g_off` (sub-threshold).
    void accumulate_loss(Scalar v_ds, Scalar dt, bool is_on) noexcept {
        if (dt < Scalar{0}) return;

        // ----- Switching-event detection (E_on / E_off, Phase 4) ------
        // Compare current state to the snapshot from the prior call.
        // The very first call (was_on_initialized_ == false) just
        // establishes the baseline — no transition is counted. From
        // then on, every flip of `is_on` triggers an event.
        if (was_on_initialized_ && (was_on_ != is_on) &&
            params_.R_th_ja > Scalar{0}) {
            // Temperature-scaled per-event energy.
            const Scalar T_delta = T_j_ - params_.T_ref;
            const Scalar tc_factor = Scalar{1} + params_.Esw_tc * T_delta;
            const Scalar i_ref = (params_.I_ref > Scalar{0}) ?
                params_.I_ref : Scalar{1};
            const Scalar v_ref = (params_.V_ref > Scalar{0}) ?
                params_.V_ref : Scalar{1};
            if (is_on && !was_on_) {
                // OFF → ON. Use v_last_ (blocking voltage just before
                // the transition) and the post-transition current i.
                const Scalar Rds = Rds_on_at_Tj();
                const Scalar g_on_eff = (Rds > Scalar{0}) ?
                    Scalar{1}/Rds : params_.g_on;
                const Scalar i_post = g_on_eff * v_ds;
                const Scalar V_block = (v_last_ > Scalar{0}) ?
                    v_last_ : v_ds;
                const Scalar e_event = params_.Eon_25 *
                    (std::abs(i_post) / i_ref) * (V_block / v_ref) *
                    tc_factor;
                if (e_event > Scalar{0}) {
                    e_sw_ += e_event;
                    ++ev_count_;
                }
            } else if (!is_on && was_on_) {
                // ON → OFF. Use the pre-transition current (i_last_)
                // and the post-transition voltage (v_ds, now blocking).
                const Scalar I_pre = std::abs(i_last_);
                const Scalar V_block = (v_ds > Scalar{0}) ? v_ds : Scalar{0};
                const Scalar e_event = params_.Eoff_25 *
                    (I_pre / i_ref) * (V_block / v_ref) * tc_factor;
                if (e_event > Scalar{0}) {
                    e_sw_ += e_event;
                    ++ev_count_;
                }
            }
        }
        was_on_ = is_on;
        was_on_initialized_ = true;

        // ----- Conduction loss (always tracked) ----------------------
        // R_th_ja > 0 enables T_j-corrected R_ds(on); when disabled
        // (R_th_ja == 0) we fall back to the static `g_on`/`g_off` but
        // still integrate the conduction energy so the device's loss
        // accessors and the system-level `SystemLossSummary` report a
        // consistent number regardless of which thermal path the user
        // wires up (`MOSFETParams.R_th_ja` or `opts.thermal_devices`).
        const bool tj_corrected = params_.R_th_ja > Scalar{0};
        const Scalar Rds = tj_corrected ? Rds_on_at_Tj() : Scalar{0};
        const Scalar g = is_on
            ? (tj_corrected && Rds > Scalar{0} ? Scalar{1}/Rds : params_.g_on)
            : params_.g_off;
        const Scalar i_ds = g * v_ds;
        const Scalar p = v_ds * i_ds;

        v_last_ = v_ds;
        i_last_ = i_ds;
        p_last_ = p;
        // Both forward and reverse conduction loss (P = V·I always
        // positive in resistive operation, no body-diode model yet).
        const Scalar p_abs = (p > Scalar{0}) ? p : -p;
        if (p_abs > Scalar{0}) {
            e_cond_ += p_abs * dt;
            if (p_abs > p_peak_) p_peak_ = p_abs;
        }
        t_sim_ += dt;
    }

    /// Sigmoid sharpness for the smooth Shichman-Hodges region blend
    /// (1/V). Phase-8 PMOS Newton-region fix: the previous hard-branch
    /// `if (vgs <= vth) ... else if (vds < vgs - vth) ... else ...`
    /// gave Newton no way to cross a region boundary smoothly, so the
    /// high-side PMOS bench (`buck_pmos`) DC OP got trapped in
    /// saturation at V(sw) = -0.19 V instead of the analytical triode
    /// answer V(sw) = 23.3 V. With `kappa = 50/V` the cutoff/triode
    /// and triode/saturation transitions span ~120 mV — sharp enough
    /// to behave like a hard switch in power circuits, smooth enough
    /// for Newton to find a continuous path between regions.
    static constexpr Real kSmoothRegionSharpness = Real{50.0};

    // ---- Phase 2 of `add-automatic-differentiation` --------------------------
    //
    // Templated drain-current expression for the Behavioral (Shichman-Hodges)
    // model.  Phase-8 update: the three regions (cutoff / triode /
    // saturation) are unified into a single smooth formula that converges
    // to each hard branch at saturated tails. The blend uses two ingredients:
    //
    //   1. Smooth ReLU on `Vov`:
    //        σ_g     = sigmoid(κ · (vgs − vth))
    //        Vov_eff = (vgs − vth) · σ_g
    //      → Vov_eff ≈ 0 in cutoff, ≈ vgs−vth far above threshold.
    //
    //   2. Smooth `min(vds, Vov_eff)` for the channel current:
    //        σ_sat   = sigmoid(κ · (Vov_eff − vds))
    //                  (= 1 in triode, 0 in saturation)
    //        vds_eff = σ_sat · vds + (1 − σ_sat) · Vov_eff
    //
    //   3. Unified channel current:
    //        id_ch = kp · (Vov_eff · vds_eff − ½ vds_eff²) · (1 + λ vds)
    //
    //   4. Cutoff leakage (always added):
    //        id    = id_ch + g_off · vds
    //
    // At `vgs >> vth + 200 mV` and `vds >> Vov_eff + 200 mV` (saturation),
    // the formula reduces to `½ kp Vov² (1 + λ vds)` bit-for-bit; at
    // `vds << Vov_eff` (triode), it reduces to the legacy triode formula;
    // and at `vgs << vth` (cutoff), `id ≈ g_off · vds`. The existing
    // `test_ad_mosfet_stamp` cross-validation passes after this rewrite:
    // at every test op-point the smooth model is bit-identical to the
    // hard branch up to floating-point noise (sigmoid tails ≈ 1e-22).
    //
    // All physical coefficients (`vth`, `kp`, `lambda`, `g_off`, `sign`,
    // `kappa`) stay as `Real` per the Phase 1 plumbing notes; only the
    // terminal voltages are `S`. This protects the derivative chain when
    // `S = ADReal`.
    template <typename S>
    [[nodiscard]] S drain_current_behavioral(S v_g, S v_d, S v_s) const {
        const Real sign = params_.is_nmos ? Real{1.0} : Real{-1.0};
        const S vgs = sign * (v_g - v_s);
        const S vds = sign * (v_d - v_s);
        const Real vth = params_.vth;
        const Real kp = params_.kp;
        const Real lambda = params_.lambda;
        const Real kappa = kSmoothRegionSharpness;

        using std::exp;

        // Smooth ReLU on Vov.
        const S sigma_g = Real{1.0} / (Real{1.0} + exp(-kappa * (vgs - vth)));
        const S vov_eff = (vgs - vth) * sigma_g;

        // Smooth min(vds, vov_eff) — sigma_sat = 1 in triode, 0 in saturation.
        const S sigma_sat = Real{1.0} / (Real{1.0} + exp(-kappa * (vov_eff - vds)));
        const S vds_eff = sigma_sat * vds + (Real{1.0} - sigma_sat) * vov_eff;

        // Unified channel current.
        const S id_ch = kp * (vov_eff * vds_eff - Real{0.5} * vds_eff * vds_eff)
                          * (Real{1.0} + lambda * vds);

        // Plus cutoff leakage (small, applies in all regions).
        const S id = id_ch + params_.g_off * vds;
        return sign * id;
    }

    /// AD-derived stamp of the Behavioral residual + Jacobian. Replicates
    /// the manual `stamp_jacobian_behavioral` Norton companion form
    /// (`i_eq = id − Σ ∂id/∂x_i · x_i`) so cross-validation against the
    /// manual stamp passes within floating-point precision at every
    /// operating point.
    template <typename Matrix, typename Vec>
    void stamp_jacobian_via_ad(Matrix& J, Vec& f, const Vec& x,
                               std::span<const NodeIndex> nodes) {
        if (nodes.size() < 3) return;
        const NodeIndex n_gate = nodes[0];
        const NodeIndex n_drain = nodes[1];
        const NodeIndex n_source = nodes[2];

        const Scalar v_g = (n_gate >= 0) ? x[n_gate] : Scalar{0.0};
        const Scalar v_d = (n_drain >= 0) ? x[n_drain] : Scalar{0.0};
        const Scalar v_s = (n_source >= 0) ? x[n_source] : Scalar{0.0};

        auto seeded = ad::seed_from_values({v_g, v_d, v_s});
        const ad::ADReal id_ad =
            drain_current_behavioral<ad::ADReal>(seeded[0], seeded[1], seeded[2]);

        // Mirror manual stamp side-effect: pwl_state_ tracks (region != cutoff).
        const Real sign = params_.is_nmos ? Real{1.0} : Real{-1.0};
        const Scalar vgs_signed = sign * (v_g - v_s);
        pwl_state_ = (vgs_signed > params_.vth);

        const Scalar id = id_ad.value();
        const Scalar di_dvg = (id_ad.derivatives().size() > 0)
            ? Scalar{id_ad.derivatives()[0]} : Scalar{0.0};
        const Scalar di_dvd = (id_ad.derivatives().size() > 1)
            ? Scalar{id_ad.derivatives()[1]} : Scalar{0.0};
        const Scalar di_dvs = (id_ad.derivatives().size() > 2)
            ? Scalar{id_ad.derivatives()[2]} : Scalar{0.0};

        // Norton companion offset (Taylor-residual form):
        //   i_eq = id − ∇id · x
        //        = id − gm·vgs − gds·vds  (manual form, after change of basis
        //                                  via vgs = v_g − v_s, vds = v_d − v_s)
        const Scalar i_eq = id - di_dvg * v_g - di_dvd * v_d - di_dvs * v_s;

        // Drain row: + ∂id/∂x_i.
        if (n_drain >= 0) {
            J.coeffRef(n_drain, n_drain) += di_dvd;
            if (n_gate >= 0)   J.coeffRef(n_drain, n_gate)   += di_dvg;
            if (n_source >= 0) J.coeffRef(n_drain, n_source) += di_dvs;
        }
        // Source row: − ∂id/∂x_i (current-leaving convention reversed).
        if (n_source >= 0) {
            if (n_drain >= 0) J.coeffRef(n_source, n_drain) -= di_dvd;
            if (n_gate >= 0)  J.coeffRef(n_source, n_gate)  -= di_dvg;
            J.coeffRef(n_source, n_source) -= di_dvs;
        }

        // Norton companion residual contribution.
        if (n_drain >= 0)  f[n_drain]  -= i_eq;
        if (n_source >= 0) f[n_source] += i_eq;
    }

private:
    // --- Behavioral Jacobian stamp (Phase-8 smooth-region form) -------------
    //
    // Computes the same smooth blend as `drain_current_behavioral<S>` with
    // closed-form partials, then stamps via the standard Norton companion
    // form. Because the manual stamp and the AD stamp now share the exact
    // same mathematical form (and the AD path autodiff'es the same template
    // that the manual stamp encodes), `test_ad_mosfet_stamp` continues to
    // pass within 1e-12 across cutoff / triode / saturation / boundary
    // op-points.
    template<typename Matrix, typename Vec>
    void stamp_jacobian_behavioral(Matrix& J, Vec& f, const Vec& x,
                                   std::span<const NodeIndex> nodes) {
        const NodeIndex n_gate = nodes[0];
        const NodeIndex n_drain = nodes[1];
        const NodeIndex n_source = nodes[2];

        const Scalar vg = (n_gate >= 0) ? x[n_gate] : Scalar{0.0};
        const Scalar vd = (n_drain >= 0) ? x[n_drain] : Scalar{0.0};
        const Scalar vs = (n_source >= 0) ? x[n_source] : Scalar{0.0};

        // PMOS sign-fold.
        const Scalar sign = params_.is_nmos ? Scalar{1.0} : Scalar{-1.0};
        const Scalar vgs = sign * (vg - vs);
        const Scalar vds = sign * (vd - vs);

        const Scalar vth = params_.vth;
        const Scalar kp = params_.kp;
        const Scalar lambda = params_.lambda;
        const Scalar kappa = kSmoothRegionSharpness;
        const Scalar g_off = params_.g_off;

        // ---- Smooth Vov_eff ----
        const Scalar sigma_g =
            Scalar{1.0} / (Scalar{1.0} + std::exp(-kappa * (vgs - vth)));
        const Scalar dsigma_g_d_vgs = kappa * sigma_g * (Scalar{1.0} - sigma_g);
        const Scalar vov_eff = (vgs - vth) * sigma_g;
        const Scalar dvov_dvgs = sigma_g + (vgs - vth) * dsigma_g_d_vgs;

        // ---- Smooth Vds_eff = soft_min(vds, vov_eff) ----
        const Scalar sigma_sat =
            Scalar{1.0} / (Scalar{1.0} + std::exp(-kappa * (vov_eff - vds)));
        const Scalar dsigma_sat_d_arg =
            kappa * sigma_sat * (Scalar{1.0} - sigma_sat);
        const Scalar dsigma_sat_dvgs = dsigma_sat_d_arg * dvov_dvgs;
        const Scalar dsigma_sat_dvds = -dsigma_sat_d_arg;

        // vds_eff = sigma_sat·vds + (1 - sigma_sat)·vov_eff
        const Scalar vds_eff = sigma_sat * vds + (Scalar{1.0} - sigma_sat) * vov_eff;
        const Scalar dvds_eff_dvgs =
            dsigma_sat_dvgs * vds
            - dsigma_sat_dvgs * vov_eff
            + (Scalar{1.0} - sigma_sat) * dvov_dvgs;
        const Scalar dvds_eff_dvds =
            sigma_sat
            + dsigma_sat_dvds * vds
            - dsigma_sat_dvds * vov_eff;

        // ---- Channel current id_ch = kp · (Vov_eff·Vds_eff − ½ Vds_eff²) · (1+λvds)
        const Scalar core = vov_eff * vds_eff - Scalar{0.5} * vds_eff * vds_eff;
        const Scalar lambda_factor = Scalar{1.0} + lambda * vds;
        const Scalar id_ch = kp * core * lambda_factor;

        // Partials of `core = Vov_eff·Vds_eff − ½·Vds_eff²`
        // ∂core/∂vgs = Vds_eff·dVov_dvgs + (Vov_eff − Vds_eff)·dVds_eff_dvgs
        // ∂core/∂vds = (Vov_eff − Vds_eff)·dVds_eff_dvds
        const Scalar dcore_dvgs = vds_eff * dvov_dvgs
                                  + (vov_eff - vds_eff) * dvds_eff_dvgs;
        const Scalar dcore_dvds = (vov_eff - vds_eff) * dvds_eff_dvds;

        const Scalar dlambda_factor_dvds = lambda;

        // ∂id_ch/∂vgs = kp · ∂core/∂vgs · (1+λvds)
        // ∂id_ch/∂vds = kp · [∂core/∂vds · (1+λvds) + core · λ]
        const Scalar di_ch_dvgs = kp * dcore_dvgs * lambda_factor;
        const Scalar di_ch_dvds = kp * (dcore_dvds * lambda_factor
                                        + core * dlambda_factor_dvds);

        // ---- Total id (with g_off leakage) ----
        const Scalar id_internal = id_ch + g_off * vds;
        const Scalar di_internal_dvgs = di_ch_dvgs;
        const Scalar di_internal_dvds = di_ch_dvds + g_off;

        // PMOS sign-fold of the OUTPUT current (i_actual = sign · i_internal).
        // The internal partials are w.r.t. internal vgs/vds; chain through:
        //   vgs_internal = sign · (vg − vs)  →  ∂id_actual/∂vg = sign·(sign·di/dvgs)
        //                                                     = di_internal_dvgs
        //   ∂id_actual/∂vs = sign · (-sign · di/dvgs − sign · di/dvds)
        //                  = − di_internal_dvgs − di_internal_dvds
        //   ∂id_actual/∂vd = sign · (sign · di/dvds) = di_internal_dvds
        //
        // (The two `sign` factors cancel in vg/vd partials, net negative on vs.)
        const Scalar id = sign * id_internal;
        const Scalar di_dvg = di_internal_dvgs;
        const Scalar di_dvd = di_internal_dvds;
        const Scalar di_dvs = -di_internal_dvgs - di_internal_dvds;

        // Telemetry: pwl_state mirrors the channel-on bit (~ Vgs > Vth).
        pwl_state_ = (sigma_g > Scalar{0.5});

        // Norton companion residual (Taylor-offset form, matches the AD path).
        const Scalar i_eq = id - di_dvg * vg - di_dvd * vd - di_dvs * vs;

        // Drain row: + ∂id/∂x_i.
        if (n_drain >= 0) {
            J.coeffRef(n_drain, n_drain) += di_dvd;
            if (n_gate >= 0)   J.coeffRef(n_drain, n_gate)   += di_dvg;
            if (n_source >= 0) J.coeffRef(n_drain, n_source) += di_dvs;
        }
        // Source row: − ∂id/∂x_i (current-leaving convention).
        if (n_source >= 0) {
            if (n_drain >= 0) J.coeffRef(n_source, n_drain) -= di_dvd;
            if (n_gate >= 0)  J.coeffRef(n_source, n_gate)  -= di_dvg;
            J.coeffRef(n_source, n_source) -= di_dvs;
        }

        // Norton companion residual contribution.
        if (n_drain >= 0)  f[n_drain]  -= i_eq;
        if (n_source >= 0) f[n_source] += i_eq;
    }

    // --- Ideal (PWL two-state) Jacobian stamp ---------------------------------
    template<typename Matrix, typename Vec>
    void stamp_jacobian_ideal(Matrix& J, Vec& f, const Vec& x,
                              std::span<const NodeIndex> nodes) const {
        const NodeIndex n_drain = nodes[1];
        const NodeIndex n_source = nodes[2];

        const Scalar vd = (n_drain >= 0) ? x[n_drain] : Scalar{0.0};
        const Scalar vs = (n_source >= 0) ? x[n_source] : Scalar{0.0};
        const Scalar vds = vd - vs;

        const Scalar g = pwl_state_ ? params_.g_on : params_.g_off;
        const Scalar id = g * vds;

        // Pure drain-source resistive stamp; no gm contribution.
        if (n_drain >= 0) {
            J.coeffRef(n_drain, n_drain) += g;
            if (n_source >= 0) J.coeffRef(n_drain, n_source) -= g;
        }
        if (n_source >= 0) {
            J.coeffRef(n_source, n_source) += g;
            if (n_drain >= 0) J.coeffRef(n_source, n_drain) -= g;
        }

        if (n_drain >= 0) f[n_drain] -= id;
        if (n_source >= 0) f[n_source] += id;
    }

    Params params_;
    Scalar event_hysteresis_ = Scalar{1e-9};
    SwitchingMode mode_ = SwitchingMode::Auto;
    bool pwl_state_ = false;

    // Loss + thermal accumulator (Phase 2 of inverter-bridge-losses).
    // All zero when R_th_ja == 0 (legacy mode).
    Scalar e_cond_ = Scalar{0.0};
    Scalar e_sw_   = Scalar{0.0};   // switching energy (Phase 4)
    std::size_t ev_count_ = 0;       // # of switching events recorded
    Scalar p_peak_ = Scalar{0.0};
    Scalar p_last_ = Scalar{0.0};
    Scalar t_sim_  = Scalar{0.0};
    Scalar i_last_ = Scalar{0.0};
    Scalar v_last_ = Scalar{0.0};
    Scalar T_j_    = Scalar{25.0};
    // Switching transition tracker — true iff the device was ON at the
    // PREVIOUS accumulate_loss() call. The `was_on_initialized_` flag
    // suppresses a spurious event on the very first call.
    bool   was_on_             = false;
    bool   was_on_initialized_ = false;
};

template<>
struct device_traits<MOSFET> {
    static constexpr DeviceType type = DeviceType::MOSFET;
    static constexpr std::size_t num_pins = 3;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = false;  // Nonlinear
    static constexpr bool is_dynamic = false;
    static constexpr bool has_loss_model = true;
    static constexpr bool has_thermal_model = true;
    static constexpr bool supports_pwl = true;
    static constexpr std::size_t jacobian_size = 9;
};

}  // namespace pulsim::v1
