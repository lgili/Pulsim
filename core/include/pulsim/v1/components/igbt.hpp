#pragma once

#include "pulsim/v1/components/base.hpp"
#include "pulsim/v1/ad/ad_scalar.hpp"

#include <cmath>

namespace pulsim::v1 {

// =============================================================================
// IGBT Device (CRTP - Nonlinear, 3-terminal, supports SwitchingMode::Ideal)
// =============================================================================
//
// Behavioral mode: voltage-controlled conductance with collector-emitter
//                  saturation (forward-drop above v_ce_sat).
// Ideal      mode: piecewise-linear two-state model gated by Vge vs vth.
//                  Collector-emitter path is exactly g_on or g_off; no
//                  saturation drop (which is loss-model territory). Tail
//                  current modeling lives in the catalog tier follow-up.
//
// Replaced legacy `mutable bool is_on_` with explicit `pwl_state_` field and
// const-correct stamping methods.
//
/// Simplified IGBT model for power electronics
/// Terminals: Gate (0), Collector (1), Emitter (2)
class IGBT : public NonlinearDeviceBase<IGBT> {
public:
    using Base = NonlinearDeviceBase<IGBT>;
    static constexpr std::size_t num_pins = 3;
    static constexpr int device_type = static_cast<int>(DeviceType::IGBT);

    struct Params {
        Scalar vth = 5.0;           // Gate threshold voltage (V)
        Scalar g_on = 1e4;          // On-state conductance (S)
        Scalar g_off = 1e-12;       // Off-state conductance (S)
        Scalar v_ce_sat = 1.5;      // Collector-emitter saturation voltage (V)

        // -------- Thermal binding + loss accumulator (Phase 2 of
        //          inverter-bridge-losses) ----------------------------------
        // Loss model: P_cond = V_ce_sat · I_c + R_ce · I_c²
        //   where R_ce(T_j) = R_ce_25 · (1 + R_ce_tc · (T_j − T_ref))
        //         V_ce_sat(T_j) = V_ce_sat + V_ce_tc · (T_j − T_ref)
        // Default R_th_ja=0 disables the thermal model (backward-compat).
        Scalar Rce       = 0.02;       // Ω — bulk on-state resistance
        Scalar Rce_tc    = 5.0e-3;     // 1/K — R_ce temperature coefficient
        Scalar V_ce_tc   = 2.0e-3;     // V/K — V_ce_sat positive coefficient
        Scalar T_ref     = 25.0;       // °C
        Scalar R_th_ja   = 0.0;        // K/W (0 = disabled)
        Scalar T_amb     = 25.0;       // °C

        // -------- Switching-loss model (Phase 4 of inverter-bridge-losses,
        //          Pulsim 0.10.0a10) — mirrors the MOSFET pattern -------
        // E_on(I, V, T_j) = Eon_25 · (I/I_ref) · (V/V_ref)
        //                          · (1 + Esw_tc · (T_j − T_ref))
        // E_off — same shape with Eoff_25.
        Scalar Eon_25    = 0.0;        // J at I_ref, V_ref, T_ref
        Scalar Eoff_25   = 0.0;
        Scalar I_ref     = 50.0;       // A — reference current (typical IGBT)
        Scalar V_ref     = 600.0;      // V — reference voltage
        Scalar Esw_tc    = 3.0e-3;     // 1/K

        // PSIM-style parasitic C_ces (C-E auto-snubber). See
        // components/mosfet.hpp::Params::C_oss for tuning notes.
        Scalar C_oss     = 0.0;        // F

        // Gate-row Jacobian anchor (harden-component-models-vs-psim-plecs
        // Phase A2). Mirrors MOSFETParams::g_gate_leak — see that
        // header for the rationale (gate row would otherwise be all-zero
        // when the gate has no other attached devices, making the MNA
        // matrix singular). 1 nS matches the SPICE GMIN default and
        // sits 1e-3 below the lowest-realistic g_off, so the device's
        // small-signal behaviour is unaffected.
        Scalar g_gate_leak = 1e-9;     // S

        // V_CE_sat Norton-shift in the Behavioral stamp
        // (harden-component-models-vs-psim-plecs Phase A1). OFF by
        // default — turning it ON swaps the legacy on-state stamp
        // `i_C = g_on · V_CE` for the PSIM/PLECS-style Norton form
        // `i_C = (V_CE − V_CE_sat) / Rce`, blended via the same
        // sigmoid alpha that gates the on/off transition.
        //
        // Legacy ON state (flag = false):
        //     V_CE = I_C / g_on = 50 A / 1e4 ≈ 5 mV   (unrealistic)
        // Norton-shifted (flag = true):
        //     V_CE = V_CE_sat + I_C · Rce
        //          = 1.5 V + 50 A · 0.02 Ω = 2.5 V   (PSIM/PLECS-parity)
        //
        // OFF by default so existing tests that pin V_CE near 0 stay
        // green. New circuits that want realistic IGBT conduction
        // losses opt in by setting `enable_vce_sat_stamp = true`.
        // The Ideal (PWL) stamp is unaffected — it is purely g·V_CE
        // and the shift would conflict with the PWL state-space form;
        // the shift lives only on the Behavioral / AD paths.
        bool   enable_vce_sat_stamp = false;

        // Antiparallel diode (harden-component-models-vs-psim-plecs
        // Phase B2). Mirror of MOSFETParams body diode block — IGBT
        // modules ship with an antiparallel "freewheel" diode between
        // emitter (anode) and collector (cathode). When the load is
        // inductive and the IGBT turns OFF, the freewheel path keeps
        // the inductor current flowing through the antiparallel diode
        // — without this, V_collector swings to −∞ at turn-off.
        //
        //   V_ec = v_e − v_c
        //   α_apd = sigmoid(κ · (V_ec − V_F0))
        //   i_apd = α_apd · (V_ec − V_F0) / R_d + (1 − α_apd) · V_ec · g_off
        //
        // The freewheel current SUBTRACTS from the main collector
        // current (it flows in the opposite reference direction,
        // emitter → collector).
        //
        // OFF by default so existing IGBT tests that don't model
        // freewheel paths stay green. Production inverter circuits
        // should set `antiparallel_diode_enable = true`.
        bool   antiparallel_diode_enable = false;
        Scalar antiparallel_diode_V_F0   = 1.0;     // forward drop (V)
        Scalar antiparallel_diode_R_d    = 20e-3;   // forward slope (Ω)
        Scalar antiparallel_diode_g_off  = 1e-9;    // reverse leakage (S)
    };

    explicit IGBT(std::string name = "")
        : Base(std::move(name)), params_() {}

    explicit IGBT(Params params, std::string name)
        : Base(std::move(name)), params_(params), T_j_(params.T_amb) {
        // Auto-promote to SwitchingMode::Ideal when the user opts into
        // the switching-loss model. Same rationale as MOSFET — see the
        // mirror constructor in `components/mosfet.hpp`. Backward-compat
        // preserved when Eon_25 == Eoff_25 == 0 (default).
        if (params.Eon_25 > Scalar{0} || params.Eoff_25 > Scalar{0}) {
            mode_ = SwitchingMode::Ideal;
            // PSIM-style auto-snubber — see components/mosfet.hpp.
            // 20 nF default tracks the larger C_ces typical of IGBTs.
            if (params_.C_oss <= Scalar{0}) {
                params_.C_oss = Scalar{2e-8};
                C_oss_user_set_ = false;
            } else {
                C_oss_user_set_ = true;
            }
        }
    }

    explicit IGBT(Scalar vth, Scalar g_on = 1e4, std::string name = "")
        : Base(std::move(name))
        , params_{vth, g_on, 1e-12, 1.5} {}

    /// Parasitic C_ces between collector and emitter (F).
    [[nodiscard]] Scalar C_oss() const noexcept { return params_.C_oss; }
    /// True only when the user explicitly assigned `Params::C_oss > 0`
    /// before constructing the device. See MOSFET::C_oss_user_set() for
    /// rationale.
    [[nodiscard]] bool C_oss_user_set() const noexcept { return C_oss_user_set_; }
    /// Mutator used by the runtime's auto-parasitics pre-flight (see
    /// `pulsim/v1/auto_parasitics.hpp`). Sets the stamped parasitic
    /// capacitance the next assembly will use.
    void set_C_oss(Scalar c) noexcept { params_.C_oss = c; }

    // -------- Loss + thermal API (Phase 2 of inverter-bridge-losses) --------
    [[nodiscard]] Scalar V_ce_sat_at_Tj() const noexcept {
        return params_.v_ce_sat + params_.V_ce_tc * (T_j_ - params_.T_ref);
    }
    [[nodiscard]] Scalar Rce_at_Tj() const noexcept {
        return params_.Rce * (Scalar{1} + params_.Rce_tc * (T_j_ - params_.T_ref));
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
    /// touch T_j. Resets the was_on_ snapshot so the first call after
    /// reset re-establishes the baseline without counting a spurious
    /// transition.
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

    void set_T_j_init(Scalar t_j) noexcept { T_j_ = t_j; }

    /// Sample V_ce · I_c over the past `dt` seconds. `is_on` is
    /// supplied by the runtime (captures both pwl_state_ and any
    /// forced_switch_state override). When ON the model is
    /// V_ce ≈ V_ce_sat(T_j) + R_ce(T_j)·I_c (Norton-shifted form);
    /// OFF state is pure g_off leakage.
    void accumulate_loss(Scalar v_ce, Scalar dt, bool is_on) noexcept {
        if (dt < Scalar{0}) return;

        // ----- Switching-event detection (E_on / E_off, Phase 4) ------
        if (was_on_initialized_ && (was_on_ != is_on) &&
            params_.R_th_ja > Scalar{0}) {
            const Scalar T_delta = T_j_ - params_.T_ref;
            const Scalar tc_factor = Scalar{1} + params_.Esw_tc * T_delta;
            const Scalar i_ref = (params_.I_ref > Scalar{0}) ?
                params_.I_ref : Scalar{1};
            const Scalar v_ref = (params_.V_ref > Scalar{0}) ?
                params_.V_ref : Scalar{1};
            if (is_on && !was_on_) {
                // OFF → ON. Use v_last_ (blocking V) and post-state i.
                const Scalar V_ce_sat_T = V_ce_sat_at_Tj();
                const Scalar R_ce_T     = Rce_at_Tj();
                const Scalar i_post = (v_ce - V_ce_sat_T) /
                    std::max<Scalar>(R_ce_T, Scalar{1e-9});
                const Scalar I_post = (i_post > Scalar{0}) ? i_post : Scalar{0};
                const Scalar V_block = (v_last_ > Scalar{0}) ?
                    v_last_ : v_ce;
                const Scalar e_event = params_.Eon_25 *
                    (I_post / i_ref) * (V_block / v_ref) * tc_factor;
                if (e_event > Scalar{0}) {
                    e_sw_ += e_event;
                }
                ++ev_count_;   // count the edge regardless of energy magnitude
            } else if (!is_on && was_on_) {
                // ON → OFF. Use i_last_ (pre-state I) and v_ce (now blocking).
                const Scalar I_pre = (i_last_ > Scalar{0}) ?
                    i_last_ : Scalar{0};
                const Scalar V_block = (v_ce > Scalar{0}) ? v_ce : Scalar{0};
                const Scalar e_event = params_.Eoff_25 *
                    (I_pre / i_ref) * (V_block / v_ref) * tc_factor;
                if (e_event > Scalar{0}) {
                    e_sw_ += e_event;
                }
                ++ev_count_;
            }
        }
        was_on_ = is_on;
        was_on_initialized_ = true;

        // ----- Conduction loss (always tracked) ---------------------
        // R_th_ja > 0 enables T_j-corrected (V_ce_sat, R_ce); when
        // disabled we fall back to the static `g_on`/`g_off` but still
        // integrate the conduction energy so the device's loss
        // accessors and `SystemLossSummary` stay consistent.
        const bool tj_corrected = params_.R_th_ja > Scalar{0};
        Scalar i_c;
        if (tj_corrected) {
            const Scalar V_ce_sat_T = V_ce_sat_at_Tj();
            const Scalar R_ce_T     = Rce_at_Tj();
            if (is_on) {
                // V_ce = V_ce_sat + I_c · R_ce  →  I_c = (V_ce − V_ce_sat) / R_ce
                i_c = (v_ce - V_ce_sat_T) / std::max<Scalar>(R_ce_T, Scalar{1e-9});
                if (i_c < Scalar{0}) i_c = Scalar{0};   // IGBT doesn't conduct in reverse
            } else {
                i_c = params_.g_off * v_ce;
            }
        } else {
            const Scalar g = is_on ? params_.g_on : params_.g_off;
            i_c = g * v_ce;
        }
        const Scalar p = v_ce * i_c;

        // Trapezoidal time-averaging (see components/mosfet.hpp for the
        // rationale — drops the over-count from rectangular-end-of-step
        // integration on fast V_CE transients).
        const Scalar p_prev = (p_last_ > Scalar{0}) ? p_last_ : Scalar{0};
        const Scalar p_now  = (p > Scalar{0}) ? p : Scalar{0};
        const Scalar p_avg = (t_sim_ > Scalar{0})
                           ? Scalar{0.5} * (p_prev + p_now)
                           : p_now;

        v_last_ = v_ce;
        i_last_ = i_c;
        p_last_ = p;
        if (p_avg > Scalar{0}) {
            e_cond_ += p_avg * dt;
            if (p_now > p_peak_) p_peak_ = p_now;
        }
        t_sim_ += dt;
    }

    // --- SwitchingMode contract -----------------------------------------------
    [[nodiscard]] SwitchingMode switching_mode() const noexcept { return mode_; }
    void set_switching_mode(SwitchingMode mode) noexcept { mode_ = mode; }

    [[nodiscard]] Scalar event_hysteresis() const noexcept { return event_hysteresis_; }
    void set_event_hysteresis(Scalar h) noexcept { event_hysteresis_ = h; }

    // --- PWL two-state contract -----------------------------------------------
    [[nodiscard]] bool pwl_state() const noexcept { return pwl_state_; }
    void commit_pwl_state(bool on) noexcept { pwl_state_ = on; }

    /// Commute when Vge crosses the gate threshold (with hysteresis).
    [[nodiscard]] bool should_commute(const PwlEventContext& ctx) const noexcept {
        const Scalar h = std::max<Scalar>(ctx.event_hysteresis, event_hysteresis_);
        return pwl_state_
            ? (ctx.control_voltage < params_.vth - h)
            : (ctx.control_voltage > params_.vth + h);
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
        if (nodes.size() < 3) return;
        const NodeIndex n_collector = nodes[1];
        const NodeIndex n_emitter = nodes[2];

        const Scalar g = pwl_state_ ? params_.g_on : params_.g_off;

        if (n_collector >= 0) {
            G.coeffRef(n_collector, n_collector) += g;
            if (n_emitter >= 0) G.coeffRef(n_collector, n_emitter) -= g;
        }
        if (n_emitter >= 0) {
            G.coeffRef(n_emitter, n_emitter) += g;
            if (n_collector >= 0) G.coeffRef(n_emitter, n_collector) -= g;
        }
    }

    static constexpr auto jacobian_pattern_impl() {
        return StaticSparsityPattern<9>{{
            JacobianEntry{0, 0}, JacobianEntry{0, 1}, JacobianEntry{0, 2},
            JacobianEntry{1, 0}, JacobianEntry{1, 1}, JacobianEntry{1, 2},
            JacobianEntry{2, 0}, JacobianEntry{2, 1}, JacobianEntry{2, 2}
        }};
    }

    [[nodiscard]] bool is_conducting() const noexcept { return pwl_state_; }
    [[nodiscard]] const Params& params() const { return params_; }

    /// Sigmoid sharpness for the smooth-gm blend (1/V). Phase-6 IGBT fix:
    /// the previous hard-step model gave Newton zero ∂ic/∂vge information
    /// across iterations, so the DC operating point on a trivial
    /// `Vdc + IGBT + Rload` failed ("All random restarts failed"). With
    /// `kappa = 50/V`, ~99% of the g_off → g_on transition happens within
    /// ±60 mV of `vth`, which is sharp enough to behave like a switch in
    /// power circuits but smooth enough for Newton to find a continuous
    /// path through the threshold.
    static constexpr Real kSmoothGmSharpness = Real{50.0};

    // ---- Phase 2 of `add-automatic-differentiation` --------------------------
    //
    // Templated collector-current expression for the Behavioral IGBT model.
    // Phase-6 update: uses a sigmoid-blended conductance instead of a hard
    // step on `vge > vth ∧ vce > 0`, so AD picks up a non-zero
    // ∂ic/∂vge term and Newton has a continuous gradient through the
    // gate threshold.
    //
    //   alpha_gate = sigmoid(κ · (vge - vth))
    //   alpha_dir  = sigmoid(κ · vce)
    //   alpha      = alpha_gate · alpha_dir
    //   g_eff      = g_off + (g_on - g_off) · alpha
    //   ic         = g_eff · vce
    //
    // For `|vge - vth| > 200 mV` the sigmoid saturates to 0 or 1 within
    // float precision, so far from the threshold this collapses to the
    // legacy hard-step behavior bit-for-bit. Inside the transition
    // window (~120 mV wide), the smooth gradient is what gives Newton
    // its missing gm.
    template <typename S>
    [[nodiscard]] S collector_current_behavioral(S v_g, S v_c, S v_e) const {
        const S vge = v_g - v_e;
        const S vce = v_c - v_e;
        const Real vth = params_.vth;
        const Real kappa = kSmoothGmSharpness;
        using std::exp;
        const S alpha_gate = Real{1.0} / (Real{1.0} + exp(-kappa * (vge - vth)));
        const S alpha_dir  = Real{1.0} / (Real{1.0} + exp(-kappa * vce));
        const S alpha = alpha_gate * alpha_dir;
        const Real g_off = params_.g_off;

        if (params_.enable_vce_sat_stamp) {
            // harden-component-models-vs-psim-plecs Phase A1: Norton-
            // shifted on-state model — PSIM/PLECS-parity. R_CE_on is
            // taken from Rce (the realistic 10-50 mΩ on-state slope);
            // the V_CE_sat offset is the fixed forward voltage that
            // remains even at zero current.
            //
            //   i_C = alpha · (V_CE − V_CE_sat) / R_CE_on
            //       + (1 − alpha) · V_CE · g_off
            //
            // Expanding gives a clean form for AD differentiation:
            //   i_C = [g_off + (g_on_eff − g_off) · alpha] · V_CE
            //         − alpha · V_CE_sat · g_on_eff
            // where g_on_eff = 1 / R_CE_on.
            const Real Rce_safe =
                (params_.Rce > Real{1e-9}) ? params_.Rce : Real{1e-9};
            const Real g_on_eff = Real{1.0} / Rce_safe;
            const Real v_ce_sat_t = V_ce_sat_at_Tj();
            const S g_eff = g_off + (g_on_eff - g_off) * alpha;
            return g_eff * vce - alpha * v_ce_sat_t * g_on_eff;
        }

        const Real g_on  = params_.g_on;
        const S g_eff = g_off + (g_on - g_off) * alpha;
        return g_eff * vce;
    }

    /// AD-derived stamp — Norton companion form, identical math to the
    /// manual `stamp_jacobian_behavioral` (including the dead-saturation
    /// branch), so cross-validation passes within 1e-12 absolute on every
    /// op-point.
    template <typename Matrix, typename Vec>
    void stamp_jacobian_via_ad(Matrix& J, Vec& f, const Vec& x,
                               std::span<const NodeIndex> nodes) {
        if (nodes.size() < 3) return;
        const NodeIndex n_gate = nodes[0];
        const NodeIndex n_collector = nodes[1];
        const NodeIndex n_emitter = nodes[2];

        // harden-component-models-vs-psim-plecs Phase A2: gate-row
        // diagonal anchor (see Params::g_gate_leak comment + MOSFET
        // mirror). Residual untouched.
        if (n_gate >= 0 && params_.g_gate_leak > Scalar{0}) {
            J.coeffRef(n_gate, n_gate) += params_.g_gate_leak;
        }

        const Scalar v_g = (n_gate >= 0) ? x[n_gate] : Scalar{0.0};
        const Scalar v_c = (n_collector >= 0) ? x[n_collector] : Scalar{0.0};
        const Scalar v_e = (n_emitter >= 0) ? x[n_emitter] : Scalar{0.0};

        auto seeded = ad::seed_from_values({v_g, v_c, v_e});
        const ad::ADReal ic_ad =
            collector_current_behavioral<ad::ADReal>(seeded[0], seeded[1], seeded[2]);

        // Mirror manual stamp side-effect.
        const Scalar vge = v_g - v_e;
        const Scalar vce = v_c - v_e;
        pwl_state_ = (vge > params_.vth) && (vce > Real{0});

        const Scalar ic = ic_ad.value();
        const Scalar di_dvg = (ic_ad.derivatives().size() > 0)
            ? Scalar{ic_ad.derivatives()[0]} : Scalar{0.0};
        const Scalar di_dvc = (ic_ad.derivatives().size() > 1)
            ? Scalar{ic_ad.derivatives()[1]} : Scalar{0.0};
        const Scalar di_dve = (ic_ad.derivatives().size() > 2)
            ? Scalar{ic_ad.derivatives()[2]} : Scalar{0.0};

        // For the linear `ic = g · (v_c − v_e)` model, the Taylor offset
        // ic − ∇ic·x simplifies to zero. We compute it generically anyway
        // so future enhancements (true forward-drop / non-linear sat) do
        // not need to revisit the stamp.
        const Scalar i_eq = ic - di_dvg * v_g - di_dvc * v_c - di_dve * v_e;

        // Collector row: + ∂ic/∂x.
        if (n_collector >= 0) {
            J.coeffRef(n_collector, n_collector) += di_dvc;
            if (n_gate >= 0)    J.coeffRef(n_collector, n_gate)    += di_dvg;
            if (n_emitter >= 0) J.coeffRef(n_collector, n_emitter) += di_dve;
        }
        // Emitter row: − ∂ic/∂x.
        if (n_emitter >= 0) {
            if (n_collector >= 0) J.coeffRef(n_emitter, n_collector) -= di_dvc;
            if (n_gate >= 0)      J.coeffRef(n_emitter, n_gate)      -= di_dvg;
            J.coeffRef(n_emitter, n_emitter) -= di_dve;
        }

        // The legacy stamp uses `f[c] += ic − g·vce` (which equals 0 for
        // the linear model). We use the standard Taylor offset i_eq with
        // `f[c] += i_eq`. Both expressions evaluate to 0 here, so manual
        // and AD agree on `f` to within numerical noise.
        if (n_collector >= 0) f[n_collector] += i_eq;
        if (n_emitter >= 0)   f[n_emitter]   -= i_eq;
    }

private:
    // --- Behavioral Jacobian stamp (Phase-6 smooth-gm form) -------------------
    //
    // Evaluates the same sigmoid-blended `ic = g_eff(vge, vce) · vce` model
    // as `collector_current_behavioral`, but with the partials computed
    // analytically (closed form on the sigmoid):
    //
    //   ∂σ_g/∂vge = κ · σ_g · (1 − σ_g)
    //   ∂σ_d/∂vce = κ · σ_d · (1 − σ_d)
    //   ∂g_eff/∂vge = (g_on − g_off) · σ_d · ∂σ_g/∂vge
    //   ∂g_eff/∂vce = (g_on − g_off) · σ_g · ∂σ_d/∂vce
    //   ∂ic/∂v_g    = ∂g_eff/∂vge · vce
    //   ∂ic/∂v_c    = ∂g_eff/∂vce · vce + g_eff
    //   ∂ic/∂v_e    = −∂ic/∂v_g − ∂ic/∂v_c   (chain on vge, vce of v_e)
    //
    // Stamped via the standard Norton companion form (matches the AD path's
    // i_eq exactly — `test_ad_igbt_stamp` passes after this rewrite).
    template<typename Matrix, typename Vec>
    void stamp_jacobian_behavioral(Matrix& J, Vec& f, const Vec& x,
                                   std::span<const NodeIndex> nodes) {
        const NodeIndex n_gate = nodes[0];
        const NodeIndex n_collector = nodes[1];
        const NodeIndex n_emitter = nodes[2];

        // harden-component-models-vs-psim-plecs Phase A2: gate-row
        // diagonal anchor (see Params::g_gate_leak + MOSFET mirror).
        if (n_gate >= 0 && params_.g_gate_leak > Scalar{0}) {
            J.coeffRef(n_gate, n_gate) += params_.g_gate_leak;
        }

        const Scalar vg = (n_gate >= 0) ? x[n_gate] : Scalar{0.0};
        const Scalar vc = (n_collector >= 0) ? x[n_collector] : Scalar{0.0};
        const Scalar ve = (n_emitter >= 0) ? x[n_emitter] : Scalar{0.0};

        const Scalar vge = vg - ve;
        const Scalar vce = vc - ve;

        // Sigmoid-blended on-factor.
        const Real kappa = kSmoothGmSharpness;
        const Scalar sigma_g = Scalar{1.0} /
            (Scalar{1.0} + std::exp(-kappa * (vge - params_.vth)));
        const Scalar sigma_d = Scalar{1.0} /
            (Scalar{1.0} + std::exp(-kappa * vce));
        const Scalar alpha = sigma_g * sigma_d;
        const Scalar dsigma_g_dvge = kappa * sigma_g * (Scalar{1.0} - sigma_g);
        const Scalar dsigma_d_dvce = kappa * sigma_d * (Scalar{1.0} - sigma_d);
        const Scalar dalpha_dvge = sigma_d * dsigma_g_dvge;
        const Scalar dalpha_dvce = sigma_g * dsigma_d_dvce;

        // Telemetry: pwl_state mirrors the on/off bit at α > 0.5.
        pwl_state_ = (alpha > Scalar{0.5});

        // Pick g_on (the on-state branch slope) based on the
        // V_CE_sat Norton-shift flag (Phase A1). When the flag is OFF
        // the legacy g_on = params_.g_on is used and the additional
        // offset term is zero — exactly the pre-A1 stamp.
        Scalar g_on_used;
        Scalar offset_shift;     // = alpha · V_CE_sat · g_on_eff (added to −i_C)
        if (params_.enable_vce_sat_stamp) {
            const Real Rce_safe =
                (params_.Rce > Real{1e-9}) ? params_.Rce : Real{1e-9};
            g_on_used = Scalar{1.0} / Rce_safe;
            offset_shift = alpha * V_ce_sat_at_Tj() * g_on_used;
        } else {
            g_on_used = params_.g_on;
            offset_shift = Scalar{0.0};
        }
        const Scalar dg = g_on_used - params_.g_off;
        const Scalar g_eff = params_.g_off + dg * alpha;

        //   i_C = g_eff · vce − offset_shift          (offset_shift = α·V_CE_sat·g_on)
        // Partials:
        //   ∂(g_eff·vce)/∂vge = dg · ∂α/∂vge · vce
        //   ∂(g_eff·vce)/∂vce = dg · ∂α/∂vce · vce + g_eff
        //   ∂offset_shift/∂vge = V_CE_sat·g_on · ∂α/∂vge
        //   ∂offset_shift/∂vce = V_CE_sat·g_on · ∂α/∂vce
        const Scalar v_ce_sat_g_on = params_.enable_vce_sat_stamp
            ? V_ce_sat_at_Tj() * g_on_used : Scalar{0.0};
        const Scalar di_dvg =
            dg * dalpha_dvge * vce - v_ce_sat_g_on * dalpha_dvge;
        const Scalar di_dvc =
            dg * dalpha_dvce * vce + g_eff - v_ce_sat_g_on * dalpha_dvce;
        const Scalar di_dve = -di_dvg - di_dvc;

        const Scalar ic = g_eff * vce - offset_shift;

        // Norton companion offset (Taylor residual form).
        const Scalar i_eq = ic - di_dvg * vg - di_dvc * vc - di_dve * ve;

        // Collector row: + ∂ic/∂x_i.
        if (n_collector >= 0) {
            J.coeffRef(n_collector, n_collector) += di_dvc;
            if (n_gate >= 0)    J.coeffRef(n_collector, n_gate)    += di_dvg;
            if (n_emitter >= 0) J.coeffRef(n_collector, n_emitter) += di_dve;
        }
        // Emitter row: − ∂ic/∂x_i.
        if (n_emitter >= 0) {
            if (n_collector >= 0) J.coeffRef(n_emitter, n_collector) -= di_dvc;
            if (n_gate >= 0)      J.coeffRef(n_emitter, n_gate)      -= di_dvg;
            J.coeffRef(n_emitter, n_emitter) -= di_dve;
        }

        if (n_collector >= 0) f[n_collector] += i_eq;
        if (n_emitter >= 0)   f[n_emitter]   -= i_eq;
    }

    // --- Ideal (PWL two-state) Jacobian stamp ---------------------------------
    template<typename Matrix, typename Vec>
    void stamp_jacobian_ideal(Matrix& J, Vec& f, const Vec& x,
                              std::span<const NodeIndex> nodes) const {
        const NodeIndex n_gate = nodes[0];
        const NodeIndex n_collector = nodes[1];
        const NodeIndex n_emitter = nodes[2];

        // harden-component-models-vs-psim-plecs Phase A2: gate-row
        // diagonal anchor. The PWL Ideal C-E stamp does not touch the
        // gate row at all (gate is queried only in the commute event
        // detector), so this anchor is the only thing keeping the row
        // non-singular when the gate is otherwise unconnected.
        if (n_gate >= 0 && params_.g_gate_leak > Scalar{0}) {
            J.coeffRef(n_gate, n_gate) += params_.g_gate_leak;
        }

        const Scalar vc = (n_collector >= 0) ? x[n_collector] : Scalar{0.0};
        const Scalar ve = (n_emitter >= 0) ? x[n_emitter] : Scalar{0.0};
        const Scalar vce = vc - ve;

        const Scalar g = pwl_state_ ? params_.g_on : params_.g_off;
        const Scalar ic = g * vce;

        if (n_collector >= 0) {
            J.coeffRef(n_collector, n_collector) += g;
            if (n_emitter >= 0) J.coeffRef(n_collector, n_emitter) -= g;
        }
        if (n_emitter >= 0) {
            J.coeffRef(n_emitter, n_emitter) += g;
            if (n_collector >= 0) J.coeffRef(n_emitter, n_collector) -= g;
        }

        if (n_collector >= 0) f[n_collector] += ic;
        if (n_emitter >= 0) f[n_emitter] -= ic;
    }

    Params params_;
    // Match MOSFET / IdealDiode hysteresis default (see those headers
    // for the rationale on why 1e-2 V/A is the right scale).
    Scalar event_hysteresis_ = Scalar{1e-2};
    SwitchingMode mode_ = SwitchingMode::Auto;
    bool pwl_state_ = false;
    // `boost-pfc-auto-parasitics`: tracks whether C_oss was explicitly
    // set by user (true) or auto-defaulted by ctor (false).
    bool C_oss_user_set_ = false;

    // Loss + thermal accumulator (Phase 2 of inverter-bridge-losses).
    Scalar e_cond_ = Scalar{0.0};
    Scalar e_sw_   = Scalar{0.0};
    std::size_t ev_count_ = 0;
    Scalar p_peak_ = Scalar{0.0};
    Scalar p_last_ = Scalar{0.0};
    Scalar t_sim_  = Scalar{0.0};
    Scalar i_last_ = Scalar{0.0};
    Scalar v_last_ = Scalar{0.0};
    Scalar T_j_    = Scalar{25.0};
    bool   was_on_             = false;
    bool   was_on_initialized_ = false;
};

template<>
struct device_traits<IGBT> {
    static constexpr DeviceType type = DeviceType::IGBT;
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
