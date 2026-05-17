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
        Scalar g_on = 1e3;          ///< On-state conductance (1/R_on) — used when V_F0 == 0
        Scalar g_off = 1e-9;        ///< Off-state conductance (leakage)
        Scalar v_threshold = 0.0;   ///< Threshold voltage (0 for ideal)
        Scalar v_smooth = 0.1;      ///< Smoothing voltage for Behavioral mode (0 = sharp)

        // -------- Realistic diode model (Phase 1 of inverter-bridge-losses) ----
        // Linear forward-characteristic fit: V_F = V_F0 + R_d · I_F (per
        // datasheet tangent fit, e.g. powerStage YAML catalog).
        //
        //   - V_F0 (V):     Forward-voltage offset at T_ref. Default 0 →
        //                    legacy pure-conductance model (backward compat).
        //   - R_d  (Ω):     Differential on-resistance. Used iff V_F0 > 0;
        //                    otherwise 1/g_on is the conducting slope.
        //   - V_F0_tc (V/K):Temperature coefficient on V_F0. Most silicon
        //                    diodes drop ~−2 mV/K, so the default sign here
        //                    is intentionally negative.
        //   - T_ref:        Reference temperature for V_F0 (°C).
        //
        // When V_F0 > 0 the diode stamps as a Norton-shifted conductance:
        //   I_diode = (V_diode − V_F0(T_j)) / R_d
        //
        // i.e. when V_diode = V_F0, I = 0; when V_diode = V_F0 + R_d, I = 1 A.
        Scalar V_F0       = 0.0;
        Scalar R_d        = -1.0;    // <0 → fall back to 1/g_on
        Scalar V_F0_tc    = -2e-3;
        Scalar T_ref      = 25.0;

        // -------- Thermal binding (simple 1-stage steady-state R_θ_JA) --------
        // Once the device has accumulated some conduction energy, the runtime
        // can report a steady-state junction temperature:
        //
        //   T_j = T_amb + P_avg · R_th_ja
        //
        // For richer Foster RC transient modeling, attach a FosterNetwork to
        // the device externally via `set_foster_network()` (Phase 2 — not
        // wired into the runtime ladder yet).
        Scalar R_th_ja    = 25.0;    ///< Junction-to-ambient (K/W)
        Scalar T_amb      = 25.0;    ///< Ambient temperature (°C)

        // -------- Reverse-recovery loss (Phase 4 of
        //          inverter-bridge-losses, Pulsim 0.10.0a10) ----------
        // When the diode is forced from conducting → blocking by an
        // external dV/dt (e.g. a freewheeling diode in a hard-switched
        // inverter leg), it stores `Qrr` Coulombs that must be swept
        // out as the cathode voltage rises. The textbook approximation
        // gives:
        //
        //     E_rec ≈ Qrr · V_r · (1 − s_rec / (1 + s_rec))
        //
        // We simplify to `E_rec = Qrr · V_r · k_shape` with k_shape = 0.5
        // (symmetric triangular recovery — soft-recovery diodes set
        // k_shape ≈ 0.33, hard-recovery ≈ 0.67). Per ON→OFF transition.
        //
        // Default Qrr = 0 → no reverse-recovery loss recorded (legacy).
        Scalar Qrr        = 0.0;     ///< Reverse-recovery charge (C)
        Scalar Erec_shape = 0.5;     ///< Shape factor (0..1; 0.5 = symmetric)

        // PSIM-style parasitic junction capacitance (F) anode-to-cathode.
        // Opt-in auto-snubber for PWL Ideal path. Default 0 = legacy.
        Scalar C_j        = 0.0;
    };

    explicit IdealDiode(Scalar g_on = 1e3,
                        Scalar g_off = 1e-9,
                        std::string name = "")
        : Base(std::move(name))
        , g_on_(g_on)
        , g_off_(g_off) {}

    /// Construct from a fully-specified Params struct (Phase 1 entry-point
    /// for realistic V_F0 + R_d models). Backward-compatible — if
    /// `params.V_F0 == 0` the device behaves exactly like the legacy
    /// (g_on, g_off) form above.
    explicit IdealDiode(const Params& params, std::string name = "")
        : Base(std::move(name))
        , g_on_(params.g_on)
        , g_off_(params.g_off)
        , v_smooth_(params.v_smooth)
        , V_F0_(params.V_F0)
        , R_d_(params.R_d)
        , V_F0_tc_(params.V_F0_tc)
        , T_ref_(params.T_ref)
        , R_th_ja_(params.R_th_ja)
        , T_amb_(params.T_amb)
        , T_j_(params.T_amb)
        , Qrr_(params.Qrr)
        , Erec_shape_(params.Erec_shape)
        // PSIM-style auto-snubber on by default whenever the user
        // opts into a "realistic" diode (V_F0 > 0 indicates the
        // Norton-shifted forward model is active; Qrr > 0 indicates
        // reverse-recovery loss is being tracked). Either one means
        // the device is being used in a power-electronics context
        // where a finite-dt commutation path is wanted — without it,
        // the PWL Ideal stamp presents an infinite-impedance jump at
        // every commutation and a fast switch (boost diode) flickers
        // off the bus ripple.
        //   Qrr > 0   : 5 nF  (fast-recovery device, biggest cap)
        //   V_F0 > 0  : 500 pF (line rectifier or general "realistic"
        //                       diode — small cap, negligible at line
        //                       frequency but enough to keep the
        //                       commutation finite at switching dt)
        //   both 0    : 0 F    (legacy ideal switch — backward compat)
        // Set `params.C_j = 0` explicitly to opt out.
        , C_j_(params.C_j > Scalar{0}
               ? params.C_j
               : (params.Qrr > Scalar{0}
                  ? Scalar{5e-9}
                  : (params.V_F0 > Scalar{0} ? Scalar{5e-10} : Scalar{0}))) {}

    /// Parasitic junction capacitance (F) — see Params::C_j.
    [[nodiscard]] Scalar C_j() const noexcept { return C_j_; }

    // --- Smoothing controls (Behavioral mode only) -----------------------------
    void set_smoothing(Scalar v_smooth) { v_smooth_ = v_smooth; }
    [[nodiscard]] Scalar smoothing() const { return v_smooth_; }
    [[nodiscard]] Scalar g_on() const { return g_on_; }
    [[nodiscard]] Scalar g_off() const { return g_off_; }

    // --- Realistic-loss model accessors (Phase 1 of inverter-bridge-losses) ---
    [[nodiscard]] Scalar V_F0()      const noexcept { return V_F0_; }
    [[nodiscard]] Scalar R_d()       const noexcept { return R_d_; }
    [[nodiscard]] Scalar V_F0_tc()   const noexcept { return V_F0_tc_; }
    [[nodiscard]] Scalar T_ref()     const noexcept { return T_ref_; }
    [[nodiscard]] Scalar R_th_ja()   const noexcept { return R_th_ja_; }
    [[nodiscard]] Scalar T_amb()     const noexcept { return T_amb_; }

    void set_V_F0(Scalar v)        noexcept { V_F0_ = v; }
    void set_R_d(Scalar r)         noexcept { R_d_  = r; }
    void set_V_F0_tc(Scalar tc)    noexcept { V_F0_tc_ = tc; }
    void set_T_ref(Scalar t_ref)   noexcept { T_ref_ = t_ref; }
    void set_R_th_ja(Scalar r)     noexcept { R_th_ja_ = r; }
    void set_T_amb(Scalar t)       noexcept { T_amb_ = t; T_j_ = t; }

    /// Temperature-corrected forward-voltage offset at the current T_j.
    [[nodiscard]] Scalar V_F0_at_Tj() const noexcept {
        return V_F0_ + V_F0_tc_ * (T_j_ - T_ref_);
    }

    /// Effective on-state conductance: 1/R_d when the realistic model is
    /// active (V_F0 > 0 and R_d > 0), else the legacy g_on.
    [[nodiscard]] Scalar effective_g_on() const noexcept {
        if (V_F0_ > Scalar{0} && R_d_ > Scalar{0}) {
            return Scalar{1} / R_d_;
        }
        return g_on_;
    }

    /// Norton-shifted current source: I_F0 = V_F0 / R_d, flowing such that
    /// when V_diode = V_F0 the net current is zero. Returns 0 when the
    /// realistic model is inactive.
    [[nodiscard]] Scalar i_f0_offset() const noexcept {
        if (V_F0_ > Scalar{0} && R_d_ > Scalar{0}) {
            return V_F0_at_Tj() / R_d_;
        }
        return Scalar{0};
    }

    // --- Loss-accumulator accessors -----------------------------------------
    [[nodiscard]] Scalar total_energy()      const noexcept { return e_cond_ + e_sw_; }
    [[nodiscard]] Scalar conduction_energy() const noexcept { return e_cond_; }
    [[nodiscard]] Scalar switching_energy()  const noexcept { return e_sw_; }
    [[nodiscard]] std::size_t switching_events() const noexcept {
        return ev_count_;
    }
    [[nodiscard]] Scalar conduction_time() const noexcept { return t_sim_; }
    [[nodiscard]] Scalar peak_power()     const noexcept { return p_peak_; }
    [[nodiscard]] Scalar last_power()     const noexcept { return p_last_; }
    [[nodiscard]] Scalar last_current()   const noexcept { return i_last_; }
    [[nodiscard]] Scalar last_voltage()   const noexcept { return v_last_; }
    /// Junction temperature used by V_F0_at_Tj() in stamping (the device's
    /// current operating-point assumption). Initialised to T_amb; the user
    /// can override via `set_T_j_init(...)` for fixed-point iterations.
    [[nodiscard]] Scalar junction_temperature() const noexcept { return T_j_; }
    /// Steady-state junction temperature *derived* from the accumulated
    /// average power and the configured R_th_ja network:
    ///     T_j = T_amb + P_avg · R_th_ja.
    /// This does NOT mutate device state — call `set_T_j_init` to push the
    /// derived value back into the stamping path for the next sim pass.
    [[nodiscard]] Scalar steady_state_junction_temperature() const noexcept {
        return T_amb_ + average_power() * R_th_ja_;
    }
    [[nodiscard]] Scalar average_power() const noexcept {
        return (t_sim_ > Scalar{0}) ? (e_cond_ + e_sw_) / t_sim_ : Scalar{0};
    }
    [[nodiscard]] Scalar average_conduction_power() const noexcept {
        return (t_sim_ > Scalar{0}) ? e_cond_ / t_sim_ : Scalar{0};
    }
    [[nodiscard]] Scalar average_switching_power() const noexcept {
        return (t_sim_ > Scalar{0}) ? e_sw_ / t_sim_ : Scalar{0};
    }

    /// Reset the loss accumulator. Called at simulation start (initialize=true
    /// path of update_history).
    /// Zero the loss accumulator (conduction + reverse-recovery). Does
    /// NOT touch T_j. Resets the was_conducting_ snapshot too so the
    /// first call after reset doesn't count a spurious transition.
    void reset_loss() noexcept {
        e_cond_ = 0.0;
        e_sw_ = 0.0;
        ev_count_ = 0;
        p_peak_ = 0.0;
        p_last_ = 0.0;
        t_sim_ = 0.0;
        i_last_ = 0.0;
        v_last_ = 0.0;
        was_conducting_ = false;
        was_conducting_initialized_ = false;
    }

    /// Sample the instantaneous power V·I over the past `dt` seconds.
    /// Called by `Circuit::update_history` for every IdealDiode after each
    /// accepted timestep.
    ///
    /// The on/off decision matches the stamping path (V_F0_at_Tj threshold
    /// using whatever T_j is currently set on the device — typically
    /// T_amb at init or a user-set value).
    ///
    /// Important: this method does NOT update T_j_. The junction
    /// temperature is exposed as a *derived* read-only quantity via
    /// `junction_temperature()` (= T_amb + P_avg · R_th_ja). If you want
    /// the device's stamping to see the elevated T_j, call `set_T_j_init()`
    /// after a thermal analysis and re-simulate (fixed-point iteration —
    /// converges in 2–3 passes for typical bridge-rectifier loads).
    void accumulate_loss(Scalar v_diode, Scalar dt) noexcept {
        if (dt < Scalar{0}) return;
        const Scalar v_thresh =
            (V_F0_ > Scalar{0}) ? V_F0_at_Tj() : Scalar{0};
        const bool conducting = (v_diode > v_thresh);
        const Scalar g  = conducting ? effective_g_on() : g_off_;
        const Scalar I0 = conducting ? i_f0_offset()   : Scalar{0};
        const Scalar i_diode = g * v_diode - I0;
        const Scalar p = v_diode * i_diode;

        // ----- Reverse-recovery event (Phase 4) ---------------------
        // E_rec ≈ Qrr · V_r · shape, charged on every ON → OFF
        // transition. V_r at the moment of transition is the reverse
        // voltage the diode will block (current v_diode, now negative —
        // we use |v_diode|).
        if (was_conducting_initialized_ &&
            was_conducting_ && !conducting && Qrr_ > Scalar{0}) {
            const Scalar V_r = (v_diode < Scalar{0}) ? -v_diode : v_diode;
            const Scalar e_event = Qrr_ * V_r * Erec_shape_;
            if (e_event > Scalar{0}) {
                e_sw_ += e_event;
            }
            // Count the recovery edge even when V_r is briefly 0 at the
            // sample after commutation (the C_j charge ramp takes a few
            // hundred ns to climb past zero — same root cause as the
            // MOSFET/IGBT event-counter fix).
            ++ev_count_;
        }
        was_conducting_ = conducting;
        was_conducting_initialized_ = true;

        v_last_ = v_diode;
        i_last_ = i_diode;
        p_last_ = p;

        // Accumulate only the forward-conduction energy (positive instantaneous
        // power). Off-state leakage (g_off · V²) is in the noise floor and
        // ignored — matches the powerStage `LossAccumulator` convention.
        if (p > Scalar{0}) {
            e_cond_ += p * dt;
            if (p > p_peak_) p_peak_ = p;
        }
        t_sim_ += dt;
    }

    /// Manually update T_j (in °C) before a simulation pass so the
    /// stamping reads V_F0 at the elevated junction temperature. Used by
    /// the powerStage-style fixed-point iteration: simulate → read
    /// junction_temperature() → set_T_j_init(value) → re-simulate.
    void set_T_j_init(Scalar t_j) noexcept { T_j_ = t_j; }

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
    ///  - Off state: commute when voltage rises above V_F0 + hysteresis. When
    ///    V_F0 == 0 (legacy model), this reduces to the original voltage>h
    ///    predicate.
    [[nodiscard]] bool should_commute(const PwlEventContext& ctx) const noexcept {
        // Voltage and current hysteresis are physically different scales —
        // voltage in V (V_F0_at_Tj typically 0.5–1 V), current in A (forward
        // current in a converter is typically A, not µA). A single shared
        // band fits one or the other but not both. The event-scheduler can
        // override per-call via `ctx.event_hysteresis`; locally we use the
        // device defaults (`event_hysteresis_` for voltage,
        // `event_hysteresis_current_` for current) so a boost diode doesn't
        // flicker on round-off-scale current dips while still commutating
        // cleanly at real zero-crossings.
        const Scalar h_v = std::max<Scalar>(ctx.event_hysteresis, event_hysteresis_);
        const Scalar h_i = std::max<Scalar>(ctx.event_hysteresis,
                                             event_hysteresis_current_);
        const Scalar v_on_threshold = (V_F0_ > Scalar{0}) ? V_F0_at_Tj() : Scalar{0};
        return pwl_state_
            ? (ctx.current < -h_i)
            : (ctx.voltage > v_on_threshold + h_v);
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
    void stamp_impl(Matrix& G, Vec& b, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2) {
            return;
        }

        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];
        const Scalar g  = pwl_state_ ? effective_g_on() : g_off_;
        const Scalar I0 = pwl_state_ ? i_f0_offset()   : Scalar{0};

        if (n_plus >= 0) {
            G.coeffRef(n_plus, n_plus) += g;
            if (n_minus >= 0) {
                G.coeffRef(n_plus, n_minus) -= g;
            }
            // Norton-shifted current source pushes I_F0 = V_F0/R_d from
            // cathode → anode internally (anode→cathode externally) so that
            // when V_diode = V_F0 the net branch current is zero.
            b[n_plus] -= I0;
        }
        if (n_minus >= 0) {
            G.coeffRef(n_minus, n_minus) += g;
            if (n_plus >= 0) {
                G.coeffRef(n_minus, n_plus) -= g;
            }
            b[n_minus] += I0;
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
        const Real v_thresh = (V_F0_ > Real{0}) ? V_F0_at_Tj() : Real{0};
        const Real g_on_eff = effective_g_on();
        const S v_shifted = v_diode - v_thresh;
        if (v_smooth_ > Real{0.0}) {
            using std::tanh;
            const S alpha = tanh(v_shifted / v_smooth_);
            const Real g_avg = (g_on_eff + g_off_) * Real{0.5};
            const Real g_diff = (g_on_eff - g_off_) * Real{0.5};
            const S g = g_avg + g_diff * alpha;
            return g * v_shifted;
        }
        // Sharp transition: g picked from a Real conditional that does not
        // touch S, so AD propagates `g * v_shifted` correctly with d(i)/d(v) = g.
        const Real g = (v_diode > v_thresh) ? g_on_eff : g_off_;
        return g * v_shifted;
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
    // When V_F0 == 0 the smoothing happens around v_diode = 0 (legacy
    // behavior). When V_F0 > 0 the threshold shifts to v_diode = V_F0(T_j)
    // and on-conduction follows I_diode = g · (V_diode − V_F0).
    template<typename Matrix, typename Vec>
    void stamp_jacobian_behavioral(Matrix& J, Vec& f, const Vec& x,
                                   std::span<const NodeIndex> nodes) {
        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];

        const Scalar v_anode = (n_plus >= 0) ? x[n_plus] : Scalar{0.0};
        const Scalar v_cathode = (n_minus >= 0) ? x[n_minus] : Scalar{0.0};
        const Scalar v_diode = v_anode - v_cathode;

        const Scalar v_thresh = (V_F0_ > Scalar{0}) ? V_F0_at_Tj() : Scalar{0};
        const Scalar v_shifted = v_diode - v_thresh;
        const Scalar g_on_eff = effective_g_on();

        Scalar g;
        Scalar dg_dv;

        if (v_smooth_ > Scalar{0.0}) {
            // Smooth tanh transition centered on v_thresh.
            const Scalar alpha = std::tanh(v_shifted / v_smooth_);
            const Scalar g_avg = (g_on_eff + g_off_) * Scalar{0.5};
            const Scalar g_diff = (g_on_eff - g_off_) * Scalar{0.5};
            g = g_avg + g_diff * alpha;

            const Scalar dalpha_dv = (Scalar{1.0} - alpha * alpha) / v_smooth_;
            dg_dv = g_diff * dalpha_dv;

            pwl_state_ = (alpha > Scalar{0.0});
        } else {
            pwl_state_ = (v_shifted > Scalar{0.0});
            g = pwl_state_ ? g_on_eff : g_off_;
            dg_dv = Scalar{0.0};
        }

        // I_diode = g(v) · (V_diode − V_F0). When V_F0 = 0 this is identical
        // to the legacy `g · v_diode` stamp.
        const Scalar i_diode = g * v_shifted;
        // Jacobian: d(g · (v − v_thresh))/dv = g + (v − v_thresh) · dg/dv.
        const Scalar j_diag = g + v_shifted * dg_dv;

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
    // Sharp two-state stamp keyed off pwl_state_. When V_F0 > 0 the
    // conducting state stamps the Norton-shifted form:
    //
    //   I_diode = g · V_diode − I_F0     where g = 1/R_d, I_F0 = V_F0(T_j)/R_d
    //
    // i.e. I = 0 exactly when V_diode = V_F0. With V_F0 = 0 (legacy default)
    // the stamp reduces to I = g_on · V_diode bit-for-bit. The kernel mutates
    // pwl_state_ via commit_pwl_state() at event boundaries; this method
    // never updates the state.
    template<typename Matrix, typename Vec>
    void stamp_jacobian_ideal(Matrix& J, Vec& f, const Vec& x,
                              std::span<const NodeIndex> nodes) const {
        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];

        const Scalar v_anode = (n_plus >= 0) ? x[n_plus] : Scalar{0.0};
        const Scalar v_cathode = (n_minus >= 0) ? x[n_minus] : Scalar{0.0};
        const Scalar v_diode = v_anode - v_cathode;

        const Scalar g  = pwl_state_ ? effective_g_on() : g_off_;
        const Scalar I0 = pwl_state_ ? i_f0_offset()   : Scalar{0};
        const Scalar i_diode = g * v_diode - I0;

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
    // PWL event-hysteresis bands. Voltage and current use different
    // scales (V_F0 ≈ 0.5–1 V, forward current ≈ A), so we keep
    // separate defaults: 10 mV on the off→on voltage check (filters
    // round-off bus ripple) and 100 mA on the on→off current check
    // (filters di/dt overshoot through the diode during commutation
    // — a real boost diode in CCM has 1–10 A of forward current, so
    // a 100 mA band is < 10 % of nominal and never trips falsely).
    Scalar event_hysteresis_         = Scalar{1e-2};
    Scalar event_hysteresis_current_ = Scalar{1e-1};
    SwitchingMode mode_ = SwitchingMode::Auto;
    bool pwl_state_ = false;                ///< on/off state (true = conducting).

    // Realistic forward-characteristic params (Phase 1 of
    // inverter-bridge-losses). Defaults preserve the legacy
    // pure-conductance model (V_F0 = 0 → effective_g_on = g_on).
    Scalar V_F0_     = Scalar{0.0};
    Scalar R_d_      = Scalar{-1.0};
    Scalar V_F0_tc_  = Scalar{-2e-3};
    Scalar T_ref_    = Scalar{25.0};
    Scalar R_th_ja_  = Scalar{25.0};
    Scalar T_amb_    = Scalar{25.0};

    // Loss accumulator + thermal state. `e_cond_` is the total integrated
    // V·I·dt over the simulation; `T_j_` is the steady-state junction
    // temperature derived from the running average power.
    Scalar e_cond_ = Scalar{0.0};
    Scalar e_sw_   = Scalar{0.0};   // reverse-recovery energy (Phase 4)
    std::size_t ev_count_ = 0;       // # of recovery events
    Scalar p_peak_ = Scalar{0.0};
    Scalar p_last_ = Scalar{0.0};
    Scalar t_sim_  = Scalar{0.0};
    Scalar i_last_ = Scalar{0.0};
    Scalar v_last_ = Scalar{0.0};
    Scalar T_j_    = Scalar{25.0};

    // Reverse-recovery params (Phase 4 — defaults = 0 = disabled).
    Scalar Qrr_         = Scalar{0.0};
    Scalar Erec_shape_  = Scalar{0.5};
    Scalar C_j_         = Scalar{0.0};
    bool   was_conducting_ = false;
    bool   was_conducting_initialized_ = false;
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
