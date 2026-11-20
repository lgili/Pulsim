#pragma once
// add-parameter-sensitivities, group 5 (design D1): forward parameter
// sensitivities of the time-stepping engines, as the stage recurrence of the
// discrete step. ONE place for every scheme: an engine calls `stage` after
// each converged stage solve (with the stage's ψ combination, `StagePsi`) and
// `commit` when the step is accepted; the scheme lives in the ψ coefficients
// only.
//
// A stage solves R(X) = f(X, t) + a·(q(X) − ψ) = 0 (conductance form: the
// engine's matrix is J = ∂f/∂x + a·∂q/∂x, a = 2/factor_dt), with ψ a fixed
// combination of the step start's q_n, q̇_n and the earlier stages' q_j, q̇_j.
// Differentiating in a parameter p:
//
//   J·S = a·(dψ/dp − ∂q/∂p) − ∂f/∂p,     dq/dp = ∂q/∂x·S + ∂q/∂p,
//   dq̇/dp = a·(dq/dp − dψ/dp)             (the stage's own derivative).
//
// The solve reuses the stage's factor (no factorization is ever added): the
// cached linear factor (exact), or the Newton workspace's factor of the last
// iteration (stale by one Newton step: the solve is iterated with that
// factor, against the true Jacobian at X, to the Newton tolerance — 5.2).
// The engine's state is never touched: with and without sensitivities the
// states are the same bits.
//
// Scope (refused by name otherwise): linear R, C, L (not coupled) and the
// static Newton laws (Shockley, smooth ideal diode, Level-1 MOSFET, excess
// loss) — their ∂f/∂p by AD on the law (`device::seed_params`). Stateful
// Newton kinds (their q history lives in the device), a mode change (a
// diode commutation or a gate edge: the jump correction is group 6), and a
// start that depends on the parameters are refused.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "pulsim/device/device.hpp"
#include "pulsim/device/laws.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/parse_real.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/assemble.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/device_refresh.hpp"
#include "pulsim/pwl/history_state.hpp"
#include "pulsim/pwl/interface_stamps.hpp"
#include "pulsim/pwl/nonlinear_solve.hpp"
#include "pulsim/pwl/nonlinear_refresh_mosfet_level1.hpp"
#include "pulsim/solver/result.hpp"
#include "pulsim/solver/step/esdirk43.hpp"
#include "pulsim/solver/unsupported.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/sparse/solver.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

namespace pulsim::step {

/// ψ of one stage: ψ = c_qn·q_n + c_dn·q̇_n + Σ_{j<n_prev} (c_q[j]·q_j +
/// c_d[j]·q̇_j), and the stage matrix's coefficient `a` of ∂q/∂x.
struct StagePsi {
    Real a = Real{0};
    Real c_qn = Real{1};
    Real c_dn = Real{0};
    int n_prev = 0;
    std::array<Real, 6> c_q{};
    std::array<Real, 6> c_d{};
};

/// The trapezoid over h (factor h): ψ = q_n + (h/2)·q̇_n.
[[nodiscard]] inline StagePsi psi_trapezoidal(Real h) noexcept {
    return {Real{2} / h, Real{1}, Real{0.5} * h, 0, {}, {}};
}
/// Backward Euler solved on the trapezoidal factor `factor_dt` (interval
/// factor_dt/2): ψ = q_n (the derivative slot is left out).
[[nodiscard]] inline StagePsi psi_backward_euler_on(Real factor_dt) noexcept {
    return {Real{2} / factor_dt, Real{1}, Real{0}, 0, {}, {}};
}
/// TR-BDF2 stage 2 on the factor γh: q̇_1 = (c1·q_1 + c2·q_γ + c3·q_n)/h,
/// so ψ = −(c2·q_γ + c3·q_n)/c1 (stage 0 is the trapezoid to γh).
[[nodiscard]] inline StagePsi psi_bdf2_stage2(Real factor_dt) noexcept {
    const Real c1 = pwl::HistoryState::trbdf2_c1();
    StagePsi p{Real{2} / factor_dt, -pwl::HistoryState::trbdf2_c3() / c1,
               Real{0}, 1, {}, {}};
    p.c_q[0] = -pwl::HistoryState::trbdf2_c2() / c1;
    return p;
}
/// SDIRK(2,2) stage 2 on `factor_dt`: backward Euler from the pseudo
/// previous state q_n + r·(q_1 − q_n).
[[nodiscard]] inline StagePsi psi_sdirk_stage2(Real factor_dt, Real r) noexcept {
    StagePsi p{Real{2} / factor_dt, Real{1} - r, Real{0}, 1, {}, {}};
    p.c_q[0] = r;
    return p;
}

/// ESDIRK4(3)6L[2]SA implicit stage s of the step hh, on the factor of hh/2:
/// ψ_s = q_n + a_{s+1,0}·hh·q̇_n + Σ_{j<s} (a_{s+1,j+1}/γ)·γ·hh·q̇_j (the
/// increments q(Y_j) − ψ_j = γ·hh·q̇_j).
[[nodiscard]] inline StagePsi psi_esdirk43(int s, Real hh) noexcept {
    using E = Esdirk43;
    StagePsi p{Real{1} / (E::gamma * hh), Real{1}, E::explicit_weight(s) * hh, s, {}, {}};
    for (int j = 0; j < s; ++j) {
        p.c_d[static_cast<std::size_t>(j)] = E::weight(s, j) * E::gamma * hh;
    }
    return p;
}

using solver::SensitivityTrace;

class StageSensitivity {
public:
    // Out of line: the engines' hot step loops keep their code when off.
    [[gnu::noinline]] StageSensitivity(const topology::Graph& graph, const pwl::DevicePool& pool,
                     const std::vector<std::string>& names, const char* engine,
                     Real tol)
        : graph_{graph}, pool_{pool}, engine_{engine}, tol_{tol} {
        if (names.empty()) return;
        refuse_kinds_();
        if (names.size() > 64) {
            throw UnsupportedError(engine_, "sensitivities", "", "",
                std::format("sensitivities: {} parameters; the forward method "
                            "is limited to 64 per run. Split the request.",
                            names.size()));
        }
        for (const auto& nm : names) params_.push_back(resolve_(nm));
        for (Index b = 0; b < graph_.num_branches(); ++b) {
            const SK k = pool_.kind_of(graph_.branch(b).id);
            has_newton_ = has_newton_ || k == SK::NonlinearDiode || k == SK::ShockleyDiode
                          || k == SK::ExcessLoss || k == SK::MosfetLevel1;
        }
        for (std::size_t i = 0; i < names.size(); ++i)
            for (std::size_t j = 0; j < i; ++j)
                if (names[i] == names[j])
                    throw std::invalid_argument(
                        "sensitivities: duplicate name '" + names[i] + "'.");
        trace_.params = names;
    }

    [[nodiscard]] bool on() const noexcept { return !params_.empty(); }

    /// What the run does beside the stamps the recurrence differentiates:
    /// in-kernel device losses (heat currents and capacitor-slot writes from
    /// earlier states), a caller's static-refresh override (another model
    /// than the pool's laws), or Levenberg–Marquardt from the start (its
    /// factor is J + λI, not the stage's). Each is refused by name.
    [[gnu::noinline]] void refuse_setup(bool losses, bool nl_override, bool lm) const {
        const char* why = losses ? "in-kernel device losses (their heat currents "
                                   "read earlier states)"
                        : nl_override ? "a caller-supplied nl_refresh (not the "
                                        "pool's device laws)"
                        : lm ? "enable_newton_lm (the factor is J + λI)"
                             : nullptr;
        if (why == nullptr) return;
        throw UnsupportedError(engine_, "sensitivities", "", "",
            std::format("sensitivities on engine '{}' with {}: not carried by "
                        "the stage recurrence; run without it.", engine_, why));
    }

    /// The factor a Newton stage ended on: the workspace's plain solver.
    /// A solve that went through Levenberg–Marquardt (an auto-promotion)
    /// has none of the stage's matrix: refused by name.
    [[nodiscard, gnu::noinline]] const sparse::DirectSolver& newton_factor(
            const pwl::NewtonWorkspace& w, Real t) const {
        if (!w.solver || w.lm_attempts != 0) {
            refuse_step(t, "solved by Levenberg-Marquardt (an auto-promotion "
                           "on a singular or stalled Newton), whose factor is "
                           "not the stage's matrix");
        }
        return *w.solver;
    }

    /// The run's start: S = 0, dq/dp = ∂q/∂p(x0), dq̇/dp = 0 (every engine
    /// starts from zero derivative slots). `x0` must not depend on the
    /// parameters (the engine refuses a DC-operating-point start; a start
    /// the engine moves before its first step is caught by `begin_step`).
    [[gnu::noinline]] void start(Real t0, const Vector& x0) {
        n_ = static_cast<Index>(x0.size());
        const auto np = static_cast<Index>(params_.size());
        pend_.on = false;
        S_.setZero(n_, np);
        dqd_n_.setZero(n_, np);
        dq_n_.setZero(n_, np);
        x_last_ = x0;
        t_last_ = t0;
        have_qx_ = false;
        trace_.n_state = static_cast<Size>(n_);
        record_(t0);
    }

    /// The engine's mode at the start: the first stage must be solved under
    /// it, or after an event registered in between (a mode change inside the
    /// first step is refused like any other that was not located).
    [[gnu::noinline]] void start_mode(const topology::SwitchStateMask& mask) { init_mode_(mask); }

    /// A step (attempt) begins at (t0, x0): the earlier attempt's stages are
    /// discarded. The engine must start it from the state this object last
    /// committed; anything else (an event landing, a probe or a split that
    /// committed a state no stage here produced) is refused by name.
    [[gnu::noinline]] void begin_step(Real t0, const Vector& x0) {
        drop_stages_(0);
        n_stage_ = 0;
        check_contiguous_(t0, x0);
    }

    /// The run ended at (t, x): the last committed state must be ours.
    /// An event whose instant moves with p, still pending (its jump is applied
    /// at the first post-event stage, and none came: it fell within the
    /// time resolution of t_end), leaves x(t_end) one-sided in p: refused.
    [[gnu::noinline]] void finish(Real t, const Vector& x) const {
        check_contiguous_(t, x);
        if (pend_.on && pend_.dtau.cwiseAbs().maxCoeff() > Real{0}) {
            refuse_step(t, std::format("an event ({}) at t = {} whose jump in the "
                                       "derivative was not applied: the run ends at it",
                                       pend_.what, pend_.tau));
        }
    }

    /// Stage `s` (0-based in the step) converged at X under `mask`. `fac`
    /// holds the factor the stage solved with; `J_lin` is the linear part of
    /// a Newton stage (nullptr: the factor is the stage's exact matrix).
    [[gnu::noinline]] void stage(int s, const StagePsi& psi, const topology::SwitchStateMask& mask,
               const Vector& X, Real t, const sparse::DirectSolver& fac,
               const sparse::Matrix* J_lin) {
        if (!have_qx_) {
            init_mode_(mask);
        } else if (mask == mask_) {
            pend_.on = false;   // an event that changed no mode: f⁻ = f⁺, no jump
        } else {
            if (!pend_.on) {
                throw UnsupportedError(engine_, "sensitivities", "", "",
                    std::format("sensitivities on engine '{}': at t = {} the mode "
                                "changed ({}) at an instant that was not located "
                                "in time{}; its jump term (f- - f+)·dτ/dp cannot "
                                "be formed (add-parameter-sensitivities group 6).",
                                engine_, t, switch_names_(mask_, mask),
                                std::string_view(engine_) == "pwl"
                                    ? ": on engine 'pwl' an event is located only "
                                      "with enable_substep_state_correction=True "
                                      "(event splitting)"
                                    : ""));
            }
            if (s != 0) refuse_step(t, "changed mode between two stages of one step");
            apply_jump_(mask, psi.a, fac, J_lin != nullptr, t);
        }
        if (s > n_stage_ || s >= kMaxStages) {
            throw std::logic_error("StageSensitivity: stages out of order");
        }
        drop_stages_(s);   // a stage taken again (pwl's stage-2 retry): drop it
        const bool newton = J_lin != nullptr;
        if (newton) {
            if (!nl_) nl_ = pwl::make_interface_static_refresh();
            if (Jnl_.rows() != n_) Jnl_.resize(n_, n_);
            fnl_.setZero(n_);
            nl_(X, Jnl_, fnl_, graph_, pool_);
            sparse::compress_in_place(Jnl_);
        }
        auto& dq = dq_st_[static_cast<std::size_t>(s)];
        auto& dqd = dqd_st_[static_cast<std::size_t>(s)];
        dq.resize(n_, S_.cols());
        dqd.resize(n_, S_.cols());
        Vector dpsi(n_), qp(n_), fp(n_), rhs(n_), S(n_), r(n_), d(n_);
        for (Index j = 0; j < S_.cols(); ++j) {
            dpsi = psi.c_qn * dq_n_.col(j) + psi.c_dn * dqd_n_.col(j);
            for (int i = 0; i < psi.n_prev; ++i) {
                const auto ii = static_cast<std::size_t>(i);
                dpsi += psi.c_q[ii] * dq_st_[ii].col(j) + psi.c_d[ii] * dqd_st_[ii].col(j);
            }
            qp.setZero();
            fp.setZero();
            add_qp_(params_[static_cast<std::size_t>(j)], X, qp);
            add_fp_(params_[static_cast<std::size_t>(j)], X, fp);
            rhs = psi.a * (dpsi - qp) - fp;
            fac.solve(rhs, S);
            ++st_solves_[static_cast<std::size_t>(s)];
            if (newton) {
                // modified Newton: iterate on the stale factor (5.2)
                for (int it = 0;; ++it) {
                    r.noalias() = *J_lin * S;
                    r.noalias() += Jnl_ * S;
                    r = rhs - r;
                    fac.solve(r, d);
                    ++st_iters_[static_cast<std::size_t>(s)];
                    S += d;
                    const Real dn = d.cwiseAbs().maxCoeff();
                    const Real sn = S.cwiseAbs().maxCoeff();
                    if (!(dn > tol_ * sn)) break;
                    if (it >= kMaxIterations || !std::isfinite(dn)) {
                        throw std::runtime_error(std::format(
                            "sensitivities on engine '{}': at t = {} the "
                            "sensitivity solve of '{}' did not converge on the "
                            "stage's factor ({} iterations, last correction "
                            "{:.3e} of {:.3e}; tol_newton_dx is the relative "
                            "tolerance on S).",
                            engine_, t, trace_.params[static_cast<std::size_t>(j)],
                            it + 1, dn, sn));
                    }
                }
            }
            dq.col(j) = Cq_ * S + qp;
            dqd.col(j) = psi.a * (dq.col(j) - dpsi);
            Sst_.resize(n_, S_.cols());
            Sst_.col(j) = S;
        }
        // the values' own recurrence (the engine's charge slots, mirrored):
        // q̇ of a stage is −f(X, t) exactly — the field an event needs
        {
            const auto ss = static_cast<std::size_t>(s);
            Vector psi_v = psi.c_qn * qv_n_ + psi.c_dn * qdv_n_;
            for (int i = 0; i < psi.n_prev; ++i) {
                const auto ii = static_cast<std::size_t>(i);
                psi_v += psi.c_q[ii] * qv_st_[ii] + psi.c_d[ii] * qdv_st_[ii];
            }
            qv_st_[ss] = Cq_ * X;
            qdv_st_[ss] = psi.a * (qv_st_[ss] - psi_v);
        }
        X_last_ = X;
        n_stage_ = s + 1;
    }

    /// The step was accepted at time `t`: its last stage is the new step start.
    /// `record` false: an intermediate sub-step end (an event inside the
    /// engine's step), not a sample of the trace.
    [[gnu::noinline]] void commit(Real t, bool record = true) {
        if (n_stage_ == 0) {
            throw std::logic_error("StageSensitivity: commit without a stage");
        }
        const auto k = static_cast<std::size_t>(n_stage_ - 1);
        dq_n_ = dq_st_[k];
        dqd_n_ = dqd_st_[k];
        qv_n_ = qv_st_[k];
        qdv_n_ = qdv_st_[k];
        S_ = Sst_;
        x_last_ = X_last_;
        t_last_ = t;
        for (int i = 0; i < n_stage_; ++i) {   // the accepted attempt's solves
            const auto ii = static_cast<std::size_t>(i);
            trace_.n_sens_solves += st_solves_[ii];
            trace_.n_sens_iterations += st_iters_[ii];
            st_solves_[ii] = st_iters_[ii] = 0;
        }
        n_stage_ = 0;
        if (record) record_(t);
    }

    // ---- events (group 6, design D3; DECISIONS 16) ----------------------
    //
    // At an event at τ(p) the differential state q (charges, fluxes) is
    // continuous and its sensitivity jumps:
    //   dq/dp⁺ = dq/dp⁻ + (q̇⁻ − q̇⁺)·dτ/dp,
    //   dτ/dp = −(g_x·S⁻ + g_p)/(g_x·ẋ⁻ + g_t)        (state event g = 0),
    //   dτ/dp from the modulator                     (time event; 0 when the
    //                                                 instant is fixed).
    // q̇⁻ is the last pre-event stage's own derivative (= −f⁻(x⁻, τ) exactly);
    // q̇⁺ = −f⁺(x⁺) at the consistent post-event point and ẋ⁻ (algebraic
    // components included) come from one bordered solve each on a factor of
    // that mode (`bordered_`). The engine registers the event; the jump is
    // applied at the first post-event stage, whose factor is the new mode's.

    /// A gate edge at τ: the gate bits go from `before` to `after`. A duty
    /// parameter whose switch turns OFF there moves it by dτ/dD = T. An edge
    /// the engine did not locate (taken at a grid point) with a non-zero
    /// dτ/dp is a staircase in p: refused.
    [[gnu::noinline]] void time_event(Real tau, const topology::SwitchStateMask& before,
                                      const topology::SwitchStateMask& after, bool located) {
        if (before == after) return;   // a no-op edge
        Vector dtau = Vector::Zero(static_cast<Index>(params_.size()));
        for (std::size_t j = 0; j < params_.size(); ++j) {
            const Param& p = params_[j];
            if (p.kind != Param::Kind::Duty || p.field >= before.size()
                || p.field >= after.size()) continue;
            if (before.get(p.field) && !after.get(p.field)) dtau[static_cast<Index>(j)] = p.value;
        }
        if (!located && dtau.cwiseAbs().maxCoeff() > Real{0}) {
            throw UnsupportedError(engine_, "sensitivities", "", "",
                std::format("sensitivities on engine '{}': at t = {} a gate edge "
                            "whose instant depends on a duty ({}) was taken at a "
                            "grid point, not located: the solution is a staircase "
                            "in the duty. Run with enable_substep_state_correction"
                            "=True (event splitting).",
                            engine_, tau, switch_names_(before, after)));
        }
        add_pending_(tau, dtau, "a gate edge");
    }

    /// A gate edge at τ ≈ t_end that no step after it opened (trbdf2's
    /// no-final-sliver rule, 64 h_min; pwl's restart rule at the last grid
    /// point): x(t_end) is then the pre-edge side, one-sided in p when the
    /// edge moves with a duty (dv/dD read 0 against 7.38): refused, as
    /// `finish` refuses a pending event.
    [[gnu::noinline]] void edge_past_end(Real tau, const topology::SwitchStateMask& before,
                                         const topology::SwitchStateMask& after) const {
        for (const Param& p : params_) {
            if (p.kind == Param::Kind::Duty && p.field < before.size()
                && p.field < after.size() && before.get(p.field) && !after.get(p.field)) {
                refuse_step(tau, std::format("an event (a gate edge) at t = {} whose "
                                             "jump in the derivative was not applied: "
                                             "the run ends at it", tau));
            }
        }
    }

    /// A PWL diode's crossing at τ, found by the engine's locator. The
    /// crossing signal reads x[var] (an ON branch-form diode) or the node
    /// voltage difference from − to (slope 1). `fac`, `a` are a factor of the
    /// PRE-event mode `mask` (G + a·C/2); `src(t, b)` writes the time-varying
    /// source rows at t (for ẋ⁻: its algebraic part follows ḃ).
    template <class Src>
    [[gnu::noinline]] void state_event(Real tau, std::string_view name, Index from, Index to,
                                       Index var, const topology::SwitchStateMask& mask,
                                       const sparse::DirectSolver& fac, Real a, Src&& src) {
        refuse_newton_event(tau, name);
        const bool cur = n_stage_ > 0;   // the pre-event stage is not committed yet
        const Vector& xm = cur ? X_last_ : x_last_;
        const DenseMatrix& Sm = cur ? Sst_ : S_;
        const Vector& qdm = cur ? qdv_st_[static_cast<std::size_t>(n_stage_ - 1)] : qdv_n_;
        if (!(mask == mask_) || pend_.on) {
            throw UnsupportedError(engine_, "sensitivities", std::string(name), "Diode",
                std::format("sensitivities on engine '{}': at t = {} the diode '{}' "
                            "crosses at the instant of another event: a tie whose "
                            "order changes with p, without a derivative.",
                            engine_, tau, name));
        }
        auto gx = [&](const auto& v) -> Real {
            if (var >= 0) return v[var];
            return (stamping::node_is_active(from) ? Real(v[from]) : Real{0})
                   - (stamping::node_is_active(to) ? Real(v[to]) : Real{0});
        };
        // ḃ(τ) by a central difference of the sources (smooth near the event).
        // `src` writes the engines' b_extra rows, whose sign is the opposite
        // of b in f = G·x − b (the engine's rhs is −(b_const + b_extra)).
        const Real eps = Real{1e-7} * std::max(std::abs(tau), Real{1e-9}) + Real{1e-15};
        Vector bp = Vector::Zero(n_), bm = Vector::Zero(n_);
        src(tau + eps, bp);
        src(tau - eps, bm);
        const Vector bdot = (bm - bp) / (Real{2} * eps);
        // ẋ⁻: C/2·y = q̇⁻ and G·y − ḃ in range(C/2)
        bordered_(fac);
        const Vector y = solve_bordered_(fac, Vector(bdot + a * qdm), qdm, tau, name);
        const Real slope = gx(y);
        Real scale = std::abs(slope);
        if (var < 0) {
            scale = (stamping::node_is_active(from) ? std::abs(y[from]) : Real{0})
                    + (stamping::node_is_active(to) ? std::abs(y[to]) : Real{0});
        }
        if (!(std::abs(slope) > Real{1e-9} * scale) || slope == Real{0}) {
            throw UnsupportedError(engine_, "grazing_event", std::string(name), "Diode",
                std::format("sensitivities on engine '{}': at t = {} the diode '{}' "
                            "grazes its threshold (dg/dt = {:.3e}): one side of the "
                            "parameter removes the event and the derivative is "
                            "one-sided.", engine_, tau, name, slope));
        }
        Vector dtau(static_cast<Index>(params_.size()));
        for (Index j = 0; j < dtau.size(); ++j) dtau[j] = -gx(Sm.col(j)) / slope;
        (void)xm;
        add_pending_(tau, dtau, name);
    }

    /// A landing interpolated between two states of the current mode, below
    /// the engine's conditioning floor (no solve: TR-BDF2's turn-off at its
    /// bracket floor). Its length dt is up to about the step (the floor
    /// bounds the bracket's width, not its distance from t: 0.2–5 ns on the
    /// 40 T DCM buck at rtol 1e-10): dq/dp moves by dt·dq̇/dp (explicit
    /// Euler on the pre-event slot, O(dt²) per landing, a fixed number of
    /// landings: second order overall like the state), the slots are
    /// kept (q̇ at x_n: exact only for a jump-free turn-off, V_f = 0 — a
    /// turn-off with a jump is refused at the floor today), and S is
    /// re-initialised consistent with dq/dp at x1 by one
    /// bordered solve per parameter on `fac` (any factor of the mode, a =
    /// 2/its dt) — no factorization. Call after `begin_step`.
    [[gnu::noinline]] void carry(Real t1, const topology::SwitchStateMask& mask, const Vector& x1,
                                 const sparse::DirectSolver& fac, Real a) {
        if (!have_qx_ || !(mask == mask_) || pend_.on || n_stage_ != 0) {
            refuse_step(t1, "an interpolated diode landing in a mode the recurrence "
                            "did not reach by a stage");
        }
        dq_n_ += (t1 - t_last_) * dqd_n_;
        qv_n_ = Cq_ * x1;
        bordered_(fac);
        Vector qp(n_), fp(n_);
        for (Index j = 0; j < S_.cols(); ++j) {
            qp.setZero();
            fp.setZero();
            add_qp_(params_[static_cast<std::size_t>(j)], x1, qp);
            add_fp_(params_[static_cast<std::size_t>(j)], x1, fp);
            S_.col(j) = solve_bordered_(fac, Vector(-fp + a * (dq_n_.col(j) - qp)),
                                        Vector(dq_n_.col(j) - qp), t1, "a landing");
        }
        x_last_ = x1;
        t_last_ = t1;
        record_(t1);
    }

    /// True while an event waits for its first post-event stage.
    [[nodiscard]] bool event_pending() const noexcept { return pend_.on; }

    /// An accepted step that is not a sequence of this object's stages (an
    /// event landing, a split, a retry, a restart): refused by name.
    [[noreturn, gnu::noinline, gnu::cold]] void refuse_step(Real t, std::string_view what) const {
        throw UnsupportedError(engine_, "sensitivities", "", "",
            std::format("sensitivities on engine '{}': at t = {} the step was "
                        "{}; the sensitivity of such a step (its event or "
                        "restart) is not implemented on this engine (add-"
                        "parameter-sensitivities group 6). Use engine='dsed' "
                        "for a switched circuit.",
                        engine_, t, what));
    }

    void drop_stages_(int from) {
        for (int i = from; i < kMaxStages; ++i) {
            const auto ii = static_cast<std::size_t>(i);
            trace_.n_sens_discarded += st_solves_[ii] + st_iters_[ii];
            st_solves_[ii] = st_iters_[ii] = 0;
        }
    }

    void init_mode_(const topology::SwitchStateMask& mask) {
        mask_ = mask;
        split_(mask, G_, b_);
        have_qx_ = true;
        // the step start's q derivative: ∂q/∂p(x0), S0 = 0
        Vector qp(n_);
        for (std::size_t j = 0; j < params_.size(); ++j) {
            qp.setZero();
            add_qp_(params_[j], x_last_, qp);
            dq_n_.col(static_cast<Index>(j)) = qp;
        }
        qv_n_ = Cq_ * x_last_;   // the engine's own start: q(x0), q̇ = 0
        qdv_n_.setZero(n_);
    }

    void split_(const topology::SwitchStateMask& mask, sparse::Matrix& G, Vector& b) {
        sparse::Matrix C;
        sa_.split(graph_, pool_, mask, G, C, b);
        Cq_ = Real{0.5} * C;   // J = G + (1/factor_dt)·C = G + a·(C/2)
    }

    /// A moving event in a circuit with a Newton law: refused by name.
    void refuse_newton_event(Real tau, std::string_view what) const {
        if (!has_newton_) return;
        throw UnsupportedError(engine_, "sensitivities", std::string(what), "",
            std::format("sensitivities on engine '{}': at t = {} an event ({}) whose "
                        "instant depends on the parameters, in a circuit with a "
                        "Newton device: the consistent post-event field needs a "
                        "nonlinear re-initialisation, not implemented (add-"
                        "parameter-sensitivities group 6). Events at fixed instants "
                        "(gate edges of a fixed schedule) are carried.",
                        engine_, tau, what));
    }

    /// Registers an event at τ (pre-event values: the uncommitted last stage
    /// when there is one, else the committed step end). Two events at one
    /// instant (a cascade) must move together: a tie whose order changes
    /// with p has no derivative.
    void add_pending_(Real tau, const Vector& dtau, std::string_view what) {
        const bool cur = n_stage_ > 0;
        if (pend_.on) {
            const Real d = (pend_.dtau - dtau).cwiseAbs().maxCoeff();
            const Real m = std::max(pend_.dtau.cwiseAbs().maxCoeff(), dtau.cwiseAbs().maxCoeff());
            if (std::abs(tau - pend_.tau) > Real{1e-9} * std::max(std::abs(tau), Real{1e-12})
                || d > Real{1e-6} * m) {
                throw UnsupportedError(engine_, "sensitivities", std::string(what), "",
                    std::format("sensitivities on engine '{}': at t = {} two events "
                                "({} and {}) fall on one instant but move differently "
                                "with the parameters (dτ/dp differ by {:.3e}): their "
                                "order changes with p, a tie without a derivative.",
                                engine_, tau, pend_.what, what, d));
            }
            pend_.what += std::string(", ") + std::string(what);
            return;
        }
        pend_.on = true;
        pend_.tau = tau;
        pend_.dtau = dtau;
        pend_.what = std::string(what);
        if (dtau.cwiseAbs().maxCoeff() == Real{0}) return;
        refuse_newton_event(tau, what);
        pend_.xm = cur ? X_last_ : x_last_;
        pend_.qdm = cur ? qdv_st_[static_cast<std::size_t>(n_stage_ - 1)] : qdv_n_;
        pend_.fm = G_ * pend_.xm - b_;   // f⁻(x⁻) without the common sources
    }

    /// The first stage after a registered event: jump dq/dp, re-initialise.
    void apply_jump_(const topology::SwitchStateMask& mask, Real a,
                     const sparse::DirectSolver& fac, bool newton, Real t) {
        mask_ = mask;
        split_(mask, G_, b_);
        pend_.on = false;
        if (pend_.dtau.cwiseAbs().maxCoeff() == Real{0}) return;   // no jump
        if (newton) refuse_newton_event(t, pend_.what);
        // f⁺(x⁻) = f⁻(x⁻) + (switch stamps' change); f⁻(x⁻) = −q̇⁻
        const Vector r0 = -pend_.qdm + (G_ * pend_.xm - b_) - pend_.fm;
        bordered_(fac);
        const Vector delta = solve_bordered_(fac, Vector(-r0), Vector::Zero(n_), t, pend_.what);
        const Vector qdp = -(r0 + G_ * delta);   // q̇⁺ = −f⁺(x⁺), x⁺ = x⁻ + δ
        const Vector xp = pend_.xm + delta;
        dq_n_.noalias() += (pend_.qdm - qdp) * pend_.dtau.transpose();
        qdv_n_ = qdp;
        // S⁺ consistent with dq/dp⁺ in the new mode (C/2·S⁺ = dq⁺ − ∂q/∂p,
        // G⁺S⁺ + ∂f/∂p in range), and dq̇/dp⁺ = −(G⁺S⁺ + ∂f/∂p): the
        // derivative slot a following trapezoid reads
        Vector qp(n_), fp(n_);
        for (Index j = 0; j < S_.cols(); ++j) {
            qp.setZero();
            fp.setZero();
            add_qp_(params_[static_cast<std::size_t>(j)], xp, qp);
            add_fp_(params_[static_cast<std::size_t>(j)], xp, fp);
            const Vector sp = solve_bordered_(fac, Vector(-fp + a * (dq_n_.col(j) - qp)),
                                              Vector(dq_n_.col(j) - qp), t, pend_.what);
            S_.col(j) = sp;
            dqd_n_.col(j) = -(G_ * sp + fp);
        }
    }

    /// The bordered (index-1) solve of a consistent point on a factor M =
    /// G + a·C/2 of one mode: y = M⁻¹(c + (C/2)·w) with (C/2)·y = target on
    /// the rows C/2 touches. Its residual G·y + a·(C/2)·y − c lies in
    /// range(C/2) for every w, so y is consistent with the mode. `bordered_`
    /// costs |rows| back-substitutions and a dense QR of that size, shared by
    /// every right-hand side; each `solve_bordered_` one more substitution.
    /// A target that cannot be met (a new constraint on q: an impulsive
    /// event, the state itself jumps) is refused.
    void bordered_(const sparse::DirectSolver& M) {
        bz_.clear();
        {
            const Vector rs = Cq_.cwiseAbs() * Vector::Ones(n_);
            for (Index i = 0; i < n_; ++i) if (rs[i] != Real{0}) bz_.push_back(i);
        }
        const auto m = static_cast<Index>(bz_.size());
        Vector col(n_), u(n_);
        bU_.resize(n_, m);
        for (Index k = 0; k < m; ++k) {
            col = Cq_.col(bz_[static_cast<std::size_t>(k)]);
            M.solve(col, u);
            bU_.col(k) = u;
        }
        const DenseMatrix CU = Cq_ * bU_;
        bK_.resize(m, m);
        for (Index i = 0; i < m; ++i) bK_.row(i) = CU.row(bz_[static_cast<std::size_t>(i)]);
        bqr_.compute(bK_);
    }
    [[nodiscard]] Vector solve_bordered_(const sparse::DirectSolver& M, const Vector& c,
                                         const Vector& target, Real t, std::string_view what) {
        Vector y(n_);
        M.solve(c, y);
        const auto m = static_cast<Index>(bz_.size());
        if (m == 0) return y;
        const Vector Cy = Cq_ * y;
        Vector rhs(m);
        for (Index i = 0; i < m; ++i) {
            const Index zi = bz_[static_cast<std::size_t>(i)];
            rhs[i] = target[zi] - Cy[zi];
        }
        const Vector w = bqr_.solve(rhs);
        const Real res = (bK_ * w - rhs).cwiseAbs().maxCoeff();
        const Real ref = rhs.cwiseAbs().maxCoeff() + (bK_.cwiseAbs() * w.cwiseAbs()).maxCoeff();
        if (!(res <= Real{1e-8} * ref) && res > Real{0}) {
            throw UnsupportedError(engine_, "sensitivities", std::string(what), "",
                std::format("sensitivities on engine '{}': at t = {} the event ({}) "
                            "has no consistent continuation of the charges and "
                            "fluxes (residual {:.3e} of {:.3e}): the state itself "
                            "jumps (an impulsive commutation), whose reset map has "
                            "no derivative here.", engine_, t, what, res, ref));
        }
        y.noalias() += bU_ * w;
        return y;
    }

    void check_contiguous_(Real t, const Vector& x) const {
        if (t == t_last_ && x.size() == x_last_.size() && x == x_last_) return;
        refuse_step(t, std::format("taken from a state the sensitivity "
                                   "recurrence did not produce (last committed "
                                   "at t = {}): an event landing, a zero-time "
                                   "settle, a split or a retry moved it", t_last_));
    }

    [[nodiscard, gnu::noinline]] SensitivityTrace take_trace() { return std::move(trace_); }
    [[nodiscard]] const DenseMatrix& S() const noexcept { return S_; }

private:
    using SK = pwl::DevicePool::StoredKind;
    static constexpr int kMaxStages = 6;
    static constexpr int kMaxIterations = 50;

    struct Param {
        enum class Kind : std::uint8_t { R, C, L, Law, Duty };
        Kind kind = Kind::R;
        Index branch = -1;   // the pool's branch id
        Index gidx = -1;     // the graph's branch index
        Index from = -1, to = -1, row = -1;
        Real value = Real{0};   // Duty: the PWM period T (dτ/dD of a falling edge)
        SK sk = SK::Resistor;
        std::size_t field = 0;  // Duty: the switch's mask bit
    };

    /// Mask bit of the graph's branch index `b` (switches in branch order).
    [[nodiscard]] Size switch_bit_(Index b) const {
        Size k = 0;
        for (Index i = 0; i < b; ++i)
            if (graph_.branch(i).kind == topology::BranchKind::Switch) ++k;
        return k;
    }
    [[nodiscard]] std::string switch_names_(const topology::SwitchStateMask& a,
                                            const topology::SwitchStateMask& b) const {
        std::string s;
        Size k = 0;
        for (Index i = 0; i < graph_.num_branches(); ++i) {
            if (graph_.branch(i).kind != topology::BranchKind::Switch) continue;
            if (k < a.size() && k < b.size() && a.get(k) != b.get(k)) {
                s += (s.empty() ? "'" : ", '") + std::string(graph_.branch_name(i))
                     + (b.get(k) ? "' turns ON" : "' turns OFF");
            }
            ++k;
        }
        return s;
    }

    void refuse_kinds_() const {
        for (Index b = 0; b < graph_.num_branches(); ++b) {
            const auto& br = graph_.branch(b);
            const SK k = pool_.kind_of(br.id);
            const char* what = nullptr;
            switch (k) {
            case SK::NonlinearCapacitor: what = "NonlinearCapacitor"; break;
            case SK::SaturableInductor: what = "SaturableInductor"; break;
            case SK::LauritzenDiode: what = "LauritzenDiode"; break;
            case SK::PmsmMna: what = "PmsmMna"; break;
            case SK::InductionMachineMna: what = "InductionMachineMna"; break;
            case SK::ExternalDevice: what = "ExternalDevice"; break;
            case SK::IgbtLevel1: what = "IgbtLevel1"; break;
            default: break;
            }
            if (what != nullptr) {
                throw UnsupportedError(engine_, "sensitivities",
                    std::string(graph_.branch_name(b)), what,
                    std::format("sensitivities on engine '{}': '{}' ({}) keeps "
                                "its own state history (or a tail), whose "
                                "sensitivity is not carried by the stage "
                                "recurrence yet. Supported: R, L, C, sources, "
                                "switches in one mode, and the static Newton "
                                "laws (Shockley, smooth ideal diode, Level-1 "
                                "MOSFET, excess loss).",
                                engine_, graph_.branch_name(b), what));
            }
        }
    }

    [[nodiscard]] Param resolve_(const std::string& nm) const {
        const auto dot = nm.substr(0, nm.find("#T=")).rfind('.');
        if (dot == std::string::npos || dot == 0 || dot + 1 == nm.size()) {
            throw std::invalid_argument("sensitivities: '" + nm +
                                        "' is not '<component>.<field>'.");
        }
        std::string comp = nm.substr(0, dot);
        std::string fld = nm.substr(dot + 1);
        // '<switch>.duty#T=<period>': Python checked the gate is a fixed-period
        // trailing-edge PWM of this switch alone (sensitivity._duty_period).
        Real period = Real{0};
        if (const auto hash = fld.find("#T="); hash != std::string::npos) {
            const std::string tv = fld.substr(hash + 3);
            if (!numeric::parse_real(tv.data(), tv.data() + tv.size(), period)
                || !(period > Real{0})) {
                throw std::invalid_argument("sensitivities: '" + nm + "': bad period.");
            }
            fld = fld.substr(0, hash);
        }
        Index bid = -1;
        for (Index b = 0; b < graph_.num_branches(); ++b) {
            if (graph_.branch_name(b) == comp) { bid = b; break; }
        }
        if (bid < 0) {
            throw std::invalid_argument("sensitivities: '" + nm +
                                        "': no component named '" + comp + "'.");
        }
        const auto& br = graph_.branch(bid);
        Param p;
        p.branch = br.id;
        p.gidx = bid;
        p.from = br.from;
        p.to = br.to;
        p.sk = pool_.kind_of(br.id);
        auto refuse = [&](const std::string& have) -> Param {
            throw UnsupportedError(engine_, "sensitivity_parameter", comp, fld,
                std::format("sensitivities: '{}': field '{}' of '{}' is not a "
                            "differentiable parameter on engine '{}' "
                            "(differentiable fields: {}).",
                            nm, fld, comp, engine_, have));
        };
        if (fld == "duty" && (p.sk == SK::Switch || p.sk == SK::Diode)) {
            if (p.sk == SK::Diode || !(period > Real{0})) {
                return refuse("'duty' only through simulate() on a switch driven "
                              "by make_pwm_switch_fn");
            }
            p.kind = Param::Kind::Duty;
            p.value = period;
            p.field = switch_bit_(bid);
            return p;
        }
        switch (p.sk) {
        case SK::Resistor:
            if (fld != "R" && fld != "R_ohms") return refuse("['R', 'R_ohms']");
            p.kind = Param::Kind::R;
            p.value = Real{1} / pool_.resistor_params(br.id).G;
            return p;
        case SK::Capacitor:
            if (fld != "C" && fld != "C_farads") return refuse("['C', 'C_farads']");
            p.kind = Param::Kind::C;
            p.value = pool_.capacitor_params(br.id).C;
            return p;
        case SK::Inductor:
            if (fld != "L" && fld != "L_henries") return refuse("['L', 'L_henries']");
            for (const auto& tc : pool_.transformer_couplings()) {
                if (tc.primary_branch_id == br.id || tc.secondary_branch_id == br.id) {
                    return refuse("none (a coupled winding: its mutual inductance "
                                  "depends on it)");
                }
            }
            p.kind = Param::Kind::L;
            p.value = pool_.inductor_params(br.id).L;
            p.row = pool_.branch_var_id_for_inductor(br.id, graph_);
            return p;
        default: break;
        }
        bool found = false;
        with_law_(p.sk, br, [&]<class L, std::size_t R>(const std::array<Index, R>&,
                                                     const typename L::template Params<Real>&) {
            for (std::size_t k = 0; k < L::kParamNames.size(); ++k) {
                if (L::kParamNames[k] == fld) { p.field = k; found = true; }
            }
            if (!found) {
                std::string have = "[";
                for (std::size_t k = 0; k < L::kParamNames.size(); ++k) {
                    have += (k ? ", '" : "'") + std::string(L::kParamNames[k]) + "'";
                }
                (void)refuse(have + "]");
            }
        });
        if (!found) return refuse("none");
        p.kind = Param::Kind::Law;
        return p;
    }

    /// Calls f.template operator()<Law, R>(terminals, params) for a static
    /// Newton kind; does nothing for any other kind.
    template <class F>
    void with_law_(SK sk, const topology::Branch& br, F&& f) const {
        namespace L = device::laws;
        switch (sk) {
        case SK::NonlinearDiode:
            f.template operator()<L::IdealDiode, 2>(
                std::array<Index, 2>{br.from, br.to},
                L::IdealDiode::from(pool_.nonlinear_diode_params(br.id)));
            break;
        case SK::ShockleyDiode:
            f.template operator()<L::ShockleyDiode, 2>(
                std::array<Index, 2>{br.from, br.to},
                L::ShockleyDiode::from(pool_.shockley_diode_params(br.id)));
            break;
        case SK::ExcessLoss:
            f.template operator()<L::ExcessLoss, 2>(
                std::array<Index, 2>{br.from, br.to},
                L::ExcessLoss::from(pool_.excess_loss_params(br.id)));
            break;
        case SK::MosfetLevel1:
            f.template operator()<L::MosfetLevel1, 3>(
                pwl::detail::mosfet_level1_terminals(pool_, br),
                L::MosfetLevel1::from(pool_.mosfet_level1_params(br.id)));
            break;
        default: break;
        }
    }

    void add_fp_(const Param& p, const Vector& X, Vector& fp) const {
        if (p.kind == Param::Kind::R) {
            // f = G·v, G = 1/R: ∂f/∂R = −G²·v on the from row, + on the to row
            const Real G = pool_.resistor_params(p.branch).G;
            const Real v = stamping::read_node_voltage(X, p.from)
                           - stamping::read_node_voltage(X, p.to);
            if (stamping::node_is_active(p.from)) fp[p.from] -= G * G * v;
            if (stamping::node_is_active(p.to)) fp[p.to] += G * G * v;
            return;
        }
        if (p.kind != Param::Kind::Law) return;
        with_law_(p.sk, graph_.branch(p.gidx),
                  [&]<class L, std::size_t R>(const std::array<Index, R>& at,
                                              const typename L::template Params<Real>& prm) {
            constexpr std::size_t NP = L::n_params;
            const auto ps = device::seed_params<L, NP>(prm);
            const std::array<Real, R> xv = pwl::iface::gather<R>(X, at, R);
            device::Out<ad::ADRealN<NP>, R> o{};
            device::eval<L>(device::aux_t<L>{}, ps, xv, o);
            for (std::size_t r = 0; r < R; ++r) {
                if (stamping::node_is_active(at[r])) fp[at[r]] += o.f[r].deriv(p.field);
            }
        });
    }

    void add_qp_(const Param& p, const Vector& X, Vector& qp) const {
        if (p.kind == Param::Kind::C) {
            // q = C·v on the from row, −C·v on the to row
            const Real v = stamping::read_node_voltage(X, p.from)
                           - stamping::read_node_voltage(X, p.to);
            if (stamping::node_is_active(p.from)) qp[p.from] += v;
            if (stamping::node_is_active(p.to)) qp[p.to] -= v;
        } else if (p.kind == Param::Kind::L) {
            // the flux row: q = −L·i (its ∂q/∂x entry, read from the matrix)
            qp[p.row] += (have_qx_ ? Cq_.coeff(p.row, p.row) / p.value : Real{-1})
                         * X[p.row];
        }
    }

    void record_(Real t) {
        trace_.times.push_back(t);
        for (Index j = 0; j < S_.cols(); ++j)
            for (Index i = 0; i < n_; ++i) trace_.states.push_back(S_(i, j));
    }

    const topology::Graph& graph_;
    const pwl::DevicePool& pool_;
    const char* engine_;
    Real tol_;
    std::vector<Param> params_;
    Index n_ = 0;
    DenseMatrix S_, dq_n_, dqd_n_, Sst_;
    std::array<DenseMatrix, kMaxStages> dq_st_, dqd_st_;
    int n_stage_ = 0;
    std::array<std::uint64_t, kMaxStages> st_solves_{}, st_iters_{};
    // the values' recurrence (q, q̇ as the engine's slots) and the event state
    Vector qv_n_, qdv_n_;
    std::array<Vector, kMaxStages> qv_st_, qdv_st_;
    std::vector<Index> bz_;
    DenseMatrix bU_, bK_;
    Eigen::ColPivHouseholderQR<DenseMatrix> bqr_;
    bool has_newton_ = false;
    struct Pending {
        bool on = false;
        Real tau = Real{0};
        Vector dtau, xm, qdm, fm;
        std::string what;
    } pend_;
    sparse::Matrix G_;
    Vector b_;
    pwl::SegmentAssembler sa_;
    Vector x_last_, X_last_;
    Real t_last_ = Real{0};
    bool have_qx_ = false;
    topology::SwitchStateMask mask_;
    sparse::Matrix Cq_;
    pwl::NonlinearRefreshFn nl_;
    sparse::Matrix Jnl_;
    Vector fnl_;
    SensitivityTrace trace_;
};

}  // namespace pulsim::step
