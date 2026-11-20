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
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "pulsim/device/device.hpp"
#include "pulsim/device/laws.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/assemble.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/device_refresh.hpp"
#include "pulsim/pwl/history_state.hpp"
#include "pulsim/pwl/interface_stamps.hpp"
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
    StageSensitivity(const topology::Graph& graph, const pwl::DevicePool& pool,
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
        for (std::size_t i = 0; i < names.size(); ++i)
            for (std::size_t j = 0; j < i; ++j)
                if (names[i] == names[j])
                    throw std::invalid_argument(
                        "sensitivities: duplicate name '" + names[i] + "'.");
        trace_.params = names;
    }

    [[nodiscard]] bool on() const noexcept { return !params_.empty(); }

    /// The run's start: S = 0, dq/dp = ∂q/∂p(x0), dq̇/dp = 0 (every engine
    /// starts from zero derivative slots). `x0` must not depend on the
    /// parameters (the engine refuses a DC-operating-point start; a start
    /// the engine moves before its first step is caught by `begin_step`).
    void start(Real t0, const Vector& x0) {
        n_ = static_cast<Index>(x0.size());
        const auto np = static_cast<Index>(params_.size());
        S_.setZero(n_, np);
        dqd_n_.setZero(n_, np);
        dq_n_.setZero(n_, np);
        x_last_ = x0;
        t_last_ = t0;
        have_qx_ = false;
        trace_.n_state = static_cast<Size>(n_);
        record_(t0);
    }

    /// A step (attempt) begins at (t0, x0): the earlier attempt's stages are
    /// discarded. The engine must start it from the state this object last
    /// committed; anything else (an event landing, a probe or a split that
    /// committed a state no stage here produced) is refused by name.
    void begin_step(Real t0, const Vector& x0) {
        n_stage_ = 0;
        trace_.n_sens_discarded += pend_solves_ + pend_iters_;
        pend_solves_ = pend_iters_ = 0;
        check_contiguous_(t0, x0);
    }

    /// The run ended at (t, x): the last committed state must be ours.
    void finish(Real t, const Vector& x) const { check_contiguous_(t, x); }

    /// Stage `s` (0-based in the step) converged at X under `mask`. `fac`
    /// holds the factor the stage solved with; `J_lin` is the linear part of
    /// a Newton stage (nullptr: the factor is the stage's exact matrix).
    void stage(int s, const StagePsi& psi, const topology::SwitchStateMask& mask,
               const Vector& X, Real t, const sparse::DirectSolver& fac,
               const sparse::Matrix* J_lin) {
        if (!have_qx_) {
            mask_ = mask;
            const Vector& x0_ = x_last_;
            sparse::Matrix G, C;
            Vector b;
            pwl::SegmentAssembler sa;
            sa.split(graph_, pool_, mask, G, C, b);
            Cq_ = Real{0.5} * C;   // J = G + (1/factor_dt)·C = G + a·(C/2)
            have_qx_ = true;
            // the step start's q derivative: ∂q/∂p(x0), S0 = 0
            Vector qp(n_);
            for (std::size_t j = 0; j < params_.size(); ++j) {
                qp.setZero();
                add_qp_(params_[j], x0_, qp);
                dq_n_.col(static_cast<Index>(j)) = qp;
            }
        } else if (!(mask == mask_)) {
            throw UnsupportedError(engine_, "sensitivities", "", "",
                std::format("sensitivities on engine '{}': at t = {} the mode "
                            "changed ({} -> {}); the jump correction of an "
                            "event (a diode commutation, a gate edge) is not "
                            "implemented on this engine (add-parameter-"
                            "sensitivities group 6). Use engine='dsed' for a "
                            "switched circuit.",
                            engine_, t, mask_.to_string(), mask.to_string()));
        }
        if (s != n_stage_ || s >= kMaxStages) {
            throw std::logic_error("StageSensitivity: stages out of order");
        }
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
            ++pend_solves_;
            if (newton) {
                // modified Newton: iterate on the stale factor (5.2)
                for (int it = 0;; ++it) {
                    r.noalias() = *J_lin * S;
                    r.noalias() += Jnl_ * S;
                    r = rhs - r;
                    fac.solve(r, d);
                    ++pend_iters_;
                    S += d;
                    const Real dn = d.cwiseAbs().maxCoeff();
                    const Real sn = S.cwiseAbs().maxCoeff();
                    if (!(dn > tol_ * sn)) break;
                    if (it >= kMaxIterations || !std::isfinite(dn)) {
                        throw std::runtime_error(std::format(
                            "sensitivities on engine '{}': at t = {} the "
                            "sensitivity solve of '{}' did not converge on the "
                            "stage's factor ({} iterations, last correction "
                            "{:.3e} of {:.3e}).",
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
        X_last_ = X;
        n_stage_ = s + 1;
    }

    /// The step was accepted at time `t`: its last stage is the new step start.
    void commit(Real t) {
        if (n_stage_ == 0) {
            throw std::logic_error("StageSensitivity: commit without a stage");
        }
        const auto k = static_cast<std::size_t>(n_stage_ - 1);
        dq_n_ = dq_st_[k];
        dqd_n_ = dqd_st_[k];
        S_ = Sst_;
        x_last_ = X_last_;
        t_last_ = t;
        n_stage_ = 0;
        trace_.n_sens_solves += pend_solves_;   // the accepted attempt's solves
        trace_.n_sens_iterations += pend_iters_;
        pend_solves_ = pend_iters_ = 0;
        record_(t);
    }

    /// An accepted step that is not a sequence of this object's stages (an
    /// event landing, a split, a retry, a restart): refused by name.
    [[noreturn]] void refuse_step(Real t, std::string_view what) const {
        throw UnsupportedError(engine_, "sensitivities", "", "",
            std::format("sensitivities on engine '{}': at t = {} the step was "
                        "{}; the sensitivity of such a step (its event or "
                        "restart) is not implemented on this engine (add-"
                        "parameter-sensitivities group 6). Use engine='dsed' "
                        "for a switched circuit.",
                        engine_, t, what));
    }

    void check_contiguous_(Real t, const Vector& x) const {
        if (t == t_last_ && x.size() == x_last_.size() && x == x_last_) return;
        refuse_step(t, std::format("taken from a state the sensitivity "
                                   "recurrence did not produce (last committed "
                                   "at t = {}): an event landing, a zero-time "
                                   "settle, a split or a retry moved it", t_last_));
    }

    [[nodiscard]] SensitivityTrace take_trace() { return std::move(trace_); }
    [[nodiscard]] const DenseMatrix& S() const noexcept { return S_; }

private:
    using SK = pwl::DevicePool::StoredKind;
    static constexpr int kMaxStages = 6;
    static constexpr int kMaxIterations = 50;

    struct Param {
        enum class Kind : std::uint8_t { R, C, L, Law };
        Kind kind = Kind::R;
        Index branch = -1;   // the pool's branch id
        Index gidx = -1;     // the graph's branch index
        Index from = -1, to = -1, row = -1;
        Real value = Real{0};
        SK sk = SK::Resistor;
        std::size_t field = 0;
    };

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
        const auto dot = nm.rfind('.');
        if (dot == std::string::npos || dot == 0 || dot + 1 == nm.size()) {
            throw std::invalid_argument("sensitivities: '" + nm +
                                        "' is not '<component>.<field>'.");
        }
        const std::string comp = nm.substr(0, dot);
        const std::string fld = nm.substr(dot + 1);
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
    std::uint64_t pend_solves_ = 0, pend_iters_ = 0;
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
