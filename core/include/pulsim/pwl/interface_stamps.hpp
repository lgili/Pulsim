#pragma once

// =============================================================================
// Pulsim — the Newton kinds stamped through their `Device` laws
// =============================================================================
//
// add-unified-device-interface, dual path (DECISIONS #23). The engines run
// these instead of the hand stamps when the run asks for
// `device::DevicePath::Interface`. Per kind: gather the instance's rows from
// the global x (pins: node voltages, ground reads 0; internal rows: x), map
// the history it keeps into per-row (q_n, q̇_n, q_γ), call the generic
// `device::stamp_fq` / `commit_fq`, and scatter F and the law's structural
// Jacobian entries (`kPattern`) on the active rows and columns. The step
// comes ONLY from the `StepCtx` of the solve.
//
// Storage: the instance state stays in each history's entries (which the
// flat device-state vector already serialises, rollback included); the
// interface path reads and writes it through the law.

#include "pulsim/device/device.hpp"
#include "pulsim/device/device_path.hpp"
#include "pulsim/device/device_state.hpp"
#include "pulsim/device/laws.hpp"
#include "pulsim/device/machine_laws.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/igbt_tail_history.hpp"
#include "pulsim/pwl/lauritzen_diode_history.hpp"
#include "pulsim/pwl/nonlinear_capacitor_history.hpp"
#include "pulsim/pwl/pmsm_mna_history.hpp"
#include "pulsim/pwl/saturable_inductor_history.hpp"
#include "pulsim/pwl/trbdf2_stage.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/topology/graph.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pulsim::pwl::iface {

[[nodiscard]] inline device::Formula formula_of(TrBdf2Stage s) noexcept {
    switch (s) {
    case TrBdf2Stage::BackwardEuler: return device::Formula::BackwardEuler;
    case TrBdf2Stage::Bdf2Stage2:    return device::Formula::Bdf2Stage2;
    default:                         return device::Formula::Trapezoidal;
    }
}
[[nodiscard]] inline device::StepCtx ctx_of(Real h, TrBdf2Stage s) noexcept {
    return device::StepCtx{Real{0}, formula_of(s), h};
}

template <std::size_t R>
[[nodiscard]] inline std::array<Real, R> gather(
    const Vector& x, const std::array<Index, R>& at, std::size_t n_pins) {
    std::array<Real, R> v{};
    for (std::size_t r = 0; r < R; ++r) {
        v[r] = r < n_pins ? stamping::read_node_voltage(x, at[r]) : x[at[r]];
    }
    return v;
}

namespace detail {
/// The law's structural entries as (row, col) pairs, row-major.
template <class D, std::size_t R>
consteval auto pattern_entries() {
    constexpr std::size_t n = [] {
        std::size_t c = 0;
        for (std::size_t k = 0; k < R; ++k)
            for (std::size_t r = 0; r < R; ++r) c += D::kPattern[r][k] ? 1 : 0;
        return c;
    }();
    std::array<std::array<std::size_t, 2>, n> out{};
    std::size_t i = 0;
    for (std::size_t r = 0; r < R; ++r)
        for (std::size_t k = 0; k < R; ++k)
            if (D::kPattern[r][k]) out[i++] = {r, k};
    return out;
}
}  // namespace detail

// Forced inlining of the scatter and its unrolled folds: left to the
// heuristic, clang keeps the fold over the pattern out of line (five
// inlined `coeffRef`) and that call costs ~4 ns of a 14 ns saturable-
// inductor stamp; a runtime loop over the pattern is not unrolled and costs
// twice the stamp (task 15.7, `.scratch/bench_stamp.cpp` measurements).
#if defined(__GNUC__) || defined(__clang__)
#define PULSIM_IFACE_INLINE __attribute__((always_inline))
#else
#define PULSIM_IFACE_INLINE
#endif

/// Add F on the active rows and J on the law's structural entries whose
/// row and column are both active, unrolled at compile time.
template <class D, std::size_t R>
PULSIM_IFACE_INLINE inline void scatter(const std::array<Index, R>& at,
                                        const std::array<Real, R>& F,
                                        const std::array<std::array<Real, R>, R>& J,
                                        sparse::Matrix& J_nl, Vector& f_nl) {
    // A row the law never fills (a gate) gets no F either, as by hand.
    static constexpr auto kRowUsed = [] {
        std::array<bool, R> u{};
        for (std::size_t r = 0; r < R; ++r)
            for (std::size_t k = 0; k < R; ++k) u[r] = u[r] || D::kPattern[r][k];
        return u;
    }();
    static constexpr auto kEntries = detail::pattern_entries<D, R>();
    const auto on = [&]<std::size_t... I>(std::index_sequence<I...>) PULSIM_IFACE_INLINE {
        return std::array<bool, R>{(I >= D::n_pins || stamping::node_is_active(at[I]))...};
    }(std::make_index_sequence<R>{});
    [&]<std::size_t... I>(std::index_sequence<I...>) PULSIM_IFACE_INLINE {
        ((kRowUsed[I] && on[I] ? (void)(f_nl[at[I]] += F[I]) : void()), ...);
    }(std::make_index_sequence<R>{});
    [&]<std::size_t... E>(std::index_sequence<E...>) PULSIM_IFACE_INLINE {
        ((on[kEntries[E][0]] && on[kEntries[E][1]]
              ? (void)(J_nl.coeffRef(at[kEntries[E][0]], at[kEntries[E][1]])
                       += J[kEntries[E][0]][kEntries[E][1]])
              : void()),
         ...);
    }(std::make_index_sequence<kEntries.size()>{});
}

// ---- static kinds -----------------------------------------------------------
/// One static branch through law `L`. Returns |i|.
template <class L, std::size_t R>
PULSIM_IFACE_INLINE inline Real stamp_one_static(const Vector& x, sparse::Matrix& J_nl, Vector& f_nl,
                             const std::array<Index, R>& at,
                             const typename L::template Params<Real>& p) {
    static_assert(R == device::rows_v<L>);
    constexpr std::array<Real, R> zero{};
    constexpr device::StepCtx dc{Real{0}, device::Formula::Dc, Real{0}};
    std::array<Real, R> F{};
    std::array<std::array<Real, R>, R> J{};
    device::stamp_fq<L>(device::aux_t<L>{}, p, gather<R>(x, at, R), dc, zero, zero,
                        zero, F, J);
    scatter<L>(at, F, J, J_nl, f_nl);
    return std::abs(F[0]);
}

// ---- the nonlinear capacitor (q-only) --------------------------------------
using CossLaw = device::laws::NonlinearCapacitor;

/// One Coss instance (`qg` = Q at the BDF2 stage point, else unused).
/// Returns |i|.
PULSIM_IFACE_INLINE inline Real stamp_entry(const NonlinearCapacitorHistory::Entry& e, const Vector& x,
                        sparse::Matrix& J_nl, Vector& f_nl,
                        const device::StepCtx& c, Real qg) {
    const std::array<Index, 2> at{e.from, e.to};
    std::array<Real, 2> F{};
    std::array<std::array<Real, 2>, 2> J{};
    device::stamp_fq<CossLaw>(CossLaw::aux_from(e.params), CossLaw::from(e.params),
                              gather<2>(x, at, 2), c, {e.q_prev, -e.q_prev},
                              {e.i_prev, -e.i_prev}, {qg, -qg}, F, J);
    scatter<CossLaw>(at, F, J, J_nl, f_nl);
    return std::abs(F[0]);
}

inline Real stamp(NonlinearCapacitorHistory& k, const Vector& x,
                  sparse::Matrix& J_nl, Vector& f_nl, Real h, TrBdf2Stage st,
                  bool /*inc*/) {
    if (k.empty() || !(h > Real{0})) return Real{0};
    const auto c = ctx_of(h, st);
    const bool bdf2 = st == TrBdf2Stage::Bdf2Stage2;
    if (bdf2 && k.q_gamma().size() != k.entries().size()) {
        throw std::invalid_argument(
            "iface::stamp(NonlinearCapacitor): the BDF2 stage needs Q at the "
            "stage point, one per device");
    }
    Real m = Real{0};
    std::size_t idx = 0;
    for (const auto& e : k.entries()) {
        const Real qg = bdf2 ? k.q_gamma()[idx] : Real{0};
        ++idx;
        m = std::max(m, stamp_entry(e, x, J_nl, f_nl, c, qg));
    }
    return m;
}

inline void commit(NonlinearCapacitorHistory& k, const Vector& x, Real h,
                   TrBdf2Stage st) {
    const bool bdf2 = st == TrBdf2Stage::Bdf2Stage2;
    if (k.empty() || (!bdf2 && !(h > Real{0}))) return;
    const auto c = ctx_of(h, st);
    std::size_t idx = 0;
    for (auto& e : k.mutable_entries()) {
        const auto xl = gather<2>(x, {e.from, e.to}, 2);
        const Real qg = bdf2 ? k.q_gamma()[idx] : Real{0};
        ++idx;
        std::array<Real, 2> qn{e.q_prev, -e.q_prev};
        std::array<Real, 2> qd{e.i_prev, -e.i_prev};
        (void)device::commit_fq<CossLaw>(CossLaw::aux_from(e.params),
                                         CossLaw::from(e.params), xl, c, qn,
                                         qd, {qg, -qg});
        e.v_prev = xl[0] - xl[1];
        e.q_prev = qn[0];
        e.i_prev = qd[0];
    }
}

inline void capture_gamma(NonlinearCapacitorHistory& k, const Vector& xg) {
    std::size_t idx = 0;
    for (const auto& e : k.entries()) {
        device::Out<Real, 2> o;
        device::eval<CossLaw>(CossLaw::aux_from(e.params), CossLaw::from(e.params),
                              gather<2>(xg, {e.from, e.to}, 2), o);
        k.mutable_q_gamma()[idx++] = o.q[0];
    }
}

// ---- the saturable inductor (flux row, internal branch current) ------------
using SatLaw = device::laws::SaturableInductor;

[[nodiscard]] inline SatLaw::Aux sat_aux(const SaturableInductorHistory::Entry& e,
                                         TrBdf2Stage st) {
    return {e.params.table.get(), e.params.ja.get(),
            &SaturableInductorHistory::ja_base(e, st)};
}

/// One saturable inductor under the stage `st` of a step `h`. Returns |i|.
PULSIM_IFACE_INLINE inline Real stamp_entry(const SaturableInductorHistory::Entry& e, const Vector& x,
                        sparse::Matrix& J_nl, Vector& f_nl,
                        const device::StepCtx& c, TrBdf2Stage st, bool inc) {
    const bool bdf2 = st == TrBdf2Stage::Bdf2Stage2;
    const std::array<Index, 3> at{e.from, e.to, e.branch_var_id};
    const auto xl = gather<3>(x, at, 2);
    std::array<Real, 3> F{};
    std::array<std::array<Real, 3>, 3> J{};
    // The increment form (TR-BDF2): a·(q − q_n) in closed form, for
    // the atan law (the hand stamp's `incr` condition).
    const bool incr = inc && !e.params.ja && !e.params.table;
    std::array<Real, 3> dq{}, dqg{};
    if (incr) {
        dq[2] = -models::SaturableInductor::flux_increment(xl[2], e.i_L_old, e.params);
        if (bdf2) {
            dqg[2] = -models::SaturableInductor::flux_increment(e.i_gamma, e.i_L_old,
                                                               e.params);
        }
    }
    device::stamp_fq<SatLaw>(sat_aux(e, st), SatLaw::from(e.params), xl, c,
                             {Real{0}, Real{0}, -e.lambda_old},
                             {Real{0}, Real{0}, -e.V_L_old},
                             {Real{0}, Real{0}, -e.lambda_gamma}, F, J,
                             incr ? &dq : nullptr, incr ? &dqg : nullptr);
    scatter<SatLaw>(at, F, J, J_nl, f_nl);
    return std::abs(xl[2]);
}

inline Real stamp(SaturableInductorHistory& k, const Vector& x,
                  sparse::Matrix& J_nl, Vector& f_nl, Real h, TrBdf2Stage st,
                  bool inc) {
    const auto c = ctx_of(h, st);
    Real m = Real{0};
    for (const auto& e : k.entries()) {
        m = std::max(m, stamp_entry(e, x, J_nl, f_nl, c, st, inc));
    }
    return m;
}

inline void commit(SaturableInductorHistory& k, const Vector& x, Real h,
                   TrBdf2Stage st) {
    const auto c = ctx_of(h, st);
    for (auto& e : k.mutable_entries()) {
        const auto xl = gather<3>(x, {e.from, e.to, e.branch_var_id}, 2);
        std::array<Real, 3> qn{Real{0}, Real{0}, -e.lambda_old};
        std::array<Real, 3> qd{Real{0}, Real{0}, -e.V_L_old};
        (void)device::commit_fq<SatLaw>(sat_aux(e, st), SatLaw::from(e.params),
                                        xl, c, qn, qd,
                                        {Real{0}, Real{0}, -e.lambda_gamma});
        e.i_L_old = xl[2];
        e.V_L_old = -qd[2];
        e.lambda_old = -qn[2];
        // The hysteretic law's State::accept: (H, M), direction, sub-steps.
        if (e.params.ja) SaturableInductorHistory::accept_ja(e, st);
    }
}

inline void capture_gamma(SaturableInductorHistory& k, const Vector& xg) {
    for (auto& e : k.mutable_entries()) {
        e.i_gamma = xg[e.branch_var_id];
        if (e.params.ja) {
            SaturableInductorHistory::capture_ja(e);
            continue;
        }
        device::Out<Real, 3> o;
        device::eval<SatLaw>(sat_aux(e, TrBdf2Stage::Trapezoidal),
                             SatLaw::from(e.params),
                             gather<3>(xg, {e.from, e.to, e.branch_var_id}, 2), o);
        e.lambda_gamma = -o.q[2];
    }
}

// ---- stored-charge kinds (one local row, condensed) -------------------------
/// One instance of a law with a local charge row: gather the pins, start
/// the local unknown at the committed charge, condense, scatter. `xl` comes
/// back holding the solved charge.
template <class Law, std::size_t NP>
PULSIM_IFACE_INLINE inline Real stamp_local(const std::array<Index, NP>& at,
                                            std::array<Real, NP + 1>& xl,
                                            const typename Law::template Params<Real>& p,
                                            const Vector& x, const device::StepCtx& c,
                                            Real q_n, Real qdot_n, Real q_g,
                                            sparse::Matrix* J_nl, Vector* f_nl) {
    const auto v = gather<NP>(x, at, NP);
    for (std::size_t r = 0; r < NP; ++r) xl[r] = v[r];
    xl[NP] = q_n;
    std::array<Real, NP + 1> qn{}, qd{}, qg{};
    qn[NP] = q_n;
    qd[NP] = qdot_n;
    qg[NP] = q_g;
    std::array<Real, NP> F{};
    std::array<std::array<Real, NP>, NP> J{};
    device::stamp_fq_local<Law>({}, p, xl, c, qn, qd, qg, F, J);
    if (J_nl != nullptr) scatter<Law>(at, F, J, *J_nl, *f_nl);
    return std::abs(F[0]);
}

/// Commit one local-charge instance: solve the local row at x, then the
/// contract's commit (q_n ← q, q̇_n ← −f on the local row).
template <class Law, std::size_t NP>
inline void commit_local(const std::array<Index, NP>& at,
                         const typename Law::template Params<Real>& p, const Vector& x,
                         const device::StepCtx& c, Real& q_n, Real& qdot_n, Real q_g) {
    std::array<Real, NP + 1> xl{};
    (void)stamp_local<Law, NP>(at, xl, p, x, c, q_n, qdot_n, q_g, nullptr, nullptr);
    std::array<Real, NP + 1> qn{}, qd{}, qg{};
    qn[NP] = q_n;
    qd[NP] = qdot_n;
    qg[NP] = q_g;
    (void)device::commit_fq<Law>({}, p, xl, c, qn, qd, qg);
    q_n = qn[NP];
    qdot_n = qd[NP];
}

using LauritzenLaw = device::laws::LauritzenDiode;
using IgbtTailLaw = device::laws::IgbtTail;

[[nodiscard]] inline std::array<Index, 2> pins_of(const LauritzenDiodeHistory::Entry& e) {
    return {e.from, e.to};
}
[[nodiscard]] inline std::array<Index, 3> pins_of(const IgbtTailHistory::Entry& e) {
    return {e.collector, e.emitter, e.gate};
}
// The three members the two histories name differently.
[[nodiscard]] inline Real& q_of(LauritzenDiodeHistory::Entry& e) { return e.q_m_prev; }
[[nodiscard]] inline Real& q_of(IgbtTailHistory::Entry& e) { return e.q_prev; }
[[nodiscard]] inline Real q_of(const LauritzenDiodeHistory::Entry& e) { return e.q_m_prev; }
[[nodiscard]] inline Real q_of(const IgbtTailHistory::Entry& e) { return e.q_prev; }
[[nodiscard]] inline Real& qg_of(LauritzenDiodeHistory::Entry& e) { return e.q_m_gamma; }
[[nodiscard]] inline Real& qg_of(IgbtTailHistory::Entry& e) { return e.q_gamma; }
[[nodiscard]] inline Real qg_of(const LauritzenDiodeHistory::Entry& e) { return e.q_m_gamma; }
[[nodiscard]] inline Real qg_of(const IgbtTailHistory::Entry& e) { return e.q_gamma; }
template <class H> struct LocalLawOf;
template <> struct LocalLawOf<LauritzenDiodeHistory> { using type = LauritzenLaw; };
template <> struct LocalLawOf<IgbtTailHistory> { using type = IgbtTailLaw; };

template <class H>
concept LocalChargeHistory = requires { typename LocalLawOf<H>::type; };

template <LocalChargeHistory H>
inline Real stamp(H& k, const Vector& x, sparse::Matrix& J_nl, Vector& f_nl, Real h,
                  TrBdf2Stage st, bool /*inc*/) {
    if (k.empty() || !(h > Real{0})) return Real{0};
    using Law = typename LocalLawOf<H>::type;
    // K0/K1 stay current for the restart helpers that keep the history's
    // arithmetic on this path (DECISIONS #26: extrapolate_previous).
    k.begin_step(h, st);
    k.iface_ctx = ctx_of(h, st);
    Real m = Real{0};
    for (const auto& e : k.entries()) {
        const auto at = pins_of(e);
        constexpr std::size_t NP = std::tuple_size_v<decltype(at)>;
        std::array<Real, NP + 1> xl{};
        m = std::max(m, stamp_local<Law, NP>(at, xl, Law::from(e.params), x,
                                                    k.iface_ctx, q_of(e), e.f_prev,
                                                    qg_of(e), &J_nl, &f_nl));
    }
    return m;
}

template <LocalChargeHistory H>
inline void commit(H& k, const Vector& x, Real h, TrBdf2Stage st) {
    if (k.empty() || !(h > Real{0})) return;
    using Law = typename LocalLawOf<H>::type;
    const auto c = ctx_of(h, st);
    for (auto& e : k.mutable_entries()) {
        commit_local<Law>(pins_of(e), Law::from(e.params), x, c, q_of(e), e.f_prev,
                          qg_of(e));
    }
}

template <LocalChargeHistory H>
inline void capture_gamma(H& k, const Vector& xg) {
    using Law = typename LocalLawOf<H>::type;
    for (auto& e : k.mutable_entries()) {
        const auto at = pins_of(e);
        constexpr std::size_t NP = std::tuple_size_v<decltype(at)>;
        // No interface stamp yet (h = 0): the hand path's K0 = K1 = 0 give 0.
        if (!(k.iface_ctx.h > Real{0})) { qg_of(e) = Real{0}; continue; }
        std::array<Real, NP + 1> xl{};
        (void)stamp_local<Law, NP>(at, xl, Law::from(e.params), xg, k.iface_ctx, q_of(e),
                                   e.f_prev, qg_of(e), nullptr, nullptr);
        qg_of(e) = xl[NP];
    }
}

// ---- the MNA machines (flux rows, increment form) ----------------------------
using PmsmLaw = device::laws::PmsmMna;
using ImLaw = device::laws::InductionMachine;

[[nodiscard]] inline std::array<Index, 9> pmsm_rows(const PmsmMnaHistory::Entry& e) {
    return {e.phase_node[0], e.phase_node[1], e.phase_node[2], e.neutral, e.omega_node,
            e.theta_node,    e.cur_row[0],    e.cur_row[1],    e.cur_row[2]};
}
[[nodiscard]] inline std::array<Index, 10> im_rows(const PmsmMnaHistory::Entry& e) {
    return {e.phase_node[0], e.phase_node[1], e.phase_node[2], e.neutral,
            e.omega_node,    e.rotor_node[0], e.rotor_node[1], e.cur_row[0],
            e.cur_row[1],    e.cur_row[2]};
}
[[nodiscard]] inline PmsmLaw::Aux pmsm_aux(const PmsmMnaHistory::Entry& e) {
    return {&e.params, e.c_theta};
}
[[nodiscard]] inline ImLaw::Aux im_aux(const PmsmMnaHistory::Entry& e) {
    return {&e.im, &e.im_sm, e.c_rotor};
}
template <class Law>
[[nodiscard]] inline device::laws::MachineParams<Real> machine_params(
    const PmsmMnaHistory::Entry& e) {
    if constexpr (std::is_same_v<Law, ImLaw>) return Law::from(e.im);
    else return Law::from(e.params);
}

/// One machine through law `Law`: the stator rows' history is the entry's
/// committed (λ_n, dλ/dt_n) as q = −λ, and a·(q − q_n) is the closed-form
/// increment −(Δλ − lam_off) (`delta_lambda`; the BDF2 stage's −dlam_gamma),
/// as the hand stamp. Returns max |i_k|.
template <class Law, std::size_t R>
PULSIM_IFACE_INLINE inline Real stamp_machine(const PmsmMnaHistory::Entry& e,
                                              const std::array<Index, R>& at,
                                              const device::aux_t<Law>& aux,
                                              const Vector& x, sparse::Matrix& J_nl,
                                              Vector& f_nl, const device::StepCtx& c) {
    constexpr std::size_t I0 = Law::n_pins;
    const auto xl = gather<R>(x, at, I0);
    // The constant-L PMSM: L(θ_e) once for the law and the increment.
    models::PmsmMna::Inductance ind;
    auto aux_x = aux;
    const models::PmsmMna::Mat3* L_at_x = nullptr;
    if constexpr (std::is_same_v<Law, PmsmLaw>) {
        if (!e.params.flux_map) {
            ind = models::PmsmMna::inductance(e.params, PmsmMnaHistory::theta_e_of(e, x));
            aux_x.ind = &ind;
            L_at_x = &ind.L;
        }
    }
    const auto dlam = PmsmMnaHistory::delta_lambda(e, x, L_at_x);
    std::array<Real, R> qn{}, qd{}, dq{}, dqg{};
    Real m = Real{0};
    for (std::size_t k = 0; k < 3; ++k) {
        qn[I0 + k] = -e.lambda_old[k];
        qd[I0 + k] = -e.dlam_old[k];
        dq[I0 + k] = -(dlam[k] - e.lam_off[k]);
        dqg[I0 + k] = -e.dlam_gamma[k];
        m = std::max(m, std::abs(xl[I0 + k]));
    }
    std::array<Real, R> F{};
    std::array<std::array<Real, R>, R> J{};
    device::stamp_fq<Law>(aux_x, machine_params<Law>(e), xl, c, qn, qd, qn, F, J, &dq,
                          &dqg);
    scatter<Law>(at, F, J, J_nl, f_nl);
    return m;
}

inline Real stamp(PmsmMnaHistory& k, const Vector& x, sparse::Matrix& J_nl, Vector& f_nl,
                  Real h, TrBdf2Stage st, bool /*inc: the machines always stamp the
                                                 increment form*/) {
    if (k.empty() || !(h > Real{0})) return Real{0};
    const auto c = ctx_of(h, st);
    Real m = Real{0};
    for (const auto& e : k.entries()) {
        m = std::max(m, e.induction
                            ? stamp_machine<ImLaw>(e, im_rows(e), im_aux(e), x, J_nl, f_nl, c)
                            : stamp_machine<PmsmLaw>(e, pmsm_rows(e), pmsm_aux(e), x, J_nl,
                                                     f_nl, c));
    }
    return m;
}

/// Commit through the law: λ_n ← λ(x) (q_n = −λ), dλ/dt_n ← −q̇_n = f(x) =
/// v − R_s·i (an internal row: the solved derivative, `commit_fq`); the
/// increment-form reference and v_n are the history's bookkeeping.
template <class Law, std::size_t R>
inline void commit_machine(PmsmMnaHistory::Entry& e, const std::array<Index, R>& at,
                           const device::aux_t<Law>& aux, const Vector& x,
                           const device::StepCtx& c) {
    constexpr std::size_t I0 = Law::n_pins;
    std::array<Real, R> qn{}, qd{};
    for (std::size_t k = 0; k < 3; ++k) {
        qn[I0 + k] = -e.lambda_old[k];
        qd[I0 + k] = -e.dlam_old[k];
    }
    (void)device::commit_fq<Law>(aux, machine_params<Law>(e), gather<R>(x, at, I0), c,
                                 qn, qd, qn);
    const Real v_n = stamping::read_node_voltage(x, e.neutral);
    PmsmMnaHistory::set_reference(e, x);
    for (std::size_t k = 0; k < 3; ++k) {
        e.lambda_old[k] = -qn[I0 + k];
        e.v_old[k] = stamping::read_node_voltage(x, e.phase_node[k]) - v_n;
        e.dlam_old[k] = -qd[I0 + k];
    }
}

inline void commit(PmsmMnaHistory& k, const Vector& x, Real h, TrBdf2Stage st) {
    // kPinQ = false: the commit takes no time formula, so h = 0 is fine.
    const auto c = ctx_of(h, st);
    for (auto& e : k.mutable_entries()) {
        if (e.induction) {
            commit_machine<ImLaw>(e, im_rows(e), im_aux(e), x, c);
            continue;
        }
        if (e.params.flux_map) PmsmMnaHistory::check_flux_map_band_(e, x);
        commit_machine<PmsmLaw>(e, pmsm_rows(e), pmsm_aux(e), x, c);
    }
}

/// The BDF2 stage's −q_γ − (−q_n) = −(λ_γ − λ_n) in increment form: the
/// history's closed form (`delta_lambda` at the stage point).
inline void capture_gamma(PmsmMnaHistory& k, const Vector& xg) { k.capture_gamma(xg); }

// ---- the registry of ported kinds ------------------------------------------
/// A stateful history whose law runs on the interface path.
template <class K> inline constexpr bool kPorted = false;
template <> inline constexpr bool kPorted<NonlinearCapacitorHistory> = true;
template <> inline constexpr bool kPorted<SaturableInductorHistory> = true;
template <> inline constexpr bool kPorted<LauritzenDiodeHistory> = true;
template <> inline constexpr bool kPorted<IgbtTailHistory> = true;
template <> inline constexpr bool kPorted<PmsmMnaHistory> = true;

/// The kind-table rows a stateful history runs: its `kStoredKind`, and for
/// the machine history also the induction machine (DECISIONS #34).
template <class K>
[[nodiscard]] constexpr bool history_covers(DevicePool::StoredKind sk) noexcept {
    if constexpr (std::is_same_v<K, PmsmMnaHistory>) {
        if (sk == DevicePool::StoredKind::InductionMachineMna) return true;
    }
    return K::kStoredKind == sk;
}

/// A kind-table row whose Newton law runs on the interface path (the
/// static rows; a stateful row is ported when its history is).
[[nodiscard]] constexpr bool static_kind_ported(DevicePool::StoredKind k) noexcept {
    using SK = DevicePool::StoredKind;
    return k == SK::NonlinearDiode || k == SK::ShockleyDiode
        || k == SK::MosfetLevel1 || k == SK::IgbtLevel1;
}

}  // namespace pulsim::pwl::iface

#undef PULSIM_IFACE_INLINE
