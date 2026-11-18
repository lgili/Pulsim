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
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/nonlinear_capacitor_history.hpp"
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
/// The law's structural entries as (row, col) pairs, column-major.
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
    for (std::size_t k = 0; k < R; ++k)
        for (std::size_t r = 0; r < R; ++r)
            if (D::kPattern[r][k]) out[i++] = {r, k};
    return out;
}
}  // namespace detail

/// Add F on the active rows and J on the law's structural entries whose
/// row and column are both active.
template <class D, std::size_t R>
inline void scatter(const std::array<Index, R>& at,
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
    std::array<bool, R> on{};
    for (std::size_t r = 0; r < R; ++r) {
        on[r] = r >= D::n_pins || stamping::node_is_active(at[r]);
        if (on[r] && kRowUsed[r]) f_nl[at[r]] += F[r];
    }
    static constexpr auto kEntries = detail::pattern_entries<D, R>();
    for (const auto& [r, k] : kEntries) {
        if (on[r] && on[k]) J_nl.coeffRef(at[r], at[k]) += J[r][k];
    }
}

// ---- static kinds -----------------------------------------------------------
/// One static branch through law `L`. Returns |i|.
template <class L, std::size_t R>
inline Real stamp_one_static(const Vector& x, sparse::Matrix& J_nl, Vector& f_nl,
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
        const std::array<Index, 2> at{e.from, e.to};
        const Real qg = bdf2 ? k.q_gamma()[idx] : Real{0};
        ++idx;
        std::array<Real, 2> F{};
        std::array<std::array<Real, 2>, 2> J{};
        device::stamp_fq<CossLaw>(CossLaw::aux_from(e.params),
                                  CossLaw::from(e.params), gather<2>(x, at, 2), c,
                                  {e.q_prev, -e.q_prev}, {e.i_prev, -e.i_prev},
                                  {qg, -qg}, F, J);
        scatter<CossLaw>(at, F, J, J_nl, f_nl);
        m = std::max(m, std::abs(F[0]));
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

inline Real stamp(SaturableInductorHistory& k, const Vector& x,
                  sparse::Matrix& J_nl, Vector& f_nl, Real h, TrBdf2Stage st,
                  bool inc) {
    const auto c = ctx_of(h, st);
    const bool bdf2 = st == TrBdf2Stage::Bdf2Stage2;
    Real m = Real{0};
    for (const auto& e : k.entries()) {
        const std::array<Index, 3> at{e.from, e.to, e.branch_var_id};
        const auto xl = gather<3>(x, at, 2);
        std::array<Real, 3> F{};
        std::array<std::array<Real, 3>, 3> J{};
        // The increment form (TR-BDF2): a·(q − q_n) in closed form, for
        // the atan law (the hand stamp's `incr` condition).
        const bool incr = inc && !e.params.ja && !e.params.table;
        std::array<Real, 3> dq{}, dqg{};
        if (incr) {
            dq[2] = -models::SaturableInductor::flux_increment(xl[2], e.i_L_old,
                                                              e.params);
            if (bdf2) {
                dqg[2] = -models::SaturableInductor::flux_increment(
                    e.i_gamma, e.i_L_old, e.params);
            }
        }
        device::stamp_fq<SatLaw>(sat_aux(e, st), SatLaw::from(e.params), xl, c,
                                 {Real{0}, Real{0}, -e.lambda_old},
                                 {Real{0}, Real{0}, -e.V_L_old},
                                 {Real{0}, Real{0}, -e.lambda_gamma}, F, J,
                                 incr ? &dq : nullptr, incr ? &dqg : nullptr);
        scatter<SatLaw>(at, F, J, J_nl, f_nl);
        m = std::max(m, std::abs(xl[2]));
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

// ---- the registry of ported kinds ------------------------------------------
/// A stateful history whose law runs on the interface path.
template <class K> inline constexpr bool kPorted = false;
template <> inline constexpr bool kPorted<NonlinearCapacitorHistory> = true;
template <> inline constexpr bool kPorted<SaturableInductorHistory> = true;

/// A kind-table row whose Newton law runs on the interface path (the
/// static rows; a stateful row is ported when its history is).
[[nodiscard]] constexpr bool static_kind_ported(DevicePool::StoredKind k) noexcept {
    using SK = DevicePool::StoredKind;
    return k == SK::NonlinearDiode || k == SK::ShockleyDiode
        || k == SK::MosfetLevel1 || k == SK::IgbtLevel1;
}

}  // namespace pulsim::pwl::iface
