// SPDX-License-Identifier: MIT
//
// Pulsim — the small-signal model of a circuit at an operating point
// (add-unified-device-interface D8 / task 10.1).
//
// The circuit reads E·dx/dt + G·x + b + Σ_d [f_d(x) + d/dt q_d(x)] = 0. At an
// operating point x0 (a DC point: every d/dt is 0 there) the small-signal
// model of x = x0 + x̂ is
//
//     (G + Σ ∂f_d/∂x) · x̂ + (E + Σ ∂q_d/∂x) · dx̂/dt = −b̂ ,
//
// `Linearization{G, C}` holds the two matrices. The linear part is the AC
// sweep's own assembly (`assemble_segment_split` at the caller's switch
// mask, E = C_split/2, unchanged operation for operation); each Newton
// device adds the Jacobians of its `Device` law (device/laws.hpp,
// device/machine_laws.hpp) from ONE forward-AD evaluation at x0. A device
// with LOCAL rows (the stored charge of the Lauritzen diode and of the
// IGBT tail) keeps them as extra unknowns appended after the circuit's
// (rows n … n+L−1): its charge is a state, so it cannot be condensed into
// a real (G, C) pair (the condensed admittance depends on ω).
//
// What is refused, by name (`UnsupportedError(engine='ac_sweep')`):
//   * kinds whose `kKindCaps` row lacks the ac_sweep bit (the induction
//     machine: no analytic small-signal test yet);
//   * a Jiles–Atherton saturable inductor: its incremental inductance at a
//     point depends on the direction of the excursion (a minor loop), so
//     there is no unique small-signal L;
//   * a nonlinear permeance on a magnetic node (its DC point needs a
//     nonlinear flux-conservation row the DC assembly lacks);
//   * a PMSM held at a non-zero speed (abc quantities rotate: no
//     equilibrium) or whose shaft is not in torque balance at x0.

#pragma once

#include "pulsim/ad/ad_scalar.hpp"
#include "pulsim/device/device.hpp"
#include "pulsim/device/laws.hpp"
#include "pulsim/device/machine_laws.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/assemble.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/device_refresh.hpp"
#include "pulsim/pwl/external_device_history.hpp"
#include "pulsim/pwl/igbt_tail_history.hpp"
#include "pulsim/pwl/interface_stamps.hpp"
#include "pulsim/pwl/lauritzen_diode_history.hpp"
#include "pulsim/pwl/nonlinear_capacitor_history.hpp"
#include "pulsim/pwl/pmsm_mna_history.hpp"
#include "pulsim/pwl/saturable_inductor_history.hpp"
#include "pulsim/solver/unsupported.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <string>

namespace pulsim::analysis {

/// The small-signal model at an operating point: the sweep solves
/// (G + jω·C)·x̂ = e. C is the MASS matrix (the coefficient of dx/dt), not
/// the transient split's 2/dt-scaled one.
struct Linearization {
    sparse::Matrix G;
    sparse::Matrix C;
    Index n_circuit = 0;   ///< the circuit's unknowns; rows ≥ this are device-local charges
};

namespace detail_lin {

/// G += ∂f/∂x, C += ∂q/∂x of law `D` at `xl`, on the rows / columns `at`
/// (pins are node rows, skipped when grounded; internal and local rows are
/// always active). One forward-AD evaluation, as the engines' stamps.
template <class D, std::size_t R>
inline void add_law(const device::aux_t<D>& aux,
                    const typename D::template Params<Real>& p,
                    const std::array<Index, R>& at, const std::array<Real, R>& xl,
                    sparse::Matrix& G, sparse::Matrix& C) {
    static_assert(R == device::rows_v<D>);
    std::array<ad::ADRealN<R>, R> xs{};
    for (std::size_t k = 0; k < R; ++k) {
        std::array<Real, R> d{};
        d[k] = Real{1};
        xs[k] = ad::ADRealN<R>{xl[k], d};
    }
    device::Out<ad::ADRealN<R>, R> o;
    device::eval<D>(aux, p, xs, o);
    const auto on = [&](std::size_t r) {
        return r >= D::n_pins || stamping::node_is_active(at[r]);
    };
    for (std::size_t r = 0; r < R; ++r) {
        if (!on(r)) continue;
        for (std::size_t k = 0; k < R; ++k) {
            if (!on(k)) continue;
            if constexpr (device::has_f_v<D>) {
                const Real g = o.f[r].deriv(k);
                if (g != Real{0}) G.coeffRef(at[r], at[k]) += g;
            }
            if constexpr (device::has_q_v<D>) {
                const Real c = o.q[r].deriv(k);
                if (c != Real{0}) C.coeffRef(at[r], at[k]) += c;
            }
        }
    }
}

/// The value of a law's one local row at its DC point: f_L(x, s) = 0 with
/// q̇_L = 0. The local row is affine in s (`kLocalAffine`), so one Newton
/// step from s = 0 solves it to rounding.
template <class D, std::size_t R>
[[nodiscard]] inline Real local_dc_value(const typename D::template Params<Real>& p,
                                         std::array<Real, R> xl) {
    static_assert(device::local_v<D> == 1 && D::kLocalAffine);
    constexpr std::size_t L = R - 1;
    xl[L] = Real{0};
    std::array<ad::ADRealN<1>, R> xs{};
    for (std::size_t k = 0; k < R; ++k) {
        xs[k] = ad::ADRealN<1>{xl[k], {k == L ? Real{1} : Real{0}}};
    }
    device::Out<ad::ADRealN<1>, R> o;
    device::eval<D>(device::aux_t<D>{}, p, xs, o);
    return -o.f[L].value() / o.f[L].deriv(0);
}

/// One local-charge instance at the DC point x0: its charge row is unknown
/// `row` of the extended system.
template <class D, std::size_t NP>
inline void add_local(const std::array<Index, NP>& pins,
                      const typename D::template Params<Real>& p, const Vector& x0,
                      Index row, sparse::Matrix& G, sparse::Matrix& C) {
    constexpr std::size_t R = NP + 1;
    std::array<Index, R> at{};
    std::array<Real, R> xl{};
    for (std::size_t r = 0; r < NP; ++r) {
        at[r] = pins[r];
        xl[r] = stamping::read_node_voltage(x0, pins[r]);
    }
    at[NP] = row;
    xl[NP] = local_dc_value<D, R>(p, xl);
    add_law<D, R>(device::aux_t<D>{}, p, at, xl, G, C);
}

[[noreturn]] inline void refuse(const topology::Graph& graph, const pwl::DevicePool& pool,
                                Index b, const char* feature, const std::string& why) {
    throw UnsupportedError("ac_sweep", feature, std::string(graph.branch_name(b)),
                           std::string(pwl::stored_kind_name(pool.kind_of(b))), why);
}

/// A PMSM is linearisable only at an equilibrium: at rest (ω0 = 0; at speed
/// the abc quantities rotate) with its shaft in torque balance — the DC
/// point holds the mechanics, so it does not enforce T(i, θ) = T_load by
/// itself. The shaft row's residual of the free system, (G·x0 + b)_ω +
/// f_ω(x0), must vanish to 1e-9 of its largest term.
inline void require_pmsm_equilibrium(const topology::Graph& graph,
                                     const pwl::DevicePool& pool,
                                     const pwl::PmsmMnaHistory::Entry& e,
                                     const sparse::Matrix& G_lin, const Vector& b,
                                     const Vector& x0) {
    using Law = device::laws::PmsmMna;
    const Index b0 = e.branch_id[0];
    const Real w = stamping::read_node_voltage(x0, e.omega_node);
    if (w != Real{0}) {
        refuse(graph, pool, b0, "",
               std::format("run_mna_sweep: the PMSM '{}' is held at omega = {} rad/s; "
                           "at speed its abc voltages and currents rotate, so the "
                           "circuit has no equilibrium to linearise about. Sweep it at "
                           "rest (omega0 = 0) or use a time-domain routine "
                           "(pulsim.fra.run_fra).",
                           graph.branch_name(b0), w));
    }
    if (!stamping::node_is_active(e.omega_node)) return;
    const Index r = e.omega_node;
    Real res = b[r], scale = std::abs(b[r]);
    for (Index k = 0; k < G_lin.outerSize(); ++k) {
        for (sparse::Matrix::InnerIterator it(G_lin, k); it; ++it) {
            if (it.row() != r) continue;
            const Real t = it.value() * x0[it.col()];
            res += t;
            scale = std::max(scale, std::abs(t));
        }
    }
    const auto at = pwl::iface::pmsm_rows(e);
    const auto xl = pwl::iface::gather<9>(x0, at, Law::n_pins);
    device::Out<Real, 9> o;
    device::eval<Law>(pwl::iface::pmsm_aux(e), Law::from(e.params), xl, o);
    res += o.f[4];
    scale = std::max({scale, std::abs(o.f[4]), std::abs(e.params.T_load)});
    if (std::abs(res) > Real{1e-9} * scale) {
        refuse(graph, pool, b0, "",
               std::format("run_mna_sweep: the PMSM '{}' is not in torque balance at the "
                           "DC point (shaft-row residual {:.3e} N·m, largest term "
                           "{:.3e}): the DC point holds the mechanics, and the rotor "
                           "would accelerate from it, so it is no equilibrium. Bias it "
                           "so the electromagnetic torque equals the load.",
                           graph.branch_name(b0), res, scale));
    }
}

}  // namespace detail_lin

/// The instances of an accepted kind that have no small-signal model,
/// refused by name BEFORE the DC point is solved: a Jiles–Atherton
/// saturable inductor (no unique incremental inductance: a minor loop) and
/// a nonlinear permeance on a magnetic node (its DC row is flux
/// conservation, linear-only in the DC assembly, so the operating point
/// itself does not exist yet).
inline void require_linearisable(const topology::Graph& graph, const pwl::DevicePool& pool) {
    for (const Index b : pool.saturable_inductor_branches()) {
        if (!pool.saturable_inductor_params(b).ja) continue;
        detail_lin::refuse(
            graph, pool, b, "hysteretic_core",
            std::format("run_mna_sweep: '{}' is a Jiles-Atherton (hysteretic) "
                        "saturable inductor; its incremental inductance at a point "
                        "depends on the direction of the excursion (a minor loop), "
                        "so it has no unique small-signal model. Use a time-domain "
                        "routine (pulsim.fra.run_fra).",
                        graph.branch_name(b)));
    }
    if (pool.flux_nodes().empty()) return;
    const auto& fx = pool.flux_nodes();
    const auto is_flux = [&](Index n) {
        return n >= 0 && std::find(fx.begin(), fx.end(), n) != fx.end();
    };
    for (const Index b : pool.nonlinear_capacitor_branches()) {
        const auto& br = graph.branch(b);
        if (!is_flux(br.from) && !is_flux(br.to)) continue;
        detail_lin::refuse(
            graph, pool, b, "nonlinear_permeance",
            std::format("run_mna_sweep: '{}' is a nonlinear permeance on a magnetic "
                        "node; the DC operating point it would be linearised at needs "
                        "a nonlinear flux-conservation row, which the DC assembly does "
                        "not have yet. Use a time-domain routine (pulsim.fra.run_fra).",
                        graph.branch_name(b)));
    }
}

/// Does the pool hold a built-in Newton kind (one whose law `linearize`
/// stamps)? External devices are not counted: their path predates this one.
[[nodiscard]] inline bool has_builtin_newton(const topology::Graph& graph,
                                             const pwl::DevicePool& pool) noexcept {
    using SK = pwl::DevicePool::StoredKind;
    for (Index b = 0; b < graph.num_branches(); ++b) {
        if (!pool.is_registered(b)) continue;
        const auto k = pool.kind_of(b);
        if (pwl::kind_caps(k).newton() && k != SK::ExternalDevice) return true;
    }
    return false;
}

/// The linear part: the sweep's (G, E) from the transient assembler at
/// `mask`, operation for operation as before (E = C_split/2: the split
/// stores the trapezoidal 2/dt coefficients). `b` (optional) receives the
/// split's constant right-hand side.
[[nodiscard]] inline Linearization linear_part(const topology::Graph& graph,
                                               const pwl::DevicePool& pool,
                                               const topology::SwitchStateMask& mask,
                                               Vector* b_out = nullptr) {
    Linearization m;
    sparse::Matrix C;
    Vector b;
    pwl::assemble_segment_split(graph, pool, mask, m.G, C, b);
    m.C = Real{0.5} * C;
    m.C.makeCompressed();
    m.n_circuit = m.G.rows();
    if (b_out != nullptr) *b_out = std::move(b);
    return m;
}

/// The small-signal model at `x0` (task 10.1): the linear part at `mask`
/// plus ∂f/∂x, ∂q/∂x of every Newton device at `x0`. `x0` must be the DC
/// operating point at `mask` (`t` its time): the device laws are evaluated
/// there, and the local charges are taken at their DC values.
[[nodiscard]] inline Linearization linearize(const topology::Graph& graph,
                                             const pwl::DevicePool& pool,
                                             const topology::SwitchStateMask& mask,
                                             const Vector& x0, Real t = Real{0}) {
    Vector b;
    Linearization m = linear_part(graph, pool, mask, &b);

    // External devices, as the sweep linearised them before: the model state
    // the DC point was solved with (init at x = 0).
    if (!pool.external_device_branches().empty()) {
        pwl::ExternalDeviceHistory xh;
        xh.set_start_time(t);
        xh.init(graph, pool);
        xh.linearise(x0, m.G, m.C, t);
        m.G.makeCompressed();
        m.C.makeCompressed();
    }
    if (!has_builtin_newton(graph, pool)) return m;
    require_linearisable(graph, pool);

    using SK = pwl::DevicePool::StoredKind;
    namespace L = device::laws;
    const sparse::Matrix G_lin = m.G;   // the free system's linear rows (PMSM balance)

    pwl::LauritzenDiodeHistory lh;
    lh.init(graph, pool);
    pwl::IgbtTailHistory th;
    th.init(graph, pool);
    const Index n = m.n_circuit;
    const Index n_local = static_cast<Index>(lh.entries().size() + th.entries().size());
    if (n_local > 0) {
        m.G.conservativeResize(n + n_local, n + n_local);
        m.C.conservativeResize(n + n_local, n + n_local);
    }

    // static kinds, in graph order (the interface refresh's loop)
    for (Index bi = 0; bi < graph.num_branches(); ++bi) {
        const auto& br = graph.branch(bi);
        if (br.kind != topology::BranchKind::Nonlinear) continue;
        const std::array<Index, 2> two{br.from, br.to};
        switch (pool.kind_of(br.id)) {
        case SK::NonlinearDiode:
            detail_lin::add_law<L::IdealDiode, 2>(
                {}, L::IdealDiode::from(pool.nonlinear_diode_params(br.id)), two,
                pwl::iface::gather<2>(x0, two, 2), m.G, m.C);
            break;
        case SK::ShockleyDiode:
            detail_lin::add_law<L::ShockleyDiode, 2>(
                {}, L::ShockleyDiode::from(pool.shockley_diode_params(br.id)), two,
                pwl::iface::gather<2>(x0, two, 2), m.G, m.C);
            break;
        case SK::MosfetLevel1: {
            const auto at = pwl::detail::mosfet_level1_terminals(pool, br);
            detail_lin::add_law<L::MosfetLevel1, 3>(
                {}, L::MosfetLevel1::from(pool.mosfet_level1_params(br.id)), at,
                pwl::iface::gather<3>(x0, at, 3), m.G, m.C);
            break;
        }
        case SK::IgbtLevel1: {
            const auto at = pwl::detail::igbt_level1_terminals(pool, br);
            detail_lin::add_law<L::IgbtLevel1, 3>(
                {}, L::IgbtLevel1::from(pool.igbt_level1_params(br.id)), at,
                pwl::iface::gather<3>(x0, at, 3), m.G, m.C);
            break;
        }
        default: break;
        }
    }

    // the nonlinear capacitor: q only, C += C(v)
    pwl::NonlinearCapacitorHistory ch;
    ch.init(graph, pool);
    for (const auto& e : ch.entries()) {
        const std::array<Index, 2> at{e.from, e.to};
        detail_lin::add_law<L::NonlinearCapacitor, 2>(
            L::NonlinearCapacitor::aux_from(e.params), L::NonlinearCapacitor::from(e.params),
            at, pwl::iface::gather<2>(x0, at, 2), m.G, m.C);
    }

    // the saturable inductor: the branch row v − L(i0)·di/dt
    pwl::SaturableInductorHistory sh;
    sh.init(graph, pool);
    for (const auto& e : sh.entries()) {
        // a hysteretic core never reaches here (require_linearisable)
        const std::array<Index, 3> at{e.from, e.to, e.branch_var_id};
        detail_lin::add_law<L::SaturableInductor, 3>(
            {e.params.table.get(), nullptr, nullptr}, L::SaturableInductor::from(e.params),
            at, pwl::iface::gather<3>(x0, at, 2), m.G, m.C);
    }

    // the stored-charge kinds: one extended unknown each
    Index row = n;
    for (const auto& e : lh.entries()) {
        detail_lin::add_local<L::LauritzenDiode, 2>(
            {e.from, e.to}, L::LauritzenDiode::from(e.params), x0, row++, m.G, m.C);
    }
    for (const auto& e : th.entries()) {
        detail_lin::add_local<L::IgbtTail, 3>(
            {e.collector, e.emitter, e.gate}, L::IgbtTail::from(e.params), x0, row++, m.G,
            m.C);
    }

    // the PMSM, at rest and in torque balance
    if (!pool.pmsm_mna_machines().empty()) {
        pwl::PmsmMnaHistory ph;
        ph.init(graph, pool);
        for (const auto& e : ph.entries()) {
            if (e.induction) continue;   // refused by kind before this point
            detail_lin::require_pmsm_equilibrium(graph, pool, e, G_lin, b, x0);
            const auto at = pwl::iface::pmsm_rows(e);
            detail_lin::add_law<L::PmsmMna, 9>(
                pwl::iface::pmsm_aux(e), L::PmsmMna::from(e.params), at,
                pwl::iface::gather<9>(x0, at, L::PmsmMna::n_pins), m.G, m.C);
        }
    }
    m.G.makeCompressed();
    m.C.makeCompressed();
    return m;
}

}  // namespace pulsim::analysis
