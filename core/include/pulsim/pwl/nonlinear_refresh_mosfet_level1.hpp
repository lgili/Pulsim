#pragma once

// =============================================================================
// Pulsim — Layer 4 V13/V14: SH1 MOSFET + Level-1 IGBT Newton refresh
// =============================================================================
//
// Per-Newton-iteration refresh for the 3-terminal AD devices
// `models::MosfetLevel1` and `models::IgbtLevel1`. Each iterates
// `BranchKind::Nonlinear` branches of its `StoredKind`, reads V(drain/
// collector), V(source/emitter), V(gate) from the state vector, evaluates the
// current + Jacobian via AD, and stamps the (drain/source) x (drain/source/
// gate) conductance block:
//
//   f[from]  += i;                f[to] -= i
//   J[from, k] += ∂i/∂v[k];       J[to, k] -= ∂i/∂v[k]   for k in {from,to,gate}
//
// (Gate KCL is unaffected — Level-1 ideal gate draws zero current. Parasitic
// gate capacitance is out of scope.)
//
// The loop/stamp body is identical to the diode's apart from terminal count
// and param/gate lookup, so all stampers delegate to the generic
// `stamp_nonlinear_branches<Device>` (nonlinear_refresh_device.hpp) and supply
// only the per-device `get_params`/`get_terminals` functors. (Audit 2026-05
// #14b: collapsed ~700 LOC of cross-device copy-paste.)

#include "pulsim/models/device_model.hpp"
#include "pulsim/models/ideal_diode.hpp"
#include "pulsim/models/igbt_level1.hpp"
#include "pulsim/models/mosfet_level1.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/nonlinear_refresh_device.hpp"
#include "pulsim/pwl/nonlinear_solve.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/topology/graph.hpp"

#include <algorithm>
#include <array>

namespace pulsim::pwl {

namespace detail {
/// Terminal list for a 3-terminal SH1 MOSFET: {drain, source, gate}.
inline constexpr auto mosfet_level1_terminals =
    [](const DevicePool& pool, const auto& branch) {
        return std::array<Index, 3>{
            branch.from, branch.to,
            pool.mosfet_level1_gate_node(branch.id)};
    };
/// Terminal list for a 3-terminal Level-1 IGBT: {collector, emitter, gate}.
inline constexpr auto igbt_level1_terminals =
    [](const DevicePool& pool, const auto& branch) {
        return std::array<Index, 3>{
            branch.from, branch.to,
            pool.igbt_level1_gate_node(branch.id)};
    };
}  // namespace detail

/// Stamp SH1 MOSFET contributions on every Nonlinear branch registered as
/// one. Clears J_nl/f_nl first. Returns max(|i_drain|) for residual reporting.
inline Real refresh_mosfets_level1(
    const Vector& x,
    sparse::Matrix& J_nl,
    Vector& f_nl,
    const topology::Graph& graph,
    const DevicePool& pool) {
    if (J_nl.rows() > 0) J_nl.setZero();
    if (f_nl.size() > 0) f_nl.setZero();
    return stamp_nonlinear_branches<models::MosfetLevel1>(
        x, J_nl, f_nl, graph, pool,
        DevicePool::StoredKind::MosfetLevel1,
        [](const DevicePool& p, Index b) {
            return p.mosfet_level1_params(b);
        },
        detail::mosfet_level1_terminals);
}

/// Stamp Level-1 IGBT contributions (linear-conduction model — Newton handles
/// it cleanly, no spurious roots). Clears J_nl/f_nl first.
inline Real refresh_igbts_level1(
    const Vector& x,
    sparse::Matrix& J_nl,
    Vector& f_nl,
    const topology::Graph& graph,
    const DevicePool& pool) {
    if (J_nl.rows() > 0) J_nl.setZero();
    if (f_nl.size() > 0) f_nl.setZero();
    return stamp_nonlinear_branches<models::IgbtLevel1>(
        x, J_nl, f_nl, graph, pool,
        DevicePool::StoredKind::IgbtLevel1,
        [](const DevicePool& p, Index b) {
            return p.igbt_level1_params(b);
        },
        detail::igbt_level1_terminals);
}

/// Combined refresh that stamps smooth-blend IdealDiode, Level-1 IGBT, and
/// SH1 MOSFET contributions in a single `NonlinearRefreshFn`. Drop-in for
/// circuits mixing those nonlinear devices. Zeroes J_nl/f_nl once, then runs
/// three ADDITIVE device passes (pass order is irrelevant — stamps accumulate
/// into the same once-zeroed matrices); the residual indicator is the max
/// stamped current across all three.
[[nodiscard]] inline NonlinearRefreshFn
make_combined_diode_mosfet_refresh() {
    return [](const Vector& x, sparse::Matrix& J_nl, Vector& f_nl,
              const topology::Graph& graph, const DevicePool& pool) -> Real {
        if (J_nl.rows() > 0) J_nl.setZero();
        if (f_nl.size() > 0) f_nl.setZero();

        const Real i_diode = stamp_nonlinear_branches<models::IdealDiode>(
            x, J_nl, f_nl, graph, pool,
            DevicePool::StoredKind::NonlinearDiode,
            [](const DevicePool& p, Index b) {
                return p.nonlinear_diode_params(b);
            },
            [](const DevicePool&, const auto& branch) {
                return std::array<Index, 2>{branch.from, branch.to};
            });

        const Real i_igbt = stamp_nonlinear_branches<models::IgbtLevel1>(
            x, J_nl, f_nl, graph, pool,
            DevicePool::StoredKind::IgbtLevel1,
            [](const DevicePool& p, Index b) {
                return p.igbt_level1_params(b);
            },
            detail::igbt_level1_terminals);

        const Real i_mosfet = stamp_nonlinear_branches<models::MosfetLevel1>(
            x, J_nl, f_nl, graph, pool,
            DevicePool::StoredKind::MosfetLevel1,
            [](const DevicePool& p, Index b) {
                return p.mosfet_level1_params(b);
            },
            detail::mosfet_level1_terminals);

        return std::max({i_diode, i_igbt, i_mosfet});
    };
}

}  // namespace pulsim::pwl
