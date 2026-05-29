#pragma once

// =============================================================================
// Pulsim — Layer 4 V3: Smooth-blend IdealDiode Newton refresh
// =============================================================================
//
// Built-in `NonlinearRefreshFn`s for the smooth-blend `models::IdealDiode`
// (the sigmoid-shifted Norton model in models/ideal_diode.hpp). The base
// `refresh_smooth_diodes` stamps every `BranchKind::Nonlinear` branch
// registered as `StoredKind::NonlinearDiode`; the `make_*_override_refresh`
// factories let `continuation_solve` ramp a single Param field (kappa and/or
// V_F0) as a homotopy while leaving the rest pool-stored.
//
// All four share the same loop/stamp body, so they delegate to the generic
// `stamp_nonlinear_branches<IdealDiode>` (nonlinear_refresh_device.hpp) and
// differ only in the `get_params` functor — the override factories copy the
// pool params and overwrite the homotopy field. (Audit 2026-05 #14b: this
// removed ~250 LOC of copy-paste from this file.)

#include "pulsim/models/device_model.hpp"
#include "pulsim/models/ideal_diode.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/nonlinear_refresh_device.hpp"
#include "pulsim/pwl/nonlinear_solve.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/topology/graph.hpp"

#include <array>

namespace pulsim::pwl {

namespace detail {
/// Terminal list for a 2-terminal nonlinear diode: {anode, cathode}.
/// Generic lambda-friendly free helper (branch type stays deduced).
inline constexpr auto nonlinear_diode_terminals =
    [](const DevicePool& /*pool*/, const auto& branch) {
        return std::array<Index, 2>{branch.from, branch.to};
    };
}  // namespace detail

/// Refresh function that stamps the smooth-blend `IdealDiode` on every
/// `BranchKind::Nonlinear` branch registered as `NonlinearDiode`. Clears
/// J_nl/f_nl, then stamps. Returns max(|i_diode|) as the residual indicator.
inline Real refresh_smooth_diodes(
    const Vector& x,
    sparse::Matrix& J_nl,
    Vector& f_nl,
    const topology::Graph& graph,
    const DevicePool& pool) {
    if (J_nl.rows() > 0) J_nl.setZero();
    if (f_nl.size() > 0) f_nl.setZero();
    return stamp_nonlinear_branches<models::IdealDiode>(
        x, J_nl, f_nl, graph, pool,
        DevicePool::StoredKind::NonlinearDiode,
        [](const DevicePool& p, Index b) {
            return p.nonlinear_diode_params(b);
        },
        detail::nonlinear_diode_terminals);
}

/// Layer 4 V8 — `NonlinearRefreshFn` that stamps smooth-blend IdealDiode
/// contributions using `kappa_override` instead of each diode's pool-stored
/// kappa. Other params (V_F0, R_d, G_off) come from the pool unchanged.
/// Used by `continuation_solve` to ramp kappa from "easy" (smooth) to "hard"
/// (target) progressively, warm-starting each step from the previous.
[[nodiscard]] inline NonlinearRefreshFn
make_kappa_override_refresh(Real kappa_override) {
    return [kappa_override](const Vector& x, sparse::Matrix& J_nl,
                            Vector& f_nl, const topology::Graph& graph,
                            const DevicePool& pool) -> Real {
        if (J_nl.rows() > 0) J_nl.setZero();
        if (f_nl.size() > 0) f_nl.setZero();
        return stamp_nonlinear_branches<models::IdealDiode>(
            x, J_nl, f_nl, graph, pool,
            DevicePool::StoredKind::NonlinearDiode,
            [kappa_override](const DevicePool& p, Index b) {
                models::IdealDiode::Params pp = p.nonlinear_diode_params(b);
                pp.kappa = kappa_override;
                return pp;
            },
            detail::nonlinear_diode_terminals);
    };
}

/// Layer 4 V9 — `NonlinearRefreshFn` that stamps using `V_F0_override`
/// instead of each diode's pool-stored V_F0. Other params (R_d, G_off, kappa)
/// come from the pool unchanged. V_F0 continuation is the homotopy that
/// bridges "always-conducting" (V_F0 = -V_amp) → "realistic diode"
/// (V_F0 = 0.7) without crossing the smooth-blend's near-zero bifurcation;
/// lets `continuation_solve` solve stiff rectifiers from x = 0.
[[nodiscard]] inline NonlinearRefreshFn
make_vf0_override_refresh(Real V_F0_override) {
    return [V_F0_override](const Vector& x, sparse::Matrix& J_nl,
                           Vector& f_nl, const topology::Graph& graph,
                           const DevicePool& pool) -> Real {
        if (J_nl.rows() > 0) J_nl.setZero();
        if (f_nl.size() > 0) f_nl.setZero();
        return stamp_nonlinear_branches<models::IdealDiode>(
            x, J_nl, f_nl, graph, pool,
            DevicePool::StoredKind::NonlinearDiode,
            [V_F0_override](const DevicePool& p, Index b) {
                models::IdealDiode::Params pp = p.nonlinear_diode_params(b);
                pp.V_F0 = V_F0_override;
                return pp;
            },
            detail::nonlinear_diode_terminals);
    };
}

/// Layer 4 V9.1 — `NonlinearRefreshFn` that stamps using BOTH
/// `kappa_override` AND `V_F0_override` (R_d, G_off pool-stored). Used to
/// build combined κ + V_F0 continuation chains, more robust than either
/// alone for the κ=20 stiff rectifier from x = 0.
[[nodiscard]] inline NonlinearRefreshFn
make_kappa_vf0_override_refresh(Real kappa_override, Real V_F0_override) {
    return [kappa_override, V_F0_override](
               const Vector& x, sparse::Matrix& J_nl, Vector& f_nl,
               const topology::Graph& graph, const DevicePool& pool) -> Real {
        if (J_nl.rows() > 0) J_nl.setZero();
        if (f_nl.size() > 0) f_nl.setZero();
        return stamp_nonlinear_branches<models::IdealDiode>(
            x, J_nl, f_nl, graph, pool,
            DevicePool::StoredKind::NonlinearDiode,
            [kappa_override, V_F0_override](const DevicePool& p, Index b) {
                models::IdealDiode::Params pp = p.nonlinear_diode_params(b);
                pp.kappa = kappa_override;
                pp.V_F0  = V_F0_override;
                return pp;
            },
            detail::nonlinear_diode_terminals);
    };
}

}  // namespace pulsim::pwl
