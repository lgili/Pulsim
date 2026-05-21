#pragma once

// =============================================================================
// Pulsim v2 — Layer 4 V3: Smooth-blend IdealDiode Newton refresh
// =============================================================================
//
// Built-in `NonlinearRefreshFn` for the smooth-blend
// `models::IdealDiode` (the sigmoid-shifted Norton model that
// already lives in models/ideal_diode.hpp and was waiting for a
// Newton consumer since Layer 2).
//
// For V0 of the nonlinear-Newton OpenSpec we need a way for the
// caller to register the smooth-blend diode's params on a
// `BranchKind::Nonlinear` branch. We piggyback on the existing
// `DevicePool` machinery: a new `add_nonlinear_diode(...)`
// method stores the params. The refresh function looks them up
// per branch.

#include "pulsim/v2/models/device_model.hpp"
#include "pulsim/v2/models/ideal_diode.hpp"
#include "pulsim/v2/numeric/dense.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/pwl/nonlinear_solve.hpp"
#include "pulsim/v2/sparse/matrix.hpp"
#include "pulsim/v2/stamping/branch_coord.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <cmath>

namespace pulsim::v2::pwl {

/// Refresh function that stamps the smooth-blend `IdealDiode`
/// on every `BranchKind::Nonlinear` branch registered as one.
///
/// Returns max(|i_diode|) across all diode branches as the
/// "residual norm" indicator.
///
/// Implementation pattern: for each nonlinear branch, read the
/// terminal voltages from `x`, call
/// `evaluate_current_and_jacobian<IdealDiode>` to get
/// `(i, partials)`, stamp:
///   f(from) += i;          f(to) -= i
///   J(from, from) += g00;  J(from, to) += g01
///   J(to,   from) -= g00;  J(to,   to) -= g01
inline Real refresh_smooth_diodes(
    const Vector& x,
    sparse::Matrix& J_nl,
    Vector& f_nl,
    const topology::Graph& graph,
    const DevicePool& pool) {
    // Zero out previous contributions.
    if (J_nl.rows() > 0) {
        J_nl.setZero();
    }
    if (f_nl.size() > 0) {
        f_nl.setZero();
    }

    Real max_abs_i = Real{0};

    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        const auto& branch = graph.branch(b_id);
        if (branch.kind != topology::BranchKind::Nonlinear) {
            continue;
        }
        // Only handle branches registered as smooth-blend diodes.
        // We use a sentinel: kind_of == Diode means SwitchedDiode
        // (the binary-on/off model). We need a distinct
        // `nonlinear_diode_params` slot in DevicePool — see the
        // add method below.
        const auto kind = pool.kind_of(branch.id);
        if (kind != DevicePool::StoredKind::NonlinearDiode) {
            continue;
        }
        const auto& p = pool.nonlinear_diode_params(branch.id);

        const stamping::BranchCoord coord{branch.from, branch.to,
                                            branch.id};
        const Real v_from =
            stamping::read_node_voltage(x, coord.from);
        const Real v_to =
            stamping::read_node_voltage(x, coord.to);
        const models::ModelInputs<models::IdealDiode> v_term{
            v_from, v_to};

        const auto [i, partials] =
            models::evaluate_current_and_jacobian<models::IdealDiode>(
                v_term, p);

        const bool from_active =
            stamping::node_is_active(coord.from);
        const bool to_active =
            stamping::node_is_active(coord.to);

        // Residual contributions.
        if (from_active) f_nl[coord.from] += i;
        if (to_active)   f_nl[coord.to]   -= i;

        // Jacobian contributions (standard 2x2 conductance pattern).
        if (from_active) {
            J_nl.coeffRef(coord.from, coord.from) += partials[0];
            if (to_active) {
                J_nl.coeffRef(coord.from, coord.to) += partials[1];
            }
        }
        if (to_active) {
            if (from_active) {
                J_nl.coeffRef(coord.to, coord.from) -= partials[0];
            }
            J_nl.coeffRef(coord.to, coord.to) -= partials[1];
        }

        max_abs_i = std::max(max_abs_i, std::abs(i));
    }

    return max_abs_i;
}

/// Layer 4 V8 — returns a NonlinearRefreshFn that stamps
/// smooth-blend IdealDiode contributions using `kappa_override`
/// instead of each diode's pool-stored kappa. Other params
/// (V_F0, R_d, G_off) come from the pool unchanged.
///
/// Used by `continuation_solve` to ramp kappa from "easy"
/// (smooth) to "hard" (target) progressively. Each step warm-
/// starts from the previous, guiding Newton through the
/// residual landscape past local minima.
[[nodiscard]] inline NonlinearRefreshFn
make_kappa_override_refresh(Real kappa_override) {
    return [kappa_override](const Vector& x,
                              sparse::Matrix& J_nl,
                              Vector& f_nl,
                              const topology::Graph& graph,
                              const DevicePool& pool) -> Real {
        if (J_nl.rows() > 0) J_nl.setZero();
        if (f_nl.size() > 0) f_nl.setZero();

        Real max_abs_i = Real{0};

        for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
            const auto& branch = graph.branch(b_id);
            if (branch.kind != topology::BranchKind::Nonlinear) {
                continue;
            }
            const auto kind = pool.kind_of(branch.id);
            if (kind != DevicePool::StoredKind::NonlinearDiode) {
                continue;
            }
            const auto& pool_p =
                pool.nonlinear_diode_params(branch.id);
            const models::IdealDiode::Params p_override{
                .V_F0  = pool_p.V_F0,
                .R_d   = pool_p.R_d,
                .G_off = pool_p.G_off,
                .kappa = kappa_override,
            };

            const stamping::BranchCoord coord{
                branch.from, branch.to, branch.id};
            const Real v_from =
                stamping::read_node_voltage(x, coord.from);
            const Real v_to =
                stamping::read_node_voltage(x, coord.to);
            const models::ModelInputs<models::IdealDiode> v_term{
                v_from, v_to};

            const auto [i, partials] =
                models::evaluate_current_and_jacobian<
                    models::IdealDiode>(v_term, p_override);

            const bool from_active =
                stamping::node_is_active(coord.from);
            const bool to_active =
                stamping::node_is_active(coord.to);

            if (from_active) f_nl[coord.from] += i;
            if (to_active)   f_nl[coord.to]   -= i;

            if (from_active) {
                J_nl.coeffRef(coord.from, coord.from) += partials[0];
                if (to_active) {
                    J_nl.coeffRef(coord.from, coord.to) += partials[1];
                }
            }
            if (to_active) {
                if (from_active) {
                    J_nl.coeffRef(coord.to, coord.from) -= partials[0];
                }
                J_nl.coeffRef(coord.to, coord.to) -= partials[1];
            }

            max_abs_i = std::max(max_abs_i, std::abs(i));
        }
        return max_abs_i;
    };
}

/// Layer 4 V9 — returns a NonlinearRefreshFn that stamps
/// smooth-blend IdealDiode contributions using
/// `V_F0_override` instead of each diode's pool-stored V_F0.
/// Other params (R_d, G_off, kappa) come from the pool
/// unchanged.
///
/// Companion to `make_kappa_override_refresh` (V8): both
/// override a single Param field while preserving the rest.
/// V_F0 continuation is the homotopy that bridges
/// "always-conducting" (V_F0 = −V_amp) → "realistic diode"
/// (V_F0 = 0.7) without crossing the operating-point
/// bifurcation that the smooth-blend exhibits near zero-
/// crossings at high κ. Used by `continuation_solve` to
/// solve stiff rectifiers from `x = 0`.
[[nodiscard]] inline NonlinearRefreshFn
make_vf0_override_refresh(Real V_F0_override) {
    return [V_F0_override](const Vector& x,
                            sparse::Matrix& J_nl,
                            Vector& f_nl,
                            const topology::Graph& graph,
                            const DevicePool& pool) -> Real {
        if (J_nl.rows() > 0) J_nl.setZero();
        if (f_nl.size() > 0) f_nl.setZero();

        Real max_abs_i = Real{0};

        for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
            const auto& branch = graph.branch(b_id);
            if (branch.kind != topology::BranchKind::Nonlinear) {
                continue;
            }
            const auto kind = pool.kind_of(branch.id);
            if (kind != DevicePool::StoredKind::NonlinearDiode) {
                continue;
            }
            const auto& pool_p =
                pool.nonlinear_diode_params(branch.id);
            const models::IdealDiode::Params p_override{
                .V_F0  = V_F0_override,
                .R_d   = pool_p.R_d,
                .G_off = pool_p.G_off,
                .kappa = pool_p.kappa,
            };

            const stamping::BranchCoord coord{
                branch.from, branch.to, branch.id};
            const Real v_from =
                stamping::read_node_voltage(x, coord.from);
            const Real v_to =
                stamping::read_node_voltage(x, coord.to);
            const models::ModelInputs<models::IdealDiode> v_term{
                v_from, v_to};

            const auto [i, partials] =
                models::evaluate_current_and_jacobian<
                    models::IdealDiode>(v_term, p_override);

            const bool from_active =
                stamping::node_is_active(coord.from);
            const bool to_active =
                stamping::node_is_active(coord.to);

            if (from_active) f_nl[coord.from] += i;
            if (to_active)   f_nl[coord.to]   -= i;

            if (from_active) {
                J_nl.coeffRef(coord.from, coord.from) += partials[0];
                if (to_active) {
                    J_nl.coeffRef(coord.from, coord.to) += partials[1];
                }
            }
            if (to_active) {
                if (from_active) {
                    J_nl.coeffRef(coord.to, coord.from) -= partials[0];
                }
                J_nl.coeffRef(coord.to, coord.to) -= partials[1];
            }

            max_abs_i = std::max(max_abs_i, std::abs(i));
        }
        return max_abs_i;
    };
}

/// Layer 4 V9.1 — returns a NonlinearRefreshFn that
/// stamps smooth-blend IdealDiode contributions using BOTH
/// `kappa_override` AND `V_F0_override`. Other params
/// (R_d, G_off) come from the pool unchanged.
///
/// Used to build combined κ + V_F0 continuation chains:
/// start at low κ AND extreme V_F0 (both make Newton
/// easy), ramp both simultaneously toward the target. The
/// combined chain is more robust than either alone for
/// the κ=20 stiff rectifier from `x = 0`.
[[nodiscard]] inline NonlinearRefreshFn
make_kappa_vf0_override_refresh(Real kappa_override,
                                  Real V_F0_override) {
    return [kappa_override, V_F0_override](
            const Vector& x,
            sparse::Matrix& J_nl,
            Vector& f_nl,
            const topology::Graph& graph,
            const DevicePool& pool) -> Real {
        if (J_nl.rows() > 0) J_nl.setZero();
        if (f_nl.size() > 0) f_nl.setZero();

        Real max_abs_i = Real{0};

        for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
            const auto& branch = graph.branch(b_id);
            if (branch.kind != topology::BranchKind::Nonlinear) {
                continue;
            }
            const auto kind = pool.kind_of(branch.id);
            if (kind != DevicePool::StoredKind::NonlinearDiode) {
                continue;
            }
            const auto& pool_p =
                pool.nonlinear_diode_params(branch.id);
            const models::IdealDiode::Params p_override{
                .V_F0  = V_F0_override,
                .R_d   = pool_p.R_d,
                .G_off = pool_p.G_off,
                .kappa = kappa_override,
            };

            const stamping::BranchCoord coord{
                branch.from, branch.to, branch.id};
            const Real v_from =
                stamping::read_node_voltage(x, coord.from);
            const Real v_to =
                stamping::read_node_voltage(x, coord.to);
            const models::ModelInputs<models::IdealDiode> v_term{
                v_from, v_to};

            const auto [i, partials] =
                models::evaluate_current_and_jacobian<
                    models::IdealDiode>(v_term, p_override);

            const bool from_active =
                stamping::node_is_active(coord.from);
            const bool to_active =
                stamping::node_is_active(coord.to);

            if (from_active) f_nl[coord.from] += i;
            if (to_active)   f_nl[coord.to]   -= i;

            if (from_active) {
                J_nl.coeffRef(coord.from, coord.from) += partials[0];
                if (to_active) {
                    J_nl.coeffRef(coord.from, coord.to) += partials[1];
                }
            }
            if (to_active) {
                if (from_active) {
                    J_nl.coeffRef(coord.to, coord.from) -= partials[0];
                }
                J_nl.coeffRef(coord.to, coord.to) -= partials[1];
            }

            max_abs_i = std::max(max_abs_i, std::abs(i));
        }
        return max_abs_i;
    };
}

}  // namespace pulsim::v2::pwl
