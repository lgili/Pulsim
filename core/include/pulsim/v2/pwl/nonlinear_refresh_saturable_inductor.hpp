#pragma once

// =============================================================================
// Pulsim v2 — Layer 4 V17: Saturable-inductor Newton refresh
// =============================================================================
//
// Per-Newton-iteration stamping for `SaturableInductor` branches.
//
// Trap rule (with L = L(i_L_new)):
//
//   V_L_new + V_L_old = (2·L_eff/dt) · (i_L_new − i_L_old)
//
// where V_L_new = v_from − v_to and L_eff = L(i_L_new).
// Rearranged as the constraint row residual (J·x + f = 0 form):
//
//   R_row = v_from − v_to + V_L_old − (2·L_eff/dt)·(i_L_new − i_L_old) = 0
//
// Linearisation (∂R/∂x):
//   ∂R/∂v_from        = +1
//   ∂R/∂v_to          = -1
//   ∂R/∂i_L_new       = -(2·L_eff/dt) - (2/dt)·(i_L_new − i_L_old)·dL/di
//
// KCL contributions at from/to (current flows i_L from `from` to `to`):
//   f[from] += i_L_new        J[from, i_L_row] += 1
//   f[to]   -= i_L_new        J[to,   i_L_row] -= 1
//
// Uses AD via `evaluate_current_and_jacobian` on the saturable
// inductor model (which returns L(i) and dL/di).

#include "pulsim/v2/models/device_model.hpp"
#include "pulsim/v2/models/saturable_inductor.hpp"
#include "pulsim/v2/numeric/dense.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/pwl/nonlinear_solve.hpp"
#include "pulsim/v2/pwl/saturable_inductor_history.hpp"
#include "pulsim/v2/sparse/matrix.hpp"
#include "pulsim/v2/stamping/branch_coord.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <algorithm>
#include <cmath>

namespace pulsim::v2::pwl {

/// Stamp the SaturableInductor contribution. Reads i_L_old
/// and V_L_old from `history`. Clears J_nl / f_nl first (so
/// this is a standalone refresh; for composition see the
/// combined refresh below). Returns max(|i_L_new|).
inline Real refresh_saturable_inductors(
    const Vector& x,
    sparse::Matrix& J_nl,
    Vector& f_nl,
    const topology::Graph& /*graph*/,
    const DevicePool& /*pool*/,
    const SaturableInductorHistory& history,
    Real dt) {
    if (J_nl.rows() > 0) J_nl.setZero();
    if (f_nl.size() > 0) f_nl.setZero();
    Real max_abs_i = Real{0};

    for (const auto& e : history.entries()) {
        const Real v_from =
            stamping::read_node_voltage(x, e.from);
        const Real v_to =
            stamping::read_node_voltage(x, e.to);
        const Real i_L_new = x[e.branch_var_id];

        // Evaluate L(i_L_new) and dL/di_L via AD.
        const models::ModelInputs<models::SaturableInductor>
            iv{i_L_new};
        const auto [L_eff, partials] =
            models::evaluate_current_and_jacobian<
                models::SaturableInductor>(iv, e.params);
        const Real dL_di = partials[0];

        const Real two_over_dt = Real{2} / dt;
        const Real two_L_over_dt = two_over_dt * L_eff;
        const Real di = i_L_new - e.i_L_old;

        // Constraint-row residual.
        const Real R_row =
            (v_from - v_to) + e.V_L_old -
            two_L_over_dt * di;

        // KCL residuals at terminals.
        const bool from_active =
            stamping::node_is_active(e.from);
        const bool to_active =
            stamping::node_is_active(e.to);
        if (from_active) f_nl[e.from] += i_L_new;
        if (to_active)   f_nl[e.to]   -= i_L_new;
        f_nl[e.branch_var_id] += R_row;

        // Jacobian: constraint row.
        if (from_active) {
            J_nl.coeffRef(e.branch_var_id, e.from) += Real{1};
        }
        if (to_active) {
            J_nl.coeffRef(e.branch_var_id, e.to)   -= Real{1};
        }
        const Real dR_di_L =
            -two_L_over_dt - two_over_dt * di * dL_di;
        J_nl.coeffRef(e.branch_var_id, e.branch_var_id)
            += dR_di_L;

        // Jacobian: KCL rows.
        if (from_active) {
            J_nl.coeffRef(e.from, e.branch_var_id) += Real{1};
        }
        if (to_active) {
            J_nl.coeffRef(e.to,   e.branch_var_id) -= Real{1};
        }

        max_abs_i = std::max(max_abs_i, std::abs(i_L_new));
    }
    return max_abs_i;
}

/// Combined refresh: smooth-blend diodes + SH1 MOSFETs +
/// IGBT Level 1 + SATURABLE INDUCTORS in one pass.
/// Inputs: history (saturable inductor (i_L, V_L)_old) and dt.
[[nodiscard]] inline NonlinearRefreshFn
make_combined_nonlinear_refresh(
    const SaturableInductorHistory* history,
    Real dt) {
    return [history, dt](const Vector& x,
                          sparse::Matrix& J_nl,
                          Vector& f_nl,
                          const topology::Graph& graph,
                          const DevicePool& pool) -> Real {
        if (J_nl.rows() > 0) J_nl.setZero();
        if (f_nl.size() > 0) f_nl.setZero();
        Real max_abs_i = Real{0};

        for (Index b_id = 0;
             b_id < graph.num_branches(); ++b_id) {
            const auto& branch = graph.branch(b_id);
            if (branch.kind !=
                    topology::BranchKind::Nonlinear) {
                continue;
            }
            const auto kind = pool.kind_of(branch.id);
            if (kind ==
                    DevicePool::StoredKind::NonlinearDiode) {
                const auto& p =
                    pool.nonlinear_diode_params(branch.id);
                const stamping::BranchCoord coord{
                    branch.from, branch.to, branch.id};
                const Real v_from =
                    stamping::read_node_voltage(x, coord.from);
                const Real v_to =
                    stamping::read_node_voltage(x, coord.to);
                const models::ModelInputs<models::IdealDiode>
                    v_term{v_from, v_to};
                const auto [i, partials] =
                    models::evaluate_current_and_jacobian<
                        models::IdealDiode>(v_term, p);
                const bool from_active =
                    stamping::node_is_active(coord.from);
                const bool to_active =
                    stamping::node_is_active(coord.to);
                if (from_active) f_nl[coord.from] += i;
                if (to_active)   f_nl[coord.to]   -= i;
                if (from_active) {
                    J_nl.coeffRef(coord.from, coord.from) +=
                        partials[0];
                    if (to_active) {
                        J_nl.coeffRef(coord.from, coord.to) +=
                            partials[1];
                    }
                }
                if (to_active) {
                    if (from_active) {
                        J_nl.coeffRef(coord.to, coord.from) -=
                            partials[0];
                    }
                    J_nl.coeffRef(coord.to, coord.to) -=
                        partials[1];
                }
                max_abs_i = std::max(max_abs_i, std::abs(i));
            } else if (kind ==
                    DevicePool::StoredKind::IgbtLevel1) {
                const auto& p =
                    pool.igbt_level1_params(branch.id);
                const Index collector = branch.from;
                const Index emitter   = branch.to;
                const Index gate      =
                    pool.igbt_level1_gate_node(branch.id);
                const Real v_c =
                    stamping::read_node_voltage(x, collector);
                const Real v_e =
                    stamping::read_node_voltage(x, emitter);
                const Real v_g =
                    stamping::read_node_voltage(x, gate);
                const models::ModelInputs<models::IgbtLevel1>
                    v_term{v_c, v_e, v_g};
                const auto [i, partials] =
                    models::evaluate_current_and_jacobian<
                        models::IgbtLevel1>(v_term, p);
                const bool c_active =
                    stamping::node_is_active(collector);
                const bool e_active =
                    stamping::node_is_active(emitter);
                const bool g_active =
                    stamping::node_is_active(gate);
                if (c_active) f_nl[collector] += i;
                if (e_active) f_nl[emitter]   -= i;
                if (c_active) {
                    J_nl.coeffRef(collector, collector) +=
                        partials[0];
                    if (e_active) {
                        J_nl.coeffRef(collector, emitter) +=
                            partials[1];
                    }
                    if (g_active) {
                        J_nl.coeffRef(collector, gate) +=
                            partials[2];
                    }
                }
                if (e_active) {
                    if (c_active) {
                        J_nl.coeffRef(emitter, collector) -=
                            partials[0];
                    }
                    J_nl.coeffRef(emitter, emitter) -=
                        partials[1];
                    if (g_active) {
                        J_nl.coeffRef(emitter, gate) -=
                            partials[2];
                    }
                }
                max_abs_i =
                    std::max(max_abs_i, std::abs(i));
            } else if (kind ==
                    DevicePool::StoredKind::MosfetLevel1) {
                const auto& p =
                    pool.mosfet_level1_params(branch.id);
                const Index drain  = branch.from;
                const Index source = branch.to;
                const Index gate   =
                    pool.mosfet_level1_gate_node(branch.id);
                const Real v_d =
                    stamping::read_node_voltage(x, drain);
                const Real v_s =
                    stamping::read_node_voltage(x, source);
                const Real v_g =
                    stamping::read_node_voltage(x, gate);
                const models::ModelInputs<
                        models::MosfetLevel1>
                    v_term{v_d, v_s, v_g};
                const auto [i, partials] =
                    models::evaluate_current_and_jacobian<
                        models::MosfetLevel1>(v_term, p);
                const bool d_active =
                    stamping::node_is_active(drain);
                const bool s_active =
                    stamping::node_is_active(source);
                const bool g_active =
                    stamping::node_is_active(gate);
                if (d_active) f_nl[drain]  += i;
                if (s_active) f_nl[source] -= i;
                if (d_active) {
                    J_nl.coeffRef(drain, drain) +=
                        partials[0];
                    if (s_active) {
                        J_nl.coeffRef(drain, source) +=
                            partials[1];
                    }
                    if (g_active) {
                        J_nl.coeffRef(drain, gate) +=
                            partials[2];
                    }
                }
                if (s_active) {
                    if (d_active) {
                        J_nl.coeffRef(source, drain) -=
                            partials[0];
                    }
                    J_nl.coeffRef(source, source) -=
                        partials[1];
                    if (g_active) {
                        J_nl.coeffRef(source, gate) -=
                            partials[2];
                    }
                }
                max_abs_i =
                    std::max(max_abs_i, std::abs(i));
            }
        }

        // Saturable inductors — handled via history.
        if (history != nullptr && !history->empty()) {
            for (const auto& e : history->entries()) {
                const Real v_from =
                    stamping::read_node_voltage(x, e.from);
                const Real v_to =
                    stamping::read_node_voltage(x, e.to);
                const Real i_L_new = x[e.branch_var_id];

                const models::ModelInputs<
                        models::SaturableInductor> iv{i_L_new};
                const auto [L_eff, partials] =
                    models::evaluate_current_and_jacobian<
                        models::SaturableInductor>(iv, e.params);
                const Real dL_di = partials[0];

                const Real two_over_dt = Real{2} / dt;
                const Real two_L_over_dt = two_over_dt * L_eff;
                const Real di = i_L_new - e.i_L_old;

                const Real R_row =
                    (v_from - v_to) + e.V_L_old -
                    two_L_over_dt * di;

                const bool from_active =
                    stamping::node_is_active(e.from);
                const bool to_active =
                    stamping::node_is_active(e.to);
                if (from_active) f_nl[e.from] += i_L_new;
                if (to_active)   f_nl[e.to]   -= i_L_new;
                f_nl[e.branch_var_id] += R_row;

                if (from_active) {
                    J_nl.coeffRef(e.branch_var_id, e.from)
                        += Real{1};
                }
                if (to_active) {
                    J_nl.coeffRef(e.branch_var_id, e.to)
                        -= Real{1};
                }
                const Real dR_di_L =
                    -two_L_over_dt - two_over_dt * di * dL_di;
                J_nl.coeffRef(e.branch_var_id, e.branch_var_id)
                    += dR_di_L;

                if (from_active) {
                    J_nl.coeffRef(e.from, e.branch_var_id)
                        += Real{1};
                }
                if (to_active) {
                    J_nl.coeffRef(e.to, e.branch_var_id)
                        -= Real{1};
                }
                max_abs_i =
                    std::max(max_abs_i, std::abs(i_L_new));
            }
        }
        return max_abs_i;
    };
}

}  // namespace pulsim::v2::pwl
