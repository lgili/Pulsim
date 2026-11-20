#pragma once

// =============================================================================
// Pulsim — Layer 4 V2: DC operating-point MNA assembly
// =============================================================================
//
// `pulsim-v2-dc-operating-point` Phase 1.
//
// `dc_assemble` is a sibling of `assemble_segment` that builds
// the DC MNA matrix for a given switch state. Differences from
// the trap-companion assembly:
//   * Capacitors are SKIPPED (g_eq = 0 at DC, no contribution).
//   * Inductors are stamped as v_from − v_to = 0 short circuits
//     (with the branch-current unknown still present so i_L is
//     recoverable from the solution).
//   * Resistors, VoltageSources, Switches, Diodes stamp as
//     usual (their stamps are dt-independent).
//
// `compute_dc_op` solves the DC system and returns the state
// vector. Throws on singular matrices.

#include "pulsim/models/current_source.hpp"
#include "pulsim/models/induction_machine_mna.hpp"
#include "pulsim/models/pmsm_mna.hpp"
#include "pulsim/models/pulse_voltage_source.hpp"
#include "pulsim/models/pwm_voltage_source.hpp"
#include "pulsim/models/resistor.hpp"
#include "pulsim/models/sine_voltage_source.hpp"
#include "pulsim/models/vcvs.hpp"
#include "pulsim/models/voltage_source.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/assemble.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/gmin.hpp"
#include "pulsim/pwl/preflight.hpp"
#include "pulsim/pwl/row_names.hpp"
#include "pulsim/pwl/nonlinear_solve.hpp"
#include "pulsim/pwl/segment.hpp"
#include "pulsim/solver/unsupported.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/sparse/solver.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/stamping/stamp_current_source.hpp"
#include "pulsim/stamping/stamp_device.hpp"
#include "pulsim/stamping/stamp_switch.hpp"
#include "pulsim/stamping/stamp_ideal_transformer.hpp"
#include "pulsim/stamping/stamp_vcvs.hpp"
#include "pulsim/stamping/stamp_voltage_source.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pulsim::pwl {

/// Stamp an inductor as a DC short: v_from − v_to = 0 with i_L
/// as a branch unknown. The constraint row reads "+1 at from,
/// −1 at to, 0 at i_L" (the i_L coefficient is zero, but the
/// branch-current unknown still exists in `x` so we can read
/// i_L from it). The KCL contributions of i_L on the terminal
/// rows are identical to the trap case.
inline void stamp_inductor_dc(sparse::Matrix& J, Vector& b,
                                const stamping::BranchCoord& coord,
                                Index branch_var_id) noexcept {
    (void)b;
    const bool from_active = stamping::node_is_active(coord.from);
    const bool to_active   = stamping::node_is_active(coord.to);

    // KCL contributions of i_L on terminal rows.
    if (from_active) {
        J.coeffRef(coord.from, branch_var_id) += Real{1};
    }
    if (to_active) {
        J.coeffRef(coord.to, branch_var_id) -= Real{1};
    }

    // Constraint row (v_from − v_to = 0).
    if (from_active) {
        J.coeffRef(branch_var_id, coord.from) += Real{1};
    }
    if (to_active) {
        J.coeffRef(branch_var_id, coord.to) -= Real{1};
    }
    // No diagonal entry on branch_var_id — DC inductor is just
    // a v=0 constraint, no L · di/dt term. (The system is
    // structurally singular along that diagonal until the v
    // constraint forces a unique solution via the rest of the
    // matrix.)
    //
    // To avoid singularity from missing diagonals we add a
    // tiny epsilon — physically equivalent to a near-zero
    // resistance in series with the inductor. KLU handles it.
    J.coeffRef(branch_var_id, branch_var_id) += Real{-1e-12};
}

/// add-permeance-magnetic-domain: the magnetic nodes whose DC row needs
/// flux conservation. At DC a permeance is open. A magnetic node that
/// touches only permeances has an empty KCL row; nodes joined only by
/// magnetic resistances and winding ports (no other path) have KCL rows
/// that sum to zero (each such branch current enters one node of the
/// group and leaves another). Either way the missing equation is that
/// the net flux leaving the group is zero, Σ ±Φ_k = 0 (the reluctance-
/// network solution from zero flux). Indexed by node: −1 for an ordinary
/// row, otherwise the group's row that carries Σ ±C·ΔF (the lowest node
/// of the group; its resistor terms sum to zero over the group, so the
/// other rows' KCL makes them vanish). Every node of a group skips the
/// conductance floor. Empty when the circuit has no magnetic node, so
/// every other circuit assembles exactly as before.
[[nodiscard]] inline std::vector<Index> charge_rows(const topology::Graph& graph,
                                                    const DevicePool& pool) {
    std::vector<Index> rows;
    if (pool.flux_nodes().empty()) return rows;
    const Index nn = graph.num_nodes();
    std::vector<char> cand(static_cast<Size>(nn), char{0});
    for (const Index n : pool.flux_nodes()) {
        if (n >= 0 && n < nn) cand[static_cast<Size>(n)] = 1;
    }
    using SK = DevicePool::StoredKind;
    auto clear = [&](Index n) {
        if (n >= 0 && n < nn) cand[static_cast<Size>(n)] = 0;
    };
    std::vector<std::pair<Index, Index>> r_edges;
    for (Index b = 0; b < graph.num_branches(); ++b) {
        const auto& br = graph.branch(b);
        const bool reg = pool.is_registered(b);
        const auto k = reg ? pool.kind_of(b) : SK::Resistor;
        if (reg && (k == SK::Capacitor || k == SK::NonlinearCapacitor)) continue;
        if (reg && k == SK::Resistor) {
            r_edges.emplace_back(br.from, br.to);
            continue;
        }
        if (reg && k == SK::Gyrator) {
            // A winding: its graph branch is the magnetic port, a group
            // edge like a resistance; its electrical port is a pair of
            // node references.
            r_edges.emplace_back(br.from, br.to);
            const auto [p, q] = pool.control_nodes(b);
            clear(p);
            clear(q);
            continue;
        }
        clear(br.from);
        clear(br.to);
        if (!reg) continue;
        // Node references that carry current without being a graph edge.
        if (k == SK::IdealTransformer) {
            const auto [p, q] = pool.ideal_transformer_primary_nodes(b);
            clear(p);
            clear(q);
        }
    }
    // Groups of candidates joined by resistors and winding ports; such an
    // edge to any other node (or to ground) anchors the group: its KCL
    // rows determine it.
    std::vector<Index> parent(static_cast<Size>(nn));
    for (Index i = 0; i < nn; ++i) parent[static_cast<Size>(i)] = i;
    auto find = [&](Index i) {
        while (parent[static_cast<Size>(i)] != i) {
            parent[static_cast<Size>(i)] = parent[static_cast<Size>(parent[static_cast<Size>(i)])];
            i = parent[static_cast<Size>(i)];
        }
        return i;
    };
    auto is_cand = [&](Index n) { return n >= 0 && n < nn && cand[static_cast<Size>(n)] != 0; };
    for (const auto [a, c] : r_edges) {
        if (is_cand(a) && is_cand(c)) {
            const Index ra = find(a), rc = find(c);
            parent[static_cast<Size>(std::max(ra, rc))] = std::min(ra, rc);
        }
    }
    std::vector<char> anchored(static_cast<Size>(nn), char{0});
    for (const auto [a, c] : r_edges) {
        if (is_cand(a) != is_cand(c)) anchored[static_cast<Size>(find(is_cand(a) ? a : c))] = 1;
    }
    rows.assign(static_cast<Size>(nn), Index{-1});
    for (Index i = 0; i < nn; ++i) {
        if (is_cand(i) && anchored[static_cast<Size>(find(i))] == 0) {
            rows[static_cast<Size>(i)] = find(i);  // the root is the lowest node
        }
    }
    return rows;
}

/// The conductance floor on every node row except those of a flux-
/// conservation group (a gmin there would be a flux leak, and the
/// gmin-stepping ramp starts at 1e-2 S against permeances of ~1e-7 H).
inline void stamp_gmin_dc(sparse::Matrix& J, Index num_nodes, Real g,
                          const std::vector<Index>& rows) {
    if (rows.empty()) {
        stamp_gmin(J, num_nodes, g);
        return;
    }
    if (!(g > Real{0})) return;
    const Index n = std::min<Index>(num_nodes, static_cast<Index>(J.rows()));
    for (Index i = 0; i < n; ++i) {
        if (rows[static_cast<Size>(i)] < 0) J.coeffRef(i, i) += g;
    }
}

/// Build the DC MNA matrix. `t_eval` is the evaluation time
/// for time-varying sources (PWM / Sine / Pulse): the DC
/// operating point is computed treating each source as if it
/// were held at its instantaneous value at t = t_eval. Default
/// 0 reproduces the historical behaviour for circuits with
/// only DC sources (VoltageSource / CurrentSource).
///
/// Time-varying sources dispatched here:
///   * VoltageSource      — V = p.V (DC)
///   * CurrentSource      — I = p.I (DC, no branch unknown)
///   * PWMVoltageSource   — V = PWMVoltageSource::value_at(p, t_eval)
///   * SineVoltageSource  — V = SineVoltageSource::value_at(p, t_eval)
///   * PulseVoltageSource — V = PulseVoltageSource::value_at(p, t_eval)
///   * VCVS               — linear gain stamp (no time)
///
/// `source_scale` multiplies every INDEPENDENT source amplitude — the
/// homotopy parameter for source stepping. Dependent sources (VCVS)
/// are deliberately left alone: their gain is a property of the
/// circuit, not an excitation level, and ramping it would change what
/// is being solved rather than how hard it is.
///
/// `gmin` is the conductance floor stamped node-to-ground; see
/// `gmin.hpp`. Zero (the default) leaves the matrix untouched.
namespace detail {

/// add-mna-native-machines 2.7 / 3.5 (DECISIONS D19): the DC rows of the
/// MNA-native machines at the held mechanical state (ω0, θ0). Stator:
/// `v − R·i − dλ/dt = 0` with di/dt = 0, so dλ/dt = ω_e·∂λ/∂θ_e (PMSM)
/// and M·dλ_r/dt = 0 (IM: the rotor rows below are dλ_r/dt = 0). Linear
/// at a held speed. `f = J·x + b = 0`, as the rest of the assembly.
/// ω0 / θ0 are fixed values, not sources: a source-stepping pass does
/// not scale them (the rows are linear and well posed at every scale).
inline void stamp_machines_dc(const topology::Graph& graph,
                              const DevicePool& pool, sparse::Matrix& J,
                              Vector& b) {
    const auto stator = [&](const std::array<Index, 3>& ids, const auto& coef,
                            const std::array<Real, 3>& emf) {
        std::array<Index, 3> row{};
        for (Size k = 0; k < 3; ++k) {
            row[k] = pool.branch_var_id_for_inductor(ids[k], graph);
        }
        for (Size k = 0; k < 3; ++k) {
            const auto& br = graph.branch(ids[k]);
            if (stamping::node_is_active(br.from)) {
                J.coeffRef(row[k], br.from) += Real{1};
                J.coeffRef(br.from, row[k]) += Real{1};
            }
            if (stamping::node_is_active(br.to)) {
                J.coeffRef(row[k], br.to) -= Real{1};
                J.coeffRef(br.to, row[k]) -= Real{1};
            }
            for (Size j = 0; j < 3; ++j) J.coeffRef(row[k], row[j]) -= coef(k, j);
            b[row[k]] -= emf[k];
        }
        return row;
    };
    for (const auto& m : pool.pmsm_mna_machines()) {
        const auto& p = m.params;
        const Real th_e = p.pole_pairs * m.theta0;
        const Real w_e = p.pole_pairs * m.omega0;
        if (p.flux_map && w_e != Real{0}) {
            // v = R·i + ω_e·∂λ/∂θ_e(i) is nonlinear in i with a map; the DC
            // point here is linear (no Newton). At ω0 = 0 it is v = R·i.
            throw std::invalid_argument(std::format(
                "DC operating point: the flux-map PMSM '{}' is held at "
                "omega0 = {} rad/s; its held-mechanics DC point is only "
                "available at omega0 = 0 (the stator law is nonlinear in i "
                "when spinning). Start from zero state instead.",
                graph.branch_name(m.branch_id[0]), m.omega0));
        }
        const auto ind = models::PmsmMna::inductance(p, th_e);
        const auto lp1 = models::PmsmMna::lambda_pm(p, th_e, 1);
        std::array<Real, 3> emf{};
        for (Size k = 0; k < 3; ++k) emf[k] = w_e * lp1[k];
        stator(m.branch_id, [&](Size k, Size j) {
            return (k == j ? p.R_s : Real{0}) + w_e * ind.dL[k][j];
        }, emf);
    }
    using IM = models::InductionMachineMna;
    for (const auto& m : pool.induction_machines()) {
        const auto& p = m.params;
        const auto row = stator(m.branch_id, [&](Size k, Size j) {
            return k == j ? p.R_s : Real{0};
        }, std::array<Real, 3>{});
        const auto dv = IM::derived(p);
        const auto C = IM::clarke();
        const Real w_e = p.pole_pairs * m.omega0;
        const Real dg_dl[2][2] = {{-dv.inv_tau_r, -w_e}, {w_e, -dv.inv_tau_r}};
        for (Size n = 0; n < 2; ++n) {   // dλ_r/dt = g(i, λ_r, ω0) = 0
            const Index rn = m.rotor_node[n];
            if (!stamping::node_is_active(rn)) continue;
            for (Size q = 0; q < 2; ++q) {
                if (stamping::node_is_active(m.rotor_node[q])) {
                    J.coeffRef(rn, m.rotor_node[q]) -= dg_dl[n][q];
                }
            }
            for (Size j = 0; j < 3; ++j) {
                J.coeffRef(rn, row[j]) -= dv.Lm_over_tau_r * C[n][j];
            }
        }
    }
}

/// Replace the KCL row of each machine's speed and angle node by
/// `v = ω0` / `v = θ0` (the DC point holds the mechanics; no torque
/// balance). A node another device also fixes (a source, an inductor,
/// a switch: anything but capacitors, resistors and current outputs)
/// would be over-determined: refused by name.
inline void hold_machine_mechanics(const topology::Graph& graph,
                                   const DevicePool& pool, sparse::Matrix& J,
                                   Vector& b) {
    using SK = DevicePool::StoredKind;
    std::vector<std::pair<Index, Real>> held;
    const auto hold = [&](Index node, Real value, Index phase_a,
                          const char* what) {
        if (!stamping::node_is_active(node)) return;
        for (const auto& [n, v] : held) {
            if (n != node) continue;
            if (v == value) return;
            throw std::invalid_argument(std::format(
                "DC operating point: two machines hold the {} node '{}' at "
                "different values ({} and {})", what, graph.node(node).name,
                v, value));
        }
        for (Index br = 0; br < graph.num_branches(); ++br) {
            const auto& e = graph.branch(br);
            if (e.from != node && e.to != node) continue;
            if (!pool.is_registered(br)) continue;
            const auto k = pool.kind_of(br);
            if (k == SK::Capacitor || k == SK::Resistor || k == SK::CurrentSource ||
                k == SK::VCCS || k == SK::CCCS || k == SK::SineCurrentSource ||
                k == SK::PulseCurrentSource || k == SK::PwlCurrentSource) {
                continue;
            }
            throw std::invalid_argument(std::format(
                "DC operating point: machine '{}' holds its {} node '{}' at "
                "its initial value (omega0 / theta0), but '{}' also fixes "
                "that node (a source, an inductor, a transformer winding, a "
                "switch: a prescribed or shared shaft). The DC point cannot "
                "hold a node twice: prescribe the speed with omega0 alone, or "
                "start the transient from zero state / initial conditions.",
                graph.branch_name(phase_a), what, graph.node(node).name,
                graph.branch_name(br)));
        }
        held.emplace_back(node, value);
    };
    for (const auto& m : pool.pmsm_mna_machines()) {
        hold(m.omega_node, m.omega0, m.branch_id[0], "speed");
        hold(m.theta_node, m.theta0, m.branch_id[0], "angle");
    }
    for (const auto& m : pool.induction_machines()) {
        hold(m.omega_node, m.omega0, m.branch_id[0], "speed");
    }
    if (held.empty()) return;
    std::vector<char> is_held(static_cast<Size>(J.rows()), 0);
    for (const auto& [n, v] : held) is_held[static_cast<Size>(n)] = 1;
    // keep the diagonal entry so the overwrite below never inserts
    J.prune([&](const Index& r, const Index& c, const Real&) {
        return r == c || is_held[static_cast<Size>(r)] == 0;
    });
    for (const auto& [n, v] : held) {
        J.coeffRef(n, n) = Real{1};
        b[n] = -v;
    }
}

}  // namespace detail

inline void dc_assemble(const topology::Graph& graph,
                         const DevicePool& pool,
                         const topology::SwitchStateMask& mask,
                         sparse::Matrix& J,
                         Vector& b,
                         Real t_eval = Real{0},
                         Real gmin = Real{0},
                         Real source_scale = Real{1},
                         bool hold_mechanics = true) {
    const Size state_size = pool.state_size(graph);
    J = sparse::Matrix(static_cast<Index>(state_size),
                        static_cast<Index>(state_size));
    b = Vector::Zero(static_cast<Index>(state_size));

    if (state_size == 0) {
        return;
    }

    const Vector x = Vector::Zero(static_cast<Index>(state_size));
    Size switch_idx = 0;
    const std::vector<Index> q_rows = charge_rows(graph, pool);
    auto q_row = [&](Index n) {
        return (q_rows.empty() || n < 0) ? Index{-1} : q_rows[static_cast<Size>(n)];
    };
    auto is_q_row = [&](Index n) { return q_row(n) >= 0; };

    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        const auto& branch = graph.branch(b_id);
        const stamping::BranchCoord coord{branch.from, branch.to,
                                            branch.id};

        switch (branch.kind) {
        case topology::BranchKind::PassiveLinear: {
            const auto k = pool.kind_of(branch.id);
            switch (k) {
            case DevicePool::StoredKind::Resistor: {
                const auto& p = pool.resistor_params(branch.id);
                stamping::stamp_device<models::Resistor>(J, b, x,
                                                          coord, p);
                break;
            }
            case DevicePool::StoredKind::Capacitor: {
                // Open circuit at DC — except into a flux-conservation
                // row, where it adds its charge ±C·(v_from − v_to) (= 0).
                const Real C = pool.capacitor_params(branch.id).C;
                for (const auto [row, s] : {std::pair{q_row(coord.from), Real{1}},
                                            std::pair{q_row(coord.to), Real{-1}}}) {
                    if (row < 0) continue;
                    if (stamping::node_is_active(coord.from)) J.coeffRef(row, coord.from) += s * C;
                    if (stamping::node_is_active(coord.to)) J.coeffRef(row, coord.to) -= s * C;
                }
                break;
            }
            case DevicePool::StoredKind::Inductor: {
                const Index branch_var_id =
                    pool.branch_var_id_for_inductor(branch.id, graph);
                stamp_inductor_dc(J, b, coord, branch_var_id);
                break;
            }
            default:
                break;
            }
            break;
        }
        case topology::BranchKind::Source: {
            if (detail::stamp_controlled_source(graph, pool, branch.id, J,
                                                b, x, coord, /*at_dc=*/true,
                                                t_eval, source_scale)) {
                break;
            }
            const auto k = pool.kind_of(branch.id);
            switch (k) {
            case DevicePool::StoredKind::VoltageSource: {
                const auto& p = pool.voltage_source_params(branch.id);
                const Index branch_var_id =
                    pool.branch_var_id_for_source(branch.id, graph);
                stamping::stamp_voltage_source(J, b, x, coord,
                                                branch_var_id,
                                                source_scale * p.V);
                break;
            }
            case DevicePool::StoredKind::CurrentSource: {
                const auto& p =
                    pool.current_source_params(branch.id);
                stamping::stamp_current_source(b, coord,
                                                source_scale * p.I);
                break;
            }
            case DevicePool::StoredKind::PWMVoltageSource: {
                const auto& p =
                    pool.pwm_voltage_source_params(branch.id);
                const Index branch_var_id =
                    pool.branch_var_id_for_source(branch.id, graph);
                const Real V_t = source_scale *
                    models::PWMVoltageSource::value_at(p, t_eval);
                stamping::stamp_voltage_source(J, b, x, coord,
                                                branch_var_id, V_t);
                break;
            }
            case DevicePool::StoredKind::SineVoltageSource: {
                const auto& p =
                    pool.sine_voltage_source_params(branch.id);
                const Index branch_var_id =
                    pool.branch_var_id_for_source(branch.id, graph);
                const Real V_t = source_scale *
                    models::SineVoltageSource::value_at(p, t_eval);
                stamping::stamp_voltage_source(J, b, x, coord,
                                                branch_var_id, V_t);
                break;
            }
            case DevicePool::StoredKind::PulseVoltageSource: {
                const auto& p =
                    pool.pulse_voltage_source_params(branch.id);
                const Index branch_var_id =
                    pool.branch_var_id_for_source(branch.id, graph);
                const Real V_t = source_scale *
                    models::PulseVoltageSource::value_at(p, t_eval);
                stamping::stamp_voltage_source(J, b, x, coord,
                                                branch_var_id, V_t);
                break;
            }
            case DevicePool::StoredKind::VCVS: {
                const auto& p = pool.vcvs_params(branch.id);
                const Index branch_var_id =
                    pool.branch_var_id_for_source(branch.id, graph);
                const auto [in_pos, in_neg] =
                    pool.vcvs_input_nodes(branch.id);
                stamping::stamp_vcvs(J, b, x, coord,
                                      in_pos, in_neg,
                                      branch_var_id, p.gain);
                break;
            }
            case DevicePool::StoredKind::IdealTransformer: {
                // An ideal transformer transforms DC too — it has no
                // frequency dependence. It is L_m across the primary
                // that carries the DC short, and that is a separate
                // device. Stamp it unchanged.
                const auto& p = pool.ideal_transformer_params(branch.id);
                const Index branch_var_id =
                    pool.branch_var_id_for_source(branch.id, graph);
                const auto [p_from, p_to] =
                    pool.ideal_transformer_primary_nodes(branch.id);
                stamping::stamp_ideal_transformer(
                    J, b, x, coord, p_from, p_to, branch_var_id, p.n);
                break;
            }
            case DevicePool::StoredKind::TransmissionLine: {
                // add-realistic-passives D18 (5): a lossless line is a wire
                // at DC, i.e. a 1:1 ideal transformer: port b's row
                // v_b − v_a = 0 (the secondary's), port a's row i_a + i_b = 0.
                const Index row = pool.branch_var_id_for_source(branch.id, graph);
                const Index other_id = pool.line_partner(branch.id);
                const Index other = pool.branch_var_id_for_source(other_id, graph);
                if (pool.is_line_port_b(branch.id)) {
                    stamping::stamp_voltage_source(J, b, x, coord, row, Real{0});
                    const auto& oa = graph.branch(other_id);
                    if (stamping::node_is_active(oa.from)) J.coeffRef(row, oa.from) -= Real{1};
                    if (stamping::node_is_active(oa.to)) J.coeffRef(row, oa.to) += Real{1};
                } else {
                    if (stamping::node_is_active(coord.from)) J.coeffRef(coord.from, row) += Real{1};
                    if (stamping::node_is_active(coord.to)) J.coeffRef(coord.to, row) -= Real{1};
                    J.coeffRef(row, row) += Real{1};
                    J.coeffRef(row, other) += Real{1};
                }
                break;
            }
            default:
                break;
            }
            break;
        }
        case topology::BranchKind::Switch: {
            detail::stamp_switch_branch(pool, branch.id,
                                        mask.get(switch_idx), J, b, x,
                                        coord, &graph);
            ++switch_idx;
            break;
        }
        case topology::BranchKind::Nonlinear: {
            // Newton devices are stamped by their refresh, not here —
            // EXCEPT the flux devices, whose DC law is exact and
            // linear: v = dλ/dt = 0, a short, whatever λ(i) does.
            // Left out, a saturable inductor was OPEN at DC, so a
            // DC operating point saw it as a break in the circuit,
            // and an ideal transformer whose magnetising branch it
            // is would have passed DC straight through to the
            // secondary (n·V_p instead of 0). The linear inductor's
            // own DC stamp is the right one.
            if (pool.kind_of(branch.id)
                    == DevicePool::StoredKind::NonlinearCapacitor &&
                (is_q_row(coord.from) || is_q_row(coord.to))) {
                // shortcut: flux rows are linear-only; a nonlinear one
                // needs the Newton DC refresh (the unified device
                // interface) — refused by name until then.
                throw std::runtime_error(std::format(
                    "DC operating point: the nonlinear permeance '{}' touches "
                    "a magnetic node with no winding, whose DC row is flux "
                    "conservation; that row is linear-only for now. Start the "
                    "transient from zero flux (the default) or from initial "
                    "conditions instead of a DC operating point.",
                    graph.branch_name(branch.id)));
            }
            if (pool.kind_of(branch.id)
                == DevicePool::StoredKind::SaturableInductor) {
                const Index branch_var_id =
                    pool.branch_var_id_for_inductor(branch.id, graph);
                stamp_inductor_dc(J, b, coord, branch_var_id);
            } else if (pool.kind_of(branch.id)
                       == DevicePool::StoredKind::ExternalDevice) {
                // add-external-device-abi: the law (f, ∂f/∂x) is stamped by
                // the DC refresh; each branch carrying an internal unknown
                // gets the same −1e-12 diagonal as a DC inductor's branch
                // row (as the transient's G_min, DECISIONS #7), so a twin's
                // DC equations match the built-in's to the bit
                const auto& xp = pool.external_device_params(branch.id);
                const auto& br = xp.branches;
                const auto k = static_cast<std::size_t>(
                    std::find(br.begin(), br.end(), branch.id) - br.begin());
                if (k < xp.inst->model().d().n_internal) {
                    const Index bv = pool.branch_var_id_for_inductor(branch.id, graph);
                    J.coeffRef(bv, bv) += Real{-1e-12};
                }
            }
            break;
        }
        }
    }

    // v2.0 Phase 2 (B.2): the conductance floor, stamped last so it
    // lands on the node block only — see `gmin.hpp` for why the
    // branch-current rows must be left alone. Defaults to zero: the
    // assembler stays neutral and the SOLVER opts in, so AC / LTI /
    // sweep consumers that build their own matrices are unaffected.
    stamp_gmin_dc(J, graph.num_nodes(), gmin, q_rows);
    if (!pool.pmsm_mna_machines().empty() || !pool.induction_machines().empty()) {
        detail::stamp_machines_dc(graph, pool, J, b);
        if (hold_mechanics) detail::hold_machine_mechanics(graph, pool, J, b);
    }
}

/// A rank deficiency of the DC system that no amount of
/// conditioning may cover for.
struct DCStructuralDefect {
    bool present = false;
    Index row = kInvalidIndex;   //!< the offending unknown, if any
    std::string detail;          //!< " — ..." suffix for a message
};

/// Decide whether the DC system is structurally rank-deficient — and
/// therefore whether a conductance floor would be CONDITIONING the
/// matrix or INVENTING equations for it.
///
/// Two defects qualify, and they are found different ways:
///
///   1. **A galvanically isolated subnet.** Its nodes have plenty of
///      stamps, so no column is empty; the block simply has no
///      reference, and only reachability sees that. A floor would
///      quietly supply the reference and report an operating point
///      for a circuit whose isolation is the entire point of its
///      design. `preflight.hpp` owns this repair, with a report.
///   2. **An unknown with no equation at all** — an empty MNA column
///      or row. Checked before the next one because it names a more
///      specific mechanism for the same node.
///   3. **A subnet with no DC path to ground.** Galvanically
///      connected — through a coupling capacitor, or fed by an ideal
///      current source — but rank-deficient in the DC system, where
///      those devices contribute nothing. Emptiness cannot see this:
///      the subnet's own internal resistors populate every column.
///
/// Neither probe may libel a healthy circuit, so both are careful:
/// reachability unions EVERY branch regardless of kind (a diode is a
/// galvanic connection even though it does not conduct at DC), and
/// the emptiness probe is taken on the linear stamps UNIONED with
/// the nonlinear ones. `dc_assemble` skips `BranchKind::Nonlinear`,
/// so an interior node of a diode chain has an empty LINEAR column
/// while being perfectly well determined once the diodes are
/// stamped — the same false accusation Phase 1's review caught in
/// the singular-matrix message.
///
/// Nonlinear structure is taken at x = 0: sparsity is a property of
/// which devices touch which nodes, not of the operating point.
[[nodiscard]] inline DCStructuralDefect dc_structural_defect(
    const topology::Graph& graph,
    const DevicePool& pool,
    const topology::SwitchStateMask& mask,
    const NonlinearRefreshFn& refresh,
    Real t_eval) {
    DCStructuralDefect out;

    // 1. GALVANIC reachability: a subnet with no path to ground
    //    through ANY device has no voltage reference at all.
    const auto isolated = detail::components_without_ground(
        graph, [](const topology::Branch&) { return true; });
    if (!isolated.empty() && !isolated.front().empty()) {
        out.present = true;
        out.row = isolated.front().front();
        out.detail = std::format(
            " — {} is in a {}-node subnet with no connection to "
            "ground through any device, so its voltage is undefined "
            "rather than merely hard to compute. A conductance floor "
            "would invent a reference and report a confident answer; "
            "run the topology preflight (Python: it is on by "
            "default) to insert an explicit 1 GΩ tie, or add one "
            "yourself",
            node_label(graph, out.row), isolated.front().size());
        return out;
    }

    // 2. Emptiness: an unknown that appears in no equation.
    sparse::Matrix J;
    Vector b;
    dc_assemble(graph, pool, mask, J, b, t_eval, Real{0});
    sparse::compress_in_place(J);
    if (refresh) {
        const Index n = static_cast<Index>(b.size());
        sparse::Matrix J_nl(n, n);
        Vector f_nl = Vector::Zero(n);
        const Vector x0 = Vector::Zero(n);
        (void)refresh(x0, J_nl, f_nl, graph, pool);
        sparse::compress_in_place(J_nl);
        if (J_nl.nonZeros() > 0) {
            J += J_nl;
            sparse::compress_in_place(J);
        }
    }
    const Index col = sparse::first_empty_column(J);
    const Index row = (col == kInvalidIndex)
        ? sparse::first_empty_row(J) : col;
    if (row != kInvalidIndex) {
        out.present = true;
        out.row = row;
        out.detail = explain_singular(graph, pool, J, nullptr);
        return out;
    }

    // 3. DC reachability. Galvanic connection is not enough: a
    //    subnet whose only path to ground runs through capacitors
    //    (open at DC) or ideal current sources (no conductance) has
    //    a rank-deficient DC block even though every one of its
    //    columns is populated by its own internal devices. Probe 2
    //    above cannot see that — emptiness is not rank — and with
    //    the floor stamped the block becomes invertible and the
    //    solver reports 0 V, or I/(2·gmin) volts, as an operating
    //    point.
    //
    //    Nonlinear branches conduct here IF AND ONLY IF a refresh
    //    will stamp them. `dc_assemble` skips them, which is why
    //    `conducts_at_dc` says no; but Newton puts them back, and
    //    treating an interior node of a diode chain as floating
    //    would be the same false accusation this function exists to
    //    avoid.
    const bool nonlinear_is_stamped = static_cast<bool>(refresh);
    using SK = DevicePool::StoredKind;
    const auto dc_floating = detail::components_without_ground(
        graph, [&](const topology::Branch& br) {
            if (br.kind == topology::BranchKind::Nonlinear) {
                // the machines' stator rows are stamped right here (D19)
                return nonlinear_is_stamped ||
                       (pool.is_registered(br.id) &&
                        (pool.kind_of(br.id) == SK::PmsmMna ||
                         pool.kind_of(br.id) == SK::InductionMachineMna));
            }
            return detail::conducts_at_dc(pool, br);
        }, detail::machine_dc_ties(pool));
    if (!dc_floating.empty() && !dc_floating.front().empty()) {
        out.present = true;
        out.row = dc_floating.front().front();
        out.detail = std::format(
            " — {} is in a {}-node subnet with no DC path to ground: "
            "it is connected only through devices that are open at "
            "DC (capacitors, ideal current sources), so the DC "
            "system is rank-deficient there. Its columns are all "
            "populated by the subnet's own devices, so no emptiness "
            "probe can see this, and a conductance floor would "
            "silently supply the missing rank. Run the topology "
            "preflight (Python: it is on by default) or add a "
            "bleeder resistance to ground",
            node_label(graph, out.row), dc_floating.front().size());
        return out;
    }

    return out;
}

/// A loop of branches whose DC voltage is fixed — ideal voltage sources
/// (DC, PWM, sine and pulse, at `t_eval`), inductors, coupled windings
/// and saturable inductors (all shorts at DC), and ideal-transformer
/// windings once the other winding's voltage is fixed — whose voltages
/// do not sum to zero has no finite DC solution: KVL around it is
/// violated, and the solve returned the EMF over the inductors' 1e-12 Ω
/// regularisation (48 V across two series inductors: 2.4e13 A,
/// ll16_flyback_closed_loop, 2026-11-13). Found structurally: node
/// potentials are propagated over the fixed-voltage branches (a
/// weighted union-find); a branch whose two ends already have a fixed
/// difference that disagrees with its own closes such a loop. A loop
/// whose EMF is zero (a 0 V source at t_eval, inductors in parallel) has
/// finite solutions and is not flagged. Controlled sources (VCVS, CCVS,
/// gyrator) do not fix a voltage by themselves and are not followed.
/// With `mask`, an ideal (branch-form, R_on = 0) switch that is ON in it
/// fixes its voltage too (0 V): an ideal closed switch is a short at DC
/// (add-branch-form-ideal-switch).
struct DCVoltageLoop {
    bool present = false;
    Real emf = Real{0};                 //!< the KVL mismatch around it [V]
    std::vector<Index> branches;        //!< the loop, closing branch last
};

[[nodiscard]] inline DCVoltageLoop find_dc_voltage_loop(
    const topology::Graph& graph, const DevicePool& pool,
    Real t_eval = Real{0},
    const topology::SwitchStateMask* mask = nullptr) {
    using SK = DevicePool::StoredKind;
    const Index nn = graph.num_nodes();
    const auto slot = [nn](Index n) { return n == kGround ? nn : n; };
    std::vector<Index> parent(static_cast<Size>(nn) + 1);
    std::vector<Real> pot(static_cast<Size>(nn) + 1, Real{0});  // v - v_root
    for (Index i = 0; i <= nn; ++i) parent[static_cast<Size>(i)] = i;
    const auto find = [&](Index x) {
        // Two passes: locate the root, then compress with potentials.
        Index r = x;
        Real acc = Real{0};
        while (parent[static_cast<Size>(r)] != r) {
            acc += pot[static_cast<Size>(r)];
            r = parent[static_cast<Size>(r)];
        }
        while (parent[static_cast<Size>(x)] != r) {
            const Index next = parent[static_cast<Size>(x)];
            const Real p = pot[static_cast<Size>(x)];
            parent[static_cast<Size>(x)] = r;
            pot[static_cast<Size>(x)] = acc;
            acc -= p;
            x = next;
        }
        return r;
    };
    // Spanning-forest adjacency, to name the loop.
    std::vector<std::vector<std::pair<Index, Index>>> adj(
        static_cast<Size>(nn) + 1);
    // Branches on the spanning-forest path a -> b (slots), appended.
    const auto tree_path = [&](Index a, Index b, std::vector<Index>& path) {
        std::vector<Index> prev_node(static_cast<Size>(nn) + 1, Index{-2});
        std::vector<Index> prev_br(static_cast<Size>(nn) + 1, Index{-1});
        std::vector<Index> queue{a};
        prev_node[static_cast<Size>(a)] = a;
        for (Size q = 0; q < queue.size(); ++q) {
            const Index u = queue[q];
            if (u == b) break;
            for (const auto& [v, e] : adj[static_cast<Size>(u)]) {
                if (prev_node[static_cast<Size>(v)] == Index{-2}) {
                    prev_node[static_cast<Size>(v)] = u;
                    prev_br[static_cast<Size>(v)] = e;
                    queue.push_back(v);
                }
            }
        }
        for (Index u = b; u != a && prev_node[static_cast<Size>(u)] != Index{-2};
             u = prev_node[static_cast<Size>(u)]) {
            path.push_back(prev_br[static_cast<Size>(u)]);
        }
    };
    DCVoltageLoop out;
    // v_a - v_b = d, through branch `br`. Returns false on a loop.
    const auto add = [&](Index a, Index b, Real d, Index br) {
        a = slot(a);
        b = slot(b);
        const Index ra = find(a), rb = find(b);
        if (ra != rb) {
            // v_ra = v_a - pot_a; v_rb = v_b - pot_b; v_ra - v_rb =
            // d - pot_a + pot_b.
            parent[static_cast<Size>(ra)] = rb;
            pot[static_cast<Size>(ra)] =
                d - pot[static_cast<Size>(a)] + pot[static_cast<Size>(b)];
            adj[static_cast<Size>(a)].push_back({b, br});
            adj[static_cast<Size>(b)].push_back({a, br});
            return true;
        }
        const Real have = pot[static_cast<Size>(a)] - pot[static_cast<Size>(b)];
        const Real mis = d - have;
        if (std::abs(mis) <= Real{1e-9} * (Real{1} + std::abs(d)
                                            + std::abs(have))) {
            return true;
        }
        // Name the loop: the tree path a -> b, then the closing branch.
        tree_path(a, b, out.branches);
        out.branches.push_back(br);
        out.present = true;
        out.emf = mis;
        return false;
    };
    struct Xfmr { Index br; Index pf, pt; Real n; bool p_done, s_done; };
    std::vector<Xfmr> xfmrs;
    Size sw_idx = 0;    // mask bit of a Switch-kind branch, as dc_assemble counts
    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        const auto& br = graph.branch(b_id);
        const bool sw_bit = br.kind == topology::BranchKind::Switch
            && mask != nullptr && sw_idx < mask->size() && mask->get(sw_idx);
        if (br.kind == topology::BranchKind::Switch) ++sw_idx;
        if (!pool.is_registered(b_id)) continue;
        // switches only: a diode's DC bit is a trial of the diode iteration,
        // which turns a reverse-biased ideal diode in such a loop OFF
        if (sw_bit && pool.kind_of(b_id) == SK::Switch && pool.is_branch_form(b_id)
            && pool.branch_R_on(b_id) == Real{0}) {
            if (!add(br.from, br.to, Real{0}, b_id)) return out;
            continue;
        }
        Real d = Real{0};
        bool fixed = true;
        switch (pool.kind_of(b_id)) {
        case SK::Inductor:
        case SK::SaturableInductor:
            break;
        case SK::VoltageSource:
            d = pool.voltage_source_params(b_id).V;
            break;
        case SK::PWMVoltageSource:
            d = models::PWMVoltageSource::value_at(
                pool.pwm_voltage_source_params(b_id), t_eval);
            break;
        case SK::SineVoltageSource:
            d = models::SineVoltageSource::value_at(
                pool.sine_voltage_source_params(b_id), t_eval);
            break;
        case SK::PulseVoltageSource:
            d = models::PulseVoltageSource::value_at(
                pool.pulse_voltage_source_params(b_id), t_eval);
            break;
        case SK::IdealTransformer: {
            const auto [pf, pt] = pool.ideal_transformer_primary_nodes(b_id);
            xfmrs.push_back({b_id, pf, pt,
                             pool.ideal_transformer_params(b_id).n,
                             false, false});
            fixed = false;
            break;
        }
        case SK::TransmissionLine:   // a wire at DC: port b = port a, 1:1
            if (pool.is_line_port_b(b_id)) {
                const auto& pa = graph.branch(pool.line_partner(b_id));
                xfmrs.push_back({b_id, pa.from, pa.to, Real{1}, false, false});
            }
            fixed = false;
            break;
        default:
            fixed = false;
            break;
        }
        if (fixed && !add(br.from, br.to, d, b_id)) return out;
    }
    // v_s = n v_p: a winding whose two ends are already tied by fixed
    // voltages fixes the other winding (a shorted secondary is a shorted
    // primary). Repeat until nothing new is fixed. A loop through a
    // transformer also names the path that fixed its other winding.
    std::vector<std::pair<Index, std::pair<Index, Index>>> fixed_by;
    const auto finish = [&]() {
        for (Size k = 0; k < out.branches.size(); ++k) {
            for (const auto& [xb, ends] : fixed_by) {
                if (xb != out.branches[k]) continue;
                std::vector<Index> extra;
                tree_path(ends.first, ends.second, extra);
                for (const Index e : extra) {
                    if (std::find(out.branches.begin(), out.branches.end(), e)
                        == out.branches.end()) {
                        out.branches.push_back(e);
                    }
                }
            }
        }
        return out;
    };
    for (bool grew = true; grew;) {
        grew = false;
        for (auto& x : xfmrs) {
            const auto& br = graph.branch(x.br);
            const Index sf = slot(br.from), st = slot(br.to);
            const Index pf = slot(x.pf), pt = slot(x.pt);
            if (!x.p_done && find(sf) == find(st)) {
                const Real vs = pot[static_cast<Size>(sf)]
                                - pot[static_cast<Size>(st)];
                x.p_done = x.s_done = true;
                grew = true;
                fixed_by.push_back({x.br, {sf, st}});
                if (!add(x.pf, x.pt, vs / x.n, x.br)) return finish();
            } else if (!x.s_done && find(pf) == find(pt)) {
                const Real vp = pot[static_cast<Size>(pf)]
                                - pot[static_cast<Size>(pt)];
                x.p_done = x.s_done = true;
                grew = true;
                fixed_by.push_back({x.br, {pf, pt}});
                if (!add(br.from, br.to, x.n * vp, x.br)) return finish();
            }
        }
    }
    return out;
}

/// Refuse a DC operating point that `find_dc_voltage_loop` shows has no
/// finite solution, naming the loop's devices and what to do instead.
inline void require_no_dc_voltage_loop(
    const topology::Graph& graph, const DevicePool& pool, Real t_eval,
    const std::string& engine,
    const topology::SwitchStateMask* mask = nullptr) {
    const auto loop = find_dc_voltage_loop(graph, pool, t_eval, mask);
    if (!loop.present) return;
    std::string names;
    Index device = loop.branches.front();
    bool have_source = false;
    Index ideal = kInvalidIndex;    // an ideal closed switch in the loop
    for (const Index b : loop.branches) {
        if (!names.empty()) names += ", ";
        names += std::string{graph.branch_name(b)};
        const auto k = pool.kind_of(b);
        if (ideal == kInvalidIndex && pool.is_branch_form(b)) ideal = b;
        if (!have_source && k != DevicePool::StoredKind::Inductor
            && k != DevicePool::StoredKind::SaturableInductor) {
            device = b;
            have_source = true;
        }
    }
    if (ideal != kInvalidIndex) {
        throw UnsupportedError(
            engine, "ideal_voltage_loop", std::string{graph.branch_name(ideal)},
            std::string{stored_kind_name(pool.kind_of(ideal))},
            std::format(
                "no DC operating point: {} form a loop of voltage sources, "
                "inductors (shorts at DC) and ideal switches closed in the "
                "starting mask (R_on = 0: shorts) whose voltages do not sum "
                "to zero ({:.6g} V around it), so its DC current is "
                "infinite. Start the transient from zero state or from "
                "explicit initial conditions instead of the DC operating "
                "point, or give the switch R_on > 0.",
                names, loop.emf));
    }
    throw UnsupportedError(
        engine, "dc_operating_point", std::string{graph.branch_name(device)},
        std::string{stored_kind_name(pool.kind_of(device))},
        std::format(
            "no DC operating point: {} form a loop of voltage sources and "
            "inductors (an inductor is a short at DC) whose voltages do "
            "not sum to zero ({:.6g} V around it), so its DC current is "
            "infinite. Start the transient from zero state or from "
            "explicit initial conditions instead of the DC operating "
            "point, or add the winding resistance in series.",
            names, loop.emf));
}

/// A DC "solution" that is not finite, or exceeds 1e10 times the
/// largest independent source (volts or amperes, at least 1), is not a
/// state: no physical circuit of the sizes this simulator takes produces
/// it, and a structural defect the probes above missed does. Refused,
/// naming the unknown.
inline void require_sane_dc_state(const topology::Graph& graph,
                                  const DevicePool& pool, const Vector& x,
                                  Real t_eval, const char* who) {
    using SK = DevicePool::StoredKind;
    Real scale = Real{1};
    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        if (!pool.is_registered(b_id)) continue;
        Real v = Real{0};
        switch (pool.kind_of(b_id)) {
        case SK::VoltageSource: v = pool.voltage_source_params(b_id).V; break;
        case SK::CurrentSource: v = pool.current_source_params(b_id).I; break;
        case SK::PWMVoltageSource:
            v = models::PWMVoltageSource::value_at(
                pool.pwm_voltage_source_params(b_id), t_eval);
            break;
        case SK::SineVoltageSource:
            v = models::SineVoltageSource::value_at(
                pool.sine_voltage_source_params(b_id), t_eval);
            break;
        case SK::PulseVoltageSource:
            v = models::PulseVoltageSource::value_at(
                pool.pulse_voltage_source_params(b_id), t_eval);
            break;
        default: break;
        }
        scale = std::max(scale, std::abs(v));
    }
    Index worst = kInvalidIndex;
    Real worst_v = Real{0};
    for (Index k = 0; k < x.size(); ++k) {
        if (!std::isfinite(x[k])) { worst = k; worst_v = x[k]; break; }
        if (std::abs(x[k]) > std::abs(worst_v)) { worst = k; worst_v = x[k]; }
    }
    if (worst == kInvalidIndex
        || (std::isfinite(worst_v) && std::abs(worst_v) <= Real{1e10} * scale)) {
        return;
    }
    throw std::runtime_error(std::format(
        "{}: no DC operating point — the solve returned {:.4g} for {}, "
        "against a largest source of {:.4g}: not a state. The DC system "
        "is ill-posed (a loop or cut set the structural checks did not "
        "name); start the transient from zero state or from explicit "
        "initial conditions instead.",
        who, worst_v, row_label(graph, pool, worst), scale));
}

/// Solve the DC operating point of the LINEAR part of the circuit.
///
/// `gmin` is the conductance floor (S) stamped from every non-ground
/// node to ground before factorization — see `gmin.hpp`. It is on by
/// default because a DC matrix is where near-open devices (every
/// diode in a bridge reverse-biased, a MOSFET below threshold) leave
/// pivots that are technically nonzero and numerically worthless.
/// Pass 0 to reproduce the un-augmented system exactly.
///
/// NOTE the floor is never allowed to substitute for topology: the
/// structural probe below runs on the UN-augmented matrix, so a node
/// that genuinely has no equation still produces the named Phase-1
/// error rather than a confident 0 V.
inline Vector compute_dc_op(const topology::Graph& graph,
                              const DevicePool& pool,
                              const topology::SwitchStateMask& mask,
                              Real t_eval = Real{0},
                              Real gmin = kDefaultGmin) {
    require_no_dc_voltage_loop(graph, pool, t_eval, "dc", &mask);
    sparse::Matrix J;
    Vector b;
    dc_assemble(graph, pool, mask, J, b, t_eval);
    sparse::compress_in_place(J);

    if (gmin > Real{0}) {
        // The floor may CONDITION a matrix; it may not INVENT
        // equations for one. Check first and let the diagnostic win.
        const auto defect =
            dc_structural_defect(graph, pool, mask, {}, t_eval);
        if (defect.present) {
            throw std::runtime_error(std::format(
                "compute_dc_op: DC matrix structurally singular for "
                "mask {}{}",
                mask.to_string(), defect.detail));
        }
        stamp_gmin_dc(J, graph.num_nodes(), gmin, charge_rows(graph, pool));
        // the held rows are equations v = value, not KCL: no floor on them
        if (!pool.pmsm_mna_machines().empty() ||
            !pool.induction_machines().empty()) {
            detail::hold_machine_mechanics(graph, pool, J, b);
        }
        sparse::compress_in_place(J);
    }

    auto solver = sparse::make_default_solver();
    if (!solver->analyze(J)) {
        throw std::runtime_error(std::format(
            "compute_dc_op: DC matrix structurally singular for "
            "mask {}{}",
            mask.to_string(),
            explain_singular(graph, pool, J, solver.get())));
    }
    if (!solver->factorize(J)) {
        // v2.0 Phase 1 (audit finding
        // `singular-errors-dont-name-the-node`): say WHERE, not just
        // that it failed. Capacitors are open at DC, so a node hung
        // off nothing but a capacitor produces a genuinely empty
        // column here — by far the most common cause, and now named.
        throw std::runtime_error(std::format(
            "compute_dc_op: DC matrix numerically singular for "
            "mask {}{}",
            mask.to_string(),
            explain_singular(graph, pool, J, solver.get())));
    }

    Vector x;
    Vector rhs = -b;   // J · x = -b solves f = J·x + b = 0
    solver->solve(rhs, x);
    require_sane_dc_state(graph, pool, x, t_eval, "compute_dc_op");
    return x;
}

/// Phase-0 fix #2 — Newton DC operating point.
///
/// `compute_dc_op` stamps `BranchKind::Nonlinear` devices as OPEN
/// CIRCUITS (see the case label above), so for any circuit with a
/// smooth diode / MOSFET L1 / IGBT L1 it converges — with no
/// warning — to the operating point of a DIFFERENT circuit (audit
/// finding dc-op-skips-nonlinear-devices, CONFIRMED). This variant
/// runs the SAME Newton machinery the transient uses
/// (`solve_with_newton_b_extra` + the caller's NonlinearRefreshFn
/// chain) on the DC-assembled system:
///
///   * linear part: the dc_assemble matrix (caps open, inductors
///     near-short) — identical to `compute_dc_op`;
///   * nonlinear part: re-stamped per Newton iterate by `refresh`,
///     exactly as in the transient inner solve;
///   * warm start: the linear DC solution (nonlinear-open), which
///     is the natural continuation seed.
///
/// For a circuit with no Newton devices the refresh stamps nothing
/// and this returns the `compute_dc_op` answer after one iteration.
inline Vector compute_dc_op_newton(
    const topology::Graph& graph,
    const DevicePool& pool,
    const topology::SwitchStateMask& mask,
    const NonlinearRefreshFn& refresh,
    Real t_eval = Real{0},
    Size max_iters = Size{50},
    Real tol_dx  = Real{1e-9},
    Real tol_res = Real{1e-9},
    bool enable_line_search = false,
    bool enable_lm = false,
    Real gmin = kDefaultGmin) {
    if (!refresh) {
        return compute_dc_op(graph, pool, mask, t_eval, gmin);
    }
    require_no_dc_voltage_loop(graph, pool, t_eval, "dc", &mask);

    // Structural check against the system Newton actually solves.
    // Doing it here rather than inheriting `compute_dc_op`'s
    // linear-only probe is the difference between naming a genuinely
    // floating node and libelling every interior node of a diode
    // chain (whose LINEAR column is empty by construction).
    {
        const auto defect =
            dc_structural_defect(graph, pool, mask, refresh, t_eval);
        if (defect.present) {
            throw std::runtime_error(std::format(
                "compute_dc_op_newton: DC system structurally "
                "singular for mask {}{}",
                mask.to_string(), defect.detail));
        }
    }

    // Warm start from the nonlinear-open linear solve. That system
    // can be singular on its own merits — nodes reachable only
    // through the nonlinear devices we just proved DO stamp them —
    // in which case a cold start is the honest fallback.
    Vector x0;
    try {
        x0 = compute_dc_op(graph, pool, mask, t_eval, gmin);
    } catch (const std::exception&) {
        x0 = Vector::Zero(
            static_cast<Index>(pool.state_size(graph)));
    }

    // Local DC "segment": same shape solve_with_newton_b_extra
    // consumes for transient steps (it reads only J, b_constant,
    // state_size — the pre-factored solver member is unused there).
    PwlSegment seg;
    dc_assemble(graph, pool, mask, seg.J, seg.b_constant, t_eval,
                 gmin);
    sparse::compress_in_place(seg.J);
    seg.state_size = static_cast<Size>(seg.b_constant.size());

    const Vector b_extra =
        Vector::Zero(static_cast<Index>(seg.state_size));
    return solve_with_newton_b_extra(
        seg, refresh, graph, pool, x0, b_extra,
        max_iters, tol_dx, tol_res,
        enable_line_search, enable_lm);
}

}  // namespace pulsim::pwl
