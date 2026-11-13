#pragma once

// =============================================================================
// Pulsim — Layer 4: assemble_segment (build one per-state matrix)
// =============================================================================
//
// `pulsim-v2-pwl-state-space-cache` Phase 3 (Layer 4 V0).
// `pulsim-v2-trapezoidal-companion` Phase 4 (Layer 4 V1 — dt-aware).
// v2.0 Phase 1: (G, C, b) split assembly.
//
// For one `SwitchStateMask`, build the MNA matrix `J` + constant
// RHS `b` by stamping every branch in the graph using Layer 3's
// stampers + Layer 2's device-model parameters from DevicePool.
//
// V1 scope: PassiveLinear (Resistor / Capacitor / Inductor),
// Source, and Switch branches. Nonlinear branches are SKIPPED.
//
// dt convention:
//   * dt == 0  → static-only build (V0 backwards compat). Cap /
//                Inductor branches are SKIPPED.
//   * dt  > 0  → V1 dynamic build. Caps / Inductors stamp their
//                trap-companion g_eq into J. History contributions
//                are NOT stamped here — Layer 5 adds them via
//                b_extra at solve time.
//
// (G, C, b) split (v2.0 Phase 1, audit finding A.5): every
// dt-dependent trap-companion entry in this kernel is EXACTLY
// linear in 1/dt —
//   * capacitor block      ±(2·C_cap)/dt
//   * inductor constraint  −(2·L)/dt on the branch-var diagonal
//   * transformer cross    −(2·M)/dt between winding branch rows
// — so the full matrix decomposes as
//       J(mask, dt) = G(mask) + (1/dt) · C
// with `G` holding every dt-independent stamp (including the
// inductor's ±1 incidence terms, which the dt-monolithic V1 path
// only emitted when dt > 0) and `C` holding the 1/dt COEFFICIENTS
// (2·C_cap, −2·L, −2·M). `C` does not depend on the switch mask.
// `b` receives source stamps and the per-mask SwitchedDiode forward
// drop (−g_on·V_f while ON), and is dt-independent.
//
// `assemble_segment_split` exposes the split directly (consumed by
// the cache's event-dt solver reuse and the EXACT continuous-LTI
// extraction); the dt-aware `assemble_segment` recombines
// J = G + (1/dt)·C so every caller shares one stamping loop.

#include "pulsim/models/resistor.hpp"
#include "pulsim/models/transformer.hpp"
#include "pulsim/models/voltage_source.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/stamping/stamp_companion.hpp"
#include "pulsim/stamping/stamp_current_source.hpp"
#include "pulsim/stamping/stamp_device.hpp"
#include "pulsim/stamping/stamp_switch.hpp"
#include "pulsim/stamping/stamp_ideal_transformer.hpp"
#include "pulsim/stamping/stamp_vcvs.hpp"
#include "pulsim/stamping/stamp_voltage_source.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace pulsim::pwl {

namespace detail {

// The topological `Switch` kind covers both user-controlled
// `IdealSwitch` and auto-commutating `SwitchedDiode`; the pool's
// StoredKind picks (g_on, g_off) and the diode's forward drop. Shared
// by the transient and DC stampers so the two cannot drift. Never
// source-scaled: V_f is a device law, not an independent source.
inline void stamp_switch_branch(const DevicePool& pool, Index branch_id,
                                bool closed, sparse::Matrix& J, Vector& b,
                                const Vector& x,
                                const stamping::BranchCoord& coord) {
    if (pool.kind_of(branch_id) == DevicePool::StoredKind::Diode) {
        const auto& p = pool.diode_params(branch_id);
        stamping::stamp_switch_fixed(J, b, x, coord, closed, p.g_on,
                                     p.g_off, p.V_f);
    } else {
        stamping::stamp_switch_fixed(J, b, x, coord, closed,
                                     pool.switch_g_on(branch_id),
                                     pool.switch_g_off(branch_id));
    }
}

// Per-column reserve from the node incidence (update-linear-time-assembly
// tier 1). An estimate: a column that overflows is grown by Eigen, so
// correctness never depends on it; it only removes the O(nnz) shift of
// every insert into an unreserved matrix.
inline void reserve_by_incidence(const topology::Graph& graph,
                                 sparse::Matrix& M) {
    const Index n = M.cols();
    Eigen::VectorXi per_col = Eigen::VectorXi::Constant(n, 4);
    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        const auto& br = graph.branch(b_id);
        if (br.from >= 0 && br.from < n) per_col[br.from] += 2;
        if (br.to >= 0 && br.to < n) per_col[br.to] += 2;
    }
    M.reserve(per_col);
}

// -----------------------------------------------------------------------------
// assemble_impl — the ONE stamping loop behind both public entry
// points.
//
// `S` is the static-stamp target (the caller's G, or the legacy
// static-only J). `C_out` selects the mode:
//   * C_out != nullptr → full split: static stamps land in S,
//     1/dt-coefficient stamps land in *C_out, and the inductor's
//     ±1 incidence terms land in S (they are dt-independent).
//   * C_out == nullptr → legacy V0 static-only assembly: Cap /
//     Inductor / transformer-coupling stamps are SKIPPED ENTIRELY
//     (including the inductor incidence — at dt = 0 the inductor
//     branch is absent from the system, matching V0 behaviour).
//
// V1 still uses `x = Vector::Zero(state_size)` during stamping —
// every device class supported here is linear, so the stamp
// doesn't depend on the operating point. The voltage source's
// `-V` contribution to `b` is captured because
// `stamp_voltage_source` reads V from its parameter, not from `x`.
// Capacitor / Inductor companions contribute 0 to b at assembly;
// the history term lives in `b_extra` at solve time.
// -----------------------------------------------------------------------------
inline void assemble_impl(const topology::Graph& graph,
                           const DevicePool& pool,
                           const topology::SwitchStateMask& mask,
                           sparse::Matrix& S,
                           sparse::Matrix* C_out,
                           Vector& b) {
    const Size state_size = pool.state_size(graph);
    if (state_size == 0) {
        return;
    }

    // Zero state vector: V1 stamping is linear, so the result
    // doesn't depend on x.
    const Vector x = Vector::Zero(static_cast<Index>(state_size));

    // Counter advances only on Switch-kind branches; matches the
    // SwitchStateMask bit-i = i-th Switch convention.
    Size switch_idx = 0;

    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        const auto& branch = graph.branch(b_id);
        const stamping::BranchCoord coord{branch.from, branch.to,
                                            branch.id};

        switch (branch.kind) {
        case topology::BranchKind::PassiveLinear: {
            // V1 dispatch: PassiveLinear may be Resistor, Capacitor,
            // or Inductor. Use the pool's StoredKind to discriminate.
            const auto k = pool.kind_of(branch.id);
            switch (k) {
            case DevicePool::StoredKind::Resistor: {
                const auto& p = pool.resistor_params(branch.id);
                stamping::stamp_device<models::Resistor>(S, b, x,
                                                          coord, p);
                break;
            }
            case DevicePool::StoredKind::Capacitor: {
                if (C_out != nullptr) {
                    const auto& p = pool.capacitor_params(branch.id);
                    // Coefficient of 1/dt: g_eq(dt) = 2C/dt →
                    // stamp the block with 2C into C_out.
                    stamping::stamp_capacitor_companion(
                        *C_out, b, coord, Real{2} * p.C);
                }
                // Static-only: V0 backwards-compat path — skip.
                break;
            }
            case DevicePool::StoredKind::Inductor: {
                if (C_out != nullptr) {
                    const auto& p = pool.inductor_params(branch.id);
                    const Index branch_var_id =
                        pool.branch_var_id_for_inductor(branch.id, graph);
                    // Incidence (±1) → S; diagonal coefficient
                    // (−2L) → C_out.
                    stamping::stamp_inductor_companion_split(
                        S, *C_out, coord, branch_var_id,
                        Real{2} * p.L);
                }
                break;
            }
            default:
                // Unexpected PassiveLinear kind — should be
                // unreachable. Silently skip.
                break;
            }
            break;
        }
        case topology::BranchKind::Source: {
            // The topological `Source` kind covers
            // VoltageSource, CurrentSource, AND
            // PWMVoltageSource (a time-varying voltage
            // source). Dispatch on the pool's StoredKind.
            const auto src_kind = pool.kind_of(branch.id);
            if (src_kind == DevicePool::StoredKind::CurrentSource) {
                const auto& p =
                    pool.current_source_params(branch.id);
                stamping::stamp_current_source(b, coord, p.I);
            } else if (src_kind ==
                       DevicePool::StoredKind::PWMVoltageSource) {
                // Structurally a VoltageSource with V=0
                // baseline. The time-varying PWM value is
                // overlaid via b_extra at runtime by
                // run_transient's PWM pass.
                const Index branch_var_id =
                    pool.branch_var_id_for_source(branch.id, graph);
                stamping::stamp_voltage_source(S, b, x, coord,
                                                branch_var_id, Real{0});
            } else if (src_kind ==
                       DevicePool::StoredKind::SineVoltageSource) {
                // V11: same V=0 baseline pattern; sine
                // value comes from b_extra each step.
                const Index branch_var_id =
                    pool.branch_var_id_for_source(branch.id, graph);
                stamping::stamp_voltage_source(S, b, x, coord,
                                                branch_var_id, Real{0});
            } else if (src_kind ==
                       DevicePool::StoredKind::PulseVoltageSource) {
                // V12: same V=0 baseline + b_extra overlay.
                const Index branch_var_id =
                    pool.branch_var_id_for_source(branch.id, graph);
                stamping::stamp_voltage_source(S, b, x, coord,
                                                branch_var_id, Real{0});
            } else if (src_kind ==
                       DevicePool::StoredKind::VCVS) {
                // V15: voltage-controlled voltage source.
                // Constraint row references the sense inputs.
                const auto& vp = pool.vcvs_params(branch.id);
                const auto [in_pos, in_neg] =
                    pool.vcvs_input_nodes(branch.id);
                const Index branch_var_id =
                    pool.branch_var_id_for_source(branch.id, graph);
                stamping::stamp_vcvs(S, b, x, coord,
                                       in_pos, in_neg,
                                       branch_var_id, vp.gain);
            } else if (src_kind ==
                       DevicePool::StoredKind::IdealTransformer) {
                // Phase 4 C.4: VCVS on the secondary plus the
                // reflected −n·i_s at the primary nodes.
                const auto& tp =
                    pool.ideal_transformer_params(branch.id);
                const auto [p_from, p_to] =
                    pool.ideal_transformer_primary_nodes(branch.id);
                const Index branch_var_id =
                    pool.branch_var_id_for_source(branch.id, graph);
                stamping::stamp_ideal_transformer(
                    S, b, x, coord, p_from, p_to, branch_var_id, tp.n);
            } else {
                const auto& p = pool.voltage_source_params(branch.id);
                const Index branch_var_id =
                    pool.branch_var_id_for_source(branch.id, graph);
                stamping::stamp_voltage_source(S, b, x, coord,
                                                branch_var_id, p.V);
            }
            break;
        }
        case topology::BranchKind::Switch: {
            stamp_switch_branch(pool, branch.id, mask.get(switch_idx),
                                S, b, x, coord);
            ++switch_idx;
            break;
        }
        case topology::BranchKind::Nonlinear: {
            // V17: Saturable inductors live as Nonlinear
            // branches but use the inductor branch-var
            // numbering. Their constraint row would be
            // ALL-ZERO in the linear assembly (the Newton
            // refresh stamps everything per-iteration),
            // which makes KLU's factorisation fail. Add a
            // tiny "G_min" diagonal here so the cache builds;
            // the Newton refresh's stamping dwarfs G_min in
            // the combined Jacobian.
            const auto src_kind = pool.kind_of(branch.id);
            if (src_kind ==
                    DevicePool::StoredKind::SaturableInductor) {
                const Index branch_var_id =
                    pool.branch_var_id_for_inductor(
                        branch.id, graph);
                S.coeffRef(branch_var_id, branch_var_id) +=
                    Real{1e-12};
            } else if (src_kind ==
                    DevicePool::StoredKind::NonlinearCapacitor) {
                // Phase 4 C.1 — same reasoning, node rows: a Coss
                // has no branch-current unknown and no linear
                // contribution at all, so without a G_min its
                // nodes can be structurally empty and the
                // factorisation fails before Newton ever stamps.
                if (stamping::node_is_active(branch.from)) {
                    S.coeffRef(branch.from, branch.from) +=
                        Real{1e-12};
                }
                if (stamping::node_is_active(branch.to)) {
                    S.coeffRef(branch.to, branch.to) +=
                        Real{1e-12};
                }
            }
            break;
        }
        }
    }

    // ----- Layer 2 V2: transformer cross-coupling pass ---------------------
    //
    // For each registered (primary, secondary) inductor pair,
    // add the mutual-inductance cross-terms:
    //   J[p_row, s_col] += -(2M/dt)  →  C[p_row, s_col] += -2M
    //   J[s_row, p_col] += -(2M/dt)  →  C[s_row, p_col] += -2M
    //
    // This runs AFTER per-branch stamping so each inductor's
    // self-inductance diagonal is already in place. Skipped in
    // static-only mode (couplings have no static contribution;
    // transformers behave as open circuits at DC, matching the
    // inductor's static path).
    if (C_out != nullptr) {
        for (const auto& tc : pool.transformer_couplings()) {
            const Index p_row =
                pool.branch_var_id_for_inductor(
                    tc.primary_branch_id, graph);
            const Index s_row =
                pool.branch_var_id_for_inductor(
                    tc.secondary_branch_id, graph);
            const Real cross =
                Real{2} * models::TwoWindingTransformer::
                              mutual_inductance(tc.params);
            C_out->coeffRef(p_row, s_row) += -cross;
            C_out->coeffRef(s_row, p_row) += -cross;
        }
    }
}

}  // namespace detail

// -----------------------------------------------------------------------------
// assemble_segment_split — build (G, C, b) for one switch state.
//
// `G` holds every dt-independent stamp; `C` holds the 1/dt
// coefficients, so J(dt) = G + (1/dt)·C for ANY dt > 0. Both are
// returned COMPRESSED. `C` is mask-invariant (switches only touch
// G) — callers may cache it across masks if the pool's L/C/M
// parameters have not changed.
// -----------------------------------------------------------------------------
inline void assemble_segment_split(const topology::Graph& graph,
                                    const DevicePool& pool,
                                    const topology::SwitchStateMask& mask,
                                    sparse::Matrix& G,
                                    sparse::Matrix& C,
                                    Vector& b) {
    const auto n = static_cast<Index>(pool.state_size(graph));
    G = sparse::Matrix(n, n);
    C = sparse::Matrix(n, n);
    detail::reserve_by_incidence(graph, G);
    detail::reserve_by_incidence(graph, C);
    b = Vector::Zero(n);
    detail::assemble_impl(graph, pool, mask, G, &C, b);
    G.makeCompressed();
    C.makeCompressed();
}

// -----------------------------------------------------------------------------
// assemble_segment — build (J, b) for one switch state.
//
// dt > 0 recombines the split: J = G + (1/dt)·C (single stamping
// loop shared with assemble_segment_split — no drift between the
// monolithic and split assemblies). dt == 0 is the V0 static-only
// path: Caps and Inductors are skipped entirely.
// -----------------------------------------------------------------------------
inline void assemble_segment(const topology::Graph& graph,
                              const DevicePool& pool,
                              const topology::SwitchStateMask& mask,
                              Real dt,
                              sparse::Matrix& J,
                              Vector& b) {
    const auto n = static_cast<Index>(pool.state_size(graph));
    if (dt > Real{0}) {
        sparse::Matrix G, C;
        assemble_segment_split(graph, pool, mask, G, C, b);
        J = G + (Real{1} / dt) * C;
        return;
    }
    J = sparse::Matrix(n, n);
    detail::reserve_by_incidence(graph, J);
    b = Vector::Zero(n);
    detail::assemble_impl(graph, pool, mask, J, nullptr, b);
}

// -----------------------------------------------------------------------------
// V0 backwards-compat overload — static-only assembly (dt = 0).
//
// Caps and Inductors are silently SKIPPED, matching V0 behaviour.
// New callers that have dynamic devices SHOULD use the dt-aware
// overload.
// -----------------------------------------------------------------------------
inline void assemble_segment(const topology::Graph& graph,
                              const DevicePool& pool,
                              const topology::SwitchStateMask& mask,
                              sparse::Matrix& J,
                              Vector& b) {
    assemble_segment(graph, pool, mask, Real{0}, J, b);
}

// -----------------------------------------------------------------------------
// SegmentAssembler — stamp-slot map (update-linear-time-assembly tier 1).
//
// Stamp positions depend on the topology only (a switch stamps the same
// four entries with g_on or g_off), so the first assembly of a regime
// (dynamic dt > 0: G and C; static dt = 0: J) becomes a pattern template.
// Later assemblies copy the template, zero its values and stamp in the
// same order: `coeffRef` finds an existing entry and `+0.0 + v` is the
// sum a fresh entry gets, so values are bit-identical to stamping into
// an empty matrix. `J = G + (1/dt)·C` is written through two slot arrays
// onto the union pattern with absent operands as +0.0 — the per-entry
// arithmetic of Eigen's sparse sum (`a + b`, `a + 0`, `0 + b`).
//
// A stamp outside the template inserts and changes the entry count; the
// assembler then rebuilds the template from a fresh assembly
// (`structure_builds` + 1). Not thread-safe: callers use it on the
// paths that already mutate the cache.
// -----------------------------------------------------------------------------
class SegmentAssembler {
public:
    /// Same result as `assemble_segment_split`.
    void split(const topology::Graph& graph, const DevicePool& pool,
               const topology::SwitchStateMask& mask,
               sparse::Matrix& G, sparse::Matrix& C, Vector& b) {
        const auto n = static_cast<Index>(pool.state_size(graph));
        if (dyn_.ready && dyn_.G.rows() == n) {
            G = dyn_.G;
            C = dyn_.C;
            zero_values_(G);
            zero_values_(C);
            b = Vector::Zero(n);
            detail::assemble_impl(graph, pool, mask, G, &C, b);
            if (G.nonZeros() == dyn_.G.nonZeros() &&
                C.nonZeros() == dyn_.C.nonZeros()) {
                return;
            }
        }
        assemble_segment_split(graph, pool, mask, G, C, b);
        build_dynamic_(G, C);
    }

    /// Same result as `assemble_segment` followed by `compress_in_place`.
    void assemble(const topology::Graph& graph, const DevicePool& pool,
                  const topology::SwitchStateMask& mask, Real dt,
                  sparse::Matrix& J, Vector& b) {
        if (dt > Real{0}) {
            split(graph, pool, mask, G_, C_, b);
            combine(G_, C_, dt, J);
            return;
        }
        const auto n = static_cast<Index>(pool.state_size(graph));
        if (stat_ready_ && stat_J_.rows() == n) {
            J = stat_J_;
            zero_values_(J);
            b = Vector::Zero(n);
            detail::assemble_impl(graph, pool, mask, J, nullptr, b);
            if (J.nonZeros() == stat_J_.nonZeros()) {
                return;
            }
        }
        assemble_segment(graph, pool, mask, Real{0}, J, b);
        sparse::compress_in_place(J);
        stat_J_ = J;
        stat_ready_ = true;
        ++structure_builds_;
    }

    /// `J = G + (1/dt)·C` on the dynamic union pattern, bit-equal to the
    /// Eigen expression. `G` and `C` must come from `split` (template
    /// patterns); `J` keeps its storage when it already has the pattern.
    void combine(const sparse::Matrix& G, const sparse::Matrix& C,
                 Real dt, sparse::Matrix& J) {
        if (G.nonZeros() != dyn_.G.nonZeros() ||
            C.nonZeros() != dyn_.C.nonZeros() || G.rows() != dyn_.J.rows()) {
            // Operands from before a template rebuild: Eigen's sum.
            J = G + (Real{1} / dt) * C;
            return;
        }
        if (J.rows() != dyn_.J.rows() || !J.isCompressed() ||
            J.nonZeros() != dyn_.J.nonZeros()) {
            J = dyn_.J;
        }
        const auto nnz = static_cast<std::size_t>(J.nonZeros());
        Real* jv = J.valuePtr();
        cs_.assign(nnz, Real{0});
        std::fill_n(jv, nnz, Real{0});
        const Real* gv = G.valuePtr();
        for (std::size_t k = 0; k < g2j_.size(); ++k) jv[g2j_[k]] = gv[k];
        const Real* cv = C.valuePtr();
        for (std::size_t k = 0; k < c2j_.size(); ++k) cs_[c2j_[k]] = cv[k];
        const Real inv_dt = Real{1} / dt;
        for (std::size_t k = 0; k < nnz; ++k) jv[k] = jv[k] + inv_dt * cs_[k];
    }

    /// `G` and `C` (from `split`, template patterns) scattered onto the
    /// union pattern of `J`, absent entries +0.0, for `combine_scattered`.
    void scatter(const sparse::Matrix& G, const sparse::Matrix& C,
                 std::vector<Real>& gs, std::vector<Real>& cs) const {
        const auto nnz = static_cast<std::size_t>(dyn_.J.nonZeros());
        gs.assign(nnz, Real{0});
        cs.assign(nnz, Real{0});
        const Real* gv = G.valuePtr();
        for (std::size_t k = 0; k < g2j_.size(); ++k) gs[g2j_[k]] = gv[k];
        const Real* cv = C.valuePtr();
        for (std::size_t k = 0; k < c2j_.size(); ++k) cs[c2j_[k]] = cv[k];
    }

    /// In-place `J[k] = gs[k] + (1/dt)·cs[k]` on the pattern `J` already
    /// has: the arithmetic of `combine`, no allocation
    /// (update-trbdf2-factor-reuse).
    static void combine_scattered(const std::vector<Real>& gs,
                                  const std::vector<Real>& cs, Real dt,
                                  sparse::Matrix& J) noexcept {
        Real* jv = J.valuePtr();
        const Real inv_dt = Real{1} / dt;
        for (std::size_t k = 0; k < gs.size(); ++k) jv[k] = gs[k] + inv_dt * cs[k];
    }

    /// Template (sparse-structure) builds so far: one per regime unless
    /// the circuit's structure changes under the cache.
    [[nodiscard]] std::uint64_t structure_builds() const noexcept {
        return structure_builds_;
    }

private:
    static void zero_values_(sparse::Matrix& M) noexcept {
        std::fill_n(M.valuePtr(), static_cast<std::size_t>(M.nonZeros()),
                    Real{0});
    }

    // Position in the union pattern `U` of every stored entry of `A`
    // (both compressed, sorted inner indices, A ⊆ U).
    static void slots_(const sparse::Matrix& A, const sparse::Matrix& U,
                       std::vector<Index>& out) {
        out.resize(static_cast<std::size_t>(A.nonZeros()));
        const Index* ap = A.outerIndexPtr();
        const Index* ai = A.innerIndexPtr();
        const Index* up = U.outerIndexPtr();
        const Index* ui = U.innerIndexPtr();
        for (Index j = 0; j < A.outerSize(); ++j) {
            Index q = up[j];
            for (Index p = ap[j]; p < ap[j + 1]; ++p) {
                while (ui[q] != ai[p]) ++q;
                out[static_cast<std::size_t>(p)] = q;
            }
        }
    }

    void build_dynamic_(const sparse::Matrix& G, const sparse::Matrix& C) {
        dyn_.G = G;
        dyn_.C = C;
        dyn_.J = G + C;   // union pattern; values are overwritten
        dyn_.J.makeCompressed();
        slots_(dyn_.G, dyn_.J, g2j_);
        slots_(dyn_.C, dyn_.J, c2j_);
        dyn_.ready = true;
        ++structure_builds_;
    }

    struct {
        bool ready = false;
        sparse::Matrix G, C, J;
    } dyn_;
    std::vector<Index> g2j_, c2j_;
    std::vector<Real> cs_;
    bool stat_ready_ = false;
    sparse::Matrix stat_J_;
    sparse::Matrix G_, C_;   // scratch of `assemble` at dt > 0
    std::uint64_t structure_builds_ = 0;
};

}  // namespace pulsim::pwl
