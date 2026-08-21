#pragma once

// =============================================================================
// Pulsim — Layer 4: MNA row → human label
// =============================================================================
//
// v2.0 Phase 1, audit findings `kernel-has-no-name-context-for-errors`
// (structural) and `singular-errors-dont-name-the-node` (the payoff).
//
// Kernel diagnostics used to speak only in integer ids, mask
// bitstrings and norms: a 200-switch MMC user staring at
//
//     numerically singular for mask 0010111…1 (dt=1e-7)
//
// has nothing to act on. Every one of those failures, however, has a
// LOCATION — the zero pivot is a matrix row, the worst Newton
// residual is a matrix row — and a row is a physical thing: a node's
// KCL equation, or a device's branch-current unknown.
//
// This header is the one place that knows the MNA row layout
//
//     x = [ v_0 … v_{N-1} | i_src_0 … i_src_{M-1} | i_L_0 … i_L_{K-1} ]
//
// (see `DevicePool::state_size`, `branch_var_id_for_source`,
// `branch_var_id_for_inductor`) and turns a row index into something
// a user can act on:
//
//     row 7  → "node vout"
//     row 12 → "current through source Vin"
//     row 14 → "current through inductor L1"
//
// Names are best-effort: a raw-kernel Graph built without names (or
// a node whose name was never set) falls back to the id, so this is
// safe to call from ANY error path regardless of how the graph was
// constructed.

#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/sparse/solver.hpp"
#include "pulsim/topology/graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <string>
#include <vector>

#include "pulsim/numeric/dense.hpp"

namespace pulsim::pwl {

/// What kind of unknown a given MNA row represents.
enum class RowKind {
    NodeVoltage,     ///< KCL equation / voltage unknown of a node
    SourceCurrent,   ///< branch-current unknown of a voltage source
    InductorCurrent, ///< branch-current unknown of an inductor
    OutOfRange,      ///< row index outside [0, state_size)
};

struct RowInfo {
    RowKind kind = RowKind::OutOfRange;
    Index   index = kInvalidIndex;  ///< node id, or branch id, per kind
    std::string label;              ///< human-facing description
};

/// Resolve MNA row `row` against the circuit's layout.
///
/// Never throws: an unresolvable row yields `RowKind::OutOfRange`
/// with a label naming the row number, which is still strictly more
/// useful in an error message than nothing.
[[nodiscard]] inline RowInfo describe_row(const topology::Graph& graph,
                                           const DevicePool& pool,
                                           Index row) {
    const auto n_nodes  = static_cast<Index>(graph.num_nodes());
    const auto n_src    = static_cast<Index>(pool.num_voltage_sources());
    const auto n_total  = static_cast<Index>(pool.state_size(graph));

    if (row < 0 || row >= n_total) {
        return {RowKind::OutOfRange, kInvalidIndex,
                std::format("MNA row {} (out of range)", row)};
    }

    // ---- Node voltages: rows [0, N) -------------------------------
    if (row < n_nodes) {
        const auto& name = graph.node(row).name;
        return {RowKind::NodeVoltage, row,
                name.empty() ? std::format("node #{}", row)
                              : std::format("node {}", name)};
    }

    // ---- Branch currents: rows [N, N + M + K) ---------------------
    //
    // The pool stores branch_id → relative offset; we need the
    // inverse. Both maps are small (one entry per source / inductor)
    // and this only runs on an error path, so a linear scan is the
    // right trade against maintaining a second index.
    const bool is_source = row < n_nodes + n_src;

    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        Index owned_row = kInvalidIndex;
        if (is_source) {
            if (!pool.is_voltage_source(b_id)) continue;
            owned_row = pool.branch_var_id_for_source(b_id, graph);
        } else {
            if (!pool.is_inductor(b_id)) continue;
            owned_row = pool.branch_var_id_for_inductor(b_id, graph);
        }
        if (owned_row != row) continue;

        const auto name = graph.branch_name(b_id);
        const char* what = is_source ? "source" : "inductor";
        return {is_source ? RowKind::SourceCurrent
                          : RowKind::InductorCurrent,
                b_id,
                name.empty()
                    ? std::format("current through {} branch #{}",
                                   what, b_id)
                    : std::format("current through {} {}",
                                   what, std::string{name})};
    }

    // Layout says this row is a branch current, but no branch claims
    // it — a pool/graph mismatch. Report honestly rather than
    // guessing a name.
    return {RowKind::OutOfRange, kInvalidIndex,
            std::format("MNA row {} (unattributed branch current)", row)};
}

/// Convenience: just the label.
[[nodiscard]] inline std::string row_label(const topology::Graph& graph,
                                            const DevicePool& pool,
                                            Index row) {
    return describe_row(graph, pool, row).label;
}

/// Describe the branch `b_id` itself (used where the failure is
/// already attributed to a device rather than to a matrix row —
/// e.g. a chattering diode).
[[nodiscard]] inline std::string branch_label(const topology::Graph& graph,
                                               Index b_id) {
    const auto name = graph.branch_name(b_id);
    return name.empty() ? std::format("branch #{}", b_id)
                        : std::format("{} (branch #{})",
                                       std::string{name}, b_id);
}

/// A localised singularity: the offending row (for tooling) and the
/// sentence to show the user. ONE resolution feeds both, so the
/// human text and the machine-readable index can never disagree.
struct SingularDiagnosis {
    Index       row = kInvalidIndex;
    std::string text;               ///< empty when unlocalisable
};

/// Diagnose WHY a factorization failed, in the user's vocabulary.
///
/// Two independent sources of location, because one is not enough:
///
///  1. A structurally EMPTY column/row in the assembled matrix.
///     Available on EVERY backend — which matters, since the DC and
///     Newton paths use the Eigen backend, and it reports no pivot
///     index at all.
///  2. The failing pivot column from `singular_index()`, when the
///     backend has one. This catches what structure cannot: a node
///     behind an OPEN switch is not empty (`g_off` is always
///     stamped) — it just pivots to ~0.
///
/// CRITICAL — an empty row is not always a wiring fault. Some
/// unknowns are reserved by `DevicePool::state_size` but
/// deliberately NOT stamped by the current assembly mode:
///   * every inductor at `dt == 0` (the static/V0 build omits
///     companion stamps entirely), and
///   * saturable inductors in the DC assembly (no DC stamp yet).
/// Their rows are empty BY CONSTRUCTION, not because the device is
/// disconnected. Telling such a user to "add a bleeder resistor to
/// L1" sends them to debug a correctly wired part — actively worse
/// than the old, merely-uninformative message. So the phrasing
/// branches on WHAT the empty row belongs to: a node gets the
/// wiring advice, a device branch-current gets the truth about the
/// assembly mode.
[[nodiscard]] inline SingularDiagnosis diagnose_singular(
    const topology::Graph& graph,
    const DevicePool& pool,
    const sparse::Matrix& J,
    const sparse::DirectSolver* solver = nullptr) {
    auto empty_explanation = [&](Index row) -> std::string {
        const auto info = describe_row(graph, pool, row);
        if (info.kind == RowKind::NodeVoltage) {
            return std::format(
                " — nothing connects {}: its column in the MNA matrix "
                "is empty, i.e. no device ties it to the rest of the "
                "circuit (a node reachable only through a capacitor "
                "has no DC path; add a bleeder/parallel resistance or "
                "tie it to ground)",
                info.label);
        }
        // A branch-current unknown with no entries: the DEVICE was
        // not stamped into this particular system.
        return std::format(
            " — {} has no equation in this system: the device owns an "
            "MNA unknown but contributed no stamps. This is expected "
            "when a static system is built with dt = 0 (inductor "
            "companions need dt > 0) or when a saturable inductor "
            "reaches the DC assembly, which has no DC stamp for it — "
            "build with dt > 0, or seed from a transient instead of a "
            "DC operating point",
            info.label);
    };

    const Index empty_col = sparse::first_empty_column(J);
    if (empty_col != kInvalidIndex) {
        return {empty_col, empty_explanation(empty_col)};
    }
    const Index empty_row = sparse::first_empty_row(J);
    if (empty_row != kInvalidIndex) {
        return {empty_row, empty_explanation(empty_row)};
    }
    if (solver != nullptr) {
        const Index col = solver->singular_index();
        if (col != kInvalidIndex) {
            return {col, std::format(
                " — elimination collapsed at {}: the pivot there fell "
                "to ~0, so that unknown is (nearly) unconstrained "
                "— typically an isolated node behind an open switch "
                "or a very large series resistance",
                row_label(graph, pool, col))};
        }
    }
    return {};
}

/// Text-only convenience over `diagnose_singular`. Returns an empty
/// string when the failure cannot be localised, so callers can
/// append it unconditionally.
[[nodiscard]] inline std::string explain_singular(
    const topology::Graph& graph,
    const DevicePool& pool,
    const sparse::Matrix& J,
    const sparse::DirectSolver* solver = nullptr) {
    return diagnose_singular(graph, pool, J, solver).text;
}

/// Phrase a row as the EQUATION it represents, rather than as the
/// unknown it solves for. A Newton residual lives in equation space:
/// row i of `f` is the KCL balance at node i, or a device's
/// constraint equation — not "the current through L1".
[[nodiscard]] inline std::string row_equation_label(
    const topology::Graph& graph,
    const DevicePool& pool,
    Index row) {
    const auto info = describe_row(graph, pool, row);
    switch (info.kind) {
        case RowKind::NodeVoltage:
            return std::format("the KCL balance at {}", info.label);
        case RowKind::SourceCurrent:
        case RowKind::InductorCurrent:
            return std::format("the branch equation of {}",
                                info.label);
        case RowKind::OutOfRange:
        default:
            return info.label;
    }
}

/// Report the `k` largest-magnitude entries of `v` by name, e.g.
///   "node vout (4.2e+00), current through inductor L1 (1.1e-01)"
/// Used by the Newton / event diagnostics to point at WHERE a
/// residual or step is worst instead of only how big it is.
[[nodiscard]] inline std::string top_entries_by_name(
    const topology::Graph& graph,
    const DevicePool& pool,
    const Vector& v,
    int k = 3) {
    const auto n = static_cast<Index>(v.size());
    if (n == 0 || k <= 0) {
        return "(empty)";
    }
    std::vector<Index> idx;
    idx.reserve(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i) idx.push_back(i);

    const int take = static_cast<int>(n) < k ? static_cast<int>(n) : k;
    std::partial_sort(
        idx.begin(), idx.begin() + take, idx.end(),
        [&v](Index a, Index b) {
            return std::abs(v[a]) > std::abs(v[b]);
        });

    std::string out;
    for (int j = 0; j < take; ++j) {
        if (j > 0) out += ", ";
        out += std::format("{} ({:.3e})",
                            row_label(graph, pool, idx[static_cast<std::size_t>(j)]),
                            v[idx[static_cast<std::size_t>(j)]]);
    }
    return out;
}

}  // namespace pulsim::pwl
