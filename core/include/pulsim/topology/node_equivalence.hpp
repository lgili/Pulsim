#pragma once

// =============================================================================
// Pulsim v2 — Layer 1: Node equivalence under switch state
// =============================================================================
//
// `pulsim-v2-topology-and-switch-enumeration` Phase 4.
//
// Given a Graph and a SwitchStateMask, compute equivalence classes
// of nodes under "closed switches short their endpoints":
//
//   for every Switch-kind branch where mask.get(i) == true:
//       union_find.union(branch.from, branch.to)
//
// Transitive shorts merge naturally — 3 switches in series with all
// closed merge 4 nodes into one class.
//
// Implementation: path-compressed union-find (Tarjan), O(α(n))
// amortised per query.
//
// Layer 4 uses this to reduce the per-segment matrix size: if 3
// nodes belong to the same equivalence class, the segment matrix
// has 1 row+column for them, not 3.
//
// Ground node convention: ground (`kGround = -1`) is its own
// representative; closed switches that touch ground promote the
// other endpoint into ground's class. Internally we represent
// ground with the sentinel index `kGroundRep = -1`.

#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <vector>

namespace pulsim::topology {

class NodeEquivalence {
public:
    NodeEquivalence(const Graph& graph,
                    const SwitchStateMask& state)
        : num_nodes_{graph.num_nodes()},
          parent_(static_cast<Size>(graph.num_nodes())),
          rank_(static_cast<Size>(graph.num_nodes()), 0),
          ground_class_(false) {
        // Initialise each node as its own representative.
        for (Index i = 0; i < num_nodes_; ++i) {
            parent_[static_cast<Size>(i)] = i;
        }

        // For every Switch-kind branch whose state bit is true, union
        // its endpoints. We iterate branches in order; the switch
        // counter advances only on Switch-kind branches, matching the
        // SwitchStateMask's bit ordering convention (bit i = i-th
        // Switch in branch-iteration order).
        Size sw_idx = 0;
        for (Index b = 0; b < graph.num_branches(); ++b) {
            const auto& br = graph.branch(b);
            if (br.kind != BranchKind::Switch) continue;
            if (state.get(sw_idx)) {
                union_nodes(br.from, br.to);
            }
            ++sw_idx;
        }
    }

    // Move-only. The internal union-find arrays are large enough
    // that we prefer move semantics by default.
    NodeEquivalence(const NodeEquivalence&) = delete;
    NodeEquivalence& operator=(const NodeEquivalence&) = delete;
    NodeEquivalence(NodeEquivalence&&) noexcept = default;
    NodeEquivalence& operator=(NodeEquivalence&&) noexcept = default;

    // -------------------------------------------------------------------------
    // Queries
    // -------------------------------------------------------------------------

    /// Representative node of the class containing `node`. Returns
    /// `kGround` (= -1) if the class includes the ground rail.
    /// Uses path compression — non-const because of that, but the
    /// observable result is purely a function of construction state.
    [[nodiscard]] Index representative_of(Index node) const noexcept {
        if (node < 0) return kGround;
        if (ground_class_for_[static_cast<Size>(find(node))]) return kGround;
        return find(node);
    }

    [[nodiscard]] bool are_equivalent(Index a, Index b) const noexcept {
        return representative_of(a) == representative_of(b);
    }

    /// Number of distinct equivalence classes (counting the ground
    /// class as one if any node was unioned with ground).
    [[nodiscard]] Size num_classes() const noexcept {
        std::vector<Index> reps;
        reps.reserve(static_cast<Size>(num_nodes_));
        for (Index i = 0; i < num_nodes_; ++i) {
            reps.push_back(representative_of(i));
        }
        std::sort(reps.begin(), reps.end());
        reps.erase(std::unique(reps.begin(), reps.end()), reps.end());
        return reps.size();
    }

    /// Flat list of node ids that share the representative `rep`.
    /// Order is ascending by node id for determinism. If `rep ==
    /// kGround`, returns the nodes unioned with ground.
    [[nodiscard]] std::vector<Index> class_members(Index rep) const {
        std::vector<Index> members;
        for (Index i = 0; i < num_nodes_; ++i) {
            if (representative_of(i) == rep) members.push_back(i);
        }
        return members;
    }

private:
    // ---- Union-find primitives ----------------------------------------------

    [[nodiscard]] Index find(Index x) const noexcept {
        // Path compression — mutate `parent_` lazily. The const_cast is
        // sound because the observable equivalence relation does not
        // depend on which representative we expose.
        while (parent_[static_cast<Size>(x)] != x) {
            const Index parent = parent_[static_cast<Size>(x)];
            const Index grandparent = parent_[static_cast<Size>(parent)];
            const_cast<std::vector<Index>&>(parent_)[static_cast<Size>(x)] =
                grandparent;
            x = grandparent;
        }
        return x;
    }

    void union_nodes(Index a, Index b) noexcept {
        // Special-case the ground sentinel: ground (-1) is not a real
        // node in parent_; we track ground-equivalence with a parallel
        // bool array indexed by representative.
        const bool a_is_ground = (a == kGround);
        const bool b_is_ground = (b == kGround);
        if (a_is_ground && b_is_ground) return;
        if (a_is_ground) {
            mark_ground(find(b));
            return;
        }
        if (b_is_ground) {
            mark_ground(find(a));
            return;
        }
        const Index root_a = find(a);
        const Index root_b = find(b);
        if (root_a == root_b) return;

        // Union by rank to keep the tree shallow.
        if (rank_[static_cast<Size>(root_a)] <
            rank_[static_cast<Size>(root_b)]) {
            parent_[static_cast<Size>(root_a)] = root_b;
            if (ground_class_for_[static_cast<Size>(root_a)])
                ground_class_for_[static_cast<Size>(root_b)] = true;
        } else if (rank_[static_cast<Size>(root_a)] >
                   rank_[static_cast<Size>(root_b)]) {
            parent_[static_cast<Size>(root_b)] = root_a;
            if (ground_class_for_[static_cast<Size>(root_b)])
                ground_class_for_[static_cast<Size>(root_a)] = true;
        } else {
            parent_[static_cast<Size>(root_b)] = root_a;
            ++rank_[static_cast<Size>(root_a)];
            if (ground_class_for_[static_cast<Size>(root_b)])
                ground_class_for_[static_cast<Size>(root_a)] = true;
        }
    }

    void mark_ground(Index root) noexcept {
        if (root >= 0 && root < num_nodes_) {
            ground_class_for_[static_cast<Size>(root)] = true;
        }
        ground_class_ = true;
    }

    // Lazy-allocate ground_class_for_ on first union — keeps the
    // common all-switches-open path cheap.
    void ensure_ground_array() noexcept {
        if (ground_class_for_.size() != static_cast<Size>(num_nodes_)) {
            ground_class_for_.assign(static_cast<Size>(num_nodes_), false);
        }
    }

    Index num_nodes_;
    mutable std::vector<Index> parent_;
    std::vector<std::uint8_t> rank_;
    std::vector<std::uint8_t> ground_class_for_ = std::vector<std::uint8_t>(
        static_cast<Size>(num_nodes_), 0);
    bool ground_class_;  // any node unioned with ground?
};

}  // namespace pulsim::topology
