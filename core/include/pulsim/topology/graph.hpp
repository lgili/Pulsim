#pragma once

// =============================================================================
// Pulsim v2 — Layer 1: Graph (nodes + branches + adjacency)
// =============================================================================
//
// `pulsim-v2-topology-and-switch-enumeration` Phase 1.
//
// Pure topology. The Graph stores nodes and branches in SoA form
// with per-node adjacency lists for O(degree) neighbour lookup.
//
// A Branch knows only its endpoints (`from`, `to`) and its
// topological category (`BranchKind`). It does NOT hold device
// parameters or device-model pointers — that responsibility belongs
// to Layer 2 (device models), which holds the back-reference from
// device-to-branch by Branch::id. This inversion keeps Layer 1
// independently testable without faking Layer 2 devices.
//
// Layer 4 (PWL state-space cache) enumerates over `Switch`-kind
// branches; the other kinds (`PassiveLinear`, `Source`, `Nonlinear`)
// contribute unconditionally to every segment.

#include "pulsim/numeric/types.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pulsim::topology {

// -----------------------------------------------------------------------------
// BranchKind — topological category of a branch.
//
// The enum captures WHAT the branch does at the topology level, not
// WHICH concrete device it implements. Layer 4 enumerates over
// `Switch`-kind branches (their state toggles the topology
// combination); the others are unconditional contributors to every
// segment matrix.
// -----------------------------------------------------------------------------
enum class BranchKind : std::uint8_t {
    PassiveLinear = 0,  // R, L, C — linear contribution every segment
    Source        = 1,  // voltage / current source — always-on input
    Switch        = 2,  // binary on/off — Layer 4's combinatorial axis
    Nonlinear     = 3,  // Behavioral-mode semis — Newton needed per segment
};

// -----------------------------------------------------------------------------
// Node — graph vertex.
//
// `id` is the canonical identity (matches the SoA index in the Graph
// vector). `name` is optional debug aid; identity is `id` not `name`.
// -----------------------------------------------------------------------------
struct Node {
    Index id;
    std::string name;
};

// -----------------------------------------------------------------------------
// Branch — graph edge with topological kind.
//
// Endpoints are node `Index` values (or `kGround = -1` for the
// ground-rail sentinel). Layer 1 does NOT validate that endpoints
// exist — the caller (Layer 6 Circuit builder) holds that contract.
// -----------------------------------------------------------------------------
struct Branch {
    Index id;
    Index from;
    Index to;
    BranchKind kind;
};

// -----------------------------------------------------------------------------
// Graph — circuit topology.
//
// SoA storage: a vector of nodes, a vector of branches, plus a
// per-node adjacency list mapping node id → branch ids touching that
// node. Adjacency includes branches where the node appears as
// either `from` or `to`.
//
// Build-once-then-consume contract: after construction, callers
// should only call const methods. Mutating methods exist only
// during the build phase. Move-only — no accidental deep copies.
// -----------------------------------------------------------------------------
class Graph {
public:
    Graph() = default;

    // Move-only to avoid accidental deep copies of potentially-large
    // graphs (industrial inverter ≤ 1000 branches, well within
    // sensible move-semantics range).
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    Graph(Graph&&) noexcept = default;
    Graph& operator=(Graph&&) noexcept = default;

    // ---- Build phase --------------------------------------------------------

    /// Append a node. Returns its (zero-based) index.
    Index add_node(std::string name) {
        const auto id = static_cast<Index>(nodes_.size());
        nodes_.push_back(Node{id, std::move(name)});
        node_to_branches_.emplace_back();   // empty adjacency for this node
        graph_id_cached_ = false;
        return id;
    }

    /// Append a branch with topological kind. Returns its (zero-based)
    /// branch index. Endpoint validity is the caller's contract;
    /// `kGround = -1` is permitted as either endpoint.
    Index add_branch(Index from, Index to, BranchKind kind) {
        const auto id = static_cast<Index>(branches_.size());
        branches_.push_back(Branch{id, from, to, kind});
        // Adjacency: only record real (non-ground) endpoints.
        if (from >= 0 && from < num_nodes()) {
            node_to_branches_[static_cast<Size>(from)].push_back(id);
        }
        if (to >= 0 && to < num_nodes()) {
            node_to_branches_[static_cast<Size>(to)].push_back(id);
        }
        graph_id_cached_ = false;
        return id;
    }

    // ---- Const queries -------------------------------------------------------

    [[nodiscard]] Index num_nodes() const noexcept {
        return static_cast<Index>(nodes_.size());
    }
    [[nodiscard]] Index num_branches() const noexcept {
        return static_cast<Index>(branches_.size());
    }
    [[nodiscard]] static constexpr Index ground() noexcept {
        return kGround;
    }

    [[nodiscard]] const Node& node(Index i) const noexcept {
        return nodes_[static_cast<Size>(i)];
    }
    [[nodiscard]] const Branch& branch(Index i) const noexcept {
        return branches_[static_cast<Size>(i)];
    }

    /// Branches touching `node` (either endpoint matches). Returns a
    /// span into the internal adjacency list — valid as long as the
    /// Graph is not further mutated.
    [[nodiscard]] std::span<const Index> branches_of(Index node) const noexcept {
        if (node < 0 || node >= num_nodes()) return {};
        const auto& vec = node_to_branches_[static_cast<Size>(node)];
        return std::span<const Index>{vec.data(), vec.size()};
    }

    /// Count of `BranchKind::Switch` branches — Layer 4's combinatorial
    /// axis size.
    [[nodiscard]] Size num_switches() const noexcept {
        Size n = 0;
        for (const auto& b : branches_) {
            if (b.kind == BranchKind::Switch) ++n;
        }
        return n;
    }

    /// Stable 64-bit identifier computed from the graph's structural
    /// content. Two graphs with identical structure (num_nodes,
    /// num_branches, and the sequence of (from, to, kind) tuples) have
    /// the same id; different structures have (with overwhelming
    /// probability) different ids.
    ///
    /// Implementation: FNV-1a 64-bit over the structural tuple. Fast,
    /// deterministic, negligible collision probability for realistic
    /// circuit counts.
    ///
    /// Cached lazily on first call; subsequent calls are O(1).
    [[nodiscard]] std::uint64_t id() const noexcept {
        if (!graph_id_cached_) {
            graph_id_ = compute_structural_hash();
            graph_id_cached_ = true;
        }
        return graph_id_;
    }

private:
    // FNV-1a 64-bit. Standard non-cryptographic structural hash.
    [[nodiscard]] std::uint64_t compute_structural_hash() const noexcept {
        constexpr std::uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
        constexpr std::uint64_t kFnvPrime  = 0x100000001b3ULL;
        std::uint64_t h = kFnvOffset;
        auto mix = [&](std::uint64_t x) noexcept {
            for (int byte = 0; byte < 8; ++byte) {
                h ^= (x >> (byte * 8)) & 0xFF;
                h *= kFnvPrime;
            }
        };
        mix(static_cast<std::uint64_t>(nodes_.size()));
        mix(static_cast<std::uint64_t>(branches_.size()));
        for (const auto& b : branches_) {
            mix(static_cast<std::uint64_t>(b.from));
            mix(static_cast<std::uint64_t>(b.to));
            mix(static_cast<std::uint64_t>(b.kind));
        }
        return h;
    }

    std::vector<Node> nodes_{};
    std::vector<Branch> branches_{};
    std::vector<std::vector<Index>> node_to_branches_{};
    mutable std::uint64_t graph_id_ = 0;
    mutable bool graph_id_cached_ = false;
};

// Layer-1 contract: Graph is move-only.
static_assert(!std::is_copy_constructible_v<Graph>,
              "Graph must be move-only — no accidental deep copies");
static_assert(std::is_move_constructible_v<Graph>);

}  // namespace pulsim::topology
