#pragma once

// =============================================================================
// Pulsim v2 — Layer 1: TopologyKey for Layer 4 cache lookup
// =============================================================================
//
// `pulsim-v2-topology-and-switch-enumeration` Phase 5.
//
// A `TopologyKey` is a `(graph_id, switch_state)` pair that uniquely
// identifies a single configuration of a circuit:
//
//   * `graph_id` is `Graph::id()` — a stable 64-bit structural hash.
//     Same graph structure → same id; different structure → different
//     id (with FNV-1a-grade collision probability ≈ 0 for realistic
//     circuit counts).
//
//   * `state` is the `SwitchStateMask` — which switches are closed.
//
// Layer 4's PWL state-space cache uses this as the key to a
// `std::unordered_map<TopologyKey, FactorizedStateSpace>` that
// stores the pre-built and pre-factorised (A, B, C, D) matrices per
// configuration. `analyze + factorize` happen ONCE per key; many
// `solve` calls reuse the cached factor.

#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <cstdint>
#include <functional>

namespace pulsim::topology {

struct TopologyKey {
    std::uint64_t graph_id;
    SwitchStateMask state;

    friend bool operator==(const TopologyKey& a, const TopologyKey& b) noexcept {
        return a.graph_id == b.graph_id && a.state == b.state;
    }
    friend bool operator!=(const TopologyKey& a, const TopologyKey& b) noexcept {
        return !(a == b);
    }

    /// Mix the graph_id and the state hash together via splitmix64.
    /// Determinism + good avalanche → realistic-load collision rate
    /// dominated by birthday on a 64-bit space (= negligible).
    [[nodiscard]] std::size_t hash() const noexcept {
        std::uint64_t x = graph_id ^ static_cast<std::uint64_t>(state.hash());
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27; x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return static_cast<std::size_t>(x);
    }
};

}  // namespace pulsim::topology

namespace std {
template <>
struct hash<pulsim::topology::TopologyKey> {
    std::size_t operator()(
        const pulsim::topology::TopologyKey& k) const noexcept {
        return k.hash();
    }
};
}  // namespace std
