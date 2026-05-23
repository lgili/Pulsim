# Layer 1 — Topology + Switch enumeration

The topology layer. Owns the graph data structure, the switch-state
bitmask, the Gray-code enumeration over switch combinations, the
union-find that computes node equivalence under a state, and the
`TopologyKey` that Layer 4's PWL state-space cache will use as its
map key.

Pure topology — no math, no stamping, no solver. Layer 2 (device
models) consumes the graph; Layer 4 consumes the enumeration + key.

## Public surface

```cpp
// pulsim/topology/graph.hpp
namespace pulsim::topology {
    enum class BranchKind { PassiveLinear, Source, Switch, Nonlinear };
    struct Node { Index id; std::string name; };
    struct Branch { Index id; Index from; Index to; BranchKind kind; };
    class Graph {
        Index add_node(std::string);
        Index add_branch(Index from, Index to, BranchKind);
        Index num_nodes() / num_branches() / ground();
        Size num_switches();
        std::span<const Index> branches_of(Index node);
        std::uint64_t id();                       // FNV-1a structural hash
        // move-only
    };
}

// pulsim/topology/switch_state.hpp
class SwitchStateMask {
    explicit SwitchStateMask(Size N);             // throws if N > 64
    bool get(i) / void set(i, v) / void flip(i) / Size count();
    operator== / != / <;
    std::size_t hash();                            // splitmix64-mixed
};

// pulsim/topology/enumerator.hpp
SwitchStateEnumerator enumerate_switch_states(Size N);
// for (auto mask : enumerate_switch_states(N)) { ... }  Gray-code order

// pulsim/topology/node_equivalence.hpp
class NodeEquivalence {
    NodeEquivalence(const Graph&, const SwitchStateMask&);
    Index representative_of(Index node);
    bool are_equivalent(Index, Index);
    Size num_classes();
    std::vector<Index> class_members(Index rep);
};

// pulsim/topology/key.hpp
struct TopologyKey { std::uint64_t graph_id; SwitchStateMask state; };
```

## Why Gray-code enumeration order

For N switches, there are 2^N states. Layer 4 pre-factorises the
system matrix per state. Two facts combine:

1. Adjacent Gray-code states differ by EXACTLY one bit (one switch
   flipped).
2. Flipping one switch is a rank-1 update of the MNA matrix.

Sherman-Morrison-Woodbury: updating an LU factor under a rank-1
modification is O(n²) — versus O(n³) for full re-factor. The
cache-build phase becomes 10-100× cheaper at large N.

Standard binary order (0, 1, 2, 3, ...) flips multiple bits between
consecutive values, breaking the rank-1 property. Gray-code order
(0, 1, 3, 2, 6, 7, 5, 4, ...) maintains it.

Implementation: textbook `state[i] = i XOR (i >> 1)`.

## How Layer 4 will use this

```cpp
// Layer 4 cache build (sketch — concrete in its own OpenSpec)
struct PwlStateSpaceCache {
    std::unordered_map<TopologyKey, FactorizedStateSpace> entries_;

    void build(const Graph& g, const Layer2Models& models) {
        for (auto mask : enumerate_switch_states(g.num_switches())) {
            NodeEquivalence eq(g, mask);
            auto matrices = build_state_space(g, eq, models);
            entries_.emplace(
                TopologyKey{g.id(), mask},
                factorize(matrices));   // analyze + factorize ONCE
        }
    }

    const FactorizedStateSpace& lookup(const SwitchStateMask& m) const {
        return entries_.at(TopologyKey{graph_.id(), m});
    }
};

// Simulator hot loop (Layer 5)
void step(Vector& x, Real dt) {
    const auto& fss = cache_.lookup(current_switch_state_);
    fss.solve(rhs, x_next);                       // O(n) triangular
}
```

## Node equivalence under closed switches

A closed switch shorts its two endpoints. Multiple closed switches
combine transitively — a chain of 3 closed switches merges 4 nodes.

Path-compressed union-find (Tarjan) is the canonical O(α(n)) ≈ O(1)
data structure for this. Each `NodeEquivalence` construction is
O(graph.num_branches() · α(graph.num_nodes())) ≈ O(branches).
Negligible compared to the matrix work Layer 4 does next.

Ground rail handling: the special sentinel index `kGround = -1` is
not stored in the union-find arrays directly; instead a parallel
boolean array tracks "which representative is in ground's class".
Closed switches touching ground promote the other endpoint's
representative into the ground class. Equivalence queries against
ground return `kGround` consistently.

## What this layer does NOT do

- No device math. `Branch` has `BranchKind` only.
- No stamping. Layer 3 builds matrices FROM the graph.
- No solver. No Newton, no events. Layer 5.
- No automatic pruning of "redundant" switch combinations. Layer 1
  enumerates all 2^N; Layer 4 decides which to materialise.
- No graph mutation after construction. Build-once-then-consume.

## Validation

`pulsim_v2_layer1_tests` covers each header independently plus the
small Buck-like topology end-to-end:

- Graph: counts, adjacency, switch-count, structural hash stability +
  divergence, move-only contract.
- SwitchStateMask: bit ops, equality, ordering, hash, unordered_map
  round-trip, N > 64 rejection.
- Enumerator: N=0 yields 1 state, N=3 yields 8 distinct, Gray-code
  one-bit-diff invariant, N=16 smoke (65536 states).
- NodeEquivalence: open → identity, single switch merges 2 nodes,
  ground promotion, transitive shorts, sorted class_members.
- TopologyKey: same-graph identity, different-state difference,
  different-graph difference, unordered_map round-trip.

Current: **126 assertions / 36 test cases, all green.**
