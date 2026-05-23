# Design — `pulsim-v2-topology-and-switch-enumeration` (Layer 1)

## Goal

Land the pure-topology primitives that Layer 4 (PWL state-space cache)
will consume to enumerate switch combinations, identify node
equivalence per combination, and key the cache entries.

Layer 1 owns:
- The graph data structure (nodes, branches, adjacency)
- The switch combinatorics (bitmask state + Gray-code enumerator)
- The node-equivalence calculator (union-find over closed switches)
- The topology key (graph_id × switch_mask) for cache lookup

Layer 1 does NOT own:
- Device parameters / math (Layer 2)
- Stamp generation (Layer 3)
- Anything matrix-shaped (Layer 4 consumes the graph + state to
  build matrices; Layer 1 just describes the graph)

## Key design decisions

### Branches store only `BranchKind`, NOT the device pointer

A `Branch` knows its endpoints and its topological category
(`PassiveLinear`, `Source`, `Switch`, `Nonlinear`). It does NOT
know which Layer-2 device model is attached.

Reasoning: Layer 1 must be independently testable WITHOUT a Layer 2
to draw devices from. If `Branch` held `DeviceModel*`, every Layer 1
test would need to construct (or fake) a Layer 2 model. Inverting
the reference — Layer 2's device model holds the branch index it
stamps into — keeps Layer 1 self-contained.

### Gray-code enumeration order

For N switches, there are 2^N states. The Layer 4 cache pre-factorises
the system matrix per state. Adjacent enumeration states that differ
by ONE bit allow Sherman-Morrison rank-1 update of the factorisation
(O(n²) vs O(n³) full re-factor) — that's a 10-100× speedup on the
cache-build phase for large N.

Standard binary order (0, 1, 2, 3, ...) flips multiple bits between
consecutive values (e.g. 3 → 4 flips three bits). Gray code (0, 1,
3, 2, 6, 7, 5, 4, ...) flips exactly one bit per step.

Implementation: `state[i] = i XOR (i >> 1)` is the textbook
binary-to-Gray conversion. The iterator just increments an internal
counter and applies this transform.

### Bitmask backed by `std::uint64_t`

Single uint64 holds up to 64 switches. Real-world coverage:
- 3φ inverter: 6 switches
- MMC arm (6 submodules): 12 switches
- 5-level NPC: 8-16 switches per phase
- 13-level cascaded H-bridge: 24-48 switches

Going past 64 is rare for a single segment cache — and at N > 20 the
2^N enumeration itself becomes the bottleneck (Layer 4 needs lazy
generation + on-demand caching, which is a follow-up enhancement).
A single uint64 is fast, hashable, comparable, and trivially
copyable. The public API is bit-indexed so a future swap to
`boost::dynamic_bitset` is non-breaking.

### Union-find for node equivalence

Closed switches short their endpoints. With multiple closed switches
in series, the shorts transitively merge multiple nodes into one
equivalence class.

Union-find (Tarjan, path compression + union-by-rank) is the
canonical O(α(n)) ≈ O(1) data structure for this. Used by:
- Compilers (constraint-graph reachability)
- Image processing (connected-component labelling)
- Network flow (boundary tracking)
- ...and by every PWL simulator in this domain.

Per-segment cost: O(switches_closed) union ops + O(num_nodes) find
ops. Negligible compared to the matrix factorise step Layer 4 will
do next.

### Stable `graph_id` via structural hash

Layer 4's cache is keyed on (graph_id, switch_mask). The `graph_id`
must:
1. Be stable across processes (same circuit → same id).
2. Be cheap to compute (cached after first call).
3. Have negligible collision probability across realistic circuit
   structures.

Implementation: FNV-1a 64-bit hash over the sequence (num_nodes,
num_branches, [(from, to, kind) per branch]). FNV-1a is fast,
non-cryptographic, has good avalanche, and is trivially
deterministic. Collision probability at 2^64 buckets is ≈ 0 for
the realistic circuit-count Pulsim will ever face.

The hash is computed lazily: `Graph::id()` caches the result so
repeated calls cost O(1).

## What this layer hands to Layer 2

```cpp
// Layer 2 device-model side
struct MosfetModel {
    Index branch_id;       // back-reference to the Layer-1 Branch
    MosfetParams params;
    // ... pure math, no topology ...
};

// Layer 4 (later) cache-build side
for (auto mask : enumerate_switch_states(graph.num_switches())) {
    auto eq = compute_node_equivalence(graph, mask);
    TopologyKey key{graph.id(), mask};
    cache[key] = build_state_space(graph, eq, layer2_models);
}
```

Layer 1 has done its job once it gives Layer 4 the `(graph, mask) →
node-equivalence + key` pipeline.

## Risks

1. **64-switch limit.** A future cascaded-H-bridge user might want
   N > 64. Mitigation: the public API takes a bit index `Size i`,
   so swapping the backing storage to `dynamic_bitset` is
   non-breaking. Defer until a real workload hits the limit.

2. **`graph_id` collision.** FNV-1a 64-bit has ≈ 2^32 expected
   distinct hashes before the first collision. For a single
   process running thousands of circuits, collision probability
   stays < 2^-30. Negligible. Mitigation if it ever matters:
   switch to SipHash-2-4 (still fast, 128-bit) without API
   change.

3. **Union-find equivalence under many closed switches.** Worst
   case: 2^64 enumeration × graph traversal per state. Each
   computation is O(num_branches · α(num_nodes)) ≈ O(num_branches).
   For a 1000-branch circuit and 2^20 states, that's 10^9 ops
   ≈ 10 s on commodity CPU. Tolerable for the one-time cache
   build. Layer 4's lazy / partial enumeration will mitigate the
   tail.

## What this layer explicitly does NOT do

- No device math. `Branch` has `BranchKind` only.
- No stamping. Layer 3 builds matrices FROM the graph.
- No solver. No Newton, no events.
- No automatic pruning of "redundant" switch combinations (e.g.
  two switches connecting the same node pair). Layer 1 enumerates
  all 2^N; Layer 4 decides which to materialise.
- No graph mutation after construction. Build-once-then-consume.

## Validation

`pulsim_v2_layer1_tests` covers each of the 5 headers in isolation
plus integrated tests that build a small Buck-like topology and
verify the full pipeline (graph → enumerator → equivalence → key).
Target: ≥ 30 assertions / ≥ 15 test cases.
