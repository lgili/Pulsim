## Phase 1 — Graph data structure (~0.5 days)

### 1.1 `topology/graph.hpp` — Node + Branch + Graph
- [x] 1.1.1 `enum class BranchKind { PassiveLinear, Source, Switch,
      Nonlinear }`. Captures the topological category — Layer 4
      enumerates over `Switch` branches; the other kinds contribute
      unconditionally to every segment.
- [x] 1.1.2 `struct Node { Index id; std::string name; }`. Name is
      optional debug aid; identity is `id`. Construction strips any
      `id == kGround` reservation so callers can use
      `Graph::ground()` for the canonical ground sentinel (-1).
- [x] 1.1.3 `struct Branch { Index id; Index from; Index to;
      BranchKind kind; }`. Endpoints are node `Index` values; `kind`
      is the topological category. NO parameters, NO device-model
      pointer — Layer 2 holds the back-reference from device model
      to branch id, not the other way.
- [x] 1.1.4 `class Graph` — SoA storage: `std::vector<Node> nodes_`,
      `std::vector<Branch> branches_`, plus a per-node adjacency
      list `std::vector<std::vector<Index>> node_to_branches_` for
      O(degree) neighbour lookup. Method surface:
      - `Index add_node(std::string name)`
      - `Index add_branch(Index from, Index to, BranchKind k)`
      - `Index num_nodes() const`
      - `Index num_branches() const`
      - `const Node& node(Index i) const`
      - `const Branch& branch(Index i) const`
      - `std::span<const Index> branches_of(Index node) const`
      - `Index ground() const` — returns `kGround = -1` (sentinel)
- [x] 1.1.5 `Graph` is move-constructible / move-assignable, NOT
      copy-constructible (large object, avoid accidental deep
      copies). Build-once-then-consume contract.

### 1.2 Tests `tests/v2/layer1/test_graph.cpp`
- [x] 1.2.1 Empty graph: `num_nodes() == 0`, `num_branches() == 0`,
      `ground() == kGround`.
- [x] 1.2.2 Add 3 nodes + 2 branches → counts increase, indices
      monotonic.
- [x] 1.2.3 `branches_of(node)` returns the right adjacency set
      after adding branches that share endpoints.
- [x] 1.2.4 `BranchKind` is correctly stored and retrievable.
- [x] 1.2.5 `Graph` is non-copyable but movable (compile-time
      `static_assert`).

## Phase 2 — Switch state masks (~0.5 days)

### 2.1 `topology/switch_state.hpp` — `SwitchStateMask`
- [x] 2.1.1 Backing storage: `std::uint64_t bits_` for ≤ 64
      switches (covers > 99 % of real PE circuits — a 3φ inverter
      has 6, an MMC arm has ~12, a 5-level NPC has ~16). For larger
      N, a future enhancement can swap to a dynamic bitset without
      changing the public API.
- [x] 2.1.2 Constructor `SwitchStateMask(Size num_switches)`
      initializes to all-zero (all switches open). `num_switches`
      ≤ 64; `> 64` throws `std::invalid_argument`.
- [x] 2.1.3 Bit ops:
      - `bool get(Size i) const`
      - `void set(Size i, bool v)`
      - `void flip(Size i)`
      - `Size count() const` — popcount of closed switches
      - `Size size() const` — total switches (stored separately)
- [x] 2.1.4 Ordering + equality for use as a `std::map` /
      `std::unordered_map` key:
      - `operator==`, `operator!=`
      - `operator<` (lexicographic on bit values)
      - `std::size_t hash() const` — stable, deterministic
      - `std::hash<SwitchStateMask>` specialization
- [x] 2.1.5 `to_string()` returns binary representation
      ("0b110010 N=8") for diagnostics.

### 2.2 Tests `tests/v2/layer1/test_switch_state.cpp`
- [x] 2.2.1 Default-constructed mask of size N has `count() == 0`.
- [x] 2.2.2 `set(i, true)` and `get(i)` round-trip.
- [x] 2.2.3 `flip` toggles correctly across multiple calls.
- [x] 2.2.4 Two masks with same bits AND same size compare equal
      and hash to the same value.
- [x] 2.2.5 Masks of different sizes are NOT equal even if bits
      coincide (size is part of identity).
- [x] 2.2.6 `operator<` gives a stable total order suitable for
      `std::set`.
- [x] 2.2.7 Throws on `Size > 64`.

## Phase 3 — Enumeration over switch states (~0.5 days)

### 3.1 `topology/enumerator.hpp` — Gray-code iterator
- [x] 3.1.1 `class SwitchStateEnumerator` — forward iterator that
      yields all 2^N `SwitchStateMask` values for N switches.
- [x] 3.1.2 Iteration order: **Gray code**. Each consecutive state
      differs from the previous by EXACTLY one bit flip. This is
      Layer 4's preferred cache-warm-up order — re-factorising a
      matrix that differs by one rank-1 update is faster than from
      scratch (Sherman-Morrison-friendly).
- [x] 3.1.3 Range-friendly: provides `begin()` + `end()` returning
      iterators that compose with C++20 ranges (`for (auto m :
      enumerator) { ... }`).
- [x] 3.1.4 Sentinel-based end iterator (no need to store the full
      end state — saves stack on large enumerations).
- [x] 3.1.5 `enumerate_switch_states(Size num_switches)` free
      function returns the enumerator inline.

### 3.2 Tests `tests/v2/layer1/test_enumerator.cpp`
- [x] 3.2.1 N = 0 → yields exactly one state (all-empty mask).
- [x] 3.2.2 N = 3 → yields exactly 2^3 = 8 distinct masks.
      Collect into `std::set` and assert size 8 + content matches
      `{000, 001, 010, ..., 111}`.
- [x] 3.2.3 Gray-code property: every consecutive pair differs by
      exactly ONE bit. Test for N = 4 (16 states).
- [x] 3.2.4 N = 16 → yields exactly 65536 distinct masks (smoke
      test for performance: should complete in << 1 second).
- [x] 3.2.5 The enumerator does NOT throw or overflow at N = 64.

## Phase 4 — Node equivalence under switch state (~0.75 days)

### 4.1 `topology/node_equivalence.hpp` — union-find over closed switches
- [x] 4.1.1 `class NodeEquivalence` — built from a `Graph` + a
      `SwitchStateMask`. Internally a path-compressed union-find
      over node indices.
- [x] 4.1.2 Build algorithm:
      1. Initialise each node as its own representative.
      2. For each switch branch where the mask bit is `true`,
         union its endpoints (`from` and `to`).
      3. Path-compression on subsequent queries.
- [x] 4.1.3 Method surface:
      - `Index representative_of(Index node) const`
      - `bool are_equivalent(Index a, Index b) const`
      - `Size num_classes() const` — number of distinct
        equivalence classes
      - `std::vector<Index> class_members(Index representative) const`
        — flat list of node ids in that class (for Layer 4's matrix
        row-merging)
- [x] 4.1.4 Ground node is ALWAYS its own representative (closed
      switches that touch ground promote the other endpoint to
      ground's class).
- [x] 4.1.5 `NodeEquivalence` is move-only (large internal state).

### 4.2 Tests `tests/v2/layer1/test_node_equivalence.cpp`
- [x] 4.2.1 Empty switch state → every node is its own class,
      `num_classes() == graph.num_nodes()`.
- [x] 4.2.2 Buck converter topology (5 nodes: vin, sw, out, vload,
      gnd; 2 switches: MOSFET vin↔sw, Diode gnd↔sw):
      - MOSFET on, Diode off → sw class includes vin and sw
      - MOSFET off, Diode on → sw class includes sw and gnd
      - Both on → sw class includes vin, sw, gnd (and sw becomes
        equivalent to gnd — short circuit on the bus, which the
        topology enumerator allows but Layer 4 may flag later)
      - Both off → all 5 nodes are their own classes
- [x] 4.2.3 Transitive shorts: 3 switches in series (sw1 closes
      n1↔n2, sw2 closes n2↔n3, sw3 closes n3↔gnd) — closing all 3
      merges {n1, n2, n3, gnd} into one class.
- [x] 4.2.4 `class_members(rep)` returns the right node ids.

## Phase 5 — Topology key for Layer 4 cache (~0.25 days)

### 5.1 `topology/key.hpp` — TopologyKey
- [x] 5.1.1 `struct TopologyKey { std::uint64_t graph_id;
      SwitchStateMask state; }`. `graph_id` is a stable hash of the
      graph's structure (assigned by `Graph::id()`); `state` is
      the switch combination.
- [x] 5.1.2 `Graph::id()` — computes a stable 64-bit hash of the
      graph's (num_nodes, num_branches, branch endpoints, branch
      kinds) tuple. Two graphs with the same structure have the
      same id; different structures have (with overwhelming
      probability) different ids. Cached lazily on first call.
- [x] 5.1.3 `TopologyKey` provides `operator==`, `operator!=`,
      `hash()`, and a `std::hash` specialization. Ready for use as
      `std::unordered_map<TopologyKey, StateSpaceMatrices>` in
      Layer 4.

### 5.2 Tests `tests/v2/layer1/test_key.cpp`
- [x] 5.2.1 Same graph + same state → same key + same hash.
- [x] 5.2.2 Same graph + different state → different key.
- [x] 5.2.3 Different graph (one extra branch) + same state →
      different `graph_id`, different key.
- [x] 5.2.4 `std::unordered_map<TopologyKey, int>` round-trip:
      insert, lookup, find returns the right value.

## Phase 6 — Documentation (~0.25 days)

### 6.1 `docs/pulsim-v2/layer1-topology-and-enumeration.md`
- [x] 6.1.1 Section "Why Gray-code enumeration order" — Sherman-
      Morrison rank-1 update is faster than re-factoring from
      scratch when only one switch changed.
- [x] 6.1.2 Section "How Layer 4 will use this" — sketch of the
      `unordered_map<TopologyKey, FactorizedMatrices>` lookup
      structure that this layer enables.
- [x] 6.1.3 Section "What this layer does NOT do" — explicit list
      of out-of-scope items (matches the proposal's NOT-do list).

## Phase 7 — Validation

- [x] 7.1 `pulsim_v2_layer1_tests` MUST pass with zero failures.
      Initial target: ≥ 30 assertions / ≥ 15 test cases.
- [x] 7.2 `pulsim_v2_layer0_tests` MUST stay green (Layer 0
      regression check — Layer 1 must not break what's below it).
- [x] 7.3 v1 suites (`pulsim_tests`, `pulsim_simulation_tests`)
      MUST stay green. No v1 touched.
- [x] 7.4 `openspec validate pulsim-v2-topology-and-switch-enumeration
      --strict` MUST pass.
