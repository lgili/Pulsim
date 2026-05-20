## Why

Layer 0 (`bootstrap-pulsim-v2-kernel`, archived) gave the v2 kernel its
foundation — numeric types, dense aliases, sparse matrix + solver
abstraction. Everything above it now needs **graph topology** to
describe what a circuit IS, before any math (Layer 2), any stamping
(Layer 3), or — most importantly — the **switch-state enumeration**
that the Layer 4 PWL state-space cache will key on.

Concretely, Layer 4's contract requires:

1. For a circuit with N binary-state switches, enumerate up to 2^N
   topology configurations.
2. For each configuration, determine which nodes are electrically
   merged by closed switches (a closed switch shorts its two
   terminals → node equivalence class).
3. Produce a stable hashable key per (graph identity, switch mask)
   so the cache can store the pre-factorized state-space per config.
4. Iterate efficiently — enumeration order matters for cache-locality
   and for lazy generation when N is large.

NONE of those four primitives exist in v1. v1 has fragmented bits
(`pwl_state_space_supports_all_devices`, `pwl_segment_engine`) but
they're 1-state-at-a-time, not the full combinatorics surface
Layer 4 needs. v2 builds them properly in one place.

This OpenSpec lands ONLY the topology + enumeration primitives. It
adds **no math, no stamping, no solver, no device models**. Pure
graph + switch combinatorics. Layer 2 (device models, the next
OpenSpec) consumes the graph; Layer 4 (PWL cache) consumes the
enumeration. Each layer is independently mergeable.

## What Changes

**Five new headers under `core/include/pulsim/v2/topology/`**:

1. **`graph.hpp`** — `Node`, `Branch`, `Graph`. SoA storage of nodes
   and branches plus adjacency lists for traversal. Pure topology:
   each branch knows only its endpoints and its topological kind
   (`PassiveLinear`, `Source`, `Switch`, `Nonlinear`). No
   parameters, no device math.

2. **`switch_state.hpp`** — `SwitchStateMask`. A fixed-width bitmask
   (backed by `std::uint64_t` for ≤ 64 switches, with a path to
   `boost::dynamic_bitset`-style growth for larger N). Provides
   `get(i)`, `set(i)`, `flip(i)`, `count()`, ordering operators
   (so it's directly usable as a `std::map` key in Layer 4),
   and a stable `hash()` for `std::unordered_map`.

3. **`enumerator.hpp`** — Forward iterator over 2^N switch states.
   Returns each `SwitchStateMask` in canonical order (Gray-code
   sequence so consecutive states differ by ONE bit — that's the
   Layer 4 cache-friendly order: re-factorizing from a neighbouring
   state's factor is faster than from scratch).

4. **`node_equivalence.hpp`** — Union-find data structure that,
   given a `Graph` + `SwitchStateMask`, computes equivalence
   classes of nodes (closed switches short their endpoints).
   Returns `representative_of(Index node) -> Index`. Layer 4 will
   use this to reduce the per-segment matrix size: if 3 nodes are
   shorted by closed switches, the segment matrix has 1 row+column
   for them, not 3.

5. **`key.hpp`** — `TopologyKey` = `(graph_id, SwitchStateMask)`.
   Stable hash + equality so Layer 4 can use it as a
   `std::unordered_map` key directly.

Plus its own test binary `pulsim_v2_layer1_tests` with isolated
test files per concept (1 file per header).

**Strict layer discipline.** Layer 1 includes:
- `pulsim/v2/numeric/types.hpp` (for `Index`, `Size`)
- `<vector>`, `<bitset>`, `<algorithm>`, `<cstdint>`

That's it. NO Eigen (graph doesn't need linear algebra), NO sparse
matrix (matrix is built by Layer 3 from the graph), NO solver
(solver is Layer 5). Compile-time enforced: if Layer 1 ever imports
a Layer 2+ header, the build breaks.

## Impact

- **Affected specs**:
  - NEW capability `kernel-v2-topology` (graph + switch enumeration).
- **Affected code** (this proposal — estimated 800-1200 LOC added,
  0 LOC modified):
  - NEW `core/include/pulsim/v2/topology/` directory (5 headers).
  - NEW `core/tests/v2/layer1/` directory (5 test files + main).
  - NEW CMake test target `pulsim_v2_layer1_tests`.
  - NEW `docs/pulsim-v2/layer1-topology-and-enumeration.md` design
    note.
- **Migration**: none. Layer 1 is pure new code in `pulsim::v2::topology`.
  Nothing in v1 is touched; no Python API change; existing tests
  unaffected.
- **Risk**: low. Layer 1 has no behavioral coupling to anything that
  runs in production today. Failure mode is "Layer 1 tests fail" —
  the layer above (Layer 2 device models, next OpenSpec) blocks
  until they pass.
- **What this proposal explicitly does NOT do**:
  - No device math — `Branch` has a `BranchKind` enum but stores
    NO parameters. Parameters live on Layer 2 device models, which
    HOLD a `BranchKind::Source` reference (or similar) without
    Layer 1 knowing what a "voltage source" is.
  - No stamping. The graph doesn't know how to build an MNA
    matrix. That's Layer 3's job.
  - No solver. No Newton, no events. Layer 5.
  - No automatic pruning of infeasible switch combinations
    (e.g. two switches connecting the same nodes — closing both
    is redundant). Pruning is a Layer 4 concern; Layer 1
    enumerates all 2^N and Layer 4 decides which to materialize.
  - No graph mutation after construction beyond the build phase.
    `Graph` is "build → freeze → consume" — no edge deletion, no
    node merging at runtime. Layer 4 builds a Graph from the
    user's Circuit, then iterates without modifying.
