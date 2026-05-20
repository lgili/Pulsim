## ADDED Requirements

### Requirement: Graph Data Structure for Circuit Topology

`pulsim::v2::topology` SHALL expose a `Graph` class that stores
nodes and branches in SoA form with per-node adjacency. The graph
is pure topology: branches know only their endpoints and a
topological `BranchKind` category. Device parameters and math live
above this layer (Layer 2) and reference Layer 1 by branch index.

The `Graph` class MUST provide:
- `Index add_node(std::string name)` — appends a node, returns its
  index.
- `Index add_branch(Index from, Index to, BranchKind kind)` —
  appends a branch with topological category `kind`, returns its
  index.
- `Index num_nodes() const`, `Index num_branches() const`.
- `const Node& node(Index i) const`, `const Branch& branch(Index i)
  const`.
- `std::span<const Index> branches_of(Index node) const` —
  adjacency list, O(degree) traversal.
- `Index ground() const` — returns `kGround = -1` (the canonical
  ground sentinel from Layer 0).

`BranchKind` MUST enumerate at minimum: `PassiveLinear`, `Source`,
`Switch`, `Nonlinear`. The enum captures topological categories,
not concrete device classes — Layer 4 enumerates over `Switch`-kind
branches; other kinds contribute unconditionally to every segment.

`Graph` MUST be move-constructible and move-assignable, MUST NOT be
copy-constructible (large object — avoid accidental deep copies).

#### Scenario: Empty graph reports zero counts and ground sentinel

- **GIVEN** a freshly-constructed `pulsim::v2::topology::Graph g`
- **THEN** `g.num_nodes()` SHALL be `0`
- **AND** `g.num_branches()` SHALL be `0`
- **AND** `g.ground()` SHALL be `kGround` (i.e. `-1`).

#### Scenario: Adding nodes and branches updates counts monotonically

- **GIVEN** an empty `Graph g`
- **WHEN** the user calls `g.add_node("n1")`, `g.add_node("n2")`,
  then `g.add_branch(0, 1, BranchKind::PassiveLinear)`
- **THEN** `g.num_nodes()` SHALL be `2`
- **AND** `g.num_branches()` SHALL be `1`
- **AND** `g.branch(0).kind` SHALL equal `BranchKind::PassiveLinear`
- **AND** `g.branches_of(0)` SHALL contain exactly the branch index
  `0`.

### Requirement: SwitchStateMask — bitmask of switch states

`pulsim::v2::topology::SwitchStateMask` SHALL be a fixed-width
bitmask (backed by `std::uint64_t` for up to 64 switches) that
represents the open/closed state of every switch in a circuit. It
MUST be cheaply hashable, comparable, and orderable so it can be
used as a `std::map` / `std::unordered_map` key by Layer 4's
state-space cache.

The class MUST provide:
- `SwitchStateMask(Size num_switches)` — initialises all-zero
  (all switches open). Throws `std::invalid_argument` if
  `num_switches > 64`.
- `bool get(Size i) const`, `void set(Size i, bool v)`,
  `void flip(Size i)`, `Size count() const`, `Size size() const`.
- `operator==`, `operator!=`, `operator<` (lexicographic).
- `std::size_t hash() const`.
- `std::hash<SwitchStateMask>` specialization in `namespace std`.
- `std::string to_string() const` for diagnostics.

#### Scenario: Default-constructed mask is all zeros

- **GIVEN** `SwitchStateMask mask(8)`
- **THEN** `mask.size()` SHALL be `8`
- **AND** `mask.count()` SHALL be `0`
- **AND** `mask.get(i)` SHALL be `false` for every `i ∈ [0, 8)`.

#### Scenario: set / get round-trip

- **GIVEN** `SwitchStateMask mask(8)`
- **WHEN** the user calls `mask.set(3, true)`
- **THEN** `mask.get(3)` SHALL be `true`
- **AND** `mask.count()` SHALL be `1`
- **AND** `mask.get(i)` for `i ≠ 3` SHALL remain `false`.

#### Scenario: Two masks with same bits and same size hash identically

- **GIVEN** two `SwitchStateMask` of size 8 with identical bit
  patterns
- **THEN** `m1 == m2` SHALL be `true`
- **AND** `m1.hash() == m2.hash()` SHALL be `true`
- **AND** `std::hash<SwitchStateMask>{}(m1) ==
  std::hash<SwitchStateMask>{}(m2)` SHALL be `true`.

#### Scenario: Constructor rejects N > 64

- **WHEN** the user calls `SwitchStateMask mask(65)`
- **THEN** the constructor SHALL throw `std::invalid_argument`
- **AND** the exception's `what()` SHALL contain "64".

### Requirement: Gray-Code Enumeration Over Switch States

`pulsim::v2::topology` SHALL expose `enumerate_switch_states(Size
num_switches)` returning a forward-iterable range that yields all
2^N distinct `SwitchStateMask` values for N switches. The iteration
order MUST be a **Gray code**: every consecutive pair of states
differs by exactly ONE bit. This order enables Layer 4 to apply
Sherman-Morrison rank-1 factor updates instead of full re-factor on
adjacent states.

The range MUST work in C++20 range-based for loops and MUST yield
exactly `2^N` distinct values.

#### Scenario: N = 0 yields one state (the empty mask)

- **WHEN** the user calls `enumerate_switch_states(0)`
- **THEN** iterating the result SHALL yield exactly 1 mask
- **AND** that mask SHALL have `size() == 0` and `count() == 0`.

#### Scenario: N = 3 yields exactly 8 distinct states

- **WHEN** the user iterates `enumerate_switch_states(3)`
- **THEN** the iteration SHALL yield exactly 8 distinct
  `SwitchStateMask` values
- **AND** collecting them into a `std::set<SwitchStateMask>` SHALL
  produce a set of size 8.

#### Scenario: Gray-code property — consecutive states differ by one bit

- **WHEN** the user iterates `enumerate_switch_states(4)`
- **AND** computes `popcount(prev XOR curr)` for every consecutive
  pair (prev, curr)
- **THEN** every such popcount SHALL be exactly `1`.

### Requirement: Node Equivalence Under Switch State

`pulsim::v2::topology::NodeEquivalence` SHALL compute, for a given
`Graph` and `SwitchStateMask`, the equivalence classes of nodes
under "closed switches short their endpoints". The implementation
MUST use path-compressed union-find with O(α(n)) amortised
operations.

The class MUST provide:
- Construction from `(const Graph&, const SwitchStateMask&)`.
- `Index representative_of(Index node) const`.
- `bool are_equivalent(Index a, Index b) const`.
- `Size num_classes() const`.
- `std::vector<Index> class_members(Index representative) const`.

The ground node SHALL always be its own representative; closed
switches touching ground promote the other endpoint into ground's
class.

#### Scenario: All switches open → every node is its own class

- **GIVEN** a `Graph` with 5 nodes and 3 switch branches
- **AND** a `SwitchStateMask` of size 3 with all bits zero
- **WHEN** the user constructs `NodeEquivalence eq(graph, mask)`
- **THEN** `eq.num_classes()` SHALL be `5`
- **AND** `eq.representative_of(i)` SHALL equal `i` for every
  node `i`.

#### Scenario: Closed switch shorts its two endpoints into one class

- **GIVEN** a 3-node graph with nodes `a, b, c` and ONE switch
  branch from `a` to `b`
- **WHEN** the user closes the switch (`mask.set(0, true)`) and
  constructs `NodeEquivalence eq(graph, mask)`
- **THEN** `eq.are_equivalent(a, b)` SHALL be `true`
- **AND** `eq.are_equivalent(a, c)` SHALL be `false`
- **AND** `eq.num_classes()` SHALL be `2`.

#### Scenario: Transitive shorts merge a chain into one class

- **GIVEN** a graph with nodes `{n1, n2, n3, gnd}` and switch
  branches `sw1: n1↔n2`, `sw2: n2↔n3`, `sw3: n3↔gnd`
- **WHEN** all 3 switches are closed
- **THEN** all 4 nodes SHALL belong to the same equivalence class
- **AND** `eq.representative_of(n1) == eq.representative_of(gnd)`.

### Requirement: TopologyKey for Layer 4 Cache Lookup

`pulsim::v2::topology::TopologyKey` SHALL be a value type that
combines a stable `Graph` identifier with a `SwitchStateMask`,
suitable for use as a key in `std::unordered_map`. Layer 4's PWL
state-space cache uses it to look up the pre-factorised state-space
matrices per (graph, switch state) tuple.

`Graph::id()` SHALL return a stable 64-bit identifier computed from
the graph's structural content (num_nodes, num_branches, and the
sequence of (from, to, kind) tuples). The identifier SHALL be cached
lazily — repeated calls cost O(1).

`TopologyKey` MUST provide `operator==`, `operator!=`, a member
`std::size_t hash() const`, and a `std::hash<TopologyKey>`
specialization in `namespace std`.

#### Scenario: Same graph, same state → same key

- **GIVEN** two pointers to the same `Graph` and two
  identical-value `SwitchStateMask`s
- **WHEN** the user constructs two `TopologyKey`s from them
- **THEN** the keys SHALL compare equal
- **AND** their hashes SHALL be identical.

#### Scenario: Different graph structure → different graph_id

- **GIVEN** two graphs `g1` and `g2` that differ in number of
  branches
- **THEN** `g1.id() != g2.id()` SHALL hold
- **AND** a `TopologyKey` built from each SHALL compare unequal
  even if both wrap the same `SwitchStateMask`.

#### Scenario: TopologyKey round-trips through `std::unordered_map`

- **GIVEN** a `std::unordered_map<TopologyKey, int>`
- **WHEN** the user inserts `(key1, 42)` then `find(key1)`
- **THEN** `find` SHALL return an iterator with `value == 42`.
