## ADDED Requirements

### Requirement: DevicePool — Per-Branch Parameters Registry

`pulsim::v2::pwl::DevicePool` SHALL be a heterogeneous registry
mapping branch ids to (kind, parameters) tuples. Layer 1's `Graph`
stores only `BranchKind`; the pool stores the parameters needed by
Layer 3 stamping.

The class MUST support three add methods (Layer 4 V0 scope):
- `void add_resistor(Index branch_id, Resistor::Params)`
- `void add_voltage_source(Index branch_id, VoltageSource::Params)`
- `void add_switch(Index branch_id, Real g_on, Real g_off)`

A `StoredKind` enum (Resistor, VoltageSource, Switch) and per-kind
getters MUST be exposed:
- `StoredKind kind_of(Index branch_id) const` — throws
  `std::out_of_range` if the branch isn't registered.
- Per-kind parameter accessors (`resistor_params`, `voltage_source_params`,
  `switch_g_on`, `switch_g_off`) — each throws `std::out_of_range`
  if the branch isn't registered as that specific kind.

State-vector layout helpers:
- `Index branch_var_id_for_source(Index branch_id) const` —
  returns the absolute state-vector index of the branch-current
  unknown for a Source-kind branch.
- `Size num_voltage_sources() const`
- `Size state_size(const Graph& graph) const` — returns
  `graph.num_nodes() + num_voltage_sources()`.

#### Scenario: Empty pool reports zero state_size

- **GIVEN** an empty `DevicePool` and an empty `Graph`
- **WHEN** the user calls `pool.state_size(graph)`
- **THEN** the result SHALL be `0`.

#### Scenario: Adding a resistor leaves num_voltage_sources at zero

- **GIVEN** a `Graph` with 2 nodes and an empty `DevicePool`
- **WHEN** the user adds a resistor on branch 0
- **THEN** `pool.state_size(graph)` SHALL equal `2` (just node
  voltages — no branch unknowns added).

#### Scenario: Adding voltage sources adds branch-current unknowns

- **GIVEN** a `Graph` with 2 nodes
- **WHEN** the user adds 2 voltage sources + 1 resistor
- **THEN** `pool.state_size(graph)` SHALL equal `4`
- **AND** the branch_var_id of the first source SHALL equal `2`
- **AND** the branch_var_id of the second source SHALL equal `3`.

#### Scenario: Wrong-kind lookup throws

- **GIVEN** a `DevicePool` with branch 0 added as a resistor
- **WHEN** the user calls `pool.voltage_source_params(0)`
- **THEN** the call SHALL throw `std::out_of_range`.

### Requirement: PwlSegment — Per-State Cached Record

`pulsim::v2::pwl::PwlSegment` SHALL be a move-only aggregate
holding the per-switch-state record:

```cpp
struct PwlSegment {
    sparse::Matrix J;                                  // assembled MNA matrix
    Vector b_constant;                                  // constant RHS
    std::unique_ptr<sparse::DirectSolver> solver;      // pre-factorized
    Size state_size = 0;
};
```

The structure MUST be move-constructible and move-assignable, MUST
NOT be copy-constructible (the `unique_ptr<DirectSolver>` member
makes copy nonsensical).

#### Scenario: PwlSegment is move-only

- **WHEN** the consumer evaluates static traits on `PwlSegment`
- **THEN** `std::is_move_constructible_v<PwlSegment>` SHALL be `true`
- **AND** `std::is_move_assignable_v<PwlSegment>` SHALL be `true`
- **AND** `std::is_copy_constructible_v<PwlSegment>` SHALL be `false`.

### Requirement: assemble_segment — Builds One Segment Matrix

`pulsim::v2::pwl::assemble_segment(graph, pool, mask, J, b)` SHALL
build the MNA matrix `J` and constant RHS `b` for a single switch
state. The function MUST:

1. Zero `J` and `b` (set sizes to `pool.state_size(graph)` if not
   already).
2. Iterate every branch in the graph in `branch_id` order.
3. Dispatch by `BranchKind`:
   - `PassiveLinear` → look up `Resistor::Params` via the pool,
     call `stamping::stamp_device<Resistor>` with the branch
     coordinate.
   - `Source` → look up params + branch_var_id, call
     `stamping::stamp_voltage_source`.
   - `Switch` → look up params, advance a per-switch counter,
     read the bit from `mask`, call
     `stamping::stamp_switch_fixed`.
   - `Nonlinear` → SKIPPED in V0 (matches the static-only scope).
4. Use a zero state vector during stamping — V0 devices are linear
   so their stamp contribution doesn't depend on `x`.

#### Scenario: Empty graph assembles to empty matrices

- **GIVEN** an empty `Graph` and an empty `DevicePool`
- **WHEN** the user calls `assemble_segment` with an empty mask
- **THEN** `J.rows()` SHALL be `0` and `b.size()` SHALL be `0`.

#### Scenario: Single Resistor stamps onto the right diagonal

- **GIVEN** a `Graph` with one node + one resistor branch from
  node 0 to ground, `DevicePool` with `G = 2.0`, empty mask
- **WHEN** the user assembles
- **THEN** `J(0, 0)` SHALL equal `+2.0`
- **AND** `b` SHALL be all zeros.

#### Scenario: Single VoltageSource adds the constraint row

- **GIVEN** a graph with one node + one voltage source from
  node 0 to ground, `DevicePool` with `V = 12.0`
- **WHEN** the user assembles
- **THEN** the state size SHALL be `2` (node 0 + branch current)
- **AND** `J(0, 1)` SHALL equal `+1` (KCL on node 0)
- **AND** `J(1, 0)` SHALL equal `+1` (constraint row)
- **AND** `b(1)` SHALL equal `-12.0`.

#### Scenario: Switch state changes the stamped conductance

- **GIVEN** a graph with one switch from node 0 to ground,
  `g_on = 1e3`, `g_off = 1e-9`
- **WHEN** the user assembles with `mask.get(0) == false`
- **THEN** `J(0, 0)` SHALL equal `+1e-9`
- **AND** when reassembling with `mask.get(0) == true`,
  `J(0, 0)` SHALL equal `+1e3`.

### Requirement: PwlStateSpaceCache — Build + Lookup + Solve

`pulsim::v2::pwl::PwlStateSpaceCache` SHALL be the main Layer 4
class. It MUST:

1. Take a `const Graph&` and `const DevicePool&` at construction
   (borrows both; caller manages lifetime).
2. `build()` enumerates all `SwitchStateMask` values via Layer 1's
   `enumerate_switch_states(graph.num_switches())` and, for each:
   - Calls `assemble_segment` to produce (J, b).
   - Calls `sparse::compress_in_place(J)`.
   - Creates a solver via `make_default_solver()`.
   - Calls `solver->analyze(J)` and `solver->factorize(J)`.
   - Stores the `PwlSegment{J, b, solver, state_size}` keyed by
     the mask.
3. `lookup(const SwitchStateMask&) const -> const PwlSegment&`
   returns the cached segment in O(1) (unordered_map probe).
   Throws `std::out_of_range` if the mask wasn't built.
4. `solve(const SwitchStateMask&, const Vector& b_extra, Vector& x)
   const` is the hot-loop entry point. It MUST:
   - Look up the segment.
   - Compute `b = seg.b_constant + b_extra`.
   - Call `seg.solver->solve(b, x)` and return.

5. `num_segments() const -> Size` returns the count of built
   segments for diagnostic purposes.

#### Scenario: N=0 switches → exactly 1 segment

- **GIVEN** a graph with no Switch-kind branches
- **WHEN** the user constructs `PwlStateSpaceCache` + `build()`
- **THEN** `cache.num_segments()` SHALL be `1`.

#### Scenario: N=4 switches → exactly 16 segments

- **GIVEN** a graph with 4 Switch-kind branches
- **WHEN** the user constructs + builds the cache
- **THEN** `cache.num_segments()` SHALL be `16`.

#### Scenario: lookup for a never-built mask throws

- **GIVEN** a cache built for a 2-switch graph
- **AND** a `SwitchStateMask` of a different size
- **WHEN** the user calls `cache.lookup(mask)`
- **THEN** the call SHALL throw `std::out_of_range`.

#### Scenario: solve on V-R-GND with switch ON returns ≈ V_dc

- **GIVEN** a graph with V_dc + switch + R + GND, cache built
- **AND** `SwitchStateMask` with the switch bit ON
- **WHEN** the user calls `cache.solve(mask, b_extra=0, x)`
- **THEN** `x[v_out_index]` SHALL be ≥ 99.99 % of `V_dc` (the
  switch closed forms a small resistive divider with
  `g_on = 1e3` vs `G_R = 0.1`).
