## ADDED Requirements

### Requirement: DevicePool — Dynamic Device Registration

`pulsim::v2::pwl::DevicePool` SHALL support registration of
`Capacitor` and `Inductor` devices alongside the existing
Resistor / VoltageSource / Switch types.

The class MUST extend the existing surface with:

- `void add_capacitor(Index branch_id, models::Capacitor::Params)`
- `void add_inductor(Index branch_id, models::Inductor::Params)`
- Two new `StoredKind` enum values: `Capacitor`, `Inductor`.
- `const models::Capacitor::Params& capacitor_params(Index) const`
- `const models::Inductor::Params& inductor_params(Index) const`
- `Index branch_var_id_for_inductor(Index, const Graph&) const`
  — returns the absolute state-vector index for an inductor's
  branch-current unknown.
- `Size num_inductors() const noexcept`.
- `Size num_dynamic_branches() const noexcept` — count of
  Capacitor + Inductor registrations.

`state_size(graph)` MUST return `num_nodes + num_voltage_sources
+ num_inductors`. Capacitors do NOT add branch-current unknowns
(their behaviour is captured purely on node-voltage rows via
the companion conductance).

#### Scenario: Pool with one capacitor preserves state_size

- **GIVEN** a `Graph` with 1 node and `DevicePool` containing
  one capacitor on branch 0
- **WHEN** the user calls `pool.state_size(graph)`
- **THEN** the result SHALL equal `1` (just the node voltage —
  no extra branch unknown).

#### Scenario: Pool with one inductor adds a branch-current unknown

- **GIVEN** a `Graph` with 2 nodes and `DevicePool` containing
  one inductor between them
- **WHEN** the user calls `pool.state_size(graph)`
- **THEN** the result SHALL equal `3` (2 node voltages +
  1 inductor branch current).

#### Scenario: branch_var_id_for_inductor returns the right offset

- **GIVEN** a `Graph` with 2 nodes, `DevicePool` with 1 voltage
  source (branch 0) + 1 inductor (branch 1)
- **WHEN** the user calls
  `pool.branch_var_id_for_inductor(1, graph)`
- **THEN** the result SHALL equal `3` (= num_nodes(2) +
  num_voltage_sources(1)).

### Requirement: PwlStateSpaceCache — dt-Aware Build

`pulsim::v2::pwl::PwlStateSpaceCache::build(Real dt)` SHALL be
the dt-aware build entry point. It MUST:

1. Store `dt` internally.
2. For every switch mask enumerated by Layer 1, call
   `assemble_segment(graph, pool, mask, dt, J, b)` so dynamic
   devices (Capacitor, Inductor) stamp their companion
   conductances via the dt-dependent `g_eq`.
3. Factorise each segment as in V0.

A no-arg `build()` overload SHALL remain available for backwards
compatibility. It MUST behave as `build(Real{0})`, which causes
`assemble_segment` to SKIP dynamic devices (the cache is then
valid only for static circuits).

The cache MUST expose `[[nodiscard]] Real dt() const noexcept`.

Calling `build(dt)` a second time with a different dt SHALL
clear the segments map and rebuild every factor.

#### Scenario: Static-only build is unchanged

- **GIVEN** a graph with Resistor + VoltageSource + Switch
  (no caps / inductors) and a `DevicePool`
- **WHEN** the user calls `cache.build()` (no dt)
- **THEN** the resulting segments SHALL be bit-identical to a
  V0 build (every Layer 4 V0 test passes unchanged).

#### Scenario: build(dt) stamps the cap's g_eq

- **GIVEN** a 1-node graph with a 1 µF capacitor to ground
- **WHEN** the user calls `cache.build(dt = 1e-6)`
- **THEN** the cached segment's `J(0, 0)` SHALL equal `2.0`
  (g_eq = 2C/dt = 2·1e-6/1e-6).
- **AND** `cache.dt()` SHALL return `1e-6`.

#### Scenario: Rebuild on dt change

- **GIVEN** a cache built at `dt = 1e-6`
- **WHEN** the user calls `cache.build(dt = 2e-6)`
- **THEN** `cache.dt()` SHALL return `2e-6`
- **AND** the cached `J(0, 0)` for the same capacitor SHALL
  now equal `1.0` (= 2C / 2µs).

### Requirement: assemble_segment — Dynamic Device Dispatch

`pulsim::v2::pwl::assemble_segment` SHALL accept a `Real dt`
parameter and dispatch `PassiveLinear` branches by their
`DevicePool::StoredKind`:

- `Resistor`  → existing `stamp_device<Resistor>` path.
- `Capacitor` → `stamp_capacitor_companion(...)` using
                `g_eq = Capacitor::g_eq(dt, params)`. History
                contribution is 0 (history flows via the
                solver's `b_extra`, not the assembled
                `b_constant`).
- `Inductor`  → `stamp_inductor_companion(...)` using
                `g_eq_inv = Inductor::g_eq_inv(dt, params)`.

When `dt == 0`, Capacitor and Inductor branches MUST be SKIPPED
(no stamps emitted). This preserves V0 static-only behaviour
for callers using `cache.build()` without a dt.

#### Scenario: dt = 0 skips capacitor stamping

- **GIVEN** a graph with one capacitor and `dt = 0`
- **WHEN** the user calls `assemble_segment`
- **THEN** `J` SHALL contain NO entries for the capacitor's
  branch coordinates.

#### Scenario: Capacitor companion stamp contributes only to b_constant=0

- **GIVEN** a graph with a 1 µF capacitor to ground, `dt = 1µs`
- **WHEN** the user calls `assemble_segment`
- **THEN** `J(0, 0)` SHALL equal `2.0`
- **AND** `b(0)` SHALL equal `0` (history is added at solve
  time via b_extra, not at assembly time).
