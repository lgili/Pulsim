## ADDED Requirements

### Requirement: DevicePool — Diode Registration

`pulsim::v2::pwl::DevicePool` SHALL support registration of
`IdealDiode` devices. Diodes are stored alongside the existing
Resistor / VoltageSource / Switch / Capacitor / Inductor
entries.

The class MUST extend with:

- `void add_diode(Index branch_id, Real g_on, Real g_off, Real V_th = 0.0)`
- New `StoredKind::Diode` enum value.
- `const models::IdealDiode::Params& diode_params(Index branch_id) const`
- `Size num_diodes() const noexcept`
- `std::span<const Index> diode_branches() const` — returns
  diode branch ids in branch order.

The diode's branch MUST be added to the graph with
`BranchKind::Switch` (the diode IS a switch from the topology's
perspective; the combinatorial space includes it).

#### Scenario: Pool with one diode reports counts correctly

- **GIVEN** an empty `DevicePool` and a Graph with one Switch-
  kind branch
- **WHEN** the user calls `pool.add_diode(0, g_on=1e3, g_off=
  1e-9, V_th=0.7)`
- **THEN** `pool.num_diodes()` SHALL be `1`
- **AND** `pool.diode_branches()` SHALL contain `[0]`
- **AND** `pool.diode_params(0).V_th` SHALL equal `0.7`.

#### Scenario: assemble_segment dispatches diodes using diode params

- **GIVEN** a graph with one branch registered as a diode with
  `g_on = 1e3`, `g_off = 1e-9`, V_th = 0.0
- **WHEN** the user calls `assemble_segment` with the mask bit
  CLEARED (diode OFF)
- **THEN** the matrix `J(from, from)` (where from is the active
  endpoint) SHALL include the diode's `g_off = 1e-9`
  contribution
- **AND** when reassembling with the mask bit SET (diode ON),
  `J(from, from)` SHALL include the diode's `g_on = 1e3`
  contribution.
