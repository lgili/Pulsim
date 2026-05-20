## ADDED Requirements

### Requirement: Component Position Hint Storage
The Circuit type SHALL expose a writable per-component position-hint registry that persists user-supplied placement intent (semantic `(layer, slot)` and/or absolute `(x, y)`) across the Circuit's lifetime. Hints are consumed by downstream rendering / GUI tooling and have no effect on simulation semantics.

#### Scenario: Setting and reading back a semantic hint
- **WHEN** code calls `circuit.set_position("R1", /*layer=*/0, /*slot=*/0, /*x=*/{}, /*y=*/{})` (or the equivalent named-argument form in the chosen impl)
- **THEN** a subsequent `circuit.position_hint("R1")` returns a populated `PositionHint` whose `layer == 0` and `slot == 0`
- **AND** the `x` and `y` optionals are empty

#### Scenario: Setting and reading back an absolute hint
- **WHEN** code calls `circuit.set_position("Cout", /*x=*/200.0, /*y=*/80.0)`
- **THEN** `circuit.position_hint("Cout")` returns a hint whose `x == 200.0` and `y == 80.0`
- **AND** the `layer` and `slot` optionals are empty

#### Scenario: Snapshot accessor is detached from mutation
- **GIVEN** `circuit.set_position("R1", layer=0, slot=0)` was called
- **WHEN** code captures `auto snap = circuit.position_hints();` and then calls `circuit.set_position("R1", layer=5, slot=5)`
- **THEN** `snap.at("R1").layer == 0` — the snapshot is independent of later writes

#### Scenario: Unhinted component returns nullopt
- **WHEN** code calls `circuit.position_hint("never_set")`
- **THEN** the return value is an empty `std::optional<PositionHint>`

#### Scenario: At least one coordinate must be set
- **WHEN** code calls `set_position` with all four coordinate fields unset
- **THEN** the call is rejected at the kernel boundary with a deterministic typed error
- **AND** the underlying registry is unchanged

#### Scenario: Determinism across runs
- **WHEN** the same Circuit is built twice with identical `set_position` calls in identical order
- **THEN** both `position_hints()` snapshots compare equal (same keys, same field values)
