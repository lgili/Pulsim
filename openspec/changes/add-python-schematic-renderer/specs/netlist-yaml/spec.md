## ADDED Requirements

### Requirement: Component Position Hint Field
The `pulsim-v1` netlist schema SHALL accept an optional `position:` field on each entry of the top-level `components:` array, carrying either a semantic `(layer, slot)` pair or an absolute `(x, y)` pair in renderer coordinates. The field is consumed by the schematic renderer; it has no effect on simulation.

#### Scenario: Semantic (layer, slot) hint
- **GIVEN** a YAML netlist entry of the form
  ```yaml
  - type: voltage_source
    name: Vdc
    nodes: [vcc, 0]
    waveform: { type: dc, value: 48.0 }
    position: { layer: 0, slot: 0 }
  ```
- **WHEN** the netlist is loaded via `YamlParser.load(...)`
- **THEN** `Circuit.position_hint("Vdc")` returns a hint whose `layer == 0` and `slot == 0`

#### Scenario: Absolute (x, y) hint
- **GIVEN** a YAML entry with `position: { x: 200.0, y: 80.0 }`
- **WHEN** the netlist is loaded
- **THEN** `Circuit.position_hint(<name>)` returns a hint whose `x == 200.0` and `y == 80.0`
- **AND** the `layer` and `slot` attributes are unset

#### Scenario: Missing position field is not an error
- **GIVEN** a YAML netlist whose components carry no `position:` field
- **WHEN** the netlist is loaded and rendered
- **THEN** `Circuit.position_hints()` returns an empty dict
- **AND** rendering proceeds with the default (auto + topology-aware) layout

#### Scenario: Invalid position field is rejected at parse time
- **GIVEN** a YAML entry with `position: { layer: 0 }` (missing slot AND no x/y) or `position: { x: 0 }` (missing y AND no layer/slot)
- **WHEN** the netlist is loaded
- **THEN** the parser raises a deterministic typed error naming the component and the missing fields
- **AND** no partial Circuit is returned

#### Scenario: Position field is optional and round-trips
- **GIVEN** a YAML netlist with a `position:` hint on at least one component
- **WHEN** the netlist is loaded and the resulting `Circuit.position_hints()` is inspected
- **THEN** the hint values match the YAML byte-for-byte for the canonical (layer, slot) or (x, y) form used
