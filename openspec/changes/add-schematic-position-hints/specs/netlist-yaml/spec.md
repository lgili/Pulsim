## ADDED Requirements

### Requirement: Schematic Position Hints in Netlist YAML
The YAML netlist schema SHALL accept an optional ``position`` field per component, carrying either a ``{layer, slot}`` semantic placement or a ``{x, y}`` absolute placement, used as a hint by the schematic renderer.

#### Scenario: Semantic position hint accepted
- **WHEN** a component declares ``position: {layer: 0, slot: 0}``
- **THEN** the YAML parser stores the hint on the component descriptor
- **AND** does not raise an error for the new field

#### Scenario: Absolute position hint accepted
- **WHEN** a component declares ``position: {x: 100, y: 50}``
- **THEN** the YAML parser stores the absolute coordinates on the component descriptor

#### Scenario: Strict-mode rejection of unknown sub-keys
- **WHEN** ``position`` contains a key other than ``layer``, ``slot``, ``x``, or ``y`` (e.g. ``z``)
- **THEN** the strict YAML parser emits a deterministic diagnostic naming the unsupported key
- **AND** does not import the circuit

#### Scenario: No hint preserved as None
- **WHEN** a component omits ``position``
- **THEN** the component descriptor carries no hint
- **AND** the schematic renderer uses its default auto-layout for that component
