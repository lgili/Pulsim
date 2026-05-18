## ADDED Requirements

### Requirement: Single-File Schematic Render
The schematic module SHALL render a Circuit to a single SVG or PNG file with all components placed and connected by drawn wires.

#### Scenario: Render a buck converter YAML
- **WHEN** the user loads `examples/buck_converter.yaml` and calls `pulsim.schematic.render(circuit, "buck.svg")`
- **THEN** the resulting SVG contains a graphic element for each component in `circuit.components()`
- **AND** wires connect terminal anchors that share a node ID

#### Scenario: PNG output when extension is .png
- **WHEN** the destination path ends in `.png`
- **THEN** the output is a valid PNG file whose first bytes match the PNG signature

#### Scenario: Format override
- **WHEN** the user passes `format="svg"` to `render(circuit, "out.bin", format="svg")`
- **THEN** the file is written as SVG regardless of extension

### Requirement: Force-Directed Layout with Electrical Priors
The layout engine SHALL place nodes using a deterministic force-directed algorithm with electrical priors: ground at the bottom, voltage sources at the left edge, resistive loads at the right edge.

#### Scenario: Ground appears below other nodes
- **WHEN** a layout is computed for any non-empty circuit that has a ground node
- **THEN** the ground node's `y` coordinate is greater than or equal to the `y` coordinate of every other placed node

#### Scenario: Voltage source positive terminal on the left
- **WHEN** the circuit contains at least one voltage source
- **THEN** the positive terminal of that voltage source has `x ≤ canvas.width / 4`

#### Scenario: Layout determinism
- **WHEN** the same circuit is laid out twice via two separate `compute_layout(circuit)` calls
- **THEN** every component placement and wire path is byte-identical between the two results

### Requirement: JSON-Serializable Layout
The `SchematicLayout` type SHALL serialize to and deserialize from JSON without loss of placement information.

#### Scenario: Round-trip a layout
- **WHEN** code calls `SchematicLayout.from_json(layout.to_json())`
- **THEN** the deserialized layout equals the original in components, wires, junctions, and canvas

#### Scenario: Reject unknown schema version
- **WHEN** `from_json()` receives a payload with `schema_version` other than `"schematic-v1"`
- **THEN** the call raises `ValueError` naming the unsupported version

### Requirement: GUI-Consumable Coordinate Schema
The layout coordinate schema SHALL be stable and documented for GUI consumers.

#### Scenario: Coordinate units and origin
- **WHEN** a GUI reads `layout.canvas`
- **THEN** the `unit` field is the string `"mm"`
- **AND** coordinates use the convention origin top-left, +x right, +y down

#### Scenario: Stable schema version
- **WHEN** any layout JSON is produced
- **THEN** the top-level `schema_version` field is the string `"schematic-v1"`

### Requirement: Optional Dependency Surface
The schematic module SHALL be importable without `schemdraw` or `networkx` installed but SHALL raise a clear error at render or layout time if either is missing.

#### Scenario: Import without optional dependency
- **WHEN** Python imports `pulsim.schematic` without `schemdraw` installed
- **THEN** the import succeeds and no `ImportError` is raised

#### Scenario: Render without optional dependency
- **WHEN** `pulsim.schematic.render(...)` is called and `schemdraw` is missing
- **THEN** an `ImportError` is raised with text including `pip install pulsim[schematic]`

#### Scenario: compute_layout without optional dependency
- **WHEN** `pulsim.schematic.compute_layout(...)` is called and `networkx` is missing
- **THEN** an `ImportError` is raised with text including `pip install pulsim[schematic]`

### Requirement: Graceful Handling of Unknown Component Kinds
The renderer SHALL not crash when it encounters a component kind that is not in the symbol mapping.

#### Scenario: Unmapped component kind
- **WHEN** a circuit contains a component whose `kind` is not present in the symbol library
- **THEN** the renderer draws a labeled rectangle in its place
- **AND** the renderer emits a `UserWarning` naming the unknown kind
