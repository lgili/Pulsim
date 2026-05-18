## ADDED Requirements

### Requirement: Component Introspection Binding
Python bindings SHALL expose `Circuit.components()` and `Circuit.node_position_hint(node_id)` matching the kernel contract, with descriptor fields exposed as read-only Python attributes.

#### Scenario: components() from Python
- **WHEN** Python code calls `circuit.components()`
- **THEN** the call returns a list of `ComponentDescriptor` objects
- **AND** each object exposes read-only `name` (str), `kind` (str), `nodes` (list[int]), and `params` (dict[str, float]) properties

#### Scenario: node_position_hint() from Python
- **WHEN** Python code calls `circuit.node_position_hint(node_id)`
- **THEN** the call returns `None` for an unclassified node, or a string in `{"ground", "source_pos", "source_neg", "load", "internal"}`

### Requirement: Schematic Rendering Surface
Python bindings SHALL provide a `pulsim.schematic` module that renders a Circuit to an image file and produces a JSON-serializable layout for GUI consumers.

#### Scenario: One-shot render to PNG
- **WHEN** Python code calls `pulsim.schematic.render(circuit, "out.png")` on a circuit built via the public Circuit API
- **THEN** the call writes a non-empty PNG file
- **AND** the call raises no exception when the `[schematic]` extra is installed

#### Scenario: One-shot render to SVG
- **WHEN** the destination path ends in `.svg`
- **THEN** the call writes a well-formed SVG file parseable by standard XML libraries

#### Scenario: Layout for GUI auto-place
- **WHEN** Python code calls `pulsim.schematic.compute_layout(circuit).to_json()`
- **THEN** the call returns a dict with keys `components`, `wires`, `junctions`, `canvas`, and `schema_version`
- **AND** every numeric coordinate is a finite float
- **AND** `schema_version` equals the string `"schematic-v1"`

#### Scenario: Missing optional dependency at import time
- **WHEN** Python imports `pulsim.schematic` without `schemdraw` or `networkx` installed
- **THEN** the import succeeds without raising

#### Scenario: Missing optional dependency at render time
- **WHEN** `pulsim.schematic.render(...)` is called and `schemdraw` is not installed
- **THEN** the call raises `ImportError` with a message containing the install instruction `pip install pulsim[schematic]`
