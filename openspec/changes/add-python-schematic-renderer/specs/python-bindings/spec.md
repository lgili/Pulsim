## ADDED Requirements

### Requirement: Position Hint Binding
Python bindings SHALL expose `Circuit.set_position(name, *, layer=None, slot=None, x=None, y=None)`, `Circuit.position_hint(name) -> Optional[PositionHint]`, and `Circuit.position_hints() -> dict[str, PositionHint]` matching the kernel contract, with `PositionHint` exposed as a read-only Python class.

#### Scenario: Set and read back a semantic-grid hint
- **WHEN** Python code calls `circuit.set_position("Q1", layer=2, slot=1)`
- **THEN** `circuit.position_hint("Q1")` returns a `PositionHint` whose `layer == 2` and `slot == 1`
- **AND** `circuit.position_hints()["Q1"]` matches

#### Scenario: Set and read back an absolute hint
- **WHEN** Python code calls `circuit.set_position("Cout", x=120.0, y=40.0)`
- **THEN** `circuit.position_hint("Cout")` returns a `PositionHint` whose `x == 120.0` and `y == 40.0`
- **AND** the `layer` and `slot` attributes are `None`

#### Scenario: Hint snapshot is detached from later mutation
- **GIVEN** `circuit.set_position("R1", layer=0, slot=0)` has been called
- **WHEN** Python code calls `hints = circuit.position_hints()` and then `circuit.set_position("R1", layer=5, slot=5)`
- **THEN** `hints["R1"].layer` is still `0` — the snapshot is independent

#### Scenario: Querying an unhinted component
- **WHEN** Python code calls `circuit.position_hint("never_pinned")`
- **THEN** the return value is `None`

#### Scenario: PositionHint is read-only
- **GIVEN** `h = circuit.position_hint("Q1")` returned a `PositionHint`
- **WHEN** Python code attempts `h.layer = 99`
- **THEN** `AttributeError` is raised

### Requirement: Default Backend Selection
Python bindings SHALL surface a default `pulsim.schematic.render(...)` that dispatches to the native Python backend unless `PULSIM_SCHEMATIC_BACKEND` is set to a non-default value.

#### Scenario: Default backend produces SVG without invoking netlistsvg
- **GIVEN** `PULSIM_SCHEMATIC_BACKEND` is unset (or set to the canonical native-backend value)
- **WHEN** `ps.schematic.render(circuit, "out.svg")` runs
- **THEN** no `netlistsvg` subprocess is invoked (verifiable by removing `node_modules/netlistsvg` and observing the call still succeeds, provided `elkjs` is present)

#### Scenario: Legacy backend warning
- **GIVEN** `PULSIM_SCHEMATIC_BACKEND=netlistsvg`
- **WHEN** `ps.schematic.render(...)` runs
- **THEN** a `DeprecationWarning` is emitted whose message includes the planned removal release

## MODIFIED Requirements

### Requirement: Schematic Rendering Surface
Python bindings SHALL provide a `pulsim.schematic` module that renders a Circuit to an image file and produces a JSON-serializable layout for GUI consumers. The module's default backend is the native Python renderer introduced by `add-python-schematic-renderer`; `compute_layout(circuit).to_json()` returns the same `schematic-v1` schema as before but with position-hint information layered into the placement step.

#### Scenario: One-shot render to PNG
- **WHEN** Python code calls `pulsim.schematic.render(circuit, "out.png")` on a circuit built via the public Circuit API
- **THEN** the call writes a non-empty PNG file
- **AND** the call raises no exception when the `[schematic]` extra is installed

#### Scenario: One-shot render to SVG
- **WHEN** the destination path ends in `.svg`
- **THEN** the call writes a well-formed SVG file parseable by standard XML libraries

#### Scenario: Layout for GUI auto-place reflects hints
- **GIVEN** a Circuit with one or more `set_position` calls
- **WHEN** Python code calls `pulsim.schematic.compute_layout(circuit).to_json()`
- **THEN** the returned dict's `components[name]` entries for hinted components carry the user's resolved `(x, y)` (in mm, origin top-left)
- **AND** `schema_version` remains `"schematic-v1"`

#### Scenario: Missing optional dependency at import time
- **WHEN** Python imports `pulsim.schematic` without `numpy` installed
- **THEN** the import succeeds without raising
- **AND** `pulsim.schematic.render(...)` raises a clear `ImportError` only when called

#### Scenario: Missing optional dependency at render time
- **WHEN** `pulsim.schematic.render(...)` is called and a required dependency for the chosen backend is missing (e.g. `node` for the native backend's layout step)
- **THEN** the call raises `ImportError` (or `RuntimeError` for non-Python dependencies) with a message naming the missing tool and the install command
