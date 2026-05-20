## ADDED Requirements

### Requirement: Native Python Rendering Backend
The `pulsim.schematic` module SHALL provide a pure-Python rendering backend (no Node-side SVG composition) that consumes the same Yosys-style cell/net JSON the legacy `netlistsvg` backend builds and produces equivalent SVG output by parsing the shipped analog skin file and composing the output document in Python.

#### Scenario: Render the canonical RC circuit with the native backend
- **GIVEN** an `RC` circuit (one voltage source, one resistor, one capacitor, all referenced to ground)
- **WHEN** `pulsim.schematic.render(circuit, "rc.svg")` runs with `PULSIM_SCHEMATIC_BACKEND` unset or set to the native value
- **THEN** the resulting SVG parses cleanly via `xml.etree.ElementTree`
- **AND** contains one symbol per component (voltage-source circle, resistor zig-zag, capacitor parallel lines)
- **AND** contains one ground rail symbol per node connected to ground
- **AND** contains at least one orthogonal `<path>` for each net edge

#### Scenario: Render a buck converter with the native backend
- **GIVEN** the buck converter built in `scripts/render_boost_pfc.py`-style code (Vdc, vcswitch, freewheel diode, inductor, output cap, load resistor, PWM gate source)
- **WHEN** the same render call runs with the native backend
- **THEN** the rendered SVG contains a MOSFET / vcswitch symbol from the Pulsim analog skin (not the generic fallback rectangle)
- **AND** wires connect the gate driver to the switch's control terminal

#### Scenario: Render with no Node-side rendering subprocess
- **GIVEN** Node.js is installed only to support the layout step (`elk_bridge.js`)
- **WHEN** the native backend renders any circuit
- **THEN** the only subprocess invoked is `node elk_bridge.js` (or its equivalent layout call) — NO `netlistsvg` binary is invoked
- **AND** removing `node_modules/netlistsvg/` from the working tree does not break the render

### Requirement: Skin SVG Parser
The native rendering backend SHALL parse the Pulsim analog skin SVG file once per process and cache the resulting symbol templates keyed by canonical kind (with `<s:alias>` indirections resolved).

#### Scenario: Skin templates cover every documented kind
- **WHEN** the parser loads `pulsim_analog.svg`
- **THEN** the resulting template dict contains keys for at least: `resistor`, `capacitor`, `inductor`, `voltage_source`, `current_source`, `diode`, `mosfet`, `igbt`, `vcswitch`, `transformer`, `ground`, `generic`
- **AND** every alias defined in the skin resolves to its primary template

#### Scenario: Skin parser exposes port anchors
- **GIVEN** a `mosfet_n` symbol in the skin with `<g s:x="X" s:y="Y" s:pid="G|D|S">` children
- **WHEN** the parser produces a `SymbolTemplate` for `mosfet`
- **THEN** the template's `ports` dict maps `"G"`, `"D"`, `"S"` to their `(x, y)` anchor coordinates as floats

#### Scenario: Custom skin path
- **GIVEN** the environment variable `PULSIM_SCHEMATIC_SKIN` set to a user-provided skin path
- **WHEN** the next render call runs
- **THEN** the parser loads the user skin instead of the built-in one
- **AND** caches separately from the default

### Requirement: Position Hint Rendering
The native rendering backend SHALL accept per-component position hints, resolve them to absolute layout coordinates, and forward them to the layout engine as position constraints so that wires re-route consistently with the moved cells.

#### Scenario: User-pinned components land at the requested position
- **GIVEN** a buck circuit with `circuit.set_position("Vdc", layer=0, slot=0)` and `circuit.set_position("S1", layer=1, slot=0)`
- **WHEN** the circuit is rendered via the native backend
- **THEN** the `Vdc` cell in the resulting SVG sits at `(0 * LAYER_PX, 0 * SLOT_PX)` within ±5 px
- **AND** the `S1` cell sits at `(1 * LAYER_PX, 0 * SLOT_PX)` within ±5 px
- **AND** the wires connecting `Vdc` and `S1` are orthogonal segments terminating at the symbols' port anchors

#### Scenario: Absolute coordinate hints
- **GIVEN** `circuit.set_position("Cout", x=200.0, y=80.0)` (no layer/slot)
- **WHEN** the circuit is rendered
- **THEN** the `Cout` cell sits at `(200, 80)` within ±5 px in the output SVG's coordinate space

#### Scenario: No-hints output matches Phase 1 baseline
- **GIVEN** a circuit with no `set_position` calls and no YAML `position:` entries
- **WHEN** the native backend renders it
- **THEN** the output is byte-equivalent to the same circuit rendered with the same backend in the Phase 1 baseline release (i.e. adding position-hint support did not change the un-hinted layout)

### Requirement: Topology-Aware Auto-Hints
The native rendering backend SHALL run the existing template recognizers (`bridge_rectifier`, `half_bridge`, `boost_stage`) before layout and emit default position hints for the recognized components when the user has not explicitly hinted them.

#### Scenario: Buck converter renders with switch and diode in canonical columns
- **GIVEN** a buck converter built programmatically with no explicit position hints
- **WHEN** the native backend renders it
- **THEN** the recognized `boost_stage` (or equivalent buck pattern) hint set is applied
- **AND** the switching device (`S1` / `Q1`) sits in column 1 and the freewheel diode (`D1`) sits in column 2 (or the convention documented in the design's auto-layout table)
- **AND** the user's `Circuit.position_hints()` snapshot is unchanged — auto-hints are not persisted on the Circuit

#### Scenario: User hint overrides auto-hint
- **GIVEN** a buck circuit where the user calls `set_position("S1", layer=5, slot=5)` (an arbitrary non-canonical spot)
- **WHEN** the native backend renders it
- **THEN** `S1` sits at `(5 * LAYER_PX, 5 * SLOT_PX)` — the user's explicit hint wins over the recognizer's default

#### Scenario: Unrecognized topology falls back to free ELK layout
- **GIVEN** a circuit with no recognized topology (e.g. an isolated RC filter with no switching devices)
- **WHEN** the native backend renders it
- **THEN** no auto-hints are emitted
- **AND** the result is identical to the no-hints baseline

## MODIFIED Requirements

### Requirement: Single-File Schematic Render
The `pulsim.schematic` module SHALL expose a top-level `render(circuit, output_path, *, format=None)` callable that writes a single SVG or PNG file describing the circuit and returns the resolved `pathlib.Path`. By default the call dispatches to the native Python backend; the netlistsvg backend remains selectable for one release via `PULSIM_SCHEMATIC_BACKEND=netlistsvg` and emits a `DeprecationWarning` on use.

#### Scenario: Default backend writes a non-empty SVG
- **WHEN** `pulsim.schematic.render(circuit, "out.svg")` runs with the default backend
- **THEN** the file at `out.svg` exists and has size > 0
- **AND** parses cleanly via `xml.etree.ElementTree`

#### Scenario: Legacy netlistsvg backend opt-in
- **GIVEN** `PULSIM_SCHEMATIC_BACKEND=netlistsvg`
- **WHEN** a render call runs
- **THEN** a `DeprecationWarning` is emitted whose message names the removal release
- **AND** the call still succeeds and writes a valid SVG

#### Scenario: PNG inference from extension
- **WHEN** the destination path ends in `.png`
- **THEN** the call produces a valid PNG via `rsvg-convert` or `cairosvg`
- **AND** the call raises a clear `ImportError` if neither rasterizer is available

## REMOVED Requirements

### Requirement: Force-Directed Layout with Electrical Priors

**Reason:** This requirement described the original `spring` backend that placed components using a force-directed graph layout with electrical-domain priors (ground south, sources west). It remains accurate for the `spring` legacy backend, but it does NOT describe the default render path that `add-python-schematic-renderer` introduces. To avoid implying the default render uses force-directed layout, the requirement is moved out of the schematic-rendering spec — the legacy backend's behavior is documented in code-only as of this change.

**Migration:** Users / GUI consumers who relied on `spring`-backend layout for tests can still select it via `PULSIM_SCHEMATIC_BACKEND=spring`. The behavior is unchanged; only the spec text moves.
