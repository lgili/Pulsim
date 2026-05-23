# Schematic Rendering — V2

## ADDED Requirements

### Requirement: Topology Recognition (Deterministic Tier)
The schematic module SHALL recognize at least the following twelve
canonical power-electronics topologies from the circuit's component
graph alone, without any user input or network access:
`buck`, `boost`, `buck_boost`, `flyback`, `forward`, `half_bridge`,
`full_bridge`, `rc_filter`, `rl_filter`, `rlc_filter`,
`half_wave_rectifier`, `full_wave_bridge_rectifier`.

#### Scenario: Recognize a buck converter
- **WHEN** a circuit consists of one voltage source, one switch, one
  freewheel diode, one inductor, one output capacitor, and one load
  resistor wired in the canonical buck configuration
- **THEN** `recognize(circuit)` returns a `RecognizedTopology` whose
  `name == "buck"` and `confidence >= 0.9`

#### Scenario: Recognize a boost converter
- **WHEN** a circuit consists of one voltage source, one inductor, one
  low-side switch, one diode, one output capacitor, and one load
  resistor wired in the canonical boost configuration
- **THEN** `recognize(circuit)` returns a `RecognizedTopology` whose
  `name == "boost"` and `confidence >= 0.9`

#### Scenario: Reject a different topology
- **WHEN** a buck-shaped circuit is fed to `detect_boost`
- **THEN** `detect_boost(...)` returns `confidence < 0.5`

#### Scenario: Return None on no match
- **WHEN** an unrelated circuit (e.g., a random graph of 8 resistors)
  is fed to `recognize()`
- **THEN** the function returns `None`

### Requirement: LLM-Augmented Classifier (Optional Tier)
The schematic module SHALL invoke an Anthropic LLM classifier as a
second-tier topology recognizer when the deterministic tier returns
`confidence < 0.7` OR returns `None`, provided the environment
variable `PULSIM_LLM_LAYOUT_HINTS` is not set to `0` AND
`ANTHROPIC_API_KEY` is present.

#### Scenario: LLM fills in a custom topology
- **WHEN** a non-canonical circuit (e.g., a buck with extra snubber)
  is passed to `compute_layout()`
- **AND** the deterministic recognizer returns `confidence < 0.7`
- **AND** `ANTHROPIC_API_KEY` is set
- **THEN** the LLM classifier is invoked
- **AND** its `RecognizedTopology.source == "llm"`

#### Scenario: Disable via env var
- **WHEN** `PULSIM_LLM_LAYOUT_HINTS=0` is in the environment
- **THEN** the LLM classifier is NEVER invoked, regardless of
  deterministic confidence
- **AND** the renderer falls through to force-directed layout

#### Scenario: Cache hits skip the API
- **WHEN** the same circuit is rendered twice in the same process
  (or in different processes sharing the same cache directory)
- **THEN** the second render reads the topology from the local cache
- **AND** does NOT invoke the Anthropic API
- **AND** the returned `RecognizedTopology.source == "cache"`

#### Scenario: Graceful failure on missing key
- **WHEN** `PULSIM_LLM_LAYOUT_HINTS` is unset (default-on)
- **AND** `ANTHROPIC_API_KEY` is NOT set
- **THEN** the LLM classifier returns `None` without raising
- **AND** the renderer falls through to force-directed layout

#### Scenario: Graceful failure on network error
- **WHEN** the Anthropic API call raises any exception
- **THEN** the LLM classifier returns `None` without propagating
- **AND** the renderer falls through to force-directed layout

#### Scenario: Cache schema version validation
- **WHEN** the cache file contains a `schema_version` that does NOT
  match `"topology-cache-v1"`
- **THEN** the cache is discarded and rebuilt on the next call

### Requirement: Topology Templates
The schematic module SHALL ship with one template per recognized
topology under `python/pulsim/schematic/templates/<name>.yaml`. The
templates for `buck`, `boost`, and `flyback` SHALL be visually polished
and golden-tested.

#### Scenario: Template instantiation places components by role
- **WHEN** `compute_layout()` is called on a buck circuit
- **AND** the recognizer returns `topology="buck"` with a complete
  `role_map`
- **THEN** the resulting `SchematicLayout` places the source on the
  WEST edge, the switch on the NORTH edge, the freewheel diode on
  the SOUTH edge below the switch, the inductor on the EAST of the
  switch, the output capacitor on the EAST edge, and the load resistor
  to the EAST of the output capacitor

#### Scenario: Buck golden SVG match
- **WHEN** `render(buck_circuit, "/tmp/buck.svg")` is called using
  the same circuit as `python/tests/fixtures/schematic/buck.svg`
- **THEN** the output SVG either matches the fixture byte-for-byte OR
  satisfies a structural-equivalence check (same component positions
  within 1 pixel tolerance, same wire endpoints)

#### Scenario: Unrecognized components inside a recognized topology
- **WHEN** a buck circuit also contains an unidentified snubber RC
  pair
- **AND** the recognizer returns `topology="buck"` with a `role_map`
  covering only the main components
- **THEN** the main components are placed via the template
- **AND** the snubber components are placed via force-directed layout
  inside the canvas region not occupied by template slots

#### Scenario: No template, no recognition — force-directed fallback
- **WHEN** the circuit fails both recognizer tiers (none of the 17
  topologies match)
- **THEN** the renderer uses the V1 force-directed layout with
  electrical priors (ground south, sources west)
- **AND** the resulting layout still respects ground-anchoring,
  source-edge, and determinism guarantees from the V1 spec

### Requirement: Single-File Schematic Render
The schematic module SHALL render a circuit to a single SVG or PNG
file with all components placed and connected by drawn wires.

#### Scenario: Render to SVG
- **WHEN** the user calls `pulsim.schematic.render(circuit,
  "buck.svg")`
- **THEN** the resulting file contains a graphic element for each
  component in `circuit.components()`
- **AND** wires connect terminal anchors that share a node ID

#### Scenario: Render to PNG via cairosvg
- **WHEN** the destination path ends in `.png`
- **AND** `cairosvg` is installed
- **THEN** the output file's first bytes match the PNG signature
  `\x89PNG`

#### Scenario: PNG fallback to schemdraw when cairosvg missing
- **WHEN** the destination path ends in `.png`
- **AND** `cairosvg` is NOT installed
- **AND** `schemdraw` IS installed
- **THEN** the renderer produces a valid PNG via schemdraw's native
  PNG export

#### Scenario: Format override
- **WHEN** the user passes `format="svg"` to `render(circuit,
  "out.bin", format="svg")`
- **THEN** the file is written as SVG regardless of extension

### Requirement: Layout Determinism
The same circuit rendered twice SHALL produce byte-identical layout
output (`compute_layout`) and, for non-PNG outputs, byte-identical
file content.

#### Scenario: Determinism across processes
- **WHEN** the same circuit is laid out in process A and process B
  with the same cache state
- **THEN** every `ComponentPlacement` and `Wire` is byte-identical
  between the two `SchematicLayout` JSON dumps

#### Scenario: Determinism after cache rebuild
- **WHEN** the cache file is deleted and the same circuit is rendered
- **THEN** the LLM classifier hits the API
- **AND** the resulting layout still matches the layout produced when
  the cache was warm (the LLM response is stable for the same
  fingerprint within the same model version)

### Requirement: JSON-Serializable Layout
The `SchematicLayout` type SHALL serialize to and deserialize from
JSON without loss of placement information. Schema version: `"schematic-v1"`.

#### Scenario: Round-trip a layout
- **WHEN** code calls `SchematicLayout.from_json(layout.to_json())`
- **THEN** the deserialized layout equals the original in components,
  wires, junctions, and canvas

#### Scenario: Reject unknown schema version
- **WHEN** `from_json()` receives a payload with `schema_version`
  other than `"schematic-v1"`
- **THEN** the call raises `ValueError` naming the unsupported version

### Requirement: Optional Dependency Surface
The schematic module SHALL be importable without `schemdraw`,
`networkx`, `cairosvg`, or `anthropic` installed, but SHALL raise a
clear actionable error when a required dependency is missing at
runtime.

#### Scenario: Import without optional deps
- **WHEN** Python imports `pulsim.schematic` without `schemdraw`,
  `networkx`, `cairosvg`, or `anthropic` installed
- **THEN** the import succeeds without raising

#### Scenario: Render without schemdraw
- **WHEN** `render(...)` is called and `schemdraw` is missing
- **THEN** an `ImportError` is raised whose message includes the
  text `pip install pulsim[schematic]`

#### Scenario: LLM classifier without anthropic
- **WHEN** the LLM classifier is invoked and `anthropic` is missing
- **THEN** the classifier returns `None` without raising
- **AND** a one-time `UserWarning` recommends
  `pip install pulsim[schematic]`

### Requirement: Jupyter Inline Display
The `SchematicLayout` type SHALL implement `_repr_svg_` so that
returning a layout from a Jupyter cell auto-displays the schematic.

#### Scenario: Notebook display
- **WHEN** a Jupyter cell returns the result of `compute_layout(...)`
- **THEN** the cell output renders an inline SVG of the layout
- **AND** no explicit `.render()` call is required

### Requirement: Graceful Handling of Unknown Component Kinds
The renderer SHALL NOT crash when it encounters a component kind not
present in the symbol library.

#### Scenario: Unmapped component kind
- **WHEN** a circuit contains a component whose `kind` is not present
  in the symbol library
- **THEN** the renderer draws a labeled rectangle in its place
- **AND** the renderer emits a `UserWarning` naming the unknown kind
