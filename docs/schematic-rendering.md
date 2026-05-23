# Schematic rendering

`pulsim.schematic` turns a `CircuitBuilder` into a publication-quality
SVG (or PNG) by recognising canonical power-electronics topologies and
laying them out with hand-tuned templates, falling back to
constraint-aware force-directed for everything else.

```python
import pulsim as p
import pulsim.schematic as sch

b = p.CircuitBuilder()
b.add_voltage_source("Vin",   "vin", "gnd", 24.0)
b.add_switch        ("Q",     "vin", "sw",   g_on=1e3, g_off=1e-9)
b.add_diode         ("D_FW",  "gnd", "sw",   g_on=1e3, g_off=1e-9, V_th=0.0)
b.add_inductor      ("L1",    "sw",  "vout", 100e-6)
b.add_capacitor     ("Cout",  "vout","gnd",  100e-6)
b.add_resistor      ("Rload", "vout","gnd",  5.0)

sch.render(b, "buck.svg")     # SVG (default)
sch.render(b, "buck.png")     # PNG (extension picks the writer)
```

In a Jupyter cell, just return the layout — the schematic is rendered
inline via `_repr_svg_`:

```python
sch.compute_layout(b)
```

## How it picks a layout

Three tiers are tried in order; the first one to produce a result wins.

### Tier 1 — Deterministic recogniser
`pulsim.schematic.topology_recognizer.recognize(circuit)` returns a
`RecognizedTopology` for any of the twelve canonical topologies it
knows: `buck`, `boost`, `buck_boost`, `flyback`, `forward`,
`half_bridge`, `full_bridge`, `rc_filter`, `rl_filter`, `rlc_filter`,
`half_wave_rectifier`, `full_wave_bridge_rectifier`. Pure Python, no
network. Confidence in `[0, 1]` — a hit is `≥ 0.9` for the canonical
shape.

### Tier 2 — LLM classifier (optional, on by default)
When Tier 1 returns `confidence < 0.7` (or `None`), the LLM classifier
sends the circuit fingerprint to the Anthropic API
(`claude-haiku-4-5` by default) and parses one of seventeen known
topologies from the JSON response. Results are cached locally at
`~/.cache/pulsim/topology-cache.json`, so the API is invoked at most
once per distinct netlist fingerprint.

Tier 2 is **disabled silently** when any of these conditions hold —
the renderer always falls through to Tier 3 instead of raising:

| Condition                                  | Effect                       |
|--------------------------------------------|------------------------------|
| `PULSIM_LLM_LAYOUT_HINTS=0`                | LLM never invoked            |
| `ANTHROPIC_API_KEY` missing                | Skip Tier 2, no error        |
| `anthropic` package not installed          | Skip Tier 2, one-time warning|
| Network error / rate limit / parse fail    | Skip Tier 2, log at DEBUG    |

Override the model with `PULSIM_LLM_MODEL=claude-sonnet-4-5` and the
cache directory with `PULSIM_TOPOLOGY_CACHE_DIR=/some/path`.

### Tier 3 — Template instantiator
Once Tiers 1 + 2 produce a `RecognizedTopology`, the instantiator loads
`python/pulsim/schematic/templates/<name>.yaml` and places every
recognised role at the canvas-fraction slot defined in the template.
The buck, boost, and flyback templates are visually polished; the
remaining nine ship as drafts and may be refined in follow-up changes.

### Fallback — Force-directed
If every tier above misses (no template, low confidence, empty
role_map, parsing error), the legacy force-directed layout runs. It
produces a sensible if generic placement with ground anchored to the
south rail and voltage sources on the west edge.

## Output formats

`render_layout(layout, path)` infers the format from the file
extension. Supported: `.svg` (default), `.png`, `.pdf`, `.jpg`,
`.jpeg`. PNG rendering uses `schemdraw`'s native matplotlib backend;
the `netlistsvg` legacy backend additionally accepts a
`cairosvg`/`rsvg-convert` SVG → PNG path for higher fidelity.

Pass `format="svg"` explicitly to override an unrecognised extension.

## Determinism

The same circuit produces a byte-identical SVG across runs (assuming
the cache state is the same). This is a hard requirement — every
detector and the LLM cache lookup are deterministic. Two consecutive
`compute_layout` calls return byte-identical placements; two
consecutive `render_layout` calls produce identical SVG bytes.

## API reference

### `pulsim.schematic`

| Function                                    | Purpose                                 |
|---------------------------------------------|-----------------------------------------|
| `compute_layout(circuit) → SchematicLayout` | Dispatch through all 3 tiers + fallback |
| `render_layout(layout, path, format=…)`     | Write SVG/PNG to disk                   |
| `render(circuit, path, format=…)`           | Convenience: layout + render in one call|

### `pulsim.schematic.topology_recognizer`

| Symbol                       | Purpose                                         |
|------------------------------|-------------------------------------------------|
| `recognize(circuit)`         | Tier-1 deterministic recogniser                 |
| `RecognizedTopology`         | `(name, confidence, role_map, source)` dataclass|
| `KNOWN_TOPOLOGIES`           | Frozen set of Tier-1 names                      |
| `detect_buck(view)` …        | Per-topology detector — exposed for advanced use|

### `pulsim.schematic.llm_classifier`

| Symbol                        | Purpose                              |
|-------------------------------|--------------------------------------|
| `classify(circuit)`           | Tier-2 LLM classifier                |
| `circuit_fingerprint(circuit)`| Deterministic text representation    |
| `LLM_KNOWN_TOPOLOGIES`        | Superset of Tier-1 + 5 LLM-only names|
| `CACHE_SCHEMA_VERSION`        | On-disk cache schema tag             |
| `DEFAULT_MODEL`               | `"claude-haiku-4-5"`                 |

### `pulsim.schematic.template_instantiator`

| Symbol                              | Purpose                              |
|-------------------------------------|--------------------------------------|
| `template_layout(circuit, recognized)` | Build layout from template     |
| `load_template(name)`               | Read a single YAML template          |
| `list_available_templates()`        | Names of templates that ship today   |
| `Template`, `Slot`                  | Loaded template dataclasses          |

## Adding a new template

1. Drop a `python/pulsim/schematic/templates/<name>.yaml` with:

```yaml
name: my_topology
quality: draft          # or "polished" once visually validated
canvas:
  width:  200          # mm
  height: 120
slots:
  source:           { x: 0.10, y: 0.50, rotation: 0 }
  inductor_main:    { x: 0.30, y: 0.30, rotation: 90 }
  # … one entry per role the recognizer assigns
```

2. If the topology is one the deterministic recogniser already covers,
   you're done — `compute_layout` will pick up the template on the
   next run.

3. If it's a new topology, either:
   * add a `detect_<name>(view)` function and append it to `_DETECTORS`
     in `topology_recognizer.py`, **or**
   * add the name to `LLM_KNOWN_TOPOLOGIES` so the LLM tier can
     classify it.

## Optional dependencies

| Package      | Required for                                   |
|--------------|------------------------------------------------|
| `schemdraw`  | SVG/PNG render via the native backend          |
| `networkx`   | Force-directed fallback layout                 |
| `PyYAML`     | Loading templates                              |
| `anthropic`  | Tier-2 LLM classifier (Anthropic API client)   |
| `cairosvg`   | High-fidelity SVG → PNG conversion path        |

Install the bundle:

```bash
pip install 'pulsim[schematic]'
```

A bare `import pulsim.schematic` succeeds even without these packages
— each tier is wrapped in a try/except so the renderer degrades
gracefully.
