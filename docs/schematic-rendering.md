# Schematic Rendering

Render any Pulsim circuit to a publication-quality schematic SVG or PNG with one Python call. The default backend produces real analog symbols (resistor zig-zag, capacitor parallel lines, MOSFET vertical body, ground rails, etc.) — not labeled boxes — and routes wires orthogonally.

## Quick start

```python
import pulsim as ps

parser = ps.YamlParser(ps.YamlParserOptions())
circuit, _ = parser.load("examples/buck_converter.yaml")

# SVG — vector, scales to any resolution
ps.schematic.render(circuit, "buck.svg")

# PNG — handy for embedding in docs / chat
ps.schematic.render(circuit, "buck.png")
```

That's it. No layout configuration required for the common case.

## What gets rendered

Every device introspected via `Circuit.components()` becomes a symbol on the canvas:

| Pulsim kind | Symbol |
|---|---|
| `resistor` | IEC rectangle (zig-zag with the alternative skin) |
| `capacitor` | Two parallel lines |
| `inductor` | Coiled spiral |
| `voltage_source` / `pwm_voltage_source` / `sine_voltage_source` / `pulse_voltage_source` | Circle with ±, sine, pulse, or DC indicator |
| `current_source` | Circle with arrow |
| `diode` | Triangle + cathode bar |
| `mosfet` | N-channel MOSFET (vertical channel, gate left, body-diode arrow) |
| `igbt` | MOSFET body + collector triangle |
| `vcswitch` | Open contacts + tilted switch arm + dashed control input |
| `transformer` | Two coupled coils |
| anything else | Labeled rectangle (graceful fallback) |

Each terminal connecting to ground gets its own `gnd` symbol — standard schematic convention.

## Install

The default backend needs **Node.js** + **netlistsvg** (the production renderer used by the Yosys / openlane ecosystem, with a Pulsim-extended analog skin). PNG output additionally requires `librsvg` or `cairosvg` for the SVG→PNG conversion.

```bash
# Repo root
npm install netlistsvg elkjs

# macOS — for PNG output
brew install librsvg

# Debian/Ubuntu — for PNG output
sudo apt-get install librsvg2-bin

# Python side: nothing extra; pulsim.schematic is part of the core wheel.
```

If `pulsim.schematic.render(...)` is called without Node installed, an `ImportError` fires with the exact install command.

## Backends and how to switch

`PULSIM_SCHEMATIC_BACKEND` selects the rendering backend:

| Value | When to use | Output |
|---|---|---|
| `netlistsvg` (default) | Default — publication quality | SVG (PNG/PDF/JPG via rsvg-convert) |
| `elk` | Need a Python `SchematicLayout` object (e.g. for GUI auto-place) | SVG/PNG via schemdraw + ELK Sugiyama layered layout |
| `spring` | No Node.js available, or comparing against the legacy layout | SVG/PNG via schemdraw + force-directed |

```bash
PULSIM_SCHEMATIC_BACKEND=elk python my_script.py
```

## GUI auto-placement

For a GUI that wants to read component coordinates and draw the canvas itself, use the ELK backend's `SchematicLayout`:

```python
import os
os.environ["PULSIM_SCHEMATIC_BACKEND"] = "elk"

import pulsim as ps
parser = ps.YamlParser(ps.YamlParserOptions())
circuit, _ = parser.load("examples/buck_converter.yaml")

layout = ps.schematic.compute_layout(circuit)
payload = layout.to_json()
# payload is a dict with keys: components, wires, junctions, canvas, schema_version
# Coordinates in mm, origin top-left, +x right, +y down.
```

`payload["components"]` is a dict keyed by component name with `{x, y, rotation, terminal_anchors, ...}`. `payload["wires"]` is a list of `{from, to, path}`. The schema version is `"schematic-v1"` and stays stable for GUI consumers.

## Custom output paths

Both SVG and raster formats infer from the path extension. Override explicitly with `format=`:

```python
ps.schematic.render(circuit, "out.svg")           # → SVG
ps.schematic.render(circuit, "out.png")           # → PNG
ps.schematic.render(circuit, "out.pdf")           # → PDF (rsvg-convert)
ps.schematic.render(circuit, "diagram", format="svg")  # explicit
```

## Known limits

- **Layout is auto-computed.** Component positions follow netlistsvg's ELK layered placement, which honors connectivity but doesn't reproduce a hand-drawn "textbook" layout for every topology. Tracked in [`openspec/changes/add-schematic-position-hints`](../openspec/changes/add-schematic-position-hints) — manual position hints are blocked by upstream netlistsvg bugs and require a fork or a Python re-implementation to land.
- **Switching-device symbols** (`mosfet`, `igbt`, `vcswitch`) ship in the Pulsim analog skin (`python/pulsim/schematic/skin/pulsim_analog.svg`). If you load a custom skin that drops them, they fall back to a labeled `generic` rectangle.
- **Probes / control blocks** (`voltage_probe`, `pi_controller`, …) don't have analog skin symbols. They render as labeled boxes.
- **SPICE-style directives** (`.tran`, `.ic`, etc.) are not part of the rendered graphic — they live in the YAML `simulation:` block.

## Programmatic circuits

The same `render()` works for circuits built in Python:

```python
import pulsim as ps

ckt = ps.Circuit()
vin = ckt.add_node("vin")
vout = ckt.add_node("vout")
gnd = ckt.ground()
ckt.add_voltage_source("V1", vin, gnd, 12.0)
ckt.add_resistor("R1", vin, vout, 1_000.0)
ckt.add_capacitor("C1", vout, gnd, 1e-6, 0.0)

ps.schematic.render(ckt, "rc.svg")
```

## Testing your schematic in code

`pulsim.schematic.render(circuit, path)` returns a `pathlib.Path`. A simple smoke test:

```python
import pulsim as ps
import xml.etree.ElementTree as ET

def test_my_circuit_renders(tmp_path):
    ckt = build_my_circuit()
    out = ps.schematic.render(ckt, tmp_path / "my.svg")
    assert out.exists() and out.stat().st_size > 0
    ET.parse(out)  # raises on malformed SVG
```

The schematic test suite in [`python/tests/test_schematic_render.py`](../python/tests/test_schematic_render.py) covers SVG/PNG output, the unknown-kind fallback, and the lazy-import gate — read that file for more patterns.
