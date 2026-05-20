## Why

Today's default schematic backend is **netlistsvg** — an unmaintained Node.js library (last release 2020-12-12) that we drive via subprocess. It works, but it has hard ceilings:

1. **Manual position hints are blocked upstream.** [`add-schematic-position-hints`](../add-schematic-position-hints/proposal.md) documents two real bugs in `netlistsvg/built/index.js`:
   - the `--layout` path resolves its promise with `undefined`, so the SVG never reaches `fs.writeFile`;
   - edge `sections` are pre-computed against the original cell positions and don't follow when we move cells.
   Both are patchable in `node_modules/`, but the patch dies on every `npm install`. Without position hints, every render is whatever ELK's layered algorithm decided — topologically right, but it almost never matches the textbook layout an engineer would draw (sources on the left, DC bus horizontal at the top, ground rail at the bottom, motor/load on the right).

2. **Node.js is an awkward dependency.** Pulsim is otherwise pure Python + C++. Shipping a Node toolchain to every user / CI runner just to draw a schematic costs install footprint, surfaces Node-version skew, and forces the schematic-smoke CI to live with its own toolchain quirks (PRs #8–#12 in this branch). On the user side, `pulsim.schematic.render(...)` requires `node_modules/netlistsvg/` to exist next to the package — an easy install-time failure.

3. **Owning the renderer unlocks "smart" layouts.** The `templates.py` recognizers already detect bridge rectifiers, half-bridges, boost stages, etc. But because we delegate placement to ELK, we can't translate "this is a buck converter" into the right (Vin-left, switch-center, freewheel-below, output-right) layout. A native renderer can.

The previous tracking proposal listed three paths forward; this proposal commits to **path B (replicate the analog-skin rendering in Python)** and obsoletes the tracking item.

## What Changes

### High-level

- ADD a new backend `pulsim.schematic.python_native` (default name TBD in design) that consumes the same Yosys-style JSON the netlistsvg backend builds today, loads `pulsim_analog.svg` as a skin (no new symbols required), and emits SVG/PNG without invoking Node for symbol composition.
- KEEP elkjs as the layout engine for the first cut (we already vendor `elk_bridge.js` for the ELK backend); the Python renderer reads layout JSON from elkjs and composes the SVG itself. Net dependency change for users: `netlistsvg` package becomes unnecessary; `elkjs` stays. Phase 4 of the tasks tracks dropping Node entirely.
- ADD position-hint storage on the kernel `Circuit` (mirroring the existing `node_position_hint`), pybind11 bindings, and a `position:` field in the YAML schema. Hints are passed to elkjs as position constraints, so wires re-route consistently with no upstream bug.
- ADD topology-aware auto-hinting: when a known sub-circuit is recognized (bridge rectifier, half-bridge, boost stage), the renderer emits default position hints for the matched components in the canonical textbook arrangement. Users can override per-component.
- SWITCH the default backend to the new Python renderer; the `netlistsvg` backend stays accessible via `PULSIM_SCHEMATIC_BACKEND=netlistsvg` for one release as a fallback, then deprecates.
- ADD a regression CI job that renders the demo set (rc / buck / half-bridge / boost-PFC) and uploads the SVGs as an artifact (the existing `schematic-smoke` workflow keeps working — the Python renderer becomes its build target).

### Public surface

```python
import pulsim as ps

ckt = ps.Circuit()
# ... add nodes & components ...

# Hint a specific component to (col=2, row=1) in the layout grid
ckt.set_position("Q1", layer=2, slot=1)

# Or with absolute coords in mm (renderer normalizes to internal units)
ckt.set_position("Cout", x=120.0, y=40.0)

# Render — hints are respected automatically
ps.schematic.render(ckt, "buck.svg")

# Inspect hints (read-only snapshot)
hints = ckt.position_hints()           # dict[str, PositionHint]
print(hints["Q1"])                     # PositionHint(layer=2, slot=1, x=None, y=None)
```

YAML form:

```yaml
components:
  - type: voltage_source
    name: Vdc
    nodes: [vcc, 0]
    waveform: { type: dc, value: 48.0 }
    position: { layer: 0, slot: 0 }       # column 0, row 0
  - type: vcswitch
    name: S1
    nodes: [ctrl, vcc, sw]
    position: { layer: 1, slot: 0 }       # next column over
  - type: diode
    name: D1
    nodes: [0, sw]
    position: { x: 200.0, y: 80.0 }       # absolute alternate form
```

Both forms are optional — circuits without `position:` keep auto-layout behavior.

### Quality bar

- Visual parity (or better) with the current netlistsvg output on the existing demo set (rc, buck, half-bridge, boost-pfc, pfc+vsi+pmsm).
- Same `schematic-v1` JSON schema for GUI consumers — `compute_layout(circuit).to_json()` keeps emitting the same fields.
- Numpy-only at the renderer boundary; no new runtime dependency beyond what the `[schematic]` extra already pulls in (schemdraw is no longer used by the default path; it stays for the `spring`/`elk` legacy backends).

## Impact

### Affected specs

- `schematic-rendering` — bulk of the new requirements (native renderer, position hints, topology-aware layouts).
- `python-bindings` — `Circuit.set_position`, `Circuit.position_hints`, `PositionHint` value type.
- `netlist-yaml` — `position:` field on the component schema.
- `kernel-v1-core` — position-hint storage on `Circuit` (read-only accessor mirrors `node_position_hint`).

### Affected code

- `python/pulsim/schematic/` — new module file `native_backend.py` (skin parsing + SVG composition), `layout.py` updated to dispatch the new default, `netlistsvg_backend.py` becomes the legacy path.
- `python/pulsim/schematic/skin/pulsim_analog.svg` — reused as-is; no new symbols.
- `core/include/pulsim/v1/runtime_circuit.hpp` — add `set_position`, `position_hint`, internal storage.
- `core/include/pulsim/v1/parser/yaml_parser.hpp` (or the .cpp split) — parse the new `position:` field.
- `python/bindings.cpp`, `python/pulsim/_pulsim.pyi` — bind/declare the new API.
- `python/tests/test_schematic_render.py` — regression cases for pin-and-render plus topology-aware defaults.
- `.github/workflows/schematic-smoke.yml` — keep the demo render; switch to the new default backend.
- `docs/schematic-rendering.md` — document the new API, deprecation timeline for netlistsvg.

### Relationship to other openspec changes

- **Supersedes** [`add-schematic-position-hints`](../add-schematic-position-hints/proposal.md) — once this change is implemented and archived, the tracking proposal can be archived too (its findings are folded into the design document here).
- Builds on `schematic-rendering` spec from the archived `add-schematic-rendering` change.

### Non-goals

- Replacing elkjs with a pure-Python layout engine. That's tracked separately as Phase 4 of `tasks.md` and may become its own proposal once we know how badly we miss ELK's quality.
- Interactive editing or GUI canvas drawing. Pulsim ships layout JSON; consumers (GUIs) render their own canvas.
- New analog symbols. Use what `pulsim_analog.svg` already has; if a new device class lands later (e.g. supercap), add the symbol in a separate change.
