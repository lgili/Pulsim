## Why

After `add-schematic-rendering` Phase 4F + `add-pulsim-analog-skin`, every Pulsim circuit renders with proper analog symbols (resistor zigzag, capacitor parallel lines, MOSFET vertical body, ground rails, etc.) and orthogonal Manhattan wires. But the **placement** of components is whatever ELK's layered algorithm decides — for cyclic power topologies (buck, boost, half-bridge) it's topologically correct but rarely matches the conventional textbook layout an engineer would draw.

Industry-standard schematic editors (Eeschema, KiCad, Altium, OrCAD) solve this the same way: they auto-layout as a starting point and let the user drag components into a "conventional" position. Pulsim should do the same — keep auto-layout as the default, expose **optional position hints** so power users can pin specific components when they care about the rendering.

### Empirical findings (MVP attempt, 2026-05-17)

The first attempt was: run `netlistsvg/bin/exportLayout.js` to dump the default ELK layout, mutate `(x, y)` per cell from hints, feed back via `netlistsvg --layout`. Two real upstream bugs blocked it:

1. **Promise resolves `undefined` when `elkData` is passed.** In `netlistsvg/built/index.js`, the `elkData` branch calls `drawModule(elkData, flatModule)` but `resolve()` is invoked WITHOUT the SVG output, so the downstream `fs.writeFile(undefined, ...)` throws `ERR_INVALID_ARG_TYPE`. A one-line patch makes the basic round-trip work, but it lives in `node_modules/` and disappears on `npm install`.
2. **Edge `sections` are pre-computed against the OLD cell positions.** When we move cells, the cached bend points reference coords that no longer exist — the rendered SVG draws tangled wires through empty space. Stripping `sections` produces a worse crash ("Cannot read properties of undefined (reading '0')") because `drawModule` assumes they're present.

Path forward — three real options, none trivial:

- **A. Fork netlistsvg** with a clean position-hint API + a proper sections re-router. ~1-2 weeks; we'd own a fork from then on.
- **B. Replicate the analog-skin rendering in Python.** Parse `analog.svg` symbol definitions, run elkjs (which we already integrate via the ELK backend) with user position constraints, draw the symbols at the computed positions. ~2-3 weeks; cleaner long-term but lots of new code.
- **C. Wait for an upstream fix or a different library.** netlistsvg is effectively abandoned (last release 2020-12-12). Realistically: never.

Until one of those lands, `pulsim.schematic.render(..., position_hints={...})` raises `NotImplementedError` pointing at this proposal.

## What Changes

### YAML schema
- ADD: optional `position` field per component in the YAML netlist:
  ```yaml
  components:
    - type: voltage_source
      name: Vdc
      nodes: [vcc, 0]
      waveform: { type: dc, value: 24 }
      position: { layer: 0, slot: 0 }        # semantic: column 0, row 0
    - type: vcswitch
      name: S1
      nodes: [ctrl, vcc, sw]
      position: { layer: 1, slot: 0 }
    - type: diode
      name: D1
      nodes: [0, sw]
      position: { layer: 2, slot: 1 }        # below sw_node, freewheel column
  ```
- ALTERNATIVE shorthand: `position: { x: 100, y: 50 }` for absolute coordinates (in netlistsvg layout units). Lets users translate from an editor that exports x/y directly.
- BACKWARDS COMPATIBLE: `position` is optional; circuits without it render as today (full auto-layout).

### Python API
- ADD: `Circuit.set_position(name, layer=None, slot=None, x=None, y=None)` for programmatic placement.
- ADD: `Circuit.position_hints()` returning a `dict[str, PositionHint]` snapshot — analogous to `Circuit.components()`.
- ADD: `PositionHint` value type carrying `(layer, slot)` semantic OR `(x, y)` absolute, plus the canonical kind & component name.

### Backend integration
- ADD: in [`netlistsvg_backend.py`](python/pulsim/schematic/netlistsvg_backend.py), an `_apply_position_hints(layout, hints)` step that:
  1. Calls `node_modules/netlistsvg/bin/exportLayout.js` on the input JSON to get the default layout.
  2. Translates each hint's `(layer, slot)` to an absolute `(x, y)` using a grid (e.g. 120 px per layer, 80 px per slot).
  3. Overrides the matching child's `x, y` in the layout JSON.
  4. Recomputes basic edge routing if cells moved more than a threshold (or accept that wires render straight between moved endpoints).
- ADD: `render_netlistsvg` and the top-level `render()` accept the user's hints automatically when the YAML netlist provides them.

### Tests
- ADD: a buck circuit hand-positioned via YAML `position` hints; assert the rendered SVG places Vdc / S1 / D1 / L1 / C1 / Rload at the user's exact (layer, slot) grid coords.
- ADD: regression guard: a circuit without any `position` field renders identically to the auto-layout output (no silent layout drift).

## Impact

- Affected specs:
  - `netlist-yaml` (the new `position` field is part of the schema)
  - `python-bindings` (new `Circuit.set_position` / `position_hints` surface)
- Affected code:
  - `core/include/pulsim/v1/runtime_circuit.hpp` — store position hints alongside `ComponentDescriptor`
  - `core/include/pulsim/v1/parser/yaml_parser.hpp` (or `core/src/v1/yaml_parser.cpp`) — parse the new field
  - `python/bindings.cpp` — bind the new accessors
  - `python/pulsim/_pulsim.pyi` — type stubs
  - `python/pulsim/schematic/netlistsvg_backend.py` — apply hints via `--layout`
  - `python/tests/test_schematic_render.py` — new tests
  - `docs/schematic-rendering.md` (currently TODO in Phase 5 of `add-schematic-rendering`) — user-facing guide
- No new runtime dependency.
