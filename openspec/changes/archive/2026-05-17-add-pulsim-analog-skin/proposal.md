## Why

After landing the netlistsvg backend in `add-schematic-rendering` Phase 4F, the rendered schematics look textbook for every passive (resistor, capacitor, inductor, diode, transformer, voltage/current source). But **three-terminal switching devices** — MOSFET, IGBT, vcswitch — fall through to the netlistsvg `generic` cell type and render as a labeled rectangle with pin names like `in0`, `out0`, `out1`. The analog skin only ships symbols for `transistor_npn` and `transistor_pnp` (bipolars); MOSFET/IGBT/vcswitch have no glyph.

This is the last visible quality gap in the schematic pipeline. For a power-electronics simulator, switching devices are the central focus — every converter has at least one MOSFET or IGBT, and rendering them as anonymous boxes defeats the readability gain from netlistsvg.

## What Changes

### Extend `analog.svg` with a Pulsim skin variant
- ADD: a forked / overlaid SVG skin shipping in `python/pulsim/schematic/skin/pulsim_analog.svg` with the upstream netlistsvg analog skin plus four new symbols:
  - `mosfet_n` (vertical body, gate left, drain top, source bottom)
  - `mosfet_p` (same shape, body diode arrow inverted)
  - `igbt` (mosfet shape + collector triangle)
  - `vcswitch` (SPDT-style switch with explicit control input)
- ADD: aliases `mosfet_n`, `mosfet_p`, `igbt`, `vcswitch` (short forms — netlistsvg matches aliases not type names, per the Phase 4F bug discovery).
- The skin SHALL be a self-contained SVG file that loads via `netlistsvg --skin <path>` and renders the new symbols inline with the existing analog symbols.

### Wire the symbols into `netlistsvg_backend.py`
- MODIFY: `_CELL_TYPE` in `netlistsvg_backend.py` to map `mosfet` → `mosfet_n`, `igbt` → `igbt`, `vcswitch` → `vcswitch`. Add `_CELL_PORTS` entries with the appropriate port directions.
- MODIFY: `_require_netlistsvg` (or a new helper) to prefer the Pulsim skin at `python/pulsim/schematic/skin/pulsim_analog.svg` over the bundled `node_modules/netlistsvg/lib/analog.svg`.

### Validation
- ADD: 3 new test cases asserting that buck (with a vcswitch / mosfet), boost PFC, and half-bridge circuits render with the proper switching-device symbols, NOT the generic rectangle. The assertion can be a simple "the SVG contains the `mosfet_n` symbol class" string match — testing the glyph itself is out of scope.

## Impact

- Affected specs: `python-bindings` (the public surface of `pulsim.schematic.render` gains better symbol coverage but no API change)
- Affected code:
  - `python/pulsim/schematic/skin/pulsim_analog.svg` (new file, ~200-400 lines of SVG paths + skin metadata)
  - `python/pulsim/schematic/netlistsvg_backend.py` (small edits to `_CELL_TYPE`, `_CELL_PORTS`, skin-path resolution)
  - `python/CMakeLists.txt` (add the `.svg` glob to the schematic submodule copy)
  - `python/tests/test_schematic_render.py` (3 new test cases)
- No new runtime dependency.
