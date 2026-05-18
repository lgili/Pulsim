## Gates & Definition of Done

- [x] G.1 `pulsim.schematic.render(buck_circuit, "out.svg")` produces an SVG where the `S1` (vcswitch) is drawn with the dedicated `vcswitch` symbol, NOT as a labeled `generic` rectangle. Asserted by `test_vcswitch_renders_with_pulsim_skin_symbol`.
- [x] G.2 Same for `Q1` (mosfet) in `boost_pfc`, and both `Q_hi` / `Q_lo` (mosfets) in a half-bridge fixture. Asserted by `test_mosfet_renders_with_pulsim_skin_symbol` (single-MOSFET fixture). Half-bridge visual smoke landed in [`build/schematic_demo/half_bridge.png`](../../../build/schematic_demo/half_bridge.png): both Q_hi and Q_lo render with the `mosfet_n` symbol, body-diode arrows point toward the channel as canonical N-channel direction.
- [x] G.3 The Pulsim skin SVG validates as well-formed XML and loads via `netlistsvg --skin <path>` without errors. Demonstrated by every render call routing through `netlistsvg` with `--skin pulsim_analog.svg`; the schematic suite (36 tests) produces non-zero-byte SVGs for every fixture.
- [x] G.4 No regression on passive-component rendering: existing buck / boost PFC / RC schematics still pass their visual smoke tests. Asserted by `test_passive_only_circuit_does_not_emit_switching_symbols` (regression guard: pure RC circuit must not emit any switching glyphs) plus the 36-test schematic suite continuing to pass.
- [x] G.5 Wider test suite (`pytest python/tests --ignore=python/tests/validation`) remains green. **410 passed, 12 skipped, 2 deselected, 0 regressions** in the closeout run on 2026-05-17.

## Phase 1: Author the Pulsim-specific analog skin

- [x] 1.1 Copied upstream `node_modules/netlistsvg/lib/analog.svg` to [`python/pulsim/schematic/skin/pulsim_analog.svg`](../../../python/pulsim/schematic/skin/pulsim_analog.svg) as the base. Pulsim additions are appended in a marked block at the end of the file, before `</svg>`.
- [x] 1.2 `mosfet_n` symbol: 30×50 SVG block with vertical channel (M20,10 V40), gate plate (M14,15 V35), gate stub (M0,25 H14), drain/source pin connectors, and an N-channel body-diode arrow (filled triangle) on the source end of the channel pointing toward drain. Three `s:pid` ports: `G` (left), `D` (top), `S` (bottom).
- [x] 1.3 `mosfet_p` symbol: identical body to mosfet_n with the arrow flipped — apex up at y=7 with base at y=12, pointing away from the channel (canonical P-channel direction).
- [x] 1.4 `igbt` symbol: mosfet_n body plus an inverted hollow triangle on the gate-to-channel side (drawn at M20,18 L17,15 L23,15 Z, stroke-only), visually distinguishing the IGBT from a pure MOSFET. Ports renamed to `G`/`C`/`E` to match IGBT convention.
- [x] 1.5 `vcswitch` symbol: two open-contact circles at the top/bottom terminals (`circle cx="20" cy="12|38" r="2"`), a switch arm in the open position (M20,12 L11,33), and a dashed control line (`stroke-dasharray="3,2"`) from the ctrl pin to the switch arm. Ports `ctrl` (left), `t1` (top), `t2` (bottom).

## Phase 2: Wire the new symbols into the Python backend

- [x] 2.1 New `_resolve_skin()` helper in `netlistsvg_backend.py` walks for `python/pulsim/schematic/skin/pulsim_analog.svg` (preferred) and falls back to `node_modules/netlistsvg/lib/analog.svg` (upstream). Behavior is uniform across source-tree and build-tree imports.
- [x] 2.2 `_CELL_TYPE` extended with `"mosfet": "mosfet_n"`, `"igbt": "igbt"`, `"vcswitch": "vcswitch"`.
- [x] 2.3 `_CELL_PORTS` extended with the per-kind port lists: `mosfet_n`/`mosfet_p` use `[("G","input"), ("D","input"), ("S","output")]`; `igbt` uses `[("G","input"), ("C","input"), ("E","output")]`; `vcswitch` uses `[("ctrl","input"), ("t1","input"), ("t2","output")]`.
- [x] 2.4 `python/CMakeLists.txt` glob extended to `pulsim/${submodule}/*/*.svg` so the `schematic/skin/` sub-directory's SVG files get copied into the build tree alongside the `.py` and `.js` files.

## Phase 3: Tests + regression coverage

- [x] 3.1 `test_mosfet_renders_with_pulsim_skin_symbol` in [`python/tests/test_schematic_render.py`](../../../python/tests/test_schematic_render.py) — builds a tiny MOSFET-driven RC circuit, renders to SVG, asserts the SVG text contains `s:type="mosfet_n"`. Passing.
- [x] 3.2 `test_vcswitch_renders_with_pulsim_skin_symbol` — same check for `buck_converter.yaml`'s `S1`. Passing.
- [x] 3.3 `test_passive_only_circuit_does_not_emit_switching_symbols` — regression guard: a pure RC circuit must NOT emit any of `mosfet_n`, `mosfet_p`, `igbt`, `vcswitch` symbol tags. Passing. All three tests gated by `_NETLISTSVG_DEFAULT` skipif so they only run when netlistsvg is the active backend.
- [x] 3.4 Visual smoke set re-rendered with the new skin: [`build/schematic_demo/buck_skin.png`](../../../build/schematic_demo/buck_skin.png) shows S1 as a proper vcswitch (open contacts + switch arm + dashed control), [`build/schematic_demo/boost_pfc_skin.png`](../../../build/schematic_demo/boost_pfc_skin.png) shows Q1 as a proper MOSFET-N with body-diode arrow, [`build/schematic_demo/half_bridge.png`](../../../build/schematic_demo/half_bridge.png) shows Q_hi and Q_lo both as MOSFET-Ns. Bridge diodes, sources, passives, ground symbols unchanged from Phase 4F — no regression on the rest of the skin.
