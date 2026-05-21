# Tasks — add-python-schematic-renderer

Status legend: `[ ]` pending, `[x]` complete, `[~]` partial / in flight.

The phases are sequential — Phase 1 unlocks 2, 2 unlocks 3, etc. Within a phase the tasks are mostly independent and can be parallelized.

---

## Phase 1 — Skin parser + Python renderer (no position hints yet, no layout changes)

Goal: bit-for-bit equivalent output to the current netlistsvg backend on the demo set, with the SVG composition done in Python.

- [x] **1.1** `python/pulsim/schematic/native_backend.py` skeleton — orchestrates the new pipeline (skin parse → elkjs layout → SVG composition); cache lives in `skin_parser._SKIN_CACHE`.
- [x] **1.2** `SymbolTemplate` (frozen dataclass) lives in `skin_parser.py` with `kind: str`, `inner: tuple[ET.Element, ...]`, `ports: Mapping[str, tuple[float, float]]`, `width: float`, `height: float`.
- [x] **1.3** `parse_skin(svg_path)` extracts every `<g s:type="...">`, collects every `<g s:pid>` anchor, and resolves `<s:alias>` indirections so alias keys share template identity with their primary `s:type`. Path-keyed cache; `clear_skin_cache()` for tests.
- [x] **1.4** Yosys JSON is unchanged — the netlistsvg backend's `_build_yosys_json` stays its source of truth. The native renderer goes straight from `Circuit.components()` to an ELK graph in `_build_elk_graph(...)`, sized from skin geometry (no intermediate Yosys layer needed for the new code path).
- [x] **1.5** `_call_elk(graph)` subprocesses `node elk_bridge.js` (the existing ELK bridge), parses the laid-out JSON containing `children` (cell positions) and `edges` (sections/bend points).
- [x] **1.6** `_compose_svg(components, laid_out, skin, ground_id)` emits the output `<svg>`: orthogonal `<path>` wires from edge sections, `<circle>` junction dots, per-component `<g transform="translate(x,y)">` wrapping a deep-copy of the skin symbol's inner XML, ground-stub instantiations per gnd-touching terminal. Skin placeholders (`s:attribute="ref"` / `"value"` / `"name"`) are substituted with the real component name / engineering-formatted value.
- [x] **1.7** `render_native(circuit, svg_path)` writes the SVG, returns its `Path`. Plumbed into `render.py` dispatcher under `PULSIM_SCHEMATIC_BACKEND=python_native` (default still netlistsvg per Phase 5).
- [x] **1.8** `python/tests/test_schematic_native_backend.py` — 15/15 pass: 4 skin-parser cases, 3 value-formatter cases, 8 end-to-end render cases (RC + buck YAML + half-bridge YAML + dispatcher + unknown-kind fallback + valid-XML + orthogonal-wires + value-label-substitution). The tests skip cleanly when `node` is absent.
- [x] **1.9** Smoke-rendered the demo set via the dispatcher (`PULSIM_SCHEMATIC_BACKEND=python_native python3 -c "..."`); outputs at `build/schematic_demo/{rc_circuit,buck_converter,half_bridge_pwm}_native.{svg,png}` confirm placeholder substitution + orthogonal wires + ground stubs + the four switching/passive symbols (zigzag resistor, parallel-line capacitor, spiral inductor, voltage-source circle, vcswitch, diode).

## Phase 2 — Position-hint storage on the kernel

Goal: hints are first-class data on `Circuit` and survive YAML round-trip. Renderer doesn't use them yet — that's Phase 3.

- [x] **2.1** `struct PositionHint` in `core/include/pulsim/v1/runtime_circuit.hpp` with four `std::optional` fields and a `bool empty()` helper. Invariant enforced by `set_position` (NOT by the struct itself — POD stays simple).
- [x] **2.2** Private `std::unordered_map<std::string, PositionHint> position_hints_` added next to `virtual_components_`. (`unordered_map` per the PR #10 lesson.)
- [x] **2.3** Public accessors on `Circuit`: `set_position(name, layer?, slot?, x?, y?)`, `position_hint(name) -> std::optional<PositionHint>`, `position_hints() -> std::unordered_map<...>` (value snapshot), `num_position_hints() -> size_t`.
- [x] **2.4** `set_position` rejects fully-empty hints (`throw std::invalid_argument`); accepts mixed `(layer, slot)` + `(x, y)` and persists all four (renderer decides priority — kernel doesn't filter). Re-setting replaces wholesale (no merging). Device-existence is NOT validated — hints can be set before the device is added (YAML parser pattern). Stale hints are silently ignored by the renderer.
- [x] **2.5** `core/tests/test_position_hints.cpp` (11 cases / 57 assertions): empty circuit, both forms round-trip, snapshot detachment, missing-component query, empty-hint rejection, re-set replacement, hints survive device adds, hints don't affect `num_components`, determinism across builds.
- [x] **2.6** YAML parser (`core/src/v1/yaml_parser.cpp`) parses optional `position:` map per component. Accepts `layer`/`slot` ints and `x`/`y` reals (independent — both forms allowed simultaneously). Invalid YAML types → typed error (`kDiagInvalidParameter`); empty `position:` map → warning, component still created.
- [x] **2.7** `test_position_hints.py` covers YAML round-trip for `(layer, slot)`, `(x, y)`, mixed, missing field, and the bad-position warn-and-ignore path.
- [x] **2.8** pybind11 bindings: `py::class_<PositionHint>` with `def_readonly` for the four fields and a `__repr__`; Circuit gets `set_position(name, *, layer, slot, x, y)`, `position_hint(name)`, `position_hints()`, `num_position_hints()`.
- [x] **2.9** Stubs in `python/pulsim/_pulsim.pyi` for the four new Circuit methods + the `PositionHint` class. `python/tests/test_position_hints.py` (17 cases) covers the Python surface + YAML round-trip; full schematic test surface (70 cases) passes with zero regressions.

## Phase 3 — Hints flow into the layout

Goal: the new renderer respects user hints.

- [x] **3.1** `_build_elk_graph` now accepts `position_hints: dict[str, tuple[float, float]]`. Hinted cells get an explicit `x, y` plus `layoutOptions["org.eclipse.elk.position"]` in the input JSON, and the root graph switches to the `INTERACTIVE` strategy chain (cycle-breaking + layering + crossing-min) so ELK doesn't re-layer hinted nodes.
- [x] **3.2** `_resolve_hints(circuit)` in `native_backend.py`: `(x, y)` passes through; `(layer, slot)` → `(layer * LAYER_PX, slot * SLOT_PX)` with `LAYER_PX = 120.0`, `SLOT_PX = 80.0`. Mixed-form hints (both `(x, y)` and `(layer, slot)` set) prefer absolute coords. Hints resolving to identical absolute coords raise `ValueError` with both component names.
- [x] **3.3** `render_native` calls `_resolve_hints(circuit)` before building the ELK graph; the no-hints path stays byte-identical to Phase 1 (regression-tested by `test_render_no_hints_matches_phase1_baseline`).
- [x] **3.4** `_apply_position_hints(laid_out, hints, components, skin)` post-processes ELK's output for the hinted path: overrides cell coordinates and rewrites every edge touching a hinted cell with an L-route (one horizontal + one vertical segment via a single elbow point). Tests assert each `<g transform="translate(...)">` lands within ±5 px of the expected grid cell for both code-built and YAML-loaded buck/RC hints.
- [x] **3.5** `test_render_no_hints_matches_phase1_baseline` confirms two runs of the same un-hinted circuit produce byte-identical SVG.
- [x] **3.6** `test_resolve_hints_detects_conflict` asserts duplicate-coord hints raise `ValueError("Position hint conflict: components 'X' and 'Y' both resolve to (x, y). …")`. Behavior documented as the deterministic-error choice (per design's Risks section).

## Phase 4 — Topology-aware auto-hints

Goal: known topologies get textbook layouts out of the box.

- [x] **4.1** Canonical layouts in `native_backend.py` as `_TOPOLOGY_CANONICAL_LAYOUTS`: `bridge_rectifier` (2×2 diamond — D1/D3 top, D2/D4 bottom), `boost_stage` (L top-left, Q below, D top-right), `half_bridge` (Q_hi above Q_lo).
- [x] **4.2** `_auto_hints(circuit, user_hints)` runs `templates.recognize_all`, walks matches in order, emits translated `(x, y)` for every matched role that's NOT already user-hinted. Multiple recognized topologies in the same Circuit are stacked left-to-right via a running `layer_offset` so they don't collide. Best-effort: never raises on collision (silent first-match-wins).
- [x] **4.3** `_resolve_hints(circuit)` splits into `_resolve_user_hints` + `_auto_hints` and merges with `{**auto, **user}` (user wins). User-hint conflict detection stays on the user-only path.
- [x] **4.4** Visual confirmation: half-bridge example now stacks S_hi above S_lo automatically; rendered to `/tmp/hb_auto.png`. A code-built bridge rectifier (D1..D4 in the canonical anode/cathode arrangement) auto-places into the 2×2 diamond. Buck (`examples/buck_converter.yaml`) doesn't match any of the current 3 recognizers — that's expected; specific buck-topology recognition is out of Phase 4 scope.
- [x] **4.5** `test_auto_hints_half_bridge` + `test_auto_hints_bridge_rectifier_diamond` assert auto-hints land in the expected grid cells without any user `set_position` calls.
- [x] **4.6** `test_auto_hints_skip_user_hinted_components` builds a half-bridge, pins `S_hi` to a far-right `(999, 999)`, confirms the user hint wins (S_hi stays at 999,999) while S_lo still gets the auto-hint.

## Phase 5 — Switch the default backend; deprecate netlistsvg

Goal: `pulsim.schematic.render(...)` calls the new renderer by default. netlistsvg stays as opt-in for one release.

- [ ] **5.1** Update `python/pulsim/schematic/render.py` / `layout.py` dispatcher: `PULSIM_SCHEMATIC_BACKEND` unset → new `python_native` backend. `=netlistsvg` still works but prints `DeprecationWarning("netlistsvg backend will be removed in pulsim 0.12. Set PULSIM_SCHEMATIC_BACKEND=python_native or unset to silence.")`.
- [ ] **5.2** Update `.github/workflows/schematic-smoke.yml`: build target stays `_pulsim`; the demo render uses the new default. The job no longer needs `npm install netlistsvg` — only `elkjs`.
- [ ] **5.3** Update `docs/schematic-rendering.md`: rewrite the "Install" and "Backends" sections; document `position:` (YAML) and `set_position` (Python). Move the netlistsvg section under a "Legacy backends" heading.
- [ ] **5.4** Update `pyproject.toml` `[project.optional-dependencies] schematic`: drop the implicit `netlistsvg` doc reference (it was never a Python dep, just a doc note).
- [ ] **5.5** Smoke-test on a clean checkout: `pip install '.[schematic]' && python -c "import pulsim as ps; ckt = ...; ps.schematic.render(ckt, 'out.svg')"` works without any `npm install netlistsvg` step.

## Phase 6 — Cross-cutting docs, CI, archive

- [ ] **6.1** Update `openspec/specs/*/spec.md` deltas as specified in `specs/` of this change (handled by `openspec archive`).
- [ ] **6.2** Update `CHANGELOG.md` (or `docs/release-notes/` if that's the convention) noting the default-backend switch and the deprecation timeline.
- [ ] **6.3** Run `openspec validate add-python-schematic-renderer --strict` clean.
- [ ] **6.4** Archive `add-schematic-position-hints` separately once this change lands: its findings are now folded into this proposal's design.md, and the API surface it described is implemented here.
- [ ] **6.5** PR description: link the prior PR #6 (initial render) and the schematic-smoke CI run on the demo set; paste before/after PNGs for buck + boost-pfc to show the textbook layout improvement.

## Phase 7 — (Optional, feasibility-only) — drop Node entirely

This is a SEPARATE go/no-go investigation. Don't start until Phase 5 is in production for at least one release and we've measured user impact.

- [ ] **7.1** Spike: implement a minimal Sugiyama layered layout in Python over the same yosys-JSON input. Use `networkx` for the layering primitives. Render the demo set. Compare visually + by metric (edge crossings, total wire length) to elkjs output.
- [ ] **7.2** If the spike's output is within ~10 % of elkjs on the demo set, propose a follow-up change `replace-elkjs-with-python-layout`. Otherwise, archive the spike and keep elkjs.

---

## Validation gates

Each phase must pass these before the next starts:

- Phase 1: `pytest python/tests/test_schematic_native_backend.py -q` passes. Visual diff on demo set OK.
- Phase 2: `./build/core/pulsim_simulation_tests "position_hint:*"` passes. YAML round-trip test passes.
- Phase 3: pin-and-render test passes; no-hints regression matches Phase 1 baseline byte-for-byte.
- Phase 4: topology-aware tests pass; demo set re-renders look textbook-like.
- Phase 5: `Schematic smoke render` CI green; no Node `npm install netlistsvg` line remaining in CI logs.
- Phase 6: `openspec validate --all --strict` passes.
