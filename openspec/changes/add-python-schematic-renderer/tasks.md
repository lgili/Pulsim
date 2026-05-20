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

- [ ] **2.1** Add `struct PositionHint` to `core/include/pulsim/v1/runtime_circuit.hpp`: `std::optional<int> layer`, `std::optional<int> slot`, `std::optional<double> x`, `std::optional<double> y`. Invariant: at least one of `(layer, slot)` or `(x, y)` is set.
- [ ] **2.2** Add private `std::unordered_map<std::string, PositionHint> position_hints_` to `Circuit`. Use `unordered_map` (not `map`) for the same clang-17 / libstdc++-14 reason captured in PR #10's commit message.
- [ ] **2.3** Add public accessors on `Circuit`:
  - `void set_position(std::string_view name, std::optional<int> layer, std::optional<int> slot, std::optional<double> x, std::optional<double> y);`
  - `[[nodiscard]] std::optional<PositionHint> position_hint(std::string_view name) const;`
  - `[[nodiscard]] std::unordered_map<std::string, PositionHint> position_hints() const;` (snapshot)
- [ ] **2.4** `set_position` validates: device with `name` must exist (or warn + ignore — choose at impl time and document in spec); at least one coordinate must be set; if both `(layer, slot)` and `(x, y)` provided, prefer absolute and emit a warning channel entry.
- [ ] **2.5** C++ tests in `core/tests/test_position_hints.cpp` (Catch2): set/get round-trip, snapshot is detached from mutation, missing device returns nullopt, `[layer, slot]`-only and `[x, y]`-only both supported, conflict-resolution behavior is what the spec says.
- [ ] **2.6** YAML parser update: in `core/include/pulsim/v1/parser/yaml_parser.hpp` (or its impl file), when reading each component, look for `position:` and route to `circuit.set_position(...)`. Accept both forms documented in `proposal.md`.
- [ ] **2.7** YAML round-trip test: load a YAML with `position:` hints, assert `Circuit.position_hints()` returns them; reload the same Circuit and confirm idempotence.
- [ ] **2.8** pybind11 bindings in `python/bindings.cpp`:
  - `py::class_<PositionHint>` with `def_readonly` for the four optional fields.
  - `Circuit.set_position(name, *, layer=None, slot=None, x=None, y=None)`.
  - `Circuit.position_hint(name) -> Optional[PositionHint]`.
  - `Circuit.position_hints() -> dict[str, PositionHint]`.
- [ ] **2.9** Update `python/pulsim/_pulsim.pyi` with the new symbols. Python test in `python/tests/test_position_hints.py` covers the Python surface.

## Phase 3 — Hints flow into the layout

Goal: the new renderer respects user hints.

- [ ] **3.1** Update `_run_elk_layout` to accept a `position_hints: dict[str, tuple[float, float]]` parameter. For each hinted cell, emit `layoutOptions: { "org.eclipse.elk.position": "(x,y)" }` and pin the cell's `x,y` in the input JSON.
- [ ] **3.2** Add `_resolve_hints(circuit) -> dict[str, tuple[float, float]]`: translate every `Circuit.position_hint(name)` into absolute coords. `(layer, slot)` → `(layer * LAYER_PX, slot * SLOT_PX)` with module constants `LAYER_PX = 120.0`, `SLOT_PX = 80.0`. `(x, y)` passes through.
- [ ] **3.3** Wire `_resolve_hints` into `render_native` and into `compute_layout` so both the SVG path and the JSON path see the same hints.
- [ ] **3.4** Test: build a buck circuit, pin Vdc / S1 / D1 / L1 / Cout / Rload to known `(layer, slot)` cells, render, parse the SVG, assert each `<g transform>` is within ±5 px of the expected pinned position.
- [ ] **3.5** Test: identical Circuit rendered with and without hints — without-hints output is identical to Phase 1 baseline.
- [ ] **3.6** Test: conflicting hints (two components pinned to the same `(layer, slot)`) raise a deterministic error or shift one with a warning — pick at impl time and document in the spec.

## Phase 4 — Topology-aware auto-hints

Goal: known topologies get textbook layouts out of the box.

- [ ] **4.1** Extend `python/pulsim/schematic/templates.py` (or a new sibling module): for each existing recognizer (`bridge_rectifier`, `half_bridge`, `boost_stage`), define a `canonical_layout: dict[str, tuple[layer, slot]]` mapping the recognizer's role names to default grid cells.
- [ ] **4.2** Add `_auto_hints(circuit) -> dict[str, tuple[float, float]]`: run `recognize_all(circuit)` and emit translated absolute positions for every matched component that the user has NOT explicitly hinted.
- [ ] **4.3** Merge order: user hints win, then auto-hints, then ELK's free placement. Implement in `_resolve_hints`.
- [ ] **4.4** Render the demo set after Phase 4 — eyeball that `buck`, `half-bridge`, and the boost-pfc + vsi + pmsm circuit now look closer to a textbook diagram.
- [ ] **4.5** Test: a buck Circuit with no explicit hints renders with switch and freewheel diode in the expected columns (the test checks recognizer-derived hints, not pixel positions).
- [ ] **4.6** Test: a buck Circuit with the user pinning S1 to (5, 5) — the user's hint wins over the auto-layout's preferred position.

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
