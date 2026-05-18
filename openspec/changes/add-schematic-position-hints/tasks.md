## Gates & Definition of Done

- [ ] G.1 A YAML netlist with `position: {layer, slot}` per component renders with the components at the user-specified grid coordinates (verified by hand-positioning a buck so it matches the conventional textbook layout).
- [ ] G.2 Circuits WITHOUT any `position` hints render identically to today (no silent layout drift; regression guard).
- [ ] G.3 Both `{layer, slot}` semantic form and `{x, y}` absolute form are accepted; strict YAML parser rejects unknown keys.
- [ ] G.4 Python API (`Circuit.set_position(...)`, `Circuit.position_hints()`) round-trips through the rendering pipeline.
- [ ] G.5 Wider Python suite (`pytest python/tests --ignore=python/tests/validation`) stays green; schematic suite gains the new tests and stays green.

## Phase 1: YAML schema + parser

- [ ] 1.1 Document the `position` field in `docs/netlist-format.md` (under a "Schematic position hints" section, with examples for both `{layer, slot}` and `{x, y}` forms).
- [ ] 1.2 Extend `core/src/v1/yaml_parser.cpp` to parse the optional `position` field per component. Validate keys (`layer`, `slot`, `x`, `y`); strict mode rejects unknown keys.
- [ ] 1.3 Add a `PositionHint` struct in `core/include/pulsim/v1/runtime_circuit.hpp` carrying `(layer, slot, x, y)` as `std::optional<int>` each (semantic or absolute, mutually exclusive).
- [ ] 1.4 Store hints in `Circuit` as `std::unordered_map<std::string, PositionHint>` (keyed by component name); expose `position_hint(name)` getter.

## Phase 2: Python API surface

- [ ] 2.1 Bind `PositionHint` value type + `Circuit.set_position(name, layer=, slot=, x=, y=)` setter in `python/bindings.cpp`.
- [ ] 2.2 Bind `Circuit.position_hints()` returning a dict snapshot.
- [ ] 2.3 Type stubs in `python/pulsim/_pulsim.pyi`.
- [ ] 2.4 Python unit tests in `python/tests/test_components_introspection.py` (round-trip through code-built and YAML-loaded circuits).

## Phase 3: netlistsvg backend integration

- [ ] 3.1 Add `_export_default_layout(yosys_json_path, skin_path) -> dict` to `netlistsvg_backend.py` — wraps a subprocess call to `node_modules/netlistsvg/bin/exportLayout.js`, returns the parsed JSON.
- [ ] 3.2 Add `_apply_position_hints(layout, hints, grid_step=120, slot_step=80) -> layout` — for each component with a hint, override the matching child's `x, y` in the layout JSON.
- [ ] 3.3 Modify `render_netlistsvg` to: if any position hints exist on the circuit, run exportLayout → apply hints → save edited layout → run netlistsvg with `--layout`. Otherwise stay on the existing fast path (single netlistsvg invocation).
- [ ] 3.4 Threshold check: if a hint moves a cell more than 200 px from its auto-layout position, log a `UserWarning` ("auto-routed wires may overlap; consider hinting the cell's neighbours as well").

## Phase 4: Tests + docs

- [ ] 4.1 Add `examples/buck_converter_positioned.yaml` — same circuit as `buck_converter.yaml` but with explicit `position` hints producing the conventional textbook layout: Vdc (0,0) → S1 (1,0) → L1 (3,0) → vout area; D1 (2,1) vertical; C1/Rload (4,0)/(4,1) vertical; Vpwm (1,1) below S1.
- [ ] 4.2 Add `test_render_respects_position_hints` in `python/tests/test_schematic_render.py` — renders `buck_converter_positioned.yaml`, asserts the SVG places Vdc / S1 / L1 at the expected grid coordinates (within ±5 px tolerance).
- [ ] 4.3 Add `test_render_no_hints_matches_auto_layout` — regression guard: a circuit with no `position` fields renders identically to the same circuit rendered without the hint pipeline. (Compare SVG byte length OR re-parse positions and confirm equality.)
- [ ] 4.4 Extend `docs/schematic-rendering.md` with a "When auto-layout isn't enough: position hints" section — example YAML + the resulting render side-by-side with the auto-layout version.
