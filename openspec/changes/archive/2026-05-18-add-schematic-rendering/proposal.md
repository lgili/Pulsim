## Why

Today users build circuits via `Circuit.add_*()` in Python or via YAML netlists, but there is no way to visually verify that the topology was wired the way they intended. A misplaced node yields silent miswiring that only surfaces as a wrong simulation result hours later. The same gap blocks a future GUI from auto-placing components — there is no public API that returns where each component should sit on a canvas.

We add a `schematic-rendering` capability that:
1. Renders a Circuit to SVG/PNG so the user can eyeball the topology.
2. Exposes a layout API returning canvas coordinates per component, consumable by a GUI for auto-place.

The first deliverable is intentionally an MVP: force-directed layout + `schemdraw` symbols. Correct and legible, not "looks-like-LTspice". Topology recognizer, orthogonal wire routing, and hierarchical blocks are explicit follow-ups.

## What Changes

### Circuit Component Introspection (kernel + bindings)
- ADD: `Circuit.components()` returning `vector<ComponentDescriptor>` enumerating every physical device added. Each entry exposes `name`, `kind` (canonical string), `nodes` (terminal node IDs in pin order), and a `params` map keyed by parameter name.
- ADD: `Circuit.node_position_hint(node_id)` returning a role classification (`ground`, `source_pos`, `source_neg`, `load`, `internal`) consumed by the layout engine as a placement prior.
- ADD: pybind11 binding for `components()`, `node_position_hint()`, plus the `ComponentDescriptor` value type with read-only `name`, `kind`, `nodes`, `params` (as a Python dict).
- COMPAT: All `add_*` APIs unchanged. No new required arguments. Existing circuits enumerate identically across runs (insertion order, deterministic).

### Schematic Rendering (new Python module)
- ADD: `pulsim.schematic` module exposing:
  - `render(circuit, path, format="auto", theme="light")` — one-shot render to file (PNG or SVG inferred from extension; explicit `format` overrides).
  - `compute_layout(circuit) -> SchematicLayout` — returns a canvas-agnostic placement consumed by the GUI for auto-place.
  - `render_layout(layout, path, ...)` — render a pre-computed layout.
- ADD: `SchematicLayout` value type with `components: dict[str, ComponentPlacement]`, `wires: list[Wire]`, `junctions: list[(float, float)]`, `canvas: BoundingBox`, plus `to_json()` / `from_json()`.
- ADD: `ComponentPlacement` carrying `(x, y)` in mm, rotation ∈ {0, 90, 180, 270}, `symbol` (schemdraw element name), and per-terminal anchor coordinates.
- ADD: Layout engine: force-directed (Fruchterman-Reingold via `networkx.spring_layout`) with electrical priors — ground pinned south, voltage sources pinned west. Wires drawn as straight lines (orthogonal routing is a follow-up).
- ADD: Schemdraw-based render to SVG; PNG via schemdraw native export.
- ADD: New optional dependencies `schemdraw>=0.18` and `networkx>=3.0` declared in `pyproject.toml` under a `[schematic]` extra so the default install stays slim.

### Out of scope (explicit follow-ups)
- Topology recognizer (buck/boost/half-bridge/full-bridge templates) — separate change.
- Orthogonal Manhattan wire routing with junction detection — separate change.
- Hierarchical block abstraction (collapse half-bridge into a block) — separate change.
- Edit-in-GUI → mutate Circuit (reverse direction) — separate change.

## Impact

- Affected specs:
  - `kernel-v1-core` (ADDED: Circuit Component Introspection, Node Position Hint)
  - `python-bindings` (ADDED: Component Introspection Binding, Schematic Rendering Surface)
  - `schematic-rendering` (NEW capability)
- Affected code:
  - `core/include/pulsim/v1/runtime_circuit.hpp` — adds `components()` accessor + `ComponentDescriptor` type
  - `core/include/pulsim/v1/components/` — minor: each device exposes its canonical kind string + terminal ordering metadata if not already present
  - `python/bindings.cpp` — binds `Circuit.components()`, `ComponentDescriptor`
  - `python/pulsim/schematic/` — new submodule: `layout.py`, `render.py`, `symbols.py`, `__init__.py`
  - `python/pulsim/__init__.py` — re-export `schematic`
  - `python/tests/test_schematic_render.py` — new test file
  - `pyproject.toml` — adds `[project.optional-dependencies] schematic = ["schemdraw>=0.18,<1.0", "networkx>=3.0"]`
  - `docs/schematic-rendering.md` — new user guide
- New runtime dep (optional): `schemdraw`, `networkx` (behind `[schematic]` extra; default install unchanged)
