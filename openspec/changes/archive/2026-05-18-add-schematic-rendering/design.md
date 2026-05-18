## Context

PulsimCore today builds circuits via two entrypoints:
1. Code-built circuits using `Circuit.add_resistor(...)`, `add_mosfet(...)`, etc.
2. YAML netlists parsed via `YamlParser.load(...)` (schema `pulsim-v1`).

Neither path produces a visual artifact. The user cannot eyeball that node `sw_node` of `S1` is the same one feeding the inductor `L1`. Miswiring is silent — the simulation runs and produces nonsense.

A future GUI (PulsimGui) needs more than a render: it needs **structured coordinates** so the canvas can auto-place components without forcing the user to drag every R/L/C into position. That data has to come from a stable, JSON-serializable API.

This change introduces:
- A kernel-level enumeration API (`Circuit.components()`) so the rendering layer is decoupled from how the circuit was built.
- A pure-Python `pulsim.schematic` module that consumes that enumeration and produces an SVG/PNG file plus a JSON layout for the GUI.

Schematic auto-layout is a research-grade problem; this change deliberately scopes to an MVP that proves the data flow end-to-end. Topology recognizer and orthogonal routing are explicit follow-ups (see Non-Goals).

## Goals

- Correct topology visualization — every component and wire in `circuit.components()` is rendered.
- Stable layout — same circuit produces same coordinates across runs (deterministic seed).
- GUI-consumable coordinates — millimeter units, JSON-serializable, no Python-specific types in the payload.
- Slim default install — `schemdraw` and `networkx` are optional, behind a `[schematic]` extra.
- Zero impact on simulation hot path — the introspection accessor is read-only and additive.

## Non-Goals

- LTspice-quality aesthetic. MVP is correct and legible, not pretty.
- Manhattan / orthogonal wire routing with junction dots.
- Topology recognizer for buck / boost / half-bridge templates.
- Hierarchical block abstraction (collapsing a half-bridge into a single box).
- Editable schematic — round-trip from GUI mutations back to Circuit changes.
- Schematic export to Altium / KiCad / Eagle formats.

## Decisions

### Where component enumeration lives
**Decision:** Add `Circuit.components()` in the kernel (`runtime_circuit.hpp`), exposed via pybind11.
**Alternatives considered:**
- YAML-only enumeration → rejected: would not cover code-built circuits.
- Python-side wrapper intercepting `add_*` → rejected: would not cover circuits built via the raw `_pulsim.Circuit` or `_pulsim.YamlParser`.
**Rationale:** Single source of truth. Works for every entrypoint, current and future.

### Render backend
**Decision:** `schemdraw` (Python-pure, MIT-licensed).
**Alternatives considered:**
- `lcapy` + circuitikz → heavy LaTeX runtime dep.
- Custom SVG primitives → months of symbol design before first render.
- Graphviz `dot`/`neato` → no electrical symbols; output looks like a graph, not a schematic.
**Rationale:** Ships with R/L/C/D/MOSFET/IGBT/V/I/GND/transformer symbols out of the box. Native SVG+PNG. Pure Python — no compilation step, no system libraries beyond pip.

### Layout algorithm
**Decision:** Fruchterman-Reingold (force-directed) via `networkx.spring_layout`, deterministic seed `42`, with per-node anchor constraints:
- Ground node fixed near `(canvas.width/2, canvas.height)` (bottom-center).
- Voltage source positive terminals pulled toward `x = 0` (left edge).
- Current source positive terminals and resistive load nodes pulled toward `x = canvas.width` (right edge).
- All other (internal) nodes free.
**Alternatives considered:**
- Sugiyama / hierarchical (layered) layout → great for digital block diagrams, awkward for power loops where the same node appears in multiple feedback paths.
- Manual templates only → not general; useless for unrecognized topologies.
- Pure spring layout without priors → produces unreadable schematics; ground floats anywhere.
**Rationale:** Simplest algorithm that produces readable output for arbitrary circuits, deterministic, and extensible — a future topology recognizer can override the layout for matched sub-graphs while the rest of the circuit still uses the spring fallback.

### Wire rendering
**Decision:** Straight lines from terminal anchor to terminal anchor in PR1. Orthogonal Manhattan routing is a follow-up change.
**Rationale:** Orthogonal routing is itself an A*-on-grid-with-obstacles problem and not blocking the visual-verification use case. Straight lines are unambiguous and good enough to confirm topology.

### Optional vs required dependency
**Decision:** Optional via `[project.optional-dependencies] schematic = [...]`. Pulsim's default install stays slim; `pip install pulsim[schematic]` enables rendering.
**Rationale:** Matches the project's existing posture of keeping core lean. CI environments that do not need schematics skip the extra wheels. Validation of the render API in CI uses the `[schematic]` extra.

### Coordinate system
**Decision:** Millimeters, origin top-left, +x right, +y down (standard CAD/screen convention). `SchematicLayout.canvas` carries the bounding box and unit string.
**Rationale:** Matches SVG and most CAD tools. GUI can scale to pixel space trivially. Avoids the silent-disaster of mixing inches and millimeters.

### JSON serialization
**Decision:** `SchematicLayout.to_json()` returns a fully JSON-serializable `dict[str, Any]`. All coordinates are floats, rotations are integers ∈ {0, 90, 180, 270}, no schemdraw object references leak.
**Rationale:** GUI runs in a separate process (and likely a separate language: TypeScript / React). The serialization boundary must be free of Python-specific or library-specific types.

### Determinism
**Decision:** `compute_layout(circuit)` is deterministic — same circuit produces byte-identical coordinates across runs.
**Rationale:** Required for CI snapshot testing and reproducible GUI behavior. Achieved by pinning the spring-layout seed and processing components in `circuit.components()` insertion order.

## Risks

- **Layout quality is mediocre for unrecognized topologies.** Mitigation: scope MVP clearly; document the limitation in `docs/schematic-rendering.md`; topology recognizer planned as immediate follow-up.
- **schemdraw API drift across versions.** Mitigation: pin `>=0.18,<1.0`; integration tests catch breakage on every CI run that installs the `[schematic]` extra.
- **`runtime_circuit.hpp` is a known friction point** (per `ROADMAP.md` — flagged for refactor in 4.1). Mitigation: keep the new introspection accessor read-only and additive; no behavioral changes; the `ComponentDescriptor` storage piggy-backs on the existing `add_*` call sites with a single `components_.push_back(...)` per add.
- **Three- and four-terminal devices** (MOSFET, IGBT, transformer) need pin-aware orientation. Mitigation: the schemdraw mapping carries pin order; ComponentPlacement.terminal_anchors expose per-pin coordinates so wires connect to the right pin without trial-and-error.
- **Optional-dep ImportError UX.** Mitigation: `pulsim.schematic` import itself succeeds without `schemdraw`/`networkx`; only `render()`/`compute_layout()` raise, with a message that names the extra (`pip install pulsim[schematic]`).

## Migration Plan

- Change is additive. No existing API is removed or modified semantically.
- Default install of pulsim continues to work as before; users opt in to rendering via the `[schematic]` extra.
- `docs/schematic-rendering.md` covers install, first render, GUI integration.
- No deprecation cycle needed.

## Open Questions

- Should virtual components (probes, control blocks) render on a separate "signal layer" or be hidden by default? **Tentative answer:** hidden by default, opt-in via `render(..., include_virtual=True)`. Final call when we have a buck-with-probes rendering side-by-side.
- Should `compute_layout()` cache results so repeated calls are O(1)? **Tentative answer:** no for PR1; profile first, add caching only if measured slow.
- Should the layout JSON include component values (e.g. `R = 10Ω`) as labels? **Tentative answer:** yes, as an opt-in `include_values=True` flag on `render()` — labels often add clutter. Hold the decision until first user feedback.
