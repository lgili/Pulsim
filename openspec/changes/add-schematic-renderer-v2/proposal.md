# Add Schematic Renderer V2 — topology-aware + LLM-augmented

## Why

The v1 `schematic-rendering` capability shipped in five phases (commits
`8e489e9` → `87dcd1e`) and produced legible SVG renders via force-directed
layout + per-component position hints. The implementation was removed
during the v1→1.0 namespace retirement (commit `bce9576`) and the
project currently has no way to visualize a circuit.

The original `Why` still holds — users build circuits via
`CircuitBuilder.add_*()` or YAML and have no way to confirm the topology
matches the intent until a wrong simulation result surfaces hours later.

But the bar has moved. The original MVP (force-directed + ground at the
bottom + sources at the left) produces a *legible* layout but not a
*recognizable* one — a buck and a boost end up looking like the same
amorphous blob to anyone who hasn't memorized the netlist. For an actual
power-electronics simulator, that's not enough. The user reaction to a
half-good schematic is "this didn't help."

This change adds a topology-aware renderer that recognizes canonical
power-electronics topologies and applies hand-tuned templates for them,
falling back to constraint-aware force-directed layout for the rest. An
LLM (Anthropic API) classifies the topology when graph heuristics are
inconclusive, with results cached locally so the API is hit at most once
per distinct netlist.

## What Changes

### Restored from `87dcd1e` (Phases 1–5)
- `pulsim.schematic` Python submodule with:
  - `compute_layout(circuit) → SchematicLayout`
  - `render(circuit, path, format="auto", theme="light")`
  - `render_layout(layout, path, ...)`
  - `SchematicLayout`, `ComponentPlacement`, `Wire`, `BoundingBox` types
  - `recognize_all(circuit)` topology recognizer (graph heuristics)
  - Native Python backend, ELK backend (node.js), netlistsvg backend
  - JSON serialization (`to_json` / `from_json`, schema `"schematic-v1"`)
  - Symbol library + skin parser (analog skin shipped)
- Optional dependencies: `schemdraw>=0.18`, `networkx>=3.0`, declared
  under `[project.optional-dependencies] schematic`.
- Test suite: `python/tests/test_schematic_render.py` (golden SVGs),
  `python/tests/test_schematic_native_backend.py`.
- CI workflow: `.github/workflows/schematic-smoke.yml`.

### New in V2 (the smart part)

#### Flat-namespace binding extensions
- `CircuitBuilder.components()` returning the `ComponentDescriptor` list
  the schematic module already expects. Each entry exposes `name`,
  `kind` (canonical string: `"resistor"`, `"voltage_source"`, `"switch"`,
  …), `nodes`, and a parameter `dict`.
- `Graph.branches` exposing the branch list with `id`, `from_`, `to`,
  `kind` enum for graph-level consumers.
- `CircuitBuilder.position_hints()` and `Circuit.node_position_hint(id)`
  for the role classification the layout engine already consumes.

#### Topology recognizer (deterministic-first, LLM-augmented second)
- **Tier 1 — graph heuristics**: recognize the 12 canonical topologies
  via structural analysis (cycle detection, port detection, branch role
  classification). Same algorithm shape the Phase 4 `recognize_all`
  used, extended with explicit detector functions:
  - SMPS: `buck`, `boost`, `buck_boost`, `flyback`, `forward`,
    `half_bridge`, `full_bridge`
  - Filters/passives: `rc_filter`, `rl_filter`, `rlc_filter`
  - Amps: `common_source`, `common_emitter`, `op_amp_inverting`,
    `op_amp_non_inverting`
  - Misc: `half_wave_rectifier`, `full_wave_bridge_rectifier`,
    `voltage_divider`
- **Tier 2 — LLM classifier**: when Tier 1 returns `None` OR returns
  a low-confidence guess, send the netlist (as a deterministic textual
  fingerprint) to the Anthropic API (default model:
  `claude-haiku-4-5`). The LLM returns one of the known topology
  names, an optional component-role mapping (e.g., `"Q_high"`,
  `"Q_low"`, `"L_main"`), and a confidence score. Always-on by default;
  opt-out via `PULSIM_LLM_LAYOUT_HINTS=0`. Local file cache keyed on
  the netlist fingerprint so a stable netlist hits the API at most
  once per `~/.cache/pulsim/topology-cache.json` entry.
- **Tier 3 — fallback**: if both tiers fail, the existing
  force-directed + electrical-prior layout runs.

#### Template library
- One YAML template per recognized topology under
  `python/pulsim/schematic/templates/<name>.yaml`, defining relative
  placements (canvas-fraction coordinates) and routing hints. Buck,
  boost, and flyback templates SHALL be visually polished and golden-
  tested. The remainder ship as best-effort and may be refined in
  follow-ups.
- The renderer SHALL match recognized topology → template → instantiate
  with the actual component IDs from the user's circuit. Unrecognized
  components within a recognized topology fall back to force-directed
  placement inside the template's reserved area.

#### Renderer outputs
- SVG remains the default (Jupyter-friendly, scalable).
- PNG export SHALL work via `cairosvg` if installed OR matplotlib
  fallback OR schemdraw native — declared optional under `[schematic]`.
- Jupyter `_repr_svg_` / `_repr_html_` on `SchematicLayout` for inline
  notebook display without an explicit `.render()` call.

### Out of scope
- Hierarchical block abstraction (collapse half-bridge into a block) —
  follow-up.
- Edit-in-GUI → mutate Circuit reverse direction — follow-up.
- Training a GNN/RL model from scratch on hand-drawn schematics — this
  proposal explicitly avoids the "train a neural network" path because
  the labeled dataset doesn't exist and building one is months of work.
  The LLM-based classifier is the practical substitute.

## Impact

- **Affected specs**:
  - `schematic-rendering` (NEW capability — re-added with V2 surface)
  - `python-bindings` (MODIFIED — add `CircuitBuilder.components`,
    `Graph.branches`, position-hint accessors)
- **Affected code**:
  - `python/pulsim/schematic/` — restored from `87dcd1e` (10 files,
    ~4 500 lines) plus new `topology_recognizer.py`,
    `llm_classifier.py`, `templates/<name>.yaml`.
  - `python/pulsim/__init__.py` — re-export `schematic` submodule.
  - `python/bindings.cpp` — bind `Graph::branches`, add
    `CircuitBuilder::components_descriptor()` adapter returning a
    Python-friendly list. Bind `BranchKind` enum.
  - `pyproject.toml` — `[project.optional-dependencies] schematic` adds
    `schemdraw>=0.18,<1.0`, `networkx>=3.0`, `cairosvg>=2.7`,
    `anthropic>=0.34`.
  - `python/tests/test_schematic_*.py` — restored + new
    `test_topology_recognizer.py`, `test_llm_classifier.py` (latter
    mocks the API).
  - `docs/schematic-rendering.md` — restored + updated user guide.
- **New runtime deps** (all optional behind `[schematic]` extra; default
  install unchanged):
  - `anthropic` (LLM classifier)
  - `cairosvg` (PNG export path)
  - `schemdraw`, `networkx` (pre-existing in archived proposal)
- **Env vars**:
  - `ANTHROPIC_API_KEY` — required when LLM classifier is enabled.
  - `PULSIM_LLM_LAYOUT_HINTS` — set to `0` to disable LLM tier
    (deterministic-only).
  - `PULSIM_LLM_MODEL` — override default `claude-haiku-4-5`.
  - `PULSIM_TOPOLOGY_CACHE_DIR` — override cache location.
