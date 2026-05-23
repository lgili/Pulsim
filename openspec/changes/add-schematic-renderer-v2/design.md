# Design — Schematic Renderer V2

## Context

The archived `schematic-rendering` capability (`2026-05-18-…`) shipped a
working force-directed renderer in five phases. The implementation was
removed during the v1 → 1.0 namespace retirement, and the user's
direct feedback on the V1 result was: *"not enough — half the circuit
looks right and the rest is unreadable."* That is the design problem
this change has to solve, not "produce *some* SVG."

## Why force-directed alone fails for power electronics

Force-directed layout (Fruchterman–Reingold and friends) treats every
node identically. A "good" layout for a generic graph minimizes edge
crossings and balances node spacing. A *power-electronics schematic*
has additional conventions that are non-negotiable for readability:

1. **Power flow direction** — sources on the left, loads on the right.
2. **Ground anchoring** — every ground-referenced node sits at the
   bottom of the canvas, not "wherever the force balance ends up."
3. **Topology-specific layout** — a buck converter has a recognizable
   shape (input cap left, switch top, freewheel diode below, inductor
   right, output cap right). A boost has a *different* recognizable
   shape (inductor first, then switch + diode + output cap). Force-
   directed produces a generic blob for both.
4. **Symmetry** — half-bridges, full-bridges, and 3-φ inverters are
   visually symmetric. Force-directed produces near-symmetry but never
   exact symmetry, which reads as wrong.

The Phase 4 `recognize_all` recognizer started addressing this, but the
fallback was still force-directed for anything unrecognized — and
unrecognized circuits got the same blob.

## The chosen approach: hybrid topology-aware

```
┌──────────────────────────────────────────────────────────────────┐
│ Stage 1 — Recognizer (deterministic, free)                       │
│   topology_recognizer.recognize(circuit) → RecognizedTopology    │
│   • Graph structural analysis (cycles, ports, branch roles)      │
│   • 12 canonical detectors, each ~30–80 lines                    │
│   • Confidence score ∈ [0, 1], 1.0 means structural match        │
└────────────────────────────────┬─────────────────────────────────┘
                                 │ confidence < 0.7 ?
                ┌────────────────┴────────────────┐
                │ yes                              │ no
                ▼                                  │
┌──────────────────────────────────────────┐       │
│ Stage 2 — LLM classifier (always on,     │       │
│   cached, ~$0.0005/call cached miss)     │       │
│   • Anthropic API claude-haiku-4-5       │       │
│   • Input: textual netlist fingerprint   │       │
│   • Output: topology name + role map +   │       │
│     confidence                           │       │
│   • Cache: ~/.cache/pulsim/topology-cache│       │
│     keyed on SHA-256 of fingerprint      │       │
└──────────────────┬───────────────────────┘       │
                   │                                │
                   └────────────┬───────────────────┘
                                ▼
            ┌────────────────────────────────────┐
            │ Stage 3 — Template instantiation   │
            │   templates/<name>.yaml → place    │
            │   each role at its canvas slot     │
            │   Unrecognized components inside   │
            │   the template area fall back to   │
            │   force-directed within bounds.    │
            └─────────────────┬──────────────────┘
                              │
                              ▼
                  ┌─────────────────────────┐
                  │ Stage 4 — Renderer       │
                  │   SVG / PNG / Jupyter    │
                  └──────────────────────────┘
```

### Stage 1 — Recognizer (deterministic)

Implemented as a pure-Python module `topology_recognizer.py`. For each
known topology, a `detect_<name>(graph, components) → float`
function returns a confidence ∈ [0, 1]:

- **1.0** = structural match (right component count, right kinds,
  right connectivity).
- **0.5–0.9** = partial match (missing one component, or extra
  unidentified components).
- **< 0.5** = treat as "no match."

Examples:
- `detect_buck`: needs exactly one voltage source, one switch, one
  freewheel diode, one inductor, one output capacitor, optional load
  resistor. The freewheel diode anode connects to switch source AND
  ground. Returns 1.0 when all six structural constraints are met.
- `detect_half_bridge`: needs exactly two switches in series between
  Vdd and GND, midpoint going to load. Returns 1.0 on exact match,
  0.6 if there are anti-parallel diodes too (still a half-bridge,
  just with explicit body diodes).

The dispatcher picks the topology with the highest confidence; ties
break by detection order (most-specific first).

### Stage 2 — LLM classifier

Invoked when Stage 1 returns `confidence < 0.7` (configurable
threshold). The LLM gets a deterministic textual fingerprint of the
circuit:

```
COMPONENTS:
  Vin: voltage_source, terminals=[vin, gnd]
  Q1: switch, terminals=[vin, sw]
  D1: diode, terminals=[gnd, sw]
  L1: inductor, terminals=[sw, vout]
  Cout: capacitor, terminals=[vout, gnd]
  Rload: resistor, terminals=[vout, gnd]

NODES: vin, gnd, sw, vout
```

The model receives a system prompt that lists the 17 supported
topology names and constrains the output to JSON:

```json
{
  "topology": "buck",
  "confidence": 0.95,
  "role_map": {
    "Vin": "source", "Q1": "switch_high", "D1": "freewheel_diode",
    "L1": "inductor_main", "Cout": "output_capacitor",
    "Rload": "load_resistor"
  }
}
```

Caching is mandatory because the same circuit must produce the same
layout across runs. The cache key is `sha256(fingerprint)`; the cache
value is the parsed JSON response. Cache lives in
`~/.cache/pulsim/topology-cache.json` by default.

**Why Haiku and not Sonnet/Opus**: topology classification is a
narrow task with a small fixed vocabulary. Haiku-4-5 hits this 99%+
of the time at ~1/8th the cost of Sonnet (~$0.0005/call vs ~$0.004).
Override with `PULSIM_LLM_MODEL` if a user wants Sonnet for
edge-case circuits.

**Failure modes** — the renderer SHALL never break because the LLM is
unreachable. If `ANTHROPIC_API_KEY` is missing, the network errors, or
the JSON is malformed, the classifier returns `None` and Stage 3 falls
through to force-directed.

### Stage 3 — Template instantiation

A template is a YAML file declaring:
- `name` — canonical topology name
- `roles` — list of role names the template needs (e.g., `["source",
  "switch_high", "freewheel_diode", "inductor_main",
  "output_capacitor", "load_resistor"]`)
- `slots` — dict mapping role → `(x_frac, y_frac, rotation)` where
  `x_frac, y_frac ∈ [0, 1]` are canvas-fraction coordinates and
  rotation ∈ {0, 90, 180, 270}.
- `wires` — explicit wire paths between role pairs (orthogonal hints).
- `canvas_aspect` — preferred aspect ratio for this topology
  (e.g., buck is 1.6:1, full-bridge is 1.0:1).

Templates are visually polished by hand. Buck, boost, and flyback
SHALL ship with a golden SVG that the test suite compares against.

The instantiator:
1. Reads the template.
2. For each role, picks the actual component from the user's circuit
   via the role_map (from Stage 1 or 2).
3. Places the component at the template slot scaled to the actual
   canvas size.
4. Routes wires per template hints; falls back to orthogonal A*
   routing for any wire the template didn't specify.

### Stage 4 — Renderer

Renderer changes from the archived V1 are limited:
- SVG output is unchanged (still via `schemdraw`).
- PNG output gets a `cairosvg` path (faster, better fonts than
  schemdraw's matplotlib PNG).
- Jupyter `_repr_svg_` so that returning a `SchematicLayout` from a
  cell auto-displays the schematic.

## Why the LLM call is the right "AI" here

Re-stating the design rejection upfront so the reasoning is explicit:

1. **Training a GNN/RL model from scratch** requires a labeled dataset
   of (netlist, human-drawn-layout) pairs. This dataset does not
   exist publicly; building it would be months of manual labeling.
   The expected gain over template + heuristics is small because power
   electronics is a small, well-known design space.

2. **Pure force-directed** is what V1 shipped. It produces "good enough"
   for trivial cases and "unreadable" for anything past 8 components.
   User's exact words.

3. **SAT/ILP** is academic-correct but doesn't scale past ~30 components
   and produces solutions that aren't recognizable to a human (optimal
   ≠ readable).

4. **LLM-as-classifier** uses the LLM for what it is good at — pattern
   matching on text — and is honest about not asking it to do pixel-
   level placement (LLMs are bad at coordinates). The LLM answers a
   small-vocabulary classification question: "given this netlist,
   which of these 17 topologies is it?" That is a well-posed task
   the LLM solves at ≥99% accuracy on the canonical cases, with the
   cost amortized by the on-disk cache.

The combined system is deterministic for repeat calls (cache hits),
graceful under network failure, and respects the conventions a power-
electronics user expects.

## Caching strategy detail

```python
def topology_cache_key(circuit_fingerprint: str) -> str:
    return hashlib.sha256(circuit_fingerprint.encode()).hexdigest()
```

`circuit_fingerprint` is a canonical, sorted, whitespace-normalized
textual representation of `(components, nodes, ground_id)`. Component
order is sorted by name; node order is sorted alphabetically; ground
is rendered as `<gnd>` regardless of its actual index. This makes the
fingerprint stable across `add_*` insertion order or trivial renames.

Cache schema (`~/.cache/pulsim/topology-cache.json`):
```json
{
  "schema_version": "topology-cache-v1",
  "entries": {
    "<sha256>": {
      "topology": "buck",
      "confidence": 0.95,
      "role_map": { "Vin": "source", … },
      "model": "claude-haiku-4-5",
      "cached_at": "2026-05-23T11:42:00Z"
    }
  }
}
```

Cache can be invalidated by deleting the file or by changing
`PULSIM_TOPOLOGY_CACHE_DIR`. The cache is read-mostly; writes only
happen on Stage 2 LLM hits, so contention is not a concern.

## Test strategy

- **Recognizer**: each of the 17 topologies gets a unit test
  constructing a representative circuit and asserting `recognize()`
  returns the expected topology with confidence ≥ 0.9.
- **LLM classifier**: mocked tests using `responses` or the Anthropic
  SDK's mock support. One real-API smoke test gated behind
  `PULSIM_LLM_REAL_API=1` so CI can skip it.
- **Templates**: buck, boost, flyback render golden SVGs (committed
  to the test fixtures). Test asserts `render(...) == fixture` byte-
  identical OR a structural equivalence check (component positions
  match within tolerance, wires connect the same anchors).
- **Determinism**: same circuit rendered twice produces the same SVG
  (already a V1 requirement, kept in V2).
- **Cache**: first call hits the network (mocked), second call reads
  the cache file, third call after deleting the cache hits the
  network again.

## Migration / compatibility

- The archived spec `schematic-rendering` was REMOVED when the v1
  retirement landed; there is no current spec to MODIFY. This change
  ADDs the capability afresh under the same name.
- Users of the archived V1 `pulsim.schematic.render(circuit, path)` API
  get the same call signature. The only observable difference is the
  output looks better for the 12 recognized topologies.
- The `[schematic]` extra dependency set gains `anthropic` and
  `cairosvg`. Existing users with `pip install pulsim[schematic]` get
  these on next install; the default `pip install pulsim` install
  remains slim and does not pull either.

## Risks and mitigations

1. **LLM hallucinates a topology name**: the prompt constrains the
   output to a closed enum and the parser validates against the
   `KNOWN_TOPOLOGIES` set. Unknown names → `None`, fall through to
   force-directed.

2. **Cache file corruption / stale schema**: cache file has a
   `schema_version` field; on mismatch the cache is dropped and
   rebuilt.

3. **Cache concurrency**: parallel `pytest -n auto` runs could race on
   the cache file. Mitigation: per-process tmpdir for tests
   (`PULSIM_TOPOLOGY_CACHE_DIR=$TMPDIR/$$`); production users rarely
   parallelize schematic rendering.

4. **Template aesthetic drift**: hand-tuned templates need maintenance
   when symbol library changes. Mitigation: golden SVG tests catch
   accidental visual regressions in CI.

5. **API cost**: at worst-case ~$0.0005/call uncached, ~10 000 cached
   calls/day is < $5/month. The cache makes this effectively free
   after first use per circuit.
