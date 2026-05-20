# Design — Python Schematic Renderer with Position Hints

## Context

`add-schematic-rendering` shipped a Pulsim → SVG pipeline whose default backend (`netlistsvg`) is a Node.js library we shell out to. The follow-up `add-schematic-position-hints` documented exactly why the obvious path forward — patch netlistsvg to accept user-supplied positions — is a dead end (upstream is unmaintained; the two blocker bugs need a fork that's painful to keep alive).

This document captures the architectural decisions for the Python rewrite committed to in `proposal.md`. The decisions are:

1. Which parts of the pipeline get re-implemented (renderer vs. layout) in phase 1 vs. later.
2. How position hints flow from Python/YAML → kernel → layout engine → SVG.
3. How topology-aware auto-layouts compose with manual hints.
4. Compatibility and deprecation of the existing netlistsvg backend.

## Goals

- **Drop the Node-side rendering**: parse the analog SVG skin and compose the output SVG entirely in Python. netlistsvg becomes legacy.
- **Make position hints first-class**: per-component (layer, slot) or (x, y), in YAML or Python, persisted on the Circuit so they survive the round-trip.
- **Make layouts look right out of the box for canonical power-electronics topologies**: when we recognize a bridge rectifier, a half-bridge, or a boost stage, default the hints to a textbook arrangement before the user even thinks to pin anything.
- **No regressions in visual quality** on the current demo set (rc, buck, half-bridge, pfc).
- **Same GUI contract**: `compute_layout(circuit).to_json()` keeps its `schematic-v1` schema.

## Non-Goals

- Replacing elkjs with a pure-Python layout engine in this change. Phase 4 of `tasks.md` scopes a feasibility study; the actual rewrite is its own proposal if the study says it's worth it.
- Interactive canvas/editor. We render to SVG; GUIs draw their own canvas from the JSON.
- New analog symbols. Skin coverage stays at today's set (resistor, capacitor, inductor, voltage/current source variants, diode, mosfet/igbt, vcswitch, transformer, ground, generic fallback).

## Decisions

### Decision 1 — Re-implement the renderer in Python; keep ELK in Node (Phase 1)

**What we do:** read `pulsim_analog.svg`, extract every `<g s:type="...">` definition with its port anchors (`<g s:x="..." s:y="..." s:pid="...">`), then instantiate each component as an `<svg:use>` (or inlined `<g>`) at the layout coordinates. Wire bend points come from the same elkjs layout JSON that the netlistsvg backend consumes today.

**What we don't do (yet):** rewrite the layout engine. elkjs stays — we already vendor `elk_bridge.js` for the ELK backend; the new renderer reuses it.

**Why:**

- 80 % of the value (drop netlistsvg, enable position hints, enable topology-aware layouts) comes from owning the **rendering** step, not the layout step.
- ELK's layered Sugiyama algorithm is high-quality and well-tested. A pure-Python replacement is months of work that we can defer once we measure whether it's worth it.
- The two upstream bugs blocking position hints are both in netlistsvg's renderer, not in elkjs. Replacing netlistsvg's renderer with our own removes the blockers cleanly: we apply hints by sending `position` constraints into elkjs, then ELK returns consistent layout + wires.

**Trade-off:** Node.js stays in the dependency graph for now. That's an acceptable cost for one release; users already have Node installed (the previous default required it). Phase 4 tracks the removal study.

### Decision 2 — Position hints live on the kernel `Circuit`, not in a Python-side dict

**What we do:** the C++ `Circuit` grows a parallel `std::vector<PositionHint>` (one per registered device, or sparse map keyed by device index — TBD in implementation). Same lifecycle as `connections_`. Accessor methods: `set_position(name, layer, slot, x, y)`, `position_hint(name)`, `position_hints()` (snapshot). YAML parser writes hints during `parser.load(...)`. pybind11 binds it.

**Why:**

- Same precedent as `node_position_hint` from `add-schematic-rendering` — kernel-owned semantic data, Python is a view.
- Hints round-trip through YAML → Circuit → render without a Python-side cache the renderer would have to look up separately.
- A GUI that builds the Circuit programmatically and then asks for a layout sees its own hints back via `Circuit.position_hints()`.

**Alternative considered:** a renderer-only `dict[str, (x,y)]` argument to `ps.schematic.render(circuit, path, position_hints=...)`. Rejected because (a) it splits truth between two locations, (b) GUI consumers want a queryable snapshot, (c) YAML round-trip would need a sidecar dict the Python API can't see.

### Decision 3 — Two coordinate spaces: semantic `(layer, slot)` and absolute `(x, y)`

**What we do:** a hint is `(name, optional layer, optional slot, optional x, optional y)`. The kernel stores all four; one of `(layer, slot)` or `(x, y)` must be set. The renderer's pre-layout step:

- For `(layer, slot)` hints: translate to absolute coordinates using a grid — `x = layer * LAYER_PX`, `y = slot * SLOT_PX`. The default grid is 120 px per layer, 80 px per slot; configurable per-render via the layout options.
- For `(x, y)` hints: pass straight through to ELK.

ELK receives one position constraint per hinted component; un-hinted components float and ELK places them with normal layered layout.

**Why two forms:**

- `(layer, slot)` is what users actually think in for textbook layouts: "Vin in column 0 row 0, switch in column 1, diode below the switch". They don't want to compute pixel offsets.
- `(x, y)` is what GUI drag-and-drop produces, and what programmatic export from other tools (KiCad, LTSpice) gives you.

Same precedent: KiCad's `(at X Y)` is absolute, `org.eclipse.elk.position` is absolute, but power-electronics textbooks reason in terms of "stages" (columns). Supporting both forms keeps the common case ergonomic and the GUI case lossless.

### Decision 4 — Topology auto-hints are computed at render time, not stored

**What we do:** before composing the SVG, the renderer runs the existing `recognize_all()` over the Circuit. For every matched template (bridge rectifier, half-bridge, boost stage, …), it generates default `(layer, slot)` hints for the matched components — but only for components the user has NOT already hinted. Then it passes the combined hint set (user + auto) to ELK.

**Why not store auto-hints on the Circuit:** they're a function of the current set of devices. Adding a device might change a recognition; storing them would force re-derivation on every mutation. Keeping them at render time means the user's `Circuit.position_hints()` returns only what they explicitly set, which is what they expect.

**Override order (highest to lowest priority):**
1. Explicit user `set_position(...)` / YAML `position:` field.
2. Auto-hint from a recognized template (bridge_rectifier, half_bridge, boost_stage, …).
3. ELK's layered placement.

### Decision 5 — Skin SVG is parsed once, cached at module load

**What we do:** on first render, parse `pulsim_analog.svg` into a `dict[symbol_type → SymbolTemplate]` where each `SymbolTemplate` holds:
- the inner `<g>` XML (everything between the opening and closing tag of `<g s:type="...">`, minus the namespace declarations);
- a `dict[port_id → (x, y)]` from each `<g s:x s:y s:pid>` child;
- the bounding box from `<g s:width s:height>` (or computed if missing).

Cache lives at module level. A separate user-supplied skin path is honored via the existing `PULSIM_SCHEMATIC_SKIN` env var (already used by `netlistsvg_backend.py`) and invalidates the cache when changed.

**Why:** zero per-render parse cost, no runtime dependency on `xml.etree` beyond a one-time pass. The SVG file is ~30 KB; parsing it is fast.

### Decision 6 — SVG composition via `xml.etree.ElementTree`, not a templating library

**What we do:** the output SVG is built with `ElementTree.Element(...)` calls — one `<g>` per cell (with `transform="translate(X,Y) rotate(R)"`), one `<path>` per wire segment, one `<circle>` per junction. The skin's inner XML is inserted as a sub-tree (deep-copied) under each `<g>`.

**Why:** the alternative (Jinja-style template + string substitution) breaks down for nested namespaces, doesn't catch malformed input, and complicates symbol rotation. ElementTree is in the standard library, handles XML namespaces correctly, and gives us cheap deep-copy.

**Edge case:** the netlistsvg `<s:alias>` indirection (mosfet_n / r_v / etc.) needs to be resolved at parse time so the symbol-lookup dict is keyed by primary type. Existing logic in `_resolve_skin()` already handles this; we lift it into the new module.

### Decision 7 — Backend selection: switch the default, keep netlistsvg as legacy

**Current state (after `add-schematic-rendering`):**
- `PULSIM_SCHEMATIC_BACKEND` unset / `netlistsvg`: netlistsvg via Node subprocess (current default)
- `=elk`: schemdraw + ELK Python-side layout
- `=spring`: schemdraw + force-directed (no Node)

**After this change:**
- unset / `python_native` (new default): the new pure-Python renderer
- `=netlistsvg`: legacy fallback, prints a `DeprecationWarning` on use, removed in a release N+2
- `=elk`, `=spring`: unchanged

**Why keep netlistsvg one release:** users with custom skins targeting netlistsvg's exact rendering can pin the old backend for one release while migrating. No user-visible breakage on day 1.

## Risks

- **Layout-quality regression.** elkjs returns the same layout JSON to netlistsvg and to our new renderer, so structural quality is identical — but visual rendering (line widths, spacing, label placement, junction radius) is in our hands now and could look subtly worse on a specific topology. **Mitigation:** the schematic-smoke CI uploads PNGs of the demo set every run; we eyeball the artifact during review.

- **Skin compatibility surprises.** netlistsvg has quirks around `s:alias`, `s:laterals`, and `genericsLaterals` that aren't fully spec'd anywhere. Our parser might miss one. **Mitigation:** spec out the supported subset of the netlistsvg skin format in a doc section; if a user's custom skin uses an unsupported feature, fail loudly with a useful error pointing at the doc.

- **Wire routing with hints.** Even with ELK respecting `position` constraints, very-tight user-supplied positions can force ELK to produce zero-length or overlapping segments. **Mitigation:** validate hint distances at the API surface (`set_position` rejects values closer than a configurable minimum) and add a regression test for "user pins two components on the same coords → renderer raises".

- **Phase 4 scope creep (Node removal).** Replacing elkjs is genuinely hard. **Mitigation:** keep Phase 4 as a feasibility study with a clear go/no-go gate; if the study finds the quality gap is wide, that work moves to a separate proposal and we stay on elkjs indefinitely.

- **Performance regression on large circuits.** netlistsvg has been optimized over years; our first cut may be slower for 100+ component circuits. **Mitigation:** add a perf assertion to the schematic test suite — `render(buck_circuit)` should complete in < 250 ms on CI hardware. If it doesn't, file a follow-up before the default switches.

## Migration / Rollout

1. Land the new backend alongside netlistsvg (Phase 1-3 of tasks). Default stays on netlistsvg.
2. Switch the default to `python_native` (Phase 5). netlistsvg backend stays selectable, prints a `DeprecationWarning`.
3. One release later: remove netlistsvg backend, drop the `npm install netlistsvg` line from docs/CI, keep `elkjs` until Phase 4 lands.

## Open Questions

- Should `(layer, slot)` be uniform across templates or template-aware (a "bridge rectifier" diamond uses `slot=0.5`)? Default to uniform grid for V1; revisit if textbook output looks visibly wrong.
- Should we honor ELK's `org.eclipse.elk.position` for both `(layer, slot)` and `(x, y)` hints, or only `(x, y)`? Honor both — `(layer, slot)` gets translated to `(x, y)` at the render edge before ELK sees it.
- Do we expose the LAYER_PX / SLOT_PX grid as user-tunable? Defer — add when someone asks.
