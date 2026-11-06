# Tasks — Add Schematic Renderer V2

## Phase 1 — Foundation (restore archived module + flat-namespace bindings)
- [x] 1.1 Restore the 10 files in `python/pulsim/schematic/` from commit
  `87dcd1e` (preserve the `skin/pulsim_analog.svg` skin asset).
  (Reconciled 2026-11: python/pulsim/schematic/ has these files and
  more, merged in 41f096df.)
- [ ] 1.2 Restore `python/tests/test_schematic_render.py` and
  `python/tests/test_schematic_native_backend.py` from the same commit.
  (Reconciled 2026-11: neither test file exists.)
- [x] 1.3 Restore `.github/workflows/schematic-smoke.yml`.
- [x] 1.4 Add `python/bindings.cpp` bindings:
  `Graph.branches` → list of branch records with `id`,
  `from_`, `to`, `kind`; expose `BranchKind` enum.
  (Reconciled 2026-11: branch records are dicts; BranchKind is
  pulsim._pulsim.BranchKind.)
- [x] 1.5 Add `python/bindings.cpp` bindings:
  `CircuitBuilder.components()` → list of `ComponentDescriptor`
  (name, kind canonical string, nodes, params dict).
  (Reconciled 2026-11: returns dicts with name, kind, nodes, params,
  branch_id.)
- [x] 1.6 Add `python/bindings.cpp` bindings:
  `CircuitBuilder.node_position_hint(node_id)` and
  `position_hints()` accessor; stub returning `"internal"` for V0 so
  the layout module gets the API it expects.
- [x] 1.7 Update `python/pulsim/__init__.py` to re-export
  `schematic` submodule.
- [x] 1.8 Update `pyproject.toml` `[project.optional-dependencies]`
  `schematic` to list `schemdraw>=0.18,<1.0`, `networkx>=3.0`,
  `cairosvg>=2.7`, `anthropic>=0.34`.
- [x] 1.9 Build (`cmake --build build -j 8`) and confirm
  `import pulsim.schematic` succeeds.
- [ ] 1.10 Run `pytest python/tests/test_schematic_*` and confirm all
  restored tests pass against the new bindings.
  (Reconciled 2026-11: no python/tests/test_schematic_* file exists.)

## Phase 2 — Topology recognizer (deterministic tier)
- [x] 2.1 Create `python/pulsim/schematic/topology_recognizer.py`
  with the public function
  `recognize(circuit) → RecognizedTopology | None`.
- [x] 2.2 Define `RecognizedTopology` dataclass: `name: str`,
  `confidence: float`, `role_map: dict[str, str]`, `source:
  Literal["heuristic", "llm", "cache"]`.
- [x] 2.3 Implement detector functions for 12 canonical topologies:
  `detect_buck`, `detect_boost`, `detect_buck_boost`,
  `detect_flyback`, `detect_forward`, `detect_half_bridge`,
  `detect_full_bridge`, `detect_rc_filter`, `detect_rl_filter`,
  `detect_rlc_filter`, `detect_half_wave_rectifier`,
  `detect_full_wave_bridge_rectifier`.
- [x] 2.4 Write `python/tests/test_topology_recognizer.py` with one
  test per topology constructing a representative circuit and
  asserting confidence ≥ 0.9.
  (Reconciled 2026-11: the test asserts confidence >= 0.85, and >= 0.7
  for forward.)
- [x] 2.5 Add an inverse test: each recognizer SHALL return
  `confidence < 0.5` when given a circuit of a *different* canonical
  topology (no false positives).
  (Reconciled 2026-11: test_no_cross_matches.)

## Phase 3 — LLM classifier
- [x] 3.1 Create `python/pulsim/schematic/llm_classifier.py` with
  `classify(circuit) → RecognizedTopology | None`.
- [x] 3.2 Implement the deterministic textual fingerprint function
  `circuit_fingerprint(circuit) → str` (sorted components, sorted
  nodes, normalized ground label).
- [x] 3.3 Implement the on-disk cache at
  `~/.cache/pulsim/topology-cache.json` with `schema_version`
  validation and atomic write.
- [x] 3.4 Implement the Anthropic API call with system prompt listing
  the 17 recognized topology names, JSON-mode output, default
  model `claude-haiku-4-5`.
- [x] 3.5 Honor env vars `ANTHROPIC_API_KEY`,
  `PULSIM_LLM_LAYOUT_HINTS` (set to `0` to disable),
  `PULSIM_LLM_MODEL`, `PULSIM_TOPOLOGY_CACHE_DIR`.
- [x] 3.6 Implement graceful failure: missing key, network error,
  malformed JSON → return `None` (Stage 3 fallback in layout).
- [x] 3.7 Write `python/tests/test_llm_classifier.py` with mocked
  Anthropic SDK calls (cache miss writes, cache hit skips API,
  malformed response returns `None`, env var disables tier).
- [x] 3.8 Add a gated real-API smoke test behind
  `PULSIM_LLM_REAL_API=1`, runs only locally.

## Phase 4 — Template library
- [x] 4.1 Create the directory
  `python/pulsim/schematic/templates/` with one YAML file per
  canonical topology.
- [x] 4.2 Polish `buck.yaml`, `boost.yaml`, `flyback.yaml` to visual
  quality (canvas-fraction slots, explicit wire paths, aspect
  ratio).
- [ ] 4.3 Ship best-effort templates for the remaining 14 topologies;
  mark them in YAML with `quality: "draft"` so future passes can
  prioritize.
  (Reconciled 2026-11: 12 templates in all, so 9 drafts; the LLM tier
  names 17 topologies.)
- [x] 4.4 Implement the template instantiator
  `template_layout(circuit, recognized) → SchematicLayout` in
  `layout.py`.
  (Reconciled 2026-11: template_layout lives in
  template_instantiator.py.)
- [x] 4.5 Wire `compute_layout()` to dispatch:
  recognizer → template if confident → fall back to existing
  force-directed otherwise.
- [ ] 4.6 Commit golden SVG fixtures for buck, boost, flyback in
  `python/tests/fixtures/schematic/` (3 files).
  (Reconciled 2026-11: no golden SVGs; test_template_layouts.py
  asserts layout positions instead, by design (see its docstring).)
- [ ] 4.7 Add `python/tests/test_template_layouts.py` asserting
  golden-SVG match for the 3 polished topologies.
  (Reconciled 2026-11: the file exists and passes, but asserts
  canvas-fraction positions, not golden SVGs.)

## Phase 5 — Renderer outputs + UX
- [x] 5.1 Implement PNG export via `cairosvg` when available; fall
  back to `schemdraw` native PNG (existing) when not.
  (Reconciled 2026-11: render.py exports png via svg_renderer, with
  cairosvg as fallback.)
- [x] 5.2 Add `_repr_svg_` and `_repr_html_` to `SchematicLayout` for
  inline Jupyter display.
- [x] 5.3 Add a top-level `pulsim.schematic.render(circuit, path)`
  shorthand that composes `compute_layout` + `render_layout`.
- [x] 5.4 Document the public API in `docs/schematic-rendering.md`
  (restore from `87dcd1e` and update with V2 sections on the
  recognizer and LLM classifier).

## Phase 6 — Validation + ship
- [ ] 6.1 Run `pytest python/tests/test_schematic_*
  python/tests/test_topology_recognizer.py
  python/tests/test_llm_classifier.py
  python/tests/test_template_layouts.py` — all green.
  (Reconciled 2026-11: the three named files pass: 63 passed, 1
  skipped (the gated real-API test). The test_schematic_* files of
  1.10 were never restored, so this waits on 1.10.)
- [ ] 6.2 Render the three benchmark circuits the user cares about
  (buck, boost, flyback) and visually confirm the SVG is publication-
  quality. Update golden fixtures if intentional adjustments are
  made.
- [x] 6.3 Run `openspec validate add-schematic-renderer-v2 --strict`.
- [ ] 6.4 Update `python/pulsim/__init__.py` `__all__` to include
  the schematic submodule.
  (Reconciled 2026-11: 'schematic' is not in pulsim.__all__.)
- [ ] 6.5 Add an entry to the user-facing `CHANGELOG.md` (or
  equivalent) announcing the schematic renderer.
  (Reconciled 2026-11: no CHANGELOG entry announces the renderer.)
- [x] 6.6 Commit + push the implementation as one or more PRs
  referencing this change-id.
  (Reconciled 2026-11: merged to main in 41f096df (Phases 1-17).)

## Parallelization notes
- Phase 1 must complete before Phase 2 (recognizer consumes
  `Circuit.components()`).
- Phase 2 and Phase 3 are independent — recognizer is deterministic
  and the LLM classifier only consults the same `components()` view.
  They can be implemented in parallel.
- Phase 4 depends on both recognizer and LLM classifier.
- Phase 5 is independent of Phase 2–4 logic (it operates on
  `SchematicLayout` regardless of source).
