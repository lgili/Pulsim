## 1. Docs scaffolding

- [x] 1.1 Pick docs generator — used the project's existing **MkDocs Material** setup (mkdocs.yml already in repo) instead of introducing a second tool (mdBook)
- [x] 1.2 Add a `Pulsim v2 Guide` section to `mkdocs.yml` nav with all new pages
- [x] 1.3 CI `mkdocs build --strict` runs in `.github/workflows/docs.yml`; new pages pass without warnings

## 2. Core pages

- [x] 2.1 `docs/v2/index.md` (1-page elevator pitch + reading order)
- [x] 2.2 `docs/v2/getting-started.md` (build, install, first transient)
- [x] 2.3 `docs/v2/mental-model.md` (Graph/Pool/Cache/Newton — ASCII diagram + prose)

## 3. Tutorials

- [x] 3.1 `tutorials/01-rc-charging.md` — V12 pulse + RC
- [x] 3.2 `tutorials/02-buck-converter.md` — buck.yaml + switch_fn
- [x] 3.3 `tutorials/03-flyback-isolated.md` — transformer + commutation
- [x] 3.4 `tutorials/04-3phase-vsi.md` — 3-phase inverter + ThreePhaseLegIndices
- [x] 3.5 `tutorials/05-ldo-feedback.md` — LDO with V15 op-amp + MOSFET
- [x] 3.6 `tutorials/06-igbt-boost.md` — V14 IGBT + ramped pulse + convergence story

## 4. Reference

- [x] 4.1 `api-reference.md` — full Python API surface (CircuitBuilder methods + simulate + source helpers + types + C++ namespace mapping)
- [x] 4.2 `gotchas.md` — Newton convergence corner cases, body-diode requirement, cache sizing, `dt` heuristics

## 5. Validation + commit

- [x] 5.1 `python3 -m mkdocs build --strict` passes with zero warnings on the new v2 pages
- [x] 5.2 `openspec validate add-pulsim-v2-tutorial-docs --strict`
- [x] 5.3 Commit and push
