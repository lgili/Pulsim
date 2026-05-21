## 1. Docs scaffolding

- [ ] 1.1 Pick docs generator (mdBook recommended — single binary, no JS, fast)
- [ ] 1.2 Initialize `docs/v2/` with `book.toml` / `mkdocs.yml`
- [ ] 1.3 Add CI job that builds docs on every push

## 2. Core pages

- [ ] 2.1 Write `docs/v2/index.md` (1-page elevator pitch)
- [ ] 2.2 Write `docs/v2/getting-started.md` (build, install, first run)
- [ ] 2.3 Write `docs/v2/mental-model.md` (Graph/Pool/Cache/Newton diagram + prose)

## 3. Tutorials

- [ ] 3.1 `01-rc-charging.md` — V12 pulse + RC, plot voltage curve
- [ ] 3.2 `02-buck-converter.md` — buck.yaml walk-through w/ switch_fn
- [ ] 3.3 `03-flyback-isolated.md` — flyback.yaml + make_pwm_switch_fn
- [ ] 3.4 `04-3phase-vsi.md` — 3-phase inverter + ThreePhaseLegIndices
- [ ] 3.5 `05-ldo-feedback.md` — LDO with V15 op-amp + MOSFET
- [ ] 3.6 `06-igbt-boost.md` — V14 IGBT + ramped pulse pattern, w/ the convergence story

## 4. Reference

- [ ] 4.1 `api-reference.md` — extract method signatures + first paragraph of each Doxygen-style header comment
- [ ] 4.2 `gotchas.md` — Newton convergence corner cases, body diode requirement, etc.

## 5. Validation + commit

- [ ] 5.1 Build docs locally; manually review each page
- [ ] 5.2 `openspec validate add-pulsim-v2-tutorial-docs --strict`
- [ ] 5.3 Commit and push
