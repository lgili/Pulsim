## Why

v2 has 12 working YAML showcases (`examples/v2/*.yaml`), but **no narrative documentation** that explains:
- How to install / build
- The mental model (PWL state-space cache, switch-by-mask segments, Newton refresh)
- How to author a circuit YAML from scratch
- Walk-throughs of the 12 showcases
- Common pitfalls (the V13.2 / V14 boost-with-MOSFET convergence story is worth a write-up)
- Python API tour with `simulate(...)` (once V_ergonomics ships)

Without docs, the audience of v2 is limited to people who can read header comments. A solid `docs/` directory unlocks adoption.

## What Changes

- **ADD** `docs/v2/` directory with:
  - `index.md` — orientation: what v2 is, why PWL caching wins
  - `getting-started.md` — install, first transient, first YAML
  - `mental-model.md` — Graph + Pool + Cache + Newton refresh
  - `tutorials/` — 6 tutorial walk-throughs:
    - `01-rc-charging.md` (simplest: V12 pulse + RC)
    - `02-buck-converter.md` (YAML + switch_fn pattern)
    - `03-flyback-isolated.md` (transformer + diode commutation)
    - `04-3phase-vsi.md` (V8 helper + 6-switch topology)
    - `05-ldo-feedback.md` (V15 op-amp + MOSFET pass element)
    - `06-igbt-boost.md` (V14 IGBT + ramped pulse pattern)
  - `api-reference.md` — auto-generated section from header comments
  - `gotchas.md` — known Newton-convergence corner cases + workarounds
- **ADD** docs build setup (mdBook or MkDocs — light, no JS framework needed).
- **ADD** CI step to verify docs build.
- **ADD** Cross-links from each `examples/v2/*.yaml` header comment to the matching tutorial page.

## Impact

- **Affected specs:** new `pulsim-v2-documentation` capability.
- **Affected code:** zero core code; only adds `docs/v2/` + `mkdocs.yml` (or equivalent).
- **Risk:** Low — docs are additive. Keeping them in sync with the code is the long-term maintenance cost (mitigated by examples-as-tests in each tutorial).
