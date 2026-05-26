# Tasks — add-magnetic-hysteresis-jiles-atherton

## 1. Kernel device

- [ ] 1.1 Create `core/include/pulsim/magnetics/hysteretic_inductor.hpp` with `HystereticInductor` device.
- [ ] 1.2 Implement simplified Jiles-Atherton `dM/dB` ODE with parameters `Ms`, `a`, `alpha`, `k`, `c`.
- [ ] 1.3 Tustin integrator (default) + BDF1 path for stiff regimes near saturation.
- [ ] 1.4 Stamp through the existing inductor branch — flux state already in MNA; magnetization `M` is the extra device state.
- [ ] 1.5 Output channels: `M`, `B`, `H`, `dM/dH` (instantaneous permeability).
- [ ] 1.6 GoogleTest unit tests: linear-regime equivalence vs plain `Inductor`, analytical major-loop match within 5 %.

## 2. Python bindings + helpers

- [ ] 2.1 pybind `add_hysteretic_inductor` on `CircuitBuilder`.
- [ ] 2.2 Extend `python/pulsim/magnetic.py` with `JilesAthertonParams`, material-name lookup mirroring the existing Steinmetz catalog.
- [ ] 2.3 Implement `fit_ja_from_bh_curve(B, H)` least-squares helper using `scipy.optimize` if available, falling back to a hand-rolled gradient-descent stub.
- [ ] 2.4 Pre-fit and ship JA parameter sets for the 6-8 ferrites already in the Steinmetz catalog.
- [ ] 2.5 pytest covering material lookup, fitting helper, and a settled-loop regression vs golden CSV.

## 3. YAML support

- [ ] 3.1 Extend the YAML parser for `device_type: hysteretic_inductor` accepting either `material: "3F3"` shorthand or explicit `params: { Ms: ..., a: ..., alpha: ..., k: ..., c: ... }`.
- [ ] 3.2 Round-trip test: YAML → builder → simulate → expected hysteresis loop within 5 %.

## 4. Examples

- [ ] 4.1 `examples/scripts/run_saturable_reactor_hysteresis.py` — 50 Hz mains transformer with JA core; demonstrates residual-flux-driven inrush and B-H loop family.
- [ ] 4.2 `examples/scripts/run_psfb_with_ja_transformer.py` — PSFB with JA transformer; compares average loss to Steinmetz / iGSE prediction.
- [ ] 4.3 `examples/yaml/hysteretic_inductor_demo.yaml` minimal YAML showcase.

## 5. Docs

- [ ] 5.1 Extend `docs/v2/magnetic-models.md` with a JA section: equation derivation summary, parameter physical meaning, fitting recipe, when-to-use guidance vs Steinmetz / iGSE.
- [ ] 5.2 Update the "magnetics" tutorial index page to link to the new section.

## 6. Validation + release

- [ ] 6.1 `openspec validate add-magnetic-hysteresis-jiles-atherton --strict` clean.
- [ ] 6.2 CI green on all platforms.
- [ ] 6.3 Ship in the 1.5.0 release (bundled with the other Phase 2 changes).
- [ ] 6.4 Archive: `openspec archive add-magnetic-hysteresis-jiles-atherton --yes`.
