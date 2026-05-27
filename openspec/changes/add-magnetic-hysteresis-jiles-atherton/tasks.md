# Tasks — add-magnetic-hysteresis-jiles-atherton

> Status (Phase 2.2 — v1.5.0 RC):
> **Post-processing** Jiles-Atherton path is COMPLETE — the
> ``hysteresis.py`` module was already present but unexported.
> This change adds the missing ``fit_ja_from_bh_curve`` helper,
> exposes the full surface via ``pulsim.*``, and ships pytest
> coverage. The **kernel-coupled C++ device** (Task 1.x) — which
> would let the JA equation drive the MNA matrix directly so the
> hysteresis voltage drop participates in KCL — is intentionally
> deferred. For most circuits the hysteresis drop is small vs ωL
> (it matters mostly for ferrorresonance / inrush / saturable
> reactor); the post-processing path quantifies the LOSS, which
> is the dominant engineering deliverable.

## 1. Kernel device — deferred to a follow-up release

- [ ] 1.1 Create `core/include/pulsim/magnetics/hysteretic_inductor.hpp` with `HystereticInductor` device.
- [ ] 1.2 Implement simplified Jiles-Atherton `dM/dB` ODE with parameters `Ms`, `a`, `alpha`, `k`, `c`.
- [ ] 1.3 Tustin integrator (default) + BDF1 path for stiff regimes near saturation.
- [ ] 1.4 Stamp through the existing inductor branch — flux state already in MNA; magnetization `M` is the extra device state.
- [ ] 1.5 Output channels: `M`, `B`, `H`, `dM/dH` (instantaneous permeability).
- [ ] 1.6 GoogleTest unit tests: linear-regime equivalence vs plain `Inductor`, analytical major-loop match within 5 %.

## 2. Python bindings + helpers

- [x] 2.1 No `add_hysteretic_inductor` binding needed for the post-processing path — `JilesAthertonModel` consumes the inductor current already in the state vector.
- [x] 2.2 `JilesAthertonParams` + material-name lookup (4 materials: `annealed_iron`, `si_steel_m19`, `ferrite_n87`, `permalloy`) shipped in `python/pulsim/hysteresis.py`.
- [x] 2.3 `fit_ja_from_bh_curve(B, H, p0=None, max_iter=400)` least-squares helper — uses `scipy.optimize.least_squares` with box constraints; falls back to a coordinate-descent loop when scipy is unavailable.
- [x] 2.4 Pre-fitted JA parameter sets for 4 representative materials shipped in `_REFERENCE_MATERIALS`. (Extending to all 6-8 Steinmetz-catalog materials is deferred — out-of-the-box parameters are typical only; precision design needs a refit.)
- [x] 2.5 pytest covering catalog lookup, state advancement, settled-loop sanity, high-c → reduced-loss monotonicity, fit round-trip — 7/7 pass locally in 1.5 s.

## 3. YAML support — deferred (waiting on the C++ device)

- [ ] 3.1 Extend the YAML parser for `device_type: hysteretic_inductor`.
- [ ] 3.2 Round-trip test: YAML → builder → simulate → expected hysteresis loop within 5 %.

## 4. Examples — deferred to v1.5.0 final

- [ ] 4.1 `examples/scripts/run_saturable_reactor_hysteresis.py` — 50 Hz mains transformer with JA core; demonstrates residual-flux-driven inrush and B-H loop family.
- [ ] 4.2 `examples/scripts/run_psfb_with_ja_transformer.py` — PSFB with JA transformer; compares average loss to Steinmetz / iGSE prediction.
- [ ] 4.3 `examples/yaml/hysteretic_inductor_demo.yaml` — depends on YAML support.

## 5. Docs — deferred to v1.5.0 final

- [ ] 5.1 Extend `docs/v2/magnetic-models.md` with a JA section.
- [ ] 5.2 Update the "magnetics" tutorial index page to link to the new section.

## 6. Validation + release

- [x] 6.1 `openspec validate add-magnetic-hysteresis-jiles-atherton --strict` clean.
- [x] 6.2 Local pytest passes (7/7). Cross-platform CI pending PR.
- [ ] 6.3 Ship in the 1.5.0 release (bundled with the other Phase 2 changes).
- [ ] 6.4 Archive: `openspec archive add-magnetic-hysteresis-jiles-atherton --yes`.
