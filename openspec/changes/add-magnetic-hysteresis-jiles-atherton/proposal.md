## Why

Pulsim already covers core-loss estimation via Steinmetz and iGSE — both excellent for **average-power** loss predictions over many cycles. They fail for problems that depend on **instantaneous** hysteresis dynamics:

- Saturable reactors and ferrorresonant transformers in HVDC / grid applications, where the B-H minor-loop shape drives DC offset and current asymmetry.
- Inrush currents in mains transformers (residual flux at the previous shutdown sets the inrush peak — Steinmetz does not see flux at all).
- Switching transients in flux-balancing topologies (PSFB, LLC) where the leakage-plus-magnetizing path crosses the B-H loop multiple times per switching period.

PSIM's "Saturable Inductor" and PLECS's "Inductor with Hysteresis" both expose a Jiles-Atherton (or equivalent Preisach) hysteretic model. Adding JA to Pulsim closes the gap for the saturable-magnetics use cases above without disturbing the existing Steinmetz / iGSE loss paths (which remain the correct tool for cycle-averaged loss).

## What Changes

- **C++ kernel** — new device file `core/include/pulsim/magnetics/hysteretic_inductor.hpp`:
  - `HystereticInductor` device implementing the **simplified Jiles-Atherton** ODE: `dM/dB = f(Ms, a, alpha, k, c, M, H)`, where `Ms` is saturation magnetization, `a` and `alpha` shape parameters, `k` the pinning-loss factor, and `c` the reversibility ratio.
  - State variables: total magnetization `M`. The current state `i = (H + alpha*M)*l_m / N` is reconstructed every step from the flux state already maintained by the inductor branch in MNA.
  - Two integrator backends: Tustin (default; matches existing kernel) and BDF1 (for stiff parameter regimes near `Ms`).
- **Python helper module** — extend `python/pulsim/magnetic.py`:
  - `JilesAthertonParams` POD with material defaults for the same 6-8 ferrites covered by the existing Steinmetz catalog.
  - `pulsim.magnetic.fit_ja_from_bh_curve(B_array, H_array)` — least-squares fit of the 5 JA parameters from a measured major B-H loop.
  - `CircuitBuilder.add_hysteretic_inductor(name, nodes, params)` returning a device handle.
- **YAML support** — `device_type: hysteretic_inductor` with material name lookup OR explicit `Ms / a / alpha / k / c` block.
- **Examples**:
  - `examples/scripts/run_saturable_reactor_hysteresis.py` — 50 Hz mains transformer modelled with JA; demonstrates inrush from residual flux and major / minor loops on a phase-plane plot.
  - `examples/scripts/run_psfb_with_ja_transformer.py` — phase-shifted full-bridge with a JA-modelled transformer instead of the linear-saturable one; compares cycle-averaged loss with the Steinmetz / iGSE pipeline.
- **Cross-validation** — pytest comparing the major B-H loop simulated from sinusoidal H excitation to the parameter-set's analytical loop within 5 %.
- **Docs** — extend `docs/v2/magnetic-models.md` with a JA section, equation summary, fitting recipe, and clear guidance on when to use JA vs Steinmetz / iGSE.

## Impact

- **Affected specs**: `magnetic-models` (new requirements for JA model + analytical-loop scenarios).
- **Affected code**: new C++ device (~400 LOC), pybind glue (~40 LOC), Python helper module (~250 LOC), YAML parser (~30 LOC).
- **Backward compatibility**: PURE ADDITION. Steinmetz / iGSE / `SaturableInductor` paths untouched.
- **Performance**: 1 extra state per hysteretic device; the ODE evaluation is local and cheap. For a single-transformer circuit the runtime overhead vs the existing `SaturableInductor` is < 10 %.
- **Risk**: parameter identification from datasheets is the main user-facing pain point. Mitigated by (a) shipping pre-fitted parameter sets for the same ferrites in the Steinmetz catalog, and (b) the `fit_ja_from_bh_curve` helper.
