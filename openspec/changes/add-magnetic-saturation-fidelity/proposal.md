## Why
PLECS markets detailed magnetic-component modeling as a distinguishing feature: B-H saturation curves, Steinmetz core loss (P = k·f^α·B^β), leakage inductance, and multi-winding transformers (3+ windings) are all routine. Pulsim has a `saturable_inductor` virtual component (we saw it in the parser) but the regression matrix never stresses it, and we don't yet validate against datasheet curves.

To close this gap we need (a) a working saturation model with hysteresis-free B-H curve specification, (b) Steinmetz-style core loss extraction in the KPI layer, and (c) benchmark circuits driven hard enough to saturate the core.

## What Changes
- Audit existing `saturable_inductor` and extend YAML parameters to support a piecewise-linear B-H curve (list of `(B_tesla, H_at_per_m)` points → derive μ_eff per operating point).
- Add `core_loss` extraction in the KPI layer: given measured B(t), apply Steinmetz to recover average loss density (W/kg) and compare against a documented k/α/β datasheet triple.
- Add benchmarks:
  - `saturating_inductor_step` — a voltage source charging a saturable inductor; observe the I_L "knee" as the core saturates and the effective inductance drops.
  - `transformer_saturation_dc_offset` — primary excited with a DC-offset square wave; observe magnetizing current walking the B-H curve and saturating asymmetrically.
  - `core_loss_steinmetz` — sinewave-driven inductor; KPI extracts core loss and compares to Steinmetz prediction within 10 %.
- New multi-winding transformer model: three or more `coupled_inductor` instances sharing a single mutual-flux representation (today coupled_inductor is 2-winding only). This enables center-tapped rectifiers and multi-output flybacks.

## Impact
- Affected specs: `magnetic-models` (saturation + core loss + multi-winding), `benchmark-suite`.
- Affected code: extend the `saturable_inductor` parser, possibly extend `coupled_inductor` for 3+ windings, KPI helper for core loss; new YAML benchmarks + baselines.
- The 3-winding transformer is the larger change here — coupled_inductor today only supports 2 windings.
