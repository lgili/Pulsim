## Why

Pulsim's `Transformer` model in `core/include/pulsim/v1/components/transformer.hpp` is a linear coupled inductor — no saturation, no hysteresis, no core loss. Real power-electronics design hinges on magnetic component fidelity:

- **Saturation** drives transformer/inductor design in flyback, forward, push-pull. A linear model misses the exponential current rise that destroys MOSFETs.
- **Steinmetz core loss** dominates efficiency calculations in PFC chokes and high-frequency transformers.
- **Hysteresis (Jiles-Atherton, Preisach)** matters for low-frequency / mains transformers and inrush analysis.
- **Eddy currents** matter for litz wire vs solid winding loss prediction.

PSIM has a magnetic core library; PLECS Multi-Physics ships saturable inductor and transformer models. OpenModelica's `Magnetic.FluxTubes` library is the reference open-source implementation. Pulsim already lists `magnetic-components` as a registered skill — this change makes it real in the kernel.

## What Changes

### Saturable Inductor Model
- New device `SaturableInductor` with B-H curve as user-supplied lookup table or analytical fit (arctan, Langevin, Brillouin).
- State variable: flux linkage λ. Current `i = i(λ)` from inverse B-H curve.
- Stamps via λ-based companion model (analogous to existing inductor companion).
- Auto-derived AD Jacobian covers nonlinear i(λ).

### Saturable Transformer
- New device `SaturableTransformer` (N windings) with:
  - Per-winding leakage inductance
  - Magnetizing branch with saturable B-H
  - Optional core-loss resistor (Steinmetz-derived equivalent)
  - Flux-linkage state per primary path

### Steinmetz Core Loss
- `SteinmetzLoss { k, alpha, beta }` configuration on saturable devices.
- Loss `P = k · f^α · B^β` integrated across simulation cycle, available in `BackendTelemetry`.
- Improved Generalized Steinmetz Equation (iGSE / MSE) variant for non-sinusoidal flux waveforms.

### Hysteresis Model (Jiles-Atherton)
- `JilesAthertonParams { Ms, a, alpha_jt, k, c }` configuration.
- ODE-state hysteresis: anhysteretic + irreversible + reversible.
- Optional, behind `simulation.hysteresis_model: none | jiles_atherton`.

### Eddy-Current Lumped Model
- Per-winding lumped RL eddy-current branch.
- Simple effective resistance `R_eddy(f)` table or skin-depth analytic for litz wire.

### Datasheet Importer for Cores
- `pulsim.import_core_datasheet(pdf, manufacturer)` extracts Steinmetz parameters and B-H curve from typical core datasheets (Magnetics Inc., TDK, Ferroxcube, EPCOS).
- Output: catalog YAML under `devices/cores/<vendor>/<material>.yaml`.

### YAML Schema
- New types: `saturable_inductor`, `saturable_transformer`.
- Per-device `core_model:` reference to catalog or inline `bh_curve`, `steinmetz`, `jiles_atherton`, `eddy_current` blocks.
- `simulation.hysteresis_model` global toggle.

### Validation Suite
- Inrush current test: 1500 VA mains transformer cold-start, peak primary current within 20% of analytical Faraday integral.
- Flyback transformer test: secondary current waveform with Rcore loss, comparison vs measurement.
- PFC choke test: core loss vs frequency sweep, comparison vs Steinmetz expected.

## Impact

- **New capability**: `magnetic-models` (proposed name).
- **Affected specs**: `magnetic-models` (new), `netlist-yaml` (new component types).
- **Affected code**: new files `core/include/pulsim/v1/components/saturable_inductor.hpp`, `saturable_transformer.hpp`, `core/include/pulsim/v1/magnetic/` (B-H curves, Jiles-Atherton ODE, Steinmetz integrator); importer in `python/pulsim/import_/`; catalog under `devices/cores/`.
- **Performance**: saturable models are nonlinear → use AD path (Phase 0); Steinmetz integration adds ≤5% overhead on cycle-end accumulation; Jiles-Atherton adds ~15–25% per-step in nonlinear region.
- **Backward compat**: linear `Inductor` and `Transformer` unchanged.

## Success Criteria

1. **Inrush accuracy**: ≤20% error vs analytical Faraday on mains transformer cold-start.
2. **Steinmetz loss**: ≤10% error vs vendor calculator across 25–500 kHz range on N87 core.
3. **Saturation onset**: B-saturation transition matches B-H curve within 5% on flyback test.
4. **Importer**: ≥3 of 4 reference core datasheets (Magnetics Kool Mµ, TDK PC95, Ferroxcube N87, EPCOS N97) yield runnable catalog YAML.
5. **Documentation**: tutorial on flyback transformer design end-to-end with saturation visualization.
