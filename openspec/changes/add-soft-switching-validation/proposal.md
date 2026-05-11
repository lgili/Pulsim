## Why
Modern power converters depend on soft-switching to reduce switching losses: **ZVS** (Zero Voltage Switching) and **ZCS** (Zero Current Switching). Resonant topologies (LLC, PSFB) are engineered specifically so that the FET drain-source voltage reaches zero before the gate turns on, eliminating turn-on loss.

We already have `lcc_resonant_inverter` in the regression matrix, but nothing measures whether ZVS is actually achieved or what fraction of losses are recovered. PSIM and PLECS surface "% of switching events in ZVS" as a standard KPI.

## What Changes
- Add a ZVS / ZCS detection helper in `benchmarks/kpi/`:
  - `compute_zvs_fraction(switch_states, v_drain_source_samples, dt)` — at each turn-on event, look at V_DS in the dt window just before. If |V_DS| < threshold, count as a ZVS event. Return % of turn-on events that achieved ZVS.
  - `compute_zcs_fraction(switch_states, i_drain_samples, dt)` — analogous, at each turn-off.
  - `compute_switching_loss(switch_states, v_ds, i_d)` — integrate ½·V·I over each transition.
- Add benchmarks:
  - `llc_half_bridge_zvs` — LLC resonant half-bridge tuned for ZVS; KPI reports ZVS fraction (target: 100 %) and average switching loss per cycle.
  - `psfb_full_bridge_zvs` — Phase-Shifted Full-Bridge with leakage inductance for ZVS; KPI reports ZVS fraction.
  - `buck_hard_switching_loss_reference` — hard-switched buck for comparison; ZVS fraction near 0 %, switching loss documented.
- Document the ZVS detection thresholds and the V_DS / I_D probe convention.

## Impact
- Affected specs: `device-models` (probe convention for V_DS / I_D), `benchmark-suite` (new soft-switching benchmarks + KPI).
- Affected code: KPI helpers + new YAML circuits + baselines.
- No C++ changes anticipated — uses existing primitives and the KPI extraction layer.
