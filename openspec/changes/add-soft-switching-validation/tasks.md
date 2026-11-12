> Reconciled 2026-11: one task is done (4.1, as
> `circuits/switching/buck/buck_hard_switching_reference.yaml`). No ZVS
> or ZCS KPI exists (`compute_zvs_fraction`, `compute_zcs_fraction`,
> `compute_switching_loss` are absent). Neither ZVS bench exists;
> `isolated/ll11_llc_resonant_converter.yaml` is an LLC with no ZVS
> KPI. Also, none of these benchmark files runs on 2.0: they use
> `schema: pulsim-v1`, which `p.load_yaml_file` refuses ("missing
> required top-level key 'circuit:'"), and
> `benchmarks/tools/benchmark_runner.py` imports
> `pulsim_python_backend`, deleted in 64837c4c, so each run fails with
> "Pulsim Python runtime backend unavailable".

## 1. ZVS / ZCS detection KPIs
- [ ] 1.1 Add `compute_zvs_fraction(switch_state, v_ds_samples, threshold, lookback)` to `benchmarks/kpi/`.
- [ ] 1.2 Add `compute_zcs_fraction(switch_state, i_d_samples, threshold, lookback)`.
- [ ] 1.3 Add `compute_switching_loss(switch_state, v_ds, i_d)` — integrate ½·V·I at each transition.
- [ ] 1.4 Unit tests against synthetic switch state + ramp V_DS to confirm ZVS / ZCS / loss counts match hand-computed expectations.

## 2. LLC ZVS benchmark
- [ ] 2.1 Create `llc_half_bridge_zvs.yaml`: half-bridge with resonant Lr-Cs-Lm tank tuned so V_DS reaches 0 before turn-on of either switch.
- [ ] 2.2 Capture baselines for V_DS on each switch and i_D.
- [ ] 2.3 KPI: `kpi__zvs_fraction` (target 100 %), `kpi__avg_switching_loss_w` (target ≈ 0 W).

## 3. PSFB ZVS benchmark
- [ ] 3.1 Create `psfb_full_bridge_zvs.yaml`: phase-shifted full-bridge with leakage inductance providing ZVS for the lagging-leg switches.
- [ ] 3.2 KPI: per-switch ZVS fraction (leading-leg ≈ 100 %, lagging-leg ≈ 100 % at full load).

## 4. Hard-switched reference
- [x] 4.1 Create `buck_hard_switching_loss_reference.yaml`: standard buck where ZVS is not achieved.
  (Reconciled 2026-11: as buck_hard_switching_reference.yaml.)
- [ ] 4.2 KPI: ZVS fraction near 0 %, document the average switching loss.

## 5. Documentation + smoke-run
- [ ] 5.1 Document the ZVS / ZCS probe convention (which node is V_DS, which branch is I_D for each device type) in `docs/SOFT_SWITCHING.md`.
- [ ] 5.2 Generate baselines for all three new benchmarks.
- [ ] 5.3 Confirm dashboard renders the new KPI columns and the benches pass.
