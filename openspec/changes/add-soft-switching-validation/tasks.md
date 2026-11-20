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
- [x] 1.1 Add `compute_zvs_fraction(switch_state, v_ds_samples, threshold, lookback)` to `benchmarks/kpi/`.
- [x] 1.2 Add `compute_zcs_fraction(switch_state, i_d_samples, threshold, lookback)`.
- [x] 1.3 Add `compute_switching_loss(switch_state, v_ds, i_d)` — integrate v·i dt over a window after each transition (the earlier "½·V·I" was not an energy: F-S-2).
- [x] 1.4 Unit tests against synthetic switch state + ramp V_DS to confirm ZVS / ZCS / loss counts match hand-computed expectations.

  (2026-11-20: 1.1/1.2 existed under these names with sample lookbacks;
  1.3 is compute_switching_loss_per_event — its old formula was W/s,
  finding F-S-2, fixed: int v*i dt over window_s. 1.4:
  test_kpi_* in python/tests/test_soft_switching_validation.py.)

## 1b. Soft-switching transitions against independent truths
- [x] 1b.1 Dead-time transition of a half-bridge with nonlinear C_oss, current driven: transition time = charge closed form, on pwl (restart off / on), trbdf2, esdirk43; dsed refuses by name.
- [x] 1b.2 Resonant transition (L to a stiff node): ZVS energy condition, partial ZVS trough v_min, trajectory against the mpmath ODE, full ZVS at the diode clamp.
- [x] 1b.3 Hard turn-on energy (ZVS lost) against the closed form: strict xfails cite the known open items (fixed engine, trbdf2); esdirk43 passes case A; F-S-1 parks its start failure.

## 2. LLC ZVS benchmark
- [x] 2.1 Create `llc_half_bridge_zvs.yaml`: half-bridge with resonant Lr-Cs-Lm tank tuned so V_DS reaches 0 before turn-on of either switch.
- [ ] 2.2 Capture baselines for V_DS on each switch and i_D.
- [ ] 2.3 KPI: `kpi__zvs_fraction` (target 100 %), `kpi__avg_switching_loss_w` (target ≈ 0 W).

  (2.1 done in benchmarks/circuits/switching/isolated/ with linear C_oss and
  R_ac for the rectifier (chosen under F-S-3, since fixed); its truth is exact_truth.py, checked in
  test_llc_* . 2.2 baselines: not captured — the test computes the truth
  each run (16 s). 2.3: the ZVS fraction (100 %) is asserted; the average
  switching loss KPI is not (no YAML kpi block, the runner is dead: see the
  reconciliation note).)

## 3. PSFB ZVS benchmark
- [x] 3.1 Create `psfb_full_bridge_zvs.yaml`: phase-shifted full-bridge with leakage inductance providing ZVS for the lagging-leg switches.
- [x] 3.2 KPI: per-switch ZVS fraction (leading-leg ≈ 100 %, lagging-leg ≈ 100 % at full load).
  (2026-11-20: F-S-3 fixed in the truth tool (the slow basis lost ~18
  digits), so the PSFB has an independent truth: in
  benchmarks/circuits/switching/isolated/, 400 V, 100 kHz, 80 ns dead time,
  L_lk 10 uH, ideal 4:1 + diode bridge, started in its freewheel state.
  test_psfb_*: truth ZVS on all 12 edges, i_Llk at turn-off, transition
  samples, time-weighted means and the per-switch ZVS fraction (100 % on
  all four) on pwl+restart, trbdf2, esdirk43, dsed; pwl without restart
  strict xfails (trap 9). F-S-4 (dsed drops L_o's initial current at a
  power-transfer start) parked as a strict xfail.)

## 4. Hard-switched reference
- [x] 4.1 Create `buck_hard_switching_loss_reference.yaml`: standard buck where ZVS is not achieved.
  (Reconciled 2026-11: as buck_hard_switching_reference.yaml.)
- [ ] 4.2 KPI: ZVS fraction near 0 %, document the average switching loss.

## 5. Documentation + smoke-run
- [ ] 5.1 Document the ZVS / ZCS probe convention (which node is V_DS, which branch is I_D for each device type) in `docs/SOFT_SWITCHING.md`.
- [ ] 5.2 Generate baselines for all three new benchmarks.
- [ ] 5.3 Confirm dashboard renders the new KPI columns and the benches pass.
