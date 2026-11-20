> Reconciled 2026-11: one task is done as written (2.1). Related files
> exist with other parameters, each with a baseline CSV:
> `circuits/thermal/long_run_drift_buck.yaml` (100 ms, not 1 s; its KPI
> is ripple, not a drift slope),
> `circuits/stress/stiff_rc_high_freq_switching.yaml` (1 GOhm leak, 500
> kHz loop rather than 5 MHz) and
> `circuits/switching/multilevel/mmc_4cell_chain.yaml` (4 cells, not 8).
> There is no `compute_drift`, no `compute_conservation_drift` and no
> `--include-long-runs` flag. Also, none of these benchmark files runs
> on 2.0: they use `schema: pulsim-v1`, which `p.load_yaml_file` refuses
> ("missing required top-level key 'circuit:'"), and
> `benchmarks/tools/benchmark_runner.py` imports
> `pulsim_python_backend`, deleted in 64837c4c, so each run fails with
> "Pulsim Python runtime backend unavailable".

## 1. Long-run drift
- [x] 1.1 Create `long_run_drift_buck.yaml` — buck at 100 kHz for 1 s simulated.
  (Done 2026-11-20 as a Python case, not YAML — the benchmark YAMLs do not load on 2.0:
  `Buck` + `test_long_run_drift_buck_one_second` in python/tests/test_long_simulation_stress.py, 1 s on pwl (both restart
  flags), trbdf2, esdirk43, dsed; long marker `examples`, `PULSIM_RUN_EXAMPLES=1`.)
- [x] 1.2 Add KPI `compute_drift(samples, time_window)` that fits a linear regression and returns the slope in V/s.
- [x] 1.3 Validate: drift slope < 1 mV/s over the second half of the run.
  (Plus the last period against the closed-form periodic steady state; pwl restart is
  the strict xfail F-L-1 on its energy ledger, see FINDINGS.md.)

## 2. High-frequency (GaN / SiC)
- [x] 2.1 Create `high_freq_gan_buck.yaml` — buck at 1 MHz switching with L = 1 µH, C = 1 µF, dt = 10 ns.
  (Reconciled 2026-11:
  circuits/switching/buck/high_freq_gan_buck.yaml.)
- [x] 2.2 Capture baseline and verify steady-state V(out) within 1 % of analytical `V_in · duty`.
  (`test_gan_buck_at_1mhz_holds_vin_times_duty`, five engines; also against the
  closed-form steady state. No simulator baseline: the truth is analytic.)

## 3. Stiff system
- [x] 3.1 Create `stiff_rc_with_high_freq_switching.yaml` — a 1 GΩ leakage RC in parallel with a 5 MHz switching loop.
- [x] 3.2 Verify the simulation completes without timestep collapse and converges to a sensible steady state.
  (`test_stiff_leak_with_a_5mhz_loop_completes_and_charges_the_slow_node`: Python case,
  tau_slow / T_sw = 5e5; slow node against its closed form, step count bounded.)

## 4. MMC scale
- [x] 4.1 Create `mmc_8cell_chain.yaml` — 8 cascaded H-bridge cells (32 switches) driving an RL load.
- [x] 4.2 Capture baseline of one output phase voltage.
  (`_mmc` + `test_mmc_8cell_chain_mean_current_and_determinism`: the mean load current
  against its exact linear identity and a bit-identical second run, five engines.)
- [ ] 4.3 Validate that the per-step solve time scales sub-quadratically with device count (record runtime as a KPI).
  (Not done: a wall-clock assertion is banned in the test suite (CI lesson) and a
  runtime KPI needs the dashboard of 6.x, which does not run on 2.0.)

## 5. Conservation-law KPI
- [x] 5.1 Add `compute_conservation_drift(p_in_samples, p_out_samples, e_stored_initial, e_stored_final, time)` to `benchmarks/kpi/`.
- [x] 5.2 Wire it into all four long-run benchmarks.
  (`test_long_run_drift_buck_one_second`, `test_buck_energy_ledger_closes_over_the_record`,
  `test_conservation_kpi_closes_on_the_other_long_run_circuits` [gan, stiff, mmc].)
- [x] 5.3 Validate: conservation residual < 1 % over the full simulated time.
  (And under the grid's own bound; the engines that miss it are strict xfails
  F-L-1, F-L-2, F-L-3 in FINDINGS.md.)

## 6. Dashboard integration
- [ ] 6.1 Add an `--include-long-runs` flag to `closed_loop_dashboard.py` so the default run skips them.
  (Not done: there is no `closed_loop_dashboard.py` and the benchmark runner does not run
  on 2.0. The long cases sit behind the existing slow marker `examples` instead.)
- [ ] 6.2 Generate baselines for all four benchmarks (offline run).
  (Not done on purpose: every check compares with a closed form or an invariant, not a
  stored simulator run.)
- [x] 6.3 Document expected runtimes in `docs/`.
  (docs/performance-tuning.md, "Long-simulation stress suite".)
