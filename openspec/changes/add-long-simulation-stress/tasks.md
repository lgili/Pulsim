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
- [ ] 1.1 Create `long_run_drift_buck.yaml` — buck at 100 kHz for 1 s simulated.
- [ ] 1.2 Add KPI `compute_drift(samples, time_window)` that fits a linear regression and returns the slope in V/s.
- [ ] 1.3 Validate: drift slope < 1 mV/s over the second half of the run.

## 2. High-frequency (GaN / SiC)
- [x] 2.1 Create `high_freq_gan_buck.yaml` — buck at 1 MHz switching with L = 1 µH, C = 1 µF, dt = 10 ns.
  (Reconciled 2026-11:
  circuits/switching/buck/high_freq_gan_buck.yaml.)
- [ ] 2.2 Capture baseline and verify steady-state V(out) within 1 % of analytical `V_in · duty`.

## 3. Stiff system
- [ ] 3.1 Create `stiff_rc_with_high_freq_switching.yaml` — a 1 GΩ leakage RC in parallel with a 5 MHz switching loop.
- [ ] 3.2 Verify the simulation completes without timestep collapse and converges to a sensible steady state.

## 4. MMC scale
- [ ] 4.1 Create `mmc_8cell_chain.yaml` — 8 cascaded H-bridge cells (32 switches) driving an RL load.
- [ ] 4.2 Capture baseline of one output phase voltage.
- [ ] 4.3 Validate that the per-step solve time scales sub-quadratically with device count (record runtime as a KPI).

## 5. Conservation-law KPI
- [ ] 5.1 Add `compute_conservation_drift(p_in_samples, p_out_samples, e_stored_initial, e_stored_final, time)` to `benchmarks/kpi/`.
- [ ] 5.2 Wire it into all four long-run benchmarks.
- [ ] 5.3 Validate: conservation residual < 1 % over the full simulated time.

## 6. Dashboard integration
- [ ] 6.1 Add an `--include-long-runs` flag to `closed_loop_dashboard.py` so the default run skips them.
- [ ] 6.2 Generate baselines for all four benchmarks (offline run).
- [ ] 6.3 Document expected runtimes in `docs/`.
