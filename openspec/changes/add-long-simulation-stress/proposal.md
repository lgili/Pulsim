## Why
Pulsim's existing benches run for 5–80 ms of simulated time. Commercial simulators routinely run **multi-second** workloads — overnight test campaigns, MPPT scans, motor warm-up analyses — without numerical drift.

To prove numerical robustness at that scale we need to actually run long simulations and validate that (a) energy/charge conservation holds bit-by-bit, (b) high-frequency switching at 1+ MHz works for the GaN/SiC age, and (c) the simulator doesn't degrade with the number of devices in the circuit (matrix-converter-scale designs with 1000+ switches).

## What Changes
- Add `long_run_drift_buck` — a buck converter running for 1 s simulated time at 100 kHz switching. The KPI extracts the linear-regression slope of V(out) over the second half of the simulation and asserts it is below 1 mV/s (no drift).
- Add `high_freq_gan_buck` — a buck at 1 MHz switching with proportionally scaled L/C, dt = 10 ns. Validates that the solver handles the cadence and that the steady-state V(out) is within 1 % of the analytical prediction.
- Add `stiff_rc_with_high_freq_switching` — a circuit combining a 1 GΩ leakage path (very slow time constant) with a 5 MHz switching loop (very fast). Time-scale ratio 10⁵. Validates that adaptive timestep / step-rejection handles the stiffness.
- Add `mmc_8cell_chain` — a Modular Multi-level Converter with 8 cascaded H-bridge cells (32 switches total) driving an RL load. Validates that the solver scales to 30+ simultaneous PWL devices without degradation.
- Add a KPI helper `compute_conservation_drift(p_in, p_out, e_stored_initial, e_stored_final, time_window)` that checks `∫(p_in − p_out)dt ≈ ΔE_stored` to within 1 %, used as a sanity check on long runs.

## Impact
- Affected specs: `benchmark-suite`, `transient-timestep`.
- Affected code: new YAML circuits + baselines + a `compute_conservation_drift` helper in `benchmarks/kpi/`.
- These benchmarks will be **slow** to run (each ~10–60 s of wall time). Mark them in the manifest so they can be opted out of the default dashboard run.
