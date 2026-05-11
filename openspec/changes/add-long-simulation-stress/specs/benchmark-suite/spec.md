## ADDED Requirements

### Requirement: Long-Duration Numerical Stability
The benchmark suite SHALL include at least one benchmark that runs for ≥ 1 s of simulated time at ≥ 100 kHz switching and validates that the output exhibits no measurable drift (linear regression slope < 1 mV/s) over the second half of the simulation.

#### Scenario: Buck steady-state shows no drift over 1 second
- **WHEN** running `long_run_drift_buck` for 1 s simulated time
- **THEN** the linear regression slope of V(out) over the last 500 ms is below 1 mV/s
- **AND** the energy-conservation KPI `kpi__conservation_residual` is below 1 % over the full run

### Requirement: High-Frequency Switching Coverage
The benchmark suite SHALL include a benchmark running at ≥ 1 MHz switching frequency with appropriately scaled passive components, validating that the timestep machinery handles the cadence.

#### Scenario: 1 MHz GaN buck holds to its analytical steady-state
- **WHEN** running `high_freq_gan_buck` (1 MHz f_sw, L = 1 µH, C = 1 µF, dt = 10 ns)
- **THEN** the steady-state V(out) matches `V_in · duty` to within 1 %

### Requirement: Many-Device Scalability
The benchmark suite SHALL include a benchmark with ≥ 30 simultaneously-active PWL switches and validate that the simulation completes and produces a deterministic output.

#### Scenario: MMC 8-cell chain runs to completion
- **WHEN** running `mmc_8cell_chain` (8 H-bridge cells, 32 switches)
- **THEN** the simulation finishes within the manifest's documented runtime budget
- **AND** the captured baseline reproduces bit-for-bit on a second run
