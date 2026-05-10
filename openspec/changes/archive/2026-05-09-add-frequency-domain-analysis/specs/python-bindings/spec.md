## ADDED Requirements

### Requirement: AC and FRA Methods on Simulator
Python bindings SHALL expose `Simulator.run_ac_sweep(options)` and `Simulator.run_fra(options)` returning structured results.

#### Scenario: Run AC sweep from Python
- **WHEN** Python code instantiates `AcSweepOptions(f_start=1, f_stop=1e6, points_per_decade=20, ...)` and calls `sim.run_ac_sweep(options)`
- **THEN** the call returns an `AcSweepResult` with frequencies, magnitudes, phases, real/imag arrays
- **AND** the arrays are contiguous numpy arrays interoperable with matplotlib

#### Scenario: Run FRA from Python
- **WHEN** Python code calls `sim.run_fra(FraOptions(...))`
- **THEN** the returned `FraResult` exposes the same shape as `AcSweepResult` plus `thd_at_frequency`
- **AND** convergence diagnostics are accessible

### Requirement: Linearization Method on Simulator
Python bindings SHALL expose `Simulator.linearize_around(x_op, t_op=0.0)` returning sparse `(E, A, B, C, D)` matrices.

#### Scenario: Linearize from Python
- **WHEN** Python calls `sim.linearize_around(dc_result.solution)`
- **THEN** the returned object exposes `E`, `A`, `B`, `C`, `D` as scipy sparse matrices
- **AND** the matrices can be passed to scipy `signal.dlti` or `control.ss` for downstream analysis

### Requirement: Bode/Nyquist Plotting Helpers
Python bindings SHALL provide `pulsim.bode_plot(result, ax=None, ...)` and `pulsim.nyquist_plot(result, ...)` for fast visualization.

#### Scenario: Bode plot end-to-end
- **WHEN** Python code runs `pulsim.bode_plot(ac_result)` with default arguments
- **THEN** matplotlib axes are returned showing magnitude (dB) and phase (deg) vs frequency (log-x)
- **AND** the plot is publication-quality without further user formatting

#### Scenario: Overlay AC and FRA
- **WHEN** Python code runs `pulsim.fra_overlay(ac_result, fra_result)`
- **THEN** both curves are plotted with distinct labels
- **AND** delta annotations highlight any region exceeding 1 dB / 5° divergence

### Requirement: Export Helpers
Python bindings SHALL provide CSV and JSON export for AC and FRA results.

#### Scenario: CSV export
- **WHEN** Python calls `pulsim.export_ac_csv(ac_result, "out.csv", format="magphase")`
- **THEN** a CSV with columns `f, mag_db, phase_deg` is written
- **AND** an alternative `format="complex"` writes `f, re, im`

#### Scenario: JSON round-trip
- **WHEN** Python calls `pulsim.export_ac_json(ac_result, "out.json")` and later `pulsim.load_ac_result("out.json")`
- **THEN** the loaded result is bit-identical to the original
