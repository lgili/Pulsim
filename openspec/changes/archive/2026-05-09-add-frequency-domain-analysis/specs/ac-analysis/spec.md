## ADDED Requirements

### Requirement: Operating-Point Linearization
The kernel SHALL provide an operating-point linearization that returns matrices `(E, A, B, C, D)` from a chosen DC operating point.

#### Scenario: Linearization at converged DC OP
- **GIVEN** a converged DC operating point `x_op` from `Simulator::dc_operating_point()`
- **WHEN** `Simulator::linearize_around(x_op, t_op)` is called
- **THEN** the returned `LinearSystem` has dimensions consistent with the circuit state
- **AND** matrix entries match the corresponding MNA decomposition

#### Scenario: Linearization fails on degenerate operating point
- **WHEN** the operating point is non-unique or has near-singular Jacobian
- **THEN** the kernel returns `LinearizationDiagnostic::Degenerate`
- **AND** the diagnostic includes condition number and rank-deficient direction

### Requirement: AC Small-Signal Sweep
The kernel SHALL support a frequency-domain AC small-signal sweep around a DC operating point.

#### Scenario: AC sweep on RC tank
- **GIVEN** an RC low-pass with `R = 1 kΩ`, `C = 1 µF`, perturbation at input
- **WHEN** `run_ac_sweep` covers 1 Hz–1 MHz with 20 points/decade
- **THEN** the magnitude at the resonant frequency matches `1/(jωRC)` form within 0.1 dB
- **AND** the phase matches within 1°

#### Scenario: AC sweep with multi-frequency factorization reuse
- **GIVEN** consecutive frequency points with identical sparsity pattern
- **WHEN** the AC sweep advances
- **THEN** the symbolic factorization is reused
- **AND** numeric factorizations vary per frequency only

#### Scenario: AC sweep multi-input
- **GIVEN** AC sweep configured with two perturbation sources and three measurement nodes
- **WHEN** the sweep runs
- **THEN** the result contains a 3×2 transfer-function matrix at each frequency
- **AND** off-diagonal entries reflect cross-coupling

### Requirement: Frequency Response Analysis (FRA)
The kernel SHALL support time-domain frequency response analysis via small-signal sinusoidal perturbation.

#### Scenario: FRA on buck open-loop
- **GIVEN** a buck converter with PWM input and a perturbation summed onto the duty signal
- **WHEN** FRA runs at 100 Hz–100 kHz with `n_cycles=50` and `amplitude=0.01`
- **THEN** the extracted transfer function `V_out/d` matches the AC sweep within 1 dB / 5°
- **AND** THD at each frequency stays below the small-signal threshold (default 5%)

#### Scenario: FRA convergence check
- **GIVEN** an FRA run with `n_cycles=10`
- **WHEN** the result has not stabilized between cycle 5 and cycle 10
- **THEN** the kernel automatically extends `n_cycles` until magnitude/phase change ≤0.1 dB / 0.5°
- **AND** the final `n_cycles` used is reported per frequency

#### Scenario: FRA flags large-signal regime
- **GIVEN** an FRA with `amplitude` too large for small-signal validity (THD >5%)
- **WHEN** the run completes
- **THEN** the result flags affected frequencies with `large_signal_warning`
- **AND** suggests reducing `amplitude` or switching to AC sweep

### Requirement: AC and FRA Result Telemetry
The result objects SHALL expose conditioning, total-harmonic-distortion, and per-frequency factorization reuse counters.

#### Scenario: AC result telemetry
- **WHEN** an AC sweep completes
- **THEN** the result includes `condition_numbers[f]` per frequency
- **AND** `factorization_cache_hits` over the sweep

#### Scenario: FRA result telemetry
- **WHEN** an FRA completes
- **THEN** the result includes `thd[f]` and effective `n_cycles[f]` per frequency
- **AND** `transient_runs_total` for cost accounting
