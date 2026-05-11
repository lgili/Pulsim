## ADDED Requirements

### Requirement: Bode Plot Validation Against Analytical Models
The benchmark suite SHALL support a `bode` validation type that sweeps a source's AC perturbation across a frequency range, extracts the magnitude/phase response of a named observable, and compares it against a Python-callable analytical model.

#### Scenario: Open-loop buck plant matches Erickson averaged model
- **WHEN** running `bode_buck_plant` against `erickson_buck.transfer_function`
- **THEN** the measured |G(jω)| matches the analytical model to within 1 dB across the swept range
- **AND** the measured ∠G(jω) matches the analytical model to within 5° across the swept range
- **AND** results JSON includes `bode__db_max_err` and `bode__phase_max_err_deg`

#### Scenario: Boost plant exhibits expected RHP zero
- **WHEN** running `bode_boost_plant` (open-loop boost)
- **THEN** the measured phase exhibits the +180° "lift" at the RHP-zero frequency predicted by the averaged model (within 10°)
- **AND** the measured gain dips at the RHP-zero frequency (within 2 dB of the analytical prediction)

### Requirement: Loop Gain Measurement and Margin Extraction
The frequency-response harness SHALL extract gain margin, phase margin, and crossover frequency from a measured Bode plot of a closed-loop system.

#### Scenario: PI-compensated buck reports compensator margins
- **WHEN** running `bode_buck_pi_loop_gain` with the same PI compensator used in `cl_buck_pi`
- **THEN** the harness emits `bode__gain_margin_db`, `bode__phase_margin_deg`, `bode__crossover_hz` in results JSON
- **AND** the measured phase margin is within ±5° of the hand-computed value documented inline in the benchmark
- **AND** the measured gain margin is within ±2 dB of the hand-computed value
