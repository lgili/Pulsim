## ADDED Requirements

### Requirement: End-to-end buck-converter showcase

An integration test SHALL load `examples/v2/buck.yaml` via `pulsim::v2::yaml::load_file`, construct a `PwlStateSpaceCache`, run `run_transient` with a 100 kHz / 50 % PWM `switch_fn`, and verify the steady-state output voltage matches the analytical buck relation `V_out = V_in · D · η` (with `η ≥ 0.95` for the YAML defaults) within ±0.5 V.

The test MUST simulate long enough for the LC filter to reach steady state (at least 4 ms ≈ 60 × τ where τ = √(LC) ≈ 68 µs) and measure mean V_out over the LAST 0.5 ms.

The test MUST also verify the output ripple (peak-to-peak V_out over the last 0.5 ms) stays under 1 V (LC filter performance check).

#### Scenario: Buck steady-state V_out matches V_in · D

- **GIVEN** the YAML circuit `examples/v2/buck.yaml`
  (V_in=24V, MOSFET-w-body-diode, free-wheeling diode,
  L=100µH, C_out=47µF, R_load=5Ω) with `dt = 100 ns`
- **AND** a 100 kHz / 50 % PWM switch_fn driving Q1
- **WHEN** the simulation runs for 5 ms
- **THEN** the mean V_out over the last 0.5 ms SHALL equal
  `V_in · D = 12 V` within ±0.5 V (the small loss from
  R_on + L's parasitic drop is tolerated)
- **AND** the peak-to-peak V_out over the last 0.5 ms
  SHALL be under 1 V.

### Requirement: Python buck-converter runner script

A Python runner script SHALL ship at `examples/v2/scripts/run_buck.py` that demonstrates loading the buck YAML, generating a PWM `switch_fn` from Python, running `run_transient`, and printing steady-state V_out statistics.

The script MUST run successfully on any system with `pulsim.v2`, `numpy`, and Python 3.10+ installed. Optional dependencies (matplotlib) MUST gracefully no-op when missing.

#### Scenario: Python runner prints expected V_out

- **GIVEN** a Python 3.10+ environment with `pulsim.v2` and `numpy` installed
- **WHEN** the user runs `python examples/v2/scripts/run_buck.py`
- **THEN** the script SHALL print a line containing `V_out mean:` followed by a value close to 12.0 V (the analytical target).
