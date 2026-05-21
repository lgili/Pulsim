# pulsim-v2-sine-source Specification

## Purpose

The `SineVoltageSource` model (Layer 2 V11) is a first-class device that produces an AC voltage `v_dc + v_amplitude · sin(2π · f · t + φ)` at every instant. It eliminates the previously-common workaround of writing a custom `b_extra_fn(t)` lambda for AC mains, audio amps, grid analysis, or harmonic studies.

Architecturally it is a `Source`-kind branch with a branch-current unknown (identical pattern to `PWMVoltageSource` V4 and `PulseVoltageSource` V12). The static MNA matrix is stamped with `V = 0` as a baseline; the time-varying sine value is overlaid via `b_extra` by `run_transient`'s built-in AC pass.

## Requirements

### Requirement: SineVoltageSource device model

The `SineVoltageSource` SHALL provide a static `value_at(params, t)` returning `v_dc + v_amplitude · sin(2π · frequency · t + phase)`.

Degenerate inputs:
- If `frequency ≤ 0`, the helper SHALL return `v_dc` (no AC component).
- If `v_amplitude == 0`, the helper SHALL return `v_dc` exactly.

The `phase` parameter SHALL be interpreted as RADIANS (consistent with the SPWM helper family; PWMVoltageSource V4 uses seconds because its phase is a cycle offset).

#### Scenario: Sine source drives a resistive load

- **GIVEN** `add_sine_voltage_source("Vac", "n0", "gnd", v_dc=0, v_amplitude=10, frequency=60 Hz)`
- **AND** a 100 Ω resistor from `n0` to ground
- **WHEN** the transient simulator runs for one full cycle (16.67 ms)
- **THEN** `v_n0(t)` SHALL match `10 · sin(2π · 60 · t)` within 0.01 V at every sample

#### Scenario: DC offset is preserved

- **GIVEN** `SineVoltageSource{v_dc=3.0, v_amplitude=2.0, frequency=1 kHz}`
- **WHEN** evaluated at `t = 0` (where sin = 0)
- **THEN** `value_at` SHALL return exactly 3.0

#### Scenario: Frequency ≤ 0 returns the DC offset

- **GIVEN** `SineVoltageSource{v_dc=5.0, v_amplitude=10.0, frequency=0}`
- **WHEN** evaluated at any `t`
- **THEN** `value_at` SHALL return 5.0 (no AC component)

### Requirement: DC-operating-point support

`compute_dc_op(graph, pool, mask, t_eval)` SHALL evaluate every `SineVoltageSource` branch at the supplied `t_eval` and stamp the corresponding voltage in the MNA system.

#### Scenario: DC OP at quarter-cycle matches +amplitude

- **GIVEN** `SineVoltageSource{v_dc=0, v_amplitude=5, frequency=1 kHz}` driving a 1 Ω load
- **WHEN** `compute_dc_op` is called with `t_eval = 250 µs` (T/4 where sin = 1)
- **THEN** the solved `v_n0` SHALL equal 5.0 V within 1 µV
