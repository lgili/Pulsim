# pulsim-v2-pulse-source Specification

## Purpose

The `PulseVoltageSource` model (Layer 2 V12) is a first-class device that produces a SPICE-style PULSE waveform, including optional linear rise/fall ramps. It is the v2 equivalent of SPICE's `V… PULSE(v_initial v_pulsed t_start t_rise t_fall pulse_width period)` syntax.

It is architecturally identical to `PWMVoltageSource` (V4) and `SineVoltageSource` (V11): a `Source`-kind branch with a branch-current unknown stamped at `V = 0` baseline; the time-varying value is overlaid via `b_extra` at runtime by `run_transient`'s built-in AC pass.

Use cases include:
- Step response: `v_initial = 0`, `v_pulsed = V`, `t_start = 0`, `pulse_width` = very large → asymptotic step.
- Clock signal: periodic single-pulse train.
- Initial-condition perturbation: a single pulse to kick a system away from equilibrium.
- IGBT / MOSFET gate drive with realistic rise/fall edges.

vs. `PWMVoltageSource` (V4): PWMVoltageSource is a continuous square wave (frequency + duty), useful for power-electronics gate drives. `PulseVoltageSource` adds explicit `t_start` delay and single-shot mode — useful for transient-analysis step inputs that V4 cannot express.

## Requirements

### Requirement: PulseVoltageSource device model

The `PulseVoltageSource` SHALL provide a static `value_at(params, t)` returning the SPICE-style PULSE waveform.

Per-cycle evaluation (with `elapsed = (t − t_start) mod period` when `period > 0`, otherwise `elapsed = t − t_start`):

- If `t < t_start`: return `v_initial`.
- If `period > 0` AND `elapsed ≥ rise_time + pulse_width + fall_time` (rest of period): return `v_initial`.
- If `0 ≤ elapsed < rise_time`: linear ramp from `v_initial` to `v_pulsed`.
- If `rise_time ≤ elapsed < rise_time + pulse_width`: return `v_pulsed`.
- If `rise_time + pulse_width ≤ elapsed < rise_time + pulse_width + fall_time`: linear ramp from `v_pulsed` back to `v_initial`.
- After the pulse window completes (single-shot, `period == 0`): return `v_initial`.

Default `rise_time = fall_time = 0` SHALL reproduce instant-transition behaviour identical to a periodic step.

#### Scenario: Single-shot pulse fires at t_start

- **GIVEN** `PulseVoltageSource{v_initial=0, v_pulsed=5, t_start=1 ms, pulse_width=1 ms, period=0}`
- **WHEN** `value_at` is evaluated at `t = 0.5 ms`, `1.5 ms`, and `2.5 ms`
- **THEN** the returned values SHALL be `0`, `5`, and `0` respectively

#### Scenario: Pulse step charges an RC integrator

- **GIVEN** `add_pulse_voltage_source` driving an RC integrator with `R = 1 kΩ`, `C = 1 µF`, `τ = R·C = 1 ms`, `v_pulsed = 10 V`
- **WHEN** the simulation runs for 5 τ
- **THEN** `v_C(τ)`, `v_C(2τ)`, `v_C(3τ)` SHALL match the analytical `V · (1 − exp(−t/τ))` within 5 % at each sample

#### Scenario: SPICE-style rise/fall ramps

- **GIVEN** `PulseVoltageSource{v_initial=0, v_pulsed=1, t_start=0, rise_time=100 ns, pulse_width=900 ns, fall_time=100 ns, period=0}`
- **WHEN** `value_at` is evaluated at `t = 50 ns` (mid-rise)
- **THEN** the returned value SHALL be approximately 0.5 (linear interpolation halfway up the rising edge)
- **AND** at `t = 1.05 µs` (mid-fall), it SHALL return approximately 0.5 again

### Requirement: DC-operating-point support

`compute_dc_op(graph, pool, mask, t_eval)` SHALL evaluate every `PulseVoltageSource` branch at the supplied `t_eval`.

#### Scenario: DC OP during pulse returns v_pulsed

- **GIVEN** `PulseVoltageSource{v_initial=0, v_pulsed=5, t_start=1 ms, pulse_width=1 ms, period=0}` driving a 1 Ω load
- **WHEN** `compute_dc_op` is called with `t_eval = 1.5 ms` (mid-pulse)
- **THEN** the solved `v_n0` SHALL equal 5.0 V within 1 µV
