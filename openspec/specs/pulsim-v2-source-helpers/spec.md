# pulsim-v2-source-helpers Specification

## Purpose

Pulsim v2 ships a family of header-only `switch_fn` factory functions that take user-friendly converter parameters (frequency, duty, dead-time, phase) and return a `Real → SwitchStateMask` callable usable directly by `run_transient`. This eliminates the most common form of pulse-train boilerplate from user code.

The helpers cover six power-electronics topologies that account for >90 % of common open-loop SMPS simulations:

| Helper | Topology | Layer |
|---|---|---|
| `make_pwm_switch_fn` | Single PWM-driven switch | V5 |
| `make_dead_time_pwm_pair_fn` | Half-bridge with anti-shoot-through dead-time | V6 |
| `make_spwm_pair_fn` | Sinusoidal PWM half-bridge | V7 |
| `make_three_phase_spwm_fn` | 3-phase SPWM inverter | V8 |
| `make_phase_shift_full_bridge_fn` | Phase-shifted full-bridge ZVS | V9 |
| `make_combined_switch_fn` | Compose two switch_fn callables | V10 |

All helpers are header-only inline functions, return `std::function<SwitchStateMask(Real)>`, and are exposed through both the C++ namespace `pulsim::v2::sources` and the Python `pulsim.v2` module.

## Requirements

### Requirement: Single-Switch PWM helper (V5)

`make_pwm_switch_fn(switch_index, frequency, duty, phase)` SHALL return a `switch_fn(t)` that drives the bit at `switch_index` to ON during the first `duty · T` of each period and OFF for the rest, with optional phase offset.

The factory SHALL:
- Compute the period as `T = 1 / frequency`.
- Treat `phase` as seconds (cycle-relative offset).
- Wrap the elapsed time into `[0, T)` so the duty test is independent of cycle count.
- Set only the bit at `switch_index`; all other switches default OFF.

#### Scenario: 50 % duty cycle produces 50 % ON-time over many periods

- **GIVEN** `make_pwm_switch_fn(switch_index=0, frequency=10 kHz, duty=0.5)`
- **WHEN** sampled at 100 evenly-spaced points across 10 periods
- **THEN** approximately 50 % of samples SHALL have bit 0 set
- **AND** the ON intervals SHALL all align with the first half of each period

#### Scenario: Phase offset shifts the rising edge

- **GIVEN** two helpers, one with `phase = 0`, one with `phase = T/4`
- **WHEN** sampled at `t = T/4`
- **THEN** the phase-shifted variant SHALL be in the OFF state while the un-shifted one SHALL be in the ON state at that instant

### Requirement: Half-bridge dead-time PWM pair helper (V6)

`make_dead_time_pwm_pair_fn(high_switch_index, low_switch_index, frequency, duty, dead_time, phase)` SHALL drive a complementary half-bridge with anti-shoot-through dead-time between the high-side and low-side switches.

The helper SHALL:
- During `[0, duty · T − dead_time)`: drive high-side ON, low-side OFF.
- During `[duty · T − dead_time, duty · T)`: drive BOTH off (dead-time).
- During `[duty · T, T − dead_time)`: drive low-side ON, high-side OFF.
- During `[T − dead_time, T)`: drive BOTH off (second dead-time).
- Throw `std::invalid_argument` if `2 · dead_time ≥ T`.

#### Scenario: Dead-time interval is observed

- **GIVEN** `make_dead_time_pwm_pair_fn(high=0, low=1, frequency=100 kHz, duty=0.5, dead_time=200 ns)`
- **WHEN** sampled at `t` inside the dead-time interval
- **THEN** both bit 0 and bit 1 SHALL be OFF
- **AND** outside the dead-time, exactly one of the two SHALL be ON (never both)

#### Scenario: Invalid dead-time raises

- **GIVEN** dead_time = 6 µs and frequency = 100 kHz (so T = 10 µs, 2·dt = 12 µs > T)
- **WHEN** the factory is called
- **THEN** it SHALL throw `std::invalid_argument`

### Requirement: Sinusoidal-PWM half-bridge helper (V7)

`make_spwm_pair_fn(high_switch_index, low_switch_index, carrier_frequency, modulating_frequency, modulation_index, phase)` SHALL drive a half-bridge using sinusoidal modulation of the duty cycle.

The instantaneous duty SHALL be `0.5 + 0.5 · m · sin(2π · f_mod · t + φ)` where `m = modulation_index ∈ [0, 1]`.

#### Scenario: Modulation index 0 produces 50 % constant duty

- **GIVEN** `make_spwm_pair_fn(high=0, low=1, carrier=10 kHz, mod_freq=60 Hz, m=0)`
- **WHEN** sampled over one carrier cycle
- **THEN** the ON-time fraction of bit 0 SHALL be approximately 0.5

### Requirement: Three-phase SPWM inverter helper (V8)

`make_three_phase_spwm_fn(leg_indices, carrier_frequency, modulating_frequency, modulation_index, phase)` SHALL drive a 6-switch 3-leg inverter using SPWM with 120° phase shifts between legs.

`leg_indices` is a `ThreePhaseLegIndices` struct holding `(a_high, a_low, b_high, b_low, c_high, c_low)`.

#### Scenario: Three legs are phase-shifted by 120°

- **GIVEN** a `make_three_phase_spwm_fn` with mod_freq = 60 Hz
- **WHEN** the modulating sinusoid is reconstructed from the duty cycle of each leg
- **THEN** legs A → B SHALL be offset by `T_mod / 3` (~5.5 ms at 60 Hz)
- **AND** legs B → C SHALL be offset by another `T_mod / 3`

### Requirement: Phase-shift full-bridge ZVS helper (V9)

`make_phase_shift_full_bridge_fn(diag1_switch_index, diag2_switch_index, frequency, phase_shift)` SHALL drive a four-switch full bridge in ZVS-friendly phase-shift mode.

The two diagonals run at fixed 50 % duty with `phase_shift` (radians, 0..π) between them.

#### Scenario: Zero phase shift gives full conduction

- **GIVEN** `make_phase_shift_full_bridge_fn(d1=0, d2=1, frequency=100 kHz, phase_shift=0)`
- **WHEN** sampled at any instant
- **THEN** both diagonals SHALL conduct synchronously (both ON for half-period each)

### Requirement: Composing multiple switch_fn callables (V10)

`make_combined_switch_fn(a, b, …)` SHALL accept any number of `switch_fn`s and return a callable whose result is the bitwise OR of each input mask.

#### Scenario: Two independent PWMs produce the OR'd mask

- **GIVEN** `pwm1` driving bit 0 at 100 kHz and `pwm2` driving bit 1 at 50 kHz
- **WHEN** `combined = make_combined_switch_fn(pwm1, pwm2)` is sampled at some `t`
- **THEN** `combined(t)` SHALL equal `pwm1(t) | pwm2(t)`
