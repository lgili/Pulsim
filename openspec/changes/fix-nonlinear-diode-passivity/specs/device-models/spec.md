## ADDED Requirements

### Requirement: Smooth Diode Law Is Passive and Monotone

The smooth diode (`models::IdealDiode`, `add_nonlinear_diode`) SHALL
conduct `i(v) = [sp(κ·(v − V_F0)) − sp(−κ·V_F0)]/(κ·R_d) + G_off·v`,
with `sp(x) = ln(1 + eˣ)` and `v = v_anode − v_cathode`. For every
finite `v` and every parameter set with `R_d > 0`, `κ > 0` and
`G_off ≥ 0`, the law SHALL satisfy `i(0) = 0`, `i·v ≥ 0` and
`di/dv = logistic(κ·(v − V_F0))/R_d + G_off > 0`. Its value and Jacobian
SHALL stay finite for `κ·|v − V_F0|` up to at least 1000. The reverse
current SHALL be bounded below by
`−sp(−κ·V_F0)/(κ·R_d) + G_off·v`.

#### Scenario: Below the knee the diode absorbs power
- **GIVEN** the default parameters (`V_F0 = 0.7 V`, `R_d = 10 mΩ`,
  `κ = 20 /V`, `G_off = 1 nS`)
- **WHEN** 0.636 V is forced across the diode
- **THEN** the current is +1.2266 A within 1e-3 A (it was −1.392 A)

#### Scenario: Monotone and passive over the whole range
- **WHEN** the law is evaluated on 550 001 equally spaced points in
  [−50 V, +5 V] with the default parameters
- **THEN** the current is strictly increasing, `i·v ≥ 0` everywhere and
  `i(0)` is exactly 0

#### Scenario: Forward asymptote is unchanged
- **WHEN** 1.0 V is forced across the diode with the default parameters
- **THEN** the current is 30.012 A within 1e-3 A, against 30.000 A for
  the line `(v − V_F0)/R_d`

#### Scenario: A sine clamp runs on the fixed engine
- **GIVEN** a 20 V, 1 kHz sine source, 10 Ω in series and the diode to
  ground
- **WHEN** 2 ms are simulated on `engine='pwl'` at `dt = 1 µs` and on
  `engine='trbdf2'`
- **THEN** both runs complete and the time-weighted RMS of the diode
  voltage is 10.010 V within 0.02 V
