## Context

`IdealDiode` is the oldest Newton device: an AD-differentiated smooth
blend used by the DC continuation strategies (κ- and `V_F0`-homotopy,
`pwl/initial_guess.hpp`, `pwl/nonlinear_refresh_diode.hpp`) and by the
SPICE importer. Its law multiplies a logistic by a line through
`V_F0`; the product is negative on the low side of the line.

## Goals / Non-Goals

- Goals: `i·v ≥ 0` and `di/dv > 0` for every finite `v` and every valid
  parameter set; same parameters; same forward asymptote; the importer
  stops producing this device for a model that is piecewise linear.
- Non-Goals: retiring `IdealDiode`; junction capacitance, series
  resistance or temperature; changing the Shockley or PWL diodes.

## Decisions

- **Law.**
  `i(v) = [sp(κ(v − V_F0)) − sp(−κ·V_F0)]/(κ·R_d) + G_off·v`.
  - The constant `sp(−κ·V_F0)` makes `i(0) = 0`. Without it the plain
    softplus gives `i(0) > 0`, which is active for `v < 0` (4.2 µA at
    the defaults). The price is a reverse floor of the same size:
    `i(v → −∞) → −sp(−κ·V_F0)/(κ·R_d) + G_off·v`.
  - Leakage is the plain `G_off·v` instead of `(1 − α)·v·G_off`: it is
    monotone for every parameter set and differs from the old term by at
    most `G_off·v` in forward conduction (1e-9 S against 100 S).
  - `sp(x) = max(x, 0) + log1p(exp(−|x|))`, derivative `logistic(x)`;
    the AD overload sets the derivative directly so no `exp` overflows
    (the reason `numeric/logistic.hpp` exists).
- **Values at the defaults** (from the formula, to be pinned by tests):
  `i(−10) = −4.168e-6 A`, `i(0) = 0`, `i(0.5) = 0.090746 A`,
  `i(0.636) = 1.22662 A`, `i(0.7) = 3.46573 A`, `i(0.75) = 6.5663 A`,
  `i(1.0) = 30.0124 A`; `v(1 A) = 0.6246 V`, `v(10 A) = 0.7927 V`
  (ideal knee: 0.710 V and 0.800 V). The knee is rounded outward by at
  most `ln 2/κ` = 34.7 mV in voltage.
- **Alternatives considered.**
  - *Algebraic smooth-max `(x + √(x² + ε²))/2`*, already used by
    `igbt_level1.hpp:170`: also passive, but its reverse tail decays as
    `1/|v|` instead of exponentially, and `κ` would lose its meaning.
  - *Clamp `i_on` at zero* (`max(·, 0)`): passive but the Jacobian is
    discontinuous, which breaks the 1e-12 smoothness the Newton devices
    need.
  - *Retire the device and alias it to `add_diode(V_f=)`*: removes the
    homotopy device the DC strategies are built on.
- **Importer.** `V_th = 0`, `V_f = Vfwd`, so the diode turns on at
  `v ≥ Vfwd` and off at zero current: LTspice's rule. `area` scales
  `g_on` and `g_off` as it scaled `R_d` and `G_off`. The existing
  refusals (`Vrev`, `Rrev`, mixed parameter sets) and the dropped
  softening knobs are unchanged.

## Risks / Trade-offs

- The `V_F0`-homotopy starts from negative `V_F0`
  (`core/tests/layer5_v4/test_vf0_continuation_rectifier.cpp`). With the
  offset the start point is a resistor `R_d` instead of a diode shifted
  left; the end point is the same law. Mitigation: the six `layer5_v4`
  rectifier test files are the oracle; if a path stops converging, the homotopy
  keeps the un-offset form internally and only the device law changes.
- A small `κ·V_F0` gives a large reverse floor (`V_F0 = 0`:
  `ln 2/(κ·R_d)` = 3.47 A at the defaults). The old law had the same
  magnitude of error there with the opposite sign. Documented; see Open
  Questions.
- An imported `Vfwd = 0` diode has a zero-width hysteresis band, like
  any `V_th = 0` PWL diode: `engine='dsed'` may fail to settle with a
  named error and `trbdf2` may report chatter. Both are loud.

## Migration Plan

One release note. No parameter migration. Rollback: revert the law
header and the importer branch independently.

## Open Questions

1. Should `add_nonlinear_diode` refuse (or warn on) a parameter set whose
   reverse floor `sp(−κ·V_F0)/(κ·R_d)` exceeds a threshold (for example
   `κ·V_F0 < 10`, floor 0.23 mA at the default `R_d`)? Not proposed,
   because the DC continuation uses such sets on purpose.
2. Importer: `V_th = 0` exactly (proposed, LTspice semantics) or
   `V_th = Vfwd + 1 mV` to give DSED and bridges a hysteresis band?
3. Should the default `κ` rise (for example to 100 /V, knee rounding
   6.9 mV) now that the law is passive? Not proposed: it changes more
   numbers than the fix needs.
