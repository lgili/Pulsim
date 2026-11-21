## ADDED Requirements

### Requirement: Retired Smooth-Blend Diode Requires an Explicit Opt-In

The smooth-blend diode (`models::IdealDiode`) SHALL NOT be built from
Python or YAML without an explicit opt-in, because its law
`logistic(κ·(v − V_F0))·(v − V_F0)/R_d` is negative for
`0 < v < V_F0`. `CircuitBuilder.add_nonlinear_diode` SHALL raise
`pulsim.RetiredDeviceError`, a subclass of `ValueError`, unless called
with `allow_non_passive=True`; nothing SHALL be stamped on refusal. The
message SHALL name the device, the measured −1.39 A at +0.636 V and the
replacements `add_diode(..., V_f=)` and `add_shockley_diode`. A YAML
device of `type: nonlinear_diode` SHALL be refused unless it sets
`allow_non_passive: true`. With the opt-in, the law SHALL be unchanged.

#### Scenario: The builder refuses the device
- **WHEN** `add_nonlinear_diode("Dx", "a", "gnd", IdealDiodeParams())`
  is called without the opt-in
- **THEN** `RetiredDeviceError` is raised, its message contains `Dx`,
  `-1.39 A`, `+0.636 V`, `add_diode`, `add_shockley_diode` and
  `allow_non_passive=True`, and the builder has no branch

#### Scenario: The opt-in keeps the legacy law
- **GIVEN** the device built with `allow_non_passive=True` and default
  parameters
- **WHEN** 0.636 V is forced across it on `engine='pwl'`
- **THEN** the diode current is −1.3923 A within 1e-3 A

#### Scenario: YAML refuses without the key
- **WHEN** a YAML circuit declares `type: nonlinear_diode` without
  `allow_non_passive: true`
- **THEN** loading fails with a message naming the device and the key
