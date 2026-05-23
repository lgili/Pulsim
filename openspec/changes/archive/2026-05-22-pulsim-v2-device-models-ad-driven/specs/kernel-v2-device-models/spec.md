## ADDED Requirements

### Requirement: DeviceModel Concept

`pulsim::v2::models::DeviceModel<T>` SHALL be a C++20 concept that
any v2 device model satisfies. The concept requires:

1. A nested `typename T::Params` — the device's parameter struct.
2. `static constexpr T::kind` of type `topology::BranchKind`.
3. `static constexpr T::num_terminals` of type `Size`.
4. `static constexpr T::is_linear` of type `bool`.
5. A templated static function `T::template current<S>(const S*
   v, const Params& p) noexcept` accepting any
   `numeric::FloatingPoint S` and returning `S`. The `v` pointer
   points to `num_terminals` terminal voltages in device-defined
   order; the return value is the current flowing from terminal 0
   to terminal 1 (convention).

The concept enables Layer 3's generic stamping pipeline to consume
any device model without knowing its concrete type. Manual derivative
implementations are NOT permitted — every device implements
`current<S>` once, and the AD scalar derives the Jacobian from the
same code path.

#### Scenario: Reference models satisfy the concept

- **GIVEN** the reference models `Resistor`, `VoltageSource`,
  `IdealDiode`
- **WHEN** the consumer evaluates
  `pulsim::v2::models::DeviceModel<Resistor>` etc. via a
  `static_assert`
- **THEN** each evaluation SHALL be `true`.

#### Scenario: A struct missing a required member fails the concept

- **GIVEN** a struct `BrokenModel` that has `Params` and `kind` but
  lacks `current<S>`
- **WHEN** the consumer evaluates `DeviceModel<BrokenModel>`
- **THEN** the result SHALL be `false`.

### Requirement: Resistor Device Model

`pulsim::v2::models::Resistor` SHALL be a linear two-terminal device
model satisfying the `DeviceModel` concept. Its `Params` exposes a
single `Real G` (conductance in siemens). The `current<S>` function
SHALL implement `i = G · (v[0] - v[1])`.

Static constants:
- `kind = topology::BranchKind::PassiveLinear`
- `num_terminals = 2`
- `is_linear = true`

#### Scenario: Forward evaluation matches Ohm's law

- **GIVEN** `Resistor::Params{ G = 2.0 }` and terminal voltages
  `v = [3.0, 1.0]`
- **WHEN** the user calls `Resistor::current<Real>(v, p)`
- **THEN** the result SHALL equal `4.0` (within 1e-12).

#### Scenario: AD-derived partials are exactly ±G

- **GIVEN** `Resistor::Params{ G = 2.0 }`
- **AND** seeded AD inputs `auto [v_pos, v_neg] = ad::seed<2>(3.0,
  1.0)`
- **WHEN** the user calls `Resistor::current<ADRealN<2>>(...)`
- **THEN** the result's `deriv(0)` SHALL equal `2.0` (= G)
- **AND** `deriv(1)` SHALL equal `-2.0` (= -G).

### Requirement: VoltageSource Device Model

`pulsim::v2::models::VoltageSource` SHALL be a constraint device
(not a current contributor). Its `Params` exposes `Real V` (the DC
source voltage). The `current<S>` function SHALL return `S(0)` —
the source's contribution is a constraint row, NOT a stamped current,
and Layer 3 detects `kind == Source` to add the constraint.

A static accessor `VoltageSource::static_voltage(const Params&)`
SHALL return the configured `V` for Layer 3's use.

Static constants:
- `kind = topology::BranchKind::Source`
- `num_terminals = 2`
- `is_linear = true`

#### Scenario: current returns zero regardless of terminal voltages

- **GIVEN** `VoltageSource::Params{ V = 12.0 }` and any terminal
  voltages
- **WHEN** the user calls `VoltageSource::current<Real>(v, p)`
- **THEN** the result SHALL equal `0.0`.

#### Scenario: static_voltage returns the configured voltage

- **GIVEN** `VoltageSource::Params{ V = 12.0 }`
- **WHEN** the user calls `VoltageSource::static_voltage(p)`
- **THEN** the result SHALL equal `12.0`.

### Requirement: IdealDiode Device Model — Smooth-Blend Behavioral

`pulsim::v2::models::IdealDiode` SHALL be a nonlinear two-terminal
device model implementing the smooth-blend Norton-shifted diode:

```
v_diode = v[0] - v[1]
alpha   = 1 / (1 + exp(-kappa · (v_diode - V_F0)))
i_on    = alpha · (v_diode - V_F0) / R_d
i_off   = (1 - alpha) · v_diode · G_off
current = i_on + i_off
```

`Params` exposes `Real V_F0`, `Real R_d`, `Real G_off`, `Real kappa`.

Static constants:
- `kind = topology::BranchKind::Nonlinear`
- `num_terminals = 2`
- `is_linear = false`

The AD-derived partial derivatives ∂current/∂v[0] and ∂current/∂v[1]
MUST match a central-difference finite-difference baseline within
absolute tolerance `1e-6` at three op-points: sub-threshold
(v_diode < V_F0), at-threshold (v_diode ≈ V_F0), and forward-biased
(v_diode > V_F0).

#### Scenario: Reverse-biased current is near zero

- **GIVEN** `IdealDiode::Params{ V_F0=0.7, R_d=0.01, G_off=1e-9,
  kappa=50 }` and `v = [0, 5]` (v_diode = -5)
- **WHEN** the user calls `IdealDiode::current<Real>(v, p)`
- **THEN** the result SHALL be in `[-1e-8, 0]` (essentially G_off
  leakage in reverse).

#### Scenario: Forward-biased current scales linearly past V_F0

- **GIVEN** `IdealDiode::Params{ V_F0=0.7, R_d=0.01, ... }` and
  `v = [1.0, 0]` (v_diode = 1.0, well above V_F0)
- **WHEN** the user calls `IdealDiode::current<Real>(v, p)`
- **THEN** the result SHALL be approximately `(1.0 - 0.7) / 0.01 =
  30 A` within 1 % (the smooth-blend asymptote).

#### Scenario: AD partials match finite-difference at three op-points

- **GIVEN** the diode params above and a finite-difference step
  `h = 1e-6`
- **WHEN** the user computes ∂i/∂v[0] via AD and via central
  difference at `v = [0.5, 0]`, `v = [0.7, 0]`, and `v = [1.0, 0]`
- **THEN** the AD result SHALL match the central-difference result
  to within `1e-6` absolute at every op-point.

#### Scenario: Current depends only on v[0] - v[1]

- **GIVEN** any IdealDiode params and any terminal voltages
- **WHEN** the user computes `∂i/∂v[0] + ∂i/∂v[1]` via AD
- **THEN** the result SHALL equal `0.0` to within `1e-9`
  (the current is a function of the difference, so the partials
  are equal in magnitude and opposite in sign).
