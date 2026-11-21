## ADDED Requirements

### Requirement: Charge Table Law

The kernel SHALL provide a `ChargeTable` charge law `Q(v)` defined by
knots `(v_k, Q_k, C_k)` with `v` strictly increasing, a knot at `v = 0`
with `Q = 0`, and `C_k > 0`. Between knots it SHALL be the cubic Hermite
interpolant with the stored `C_k` as tangents, and `C(v)` SHALL be the
exact derivative of the evaluated `Q(v)`. Beyond the end knots it SHALL
continue linearly with the supplied tail slopes. Construction SHALL
check the Fritsch–Carlson monotonicity condition and SHALL refuse, with
an error naming the device, fewer than three knots, a missing `v = 0`
knot, non-increasing `v`, or a non-positive `C`.

#### Scenario: Reproduces the junction law
- **GIVEN** a table built at 64 knots with exact slopes from
  `C0 = 2 nF`, `V0 = 25 V`, `m = 0.5`
- **WHEN** it is evaluated at 400 V
- **THEN** `Q = 312.31 nC` within 1e-4 relative and `C = 485.07 pF`
  within 1e-3 relative

#### Scenario: Monotone at picovolt spacing
- **WHEN** `Q(v)` is evaluated at 1 pV spacing around 0 V, 1 V and 400 V
- **THEN** it is strictly increasing

#### Scenario: Non-monotone data refused
- **WHEN** a table is built with a knot whose `C` is zero or negative
- **THEN** construction fails with an error naming the device and the
  offending knot

### Requirement: Nonlinear Capacitor Charge-Law Variants

`NonlinearCapacitor` SHALL accept one of three charge laws: `junction`
(`C0/(1 + v/V0)^m`), `table` (a `ChargeTable`), and `vdmos_cgd`, the
closed form with `s = (CGDMAX − CGDMIN)/(1 + π/2)`, `y = CGDMAX − s`:
`C = y + s·tanh(a·u)` for `u > 0` and `C = y + s·atan(a·u)` for `u ≤ 0`,
with `Q(u)` its exact integral. Every law SHALL be stamped in charge
form, so that the charge moved over a closed voltage cycle is zero. The
`junction` law SHALL be bit-identical to the device before the variant
existed.

#### Scenario: VDMOS gate-drain law matches ngspice
- **GIVEN** `CGDMAX = 1 nF`, `CGDMIN = 50 pF`, `A = 0.5`
- **WHEN** `C` is evaluated at `v_gd` = 5, 2, 1, 0, −1, −2, −5, −20,
  −100 and −400 V
- **THEN** it equals the ngspice 45.2 VDMOS small-signal C_gd (rounded:
  995.05, 911.90, 801.23, 630.47, 459.13, 340.23, 190.61, 86.83, 57.39,
  51.85 pF) within 1e-6 relative

#### Scenario: Charge conserved over a cycle
- **GIVEN** a capacitor with any of the three laws charged from 0 to
  400 V and back through 1 kΩ
- **WHEN** the run completes on `pwl` or `trbdf2`
- **THEN** the integral of its current is zero within 1e-9 of the
  charge moved

### Requirement: Datasheet Power MOSFET Device

The system SHALL provide a power MOSFET device with pins drain, source
and gate, composed of: a channel (the Level-1 law with its
third-quadrant symmetry), a drift resistance `R_d`, a constant `C_gs`,
charge-law capacitances `C_gd(v_dg)` and `C_ds(v_ds)`, an internal gate
resistance `R_g`, an optional common source inductance `L_s`, and an
optional Lauritzen or Shockley body diode. A run-constant junction
temperature `T_j` SHALL scale `R_d` by `r(T) = 1 + tc1·ΔT + tc2·ΔT²` and
`K` by `1/r(T)`, and shift `V_T` by `vt_tc·ΔT`, with `ΔT = T_j − 25 °C`.
The device SHALL be available through the builder, Python and the YAML
type `power_mosfet`. Invalid parameters SHALL be refused with an error
naming the device and the parameter.

#### Scenario: Gate resistance sets the Miller plateau
- **GIVEN** a clamped inductive double pulse at 400 V and 10 A with
  `K = 1.43 A/V²`, `V_T = 3 V`, `C_gs = 1 nF`, a linear `C_gd = 100 pF`
  and a 0 → 12 V gate drive
- **WHEN** it runs with a total gate resistance of 10 Ω
- **THEN** the gate voltage shows a plateau whose duration is within
  10 % of `Q_gd / I_g` = 62.9 ns, and the plateau lengthens when the
  gate resistance is raised to 47 Ω

#### Scenario: Engines agree on the switching energies
- **WHEN** the same double pulse runs on `pwl` and on `trbdf2`
- **THEN** E_on and E_off differ by less than 0.5 %

#### Scenario: On-resistance follows the temperature factor
- **GIVEN** `tc1 = 5e-3 /K`, `tc2 = 2e-5 /K²`
- **WHEN** the device conducts 0.1 A at the datasheet gate voltage with
  `T_j = 125 °C` and with `T_j = 25 °C`
- **THEN** the ratio of the two on-state voltages is 1.700 within 1e-3

#### Scenario: Third quadrant
- **WHEN** 10 A flows source to drain with the gate on
- **THEN** `v_DS = −10·R_DS(on)` within 1e-6
- **AND** with the gate off the body diode carries the current

#### Scenario: Unsupported engine refused
- **WHEN** a circuit containing the device is run on `dsed`
- **THEN** it raises the conformance error naming the device and the
  engine, and does not run with the capacitances removed

### Requirement: Part Validation by Held-Out Datasheet Data

A shipped power-MOSFET part SHALL carry, for every fitted curve, the
datasheet document, revision and figure it came from, and SHALL declare
the datasheet double-pulse test condition. Its `fitted_on` and
`validated_on` lists SHALL be disjoint, and switching energies SHALL NOT
appear in `fitted_on`. A part SHALL be labelled `validated` only while
its double-pulse acceptance check reproduces the datasheet E_on and
E_off at that condition within the tolerance the part declares, which
SHALL NOT exceed 25 %. A part whose data is not the manufacturer's SHALL
remain `provenance.method: synthetic` and refused by part number.

#### Scenario: Acceptance check gates the label
- **GIVEN** a part with a `double_pulse` block and a declared tolerance
  of 20 %
- **WHEN** its acceptance check runs in CI
- **THEN** simulated E_on and E_off are each within 20 % of the
  datasheet values, or the check fails and the part cannot be labelled
  `validated`

#### Scenario: Tuned data is not validation
- **WHEN** a part file lists the same curve in `fitted_on` and
  `validated_on`
- **THEN** loading it fails with an error naming the curve

### Requirement: External Reference for Newton Devices

The reference benchmark set SHALL contain ngspice decks for the Level-1
MOSFET static curves, the Shockley diode static curve, the three charge
laws and a VDMOS double pulse, each with a stated tolerance, run by the
nightly reference job.

#### Scenario: Charge laws against ngspice
- **WHEN** the capacitance-sweep decks run
- **THEN** the junction and `vdmos_cgd` capacitances equal ngspice's
  within 1e-6 relative

#### Scenario: Imported card against ngspice
- **GIVEN** one VDMOS card used in the same double-pulse circuit in
  ngspice and in Pulsim
- **WHEN** both run
- **THEN** E_on, E_off and the plateau duration differ by less than the
  tolerance recorded in the deck (target 5 %, 5 % and 3 %)
