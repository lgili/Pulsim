## ADDED Requirements

### Requirement: Linear Controlled Sources

The kernel SHALL provide a voltage-controlled current source
(`i = g_m·v_c`), a current-controlled voltage source (`v = r_m·i_c`), a
current-controlled current source (`i = β·i_c`) and a gyrator
(`i_1 = g·v_2`, `i_2 = −g·v_1`). Each SHALL be a constant linear stamp
that does not depend on the switch mask. A current output SHALL follow
the terminal convention of the DC current source: a positive value
leaves the source at `from`. The controlling branch of a
current-controlled source SHALL carry a current unknown; any other
branch SHALL be refused with an error naming both elements and the
valid kinds. A non-finite parameter SHALL be refused by name.

#### Scenario: Transconductance
- **GIVEN** a VCCS with `g_m = 2 mS` controlled by 5 V, its `from` node
  loaded with 1 kΩ to ground and `to` grounded
- **WHEN** the DC operating point is solved
- **THEN** the `from` node is at +10.000 V within 1e-12 relative

#### Scenario: Transresistance
- **GIVEN** 10 mA flowing from `from` to `to` through a 0 V sense
  source, and a CCVS with `r_m = 100 Ω` controlled by it
- **WHEN** the circuit is solved
- **THEN** the CCVS output is 1.000 V within 1e-12 relative

#### Scenario: Current gain
- **GIVEN** 1 mA sensed, a CCCS with `β = 50` into a 100 Ω load
- **WHEN** the circuit is solved
- **THEN** the load voltage magnitude is 5.000 V within 1e-12 relative

#### Scenario: Controlling branch without a current unknown
- **WHEN** `add_cccs("F1", control="R1", ...)` names a resistor
- **THEN** it raises an error naming `F1`, `R1` and the branch kinds
  that can be sensed

### Requirement: Gyrator Is Lossless and Inverts Impedance

The gyrator SHALL absorb zero net power at every instant and SHALL
present at one port the impedance `1/(g²·Z)` of the element `Z`
connected to the other.

#### Scenario: Capacitor seen as inductor
- **GIVEN** a gyrator with `g = 0.01 S` and 1 µF on port 2, driven on
  port 1 by a 1 V step through 10 Ω
- **WHEN** it runs on `pwl` beside a 10 mH inductor driven the same way
  on the same grid
- **THEN** the two port-1 currents agree within 1e-9 relative, and the
  current at 1 ms is 63.21 mA

#### Scenario: Power balance
- **WHEN** any transient with a gyrator is run
- **THEN** `|v_1·i_1 + v_2·i_2| ≤ 1e-12·|v_1·i_1|` at every step

#### Scenario: Small-signal impedance
- **WHEN** the AC sweep is run on the gyrator with 1 µF on port 2
- **THEN** the port-1 impedance at 1 kHz is 62.83 Ω at +90° within
  1e-9 relative

### Requirement: Time-Varying Current Sources

The kernel SHALL provide sine, pulse and piecewise-linear table current
sources evaluated without a Python callback. The sine source SHALL
produce `i_dc + i_amplitude·sin(2π·f·t + φ)`. The pulse source SHALL
take the parameters and edge rules of the pulse voltage source. The
table source SHALL interpolate linearly between knots `(t_k, i_k)`,
hold the first value before the first knot, and after the last knot
either hold the last value or repeat the table when `periodic` is set.
A table with fewer than two knots, non-increasing times, a non-finite
value, or `periodic` with unequal end values SHALL be refused naming the
source. The DC operating point SHALL use the value at `t = 0`.

#### Scenario: Sine current into a resistor
- **GIVEN** a sine current source of 1 A amplitude at 1 kHz into 10 Ω
- **WHEN** it runs on `pwl`
- **THEN** the resistor voltage equals `10·i(t)` at every grid point
  within 1e-12 relative

#### Scenario: Table source delivers its charge
- **GIVEN** a table source ramping 0 → 1 A → 0 over 2 ms into 1 mF
- **WHEN** it runs on `pwl` with the knots on the grid
- **THEN** the capacitor voltage rises by 1.000 V within 1e-9 relative

#### Scenario: Breakpoints are not stepped over
- **WHEN** the same source runs on `trbdf2` with knots that no natural
  step would hit
- **THEN** the capacitor voltage rises by 1.000 V within 1e-6 relative

#### Scenario: Invalid table refused
- **WHEN** a table source is given decreasing times
- **THEN** building fails with an error naming the source

### Requirement: Controlled Sources in Every Analysis

The controlled sources and the time-varying current sources SHALL give
the same result on `pwl` and `trbdf2`, SHALL be part of the DC operating
point and of the AC sweep, and on `dsed` SHALL either run and agree or
be refused with the conformance error naming the element and the
engine. Circuits that use none of them SHALL be bit-identical to before.

#### Scenario: Engines agree
- **WHEN** a circuit with each of the four controlled elements runs on
  `pwl` and `trbdf2`
- **THEN** node voltages agree within 1e-6 relative

#### Scenario: No silent engine
- **WHEN** the same circuit is run on `dsed`
- **THEN** it either agrees within 1e-6 or raises the conformance error
  naming the element
