## ADDED Requirements

### Requirement: Branch-Form Switch and Diode

The device library SHALL offer a branch form of the switch and of the
PWL diode in which the device owns its current as an unknown and
contributes one row: `v_a − v_b − R_on·i = V_f` while ON and
`i − g_off·(v_a − v_b) = 0` while OFF, with `R_on = 0` and `g_off = 0`
both legal. A branch-form diode SHALL report the current unknown as
its ON crossing signal and the gap `V_f + R_on·i − v` as its OFF
crossing signal, and its mode at an event SHALL come from the
complementarity mode selection.

#### Scenario: Ideal closed switch
- **GIVEN** 10 V, an ideal switch (`R_on = 0`), 1 Ω and 1 mH in series,
  switch closed at t = 0
- **WHEN** it is simulated to 10 ms
- **THEN** the current is 9.99954600 A within the engine tolerance (the
  conductance form with `g_on = 1e3` gives 9.98955 A)

#### Scenario: Equivalence at finite values
- **GIVEN** a buck, a bridge rectifier and a three-phase inverter built
  in branch form with `R_on = 1e-3`, `g_off = 1e-9`
- **WHEN** each is compared with the conductance form
  (`g_on = 1e3`, `g_off = 1e-9`)
- **THEN** the states agree within 1e-10

#### Scenario: Ideal diode
- **GIVEN** 10 A forced through a branch-form diode with `V_f = 0.7`,
  `R_on = 0`
- **WHEN** the circuit is solved
- **THEN** `v_D = 0.7 V`, the conduction power is 7.0 W and
  `res.i` of the diode is the solved unknown, 10 A

#### Scenario: DCM with ideal elements
- **GIVEN** the buck of `numerics/p2_buck.py` (R = 50 Ω, D = 0.3) with
  an ideal switch and an ideal diode
- **WHEN** it reaches its periodic orbit on `trbdf2`
- **THEN** `v_C` at the rising edge is 8.9963970815 V within 1e-6 and
  the inductor current in the idle interval is exactly 0 (6.0e-9 A with
  the resistor pair)

### Requirement: Ideal Switch As The Default Switch Model

The builder SHALL, after the default flip, build the branch form for
`add_switch`, `add_diode`, the topology helpers and YAML switches and
diodes, ideal
(`R_on = 0`, `g_off = 0`) unless `R_on` or `g_off` is given, and a
call that gives `g_on` and `g_off` SHALL build the branch form with
`R_on = 1/g_on`. Before the flip the conductance form SHALL remain the
default and SHALL be bit-identical to the device before this change.
`form='conductance'` SHALL remain accepted for one release after the
flip with a deprecation warning.

#### Scenario: Default is unchanged before the flip
- **WHEN** any existing circuit is built without `form='branch'` before
  the flip
- **THEN** its states are bit-identical to the build before this change

#### Scenario: Resistor pair stated explicitly after the flip
- **GIVEN** a buck built after the flip with
  `add_switch("S", "vin", "sw", 1e3, 1e-9)`
- **WHEN** it is compared with the same buck built before the flip
- **THEN** the states agree within 1e-10

### Requirement: Switch Parameter Validation

The builder and the YAML loader SHALL refuse, with an error naming the
device and the field, a conductance-form switch or diode whose `g_on`
is not finite and positive or whose `g_off` is not finite and
non-negative, and a branch-form device whose `R_on` or `g_off` is
negative or not finite.

#### Scenario: Infinite on-conductance
- **WHEN** `add_switch("S1", "a", "b", float("inf"), 1e-9)` is called
- **THEN** it raises an error naming `S1` and `g_on` and pointing to
  the branch form (a singular-matrix abort at run time before this
  change)

#### Scenario: Negative leak
- **WHEN** a YAML switch declares `g_off: -1e-9`
- **THEN** loading fails with an error naming the switch and `g_off`

### Requirement: Conservative Re-Initialisation at Ideal Commutations

The kernel SHALL continue, after an event whose new mode makes inductor
currents or capacitor voltages dependent, from the state that conserves
flux linkage on every inductor cut-set and charge on every capacitor
loop, through one re-initialisation service shared by every strategy:
the exact propagator SHALL apply the projection explicitly and continue
on the reduced state of the mode, and the implicit strategies SHALL
realise it with the restart rule of three consecutive Alexander
SDIRK(2,2) steps and SHALL NOT take the first step after such an event
with the trapezoidal rule. The service SHALL record for the event the
devices, the kind of constraint, the transferred flux or charge and the
energy removed, SHALL mark the pulse sample in the result, and SHALL
apply the `on_impulse` policy (`'warn'` by default, `'error'`,
`'allow'`). A consistent commutation SHALL produce no record.

#### Scenario: Interrupted inductor
- **GIVEN** the RL circuit above with `g_off = 0`, the switch opened at
  5 ms with no other path, on the fixed engine at `h = τ/20`
- **WHEN** it is simulated with `on_impulse='warn'`
- **THEN** the inductor current is 0 from the first step after the
  event and the node voltage is below 1e-9 V in magnitude from the
  second step after the event (±397.3 V for the rest of the run with
  the trapezoidal rule alone)
- **AND** the record reports `½·L·i_L(t⁻)²` of the run within 1e-9
  relative (49.3120 mJ at this step; 49.3285 mJ for the exact
  `i_L(t⁻)`), and one warning names the switch

#### Scenario: Capacitor charge sharing
- **GIVEN** 1 µF at 10 V and 3 µF at 2 V joined by an ideal switch
- **WHEN** the switch closes
- **THEN** both capacitors are at 4.000000 V within 1e-12 after the
  event, the record reports 6.0 µC and 24.0 µJ within 1e-9 relative,
  and the switch current is below 1e-9 A from the second step after the
  event

#### Scenario: Policy error
- **GIVEN** the interrupted-inductor circuit with `on_impulse='error'`
- **WHEN** the switch opens
- **THEN** the run raises an error naming the switch, the inductor and
  the energy

#### Scenario: Consistent commutation
- **GIVEN** the ideal DCM buck
- **WHEN** the diode turns off at zero current
- **THEN** no impulse record is produced

#### Scenario: Second order kept through events
- **GIVEN** the ideal DCM buck on the fixed engine at `T/200` and
  `T/400`, started on its orbit, 300 periods
- **WHEN** `v_C` is compared with 8.9963970815477 V
- **THEN** the errors are at most 2.6e-7 V and 6.6e-8 V (prototype
  1.709e-7 V and 4.345e-8 V; the backward-Euler pair gives 6.8e-5 V and
  1.7e-5 V)

### Requirement: Event Switching Energy From Device Tables

The loss accounting SHALL book, at every commutation of a switch or
diode that carries switching-energy tables, the energy
`E_on(i, v, T_j)`, `E_off(i, v, T_j)` or `E_rr(i, v, T_j)` interpolated
at the commutated current and the blocked voltage of the event, SHALL
compute conduction loss as `R_on·i² + V_f·i` time-weighted over the
solved current, and SHALL report impulse energies of the
re-initialisation service in a separate category.

#### Scenario: Turn-on energy from a table
- **GIVEN** a buck whose switch has a turn-on table
  `E_on = 10 µJ·(i/1 A)·(v/24 V)`, commutating 0.5 A against 24 V
- **WHEN** one period is simulated
- **THEN** the loss record books 5.0 µJ for the turn-on within 1e-12
  relative, at the event time
