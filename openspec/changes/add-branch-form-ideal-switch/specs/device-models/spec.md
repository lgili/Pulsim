## ADDED Requirements

### Requirement: Branch-Form Switch and Diode

The device library SHALL offer a branch form of the switch and of the
PWL diode in which the device owns its current as an unknown and
contributes one row: `v_a − v_b − R_on·i = V_f` while ON and
`i − g_off·(v_a − v_b) = 0` while OFF, with `R_on = 0` and `g_off = 0`
both legal. The conductance form SHALL remain the default and SHALL be
bit-identical to the device before this change. In branch form a diode
SHALL stay ON while its current unknown is positive and SHALL turn ON
when `v ≥ V_on`.

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
- **WHEN** it reaches its periodic orbit
- **THEN** `v_C` at the rising edge is 8.9963970815 V within 1e-6 and
  the inductor current in the idle interval is exactly 0 (6.0e-9 A with
  the resistor pair)

#### Scenario: Default is unchanged
- **WHEN** any existing circuit is built without `form='branch'`
- **THEN** its states are bit-identical to the build before this change

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

The engine SHALL continue, after a commutation of branch-form devices
that makes inductor currents or capacitor voltages dependent, from the
state that conserves flux linkage on every inductor cut-set and charge
on every capacitor loop. It SHALL NOT take the first step after such
an event with the trapezoidal rule, and SHALL record for the event the
switch, the kind of constraint, the transferred flux or charge and the
energy removed. The `on_impulse` policy SHALL select between a
warning, an error and silence; a consistent commutation SHALL produce
no record.

#### Scenario: Interrupted inductor
- **GIVEN** the RL circuit above with `g_off = 0`, the switch opened at
  5 ms with 9.93262053 A flowing and no other path
- **WHEN** it is simulated with `on_impulse='warn'`
- **THEN** the inductor current is 0 from the first step after the
  event and the node voltage is below 1e-9 V in magnitude from the
  second step after the restart (±397 V for the rest of the run on the
  fixed engine and 7.0e10 V on TR-BDF2 before this change)
- **AND** the event record reports `L·i_L(t⁻)` and `½·L·i_L(t⁻)²` of
  the run within 1e-9 relative (9.93262053e-3 V·s and 49.3285 mJ for
  the exact `i_L(t⁻)`), and one warning names the switch

#### Scenario: Capacitor charge sharing
- **GIVEN** 1 µF at 10 V and 3 µF at 2 V joined by an ideal switch
- **WHEN** the switch closes
- **THEN** both capacitors are at 4.000000 V within 1e-12 after the
  event, the record reports 6.0 µC and 24.0 µJ within 1e-9 relative,
  and the switch current is below 1e-9 A after the restart

#### Scenario: Policy error
- **GIVEN** the interrupted-inductor circuit with `on_impulse='error'`
- **WHEN** the switch opens
- **THEN** the run raises an error naming the switch, the inductor and
  the energy

#### Scenario: Consistent commutation
- **GIVEN** the ideal DCM buck
- **WHEN** the diode turns off at zero current
- **THEN** no impulse record is produced
