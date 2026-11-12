## ADDED Requirements

### Requirement: Diode Commutation Signals With Forward Drop

Every engine SHALL decide a PWL diode's state through the single
`SwitchedDiode` law (`decide`, `crossing_signal`), so that no engine
lets an ON diode conduct backwards: the pwl event iteration, DC
operating point and seeding through `DiodeEventState::update_from_state`;
the pwl sub-step watcher on `v − V_on`; TR-BDF2 crossing localisation,
held and forbidden checks on `v − V_f` while ON and `v − V_on` while OFF;
the DSED turn-ON predicate on `v − V_on`, the turn-OFF predicate on
`v − V_f`, and the DSED settle through `decide`. When the DSED settle
cannot converge, its error SHALL suggest a hysteresis band
`V_th = V_f + 1e-3`.

#### Scenario: Resistive half-wave turns on and off at V_f
- **GIVEN** 10 V peak, 50 Hz into a diode (`g_on = 20`, `V_f = 0.7`) and
  10 Ω
- **WHEN** simulated on pwl, trbdf2 and auto
- **THEN** conduction starts near 223 µs and ends near 9.777 ms, the
  output averages 2.8268 V, and the diode current never falls below
  −1e-9 A

#### Scenario: DSED needs a band
- **GIVEN** a circuit whose diode turns on at zero current
- **WHEN** it runs on `engine='dsed'` with `V_th = V_f + 1e-3`
- **THEN** it matches pwl
- **AND** with `V_th ≤ V_f` it either settles or raises an error that
  names the band advice
