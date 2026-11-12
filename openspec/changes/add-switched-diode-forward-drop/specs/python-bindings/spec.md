## ADDED Requirements

### Requirement: Switched Diode Forward Drop Parameter

`CircuitBuilder.add_diode` SHALL accept `V_f` (keyword, default 0.0),
and `components()` SHALL report it in the diode's `params`. `res.i`,
`device_loss_summary` and `device_thermal_summary` SHALL compute a PWL
diode's current as `g_on·(v − V_f)` while ON and `g_off·v` while OFF;
the loss summary SHALL report `V_f` and its `P_avg` SHALL include
`V_f·i`. Loss windows SHALL still add up to the whole run.
`add_bridge_rectifier` SHALL accept `V_f` (default 0.0) and pass it as
each diode's drop, with its `V_th` default unchanged.

#### Scenario: Loss includes the drop
- **GIVEN** 10 A forced through `g_on = 20 S`, `V_f = 0.7 V`
- **WHEN** `device_loss_summary` is evaluated
- **THEN** the diode's `P_avg` is 12.0 W and `res.i` reads 10 A

#### Scenario: Thermal and loss summaries agree
- **WHEN** both summaries run on a `V_f > 0` circuit
- **THEN** their conduction energies are equal

### Requirement: Three-Phase VSI Freewheel Diode Drop

`add_three_phase_vsi(V_f=0.7)` SHALL pass `V_f` to each freewheel diode
as its forward drop (BREAKING: it was the turn-on threshold), with a new
`V_th` keyword (default 0.0) as the threshold.

#### Scenario: VSI diode loss contains V_f·I_avg
- **GIVEN** the VSI with `R_d = 50 mΩ`, `V_f = 0.7`
- **WHEN** the freewheel diode carries an inductor current
- **THEN** its `P_avg` equals `V_f·i_avg + R_d·i_rms²` within 1e-3

#### Scenario: No sharing below V_f
- **GIVEN** a half bridge with dead time, `R_on·|i| < V_f`
- **WHEN** the diode's parallel switch is gated on
- **THEN** the diode carries less than 1e-6 A and conducts only during
  the dead time
