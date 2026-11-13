## ADDED Requirements

### Requirement: PWL Diode Complementarity Law

The PWL diode SHALL be modelled as an ideal complementarity element in
series with the forward drop `V_f` and the on-resistance `R_on`
(`R_on = 1/g_on` in conductance form), in parallel with the leak
`g_off`: `0 ≤ i_d ⟂ V_f + R_on·i_d − v ≥ 0`. A diode SHALL turn off at
`i_d = 0` and SHALL turn on at `v = V_f`, or at `v = V_th` when
`V_th > V_f` (hysteresis: an OFF diode is held OFF while its
open-circuit voltage is below `V_th`). A threshold `V_th < V_f` SHALL
have no effect and SHALL produce a deprecation warning during the
legacy period.

#### Scenario: Conductance form within the leak
- **GIVEN** a diode with `g_on = 1e3`, `g_off = 1e-9`, `V_f = 0.7`
  forward-biased at 1.7 V
- **WHEN** its current is evaluated under the complementarity law
- **THEN** it equals today's `g_on·(v − V_f)` = 1000 A plus
  `g_off·v` = 1.7e-9 A

#### Scenario: Hysteresis threshold kept
- **GIVEN** a half-wave rectifier: 100 V, 50 Hz sine, one diode with
  `V_f = 0`, `V_th = 0.5 V`, `R_on = 0`, `g_off = 0`, and 10 Ω
- **WHEN** it is simulated for one period
- **THEN** the diode turns on at `t = asin(0.005)/ω` (15.9155 µs) and
  off at `t = T/2` within the event tolerance
