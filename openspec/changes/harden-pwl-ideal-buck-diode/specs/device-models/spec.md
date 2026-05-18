## ADDED Requirements

### Requirement: PWL Ideal diode admissibility recovers from cold-start OFF

The PWL admissibility check SHALL flag the OFF state as non-admissible
and trigger a re-solve with the ON-state committed before stamping the
linear companion model, whenever the simulator enters a transient step
with a switching device whose `pwl_state_` is OFF (the cold-start
default) but whose terminal voltage is forward-biased above a
threshold.

The current behaviour stamps the `g_off` blocking conductance against the
forward voltage, producing a numerically invalid current and a megavolt
device voltage on the resulting node solve.

#### Scenario: Cold-start diode under forward bias recovers without spike

- **GIVEN** a series circuit `Vsrc(50 V) → IdealDiode → Rload(0.5 Ω)`
  with the diode's `pwl_state_` initialised to OFF (cold start) and
  `SwitchingMode::Ideal`
- **WHEN** the simulator runs a single transient step
- **THEN** the PWL admissibility check flags the OFF state as
  non-admissible (forward voltage above threshold)
- **AND** the re-solve commits the ON-state before stamping
- **AND** the resulting `V_diode` is within `±5%` of the analytical
  ON-state value `V_F0 + R_d · I`
- **AND** `V_diode > −2 V` (no megavolt regression)

### Requirement: PWL Ideal freewheel diode commutes on inductor-driven OFF transition

The PWL engine SHALL force the downstream freewheel diode's
`pwl_state_` to ON before stamping, regardless of the diode's terminal
voltage observable at that instant, whenever the upstream switch in a
buck-style topology commits OFF within a transient step.

The current behaviour leaves the diode OFF for one full timestep when the
upstream switch's OFF transition is event-detected mid-step, producing the
closed-loop overshoot symptom.

#### Scenario: Closed-loop buck with PWL Ideal tracks setpoint within ±1 V

- **GIVEN** a closed-loop buck converter `Vin = 24 V`, `Vref = 12 V`,
  with PI duty control, `SwitchingMode::Ideal`, and a duty-step large
  enough to trigger the OFF transition mid-step
- **WHEN** the simulator runs the closed-loop transient
- **THEN** the steady-state output voltage `vout` settles within ±1 V
  of `Vref = 12 V`
- **AND** the duty cycle settles within `[0.30, 0.70]`
- **AND** no individual timestep produces `vout > 1.5 · Vref`

### Requirement: Auto-parasitics enforces PWL Ideal infeasibility downgrade

The auto-parasitics analyzer SHALL automatically downgrade the affected
device's `mode_` to `SwitchingMode::Behavioral` BEFORE the simulator
constructs its operator network, whenever the analyzer detects a topology
that is "PWL Ideal infeasible" (e.g. a buck-converter switch with high
stored inductive energy and no snubber, where the PWL commutation would
produce a stamp voltage exceeding `V_bus + V_overshoot_threshold`). The
analyzer SHALL NOT merely log a recommendation in this case.

A user-controllable flag `AutoParasiticsConfig::enable_auto_downgrade`
(default `true`) SHALL gate this behaviour for power users who want the
warning without the downgrade.

#### Scenario: Buck stress with 1200 % overshoot auto-downgrades to Behavioral

- **GIVEN** a buck-converter circuit where the auto-parasitics analyzer
  reports `[CRIT] D ← L … V_overshoot=2400V (1200%)` for the freewheel
  diode
- **WHEN** the simulator is constructed with the default
  `AutoParasiticsConfig` (`enable_auto_downgrade = true`)
- **THEN** the diode's `mode_` is set to `SwitchingMode::Behavioral`
  before the first transient step
- **AND** the resulting `Vsw` waveform stays within
  `[−V_bus, V_bus + 10%·V_bus]` for all steps (no megavolt regression)
- **AND** an INFO log explains "auto-downgraded D from Ideal to
  Behavioral (PWL infeasible — V_overshoot 1200%)"

#### Scenario: Power user disables auto-downgrade

- **GIVEN** the same circuit as above and the user sets
  `opts.auto_parasitics.enable_auto_downgrade = false`
- **WHEN** the simulator is constructed
- **THEN** the diode's `mode_` is left at its declared value (Auto →
  Ideal in v0.11)
- **AND** the CRIT warning still fires
- **AND** the user is responsible for the numerical result
