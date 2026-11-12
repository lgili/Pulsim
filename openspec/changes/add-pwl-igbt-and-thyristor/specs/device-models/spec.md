## ADDED Requirements

### Requirement: PWL IGBT with Knee and Reverse Blocking

The PWL IGBT SHALL accept a collector-emitter knee `V_ce0` (default 0)
and a `reverse_blocking` flag. With reverse blocking and the gate on it
SHALL conduct `i = (v − V_ce0)/R_on` (`v = v_collector − v_emitter`)
while that current is positive, and `i = v/R_off` otherwise; with the
gate off it SHALL conduct `i = v/R_off` whatever the bias. It SHALL turn
on when the gate is on and `v ≥ V_on`, where `V_on = V_th` if
`V_ce0 = 0` and `max(V_th, V_ce0)` otherwise, and SHALL turn off when
the gate goes off or the on-state current reaches zero. Its conduction
power SHALL be `V_ce0·i + R_on·i²`. `reverse_blocking=False` with
`V_ce0 = 0` SHALL be bit-identical to the IGBT before this requirement.
A `V_ce0` that is negative, not finite, or positive without reverse
blocking SHALL be refused with an error naming the device. The device
SHALL be usable on `pwl`, `trbdf2` and `dsed`, in the DC operating
point, and through the YAML `igbt` keys `V_ce0` and `reverse_blocking`.

#### Scenario: Forward conduction shows the knee
- **GIVEN** 10 A forced collector to emitter through an IGBT with
  `V_ce0 = 0.9 V`, `R_on = 10 mΩ`, gate on
- **WHEN** the circuit is solved (DC operating point, `pwl` or `trbdf2`)
- **THEN** `v_CE = 1.000 V` and the conduction power is 10.0 W

#### Scenario: Reverse voltage is blocked
- **GIVEN** the same device, which conducted 10 A forward in the control
  case
- **WHEN** −100 V is applied collector to emitter with the gate on
- **THEN** the current is `−1e-7 A` (`R_off = 1e9 Ω`), not `−1e4 A`

#### Scenario: Gate off blocks forward voltage
- **WHEN** +100 V is applied with the gate off, on `pwl`, `trbdf2` and
  `dsed`
- **THEN** `|i| ≤ 1e-7 A` on each engine

#### Scenario: Legacy device is one flag away
- **WHEN** a three-phase inverter is built with
  `add_igbt(..., reverse_blocking=False)` and `V_ce0 = 0`
- **THEN** its states, switch-state record and loss summary are
  bit-identical to the same circuit before this requirement

#### Scenario: Knee without reverse blocking refused
- **WHEN** `add_igbt("Q1", ..., V_ce0=0.9, reverse_blocking=False)` is
  called
- **THEN** it raises an error naming `Q1`, `V_ce0` and
  `reverse_blocking`

### Requirement: Kernel Thyristor Latch

The kernel SHALL provide a thyristor whose latch is a device state. It
SHALL turn on when its gate bit is on and `v ≥ V_on` (`v = v_anode −
v_cathode`), conduct `i = (v − V_f)/R_on` while on, stay on while the
gate is on and `i > 0`, and, once the gate is off, stay on while
`i > I_H`. It SHALL NOT conduct reverse current beyond `v/R_off`. The
turn-off SHALL be located by the engine's event detection on the
current, not by comparing a sampled current with a fixed epsilon. `V_f`
and `I_H` SHALL be finite and non-negative, else refused by name. The
thyristor SHALL be usable on `pwl`, `trbdf2` and `dsed`, and through
the YAML type `thyristor`.

#### Scenario: Phase-controlled half-wave rectifier off the grid
- **GIVEN** a 100 V peak, 60 Hz source, a thyristor and a 10 Ω load,
  fixed step 10 µs (the current zero is not a grid point), gate pulses
  1 ms wide at α = 30°
- **WHEN** the run reaches the third cycle
- **THEN** the mean load voltage is 29.699 V within 0.5 %
- **AND** the load voltage never goes below −1e-6 V

#### Scenario: Latch outlives the gate pulse
- **GIVEN** a thyristor feeding an RL load, fired by a 100 µs gate pulse
- **WHEN** the gate goes off while the current is rising
- **THEN** the device keeps conducting until the current returns to
  zero

#### Scenario: Holding current
- **GIVEN** `I_H = 0.05 A` and a current decaying through 0.05 A
- **WHEN** the gate is off
- **THEN** the device is OFF in the first accepted state with
  `i ≤ 0.05 A`
- **AND** with the gate held on it keeps conducting down to `i = 0`

### Requirement: Line-Commutated Bridge Accuracy

A six-pulse thyristor bridge SHALL reproduce the ideal-converter mean
output voltage `(3√2/π)·V_LL·cos α − (3/π)·ω·L_s·I_d` when built from
kernel thyristors and fired by the kernel's phase-control schedule, on
every engine that accepts the circuit.

#### Scenario: Firing-angle law
- **GIVEN** 400 V line-to-line at 60 Hz, a 20 A constant-current load,
  no source inductance, gate edges on step boundaries
- **WHEN** the bridge runs at α = 0°, 30° and 60°
- **THEN** the mean output is 540.2 V, 467.8 V and 270.1 V, each within
  0.5 %

#### Scenario: Commutation overlap
- **GIVEN** the same bridge with 1 mH source inductance per phase at
  α = 30°
- **WHEN** it runs
- **THEN** the mean output is 460.6 V within 0.5 %

#### Scenario: Engines agree
- **WHEN** the α = 30° case runs on `pwl` and on `trbdf2`
- **THEN** the mean outputs differ by less than 0.5 %

### Requirement: GTO and TRIAC Devices

The kernel SHALL provide a GTO, which conducts with a forward drop while
its gate bit is on and it is forward biased and turns off when the gate
bit goes off, and a TRIAC, built from two anti-parallel latching
branches. Both SHALL be available through the builder, Python and the
YAML types `gto` and `triac`.

#### Scenario: GTO forced turn-off
- **GIVEN** a GTO conducting 10 A into a load that has a freewheel diode
- **WHEN** its gate bit goes off
- **THEN** the GTO current falls to `v/R_off` and the diode takes the
  load current

#### Scenario: TRIAC phase control
- **GIVEN** a 100 V peak source, a TRIAC and a resistive load, both
  gates fired at α = 30° in their half-cycles
- **WHEN** the run reaches steady state
- **THEN** the RMS load voltage is 69.68 V within 0.5 %

### Requirement: Gated PWL Devices in the Loss Summary

`res.i`, `device_loss_summary` and `device_thermal_summary` SHALL use
the gated devices' law: current and power from the recorded conduction
state, with the power `V_f·i + R_on·i²` while conducting.

#### Scenario: IGBT conduction loss includes the knee
- **GIVEN** an IGBT with `V_ce0 = 0.9 V`, `R_on = 10 mΩ` carrying 10 A
  DC
- **WHEN** `device_loss_summary` is computed
- **THEN** its `P_avg` is 10.0 W, not 1.0 W
