## ADDED Requirements

### Requirement: Complementarity Mode Selection At Every Event

The kernel SHALL decide the conduction state of every PWL diode at an
event (a gate edge, a located diode crossing, a source breakpoint,
`t = 0`, a resume) and at the DC operating point by solving one linear
complementarity problem on the diode ports of the companion network
of an event step `h_e`, with the gate mask and the source values at
`t⁺` and the energy-storage states held at `t⁻`, and SHALL apply the
returned mode without re-solving and re-deciding in rounds. Every
engine (`pwl`, `trbdf2`, `dsed` and the exact propagator), the DC
operating point and `periodic_steady_state` SHALL use the same
selection service when `mode_selection='lcp'` is in effect.

#### Scenario: Diodes at rest with no threshold
- **GIVEN** a buck with every switch open, `V_th = 0`, `V_f = 0`, all
  states zero
- **WHEN** it is simulated on `dsed` with `mode_selection='lcp'`
- **THEN** the run completes, every diode is OFF and the states stay 0
  (today: "did not settle after 64 rounds at t = 0.000000")

#### Scenario: DCM without a band
- **GIVEN** the DCM buck (24 V, 100 µH, 47 µF, 50 Ω, D = 0.3,
  100 kHz) with `V_th = 0`, and the same with L scaled by
  `1 ± δ`, δ ∈ {1e-5, 1e-4, 1e-2}
- **WHEN** each is simulated for 4000 periods on `dsed`
- **THEN** every run completes with no chatter error (today: "D1 is
  chattering at t = 0.000996")

#### Scenario: Simultaneous commutation in a bridge
- **GIVEN** a single-phase diode bridge, 100 V 50 Hz, 1000 µF ‖ 100 Ω,
  ideal diodes with `V_f = 0.7`
- **WHEN** it is simulated for 10 cycles
- **THEN** each diode event is one selection, and every event time
  after the first turn-off is within 1e-12 s of the closed-form
  conduction interval 66.672808323° to 91.797645707°

#### Scenario: Same answer on every engine
- **GIVEN** the three-phase bridge with `L_s = 1 mH` and a 10 A DC load
- **WHEN** it is simulated on `pwl`, `trbdf2` and `dsed` with
  `mode_selection='lcp'`
- **THEN** the sequence of diode modes is identical on the three
  engines and the overlap angle is 15.479928° within 1e-3° on each

### Requirement: Mode Selection Termination And Infeasibility

The selection SHALL use Lemke's method with a lexicographic ratio test
and SHALL terminate in a finite number of pivots on every event. For a
circuit classified passive at build time a secondary ray SHALL be
reported as a typed error naming the diodes and the time (no
consistent mode exists). For a circuit not classified passive the
selection SHALL fall back to enumeration of the modes for at most 12
diode ports and SHALL otherwise raise
`pulsim.UnsupportedError` with `feature = 'non_passive_ports'`, the
device that makes the circuit non-passive as `device` and its device
kind as `kind`. There SHALL be no round limit, mask-cycle detector or
chatter guard on the selection path.

#### Scenario: No consistent mode
- **GIVEN** a DC current source of 1 A driven into two ideal diodes
  (`g_off = 0`) in anti-series with no other path
- **WHEN** the operating point is computed
- **THEN** a typed error names both diodes and states that no
  consistent conduction state exists

#### Scenario: Non-passive refusal
- **GIVEN** a circuit classified non-passive by a VCVS, whose 13-port
  problem at an event is replaced through the service's test entry
  point by `M = −I₁₃`, `q = −1` (no solution: Lemke ends at a secondary
  ray)
- **WHEN** the selection runs
- **THEN** `pulsim.UnsupportedError` is raised with `feature =
  'non_passive_ports'`, the VCVS as `device` and `kind = 'VCVS'`
- **AND** the same problem with 2 ports (`M = −I₂`, `q = −1`) is
  enumerated and raises the "no consistent mode" error

### Requirement: Mode Selection Tie Resolution

The selection SHALL treat a port whose current and gap are both within
the circuit-scale tolerances as a tie, SHALL re-solve the problem on
the ties alone at `100·h_e` and then `1e4·h_e` with the decided ports
held in their mode (releasing a held port whose complement the wider
solution violates), and SHALL set a port still tied to OFF unless its
current pins a node (current times its driving-point resistance above
the gap tolerance) and the mode with every remaining tie OFF fails the
exact consistency judge, in which case the pinned ports SHALL be ON.
Ties, widenings, released ports, pins and the tolerances used SHALL be
counted in `result.diagnostics`.

#### Scenario: Series pair turning off together
- **GIVEN** the single-phase bridge of the requirement above
- **WHEN** a conducting pair reaches zero current
- **THEN** both diodes turn OFF at the same event, the tie is counted,
  and no event at the same time follows

#### Scenario: Leak current holding a floating DC bus
- **GIVEN** the three-phase bridge with a 1 mF ‖ 20 Ω DC filter,
  `V_f = 0.7`, resistor-pair diodes (`g_off = 1e-9`), `V_th = 0`
- **WHEN** a conducting pair turns off and one diode is left carrying
  the ~49 nA leak that sets the DC bus potential
- **THEN** that diode is selected ON (a pin), no OFF diode is
  forward-biased after the event, and the shadow report over 0.06 s
  lists 0 disagreements with 18 pins

### Requirement: Default-Mode Mode Guard

Without `mode_selection`, on `dsed`, the kernel SHALL judge the diode
mode the settle selected at `t = 0` and at every event by propagating
it exactly from the held state, and SHALL decide an inconsistent mode
with the selection instead, recording each repair (time, diode,
violation) in `result.diagnostics['mode_guard']`. A selection refusal
at a repaired event SHALL raise `pulsim.UnsupportedError` naming the
diode. A run whose settled modes are all consistent SHALL be
bit-identical to the run without the guard.

#### Scenario: Boost without a band
- **GIVEN** the boost (12 V, 100 µH, 47 µF, 500 Ω, D = 0.5,
  100 kHz) with `V_th = 0`, 400 periods
- **WHEN** it is simulated on `dsed` with no `mode_selection`
- **THEN** `diagnostics['mode_guard']['repairs']` is positive and the
  output is within 2e-5 V of trbdf2 at rtol 1e-10 (the settle alone:
  1.01 V low)

#### Scenario: Consistent circuit unchanged
- **GIVEN** the DCM buck with `V_th = 0`, 400 periods
- **WHEN** it is simulated on `dsed` with the guard and without it
- **THEN** times and states are bit-identical and the guard reports 0
  repairs

### Requirement: Floating Islands As A Gauge

The kernel SHALL treat the potential of a sub-network connected to the
rest only through OFF diodes with `g_off = 0` as a gauge: it SHALL locate diode
crossings on gauge-free combinations of gaps, SHALL report the
island's potential at the centre of the interval its OFF diodes allow,
and SHALL list the island's nodes in
`result.diagnostics['gauge_nodes']`. A sub-network bounded only by
open gated switches SHALL remain a floating-node error.

#### Scenario: DC side of a rectifier between conduction intervals
- **GIVEN** the single-phase bridge with ideal diodes (`g_off = 0`)
- **WHEN** it is simulated on `pwl`, `trbdf2` and `dsed`
- **THEN** the reported potential of the positive DC node in a
  non-conducting interval agrees between the three engines within
  1e-9 V (today, with the leak: 0.000000 V on `trbdf2` against
  47.585852 V on `pwl` and 47.585559 V on `dsed`) and the node is
  listed as a gauge node

### Requirement: Mode Selection Shadow Mode

With `mode_selection='shadow'` the kernel SHALL keep today's diode
decision for the run, SHALL run the selection on the same held state
at every event, and SHALL report every disagreement in
`result.diagnostics['mode_shadow']` with the time, both modes, the
number of settle rounds and the pivots. The states of a shadow run
SHALL be bit-identical to a run without the option.

#### Scenario: Shadow on a banded circuit
- **GIVEN** the DCM buck with `V_th = 1e-2` on `dsed` (requires the
  hysteresis handling of the selection, since `V_th > V_f = 0`)
- **WHEN** it is simulated with and without `mode_selection='shadow'`
- **THEN** the states are bit-identical and the shadow report lists 0
  disagreements

### Requirement: Mode Selection Counters

Every run SHALL report `lcp_count`, `lcp_pivots_total`,
`max_pivots_per_event`, `ties`, `widenings` and `enumerations` in
`result.diagnostics` when the selection is active.

#### Scenario: Previous mode still consistent
- **GIVEN** the CCM boost (12 V, 100 µH, 47 µF, 50 Ω, D = 0.5) for
  20 000 periods
- **WHEN** it is simulated with `mode_selection='lcp'`
- **THEN** at least 99 % of the selections take zero pivots and
  `max_pivots_per_event` is at most 2
