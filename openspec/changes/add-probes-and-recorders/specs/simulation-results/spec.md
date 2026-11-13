## ADDED Requirements

### Requirement: Probes Computed In The Kernel

A probe SHALL name one scalar quantity of the circuit: a node voltage,
a voltage difference, a device pin current, a device power, a device
output channel, a mode bit or a raw unknown. Probes SHALL be declarable
on the circuit (builder and YAML netlist) and on the run, SHALL be
resolved by name before the run starts, and SHALL be evaluated by the
kernel from the committed step. A device current SHALL be the current
of the device law the solve used; it MUST NOT be obtained by
differentiating stored samples or by a second implementation of the
law.

#### Scenario: Current of every device kind
- **GIVEN** each built-in device kind in series with a 1 mΩ resistor
- **WHEN** `i(device)` is recorded on the fixed and on the
  variable-step engine
- **THEN** it equals the resistor current within 1e-6 relative at every
  sample, and no kind raises `NotImplementedError`

#### Scenario: Recorded equals recomputed
- **GIVEN** a probe that is a function of the stored state and mode
  record alone
- **WHEN** it is recorded during the run and recomputed from a
  full-state record afterwards
- **THEN** the two columns are bit-identical

#### Scenario: Unknown probe refused before the run
- **WHEN** a probe names a device or node that does not exist
- **THEN** a `NetlistError` naming the probe is raised before any step
  is taken

#### Scenario: Unrecorded quantity is a typed refusal
- **GIVEN** a run that recorded only `v(out)`
- **WHEN** the current of a charge-based capacitor is requested from
  the result
- **THEN** an `UnsupportedError` names the probe to declare, and no
  array of zeros is returned

#### Scenario: Probes travel with the circuit
- **WHEN** a YAML netlist that declares probes is loaded and run, and
  the same probes are declared through the builder
- **THEN** both runs record the same columns under the same names

### Requirement: Recorders Select What And When

The run SHALL record through a recorder configured by what to store
(every unknown, the default, or a selection of probes) and when (every
step, every n-th step, a list of times, or a period). The default
SHALL store what is stored today. A selection SHALL store only the
selected columns. Recording SHALL be available on every engine; an
option an engine cannot honour SHALL be refused by name.

#### Scenario: Default record is unchanged
- **WHEN** a run uses the default record options on the fixed engine
- **THEN** `times`, `states` and the mode record are bit-identical to
  the previous build

#### Scenario: Probe selection reduces storage
- **GIVEN** a 12-node inverter and three selected probes
- **WHEN** it runs for 10⁶ steps
- **THEN** the result table holds three reals per row

#### Scenario: Decimation on the variable-step engine
- **WHEN** `store_every = 10` is requested with no `dt`
- **THEN** the run is accepted, every tenth accepted step and every
  event row are stored, and `avg` of any probe equals the value from
  the full record within 1e-9 relative

#### Scenario: Requested times are solved states
- **GIVEN** `t_eval` with 100 times on the variable-step engine
- **WHEN** the run completes
- **THEN** the result has a row at each requested time, each a state
  the stepper solved at that time (not interpolated), and the result's
  grid kind is `Requested`

#### Scenario: Interpolated rows are flagged
- **WHEN** a requested time is not a step boundary and landing is
  disabled
- **THEN** the row is linearly interpolated and carries the
  `interpolated` flag

### Requirement: One Result Type That Knows Its Grid

Every engine SHALL return the same result type. It SHALL be
constructible, SHALL state its grid kind (`Uniform`, `Accepted` or
`Requested`), and SHALL carry as typed fields the time axis, the
sample table with named columns, the mode record, the run statistics,
the diagnostics, the plan and the final snapshot. No engine SHALL
return a look-alike type, and no information SHALL exist only as an
attribute attached after the run.

#### Scenario: Same type on every engine
- **WHEN** a circuit runs on the fixed, the variable-step and the
  event-driven engine
- **THEN** each result is a `SimulationResult`, and `grid` is
  `Uniform`, `Accepted` and `Accepted` respectively

#### Scenario: Statistics are typed and engine-independent
- **WHEN** `result.stats` is read on any engine
- **THEN** it has the same fields, with counters an engine does not
  use equal to zero

#### Scenario: Existing accessors keep working
- **WHEN** code written for 2.0 reads `res.times`, `res.states`,
  `res.v(name)`, `res.i(name)`, `res.switch_states` and
  `res.final_snapshot`
- **THEN** it obtains the same values as before for linear and PWL
  devices

### Requirement: Left And Right Limits At Events

A result SHALL represent a discontinuity by two rows at the same time:
the left limit under the mask that produced it, in the sample table,
and the right limit under the mask the next step starts from, in the
event rows. Every engine SHALL provide them, and decimation SHALL NOT
drop either.

#### Scenario: Diode average on every engine
- **GIVEN** a boost converter in discontinuous conduction
- **WHEN** the diode's average current is computed from the result
- **THEN** it is 0.0535 A within the engine tolerance on the fixed,
  the variable-step and the event-driven engine (a left-limit
  trapezoid gives 0.0234 A)

#### Scenario: Switched-branch current jumps between the two rows
- **WHEN** a switch turns off at a recorded instant
- **THEN** its current is `g_on·v` in the left-limit row and `g_off·v`
  in the right-limit row

### Requirement: Time-Weighted Measures On The Result

The result SHALL provide `avg`, `rms`, `integral`, `at` and `minmax`
for any probe over any time window. They SHALL integrate the
piecewise-linear signal between rows exactly, starting each interval
from the right limit where an event row exists, and SHALL therefore be
independent of how samples cluster. On a result whose grid is not
uniform, an unweighted sample mean taken through the result's arrays
SHALL produce a warning that names the time-weighted measure.

#### Scenario: Clustered grid gives the same average
- **GIVEN** the same waveform recorded on a uniform grid and on an
  accepted grid that clusters at events
- **WHEN** `res.avg` and `res.rms` are evaluated
- **THEN** the two results agree within the variable-step tolerance,
  while the unweighted sample means differ

#### Scenario: Engines agree on the turn-off regression
- **GIVEN** the diode turn-off circuit with a series inductor of the
  2026-09 regression
- **WHEN** the output average is taken with `res.avg` on six engine and
  core combinations
- **THEN** all agree within 0.3 %

#### Scenario: Unweighted mean warns
- **WHEN** `.mean()` is called on an array obtained from a result with
  an accepted grid
- **THEN** a warning naming `avg()` is emitted

### Requirement: Recording Does Not Allocate Per Step

Recording SHALL perform no heap allocation per step on a uniform grid,
SHALL evaluate probes through a precompiled flat plan with no lookup
by name, and SHALL NOT make the default record slower or larger than
today's.

#### Scenario: Zero allocations while recording probes
- **GIVEN** ten probes on the fixed engine
- **WHEN** 10 000 steps run after warm-up under the allocation counter
- **THEN** the number of heap allocations is zero

#### Scenario: Default cost is unchanged
- **WHEN** the audit's buck is timed with default record options
  before and after
- **THEN** ns/step is not above the previous value and the stored
  bytes per sample are equal
