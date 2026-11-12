## ADDED Requirements

### Requirement: Unified Stepper With Integrator And Controller Strategies

The MNA transient simulation SHALL be performed by one stepper
parameterised by an integrator (`BackwardEuler`, `Trapezoidal`,
`TrBdf2`) and a step controller (`Fixed{dt}`, `Lte{rtol, atol}`). An
integrator SHALL supply only its stage table and, when it has one, a
local-error estimate; a controller SHALL supply only the step proposal
and the accept / retry decision. Event handling, restart after a
discontinuity, retry after a failed step and guards SHALL be policies
of the one loop, not separate loops.

#### Scenario: Engine names select strategies
- **WHEN** a circuit runs with `engine='pwl'`, with `engine='trbdf2'`
  and with the backward-Euler entry point
- **THEN** the three runs execute the same step loop with
  `Trapezoidal` + `Fixed`, `TrBdf2` + `Lte` and `BackwardEuler` +
  `Fixed` respectively

#### Scenario: A new integrator needs no loop
- **GIVEN** a test-only integrator that defines a stage table
- **WHEN** it is instantiated with either controller
- **THEN** it runs every device kind and every event policy without any
  edit to the step loop

### Requirement: Fixed-Step Trapezoidal Mode Reproduces The Fixed Engine

The stepper with `Trapezoidal` and `Fixed{dt}` SHALL reproduce the
fixed engine it replaces bit for bit before `engine='pwl'` selects it:
states, sample times (`t0 + k·dt` from an integer counter), the mode
record, commutation events, event-iteration counts, dt-retry and
inductor-guard records. The replaced driver SHALL remain selectable as
`engine='pwl-legacy'` for one release.

#### Scenario: Golden hashes are equal
- **WHEN** every circuit of the reference harness and every
  fixed-engine circuit of the test suite runs on the stepper
- **THEN** the checkpoint hash of each trajectory equals the golden
  hash recorded from the replaced driver

#### Scenario: Fixed-only features are reproduced one by one
- **WHEN** `start_from_dc_op`, `store_every`, `strict_event_iterations`,
  `max_dt_halvings`, the sub-step state correction, the inductor
  guards, observers, C blocks, `on_cache` and `mmc_arms` are each
  exercised by their existing tests
- **THEN** results, records and warnings are identical on the stepper
  and on `pwl-legacy`

#### Scenario: Legacy driver stays reachable
- **WHEN** `engine='pwl-legacy'` is requested during the compatibility
  release
- **THEN** the replaced driver runs and the result reports
  `engine_used == 'pwl-legacy'`

### Requirement: Variable-Step Mode Reproduces The TR-BDF2 Engine

The stepper with `TrBdf2` and `Lte` SHALL reproduce the variable-step
engine it replaces step for step before `engine='trbdf2'` selects it:
the sequence of accepted times, the states, the mode record and the
accept, reject, solve, gate-event and diode-event counts. The replaced
driver SHALL remain selectable as `engine='trbdf2-legacy'` for one
release.

#### Scenario: Step sequences are equal
- **WHEN** every harness circuit runs on the stepper and on
  `trbdf2-legacy`
- **THEN** the accepted time grids are equal element by element and
  the states are bit-identical

#### Scenario: Agreement is judged at equal times
- **WHEN** a stepper result is compared with a result on another grid
- **THEN** the comparison uses states at equal times or time-weighted
  integrals, never an unweighted mean over samples

### Requirement: Single Solve Point Carrying Its Step Context

Every solve of a run SHALL go through one function that receives the
mask and a complete description of the solve (formula, derivative
coefficient, factor key, solve time, elapsed time): stage solves, event
probes, zero-time settles, gate landings, retry sub-steps and restart
steps. No device stamp, history term or source value SHALL be taken
from state left by a previous solve.

#### Scenario: Probe solves beside stateful devices
- **GIVEN** a PWL diode beside a saturable inductor, an MNA PMSM and a
  charge-based capacitor
- **WHEN** the variable-step mode settles the diodes at `t = 0` and
  probes events
- **THEN** the run completes without a NaN, and every stateful device
  was stamped with the probe's own step

#### Scenario: No bypass
- **WHEN** the stepper sources are checked
- **THEN** the factor cache and the Newton solver are called from the
  single solve function only

### Requirement: Single-Factor TR-BDF2 And Cached Fixed Factor Preserved

The stepper SHALL keep the factorisation economy of the engines it
replaces. Both TR-BDF2 stages SHALL use the factor keyed by
`(mask, γh)`; the fixed trapezoidal mode SHALL use the primary cached
segment and SHALL perform no error-estimate work and no rollback copy
when no policy can reject the step; a backward-Euler step over `h`
SHALL reuse the trapezoidal factor of `2h`.

#### Scenario: Factorisation counters are equal
- **WHEN** the harness circuits run on the stepper and on the legacy
  drivers
- **THEN** solves, factorisations, cache hits and cache misses are
  equal for every circuit

#### Scenario: Restart half-steps reuse the nominal factor
- **WHEN** two backward-Euler half-steps of `h/2` are taken on the
  fixed grid
- **THEN** no factorisation is performed beyond the nominal one for
  `(mask, h)`

### Requirement: Elapsed-Time Consistency Of Every Step

The time a stage commits over SHALL equal the time it advances. The
stepper MUST NOT obtain one formula by assembling another at a
different step size in a way that changes the interval a
charge-integrating device commits over.

#### Scenario: Charge device commits over the real interval
- **GIVEN** a charge-based capacitor
- **WHEN** a restart step, a retry sub-step and a normal step are taken
- **THEN** after each, the committed charge equals `q(v)` at the
  accepted state and the committed interval equals the step's span

### Requirement: Features Are Engine-Independent Or Refused For A Stated Reason

Every transient feature SHALL either work with both step controllers
or be refused by `UnsupportedError` with a structural reason. A feature
MUST NOT be accepted and ignored, and a feature SHALL NOT be refused
merely because it was implemented in one loop.

#### Scenario: Formerly fixed-only features on the variable-step controller
- **WHEN** `start_from_dc_op`, `strict_event_iterations` and the
  inductor guards are requested with no `dt`
- **THEN** the run uses the variable-step controller and agrees with
  the fixed-step run at equal times within the variable-step tolerance

#### Scenario: Structural refusal states its reason
- **WHEN** `max_dt_halvings` or the sub-step state correction is
  requested on the variable-step controller
- **THEN** `UnsupportedError` is raised and its message states that the
  LTE controller retries with a smaller step, or that events are
  localised, respectively

#### Scenario: Conformance matrix has no silent cell
- **WHEN** the feature × engine conformance test runs after each port
- **THEN** every cell is either "agrees with the reference" or "refused
  by name"

### Requirement: Backward-Euler Fixed Mode Replaces The bdf1 Driver

The backward-Euler entry point SHALL be the stepper with
`BackwardEuler` and `Fixed{dt}` and SHALL support every device kind the
stepper supports. The separate `bdf1` driver SHALL be removed.

#### Scenario: Nonlinear capacitor is not deleted
- **GIVEN** an RC with a linear capacitor and a charge-based capacitor
- **WHEN** it runs through the backward-Euler entry point
- **THEN** the result agrees with the fixed trapezoidal engine within
  the first-order error bound, and differs from the circuit without the
  charge-based capacitor

#### Scenario: First order
- **WHEN** an RC decay is run at `dt`, `dt/2`, `dt/4`
- **THEN** the error against the analytic solution halves with `dt`

### Requirement: Legacy Transient Code Retired On A Schedule

The legacy code SHALL be removed in this order and no earlier: the
legacy drivers one release after the engine names were switched; the
history-less `run_transient` overload once its tests are moved; the
`integrators/` directory and the Python Runge–Kutta modules after one
deprecation release. Each removal SHALL be preceded by a
`DeprecationWarning` naming the replacement.

#### Scenario: Deprecated names warn before they disappear
- **WHEN** `run_transient_adaptive`, `DormandPrince5` or
  `run_transient_bdf1` is used during the deprecation release
- **THEN** a `DeprecationWarning` naming the replacement is emitted and
  the call still works

### Requirement: Snapshot Resume Across Strategies

A snapshot SHALL be resumable by any integrator and controller of the
stepper. Its physical part (`t`, `x`, device state, mode bits) SHALL be
strategy-independent; an optional controller block SHALL make a resume
on the same controller exact.

#### Scenario: A snapshot taken on engine A resumes on engine B
- **GIVEN** a snapshot taken at the end of a fixed-step run
- **WHEN** the variable-step mode resumes from it, and a variable-step
  snapshot is resumed by the fixed-step mode
- **THEN** the first resumed sample equals the snapshot state and the
  resumed trajectory agrees with the continuous run at equal times
  within the variable-step tolerance

#### Scenario: Exact variable-step resume
- **WHEN** a variable-step run is split at an accepted step and resumed
  from a snapshot carrying the controller block
- **THEN** the concatenated trajectory is bit-identical to the
  continuous run

### Requirement: Stepper Hot Loop Allocates Nothing

The stepper SHALL perform no heap allocation per accepted, rejected or
retried step, in either controller, and its per-step cost SHALL NOT
exceed that of the driver it replaces.

#### Scenario: Zero allocations
- **WHEN** the zero-allocation tests run on the stepper (plain steps,
  diode events, sub-step correction, rejected steps, Newton retries)
- **THEN** the allocation count per step is zero

#### Scenario: No slowdown
- **WHEN** the audit's buck, VSI and RLC-ladder cases are timed on the
  stepper and on the legacy drivers
- **THEN** ns/step on the stepper is not above the legacy value (best
  of five, same machine)
