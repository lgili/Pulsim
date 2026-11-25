## ADDED Requirements

### Requirement: Fixed-Step Schedule Evaluation Count

The fixed-step engine SHALL evaluate `switch_fn` exactly once for each
distinct time at which it solves during a step. Event iterations at the
same time SHALL reuse that gate mask and combine it with the current
diode mask. The result SHALL be identical to evaluating `switch_fn`
before every solve whenever `switch_fn` returns the same mask for the
same `t` within a step.

#### Scenario: One call per nominal step
- **GIVEN** a circuit with a PWL diode and a schedule that counts its
  calls, run without dt-retries or sub-step correction
- **WHEN** it is run for `N` steps and again for `2N` steps
- **THEN** the second run makes exactly `N` more calls than the first

#### Scenario: A retry sub-step reads the schedule at its own time
- **GIVEN** a step that is re-taken as two half-steps
- **WHEN** the step completes
- **THEN** the schedule was called once for each of the two sub-step
  times and not again for the event iterations inside them

#### Scenario: Result unchanged
- **WHEN** the reference golden circuits are run
- **THEN** states, the switch-state record and
  `event_iteration_count` are equal to the goldens (`np.array_equal`)

### Requirement: Fixed-Step Segment Lookup Memoisation

The fixed-step engine SHALL NOT search the mask table for a solve whose
mask equals the mask of the previous solve, provided the cache has not
been structurally modified in between. The cache SHALL expose a
generation counter that changes on every insert, eviction, parametric
refactor and clear; a memoised segment SHALL be discarded when the
generation differs. A memoised hit SHALL update the segment's recency
exactly as a searched hit does.

#### Scenario: Searches bounded by mask changes
- **GIVEN** a buck converter run for 10 000 steps over 100 PWM periods
- **WHEN** the run completes
- **THEN** `counters["segment_searches"]` is at most the number of mask
  changes plus one

#### Scenario: Eviction invalidates the memo
- **GIVEN** a lazy cache with a byte budget that holds one segment
- **WHEN** the mask alternates between two values
- **THEN** every solve uses a valid segment and the states equal those
  of a run with an unlimited budget

### Requirement: Fixed-Step Loop Has No Per-Step Scratch Initialisation

The fixed-step engine SHALL allocate and initialise its event-iteration
scratch once per run. Rollback snapshots SHALL copy only state that a
rollback restores, and SHALL NOT be taken when no rollback path is
enabled for the run.

#### Scenario: No rollback path, no snapshot
- **GIVEN** `max_dt_halvings = 0` and
  `enable_substep_state_correction = false`
- **WHEN** a run completes
- **THEN** `counters["rollback_snapshots"]` is 0 and the states equal
  the goldens

#### Scenario: Rollback still restores the step start
- **GIVEN** a circuit whose step is re-taken at a smaller `dt`
- **WHEN** the run completes
- **THEN** the states equal those produced before this change

### Requirement: Sparse Source And History Passes

The right-hand-side contribution of sine, PWM and pulse sources SHALL
be written only to the rows those sources own, from index lists built
once per run. Companion coefficients that depend only on `dt` and a
device parameter SHALL be computed once per `dt` value. The assembled
right-hand side SHALL be bitwise equal to the one produced by the dense
passes.

#### Scenario: Right-hand side unchanged
- **WHEN** the reference golden circuits with sine, PWM and pulse
  sources are run, including one that takes a dt-retry
- **THEN** the states match the goldens by value and by byte hash

### Requirement: Fixed-Step Loop Counters

A fixed-step run SHALL report `switch_fn_calls`, `segment_searches`
and `rollback_snapshots`. The counters SHALL be deterministic for a
given circuit and options.

#### Scenario: Counters are reproducible
- **WHEN** the same circuit is run twice with the same options
- **THEN** the three counters are equal between the runs
