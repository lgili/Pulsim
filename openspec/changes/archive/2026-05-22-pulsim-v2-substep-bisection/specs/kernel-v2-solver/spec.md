## ADDED Requirements

### Requirement: DiodeEventState — per-update event reporting

`DiodeEventState::last_update_events()` SHALL return a
`std::span<const DiodeUpdateEvent>` listing every diode that
flipped during the most recent `update_from_state` call. The
events MUST include the branch id, the switch-mask index, and
the new ON/OFF state.

`update_from_state` MUST clear the events vector and re-populate
it on each call.

#### Scenario: No flips → empty event list

- **GIVEN** a `DiodeEventState` and an `x` that doesn't flip
  any diodes
- **WHEN** the user calls `update_from_state(x)`
- **THEN** `last_update_events()` SHALL return an empty span.

#### Scenario: One flip → one event

- **GIVEN** a `DiodeEventState` with one diode initially OFF
- **WHEN** the user calls `update_from_state` with an `x`
  whose v_diode > V_th (forcing OFF → ON)
- **THEN** `last_update_events()` SHALL return a span of size
  1, with that diode's `branch_id`, the right `switch_idx`,
  and `new_state == true`.

### Requirement: SimulationResult.commutation_events

`SimulationResult` SHALL include a vector
`std::vector<CommutationEvent> commutation_events` populated by
`run_transient` with estimated commutation times. Each event
records:

```cpp
struct CommutationEvent {
    Real t_estimated;   // linear-interp zero crossing
    Index branch_id;
    bool new_state;
};
```

The events MUST be ordered by `t_estimated` (a side-effect of
the time-marching loop pushing them in order).

#### Scenario: Diode-free circuit → empty commutation list

- **GIVEN** a circuit with no `SwitchedDiode` branches
- **WHEN** the user calls `run_transient`
- **THEN** `result.commutation_events` SHALL be empty.

### Requirement: run_transient — linear-interpolation timing

`run_transient` SHALL push a `CommutationEvent` to
`result.commutation_events` every time a diode flips during the
event-iteration loop. The `t_estimated` field MUST be computed
by linear interpolation of the watched signal between `t_n` and
`t_n+1`:

- OFF → ON watch signal: `v_diode − V_th`
- ON → OFF watch signal: `i_diode`

If the two endpoint values have the same sign (no actual zero
crossing in the interval), `t_estimated` MUST be clamped to
`t_n+1`.

The recorded state vectors (`result.states`) MUST remain at the
dt grid (sub-step state correction is deferred to V1).

#### Scenario: Half-wave rectifier reports zero-crossing events

- **GIVEN** the half-wave rectifier test from Layer 5 V2
  (V_sine at 60 Hz with `SwitchedDiode`)
- **WHEN** the user runs a 2-cycle transient
- **THEN** `result.commutation_events` SHALL contain
  approximately 4 events (2 per cycle: ON→OFF at the descending
  zero-crossing, OFF→ON at the ascending zero-crossing)
- **AND** each `t_estimated` SHALL match the analytical
  zero-crossing time (k · T / 2 for integer k) within `dt`.
