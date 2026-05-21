# Design — `pulsim-v2-substep-bisection` (Layer 5 V2.2 — diag only)

## Why V0 is diagnostic-only

True sub-step bisection requires three pieces:
1. **Detection**: catch that a flip happened in (t_n, t_n+1). ✓
   easy — comparing pre-iter and post-iter masks.
2. **Estimation**: find t* within (t_n, t_n+1). ✓ easy — linear
   interpolation on the watched signal.
3. **State correction**: take a partial step from t_n to t*
   with the OLD mask, then a step from t* to t_n+1 with the
   NEW mask. ✗ **HARD** — the trap-companion cache has
   dt-specific factors. Partial steps at dt_partial < opts.dt
   need a separate cached factor per partial-dt value.

Piece (3) is the architectural work. V0 of this OpenSpec ships
(1)+(2) — the user gets accurate commutation timestamps for
diagnostics. The state vectors stay at the dt grid (Layer 5 V2.1
behaviour).

V1 of this OpenSpec (or a sibling `pulsim-v2-multi-dt-cache`
OpenSpec) will add piece (3).

## Linear interpolation

For a diode transitioning ON → OFF, the watched signal is
`i_diode`. Between `t_n` (where i was positive) and `t_n+1`
(where i has gone negative — that's why the event iteration
flipped the state), the linear-zero crossing is:

```
t* = t_n + dt · i_n / (i_n − i_n+1)
```

For OFF → ON: watch `v_diode − V_th`. Same formula:

```
t* = t_n + dt · (v_n − V_th) / ((v_n − V_th) − (v_n+1 − V_th))
   = t_n + dt · (v_n − V_th) / (v_n − v_n+1)
```

(With (v_n − V_th) negative — diode was reverse-biased —
and (v_n+1 − V_th) positive at the just-converged state.)

If the linear interpolation gives `t*` outside [t_n, t_n+1]
(due to numerical noise or non-linearity), we clamp to that
range. If both endpoints have the same sign (no actual sign
change), we report `t* = t_n+1` and flag the event as
"untimed" (the iteration found a flip, but the linear-zero
crossing isn't well-defined).

## What we need from DiodeEventState

Currently `DiodeEventState::update_from_state` returns a
single bool ("any diode flipped"). We need MORE: which
diodes flipped, and what their pre- and post-update
(v_diode, i_diode) values were.

Two options:
1. Extend the return type to a vector of structs.
2. Add a new method `report_last_update_events()` that
   returns the events from the most recent update call.

Option 2 is cleaner — keeps the API of `update_from_state`
unchanged.

```cpp
struct DiodeUpdateEvent {
    Index branch_id;
    Index switch_idx;
    bool new_state;
    Real v_diode_before;
    Real v_diode_after;
    Real i_diode_before;
    Real i_diode_after;
};

std::span<const DiodeUpdateEvent> last_update_events() const;
```

The events are recorded internally during `update_from_state`
and overwritten on the next call.

## Run_transient changes

```
prev_x = x   // before the event-iter loop
do (event-iter):
    ...
    cache.solve / Newton
    flipped = diodes.update_from_state(x)
while ...

// AFTER the event-iter converges:
if (diodes.last_update_events().size() > 0) {
    for each event:
        // Compute pre/post watched signal.
        // Linearly interpolate t*.
        // Push to result.commutation_events.
}
```

Wait — the "pre" state is `prev_x` (state at t_n), but
`DiodeEventState`'s internal events array uses (v, i) at the
moment of the LAST `update_from_state(x)` call. The "before"
in that record is the state at the start of the just-completed
iter, NOT the state at t_n.

Refinement: at the start of each step, snapshot `prev_x = x`
BEFORE the event-iter loop. After the loop, for each event in
`last_update_events`, compute the watched signal at `prev_x`
(at t_n) and at the post-loop `x` (at t_n+1). Use those for
the interpolation.

This is cleaner — the DiodeEventState doesn't need to track
per-event "before" / "after" values; the caller already has
them.

```cpp
// Simpler: the event reports just (branch_id, new_state) and
// the calling code uses (prev_x, x) for interpolation.
struct DiodeUpdateEvent {
    Index branch_id;
    Index switch_idx;
    bool new_state;
};
```

## Tests

1. Half-wave rectifier (existing Layer 5 V2 test): for a
   60 Hz, V_amp=10, V_th=0 diode, commutations should happen
   at the zero-crossings of V_sine (every T/2 = 8.33 ms). The
   interpolated `t*` values should match these zero-crossing
   times within ~1 dt.

2. Boost converter (existing Layer 5 V2.1 test): each PWM
   cycle has 1-2 diode commutations. The interpolated times
   should align with the PWM transitions (within ~dt).

## What V1 will add

`pulsim-v2-multi-dt-cache`:
- Lazy cache that builds factors for any requested dt.
- `run_transient` uses bisected t* to take partial steps with
  the matching dt-cache.
- State vectors at the actual commutation times appear in
  `result.times` / `result.states` (not just at the dt grid).
