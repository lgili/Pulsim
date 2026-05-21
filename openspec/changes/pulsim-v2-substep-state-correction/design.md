# Design — `pulsim-v2-substep-state-correction` (Layer 5 V3)

## The wobble problem

In V2 + V2.1 + V2.2, the dynamic-path step at time `t = t_prev + dt`:

1. Compute history contribution at `dt`.
2. Iterate diode-state until consistent (V2.1 event iteration).
3. Solve via `cache.solve(mask, b_extra, x)` for the entire `dt`.
4. Record `x` at `t`.
5. (V2.2) Compute `t_est` of any zero-crossings via linear
   interpolation between `x_prev` and `x_new`. STORE as
   diagnostic. State `x` is NOT corrected.

When the commutation event happens in the middle of a step
(e.g. `t_est ≈ t_prev + 0.5·dt`), the single-shot solve uses
the diode state CONVERGED-TO by event iteration. That state
is "correct" for `t > t_est` but applies to the entire
interval `[t_prev, t]`. The state at `t` reflects a step
where the diode had its post-event behaviour over a
duration of `dt` — over-conducting (or under-conducting) by
roughly `(t_est − t_prev)` worth of trap-companion error.

For dt ≈ 100 µs at 60 Hz with V_sine = 10 V, the per-step
sweep of V_sine is ~0.38 V near zero-crossings. The
commutation wobble translates to an output error of up to
~0.4 V at coarse dt — visible to the eye in V_out plots.

## The fix: split the step at `t_est`

When V_2.2 detects a zero-crossing in step `k`:
- `t_est` = linear-interpolated event time
- `dt₁ = t_est − t_prev` (pre-event sub-step)
- `dt₂ = t − t_est` (post-event sub-step)

Replace the single solve with TWO sub-steps:

```
# Roll back to pre-step state.
x = x_prev
history.restore(history_snap)
diodes.restore(diodes_snap)

# Sub-step 1: pre-event mask, duration dt₁.
b_extra₁ = b_extra_user(t_est) + history.compute_b_extra(dt₁)
cache.solve_at(mask_pre, dt₁, b_extra₁, x)
history.update_from_state(x, dt₁)

# Apply commutation.
diodes.update_from_state(x)
mask_post = combine_masks(switch_fn(t_est), diodes.current_diode_mask(), ...)

# Sub-step 2: post-event mask, duration dt₂.
b_extra₂ = b_extra_user(t) + history.compute_b_extra(dt₂)
cache.solve_at(mask_post, dt₂, b_extra₂, x)
history.update_from_state(x, dt₂)

# x now reflects the correct end-of-step state.
```

The auxiliary segments at `dt₁` and `dt₂` are built on demand
by `solve_at` (V7's lazy-cache mechanism). For a steady-state
simulation, only a few distinct `(dt₁, dt₂)` pairs occur (the
event-time pattern is periodic with the line frequency), so
the auxiliary cache stays small.

## Required helpers

### `HistoryState` snapshot/restore

Save and restore the per-device `(v_prev, i_prev)` pairs:

```cpp
struct HistoryEntry { ... };

[[nodiscard]] std::vector<HistoryEntry> snapshot() const {
    return entries_;   // copy
}
void restore(const std::vector<HistoryEntry>& snap) {
    entries_ = snap;
}
```

### `DiodeEventState` snapshot/restore

Save and restore the per-diode `is_on` bit:

```cpp
[[nodiscard]] std::vector<bool> snapshot_on_bits() const {
    std::vector<bool> bits;
    bits.reserve(diodes_.size());
    for (const auto& d : diodes_) bits.push_back(d.is_on);
    return bits;
}
void restore_on_bits(const std::vector<bool>& bits) {
    if (bits.size() != diodes_.size()) {
        throw std::invalid_argument(
            "DiodeEventState::restore_on_bits: size mismatch");
    }
    for (Size i = 0; i < diodes_.size(); ++i) {
        diodes_[i].is_on = bits[i];
    }
}
```

## Algorithm in run_transient

Pseudo-code for the dynamic path with substep correction:

```
for k in 1..n_steps:
    t = t_start + k * dt
    t_prev = t_start + (k-1) * dt
    x_prev = x

    # SNAPSHOT for potential rollback.
    history_snap = history.snapshot()
    diodes_snap  = diodes.snapshot_on_bits()
    mask_pre     = combine_masks(...)   # mask before event iteration

    # Normal single-shot step (V2.1 event iteration).
    do {
        mask = combine_masks(switch_fn(t),
                              diodes.current_diode_mask(), ...)
        b_extra = b_extra_user(t) + history.compute_b_extra(dt)
        cache.solve(mask, b_extra, x)
        flipped = diodes.update_from_state(x)
    } while flipped

    history.update_from_state(x, dt)
    mask_post = combine_masks(...)   # mask after event iteration

    # V2.2: detect events between x_prev and x.
    events_in_step = detect_events(x_prev, x, t_prev, t)

    # V3: substep correction if requested AND there's an event.
    if opts.enable_substep_state_correction AND !events_in_step.empty():
        t_est = events_in_step[0].t_estimated
        dt1 = t_est - t_prev
        dt2 = dt - dt1

        # Roll back.
        x = x_prev
        history.restore(history_snap)
        diodes.restore_on_bits(diodes_snap)

        # Sub-step 1: pre-event.
        b_extra_1 = b_extra_user(t_est) +
                    history.compute_b_extra(dt1)
        cache.solve_at(mask_pre, dt1, b_extra_1, x)
        history.update_from_state(x, dt1)
        diodes.update_from_state(x)

        # Sub-step 2: post-event.
        b_extra_2 = b_extra_user(t) +
                    history.compute_b_extra(dt2)
        cache.solve_at(mask_post, dt2, b_extra_2, x)
        history.update_from_state(x, dt2)

    record(t, x)
    record events_in_step
```

## What V3 deliberately does NOT do

- **Multi-event sub-stepping**: if two zero-crossings happen
  within the same `dt`, V0 corrects only the first. The
  second carries over to the next step's detection logic.
- **Newton in sub-steps**: the sub-step solves use linear
  `cache.solve_at`. Nonlinear branches refresh via the
  device pool's switched-state approach (V2/V2.1 behaviour
  preserved). Sub-step Newton is V1.
- **Adaptive dt around events**: the user's overall `dt`
  remains constant. The sub-steps' `dt₁` + `dt₂` always
  equal `dt`. True adaptive time-stepping is a separate
  OpenSpec.

## Test plan

**Coarse-dt half-wave rectifier** — designed to make the
wobble visible:

- V_sine(amp=10V, f=60Hz) → ideal-switch diode → R(10Ω) → GND
- dt = 5e-4 (200 µs ≈ 80 samples/cycle, ~10× coarser than V2)
- 2 cycles

Verification:
- Without substep correction: at samples near zero-crossings,
  v_out deviates from expected by up to `0.4 V` (the
  wobble).
- With substep correction enabled: those samples are within
  `0.05 V` of the expected (effectively zero-error apart
  from interpolation residual).
- Mean output power: both match analytical within 5%; the
  diff between them is < 1% (the wobble averages out over
  a cycle).

The unit test checks both cases on the same simulation
parameters and compares per-sample errors.

## Files

- MODIFIED `core/include/pulsim/v2/solver/options.hpp`
- MODIFIED `core/include/pulsim/v2/solver/run_transient.hpp`
- MODIFIED `core/include/pulsim/v2/pwl/history_state.hpp`
- MODIFIED `core/include/pulsim/v2/pwl/diode_event_state.hpp`
- NEW `core/tests/v2/layer5_v3/test_substep_correction.cpp`
- NEW `docs/pulsim-v2/layer5-v3-substep-correction.md`
- MODIFIED `core/CMakeLists.txt`
