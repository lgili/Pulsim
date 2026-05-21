# Layer 5 V3 — Substep state correction

V2.2 added sub-step commutation timing diagnostics: the
estimated event time `t_est` from linear interpolation of
the watched diode signal (`v_diode − V_th`) between
`t_prev` and `t`. But the state vector `x` itself stayed at
the dt grid — V2.2 only reported `t_est`.

V3 wires `t_est` into actual state correction. When
`SimulationOptions::enable_substep_state_correction = true`
AND a commutation event is detected in step `k`, the
single-shot solve is replaced by two sub-steps via Layer 4
V7's `solve_at`:

- `dt₁ = t_est − t_prev`: pre-event mask.
- `dt₂ = dt − dt₁`: post-event mask (V2.2's `new_state`
  applied directly to the diode bits).

The result is an `x(t)` that reflects the diode being in
its pre-event state for `[t_prev, t_est]` and its
post-event state for `[t_est, t]` — instead of a single
state for the entire interval.

## API

```cpp
#include "pulsim/v2/solver/options.hpp"

SimulationOptions opts;
opts.enable_substep_state_correction = true;   // opt-in
// All other options unchanged.
```

When the flag is `false` (default), `run_transient`
behaves bit-identically to V2.2 (timestamp diagnostics
only).

## Mechanics

```
for k in 1..n_steps:
    x_prev    = x
    snap_hist = history.snapshot()
    snap_bits = diodes.snapshot_on_bits()
    mask_pre  = combine_masks(switch_fn(t),
                                diodes.current_diode_mask(), ...)

    # Normal V2.1 event iteration → x, history updated.
    ...

    # V2.2 event detection (sign change of v_diode − V_th).
    step_events = detect_events(x_prev, x)

    # V3 correction.
    if enable_substep_state_correction AND step_events:
        t_est = step_events.first.t_estimated
        dt1 = t_est - t_prev
        dt2 = dt - dt1
        if dt1 > 0.01·dt AND dt2 > 0.01·dt:
            x = x_prev
            history.restore(snap_hist)
            diodes.restore_on_bits(snap_bits)

            # sub-step 1
            cache.solve_at(mask_pre, dt1,
                           b_extra_user(t_est) +
                           history.compute_b_extra(dt1),
                           x)
            history.update_from_state(x, dt1)

            # apply commutation from V2.2 events
            new_bits = snap_bits
            for ev in step_events: new_bits[ev.diode_idx] = ev.new_state
            diodes.restore_on_bits(new_bits)

            # sub-step 2
            mask_post = combine_masks(switch_fn(t),
                                       diodes.current_diode_mask(), ...)
            cache.solve_at(mask_post, dt2,
                           b_extra_user(t) +
                           history.compute_b_extra(dt2),
                           x)
            history.update_from_state(x, dt2)
```

### Why apply commutation via `step_events.new_state` (not `update_from_state`)?

After sub-step 1, calling `diodes.update_from_state(x)`
would re-decide each diode based on `(v_diode, i_diode)`
at the sub-step-1 end. For an ideal switch with `V_th = 0`,
both signals are at the threshold there, and the
`SwitchedDiode` decision logic keeps the pre-event state
(it lacks numerical hysteresis to break the tie). The
substep correction then uses the WRONG mask for sub-step
2.

By applying `step_events.new_state` directly, we honour
V2.2's already-correct event detection (the sign change
of `v_diode − V_th` between `x_prev` and `x` post-
iteration), bypassing the boundary-decision ambiguity.

### Why skip events near step boundaries?

The trap-companion `g_eq = 2·C / dt` blows up as `dt → 0`.
If `t_est` lands within 1 % of either `t_prev` or `t`, the
shorter sub-step's matrix becomes ill-conditioned and the
solve produces garbage. The 1 % threshold (V0's safety
margin) catches these without false rejections of
real-mid-step events.

## Snapshot/restore helpers

V3 added small methods to `HistoryState` and
`DiodeEventState` to support roll-back:

```cpp
// HistoryState
std::vector<HistoryEntry> snapshot() const;
void restore(const std::vector<HistoryEntry>& snap);

// DiodeEventState
std::vector<bool> snapshot_on_bits() const;
void restore_on_bits(const std::vector<bool>& bits);   // throws on size mismatch
```

Round-trip invariant: `restore(snapshot())` is a no-op.

## Honest scope

V3 ships substep correction as the OPT-IN mechanism. The
empirical observations:

- **Ideal-switch diodes (V_th = 0)**: V2.2's `t_est` is
  biased toward step boundaries because `v_diode` is
  clamped to ≈0 during conduction. Sub-step durations
  collapse, and the correction is skipped (correctly —
  there's no accuracy to gain at the boundary). Output
  with the flag ON is essentially identical to OFF.
- **Smooth-blend diodes (more accurate t_est)**: these
  flow through `solve_with_newton_b_extra`, not
  `cache.solve`. V3 doesn't wire substep correction into
  the Newton path (V1 work). The flag has no effect on
  Newton-path diode events.
- **Mid-step events in mixed circuits**: when V2.2's
  `t_est` lands well inside a step (between 1 % and 99 %
  of `dt`), V3 splits the step. The result differs from
  the single-shot by a bounded amount. Long-time-averaged
  behaviour matches within 10 %.

The integration test verifies the mechanics (output stays
finite + bounded, total dt preserved, mean cap voltage
matches within 10 %), NOT a strict "always more accurate"
claim. The latter requires a t_est estimator that lands
mid-step for ideal-switch circuits — future work.

## What V3 deliberately does NOT do

- **Multi-event substepping**: V0 corrects only the first
  detected event per step. Multiple events in one step
  trigger correction at the earliest; the rest carry over.
- **Newton-path substep correction**: the smooth-blend
  diode path doesn't trigger V2.2 events (it uses the
  nonlinear Newton solve, not `cache.solve`). Wiring V3
  into the Newton path is V1.
- **Higher-order t_est interpolation**: V2.2 uses linear
  interp. Cubic or polynomial schemes might land closer
  to the true commutation time, but that's V2.2's
  responsibility, not V3's.
- **Adaptive dt around events**: V3 keeps the user's
  `opts.dt` constant. Sub-steps sum to `opts.dt`.

## Files

- MODIFIED `core/include/pulsim/v2/solver/options.hpp`
- MODIFIED `core/include/pulsim/v2/solver/run_transient.hpp`
- MODIFIED `core/include/pulsim/v2/pwl/history_state.hpp`
- MODIFIED `core/include/pulsim/v2/pwl/diode_event_state.hpp`
- NEW `core/tests/v2/layer5_v3/test_main.cpp`
- NEW `core/tests/v2/layer5_v3/test_snapshot_restore.cpp`
- NEW `core/tests/v2/layer5_v3/test_substep_correction.cpp`
- MODIFIED `core/CMakeLists.txt` (V3 test target added)
- NEW `openspec/changes/pulsim-v2-substep-state-correction/`
