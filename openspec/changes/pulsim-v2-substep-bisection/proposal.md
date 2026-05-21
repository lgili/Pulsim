## Why

Layer 5 V2.1 implemented event-iteration on the switch state
per step. This ELIMINATED the chatter that broke the boost
converter, but the time-resolution of each commutation is
still O(dt) — the diode state changes between `t_n` and
`t_{n+1}`, and we only know the change happened "somewhere in
that interval".

For most PE workloads this is fine (dt = T_sw/100 means < 1 %
timing error per commutation). But for **high-fidelity
analysis** — e.g., measuring switch losses, gate-drive timing
margins, EMC predictions — the user wants to know WHEN
commutations actually happen, not just that they did.

**Full sub-step bisection** (find t* exactly, take a partial
step from t_n to t*, restart from t* with the new mask) would
require multi-dt cache support — the trap companion's
`g_eq = 2C/dt` makes the matrix dt-specific, so a partial step
at dt_partial < opts.dt would need a separate cached factor.
That's a substantial architectural change.

This OpenSpec ships the **diagnostic V0**: report the
LINEAR-INTERPOLATED commutation time in the result. The
recorded state vectors are still at the dt-grid points (no
sub-step state correction), but the user can extract the
estimated commutation time of every event for downstream
analysis. Full sub-step state correction is deferred to a
follow-up `pulsim-v2-multi-dt-cache` OpenSpec.

## What Changes

**Scope decision — Layer 5 V2.2** (commutation-time
diagnostics):

- Extend `SimulationResult` with:
  ```cpp
  struct CommutationEvent {
      Real t_estimated;       // linear-interp zero of watched
                              // signal between t_n, t_{n+1}
      Index branch_id;        // which diode flipped
      bool new_state;         // true = ON, false = OFF
  };
  std::vector<CommutationEvent> commutation_events;
  ```

- After each step's event iteration converges with a new
  mask, for each diode that flipped:
  1. Compute the watched signal (`i_diode` for ON→OFF,
     `v_diode` for OFF→ON) at `t_n` (previous step's state)
     and `t_{n+1}` (just-converged state).
  2. Linearly interpolate `t* ≈ t_n + dt · (s_n / (s_n − s_n+1))`
     where `s` is the watched signal.
  3. Push `{t*, branch_id, new_state}` to
     `commutation_events`.

- **No change to state recording**. `times[k]` and `states[k]`
  remain at the dt grid. Full sub-step correction is V1.

## Impact

- **Affected specs**: ADDED requirement on `kernel-v2-solver`
  for the new diagnostic vector.
- **Affected code**:
  - MODIFIED `solver/result.hpp` (~10 LOC for the new vector)
  - MODIFIED `solver/run_transient.hpp` (~30 LOC for the
    interpolation logic inside the event-iter loop)
  - MODIFIED `pwl/diode_event_state.hpp` (~5 LOC to expose
    the per-diode "what flipped" info)
  - NEW `tests/v2/layer5_v2/test_commutation_events.cpp`
- **Migration**: zero. The new vector is purely additive;
  existing tests don't read it.
- **Risk**: low. Linear interpolation is monotone for
  well-behaved signals; pathological cases (s_n and s_n+1
  same sign) fall back to `t* = t_{n+1}` (the dt-grid time).
