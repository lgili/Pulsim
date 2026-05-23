## Phase 1 — DiodeEventState event reporting (~0.25 days)

- [ ] 1.1 Add `struct DiodeUpdateEvent { Index branch_id;
      Index switch_idx; bool new_state; }`.
- [ ] 1.2 Add private member `std::vector<DiodeUpdateEvent>
      last_events_`.
- [ ] 1.3 `update_from_state` writes to `last_events_` (clears
      it first).
- [ ] 1.4 `last_update_events() const` returns the events.

## Phase 2 — SimulationResult.commutation_events (~0.15 days)

- [ ] 2.1 Add `struct CommutationEvent { Real t_estimated;
      Index branch_id; bool new_state; }` to result.hpp.
- [ ] 2.2 Add `std::vector<CommutationEvent>
      commutation_events` to `SimulationResult`.

## Phase 3 — run_transient interpolation logic (~0.4 days)

- [ ] 3.1 At the start of each step's event-iter loop,
      snapshot `prev_x = x`.
- [ ] 3.2 After the loop converges, for each event in
      `diodes.last_update_events()`:
      - Find the diode's terminals from the pool.
      - Compute v_diode at prev_x and at x.
      - Compute i_diode at prev_x and at x using
        the diode's current g_on or g_off.
      - Choose the watched signal:
          OFF → ON: v_diode − V_th
          ON → OFF: i_diode
      - Linearly interpolate t* in [t_prev, t]:
          t* = t_prev + (t - t_prev) · |s_n| / (|s_n| + |s_n+1|)
        when s_n and s_n+1 have opposite signs.
      - Clamp to [t_prev, t].
      - Push to `result.commutation_events`.

## Phase 4 — Tests (~0.3 days)

- [ ] 4.1 Half-wave rectifier: verify ~2 events per cycle
      at the zero-crossings within 1 dt.
- [ ] 4.2 Boost converter: verify 1-2 events per PWM cycle
      at PWM transitions.

## Phase 5 — CMake + regression + docs (~0.15 days)

- [ ] 5.1 No new CMake target needed — tests go in
      `tests/v2/layer5_v2/`.
- [ ] 5.2 All previous tests stay green.
- [ ] 5.3 `openspec validate pulsim-v2-substep-bisection
      --strict` passes.
- [ ] 5.4 `docs/pulsim-v2/layer5-v2.2-substep-diagnostics.md`.
