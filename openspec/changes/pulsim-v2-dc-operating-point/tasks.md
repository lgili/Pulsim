## Phase 1 — DC assembly + solver (~0.5 days)

- [ ] 1.1 `pwl/dc_assemble.hpp`: dc_assemble dispatches like
      assemble_segment but with caps SKIPPED and inductors as
      v=0 constraints (using the existing branch-current
      unknown layout).
- [ ] 1.2 `pwl/dc_operating_point.hpp`: `compute_dc_op(graph,
      pool, mask)` → solves the DC system, returns Vector.
- [ ] 1.3 Throws on singular DC matrix.

## Phase 2 — Seeding helpers (~0.25 days)

- [ ] 2.1 `pwl/seeding.hpp`: `seed_history_from_dc_op(history,
      dc_x, graph, pool)` populates each HistoryState entry's
      v_prev / i_prev from `dc_x`:
      - Cap: v_prev = v_C from dc_x, i_prev = 0 (DC current is 0).
      - Inductor: v_prev = 0, i_prev = i_L from dc_x.
- [ ] 2.2 `seed_diodes_from_dc_op(diodes, dc_x, graph, pool)`
      decides each diode's initial state from the DC v_diode.

## Phase 3 — Layer 5 V3 run_transient overload (~0.5 days)

- [ ] 3.1 Add `bool start_from_dc_op = false` parameter to the
      6-arg run_transient.
- [ ] 3.2 When true:
      - Iterate diode consistency on the DC solve.
      - Seed HistoryState from dc_x.
      - Record sample 0 = dc_x (NOT zero).
      - Continue loop from sample 1.
- [ ] 3.3 When false: V2.1 behaviour unchanged.

## Phase 4 — Tests (~0.5 days)

- [ ] 4.1 DC OP unit tests (V-R-GND, V-R-C-GND, V-R-L-GND).
- [ ] 4.2 Seeding tests (after seed_history, the next trap
      step computes correct values).
- [ ] 4.3 Integration: RC circuit with DC OP → v_C stays at
      V_dc throughout (already at steady state).
- [ ] 4.4 Integration: LC tank with DC OP → no transient
      oscillation (within numerical noise).

## Phase 5 — CMake + regression + docs (~0.25 days)

- [ ] 5.1 New target `pulsim_v2_layer4_v2_tests`.
- [ ] 5.2 Regression: all previous tests stay green.
- [ ] 5.3 `openspec validate pulsim-v2-dc-operating-point
      --strict` passes.
- [ ] 5.4 `docs/pulsim-v2/layer4-v2-dc-operating-point.md`.
