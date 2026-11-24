## 1. Model law
- [x] 1.1 `SwitchedDiode::Params::V_f`; `turn_on_voltage`, `branch_current`, `decide`, `crossing_signal`

## 2. Stamping
- [x] 2.1 `stamp_switch_fixed(..., v_drop_on)`; `pwl::detail::stamp_switch_branch` shared by `assemble_impl` and `dc_assemble`

## 3. Pool, census, decisions
- [x] 3.1 `DevicePool::add_diode(V_f)`, `DiodeEventState` carries `V_f` and decides through `decide`
- [x] 3.2 `CircuitBuilder::add_diode(V_f)` validated by name

## 4. pwl watcher
- [x] 4.1 sub-step watcher on `v − V_on`

## 5. TR-BDF2
- [x] 5.1 `crossing_signal` and `held_tolerance` use the law

## 6. DSED
- [x] 6.1 predicates on `v − V_on` / `v − V_f`; settle through `decide`; band hint in the errors

## 7. Python law and consumers
- [x] 7.1 `components()` exports `V_f`
- [x] 7.2 `_diode_state`: `current`, `current_start`, `conducting`, replay rule, rule-violation check
- [x] 7.3 `losses`: `_conduction_stats`, `_branch_energy`, `_window_conductance`, diode block, `_chatter_gaps`
- [x] 7.4 `thermal`: diode power and energy

## 8. Topology helpers
- [x] 8.1 `add_three_phase_vsi(V_f=)` is a drop, new `V_th=0.0` (BREAKING)
- [x] 8.2 `add_bridge_rectifier(V_f=0.0)` passthrough

## 9. Plumbing and docs
- [x] 9.1 binding kwarg and docstrings; YAML `V_f`; stubs
- [x] 9.2 AI_GUIDE traps 1 and 13 and the executed snippet; docs; CHANGELOG

## 10. Tests
- [x] 10.1 C++: law tables, stamp deltas, forced current, event-state hold/drop at `V_f ± ε`
- [x] 10.2 Python: forced current, half-wave, RL extinction, bridge balances, VSI drop, dead-time duty, `V_f = 0` identity, engine agreement, DSED band, losses vs thermal, windows, plumbing
