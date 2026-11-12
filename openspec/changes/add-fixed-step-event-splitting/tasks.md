## 1. Characterisation (after `add-discontinuity-restart-rule`, before any change here)
- [ ] 1.1 `python/tests/test_fixed_step_event_splitting.py`: pin `p16_events.py` — every OFF event's `t_estimated/dt` is an integer and 0.32–0.33 dt before the current's zero crossing; strict xfail of "within 0.05 dt"
- [ ] 1.2 Pin `p15_substep.py`: results bit-identical with `enable_substep_state_correction` on and off
- [ ] 1.3 Measure `p15_substep.py` and `p12.py` at N = 100, 101, 400 with the restart rule on and the split off; record the three values (this is the split's "before")
- [ ] 1.4 Port the off-grid cases of `p1_edge.py` (+0.3 dt, +0.7 dt): pin order 1.00; strict xfail of "error ≤ 1e-5 A at dt = 2e-6, order ≥ 1.9"
- [ ] 1.5 Buck CCM with D = 0.4567 against the `expm` reference: pin the snapped error; strict xfail of 5.2

## 2. Locating
- [ ] 2.1 Diode watcher reads the first solve of the event iteration (pre-event bits) and `crossing_signal`; C++ test on a ramp current with a known zero
- [ ] 2.2 Gate edges: thread `next_edge_after` from the binding to `run_transient`; bisection of an opaque `switch_fn` on steps whose end masks differ
- [ ] 2.3 Source jumps inside a step from the jump table

## 3. Splitting, linear path
- [ ] 3.1 Split through `run_event_iteration` (sub-step 1 trapezoidal at `dt1`, or two backward-Euler half-steps if the step restarts; sub-step 2 the restart rule at `dt2/2`); remove the direct `cache.solve_at` calls of the old correction path
- [ ] 3.2 Several events per step, in time order, cap 8, breach record beyond
- [ ] 3.3 Snap within 1 % of dt to the boundary
- [ ] 3.4 `commutation_events.t_estimated` = located time; `SimulationResult::n_split_steps`

## 4. Splitting, Newton path and stateful devices
- [ ] 4.1 Newton circuits split with the segment assembled at `dt1` and `dt2`; `refresh_dt` and `refresh_stage` asserted for both sub-steps
- [ ] 4.2 One test per stateful device (nonlinear capacitor, saturable inductor, Lauritzen diode, IGBT tail, PMSM) with an off-grid gate edge: the result at T/200 agrees with T/800 to the second-order bound
- [ ] 4.3 dt-retry inside a split sub-step: rollback restores the pre-step snapshot; `dt_retries` records the outer step

## 5. Acceptance (restart rule on, split on)
- [ ] 5.1 RL off-grid edge: error ≤ 1e-5 A at dt = 2e-6 for +0.3 dt and +0.7 dt (prototype −1.96e-6 and +4.72e-7), order ≥ 1.9
- [ ] 5.2 Buck CCM, D = 0.4567: `v_C` error ≤ 2e-5 V at T/50 (prototype 7.3e-6; 8.0e-2 snapped) and halving factor ≥ 3.5 from T/50 to T/800
- [ ] 5.3 `p15_substep.py` / `p12.py`: `v_C` at N = 100, 101 and 400 agree to below 1e-4 relative. Investigate the −1.9 mV bias against the exact value and report its cause if found; not an acceptance criterion
- [ ] 5.4 `p16_events.py`: `t_estimated` within 0.05 dt of the zero crossing for OFF events
- [ ] 5.5 Cross-engine: DCM buck at T/200, `engine='pwl'` against `'trbdf2'` (rtol 1e-6) and `'dsed'`, time-weighted `v_out` agreement ≤ 1e-4 relative; commutation times agree to 0.05 dt
- [ ] 5.6 Bridge rectifier (four diodes, two crossings in one step at coarse dt): no breach, charge balance unchanged in tolerance
- [ ] 5.7 Option off: results bit-identical to the restart rule alone

## 6. Cost
- [ ] 6.1 Count factorisations and solves per period on the buck at T/100 with fixed off-grid edges and with a duty that moves every period; record before and after
- [ ] 6.2 Gate: if the moving-duty case is more than 25 % slower than the restart rule alone on the perf harness, implement alternative B (interpolated sub-step 1) and re-run section 5

## 7. Python, docs
- [ ] 7.1 `steady_state.py` and `_diode_state.py` no longer assume `t_estimated` is on the grid; their tests pass
- [ ] 7.2 `simulate(enable_substep_state_correction=...)` docstring; error when set without the restart rule; stubs
- [ ] 7.3 `CHANGELOG.md` **BREAKING**; `AI_GUIDE.md` §2, §3.2 (`grid_pwm`) and trap 10: edges no longer have to lie on the grid, with the measured numbers; `docs/known-limitations.md`: delete the "`make_pwm_switch_fn` edges jitter by one dt step" row if 5.1 holds for that helper
- [ ] 7.4 Default flip and golden regeneration in one commit, with the harness table

## 8. Verification
- [ ] 8.1 `ruff check`, `pytest` (throttled), `ctest`, stub check
- [ ] 8.2 `openspec validate add-fixed-step-event-splitting --strict` and `openspec validate --all --strict`
