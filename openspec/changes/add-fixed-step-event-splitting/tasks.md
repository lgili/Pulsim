## 1. Characterisation (after `add-discontinuity-restart-rule`, before any change here)
- [x] 1.1 `python/tests/test_fixed_step_event_splitting.py`: pin `p16_events.py` — every OFF event's `t_estimated/dt` is an integer and 0.32–0.33 dt before the current's zero crossing (default engine; 0.98 dt with the rule alone); the split target "within 0.05 dt" asserted directly (both sides in one test, DECISIONS.md)
- [x] 1.2 Pin `p15_substep.py`: results bit-identical with `enable_substep_state_correction` on and off — superseded: the option now requires the rule and is the split; "off = the rule alone, bit for bit" is pinned instead (`probes/split_probe.py check`, `test_split_off_is_the_restart_rule_bit_for_bit`)
- [x] 1.3 Measure `p15_substep.py` and `p12.py` at N = 100, 101, 400 with the restart rule on and the split off: relative error 8.6e-7 / 6.9e-6 / 2.1e-7 (default engine −2.3e-4 / −5.0e-3 / −2.2e-4). The rule alone removes the T/101 outlier
- [x] 1.4 Port the off-grid cases of `p1_edge.py` (+0.3 dt, +0.7 dt): rule alone order 1.00 (−8.497e-3 / −3.635e-3 A at 2 µs), pinned beside the split target
- [x] 1.5 Buck CCM with D = 0.4567 against the `expm` reference: snapped error 8.08e-2 V at T/50 and T/100, pinned beside the split target

## 2. Locating
- [x] 2.1 Diode watcher reads the first solve of the event iteration (pre-event bits) and `crossing_signal`; C++ test on a ramp current with a known zero
- [x] 2.2 Gate edges: thread `next_edge_after` from the binding to `run_transient` (C++ schedules, and a Python schedule object with the method); no bisection of an opaque `switch_fn` (open question 4, DECISIONS.md)
- [x] 2.3 Source jumps inside a step from the jump table (`sources::next_source_jump_after`)

## 3. Splitting, linear path
- [x] 3.1 Split through `run_event_iteration` (sub-step 1 trapezoidal at `dt1`, or an SDIRK(2,2) step if the step restarts; sub-step 2 an SDIRK(2,2) step of `dt2`, then the window's two grid steps); the direct `cache.solve_at` calls of the old correction path are removed
- [x] 3.2 Several events per step, in time order, cap 8, breach record beyond
- [x] 3.3 Snap within 1 % of dt to the boundary
- [x] 3.4 `commutation_events.t_estimated` = located time; `SimulationResult::n_split_steps`

## 4. Splitting, Newton path and stateful devices
- [x] 4.1 Newton circuits split with the segment assembled at `dt1` and `2γ·dt2` (the single choke point `run_event_iteration` sets `refresh_dt`/`refresh_stage`)
- [x] 4.2 Stateful devices with an off-grid event: saturable inductor, coupled transformer, Lauritzen diode, IGBT tail, Richardson ratio 3.4–4.7 with the split, not second order without; PMSM excluded (every fixed step of an MNA PMSM is retried, and a retried step is not split)
- [x] 4.3 dt-retry inside a split step: rollback restores the pre-step snapshot, the step is re-taken without its interior events; `dt_retries` records the outer step

## 5. Acceptance (restart rule on, split on)
- [x] 5.1 RL off-grid edge: +1.012e-6 / +1.014e-6 A at dt = 2 µs for +0.3 / +0.7 dt, order 1.98
- [x] 5.2 Buck CCM, D = 0.4567: `v_C` error 1.07e-6 V at T/50 (prototype 7.3e-6; 8.08e-2 snapped), factors 4.2 / 4.2 / 4.3
- [x] 5.3 `p15_substep.py` / `p12.py`: relative errors 7.2e-8 / 7.3e-8 / 1.3e-9 at N = 100 / 101 / 400. The −1.9 mV bias of the default engine is gone with the rule alone (+7.7e-6 V at T/100); since the restart alone removes it, it came from the first-order history after the three events per period, not from the diode law
- [x] 5.4 `p16_events.py`: `t_estimated` within 1e-4 dt of the zero crossing for OFF events
- [x] 5.5 Cross-engine: DCM buck at T/200, time-weighted `v_out` against `trbdf2` within 1e-4 relative
- [x] 5.6 Bridge rectifier (four diodes, two crossings in one step at coarse dt): no breach
- [x] 5.7 Option off: results bit-identical to the restart rule alone (40 probe runs, keyword absent and `False`)
- [x] 5.8 Resonant gates of the restart rule with every PWM edge 0.37 dt off the grid: tank −0.592 % / −0.011 % at T/50 / T/100 (rule alone −0.663 / −0.024, flag off −0.855 / −0.055); isolated LC 0.9980 (rule 0.9973, off 0.9961)

## 6. Cost
- [x] 6.1 Buck at T/100, 200 periods: off-nominal refactorisations per period 0.76 (fixed off-grid duty) and 2.96 (duty moving every period), 0 with the rule alone; restart factors 3–4 either way
- [x] 6.2 Gate: moving duty +6.6 % ns/step against the rule alone (fixed +10.4 %), under 25 %: alternative B not implemented

## 7. Python, docs
- [x] 7.1 `steady_state.py` and `_diode_state.py` reviewed: `_diode_state` reads `t_estimated` only for `trbdf2`; `steady_state` rounds it to the nearest grid index for its pattern check, which stays a valid "same commutation sequence" test
- [x] 7.2 Option comment in `SolverOptions`; `UnsupportedError` when set without the restart rule; stubs
- [x] 7.3 `CHANGELOG.md` **BREAKING**; `AI_GUIDE.md` §3.2 and trap 10; `docs/known-limitations.md` and `KNOWN_LIMITATIONS.md` rows updated (the `make_pwm_switch_fn` row stays: the default is off)
- [ ] 7.4 Default flip and golden regeneration in one commit, with the harness table — owner decision, together with the restart rule's flip

## 8. Verification
- [x] 8.1 `ruff check`, `pytest`, `ctest`, stub check
- [x] 8.2 `openspec validate add-fixed-step-event-splitting --strict` and `openspec validate --all --strict`
