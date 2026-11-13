## 1. Characterisation (committed first)
- [x] 1.1 `python/tests/test_trbdf2_error_control.py`: pin `p10_lte.py` in C++ or through one accepted step of an RC — true local error / estimate between 1.9 and 2.0; strict xfail of "between 0.9 and 1.1" — pinned through a whole RC run instead of one step (`test_estimate_is_the_true_local_error`: 58 → 66 steps, 8.17e-4 → 5.22e-4 V at rtol 1e-5); characterisation and fix in one commit (DECISIONS 6)
- [x] 1.2 Pin `p17_span.py`: output error at 40 T is 8.2e-4 V with `t_end` = 40 T and 1.04e-3 V with 400 T (1541 and 761 steps); strict xfail of 3.3 — replaced by `test_floors_do_not_depend_on_t_end` (dt = T/20; diverged at sample 24 before)
- [x] 1.3 Pin `p13b.py`: 32 rejects, 8 accepted steps below 1e-10 s, steps of 1.2e-10 and 6.3e-11 s before the 3.3 µs jump and 1.4e-14 s after it; and the tolerance sweep 2.69e-3 … 8.7e-6 V (accuracy "before") — measured before: 32 rejects, 8 steps < 1e-10 s, 2.69e-3 / 7.36e-4 / 1.82e-4 / 8.68e-6 V at rtol 1e-4/1e-5/1e-6/1e-8
- [x] 1.4 Record accepted steps and solves of the reference harness's variable-step circuits at the default tolerances — `update-trbdf2-post-event-restart/probes/trbdf2_baseline.py` (before = that change; table in CHANGELOG)

## 2. Source breakpoints
- [x] 2.1 `sources/`: breakpoint table (pulse start, end of rise, start and end of fall, per period; PWM voltage source edges); `next_breakpoint_after(t)`; left-limit evaluation (shared with `add-discontinuity-restart-rule` if present)
- [x] 2.2 Combine with the gate-edge landing; a landed jump reads the left limit
- [x] 2.3 Acceptance on `p13b.py`: the sample at each jump is reached by a clamped step; no accepted step shorter than 1e-9 s in the 1 µs before a jump; max error at rtol 1e-6 not above 1.9e-4 V (prototype: unchanged at 1.824e-4). With `update-trbdf2-post-event-restart` present, also no accepted step shorter than 1e-10 s after a jump — met: rejects 32 → 4, error 1.16e-4 V; the only steps < 1e-10 s are the two committed restart probes
- [x] 2.4 A pulse source with finite rise time: both corners are sample times

## 3. Span-free bounds
- [x] 3.1 `h_ref` and `t_ulp`; the five constants of design D3; C++ test: with `h_ref = span/1000` every constant equals today's value — by construction (`span_ref` = span, DECISIONS 4); tested end to end instead: `test_doubled_tolerances_restore_the_previous_density` (bridge at 2e-5 / 2e-8 repeats the old 1242 steps)
- [x] 3.2 Default ceiling from the circuit's caps; `TrBdf2Stats::h_max_basis`; binding and stub — ceiling keeps `span/1000` in its minimum (DECISIONS 1); stats are a dict (no stub change)
- [x] 3.3 Acceptance: a buck driven by `NativePwm2Switch`, `t_end` = 40 T, 400 T and 4000 T — the accepted step sequence up to 40 T is identical and the state at 40 T agrees to 1e-12 relative; with an opaque `switch_fn` the three runs differ only before the first learned pulse — met only with a fixed ceiling (`dt` given): identical times and states to 39 T; with the default ceiling `span/1000` stays (DECISIONS 1)
- [x] 3.4 The narrow-pulse regression (a 100 kHz pulse train on a 10 ms run, peak detector 9.87 V) and the long-run boost regression (3999 gate events) still pass — full pytest

## 4. Stiff filter
- [ ] 4.1 (deferred, DECISIONS 5) `PwlStateSpaceCache`: solve on an existing `(mask, dt)` factor with a caller-supplied right-hand side
- [ ] 4.2 (deferred, DECISIONS 5) `HistoryState`: error right-hand side and read-back; filtered estimate on the linear path; statistics field for "unfiltered (Newton)"
- [ ] 4.3 (deferred, DECISIONS 5) C++ test on an RC with τ = 1e-12 s beside a 1 ms RC, the fast state at equilibrium to 1e-9: the raw estimate rejects a 1 µs step and the filtered one accepts it; prototype table of design D2 reproduced for hλ = −0.05 and −5e3
- [ ] 4.4 (deferred, DECISIONS 5) Run the `h_chatter` regression circuits (the all-OFF bridge with a 1 mH inductor, the flyback at rtol 1e-6) with the floor disabled; report; remove the floor only if all pass
- [ ] 4.5 (deferred, DECISIONS 5) Measurement gate: solves per accepted step before and after (2 → 3 on the linear path); wall time of the harness's variable-step cases

## 5. Estimator scale
- [x] 5.1 Factor 2 in the estimate; un-xfail 1.1
- [x] 5.2 Acceptance: against 1.4, accepted steps rise by no more than 35 % on any harness circuit (estimate 26 %) and no reference error grows; buck output error at rtol 1e-4, 1e-6, 1e-8 is not above the 8e-4, 9e-5, 2e-6 V measured before — steps +1 % (buck CCM) to +27 % (buck at rtol 1e-8, LC tank); buck errors 7.18e-5 / 1.55e-5 / 7.55e-7; two reference errors grew (Shockley rectifier, bridge time-weighted), explained in CHANGELOG: same rtol is twice as strict and both are non-monotone in rtol
- [x] 5.3 Re-pin tests that hold variable-step step counts; list each with its old and new value — see CHANGELOG and the commit message

## 6. Documentation
- [x] 6.1 `CHANGELOG.md` **BREAKING**: tolerance meaning (double `rtol` / `atol` for the previous density), span-free bounds and the default ceiling, source breakpoints
- [x] 6.2 `docs/solvers.md`, `AI_GUIDE.md` §2 ("omit dt") and trap 13: what the ceiling is and when to pass `dt`; docstring of `simulate` for `rtol` / `atol`

## 7. Verification
- [ ] 7.1 `ruff check`, `pytest` (throttled), `ctest`, stub check
- [ ] 7.2 `openspec validate update-trbdf2-error-control --strict` and `openspec validate --all --strict`
