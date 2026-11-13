# Decisions taken while applying add-discontinuity-restart-rule

Rule for every entry: the open question's proposed option, else the
most conservative one. Format: question → option → why → how to reverse.

## Open questions of design.md

1. **Coss charge test (D2).** → Accept the reading "the half-step
   offset is the defect, `Q = I·t` is the fix". → Measured with the
   flag on, `Q/(I·t)` = 1.000000 at samples 1, 2, 11 and 40 (flag off:
   0.5, 0.75, 0.955, 0.988), so the stop condition (`I·(t − h)`) did not
   occur. The flag-on test asserts `Q = I·t` to 1e-6; the flag-off pin
   in `test_nonlinear_capacitor.py` is untouched (default off). →
   Reverse: delete `test_coss_charge_is_exact_from_the_first_step`.

2. **One pair or two (D8).** → One pair (the paper's rule, the
   proposed option). → The capacitor loop meets 10.00 ± 0.01 A at
   dt = 1e-8 (9.995 … 10.003) and ± 0.5 A at dt = 1e-9 (9.821 … 10.178);
   no target needed the second pair. → Reverse: loop `run_restart`
   twice in `run_transient.hpp`.

3. **Slot-zeroing or an explicit stage (D3).** → Explicit
   `TrBdf2Stage::BackwardEuler` / `CossStage::BackwardEuler`, as D3
   proposes; the slot-zeroing identity is kept as the oracle of
   `core/tests/layer5_v1/test_backward_euler_stage.cpp` (1e-14). → The
   stateful devices (Lauritzen, IGBT tail) have no slot to zero, so the
   stage is needed anyway. → Reverse: route the BE branches through the
   trapezoidal code with zeroed slots; the oracle test stays valid.

4. **Release of the default flip; does the flag-off path survive.** →
   Most conservative: the default stays `false` in this change (Migration
   step 1); tasks 7.2/7.3 are left open; the flag-off path stays and is
   bit-identical (20 baseline runs, `probes/pwl_baseline.py check`). →
   The flip needs an owner decision on the release. → Reverse: n/a.

5. **Step-observer label.** → Keep `(t_k, x_{k−1})` and the `t_start`
   prime call (the proposed option and D4's text); only the mask moves to
   `switch_fn(t_{k−1})`. → The label belongs to
   `fix-control-tick-semantics`. A first draft moved it to `t_{k−1}`; it
   was reverted to follow D4, and
   `test_the_observer_keeps_its_label_under_the_rule` pins the flag-on
   calls equal to the flag-off ones. → Reverse: in `run_transient.hpp`,
   pass `gate_t` to `step_observer` and skip the prime call when
   `restart_rule`.

6. **Consistent initialisation as its own change.** → Yes, a separate
   change (not opened here; no change outside this one is implemented).
   Sample 0 of node voltages still reads 0 V. → Reverse: n/a.

## Choices the design left open

- **Left-limit offset.** A source whose jump coincides with the step end
  is read at `t − left_by`, `left_by = 1e-3·dt`. Jumps are detected with
  the same `pwm_phase` snapping as `value_at`, so the offset only picks
  the side and never lands in a ramp of finite rise. Reverse: change
  `left_by` in `run_transient.hpp`.
- **dt-retry ladder.** Sub-step 0 of a restart step is a restart
  sub-step; later sub-steps are trapezoidal and are not rolled back on a
  flip — a flip there sets `restart_pending` for the next step. Ladder
  depth, sizes and `DtRetry` records unchanged.
- **Sub-step correction (V3)** forces the next step to restart (its
  history is the corrected state, not a trapezoidal one).
- **Characterisation tests** were written as flag-off pins plus flag-on
  targets in one file, not as strict xfails later un-xfailed: with the
  default off both behaviours coexist, and the flag-off pins survive the
  future flip as `discontinuity_restart=False` checks. The four
  `test_first_step_order.py` xfails stay (they describe the default
  path); the flag-on versions are new tests.
- **Capacitor-loop power target.** Task 5.1's 0.0490 W is not the window
  the probe integrates; the test asserts the exact value of its window,
  0.04599 W (flag on 0.0459 W; flag off 0.1968 W).
- **PMSM in the composition test (4.6).** Excluded: an isolated MNA PMSM
  forces whole-circuit dt/2 retries at 1 ns with the flag on and off
  alike (pre-existing); its BE stamp is covered by the oracle test.
- **Task 6.3** needed no code: `losses.py` and `_diode_state.py` already
  read the recorded `switch_states` and fall back to `switch_fn` only
  when no record exists; the pwl engine always records one.
- **Restart triggers on unsupported engines.** `engine='dsed'` refuses
  the flag; `engine='trbdf2'`/`auto` treat it as a blocker toward the
  fixed engine; `mmc_arms` refuses it. trbdf2 and dsed are bit-identical.
