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

2. **One pair or two (D8).** → SUPERSEDED by the rework below (three
   SDIRK(2,2) steps). The original answer was one pair (capacitor loop
   9.995 … 10.003 A at 1e-8, 9.821 … 10.178 A at 1e-9).

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
- **dt-retry ladder.** Superseded by the rework: every sub-step of a
  restart step is an SDIRK step. Ladder depth, sizes and `DtRetry`
  records are unchanged.
- **Sub-step correction (V3).** Superseded by the rework: the
  combination is refused.
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
  alike (pre-existing); its BE stamp is covered by the oracle test. The
  rework adds a PMSM test of its own through a restart (below).
- **Task 6.3** needed no code: `losses.py` and `_diode_state.py` already
  read the recorded `switch_states` and fall back to `switch_fn` only
  when no record exists; the pwl engine always records one.
- **Restart triggers on unsupported engines.** `engine='dsed'` refuses
  the flag; `engine='trbdf2'`/`auto` treat it as a blocker toward the
  fixed engine; `mmc_arms` refuses it. trbdf2 and dsed are bit-identical.

## Rework: three SDIRK(2,2) steps replace the backward-Euler pair

Coordinator decision after review of 41711488: the pair drained resonant
modes (tank −11.9 % / −3.1 % at T/50 / T/100; isolated-PWM LC kept
0.019 of its energy). The decision itself (SDIRK(2,2), γ = 1 − 1/√2,
three steps, stage 2 from `x0*`, one factor per mask, counter in the
snapshot) is the reviewer's; the entries below are the choices made
while applying it.

- **Diodes per stage or per step.** → Per stage (each stage runs its
  own event iteration; a flip in either sets the window to
  `max(left, 2)`). → Measured in the engine with a temporary switch: per
  step (stage 1 under the old bits, step redone on a flip) collapses the
  buck (CCM `v_C` error −11.9 V, DCM output 0.83 V instead of 13.3 V);
  per stage gives CCM `i_L` 1.17e-7 A at T/50, order 2, DCM idle 0.05 %.
  → Reverse: n/a (the alternative is wrong).
- **Where the stage factors live.** → A second lazy
  `PwlStateSpaceCache` at `2γ·dt`, same byte budget, used when a solve's
  dt equals it. → The 8-entry event LRU of `solve_at` refactored on every
  restart step of a VSI (3172 builds in 3157 restart steps, dt = 1e-6);
  now one per visited mask (`n_restart_factorizations`). The ladder's
  sub-steps (2γ·dt/2^j) still use `solve_at`. The Newton path assembles
  per stage as before. → Reverse: delete `restart_cache` in
  `run_transient.hpp`; the solves fall back to `solve_at`.
- **Window counter on a diode flip.** → `max(left, kRestartSteps − 1)`:
  the step with the flip is itself a restart step, the window covers
  the next two. A flip in a trapezoidal step rolls back and re-takes it
  as a restart step with the same rule. → Reverse: change the constant.
- **Legacy pickles.** → Field 12 as a bool `True` loads as 3 steps,
  `False` as 0. → Old snapshots predate the rework; 3 restarts as from
  a fresh discontinuity, the conservative reading. → Reverse:
  `bindings.cpp` setstate.
- **`b_extra_fn` jumps (fix 1).** → An opt-in attribute
  `fn.next_jump_after(t) -> float` (the next jump strictly after `t`, or
  `inf`). A declared jump in `(t − left_by, t]` is read at
  `t − left_by` (left limit) and a jump in the step arms the window.
  Composed callbacks (`_compose_step_hooks`, C-blocks) carry the minimum
  of their parts (`_carry_jumps`). Undeclared jumps are not restarted;
  documented in CHANGELOG, AI_GUIDE and known limitations. → Reverse:
  drop the attribute lookup in `bindings.cpp`.
- **Restart with sub-step correction (fix 2).** → Refused
  (`std::invalid_argument` → `ValueError`, naming both options). → The
  V3 path commits a trapezoidal history from an interpolated state that
  the SDIRK window would not see; the split change replaces V3. →
  Reverse: remove the check in `run_transient.hpp` validation.
- **MNA PMSM through a restart (fix 3).** → Tested, not refused. Every
  fixed step of an MNA PMSM goes through the dt/2 retry (its linear
  segment has no stamp for the phase currents), flag on or off. Inside
  the window the retry's sub-steps are SDIRK steps. Flag on, the
  step-halving ratios are 4.05 / 4.03; flag off the differences do not
  shrink (1.6e-3, 2.3e-3, −1.2e-3). → Reverse: n/a.
- **Static path (fix 5).** → Same sampling as the dynamic path (gate at
  `t_{k−1}`, left limits) and no restart (no dynamics). Tested through
  `run_transient` on a cache built at dt = 0 (`simulate` always builds
  with a dt). → Reverse: `gate_t`/`src_left` in the static loop.
- **Energy gates (fix 6).** → The tank gate is "flag-on error ≤ 1.5×
  flag-off" at T/50 and T/100 (measured 0.85× and 0.73×). The LC gate is
  "|E_on − E_off| ≤ 0.5 %" (measured 0.12 %). The smooth-segment test
  measures from E[0]: 5.49e-6 with the flag on (three SDIRK steps, order
  4 in energy), 3.93e-3 with it off (the half-step start), and < 1e-10
  after the window for both.
- **Jiles–Atherton stage-1 flux.** → Evaluated from the committed JA
  state, not a stage-1 JA update. → No JA state per stage exists; the
  error is O(h) at a loop tip only, and is documented in
  `saturable_inductor_history.hpp`. → Reverse: n/a.
- **Pinned expectations moved by the rework (flag on only):**
  - `test_gate_edge_on_the_grid_is_second_order`: bound 1e-5 → 2e-6.
  - buck CCM: 5e-5 → 2.5e-7, T/400 and T/800 dropped (they reach the
    reference's rounding).
  - DCM `v_out`: 1.5e-3 → 6e-5.
  - cap loop at 1e-9: ± 0.5 → ± 0.02 A, plus the power.
  - RC step: 0.03 → 5e-3 V.
  - PWM source restarts: 20 → 60.
  - LC restarts: 1 → 3.
  - nominal-factor test → one factor per mask (`event_builds` 0,
    nominal segments 2 instead of 4).
  - first-step test: the SDIRK amplification on steps 1–3, trapezoid
    on step 4, `n_restart_steps` 3.
  - snapshot `restart_pending` → `restart_steps_left`.

## Integration with update-trbdf2-factor-reuse (2026-11-13)

- **Where the stage factors live, revisited.** → The second lazy
  cache is removed; the stages call `cache.solve_at(mask, 2γ·dt)`. →
  The 8-entry LRU that motivated it became a byte-budgeted per-mask
  table with an in-place refactor and the regime's shared symbolic
  analysis. Measured, VSI 10 ms at dt = 1e-7: results bit-identical
  (golden `vsi_pwl_restart`, `buck_pwl_restart`); 56 restart
  factorisations = 56 event builds, 0 refactors; one symbolic analysis
  and one structure build per run (the second cache ran its own). A
  `refactor_parametric` (on_cache) now reaches the restart steps: it
  clears the event table, which the private cache never saw.
  `n_restart_factorizations` counts the event builds + refactors of the
  restart solves. → Reverse: restore `restart_cache` in
  `run_transient.hpp` (and refuse `on_cache` with the rule).
- **Pulse current source** (add-controlled-source-set) → a jump source
  like the pulse voltage source (`source_jumps.hpp`); the table current
  source is continuous by validation and does not restart.

## Stage-2 fallback (fix after the integration-branch report, 2026-11-13)

- **Defect.** → With the flag on, `three_phase_diode_rectifier` at
  dt/τ = 10 charged its bus to 37.20 V on 24 V-peak phases. Measured
  mechanism: stage 1 (diode ON) gives x1 = 15.4 V, so the stage-2
  pseudo-state is x0* = x0 + ((1−γ)/γ)(x1 − x0) = 37.2 V. Stage 2 ON
  overshoots to 25 V (> the 20.7 V phase; R(−10) ≈ −0.2), the diode
  turns OFF, and the OFF solve of an undamped capacitor returns x0*
  itself. An OFF solve from x0* = 37.18 V reproduces 37.203 V.
- **Fix chosen.** → When stage 2's event iteration flips a diode,
  re-try it as backward Euler over (1−γ)h from x1. Keep the BE step only
  when its mode equals stage 1's (the flip was the overshoot); otherwise
  restore the SDIRK stage 2 bit for bit. → The rule keeps its per-stage
  diode decision (deciding once per step collapsed the buck to
  −11.9 V), and real mode changes keep SDIRK accuracy.
- **Rejected.** → (a) BE remainder for every stage-2 flip: rectifier
  equal, but DCM buck T/50 −2.41e-4 instead of +2.93e-5 (8× worse,
  still order 2). (b) Recursive SDIRK on the remainder: depth 1 DCM
  −1.67e-4 with 2948 event-LRU refactors in 1000 periods; depth 4
  −1.26e-4 with 6412.
- **Numbers** (probe `probes/rectifier_stage2.py`, max |err| vs trbdf2
  rtol 1e-8 over 40 ms):

  | circuit, dt | off | on, before | on, after |
  |---|---|---|---|
  | benchmark, 50 µs: peak | 33.99 V | 37.20 V | 23.999 V |
  | benchmark, 50 µs: err | 13.45 | 16.66 | 0.762 |
  | L-L bridge, 50 µs: peak | 59.35 V | 59.63 V | 41.566 V |
  | L-L bridge, 50 µs: err | 17.84 | 18.35 | 3.44 |
  | benchmark / bridge, 5 µs: err | 6.21 / 8.04 | 0.363 / 0.167 | 0.363 / 0.167 |

  The first version re-tried every stage-2 flip and, on the buck CCM,
  kept backward Euler once, at t = 0 (the diode turning ON on
  round-off): a dt-independent +7e-10 A (1.182e-7 / 3.015e-8 /
  7.295e-9 A at T/50/100/200 against 1.175e-7 / 2.946e-8 / 6.608e-9),
  and one more nominal segment (2 → 3) plus one event entry. Review
  finding H1: the trigger is now "stage 2 turned at least one diode OFF
  and none ON", the mechanism isolated above. Buck CCM, DCM, tank and LC
  are bit-identical to before the fallback again (0 re-tries; order 2
  through T/1600 by successive differences: factors 3.93 / 3.98 /
  4.20); the factor test is back to 4 builds, 0 refactors, 2 segments.
  The rectifier: 19 re-tries, 5 kept at dt = 50 µs, 18 / 5 at 5 µs
  (L-L bridge 38 / 11 and 36 / 0): the count does not grow as dt
  shrinks. A re-try that throws keeps the SDIRK stage
  (`stage2_retry_failures`; review finding M2).
- **Reverse.** → Delete the `has_diodes && iters_2 > 1` block in
  `run_restart` and set `be_commit_h` back to γh; the two rectifier
  tests then fail.
