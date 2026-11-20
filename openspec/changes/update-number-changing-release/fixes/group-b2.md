# Release finding F-4-4 — the in-loop recorder at restart steps (group B, part 2)

Branch `fix/core-release-b2-recorder-restart`, from group B's finished
branch `fix/core-release-b-event-samples` (36836230). Tests:
`python/tests/test_release_findings_b2.py` (the release options explicit:
`engine='pwl', discontinuity_restart=True, enable_substep_state_correction=True`).

## Reproduction (base 36836230)

| check | truth | base |
|---|---|---|
| buck window charge, windows on stored samples | C·Δv of the stored samples (identity, round-off) | 4.0e-4 off |
| rectifier, same | same | 2.6e-8 off (> 1e-11 of scale) |
| buck / rectifier mid-step windows (test_f44's) | C·Δv, 2e-4 | 4.1e-4 / 4.4e-3 |
| buck energy balance from recorder outputs | supplied = dissipated + stored change | 3.57e-2 of supplied |
| in-loop vs stored-array feed, same run | equal (1e-12) | differ |

## Trace (buck, dt = 1 us, in-loop i(C) and v(out) kept every step)

Per step, `h/2·(i_C(k−1) + i_C(k)) − C·(v(k) − v(k−1))`: 123 steps of 1000
exceed 1e-15; the run has 122 restart steps and 1 inner event; all 122
restart steps start at a sample in `right_limit_index`; every other step is
exact. First rows: (0, 1 us] −2.1e-10, (1, 2 us] −2.2e-10, (2, 3 us]
−2.2e-10 (the run's start), (20, 21 us] −5.7e-11 ... (a gate edge's three).

Cause (group B's reading, confirmed): the feed sees a restart step's ENDS
only, and a capacitor reading is the companion's END current; an SDIRK(2,2)
step's charge is h·((1 − γ)·i1 + γ·i2), not the trapezoid of its ends.

A second trace (rectifier, dt = 0.25 us, after feeding the lines): one
step (21.78950, 21.78975] ms left 1.0e-7 C (window 1 off 3e-4), and
periodic 1.7e-12 steps near 5.98 ms. Each is a restart (sub)step whose
stage 2 ran in another mode than stage 1 (`lines.m1 != stamped_mask`):
group B gave such a step no line (no single state is KCL-consistent in
two modes), so it fell back to its ends. 128 of the rectifier's restart
(sub)steps at 0.25 us are of that kind.

## Fix (5b80e15b)

- `StepHooks::line` / `step_fns` (events/step_hooks.hpp): a lined step goes
  to the step sink line by line, its `on_step` functions see it whole;
  `StepRunView::cap_start / cap_end / line` carry the capacitor currents
  for the line being handed over (nullptr outside a line).
- `LoopFeed` (record/loop_feed.hpp): a capacitor reading takes the line
  point's current when given; a line's start right limit is a jump row
  (the existing `n_jumps` path); at the run's first step the start sample
  is the run's start state, the line's start a jump from it.
- `run_transient`: `restart_lined` fills `StepLines::c_*` (only with a step
  sink): at a restart start `2·w1·i1 + (1 − 2·w1)·i2`, i1 = C/(γh)·(v(X1) −
  v0), i2 = the commit's own (C/h_e)·(v(X2) − v_prev); an inner event's
  left limit the committed `i_prev`. A stage 2 in another mode than stage
  1: X1 (mode 1) held over w1·h, then X2 (mode 2) — an extra row at
  a + w1·h, the same SDIRK quadrature with every point a solved state in
  its own mode (the loss port and the stored limits take it too). The hook
  site calls `feed_lines` for a step with lines, `hooks->step` otherwise.
- `_recorders.record_result`: the stored pass reads `with_inner_events`.

## Numbers (release options explicit; truths)

| check | truth | before | after |
|---|---|---|---|
| buck window charge, windows on samples | C·Δv (identity) | 4.0e-4 | ≤ 1e-11 of scale |
| rectifier, same | C·Δv | 2.6e-8 | ≤ 1e-11 |
| buck energy balance, recorder outputs | Tellegen / stored change | 3.57e-2 | 3.0e-5 (7.6e-6 at dt/2, 1.9e-6 at dt/4: 2nd order) |
| in-loop vs stored-array feed | equal | differ | windows/periods/values 1e-12, jump rows = right-limit rows |
| window integrals vs trbdf2 rtol 1e-11 (p(V), p(L), p(C), p(S1), p(D1), i(L), v(out), p(R)) | 2nd order | p(V) 3.8e-2 at 1 us, not converging (ratio 2) | 2.1e-5 at 1 us, ratios 3.6-6.0 -> 4.0 to dt/8 |
| rectifier mid-step windows | C·Δv, 2e-4 | 4.4e-3 | 6e-8 |
| buck mid-step windows (test_f44) | C·Δv, 2e-4 | 4.1e-4 | 7.4e-4 / 1.7e-3 (2nd order: 1.9e-4 / 4.2e-4 at dt/2) — NOT met |

The exact engine is NOT the truth on this buck: at output_rtol 1e-10 it
differs from trbdf2 rtol 1e-10 AND the rule run at dt/16 by 1.08e-4 on
∫i(L), 1.6e-4 on ∫v(out), 2.0e-4 on ∫p(R) over [A, B], while those two
agree to 3e-8; a model difference of the exact route (not looked into;
for the orchestrator).

## What did not move

- Rule off: the 12-case recorder set (`.scratch/rec12.py`, rebuilt from
  the probes DECISIONS 22-24 description: buck pwl/trbdf2 ten probes incl.
  i(D1), i(C), p(C), products, period + declared windows; store_every 10
  with decimate + times; vsi; Shockley buck; MOSFET level 1; saturable
  inductor; 8-section RLC) bit-identical, 1296 arrays; recorder on vs off
  the same step sequence on all 12 (and on the 7 rule cases).
- pwl_identity (`--goldens --reference`): 335/379 identical; the 44 that
  differ are all `pwl+restart` / `pwl+split` configs (rule on), first
  difference `inner_events.after` (the held pieces' rows). Rule-on
  trajectories (times, states) identical to the base on buck, rectifier,
  bridge. trbdf2_identity (`--goldens --reference --extra --modes
  ',lcp,shadow'`) 371/371.
- Cost, instructions per accepted step, base 36836230 vs tip, min of 6
  (2 interleaved rounds x 3): no recorder buck pwl 3115.2 -> 3118.6
  (+0.11 %), vsi pwl 4779.3 -> 4781.4 (+0.04 %), buck trbdf2 11028.3 ->
  11029.2 (+0.01 %), bridge trbdf2 12770.8 -> 12774.2 (+0.03 %). One probe
  (keep='every'), rule off: buck pwl 3374.1 -> 3380.9 (+0.20 %), vsi pwl
  5016.6 -> 5022.9 (+0.13 %), buck trbdf2 11606.2 -> 11608.1 (+0.02 %)
  — the per-step work on a non-restart step is one null test of `L` at the
  hook site; half the pwl rise is the same shift the no-recorder runs show
  (layout). Rule on, one probe (information): buck 3886.2 -> 3946.5
  (+1.6 %), vsi 5718.0 -> 5744.6 (+0.5 %).

## Behaviour changes under the rule (for the release branch)

- pwl recorders now have `settled` event rows (the line starts' jumps,
  `n_jumps` > 0) beside the `within_dt` brackets: the release's
  `test_recorders.test_event_rows_against_the_bridge_truth` asserts
  `{"within_dt"}` for pwl and must read `{"within_dt", "settled"}` there
  (checked: with the rule explicit it fails only on that set). KCL at
  every sample and every event row's right side holds (test_kcl_holds…
  with the rule: rc, buck, rectifier pass).
- keep='times' at an event instant returns the right limit (documented);
  on a restart line that is the line start, a continuous state's value to
  O(h²) only.
- A mixed-mode restart (sub)step now has inner-event rows at a + w1·h in
  `result.inner_events` (not a located event: the mode change between the
  stages), and its loss-port energy is the SDIRK quadrature of the two
  stage powers.

## Left open

- test_f44 (strict xfail, reason updated): a window END inside a restart
  step, or at an event read by keep='times', reads the line's start value
  2·w1·X1 + (1 − 2·w1)·X2, a right limit to O(h²) only. Needs a TRUE right
  limit (group A's consistent start x0+, `event_rl_x`) as the jump row and
  a mid-step row (at a + h/2, value chosen so the two lines' trapezoid is
  still the step's quadrature), in run_transient's restart_lined, the
  stored limits and the loss port alike.
- Stateful device readings (nonlinear capacitor, saturable inductor,
  Lauritzen diode) at a line point read the committed (end) histories:
  not per line point.
- A waveform current source reads its value at the line point's time,
  not the stages' weighted values.

## Verification

Extension build; pulsim_core_record_tests, _dsed_tests, _perf_tests
(Eigen 5) pass; test_release_findings_b2 6/6, _b 15 + 1 xfail; area tests
(losses, loss port, recorders, probes, switch-state record, identity tool)
342 pass; full pytest 6114 passed / 110 skipped / 9 xfailed; ruff clean;
stubs --check up to date. NOT run: full ctest, Eigen 3.4.0, g++.
