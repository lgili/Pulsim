# R2 bucket 5 — notes

Branch `release/r2-bucket-5` from `origin/release/core-next-defaults` d2a15c91, extension
`so_release_r1`. Baseline run of the 21 files: 77 failed / 305 passed, the failing set equal to
R2_BUCKETS.md's list (diffed, no difference). After: 360 passed / 22 xfailed / 0 failed
(4 of the xfails are the first-step module's pre-existing strict xfail, now on the legacy start).

Counts per rule (77 failures): (a) 26, of which 18 parked as findings (xfail strict) and 8
passing against their truth; (b) 41; (c) 1; (d) 9 (3 of them doc-snippet outputs). A test that
needed two rules is counted once, by the rule that fixed its failing assertion.

## Per file

| file | fails | fix |
|---|---:|---|
| test_passives.py | 1 | (a) ESR step: expected value is now the closed-form RC response across the sample gap (the restart samples the left limit at the source edge). rel 1e-6 -> 1e-9, measured 7.9e-13. |
| test_trbdf2_post_event_restart.py | 1 | (b) the probe-floor island's two diodes `form='conductance'`: the subject is the 1e-9 S OFF leakage making a zero pivot. |
| test_mode_shadow.py | 1 | (b) the shadow's reference run is `mode_selection='legacy'` (the settle the shadow keeps deciding; the default is now 'auto'). |
| test_igbt_no_reverse_conduction.py | 1 | (a) FINDING F-5-1, xfail strict. |
| test_branch_form_reinit.py | 1 | (b) the "refused without the rule" case passes `discontinuity_restart=False`. |
| test_circuit_sweep.py | 1 | (c) the kpi's inner run (undriven Q1 on purpose) names `undriven='open'`. |
| test_control_graph_queue_resume.py | 1 | (b) the 'label' parametrisation names `control_tick='label'` (it was the default); the resume tests now cover both ticks explicitly. |
| test_design_evaluation.py | 1 | (a) FINDING F-5-2, xfail strict. |
| test_exact_period_end_pulse.py | 1 | (a) FINDING F-5-3, xfail strict on the d = 1e-7 parameter only. |
| test_result_i_all_kinds.py | 2 | (a) switch KCL reads the stamped mask (`res.switch_states`) instead of re-evaluating `sf(t)`, which misreads the left-limit sample at a gate edge; closed bound 0.5 A -> 2.4e-8 A, open bound 1e-5 A -> 2.4e-8 A (2·V·g_off; measured 1.20e-8 A both). (b) the no-record `NotImplementedError` exists only for a conductance-form switch (branch form stores the current): `form='conductance'`. |
| test_mode_selection_today.py | 2 | (b) the settle-alone run and the guard run pass `mode_selection='legacy'` (the r0 run passed by accident on the LCP before; now it tests the settle again). |
| test_exact_scoreboard.py | 2 | (a) against the mpmath truth the pwl cells improved: diode_rectifier 0.0097 -> 1.8e-4, flyback 0.95 -> 2.5e-6. Bounds moved by the file's own rule (30x above / 3x below): (3e-3, 0.03) -> (6e-5, 5.5e-3) and (0.3, 3.0) -> (8e-7, 7.5e-5). Both tighter. The scoreboard file itself was not regenerated (not this bucket's). |
| test_partial_results.py | 4 | (b) the failing rectifier passes `discontinuity_restart=False`: with the restart it converges (peak 399.19 V of 400 V); the subject is an abort's partial result. |
| test_doc_snippets.py | 3 | (a)/(d) in the DOCS the test runs (see "Files outside tests" below). |
| test_permeance_magnetic.py | 3 | (a) the midpoint energy balance is exact on every trapezoid step after the SDIRK start (`res.n_restart_steps` = 3, asserted); the start's residual is the restart's local error (transformer 8.2e-11 J at dt = 2e-7, 2.3e-15 J at dt/10). Tolerances unchanged (1e-9, 1e-8); measured 9e-14, 1.0e-10, 1.1e-13. NEW assertion: the EE core's stored energy equals t²/(2L) at rel 1e-9 (measured 1.1e-13; the restart-less run lags half a step, 5e-4). |
| test_first_step_order.py | 5 | (b) the defect's tests (the strict xfail, which XPASSed, and the half-BE first step) pass `discontinuity_restart=False`; docstring says the default now restarts. |
| test_engine_auto.py | 7 | (b) six tests read `_trbdf2_stats` / the TR-BDF2 step ceiling: `engine='trbdf2'`. (d) the no-dt surface test asserts the new route (`engine_used == 'dsed'`, `n_exact_steps > 0`). (a) FINDING F-5-10, xfail strict. |
| test_engine_auto_closed_loop.py | 4 | (b) three tests read `_trbdf2_stats`: `engine='trbdf2'`. (a) the fixed-engine tick test pinned its own restart-less tail mean 4.989752 V at 1e-4; the truth (same loop, exact edges) is 4.968199 V — exact engine 4.9681986, trbdf2 rtol 1e-8 4.9681979, pwl dt = 20 ns 4.9681630. The restart-less value was 2.2e-2 V off it, the release pwl at dt = 2 us is 3.9e-3 V off. Now asserted against the truth at abs 5e-3. JUDGMENT CALL for the merge review: the number grew (1e-4 -> 5e-3) because the reference changed from a self-pin to the truth; the old engine fails the new assertion. |
| test_mode_selector.py | 10 | (b) task 3 poses the port model on the conductance form; the oracle bridge's families pass no `form=`. A local fixture defaults `add_switch` / `add_diode` to `form='conductance'` for the two port tests. |
| test_diode_forward_drop.py | 12 | (d) on auto the half-wave / RL half-wave run exact: events from `res.event_times`, tol 1e-8 -> 1e-9 s (measured 2.2e-12 / 2.2e-11 s). (d) the V_f refusal states the branch form's exact message (`ValueError`, `add_diode("D_bad"): diode conduction drop V_F must be a finite number >= 0 V`). (b)+(e)-style: V_f = 0 vs omitted asserted BIT-IDENTICAL (times, states, switch states, every P_avg) under the release defaults on pwl / trbdf2 / dsed / auto; the parent commit's pinned numbers kept as the legacy-period proof under `next_defaults.legacy()` (hold at <= 1.3e-12 pwl / dsed buck & boost, 2.8e-4 dsed bridge, <= 7.5e-5 trbdf2). (a) bridge charge/power: FINDING F-5-5, xfail strict. |
| test_diode_state_reconstruction.py | 14 | (b) the TR-BDF2 midpoint-mask test names `engine='trbdf2'`. (a) 13 parked: F-5-4 (3), F-5-5 (1), F-5-6 (2), F-5-7 (2), F-5-8 (4), F-5-9 (1). Parameter-specific parks use `request.applymarker(xfail(strict=True))`. |

Tolerances tightened: test_passives 1e-6 -> 1e-9 (rel); test_result_i closed 0.5 A -> 2.4e-8 A and
open 1e-5 A -> 2.4e-8 A; test_exact_scoreboard pwl bounds (above); test_diode_forward_drop event
times 1e-8 s -> 1e-9 s on auto. Loosened: none, except the re-based closed-loop tail mean above
(flagged).

## Files outside tests (needed by test_doc_snippets.py; no bucket owns them)

`docs/getting-started.md`, `docs/tutorials/01-rc-charging.md`, `python/pulsim/AI_GUIDE.md`:
expected-output fences and the sentences next to them that became false.
- RC examples: the fixed engine restarts at the source step, so v_C(τ) = 3.161 V (was 3.151,
  the half-step delay), 6.321 / 8.647 V (was 6.319 / 8.646), i_R(0+) = 4.950 mA (= 5 mA·e^-0.01;
  was 4.975). The prose that explained the delay now says the restart removes it (the
  tutorial's error at τ: 3.1e-7 V, 3.1e-9 V at dt/10).
- `engine='auto'` without dt prints `[dsed]` (getting-started, AI_GUIDE 3.1, 3.3, 6.2); the
  route sentences in AI_GUIDE (bullet in section 1, engine table, migration table) name the
  exact engine for piecewise-linear circuits.
- AI_GUIDE buck ripple I_L 0.601 A pp (analytic 0.600; was 0.589); DCM sweep 15.75 V
  (analytic M = 2/(1+sqrt(1+4K/D²)) gives 15.74 V; was 15.79); reverse recovery 499 turn-offs
  (the 500th sits at t_end, whose sample is the left limit), 0.986 W.
- AI_GUIDE 3.4 loss table moved by 0.2 %: Q1 cond 0.905 -> 0.907, sw 5.034 -> 5.035, D1 2.453 ->
  2.451 W. AGAINST THE EXACT ENGINE (0.90486 / 5.03117 / 2.45585) the release pwl at dt = 1e-7
  is FURTHER off than the old one: Q1 cond rel err 3.4e-5 -> 2.0e-3, D1 -1.0e-3 -> -2.1e-3,
  sw +5.1e-4 -> +7.9e-4 (at dt = 2e-9 the release run converges to the exact values; the old
  sw converged to 5.0349, 7e-4 off). Same root as F-5-4 / F-5-7 (commutation inside a step,
  sample-based integration). Not parked: the fence is an illustration with no truth tolerance.
- AI_GUIDE 6.1 charge check: annotated `<!-- doctest: xfail reason="release finding F-5-4 ..." -->`
  (the runner's strict per-block xfail); its fence shows what it prints (199.1 / 199.4 mC).

## FINDINGS

Each was checked against an independent truth before parking. "New" = release defaults on this
tree; "old" = the legacy profile (or the value the test pinned).

- **F-5-1** (pwl restart, pathless inductor). `b.add_voltage_source('Vg','g','gnd',15);
  b.add_inductor('L','sw','gnd',1e-3,i0=100); b.add_igbt_level1('QL','sw','gnd','g',1.5,0.05,5.0,
  with_fwd=False); simulate(b,t_end=1e-6,dt=1e-9,engine='pwl')`. Truth: no conduction path —
  an impulse; the voltage-sanity guard must say so (it does with `discontinuity_restart=False`:
  -2e8 V and the "conduction path opens" warning). New: i_L 100 A -> -141.42 A in ONE step with
  v_sw <= 8.57 V (energy 5 J -> 10 J), no reinit record, no warning; same with
  `device_path='legacy'` and with the split off. trbdf2 does the same (-20.7 A). Reading: the
  restart's consistent re-initialisation projects the inductor onto the IGBT's forward branch
  instead of treating the cut set (no `dependent_states` refusal, no impulse record).
- **F-5-2** (design settle detector). `Evaluator(_opaque_buck, period=1e-5, loads=('RL',),
  output='vout', engine='dsed', steady=SteadySpec(method='transient', min_periods=8,
  max_periods=256, tol=1e-6)).evaluate(CCM)` (test_design_evaluation). Truth: the polled run's
  waveform equals the oracle'd one (1.4e-10 V abs at 300 periods); it is settled. New:
  `not_settled`, distance 1.3e-4. Old: ok. With `output_rtol=0`: ok. Reading: under the output
  rule the polled edge lands at kT + 1e-16 and `design._settle_error` np.interp's the OFF-branch
  current (2.4e-8 A, jumps at the edge) across it; needs a design.py change (read the state
  at kT from the right side, or skip algebraic currents).
- **F-5-3** (exact engine, LCP selection drops a 1 ps pulse). Buck of
  test_exact_period_end_pulse with D = 1e-7 (1 ps on-time at 100 kHz):
  `simulate(b, 400*T, engine='dsed', switch_fn=fn)`. Truth: 0.987·D·Vin = 2.37e-6 V — the ratio
  BOTH selectors give at D = 1e-4 and 1e-2 (2.3696e-3, 0.23698 V), and the settle gives at
  1e-7. New (`mode_selection='auto'` -> lcp): v_out 2.4000e-8 V = the OFF leakage only, the
  pulse lost. Old (settle): 2.3697e-6 V. Same with `output_rtol=0`. (trbdf2 also loses it:
  2.4e-8, pre-existing.) Through Evaluator(pss) it shows as a PeriodicityError (49 %).
- **F-5-4** (event jump stored without its right limit). `_bridge(0.7)` of
  test_diode_state_reconstruction (325 V bridge, V_th = 0.7, 1 mΩ source):
  `simulate(b, 0.06, engine='pwl', dt=1e-6)` and `simulate(b, 0.06)`. Truth: ∫(i_D1+i_D2) =
  the DC-side charge from node voltages, 0.169880 C (agrees across pwl / dsed / trbdf2 to 1e-9).
  New: trapezoid of the diode trace 0.170717 (pwl, +4.9e-3), 0.169672 (exact, -1.2e-3); D2
  i_avg 0.1706 A vs 0.1611 (exact). Old: 3e-12 (pwl), 1.6e-4 (trbdf2). Reading: at a V_th
  turn-on the current jumps 0 -> 1.4 V / 3 mΩ; the stored trace keeps only the left limit at
  the event (pwl: the event is inside a step and not sampled at all), so integrating the trace
  ramps the jump. Also the AI_GUIDE 6.1 charge check (1.3e-3 > 1e-3).
- **F-5-5** (V_th = 0 commutations between samples). Same bridge at V_th = 0, pwl dt = 1e-6:
  charge closes to 2.6e-5 (O(dt²): 3.9e-7 at dt/10); test_diode_forward_drop's bridge: charge
  3.6e-6, power 2.2e-6 (5e-8 at dt/10). Old: 3e-12 / 4e-13 — the trapezoid-integrator
  identity (commutations snapped to the grid, integrator = quadrature). The waveform agrees
  with the exact engine; the sample-trapezoid can no longer be exact. Possibly a test-estimator
  limit rather than an engine defect — parked because the tolerance cannot be met without
  loosening; the orchestrator decides.
- **F-5-6** (exact engine: summary peak vs trace). `_bridge(0.7)`, `simulate(b, 0.06)`
  (-> dsed; also `engine='dsed', mode_selection='legacy', output_rtol=0`):
  `device_loss_summary` i_peak = 466.664 A (= 1.4 V / 3 mΩ, the right limit at a turn-on)
  while max|result.i('BR__D1')| = 463.81 A and the source's 463.81 A (output_rtol=0: 253.99 A).
  trbdf2 (old auto): all three 466.665 A. Same root as F-5-4; pre-existing on the exact
  engine, exposed by the route.
- **F-5-7** (in-step commutation duty). Boost of test_diode_state_reconstruction, R = 500 Ω
  (DCM), pwl dt = T/100: D1 duty_conducting 0.21713 (V_th = 0) / 0.21543 (0.7). Truth (exact
  engine, trbdf2 rtol 1e-7 agree to 1e-6): 0.22177 / 0.22010. Old pwl: 0.2203 / 0.2187. dt/10:
  0.22123 (first order). Reading: the right-hold duty over grid samples cannot credit the part
  of a step before a commutation the restart now places inside it; `losses._time_fraction`
  would need the event times.
- **F-5-8** (recovery V_R on the auto route). Same boost, `simulate(b, 3e-3, switch_fn=pwm)` +
  `device_loss_summary(..., diode_specs={'D1': {'Q_rr': 100e-9}})`: "60 of 299 (CCM) / 258 of
  300 (DCM) turn-offs happen on a step longer than t_rr" RuntimeWarning; CCM E_sw 6.775e-4 J
  vs 6.950e-4 J (pwl dt = 10 ns), 6.933e-4 (trbdf2 rtol 1e-7). Old auto (trbdf2): no warning,
  6.898e-4. The warning's advice (smaller dt_max) has no keyword on the exact route.
- **F-5-9** (missed commutation, floating-DC bridge). test_diode_state_reconstruction's
  floating-DC bridge, `simulate(b, 0.06, engine='pwl', dt=1e-6)`: RuntimeWarning "BR__D1 ... OFF
  while forward-biased above V_th = 0.7 V on 3 samples, by up to 0.229 V". Quiet with
  `mode_selection='legacy'` and in the legacy profile; with only the restart off it still warns
  (2 samples). Reading: the pwl LCP selection ('auto') misses a turn-on where the floating DC
  side is held only by leakage. (V_dc floating vs grounded differs by 0.10 V new, 0.17 V legacy
  at dt = 1 us — not conclusive alone.)
- **F-5-10** (exact engine drops an interrupted inductor current). `b.add_voltage_source('Vin',
  'in','gnd',48); b.add_inductor('L','in','sw',1e-3); b.add_switch('S','sw','gnd',1e3,1e-9);
  simulate(b, 3e-4, switch_fn=make_pwm_switch_fn(1e4,0.5,0,1))` (auto -> dsed). Truth: the
  interrupted 2.4 A must go somewhere — an impulse (pwl: 1.16e5 V spike) or a recorded
  reinit with its energy. New: i_L 2.3999 A -> 4.8e-8 A at each turn-off, v_sw max 48 V,
  2.88 mJ per turn-off gone, no reinit record, no warning. Old auto (trbdf2): the
  voltage-sanity warning. Same for `form='conductance'` and `mode_selection='legacy'` on dsed:
  pre-existing exact-engine behaviour (event projection), exposed by the route.

## For bucket 0 / the orchestrator

- `python/tests/oracles/lcp_pulsim_bridge.py` builds "the conductance form" (its docstring) but
  passes no `form=`; it should pass `form='conductance'` (worked around with a local fixture in
  test_mode_selector.py; test_mode_shadow uses the same families on the branch form and passes).
- `benchmarks/tools/next_defaults.legacy()` reads `dt` from `args[1]`, which is `t_end` in
  `simulate(b, t_end, ...)`: an `engine='auto'` call with a positional t_end routes to pwl and
  then refuses for lack of dt. Worked around by passing the legacy route's engine explicitly.
- The branch form's refusal names the drop `V_F` while the keyword is `V_f`.
- `AI_GUIDE.md` and the docs hold more route-dependent prose than the fences checked here
  (e.g. 6.2's "15.4 mV without dt" ripple sentence was not re-measured); a docs pass belongs to
  the release's docs task.

## Deleted / skipped

None.
