# R2 bucket 4 — notes

Branch `release/r2-bucket-4` (from `origin/release/core-next-defaults`), worktree
`/Users/lgili/pulsim_wt/p3-events`, extension `so_release_r1`. Baseline: the bucket's 20 files
gave exactly the 77 failures listed in R2_BUCKETS.md (same set). After: 0 failed, 9 xfailed
(strict, findings: 8 of the 77 + 1 new split-out test), every other test passes.

Counts per rule (77 failures): (a) 12 · (b) 49 · (c) 1 · (d) 6 · findings parked 9 (the
two overhead tests also got a (d) route fix before parking).
(Some tests got two treatments, e.g. a legacy pin for a counter plus a release-default
physics check; each is counted once, under its main rule.)

## Per file

| file | fails | rule | what changed |
|---|---:|---|---|
| test_fixed_step_discontinuity_order.py | 18 | (b) 17, (d) 1 | every "without the rule" measurement pins `OFF = dict(discontinuity_restart=False)`; the with-rule sides keep their exact references and pass. `test_substep_correction_requires_the_restart_rule`: the refusal now needs an explicit `discontinuity_restart=False` (the rule is on by default). |
| test_restart_rule_integration.py | 12 | (b) 11, (d) 1 | off sides pin `discontinuity_restart=False`. Gate-edge test: with the default event split the off-grid duty edge keeps its sample (12 -> 12) and the on-grid period-start edge moves one sample (49 -> 50); asserted exactly. |
| test_newton_roundoff_floor.py | 9 | (a) 8, (d) 1 | expectation 28.754 was the restart-less start's half-step deficit (199.5/200 of the truth). Now the linearised closed form per case, `100 (1 - v_D)(1 - e^-x)/x`; tolerance 1e-3 -> 1e-5 (measured max 4.5e-6, at dt = 1e-12 s). Stall flag (100 uH, 1e-11 s) flips False -> True (roundoff luck of the new start, 7 acceptances). |
| test_dsed_exact_propagator.py | 5 | (b) 5 | `mode_guard` diagnostics exist only under `mode_selection='legacy'`: the guard counters pin it; the boost / half-wave waveforms are also asserted on the release default (same trbdf2 truth, same tolerance, pass). Synchronous-buck step counter pins `output_rtol=0` (event grid). |
| test_release_uniform_rules.py | 5 | (b) 5 | `today` fixture pins `_plan.UNIFORM_ENGINE_RULES = False` (LEGACY_PLAN). |
| test_exact_branch_form.py | 4 | (b) 1, findings 3 | PSFB conductance twin: YAML `g_on` without `form:` became branch (R_on 1e-6) since the YAML flip; now `form: conductance`. The ideal PSFB meets the twin on the release default (24.07106 vs 24.07142, rel 1.5e-5 < 1e-4). F-4-1 (2 tests), F-4-3 (1 test). |
| test_switch_conductance_today.py | 4 | (b) 3, (a) 1 | file subject = the conductance form: `add_switch(..., form="conductance")`; pwl ringing test pins `discontinuity_restart=False` (its +-397 V; with the rule the same defect rings at 958 V). Closed control: RL closed form, abs 1e-5 -> 2e-6 (measured 9.3e-7; the restart-less start was 1.06e-5 low). |
| test_exact_guard_transient.py | 3 | (c) 1, (b) 1 (+1 same test) | PSFB zero-band: `undriven='closed'` (the run means every switch closed). Forward converter: guard counter pinned `mode_selection='legacy'`; waveform vs trbdf2 on the release default passes unchanged. |
| test_recorder_device_readings.py | 3 | findings 3 | F-4-4 (pwl cases only; `request.applymarker` strict xfail). |
| test_loss_port_seams.py | 2 | (d) 1, (b) 1 | heat-pending split moves 250.3 -> 251.3 us (the gate of a step is sampled at its start, so the 250 us edge's energy is booked at 251 us); resume identity holds. Stateful byte pins (a regression record of e2e55a71, not a truth) pin `discontinuity_restart=False`; the loss-port-only-heats and lost-stamp checks run on the release default. |
| test_phase2_review_fixes.py | 2 | (b) 2 | both tests need a Newton failure as the trigger; the release rule converges the rectifier (0 retries; completes at `max_dt_halvings=0`). Trigger runs pin `discontinuity_restart=False`. |
| test_simulate_overhead.py | 2 | (d) + finding 2 | the "trbdf2" script names `engine="trbdf2"` (no-dt linear routes to the exact engine). Ceilings NOT raised: F-4-6. |
| test_circuit_params.py | 1 | (d) 1 | the Switch row edits the branch form's `R_on` (1e-3 -> 5e-4); the conductance `g_on` keeps a legacy row `Switch[conductance]` (form pinned). Coverage kept. |
| test_control_graph_export_c.py | 1 | finding 1 | bit-for-bit export check kept (passes); `n_fires == 10_000` moved to its own strict xfail test (F-4-7); the original keeps a `>= 9_999` bound. |
| test_dc_machine.py | 1 | (a) 1 | the trapezoid's discrete energy identity is read past the 3 SDIRK start steps: 1e-8 -> 1e-11 (measured 7.7e-15; whole run 1.45e-7; restart-less 4.2e-9). |
| test_engine_default_routing.py | 1 | (d) 1, (a) | engine_used `dsed`, route reason None; closed-form check 1e-3 -> 1e-11 (measured 7.3e-14). |
| test_ideal_losses.py | 1 | (a) 1 | duty vs the gate's 0.4 / 0.6 (was 0.399 / 0.601), same 1e-12. |
| test_m3c_fast_svm.py | 1 | (b) 1 | the cost/heuristic terminal invariance is the trapezoid's; the rule restarts after internal mask changes too (0.51 % apart at dt, 0.064 % at dt/8; the rule is the more accurate, 1.05 % vs 1.63 % from a dt/8 reference). The test patches `pulsim.simulate` with `discontinuity_restart=False` (model code takes no options). |
| test_nonlinear_capacitor.py | 1 | (a) 1 | Q = I t from sample 1 (was I (t - h/2) past sample 10): 1e-6 -> 1e-9 (measured 1.0e-10). |
| test_snapshot_resume.py | 1 | (b) 1 (+a) | the initial_state defect pinned legacy (2.26e-4); the release continuation now asserted `0 < err < 1e-8` (measured 5.8e-10). |

## Tolerances tightened

| test | old | new | measured |
|---|---|---|---|
| newton roundoff floor (9 cases) | rel 1e-3 (vs 28.754) | rel 1e-5 (closed form) | 4.5e-6 |
| switch conductance closed control | abs 1e-5 | abs 2e-6 | 9.3e-7 |
| dc machine energy identity | 1e-8 (whole run) | 1e-11 (past step 3) | 7.7e-15 |
| default routing RC closed form | 1e-3 | 1e-11 | 7.3e-14 |
| nonlinear capacitor Q = I t | 1e-6 (past sample 10, shifted ref) | 1e-9 (from sample 1) | 1.0e-10 |

## Legacy pins (option, reason)

- `discontinuity_restart=False`: fixed-step discontinuity order + restart rule integration
  (the "off" sides), switch-conductance pwl ringing, loss-port byte pins, phase-2 failing-step
  triggers, M3C terminal invariance, snapshot initial_state defect.
- `mode_selection='legacy'`: the settle's `mode_guard` counters (exact propagator, guard transient).
- `output_rtol=0`: the exact engine's event-grid step counter (synchronous buck).
- `form='conductance'` / YAML `form: conductance`: switch-conductance file subject; PSFB twin;
  `Switch[conductance]` with_value row.
- `_plan.UNIFORM_ENGINE_RULES = False`: the `today` tests of test_release_uniform_rules.py.

## FINDINGS

- **F-4-1** refusal names the wrong cause. `b = CircuitBuilder(); add_voltage_source("V","a","gnd",10); add_diode("D","a","gnd", form="branch", R_on=0.0, g_off=0.0); add_resistor("R","a","gnd",1); simulate(b, 1e-6, engine="dsed")`. Truth / old (`mode_selection='legacy'`): `UnsupportedError(feature='ideal_voltage_loop', device='D')`, "closes a loop of voltage sources ... (loop: D -> V)". New: `feature='consistent_mode'`, "no conduction state ... is consistent (secondary ray) ... a current source into blocking diodes, an inductor with no path: add the missing path" — a wrong hint. Same for the clamp onto a 3 V source. Cause (reading): the 'auto' selector's LCP refusal fires before the structure check that names the loop. Tests: `test_an_ideal_diode_forward_across_a_source_is_refused_by_name`, `test_an_ideal_clamp_onto_a_lower_source_is_refused_by_name`.
- **F-4-2** (no failing test of its own; seen fixing the PSFB twin) branch form with a small R_on refused under 'auto'. `examples/phase_shift_full_bridge.yaml` with every `g_on: 1.0e3` replaced by `g_on: 1.0e6` (now branch, R_on 1e-6), `simulate(lc, 2e-3, engine="exact", switch_fn=make_phase_shift_full_bridge_fn(100e3, pi/2, ..., 100e-9), rtol=1e-9, atol=1e-12)`. Truth: the conductance twin 24.0714 V; old (`mode_selection='legacy'`): 24.07141 V; new: `UnsupportedError` "mode selection at t = 3.390e-04 s: no conduction state of the diodes (...) is consistent". At g_on 1e3 (the shipped value) 'auto' runs (24.06614 V). Cause (reading): the LCP selector's consistency test is scale-sensitive at R_on 1e-6.
- **F-4-3** the ideal PSFB with C_oss aborts under 'auto'. `test_an_ideal_body_diode_across_its_closed_switch_does_not_chatter` (the shipped PSFB, all devices branch R_on = 0, C_oss / C_j added), `simulate(lc, 3e-5, engine="exact", switch_fn=..., rtol=1e-9, atol=1e-12)`. Truth / old (`mode_selection='legacy'`): runs to 30 us. New: `DiodeModeInconsistentError` "diode 'D1 (branch #12)' is ON while carrying -2.205 A at t = 1.4987e-05 s (119 samples break the diode law)". `output_rtol=0` does not change it: the cause is the mode selection, not the output rule.
- **F-4-4** recorder window integrals on pwl across restart-rule edges. `test_recorder_device_readings._buck()`, `simulate(b, 1e-3, engine="pwl", dt=1e-6, switch_fn=fn, recorders=[Recorder("e", ["p(V)","p(S1)","p(D1)","p(R)","p(L)","p(C)"], keep="none", windows=[(A, B)]), ...])`. Truth: supplied - dissipated = stored-energy change (Tellegen still holds to 1e-14). Old (restart off): residual 1.4e-5 of supplied; new: 3.6e-2 (the reactive integrals alone are 3.6 % off). Charge windows 4e-4 (buck) / 4.5e-3 (rectifier) vs tolerance 2e-4. Cause: with the rule an edge's stored sample is the left limit (old mode) and the step after it runs the new mode; the trapezoid of samples smears the jump over one step (the same off-by-h*dv*i/2 per edge appears in a plain `np.trapezoid` over `res.v`/`res.i`, 3.9 % on this buck). The recorder needs the edge's right-side value (the exact engine's `event_right`) on pwl. Tests: 2 charge-window cases + 1 energy balance, pwl only.
- **F-4-5** (pre-existing, not from the flip; no test of mine fails on it) trbdf2 loss port books ~1000x the switching energy at tight tolerance with a nonlinear C_oss. `test_loss_port_seams._stateful_with_loss(True, True)`, `simulate(b, 4e-4, engine="trbdf2", switch_fn=fn, rtol=1e-6, atol=1e-9)`. Truth: E_on = E_off = 1e-4 J per edge, 40 edges -> 4e-3 J (pwl 3.93e-3). Got: heat_injected 7.40 J, T_j 6806 C; legacy options give the same (7.40 J). At the default rtol/atol the test value is 3.93e-3. Without the C_oss (`_builder()`) trbdf2 is right at every tolerance. Cause (reading): the event's energy is read at a resolved C_oss discharge current spike (table scaling with i) or booked per rejected/probe step.
- **F-4-6** simulate() fixed overhead past its ratchet. `test_simulate_overhead` script: pwl 3056 / trbdf2 3859 bytecodes per warm call on the release defaults; 2950 / 3767 with every legacy option pinned; ceilings 2890 / 3760 (measured 2834 / 3683 on 2026-11-19 before the release merge). Both the merged release options and the default resolution (mode_selection auto census etc., ~+100) add. Not raised (rule: never raise to pass).
- **F-4-7** `control_tick='state'` loses the last tick. `test_control_graph_export_c._pi_pwm` exported as a C block on `_rc()`, `simulate(b, 10*dt, dt=dt)`: ticks are due at k*dt < t_end (k = 0..9) -> 10 fires; got 9 (10000 steps -> 9999). `control_tick='label'` fires 10. The last step (t_end - dt, t_end] runs on the previous hold: an n-step run differs from a longer run's prefix exactly when the drive changes at tick n-1. Cause (reading): the state-time due test uses the label-time bound (tau < t_end at the step END) one step late.

## Deleted / skipped

None. No test deleted or skipped. The C-block exact tick count moved into a new strict-xfail
test in the same file (F-4-7); the original keeps a `>= 9_999` bound and its bit-for-bit check.

## From bucket 0

Nothing needed. (The M3C invariance test patches `pulsim.simulate` locally because
`projects/inverters/m3c_3phase/m3c_3phase_model.py` takes no simulate options; if bucket 0 wants
the model runners to accept `**sim_kw`, the patch can become a keyword.)
