"""Generate FINDINGS.md: the release's engine findings, one index for the fix agents.

Usage: findings_index.py JUNIT_XML [JUNIT_XML ...]  (writes ../FINDINGS.md)

The table below is transcribed from r2/bucket-N.md (each bucket's own measurements) plus the two
findings of the output-rule cost agent (F-X-*). The parked tests are NOT transcribed: they are read
from pytest JUnit XML runs (an xfailed or failed test whose reason / message names
"release finding F-..."), so the index shows what the suite actually parks.
"""
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

GROUPS = {
    "A": "impulsive events / the restart rule",
    "B": "event samples and result integrals",
    "C": "the new mode selection (`mode_selection='auto'`)",
    "D": "trbdf2",
    "E": "control tick (`control_tick='state'`)",
    "F": "the exact route / output rule",
    "G": "thermal summary on branch-form devices",
    "H": "call overhead",
}
SEV = {"W": "wrong answer silently", "R": "refusal / abort of something that used to run", "C": "cosmetic"}

# id, group, severity, one line, smallest reproducing call, truth, old, new
F = [
    ("F-2-1", "G", "W", "device_thermal_summary ignores conduction_specs on a branch-form diode",
     "`test_conduction_model._dc_diode_circuit()`, `simulate(b, 2e-3, 1e-6, start_from_dc_op=True)`, `device_thermal_summary(..., conduction_specs={'D': {'V_f0': 0.7, 'r_on': 0.01}})['P_cond_avg']`",
     "7.991 W", "7.991 W", "0.0998 W"),
    ("F-2-2", "B", "W", "pwl restart steps integrated as sample trapezoids: diode charge first order in dt",
     "`_snubbed_boost(0.0, 1e-9)` (test_switch_state_record), `engine='pwl', dt=T/400`, trapezoid of `r.i('D1')` vs the load's KCL charge",
     "charge conserved (0.0106361 C)", "closes to 6e-12", "1.26e-3 off at T/400 (3.4e-5 at T/40000); boost DCM D1 charge -5.6e-3, P_avg -0.8 %"),
    ("F-2-3", "C", "W", "pwl LCP holds a bridge diode OFF while forward-biased (warned only)",
     "`_reference_bridge(0.0)` (325 V bridge), `engine='pwl', dt=1e-6`, `device_loss_summary`",
     "no rule violation", "no warning", "BR__D1 OFF forward-biased by up to 1.72 mV on 27 samples"),
    ("F-2-4", "F", "W", "thermal summary's junction temperature off on the exact engine's default output grid",
     "`_boost(0.7, 20.0)`, `engine='dsed'` without dt, `device_thermal_summary` Foster 1000 K/W 0.1 ms; ratio (T_j_avg-25)/R_th/P_cond_avg",
     "0.9946 (Q) / 0.9943 (D1) (pwl; output_rtol=0 0.9942 / 0.9945)", "0.9942 / 0.9945", "0.9776 / 1.0109"),
    ("F-3-1", "A", "W", "the event split is not rounding-stable at a diode turn-on",
     "test_mode_pwl `_sim(_perturbed('bridge1_vf07', 5), 0.04, 1e-5)` vs k = 0",
     "exact engine dt = 1e-5 (both runs 3.4e-3 V off it after 13.71 ms)", "runs agree to 2.3e-11 V", "runs differ by 3.07e-4 V at 13.71 ms (split off: 2.4e-11)"),
    ("F-3-2", "C", "C", "an LCP refusal names 'D (branch #1)' as `.device`",
     "V 10 V a-gnd, `add_diode('D','a','gnd',form='branch')`, R a-c 1, C c-gnd 1 uF, `simulate(b, 1e-4, dt=1e-6, engine='pwl')`",
     "`.device == 'D'`", "'D'", "'D (branch #1)'"),
    ("F-3-3", "A", "W", "default fixed step turns a flyback's MOSFET off INTO its body diode",
     "test_saturable_transformer `_flyback(False, 1.0)`, 100 kHz D 0.5, `simulate(b, 1e-3, dt=1e-8)`",
     "exact: i_p -> 1.6e-7 A at turn-off, peak 16.382 A, v_out 16.365 V", "i_p -15.48 A, v_out 16.80 V", "i_p 14.83 -> -20.92 A, v_out 17.15 V, no reinit record"),
    ("F-3-4", "D", "R", "trbdf2 aborts at h_min when switch AND diode are branch form",
     "test_saturable_transformer `test_engines_agree_below_saturation` circuit, `engine='trbdf2', rtol=1e-6, atol=1e-9`",
     "pwl dt = 1e-8: 15.3095 V", "15.3104 V", "SimulationAborted at t = 0.41 ms"),
    ("F-3-5", "E", "C", "a control-graph build refusal arrives as SimulationAborted, not ControlGraphBuildError",
     "`simulate(rc, 1e-5, engine='pwl', dt=1e-7, step_observer=MixedDomainBlockChain(rate=Rate(5e-8))...)`",
     "ControlGraphBuildError (ValueError)", "ControlGraphBuildError", "SimulationAborted (RuntimeError)"),
    ("F-3-6", "F", "C", "numeric segments' step counters read 0 under the output rule",
     "test_dsed_auto_integrator pulse RC, `engine='dsed', b_extra_fn=zero, allow_implicit_segments=True`",
     "n_rk45_steps = 2022 (documented counters)", "2022", "0 (same samples and values)"),
    ("F-3-7", "A", "R", "restart rule aborts the saturable-primary PSFB at dt = 50 ns (diode-mask chatter)",
     "test_newton_stall_and_mask_chatter `_run('saturable', 50e-9)`",
     "exact engine 6.8577 V (linear primary)", "6.913 V", "SimulationAborted at 61.55 us"),
    ("F-4-1", "C", "C", "LCP refuses an ideal voltage loop as consistent_mode with a wrong hint",
     "V 10 V a-gnd, `add_diode('D','a','gnd',form='branch',R_on=0,g_off=0)`, R 1, `simulate(b, 1e-6, engine='dsed')`",
     "UnsupportedError feature='ideal_voltage_loop', device='D'", "ideal_voltage_loop", "feature='consistent_mode', 'add the missing path'"),
    ("F-4-2", "C", "R", "PSFB with branch-form devices at R_on = 1e-6 refused at 339 us under 'auto'",
     "`examples/phase_shift_full_bridge.yaml` with g_on 1e6, `simulate(lc, 2e-3, engine='exact', switch_fn=..., rtol=1e-9, atol=1e-12)`",
     "conductance twin 24.0714 V", "24.07141 V", "UnsupportedError at t = 3.390e-4 s"),
    ("F-4-3", "C", "R", "ideal PSFB with C_oss aborts at 14.99 us under 'auto' (D1 ON at -2.2 A)",
     "test_exact_branch_form `test_an_ideal_body_diode_across_its_closed_switch_does_not_chatter`, `simulate(lc, 3e-5, engine='exact', ...)`",
     "runs to 30 us", "runs to 30 us", "DiodeModeInconsistentError at 1.4987e-5 s"),
    ("F-4-4", "B", "W", "pwl recorder window integrals wrong across restart-rule edges",
     "test_recorder_device_readings `_buck()`, `simulate(b, 1e-3, engine='pwl', dt=1e-6, recorders=[Recorder(..., windows=[(A, B)])])`",
     "energy balance residual ~0 (Tellegen 1e-14)", "1.4e-5 of supplied", "3.6e-2; charge windows 4e-4 / 4.5e-3 vs 2e-4"),
    ("F-4-5", "D", "W", "trbdf2 loss port books ~1000x the switching energy with a nonlinear C_oss (pre-existing)",
     "test_loss_port_seams `_stateful_with_loss(True, True)`, `simulate(b, 4e-4, engine='trbdf2', rtol=1e-6, atol=1e-9)`",
     "4e-3 J (pwl 3.93e-3)", "7.40 J (legacy options too)", "7.40 J, T_j 6806 C"),
    ("F-4-6", "H", "C", "simulate() fixed overhead past its ratchet",
     "`test_simulate_overhead` script", "ceilings pwl 2890 / trbdf2 3760 bytecodes", "2834 / 3683", "3056 / 3859 (2950 / 3767 with every legacy option)"),
    ("F-4-7", "E", "W", "control_tick='state' never fires the tick at t_end - dt",
     "test_control_graph_export_c `_pi_pwm` C block on `_rc()`, `simulate(b, 10*dt, dt=dt)`",
     "10 fires", "10", "9 (10000 steps -> 9999)"),
    ("F-5-1", "A", "W", "pwl restart turns a pathless inductor's 100 A into -141 A in one step",
     "`add_voltage_source('Vg','g','gnd',15); add_inductor('L','sw','gnd',1e-3,i0=100); add_igbt_level1('QL','sw','gnd','g',1.5,0.05,5.0,with_fwd=False); simulate(b,t_end=1e-6,dt=1e-9,engine='pwl')`",
     "an impulse: voltage-sanity warning", "-2e8 V + 'conduction path opens' warning", "i_L -141.42 A, energy 5 J -> 10 J, no record, no warning"),
    ("F-5-2", "F", "R", "design settle detector reads 1.3e-4 on a settled run under output_rtol",
     "`Evaluator(_opaque_buck, period=1e-5, loads=('RL',), output='vout', engine='dsed', steady=SteadySpec('transient', 8, 256, tol=1e-6)).evaluate(CCM)`",
     "settled (1.4e-10 V)", "ok", "not_settled, distance 1.3e-4"),
    ("F-5-3", "C", "W", "exact engine with LCP selection drops a 1 ps gate pulse",
     "test_exact_period_end_pulse buck D = 1e-7, `simulate(b, 400*T, engine='dsed', switch_fn=fn)`",
     "v_out 2.37e-6 V", "2.3697e-6 V", "2.4000e-8 V (OFF leakage only)"),
    ("F-5-4", "B", "W", "a current that jumps at an event is stored without its right limit",
     "test_diode_state_reconstruction `_bridge(0.7)`, `simulate(b, 0.06, engine='pwl', dt=1e-6)` and `simulate(b, 0.06)`",
     "charge 0.169880 C", "3e-12 (pwl)", "+4.9e-3 (pwl), -1.2e-3 (exact); D2 i_avg +6 %"),
    ("F-5-5", "B", "C", "V_th = 0 commutations between samples: sample-trapezoid charge O(dt^2) (estimator limit)",
     "`_bridge(0.0)`, `engine='pwl', dt=1e-6`",
     "charge identity", "3e-12", "2.6e-5 (3.9e-7 at dt/10; waveform right)"),
    ("F-5-6", "B", "W", "exact engine's loss summary reports a peak no stored sample holds",
     "`_bridge(0.7)`, `simulate(b, 0.06)`, `device_loss_summary` i_peak",
     "466.665 A (trbdf2)", "466.665 A (trbdf2 route)", "466.664 A vs trace max 463.81 A"),
    ("F-5-7", "B", "W", "pwl duty loses the part of a step before an in-step commutation",
     "test_diode_state_reconstruction boost R = 500, `engine='pwl', dt=T/100`, D1 duty_conducting",
     "0.22177 (V_th 0)", "0.2203", "0.21713"),
    ("F-5-8", "F", "W", "auto -> exact: the step after a diode turn-off is longer than t_rr (warned)",
     "same boost, `simulate(b, 3e-3, switch_fn=pwm)`, `device_loss_summary(..., diode_specs={'D1': {'Q_rr': 100e-9}})`",
     "E_sw 6.950e-4 J (pwl dt 10 ns)", "6.898e-4 (trbdf2), no warning", "6.775e-4 + warning with no keyword to act on"),
    ("F-5-9", "C", "W", "floating-DC bridge on pwl misses a commutation under LCP (warned)",
     "test_diode_state_reconstruction floating-DC bridge, `simulate(b, 0.06, engine='pwl', dt=1e-6)`",
     "no forward-biased OFF diode", "quiet", "BR__D1 OFF up to 0.229 V above V_th"),
    ("F-5-10", "A", "W", "exact engine silently drops an interrupted inductor current",
     "`add_voltage_source('Vin','in','gnd',48); add_inductor('L','in','sw',1e-3); add_switch('S','sw','gnd',1e3,1e-9); simulate(b, 3e-4, switch_fn=make_pwm_switch_fn(1e4,0.5,0,1))`",
     "an impulse or a recorded reinit", "voltage-sanity warning (trbdf2 route)", "i_L 2.4 A -> 4.8e-8 A, 2.88 mJ per turn-off gone, no record"),
    ("F-6-1", "A", "W", "steady_state's default probe_scale is inside the restart engine's per-period round-off",
     "`steady_state(buck, period=1e-5, dt=2e-8, switch_fn=grid_gate)` (test_steady_state `_buck`)",
     "residual -> 0 (probe_scale=0.1: 2.0e-11)", "1.8e-10", "1.9e-8"),
    ("F-6-2", "B", "W", "exact engine stores the PRE-event value at a diode turn-on",
     "`simulate(_bridge(), t_end=0.1)` (test_loss_window)",
     "i_avg 0.161005 A (pwl dt 1e-7)", "identity held (trbdf2 route)", "0.160684 (-0.21 %)"),
    ("F-6-3", "G", "W", "= F-2-1 (thermal summary conduction model misses a branch-form diode)",
     "`device_thermal_summary(..., conduction_specs={'D1': {'V_f0': .7, 'r_on': .02}})` on a pwl buck",
     "2.449 W", "2.449 W", "0.0186 W"),
    ("F-6-4", "B", "W", "loss port's integral of v*i over restart SDIRK steps first order, 10x constant",
     "test_device_losses_variable_step RL chopper, `engine='pwl', dt=1e-7`",
     "closed-form E_cond", "rel 1.8e-4 (S1) / 2.4e-4 (D1)", "1.86e-3 / 2.0e-3 (tol 1.5e-3)"),
    ("F-6-5", "B", "W", "restart edge row solved with the pre-edge source value; res.i(source) evaluates at t",
     "pulse current source 0->1 A at 1e-4..3e-4 into 1 mOhm + 5 Ohm, `engine='pwl', dt=1e-6`",
     "one value per row", "row 3e-4: Ip 0, Rsq 0", "row 3e-4: Ip 0.0, Rsq 1.0"),
    ("F-6-6", "F", "C", "= F-3-6 (numeric segments' counters 0 under the output rule)",
     "RC + `b_extra_fn`, `engine='exact', allow_implicit_segments=True`",
     "steps counted", "n_rk45_steps 1005", "0"),
    ("F-6-7", "F", "W", "exact engine: the trajectory depends on the output grid",
     "`simulate(_paralleled_bridges(10), t_end=1e-4, engine='exact')` (test_exact_route)",
     "i(L0a)(1e-4) = 0.12972284951463489 A (mpmath)", "2.6e-10 off (output_rtol=0)", "6.6e-8 off (default rule, before the cost fix 1e2b90ff); 2.1e-6 at output_rtol=1e-8. "
     "After the merge of the cost fix the default run is within the test's 1e-8 bound (the test passes, so it stays unparked); "
     "the dependence on the grid at output_rtol=1e-8 was NOT re-measured"),
    ("F-X-1", "F", "R", "ten paralleled bridges refused at 19.969 ms under the full profile",
     "`simulate(_paralleled_bridges(10), t_end=0.02, engine='exact')` (test_exact_route)",
     "runs (agrees with pwl dt 1e-7)", "runs (output_rtol=0)", "DiodeModeInconsistentError at 19.969 ms (Db8b OFF forward-biased 0.73 V; "
     "reproduced on the merged release tip with the cost fix, both selections)"),
    ("F-X-3", "F", "R", "the output rule's chord bound does not finish a 500-state RC ladder (counters.py's 88-minute case)",
     "`benchmarks/tools/counters.py` `ladder500_rc` on `engine='dsed', integrator='rk45'` without dt (t_end 2e-5)",
     "a 2e-5 s RC run (pwl / trbdf2 take 0.1 s)", "seconds (old grid)", "> 29 min with the cost fix merged, stopped by PID; "
     "`sample`: dsed::exact_chord_step / ChordWork / ExactLTI (the integrator's own work is not the cost)"),
    ("F-X-4", "F", "C", "`auto` without dt writes 2,000,002 samples for a 2 s DC resistor (triage freeze migration)",
     "`next_defaults_rehearsal._no_dt_run(v1_reader.load('benchmarks/circuits/thermal/electrothermal_resistor_self_heating.yaml'), 'auto')` "
     "(5 V across 1 ohm, t_end 2 s, uic); graded by `reference_status.py` against trbdf2 rtol=1e-8",
     "V(vin) = 5 V from t = 0+ (dsed with the netlist dt agrees, err_rms 0; 4002 samples)",
     "auto routed to trbdf2: 1004 samples, agree",
     "routes to dsed: 2,000,002 samples on a 1 us grid; V(vin) right, but the reference's uic sample "
     "V(vin)(0) = 0 interpolated to 5 V at 2e-5 s spans 20 output samples: err_rms 1.78e-3 > tol 1e-3, err_max 1.0 "
     "(disagree, frozen `auto` in v2_triage.yaml citing this id)"),
    ("F-X-7", "F", "W", "the default output rule does not bound an output that is not a state (a current-fed bridge's bus reads 39.6 V off between samples)",
     "`simulate(B.FAMILIES['bridge3_overlap'].builder(0.0, form='branch'), t_end=0.02, engine='exact')`, bus = v(p) - v(n) "
     "interpolated linearly at 4001 instants (test_exact_route::test_inductor_cut_set_runs_and_agrees_with_pwl)",
     "pwl + restart + split at dt 1e-7 (it moves 6.3e-3 V from dt 2e-7); the exact engine with output_rtol=0 is 0.0107 V from it",
     "0.0107 V (20036 samples: before the merge of the PR line's batch 14s the rule also stored every step end)",
     "39.56 V at t = 0.12 ms of a 173 V range with the default rule (2713 samples); 27.9 V with output_rtol=1e-7 (24033 samples). "
     "The rule's promise covers the states only; a node voltage recovered algebraically is a chord between two stored samples. "
     "The bound on recovered rows is on fix/core-release-f4-chord-bound (not mergeable: three defects found by its verification)"),
    ("F-X-2", "H", "C", "the `auto` routing costs 4745 bytecodes per call",
     "`test_simulate_overhead` no-dt script", "ceiling 3760 bytecodes", "3683 (trbdf2 route)", "5456 (pulsim.exact_route alone 4745)"),
]

# Closed by a fix merged into the release branch: id -> where and the measured result.
# A closed finding leaves the open tables and is listed under "Closed".
FIXED = {
    "F-4-6": "group H, `release/h-call-overhead` 26d059b0 (merged 1b3f7ecc): fixed step 3057 -> 2800 bytecodes "
             "(ceiling 2890); `test_simulate_overhead.py` no longer xfail",
    "F-X-2": "group H, `release/h-call-overhead` 26d059b0 (merged 1b3f7ecc): no-dt call on the exact route "
             "5609 -> 3634 bytecodes (ceiling 3760); gate case `[exact]` added",
    "F-2-1": "merge of the PR line 03c8b67c / a90d84b8 (R3, 9e051abe .. 9f1f0747): its parked test(s) pass "
             "unchanged and the mark is removed: `test_thermal_summary_uses_conduction_model`",
    "F-2-4": "merge of the PR line 03c8b67c / a90d84b8 (R3, 9e051abe .. 9f1f0747): its parked test(s) pass "
             "unchanged and the mark is removed: `test_dsed_junction_temperature_follows_the_step_start_power[500.0]`; `test_dsed_junction_temperature_follows_the_step_start_power[20.0]`",
    "F-3-1": "merge of the PR line 03c8b67c / a90d84b8 (R3, 9e051abe .. 9f1f0747): its parked test(s) pass "
             "unchanged and the mark is removed: `test_rounding_fuzz_gives_one_mode_sequence[bridge1_vf07-default]`; `test_rounding_fuzz_gives_one_mode_sequence[bridge1_vf07-lcp]`",
    "F-3-2": "merge of the PR line 03c8b67c / a90d84b8 (R3, 9e051abe .. 9f1f0747): its parked test(s) pass "
             "unchanged and the mark is removed: `test_h1_forward_biased_ideal_diode_across_a_source_is_refused[auto]`",
    "F-3-3": "merge of the PR line 03c8b67c / a90d84b8 (R3, 9e051abe .. 9f1f0747): its parked test(s) pass "
             "unchanged and the mark is removed: `test_past_the_knee_the_core_runs_away_and_the_output_collapses`",
    "F-3-4": "merge of the PR line 03c8b67c / a90d84b8 (R3, 9e051abe .. 9f1f0747): its parked test(s) pass "
             "unchanged and the mark is removed: `test_engines_agree_below_saturation`",
    "F-3-5": "merge of the PR line 03c8b67c / a90d84b8 (R3, 9e051abe .. 9f1f0747): its parked test(s) pass "
             "unchanged and the mark is removed: `test_a_rate_period_below_the_fixed_step_is_refused[default]`",
    "F-3-6": "merge of the PR line 03c8b67c / a90d84b8 (R3, 9e051abe .. 9f1f0747): its parked test(s) pass "
             "unchanged and the mark is removed: `test_a_mode_without_an_exact_stepper_still_runs_on_auto[default]`",
    "F-3-7": "merge of the PR line 03c8b67c / a90d84b8 (R3, 9e051abe .. 9f1f0747): its parked test(s) pass "
             "unchanged and the mark is removed: `test_at_a_moderate_step_every_primary_agrees`",
    "F-4-3": "merge of the PR line 03c8b67c / a90d84b8 (R3, 9e051abe .. 9f1f0747): its parked test(s) pass "
             "unchanged and the mark is removed: `test_an_ideal_body_diode_across_its_closed_switch_does_not_chatter`",
    "F-4-7": "merge of the PR line 03c8b67c / a90d84b8 (R3, 9e051abe .. 9f1f0747): its parked test(s) pass "
             "unchanged and the mark is removed: `test_the_c_block_fires_on_every_tick_before_t_end`",
    "F-5-2": "merge of the PR line 03c8b67c / a90d84b8 (R3, 9e051abe .. 9f1f0747): its parked test(s) pass "
             "unchanged and the mark is removed: `test_a_polled_dsed_schedule_is_counted_and_its_kpis_are_unaffected`",
    "F-5-3": "merge of the PR line 03c8b67c / a90d84b8 (R3, 9e051abe .. 9f1f0747): its parked test(s) pass "
             "unchanged and the mark is removed: `test_a_short_pulse_counts_wherever_it_sits_in_the_period[1e-07]`",
    "F-5-7": "merge of the PR line 03c8b67c / a90d84b8 (R3, 9e051abe .. 9f1f0747): its parked test(s) pass "
             "unchanged and the mark is removed: `test_boost_duty_agrees_across_engines[500.0-0.7]`; `test_boost_duty_agrees_across_engines[500.0-0.0]`",
    "F-5-8": "merge of the PR line 03c8b67c / a90d84b8 (R3, 9e051abe .. 9f1f0747): its parked test(s) pass "
             "unchanged and the mark is removed: `test_boost_recovery_energy_agrees_across_engines[20.0-0.7]`; `test_boost_recovery_energy_agrees_across_engines[20.0-0.0]`; `test_boost_recovery_energy_agrees_across_engines[500.0-0.7]`; `test_boost_recovery_energy_agrees_across_engines[500.0-0.0]`",
    "F-6-1": "merge of the PR line 03c8b67c / a90d84b8 (R3, 9e051abe .. 9f1f0747): its parked test(s) pass "
             "unchanged and the mark is removed: `test_the_default_probe_finds_the_orbit_to_1e_8`",
    "F-6-3": "merge of the PR line 03c8b67c / a90d84b8 (R3, 9e051abe .. 9f1f0747): its parked test(s) pass "
             "unchanged and the mark is removed: `test_thermal_conduction_cuts_a_flip_step_like_the_loss_summary`",
    "F-6-4": "merge of the PR line 03c8b67c / a90d84b8 (R3, 9e051abe .. 9f1f0747): its parked test(s) pass "
             "unchanged and the mark is removed: `test_event_and_conduction_energy_converge_to_the_closed_form_truth[pwl-kw0-0.0015-0.0015]`",
    "F-6-5": "merge of the PR line 03c8b67c / a90d84b8 (R3, 9e051abe .. 9f1f0747): its parked test(s) pass "
             "unchanged and the mark is removed: `test_res_i_matches_a_series_milliohm`",
    "F-6-6": "merge of the PR line 03c8b67c / a90d84b8 (R3, 9e051abe .. 9f1f0747): its parked test(s) pass "
             "unchanged and the mark is removed: `test_a_numeric_segment_counts_its_integrator_steps`",
}

PAT = re.compile(r"release finding (F-[0-9X]+-[0-9]+)")


def parked(paths):
    out = defaultdict(set)
    for p in paths:
        for tc in ET.parse(p).getroot().iter("testcase"):
            for child in tc:
                text = (child.get("message") or "") + " " + (child.text or "")
                for fid in set(PAT.findall(text)):
                    tag = "xfail" if child.tag == "skipped" else child.tag
                    out[fid].add(f"`{tc.get('classname')}::{tc.get('name')}` ({tag})")
    return out


def main(argv):
    park = parked(argv[1:])
    closed = [r for r in F if r[0] in FIXED]
    open_ = [r for r in F if r[0] not in FIXED]
    by_g = defaultdict(list)
    for row in open_:
        by_g[row[1]].append(row)
    sev = defaultdict(int)
    lines = ["# Release findings — update-number-changing-release", "",
             "GENERATED by `probes/findings_index.py` (data: r2/bucket-N.md + the output-rule cost agent; parked tests: "
             "the JUnit XML of the suite run named below). Do not edit by hand.", "",
             f"Runs read: {', '.join(Path(a).name for a in argv[1:])}", "",
             "Severity: " + "; ".join(f"**{k}** {v}" for k, v in SEV.items()) + ".", ""]
    total = defaultdict(int)
    for g in sorted(by_g):
        for r in by_g[g]:
            total[(g, r[2])] += 1
            sev[r[2]] += 1
    lines += ["| group | " + " | ".join(SEV) + " | all |", "|---|" + "---:|" * (len(SEV) + 1)]
    for g in sorted(by_g):
        lines.append(f"| {g} {GROUPS[g]} | " + " | ".join(str(total[(g, s)]) for s in SEV) + f" | {len(by_g[g])} |")
    lines.append("| all | " + " | ".join(str(sev[s]) for s in SEV) + f" | {len(open_)} |")
    lines += ["", "Bucket-0 items (fixed by the integrator, not findings): `oracles/lcp_pulsim_bridge.py` builds "
              "`form='conductance'` (keyword `form=`); `next_defaults.legacy()` reads `dt` from the third positional "
              "argument; the branch-form refusal names `V_f` for `add_diode`.", ""]
    for g in sorted(by_g):
        lines += [f"## {g} — {GROUPS[g]}", ""]
        for fid, _, s, title, call, truth, old, new in by_g[g]:
            tests = sorted(park.get(fid, ())) or ["none parked"]
            lines += [f"### {fid} [{s}] {title}", "",
                      f"- call: {call}", f"- truth: {truth}", f"- old: {old}", f"- new: {new}",
                      "- parked: " + "; ".join(tests), ""]
    if closed:
        lines += ["## Closed (fix merged into the release branch)", ""]
        for fid, g, s_, title, *_ in closed:
            tests = sorted(park.get(fid, ())) or ["none parked"]
            lines.append(f"- {fid} [{g}/{s_}] {title}: {FIXED[fid]}; parked: " + "; ".join(tests))
        lines.append("")
    unknown = sorted(set(park) - {r[0] for r in F})
    if unknown:
        lines += ["## Parked under an id not in the table", ""] + [f"- {u}: " + "; ".join(sorted(park[u])) for u in unknown]
    Path(__file__).resolve().parent.parent.joinpath("FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(open_)} open findings, {len(closed)} closed, {sum(len(v) for v in park.values())} parked tests, unknown ids {unknown}")


if __name__ == "__main__":
    main(sys.argv)
