# R2 bucket 2 — notes

Branch `release/r2-bucket-2` (from `origin/release/core-next-defaults` d2a15c91), extension
`so_release_r1`. Baseline: `test_switch_state_record.py` alone 52 failed / 63 passed (the map's
52, same tests); the other 15 files run one by one before their fix showed exactly the map's 25
(77 in all). After: the 16 files collect 245 tests: **227 passed, 0 failed, 18 xfailed**
(all 18 are the new strict xfails of the 4 findings: F-2-1 1, F-2-2 13, F-2-3 2, F-2-4 2).

No test deleted or skipped. No shared helper / conftest touched; nothing needed from bucket 0.

## Findings (rule a: parked as `xfail(strict=True)`, expectation untouched)

| id | smallest reproducing call | truth | old (legacy) | new (default) | cause (one line) |
|---|---|---|---|---|---|
| F-2-1 | `test_conduction_model._dc_diode_circuit()` (10 V, `add_diode(g_on=1e3, g_off=1e-9, V_th=0)`, 1 Ω, `simulate(b, 2e-3, 1e-6, start_from_dc_op=True)`), then `device_thermal_summary(..., conduction_specs={"D": {"V_f0": 0.7, "r_on": 0.01}})["P_cond_avg"]` | 0.7·9.99 + 0.01·9.99² = 7.991 W (`device_loss_summary` gives 7.991) | 7.991 W (conductance form) | 0.0998 W | `thermal.py` branch-form path reads v·i of the branch and never consults `conduction_specs` |
| F-2-2 | `_snubbed_boost(0.0, 1e-9)` on `engine='pwl', dt=T/400` — diode charge from the record (`trapezoid(r.i("D1"))` or summary `i_avg`) vs the load's KCL charge `C1·Δv + ∫v/R − Cs·Δv_s` | equal (charge conservation); converged run (T/40000, restart off) 0.0106361 C | closes to 6e-12 (restart-less) | 1.26e-3 off at T/400, 5e-4 at T/4000, 3.4e-5 at T/40000 (first order); boost DCM D1 charge −5.6e-3, P_avg −0.8 % (truth from trbdf2/dsed, which agree to 5e-5) | the restart steps (SDIRK / split sub-steps) are integrated by the reporting as trapezoids between stored samples; form-independent (conductance form identical) |
| F-2-3 | `_reference_bridge(0.0)` (325 V bridge, 1 mΩ, 470 µF ‖ 1 kΩ) on `engine='pwl', dt=1e-6`, `device_loss_summary` | no rule violation (the settle, `mode_selection='legacy'`, has none) | no warning | `BR__D1` OFF while forward-biased by up to 1.72 mV on 27 samples (2 samples, 1.52 mV at V_th 0.7, dt 1e-7) | pwl `mode_selection='auto'` → `'lcp'` decides a diode OFF that the commutation rule check sees forward-biased |
| F-2-4 | `_boost(0.7, 20.0)` on `engine='dsed'` without dt, `device_thermal_summary(thermal_specs={D1, Q: Foster 1000 K/W, 0.1 ms})`, ratio (T_j_avg−25)/R_th/P_cond_avg | ≈ the pwl ratio (0.9946 Q, 0.9943 D1); with `output_rtol=0`: 0.9942 / 0.9945 | 0.9942 / 0.9945 | 0.9776 (Q), 1.0109 (D1); DCM 0.9947 / 1.0049 | the thermal trace on the exact engine's default output-rule grid misplaces the step-start power (P_cond_avg itself unchanged to 1e-11) |

Tests parked: F-2-1 `test_conduction_model.py::test_thermal_summary_uses_conduction_model`;
F-2-2 `test_switch_state_record.py` snubber_closes_the_charge ×4, rectifier_charge (pwl) ×4,
line_inductance_bridge[pwl], reference_bridge_charge_and_source_kcl[0.7-pwl],
switched_branch_charge[pwl] ×2, conduction_loss_agrees[500.0]; F-2-3
reference_bridge_agrees_at_v_th_0_7, reference_bridge_charge_and_source_kcl[0.0-pwl]; F-2-4
dsed_junction_temperature ×2. Each was checked with `--runxfail` to fail on the asserted line.

## Per file

**test_cross_cutting_on_every_engine.py** (3): (b) 1 — the closed run is the legacy pwl/exact
behaviour, pinned `undriven='closed'`, read from `result.diagnostics["undriven"]` (the explicit
policy emits no warning; renamed `..._runs_closed_when_asked`); (d) 2 — the refusal is now
asserted by name on every engine (`..._refused_on_every_engine`, parametrized; was trbdf2 only).

**test_named_diagnostics.py** (1): (c) the undriven switch was incidental, `undriven='closed'`.

**test_branch_form_switch.py** (1): (b) the g_on/g_off refusal names `form='conductance'`;
(d) added: the default `add_diode("Y", a, b)` builds a branch diode.

**test_device_thermal_summary.py** (1): (b) the mask-record path is the conductance switch's
(a branch switch reads its branch current; a record-less result warns and drops it), pinned
`form='conductance'`.

**test_gate_event_counts.py** (1): (b) the snap merge is the label tick's, pinned
`control_tick='label'`; (d) added: the default state tick lands all 11 edges, 0 unresolved,
no warning.

**test_conduction_model.py** (1): F-2-1.

**test_control_tick_state.py** (1): (b) the label run names `control_tick='label'`; (d) the
default equals the state run byte for byte.

**test_induction_machine_mna.py** (1): (a) the 0.75 N·m offset on interval 0 was the old
first-order start; under the restart rule the identity holds on every interval. Tightened: all
intervals < 1e-7 of peak torque (measured 3.7e-8 N·m on the SDIRK intervals 0–2), the rest
1e-7 → 1e-9 of peak (measured 2.0e-11 N·m).

**test_sensitivity_reference.py** (1): (b) the −9.5 .. +2.0 FD record is the restart-less engine's,
pinned `discontinuity_restart=False` (reproduces it exactly); (a) added
`test_fixed_engine_finite_difference_with_the_restart_rule`: FD within 1e-4 of the analytic
sensitivity at relative steps 1e-5..1e-3 (measured 1.7e-5).

**test_adaptive_segments_resume.py** (1): (a) 8.0431 was the restart-less value, itself 2.7e-3 off
the truth. Truth 8.045726 V (trbdf2 rtol 1e-9; pwl dt/100 and dt/1000 agree to 3e-7); default
reads 8.045714. Tolerance 2e-3 → 1e-4.

**test_mmc_thevenin.py** (3): (b) `mmc_arms` skips the restart rule (route reason); the restart
rule samples the gate at the step start, so the explicit chain read one step late at every edge
(108 V at the edge samples, final capacitor voltages equal). The explicit chain names
`discontinuity_restart=False`, the stepping the aggregated arm runs: parity 5e-8 V < 1e-6.
Observation (not a finding): an MMC arm and an explicit chain under defaults differ by that one
step of gate timing until `mmc_arms` gets the restart rule.

**test_mode_passivity.py** (4): (b) 2 — the mode guard (repairs, guard-off refusal) is the
settle's, pinned `mode_selection='legacy'`; (d) 2 + the lcp half of the first — the passivity
class is read from `diagnostics["mode_selection"]` of the default run (`auto` → `lcp`, asserted).
Bus voltages unchanged against pwl (31.8762412 V).

**test_dsed_control_ticks.py** (2): (c) undriven switch incidental to the observer refusals,
`undriven='closed'`.

**test_mode_auto.py** (2): (d) `..._not_the_default` → `test_auto_is_validated_and_the_default`
(record `{"requested": "auto", "chosen": "lcp"}`; `'legacy'` has no record); the fallback test
compared auto to the default — now auto, a tautology — it names `mode_selection='legacy'`;
(c) the thyristor's gate undriven on purpose, `undriven='closed'`.

**test_restart_review_fixes.py** (2): (b) `RULE` meant the rule without the split, which now
follows the rule: named `enable_substep_state_correction=False` (fixes L8b and the M3 config);
the M3 re-try count moves with the mode rule (19 settle, 22 `auto`; 10 / 13 with the trap) —
a count, now a bound (`>= 19`, `== retries >= 10`); kept 5, every re-try failing, 37.2032291 V
bus still exact.

**test_switch_state_record.py** (52): (b) `'auto'` named the pre-release no-dt route (TR-BDF2),
whose record the tests describe (midpoint gate bit, resume rows, adaptive TOL): `_run` maps it to
`engine='trbdf2'` (the release's exact route is covered by the file's `'dsed'` cases) — fixed
18 at once. Then: (d) the fixed engine stamps the step-start gate (restart rule; it read the
step end); (b) `form='conductance'` for the unsolved-row tests (2 tests ×2: a branch diode's
row-0 current is the seeded state current, KCL-exact) and the no-record fallback; (b)
`discontinuity_restart=False` for the chatter detector (snubber_chatter ×2, paired_diode,
rectifier_chatter[2e-6]) and blip tests (immaterial_reconductions, unresolved_recoveries ×2):
the restart rule reflects nothing; (d) rl_half_wave_recovery / curve_through_the_origin on pwl:
the restart rule leaves no blips, `n_turn_off_events_dropped == 0` asserted (was `> 0`);
snubber test: no chatter at either V_th under the rule (asserted), charge part F-2-2;
(b) dsed_summary_right_limit: the default step's raw grid pinned `output_rtol=0` (reproduces
−4.22e-3 / +1.39e-4); findings F-2-2 / F-2-3 / F-2-4 as above.

Legacy pins and their reasons are each on a one-line comment at the pin.
