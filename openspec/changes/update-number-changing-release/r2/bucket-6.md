# R2 bucket 6 — notes

Branch `release/r2-bucket-6` (from `origin/release/core-next-defaults`), extension
`so_release_r1`. 76 failures in 20 files at the map's run.

Legend: (a) physics under the new defaults, (b) legacy subject pinned, (c) undriven policy,
(d) new documented behaviour stated. "F-6-n" = engine finding, parked as
`xfail(strict=True)` with the expectation untouched.

## Per file

| file | fails | fixed by |
|---|---:|---|
| test_switch_conductance_validation.py | 13 | (b) 13: `form='conductance'` (builders), `switch_form='conductance'` (YAML) — the subject is the conductance form's validation |
| test_fixed_step_event_splitting.py | 13 | (b)/(d) 13: `RULE` now spells `enable_substep_state_correction=False` ("the rule alone"; the split follows the rule by default) — fixes 9; pre-release default-engine numbers pin `discontinuity_restart=False` (+ `form='conductance'` where the conductance form's event REPORT is the subject) — 2; the split-without-rule refusal passes `discontinuity_restart=False` — 1; `test_split_off_is_the_restart_rule_bit_for_bit` (not in the map, broke on the new `RULE`): "off" is now the split at its default |
| test_mode_lcp.py | 9 | (b) 9: "the default" of this file is the guarded settle: `mode_selection='legacy'` in `_sim` and the two direct calls ('auto' is 'lcp' on dsed since the release) |
| test_circuit_scopes.py | 7 | (c) 7: `undriven='closed'` on the flat-vs-scoped example runs |
| test_exact_route.py | 5 | (d) 3: `allow_implicit_segments` with `engine='auto'` — refused with a dt (pwl), accepted without (exact; new test); `auto_would_route` test states `engine_used == 'dsed'`; (a) 1: F-6-6 split out as its own strict xfail, the main test keeps the refusal + physics + "the numeric tier ran" (segment report); ten bridges first mode: left exactly as it was (coordinator) — see F-6-7 |
| test_controlled_sources.py | 4 | (a) 3 tightened + (b) legacy halves; F-6-5 1 |
| test_fixed_step_loop_bookkeeping.py | 4 | (b) 4 restart-less counts pinned; (d) the default's counts stated analytically (new test `test_the_split_adds_one_read_per_gate_edge`) |
| test_cache_budget.py | 3 | (c) 3: `undriven='closed'` (storage tests, switches never driven) |
| test_hostile_circuits.py | 3 | (a) 1: physics asserted under the default (no retry needed); (b) 3: the retry / hard failure pin `discontinuity_restart=False` |
| test_device_state_snapshot.py | 2 | (b) 2: the 10-field 2.0 layout resumes on the restart-less step |
| test_loss_window.py | 2 | F-6-2, F-6-3 |
| test_restart_default_resolution.py | 2 | (d) 2: both plan defaults True; pre-release resolution kept under monkeypatched pre-release defaults (new test) |
| test_steady_state.py | 2 | (a) 1 tightened + F-6-1 (new strict-xfail test keeps the default-probe call visible); (d) 1: the non-affine refusal comes from the pattern check now (the branch form reports the diode's commutations; the conductance form does too under the restart rule; only conductance + restart-less hits the residual backstop) |
| test_closed_loop_reference.py | 1 | (b) 1: "the restart rule alone" = split off |
| test_control_graph_rates.py | 1 | (b) 1: the first-order ZOH test pins `discontinuity_restart=False` |
| test_device_losses_variable_step.py | 1 | F-6-4 |
| test_form_keyword.py | 1 | (d) 1: default form is branch; conductance refusals pinned |
| test_igbt_thyristor.py | 1 | (b) 1: reference `add_switch` pins `form='conductance'` (add_igbt not flipped) |
| test_monitor_kind.py | 1 | (d) 1: default output rule runs rk45 → Monitor accepted, bit-identical; refusal pinned `output_rtol=0` |
| test_result_type.py | 1 | (c)/(d) 1: undriven is refused / an explicit policy is silent; the warning under test is now the split's unlocated-gate warning of a driven switch |

## Tolerances tightened (old → new, measured)

- `test_controlled_sources::test_gyrator_turns_a_capacitor_into_an_inductor`: error vs the
  exponential `== dt/(2τ)` (5e-4) → `< 4e-8` (measured 3.06e-8; 7.66e-9 at dt/2; order 4 asserted
  ±5 %). The dt/(2τ) statement stays under `discontinuity_restart=False`.
- `test_controlled_sources::test_vccs_integrator_on_the_fixed_grid`: first step `1.5·dt` → every
  step `3·dt` to 1e-9 relative (measured 5e-19). 1.5·dt kept under the legacy pin.
- `test_controlled_sources::test_fixed_engine_converges_to_them_at_second_order` (new; the
  first-order test kept under the legacy pin): truth = exact engine on a 10 ns grid (trbdf2's
  sample grid is a 2.8e-7 floor); `e1 < 1e-6` (measured 3.0e-7), `e2 < 0.02·e1` (measured 0.011;
  first order asserted 0.15).
- `test_steady_state::test_orbit_is_a_fixed_point_and_stays_one`: residual / drift / 100-period
  move 1e-8 → 1e-10 (measured 2.0e-11) with `probe_scale=0.1`; the default probe is F-6-1.

## Legacy pins (reason)

- conductance form: its validation (13), add_igbt's default switch (1), the conductance diode's
  event report "0.98 / 0.33 dt early" (event splitting, 2 blocks).
- `discontinuity_restart=False`: first-order start / ZOH / RL statements (controlled sources 3,
  control graph 1), dt-retry mechanism (hostile 3), restart-less bookkeeping counts (4),
  the pre-release default engine's DCM numbers (event splitting 2), the 2.0 10-field snapshot
  (no stamped mask → the rule restarts at the resume: 34 vs 37 restart steps, 1.1e-7 apart).
- `mode_selection='legacy'`: test_mode_lcp's "default" = the guarded settle (9).
- `output_rtol=0`: the Monitor refusal needs the 'auto' integrator.

## Findings

| id | smallest call | truth | old | new | cause (reading) |
|---|---|---|---|---|---|
| F-6-1 | `steady_state(buck, period=1e-5, dt=2e-8, switch_fn=grid_gate)` (test_steady_state `_buck`) | fixed point (residual → 0; probe_scale=0.1: 2.0e-11) | residual 1.8e-10 | 1.9e-8 (> 1e-8) | the default `probe_scale=1e-4` (probe 5e-4) is inside the restart engine's per-period round-off; the map is affine (larger probes are better) |
| F-6-2 | `simulate(_bridge(), t_end=0.1)` (test_loss_window: 230 V bridge, 0.1 Ω, 470 µF‖1 kΩ) → exact | i_avg(80–100 ms) 0.161005 A (pwl dt=1e-7), summary 0.161027 | (trbdf2 route) identity held | `time_average(res.i('BR__D1'))` 0.160684 (−0.21 %) | at a diode turn-on the exact engine stores ONE sample with the pre-event value (7e-10 A), not the right limit (13.6 A: the 2×0.7 V threshold across 0.1 Ω); the trapezoid loses 6.8 A·µs per turn-on |
| F-6-3 | `device_thermal_summary(b, res, thermal_specs, conduction_specs={'D1': {V_f0: .7, r_on: .02}})` on a pwl buck with a default (branch) diode | 2.449 W (loss summary; the conductance form gives it in both) | 2.449 W | 0.0186 W (the device's own R_on loss) | the thermal summary's conduction-model path does not reach a branch-form diode — a silent wrong junction temperature |
| F-6-4 | RL chopper of test_device_losses_variable_step, `engine='pwl', dt=1e-7` | closed form E_cond | rel 1.8e-4 (S1) / 2.4e-4 (D1) | 1.86e-3 / 2.0e-3 (tol 1.5e-3), first order | the loss port's ∫v·i over the restart rule's SDIRK steps is first order with a 10× larger constant, while i_L is 260× closer (7.4e-5 vs 1.9e-2 A) |
| F-6-5 | pulse current source 0→1 A at 1e-4..3e-4 into 1 mΩ + 5 Ω, `pwl dt=1e-6` | one value per row | row 3e-4: Ip 0, Rsq 0 | row 3e-4: Ip 0.0, Rsq 1.0 | under the restart rule the edge row is solved with the pre-edge source value; `res.i(source)` evaluates the waveform at t |
| F-6-6 | RC + `b_extra_fn`, `engine='exact', allow_implicit_segments=True` | numeric steps counted | n_rk45_steps 1005 | 0 (n_accept 1005) | under the default output rule a tier-3 segment's integrator steps are counted nowhere |

| F-6-7 (not parked, by the coordinator's instruction) | `simulate(_paralleled_bridges(10), t_end=1e-4, engine='exact')` (test_exact_route) | i(L0a)(1e-4) = 0.12972284951463489 A (mpmath, 50 digits) | 2.6e-10 off (output_rtol=0, 101 samples) | 6.6e-8 off (tol 1e-8; 12845 samples) | the trajectory depends on the output grid: more samples, more error (dt=1e-6: 4.3e-9; output_rtol=1e-8, 406132 samples: 2.1e-6, bus 1.4e-8) — output points feed back into the propagation (re-start from a rounded / projected state at each sample) |

## Not done

- `test_exact_route.py`'s two ten-paralleled-bridges cases: not run to completion in R2: slow under
  the pre-fix extension; the merge step runs them (fix `fix/core-exact-output-rule-cost` 1e2b90ff).
  Left exactly as they were. My probe of the first-mode case DID complete (data in F-6-7): under
  this extension it fails its 1e-8 bound by 6.6e-8; recheck at the merge with the fix.
- Baseline re-check of the map: my first baseline's output went to a shared scratch file another
  agent overwrote; the confirmation re-run was stopped on the coordinator's word. Every one of the
  76 mapped failures was reproduced individually while fixing it (per-file runs), none extra seen
  except `test_split_off_is_the_restart_rule_bit_for_bit`, which broke on the new `RULE` constant
  (fixed).

## Deleted / skipped / renamed

None deleted or skipped. Renamed (their subject changed with the flip):
`test_todays_defaults_are_both_off` → `test_the_release_defaults_are_both_on`;
`test_explicit_restart_alone_keeps_the_split_off_today` → `test_explicit_restart_alone_turns_the_split_on`
+ `test_explicit_restart_alone_kept_the_split_off_before_the_release` (the old assertion, under the
pre-release plan defaults). New tests: `test_the_split_adds_one_read_per_gate_edge`,
`test_fixed_engine_converges_to_them_at_second_order`,
`test_auto_without_dt_takes_allow_implicit_segments_on_the_exact_engine`, and the strict-xfail
finding tests `test_the_default_probe_finds_the_orbit_to_1e_8` (F-6-1),
`test_a_numeric_segment_counts_its_integrator_steps` (F-6-6).

## Needed from bucket 0

Nothing. Observed: `examples/*.yaml` with controlled switches load without a driver, so a plain
`simulate()` of a built `Circuit` from them refuses by default (circuit_scopes worked around with
`undriven='closed'`); the shipped examples' own tests are another bucket's.
