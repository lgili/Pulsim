## 1. Blockers (from `benchmarks/NEXT_DEFAULTS_REHEARSAL.md`, ranked)
Rehearsal at fe75ee01 (the regenerated report's header names the runs its sections now come from): the full profile makes NO cell worse on pwl, trbdf2 or the exact engine (62 better on pwl, 0 moved in outcome on trbdf2 / dsed); every worse cell is on the `engine='auto'` route.
- [x] 1.1 `switching/buck/ll06_buck_vcswitch.yaml`, exact engine without `dt` (the release route): the algebraic node `V(sw)` (and `V(n_l)`) is wrong at the D1 turn-off event sample (t = 1.82598e-5 s, 99 % of span; `exact` 1.1e-2) while the states are right (`V(out)` 3.8e-7, `I(L1)` 1.6e-5); with the netlist dt the same engine scores 9.5e-12. Route alone, no profile option needed. Repro: `next_defaults_rehearsal._no_dt_run(v1_reader.load('benchmarks/circuits/switching/buck/ll06_buck_vcswitch.yaml'), 'dsed')`, graded by `exact_scoreboard.grade`
  - Done (DECISIONS 2): `project_` cut the real ~100 ns transient (100/dt_max); under `output_rtol` only the parasitic cluster is projected. `exact` 1.1e-2 -> 8.9e-12 (mpmath truth); test `test_exact_output_grid.py::test_ll06_d1_turn_off_matches_the_truth_without_dt`
- [x] 1.2 `closed_loop/cl_buck_aggressive_pi.yaml` on the exact engine without `dt`: strict err_rms 3.95e-3 > tol 1e-3 against the trbdf2 reference (no truth for closed loops); with the netlist dt it agrees. Decide grid or controller with the closed-loop reference (`add-compiled-control-graph/probes/closed_loop_reference.py`)
  - Done (DECISIONS 2): no engine defect; trbdf2 rtol 1e-9 + tick 'state' and pwl split dt 2e-8 agree within 8.4e-6, the dsed samples with both to 1.2e-6; the grid alone. `auto+next` strict 3.95e-3 -> 6.7e-6
- [x] 1.3 The exact engine's output grid without `dt` (25 circuits that disagree, 22 more whose strict error grows > 2x but still agree, among them `cl_boost_pi` 1.3e-6 -> 1.5e-4 and `cl_buck_pi` 1.9e-5 -> 8.5e-4): the samples are within tolerance of the truth (largest `exact` 3e-6) but 10 µs apart, so the waveform between them fails the strict gate (worst `lcc_resonant_inverter` err_rms 0.288). The route needs an output rule before the flip (e.g. `dt_max` from the circuit's fastest source / gate period, or dense output); then re-run the rehearsal's `auto+next`
  - Done (DECISIONS 2): `output_rtol` (chord bound on the mode's closed form, cap 4096 with `diagnostics['output_grid']`), profile 1e-5. `auto+next` 106/2/1 (= `auto`), 73 better, 6 worse (none from the grid: 5 reference-basis with the truth at 1e-11..1e-16, 1 electrothermal reference ramp); `lcc_resonant_inverter` 0.288 -> 5.4e-6, `cl_boost_pi` 1.5e-4 -> 4.3e-6, `cl_buck_pi` 8.5e-4 -> 3.0e-6. Cost on the 100 dsed-routed circuits: samples +20.7 %, wall 1.97 -> 3.44 s
- [x] 1.4 The exact period map / integrals refuse a branch-form diode (`periodic_steady_state`, sensitivities, shooting: 103 tests under the profile). Branch current in the period map, or those diodes stay conductance-form. Repro: `python/tests/test_periodic_steady_state.py` with `-p next_defaults_pytest`. Done (DECISIONS 1.1): branch current in the map, no fallback; buck CCM/DCM and boost DCM orbits = transient steady state within 1e-9 and the g_on = 1e6 twin within 1e-5 (multipliers within 1e-5), bridge vs g_on = 1e4 twin within 1e-4, ideal buck <vout> = D*Vin rel 1e-9, dx/dp vs central differences within 1e-7 (`test_branch_form_period_map.py`); `dependent_states` stays refused
- [x] 1.5 `simulate(mmc_arms=..., discontinuity_restart=True)` is refused (full-dt Thevenin stamp at dt/2; 13 tests). Engine fix, or the rule stays off under `mmc_arms` (decide with 2.2 block 2). Repro: `python/tests/test_mmc_thevenin.py` with the plugin. Done (DECISIONS 1.2): tri-state, the rule's default is not applied under `mmc_arms` (one route reason), explicit True still refused (`test_restart_default_resolution.py`)
- [x] 1.6 A branch-form switch refuses a bound `g_on` (`bind_param`; 7 tests in `test_circuit_scopes.py`). Done (DECISIONS 1.3): g_on -> R_on = 1/g_on, g_off bindable; parameters stay finite, R_on = 0 by `set_value`; twin within 1e-4 (`test_branch_form_bind_param.py`)
- [x] 1.7 bdf1 and `ac_sweep` cannot simulate a branch-form switch / diode (3 tests). Done (DECISIONS 1.4): `ac_sweep` stamps branch form (RC closed form, twin rtol 1e-7); bdf1 refuses by name, saying it is retired and naming pwl / trbdf2; conformance table updated
- [x] 1.8 `enable_substep_state_correction`'s default must follow `discontinuity_restart`: a caller that sets the rule off is refused once the split defaults on (1 test, `test_device_losses_kernel.py`). Done (DECISIONS 1.2): the split's default follows the rule's resolved value
- [ ] 1.9 Re-run `next_defaults_rehearsal.py run` and the pytest blast radius on the release candidate; no cell left in the first three worse classes, no refusal group in section 1b

## 2. Owner decisions before the flip
- [x] 2.1 One undriven-switch policy (`add-kernel-simulation-object` DECISIONS 23 item 1, task 11.3 there): today pwl and dsed run all CLOSED with a warning, trbdf2 refuses, `auto` without dt refuses; under the release route `auto` would run it on the exact engine, CLOSED
- [x] 2.2 The other per-engine blocks 2–8 of the same decision (`mmc_arms` / `on_cache`, `progress` / `live_stream`, `resume_from` time, `mode_selection='auto'` resolution site, `on_impulse='error'`, pwl-only value checks, `sweep()` recipe)
- [x] 2.3 One voltage-sanity check (`add-kernel-simulation-object` task 6.2)
  - 2.1–2.3 decided by the orchestrator (2026-11-19) and implemented behind options whose default is today's (DECISIONS 3 here; DECISIONS 26 of `add-kernel-simulation-object`): `undriven='refuse'` (`_plan.DEFAULT_UNDRIVEN`) and `_plan.UNIFORM_ENGINE_RULES = True` are the release values in `next_defaults.py` `PLAN`; `mode_selection='auto'`, the voltage check and the resume clock are one rule each already; `mmc_arms` / `on_cache`, `progress` / `live_stream`, the exact loss port stay per engine (engine-inherent). Open: the exact engine refuses `on_impulse='error'` after the run, not at the event (its scheduler can retract a record; `dsed/scheduler.hpp` is the exact-engine track's). Blast radius with `PLAN`: 480 -> 518 failed, 26 tests in 11 files rely on an undriven switch (DECISIONS 4).
- [ ] 2.4 TR-BDF2 step grid: flip only with its option and its accuracy sign-off (`update-trbdf2-factor-reuse` 5.6, 6.1)
- [x] 2.5 The exact route's output grid without `dt` (rehearsal section "engine='auto' under the profile")
  - Decided (orchestrator, DECISIONS 2): every event instant plus a rule grid, `output_rtol = 1e-5` in the profile; explicit dt / store_every / recorders keep today's samples (simulate() has no t_eval)

## 3. The flip
- [ ] 3.1 Defaults in `simulate()` and `sim::kRunOptionTable` take the values of `benchmarks/tools/next_defaults.py` (`--list`)
  - Includes the `PLAN` section: `pulsim._plan.DEFAULT_UNDRIVEN = 'refuse'`, `pulsim._plan.UNIFORM_ENGINE_RULES = True` (then delete the switch and the today-branches it guards).
- [ ] 3.2 `add_switch` / `add_diode` default to `form='branch'` (design D6 of `add-branch-form-ideal-switch`); YAML loader the same
- [ ] 3.3 `engine='auto'` route as `add-exact-propagator-strategy` D11 (task 19.1 there) once its default-flip gate holds
- [ ] 3.4 Delete what the owning changes list at the flip (device-interface DECISIONS 25: the hand stamps)
- [ ] 3.5 Re-run the rehearsal on the release tree: configuration (a) is then the new default and must equal the regenerated `REFERENCE_STATUS.md`

## 4. Regeneration (one re-baseline)
- [ ] 4.1 Goldens, `pwl_identity` / `trbdf2_identity` baselines
- [ ] 4.2 `benchmarks/REFERENCE_STATUS.md` and the triage freezes (`known:` / `known_err:`) per configuration
- [ ] 4.3 `benchmarks/EXACT_SCOREBOARD.md` and `exact_scoreboard.json`
- [ ] 4.4 Tests that pin old numbers (the rehearsal's pytest blast radius, by file), each re-pinned against a reference or truth, never loosened
- [ ] 4.5 Docs and examples that print numbers; `python/pulsim/AI_GUIDE.md`

## 5. Migration and deprecation notes
- [ ] 5.1 CHANGELOG: one BREAKING entry per flipped option with its old value and the keyword that restores it
- [ ] 5.2 Deprecation notes: the old values stay reachable for one release; `'dsed'` as an alias of `'exact'` (roadmap decision 8)
- [ ] 5.3 `integrator='esdirk43'` documented as opt-in (not flipped)

## 6. Verification
- [ ] 6.1 ruff, full pytest, full ctest, `pytest -m reference`, `reference_status` exit 0, `DO_NOT_TRACK=1 openspec validate --all --strict`, stubs `--check`, Eigen 3.4.0 for touched C++
