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
- [x] 1.10 The output rule's cost: ten paralleled bridges ran > 8 min under `output_rtol=1e-5` (`test_exact_route.py::test_ten_paralleled_bridges_agree_with_pwl`). Done (DECISIONS 5): a projected parasitic mode's rounding x lambda^2 set a constant step; each mode's chord contribution now saturates at its amplitude, per-mode data cached, no per-step allocation, a per-run sample budget (`output_budget_t`)
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
- [x] 3.1 Defaults in `simulate()` and `sim::kRunOptionTable` take the values of `benchmarks/tools/next_defaults.py` (`--list`)
  - Includes the `PLAN` section: `pulsim._plan.DEFAULT_UNDRIVEN = 'refuse'`, `pulsim._plan.UNIFORM_ENGINE_RULES = True` (R1: flipped; deleting the switch and the today-branches it guards is R3).
  - Done in R1 (DECISIONS 5): every old value restorable by an explicit keyword; `next_defaults.legacy()` reproduces the pre-release numbers (goldens, stateful, engines, steps, pwl_identity, trbdf2_identity byte-identical to a32b97ed).
- [x] 3.2 `add_switch` / `add_diode` default to `form='branch'` (design D6 of `add-branch-form-ideal-switch`); YAML loader the same
- [x] 3.3 `engine='auto'` route as `add-exact-propagator-strategy` D11 (task 19.1 there) once its default-flip gate holds
- [ ] 3.4 Delete what the owning changes list at the flip (device-interface DECISIONS 25: the hand stamps) — R3
- [ ] 3.5 Re-run the rehearsal on the release tree: configuration (a) is then the new default and must equal the regenerated `REFERENCE_STATUS.md`
- [ ] 3.6 `add_igbt(reverse_blocking=...)` defaults to `True` (YAML `igbt` the same); update helpers and examples that relied on reverse conduction (`add-pwl-igbt-and-thyristor` 8.1, DECISIONS 4 there)
- [ ] 3.7 YAML `induction_motor` defaults to `model: mna` (BREAKING, CHANGELOG) (`add-mna-native-machines` 5.3)

## 4. Regeneration (one re-baseline)
- [x] 4.1 Goldens (R1: `benchmarks/goldens/pwl` regenerated, `pwl_legacy` kept, `RELEASE_GOLDEN_DIFF.md`; `stiff_rlc` and `back_to_back_diodes_sine` kept at their old golden: the new run is farther from the truth), `pwl_identity` / `trbdf2_identity` baselines (unchanged: they pin explicit configurations, compared under the legacy profile)
- [ ] 4.2 `benchmarks/REFERENCE_STATUS.md` and the triage freezes (`known:` / `known_err:`) per configuration
- [ ] 4.3 `benchmarks/EXACT_SCOREBOARD.md` and `exact_scoreboard.json`
- [ ] 4.4 (R2) Tests that pin old numbers (the rehearsal's pytest blast radius, by file), each re-pinned against a reference or truth, never loosened
- [ ] 4.5 Docs and examples that print numbers; `python/pulsim/AI_GUIDE.md`

## 5. Migration and deprecation notes
- [x] 5.1 CHANGELOG: one BREAKING entry per flipped option with its old value and the keyword that restores it
- [ ] 5.2 Deprecation notes: the old values stay reachable for one release; `'dsed'` as an alias of `'exact'` (roadmap decision 8)
- [ ] 5.3 `integrator='esdirk43'` documented as opt-in (not flipped)
- [ ] 5.4 One release AFTER this one: delete the four deprecated `update_*` bindings and the subsystem proxy's string-prefix code (`add-immutable-circuit-ir` 11.1 / 11.2; deprecated with a `DeprecationWarning` naming `with_value` / `set_value`, DECISIONS 10 there)

## 6. Verification
- [ ] 6.1 ruff, full pytest, full ctest, `pytest -m reference`, `reference_status` exit 0, `DO_NOT_TRACK=1 openspec validate --all --strict`, stubs `--check`, Eigen 3.4.0 for touched C++

## R1. The flip (branch `release/core-next-defaults`)
- [x] R1.1 Flip every default together, one place per layer (`_plan.py`, `sim::Defaults`, `kDefaultSwitchForm` in bindings and YAML loader); `next_defaults.check_current()` proves the layers agree
- [x] R1.2 `next_defaults.py` = legacy -> current data; `legacy()` profile; legacy byte compare identical to a32b97ed
- [x] R1.3 `R2_BUCKETS.md` from one full pytest and one full ctest run on the flipped tree (`probes/r2_buckets.py`)
- [x] R1.4 Bucket 0: shared helpers / tools / stub overlay; pytest plugin = the no-op check
- [x] R1.5 Goldens regenerated with `RELEASE_GOLDEN_DIFF.md`; analytic ceilings tightened where earned (none loosened)
- [x] R1.6 `REFERENCE_STATUS.md` + triage freezes and `EXACT_SCOREBOARD.md` regenerated; proofs B(ii) / B(iii) (proofs + evidence 4c6ba3cb; the 72 stale triage freezes migrated entry by entry 2ae30180, DECISIONS.md section 9: 69 a + 2 a* renamed, 0 b, 1 c = F-X-4; `reference_status` 0 `UNEXPECTED`, rc 0)

## R2. The red suite (six agents, `R2_BUCKETS.md`)
- [x] R2.1 Bucket 1 (eb7dec5a): 79 -> 0 (79 digests re-pinned, rule e)
- [x] R2.2 Bucket 2 (2098d226): 77 -> 0, 18 strict xfails = 4 findings
- [x] R2.3 Bucket 3 (fc41f7ed): 77 -> 0, 8 parked = 7 findings
- [x] R2.4 Bucket 4 (01491069): 77 -> 0, 9 parked = 7 findings
- [x] R2.5 Bucket 5 (01ef7dd6): 77 -> 0, 18 parked = 10 findings
- [x] R2.6 Bucket 6 (3fc21e89): 76 -> 0, 6 parked + F-6-7 = 7 findings
- [x] R2.7 Rules (a)-(e) of `R2_BUCKETS.md`; a bucket edits only its own files; a shared helper goes back to the orchestrator. Merged with no textual conflict (2758dc7b..b437d7a3); out-of-bucket files: bucket 5's three docs (doc snippets) and bucket 1's record under `add-immutable-circuit-ir/probes/` (both accepted); the three bucket-0 items fixed in 3210315e; `FINDINGS.md` (generated by `probes/findings_index.py`) indexes the 38 findings (35 of the buckets, F-X-1..3)
- [x] R2.8 (moved to R2.5: F-4-6 / F-X-2 overhead; the two goldens were accepted by the orchestrator, regeneration is R1 leftover 1) Findings the map hands to R2 as engine work, not test edits: `simulate()` fixed overhead (pwl 3056 > 2890 bytecodes; `auto` without dt now lands on the exact engine, 5456 > 3760), `test_golden[stiff_rlc]` / `[back_to_back_diodes_sine]` (new run farther from the truth)

- [x] R2.9 Second integrator (2026-11-20): merged `feat/core-integration` b9e46778 (fad54e51) and `release/h-call-overhead` (1b3f7ecc); ctest 1071/1071; full pytest 11 failed (2 accepted goldens + 9 counter gates, both pending their regeneration) / 6123 passed / 110 skipped / 67 xfailed, no XPASS; FINDINGS 36 open / 2 closed

## R2.5 Findings (FINDINGS.md; each group handed to a fix agent on the PR line, arriving here by merge)
- [ ] R2.5.A Impulsive events / the restart rule: F-3-1, F-3-3, F-3-7, F-5-1, F-5-10, F-6-1 — IN PROGRESS (`fix/core-release-a-impulsive` a324a2db, not merged here)
- [ ] R2.5.B Event samples and result integrals: F-2-2, F-4-4, F-5-4, F-5-5, F-5-6, F-5-7, F-6-2, F-6-4, F-6-5 — FIXED on the PR line (36836230), waits for PR batch 14q; still parked here
- [ ] R2.5.C The new mode selection: F-2-3, F-3-2, F-4-1, F-4-2, F-4-3, F-5-3, F-5-9 — FIXED on the PR line (d014f049), waits for PR batch 14q; still parked here
- [ ] R2.5.D trbdf2: F-3-4, F-4-5 — `fix/core-release-de-trbdf2-tick` 85637327 (F-3-4 traced, still OPEN; F-4-5 reproduced and bounded there, status per that branch's notes); not merged here
- [ ] R2.5.E Control tick: F-3-5, F-4-7 — `fix/core-release-de-trbdf2-tick` 85637327 (F-4-7 fixed fc16563e); not merged here
- [ ] R2.5.F The exact route / output rule: F-2-4, F-3-6, F-5-2, F-5-8, F-6-6, F-6-7, F-X-1, F-X-3, F-X-4 — IN PROGRESS (`fix/core-release-f-exact-output` a5b9a485); F-X-3 keeps `ladder500_rc` / dsed out of the counter freeze
- [ ] R2.5.G Thermal summary on branch-form devices: F-2-1, F-6-3 — FIXED on the PR line (6db843c0), waits for PR batch 14q; still parked here
- [x] R2.5.H Call overhead: F-4-6, F-X-2 — CLOSED: `release/h-call-overhead` 26d059b0 merged (1b3f7ecc); warm call 3057 -> 2800 (fixed step, ceiling 2890), 5609 -> 3634 (no dt, exact route, ceiling 3760); gate 3/3 pass

### R2.5 / R3 state (hand-off, 2026-11-20, tip 2ae30180)
- FINDINGS.md: 37 open + 2 closed. New: F-X-4 (group F, C), from the triage migration: `auto` without dt lands on dsed and writes 2,000,002 samples for a 2 s DC resistor; frozen `auto: disagree (0.00178)` on `thermal/electrothermal_resistor_self_heating.yaml`.
- Fixed on the PR line, waiting for integration batch 14q, still parked here: G (6db843c0), C (d014f049), B (36836230).
- D + E: `fix/core-release-de-trbdf2-tick` 85637327, F-3-4 open (traced, not fixed).
- A (`fix/core-release-a-impulsive` a324a2db) and F (`fix/core-release-f-exact-output` a5b9a485, now with F-X-4): in progress.
- H: merged here (1b3f7ecc), F-4-6 / F-X-2 closed.
- R3 stays BLOCKED until every finding is closed or accepted by the owner.
- Full pytest at 2ae30180 (extension `so_release_1b3f7ecc`, identical to the worktree's): 1 failed / 6133 passed / 110 skipped / 68 xfailed, no XPASS. The one failure, `test_flip_evidence_resumes_per_circuit`, was a race: its cache key is HEAD + the uncommitted diff, and two docs commits landed during the run. Alone on the stable tree it passes. Full ctest not run (no C++ change).

## R3. Delete the legacy paths (after one release) — BLOCKED on R2.5 (every finding closed or accepted by the owner)
- [ ] R3.1 Remove the `legacy` values and the today-branches they keep: `form='conductance'` default paths, `mode_selection='legacy'`, `control_tick='label'`, `device_path='legacy'` (and the hand stamps, 3.4), `undriven=None` per-engine behaviour, `UNIFORM_ENGINE_RULES` switch
- [ ] R3.2 Remove `benchmarks/goldens/pwl_legacy`, the `@legacy` reference columns and `next_defaults.legacy()`
- [ ] R3.3 `bdf1` retirement (separate change)
