## 1. Blockers (from `benchmarks/NEXT_DEFAULTS_REHEARSAL.md`, ranked)
Rehearsal at fe75ee01: the full profile makes NO cell worse on pwl, trbdf2 or the exact engine (62 better on pwl, 0 moved in outcome on trbdf2 / dsed); every worse cell is on the `engine='auto'` route.
- [ ] 1.1 `switching/buck/ll06_buck_vcswitch.yaml`, exact engine without `dt` (the release route): the algebraic node `V(sw)` (and `V(n_l)`) is wrong at the D1 turn-off event sample (t = 1.82598e-5 s, 99 % of span; `exact` 1.1e-2) while the states are right (`V(out)` 3.8e-7, `I(L1)` 1.6e-5); with the netlist dt the same engine scores 9.5e-12. Route alone, no profile option needed. Repro: `next_defaults_rehearsal._no_dt_run(v1_reader.load('benchmarks/circuits/switching/buck/ll06_buck_vcswitch.yaml'), 'dsed')`, graded by `exact_scoreboard.grade`
- [ ] 1.2 `closed_loop/cl_buck_aggressive_pi.yaml` on the exact engine without `dt`: strict err_rms 3.95e-3 > tol 1e-3 against the trbdf2 reference (no truth for closed loops); with the netlist dt it agrees. Decide grid or controller with the closed-loop reference (`add-compiled-control-graph/probes/closed_loop_reference.py`)
- [ ] 1.3 The exact engine's output grid without `dt` (25 circuits): the samples are within tolerance of the truth (largest `exact` 3e-6) but 10 µs apart, so the waveform between them fails the strict gate (worst `lcc_resonant_inverter` err_rms 0.288). The route needs an output rule before the flip (e.g. `dt_max` from the circuit's fastest source / gate period, or dense output); then re-run the rehearsal's `auto+next`
- [ ] 1.4 Re-run `next_defaults_rehearsal.py run` and the pytest blast radius on the release candidate; no class-1/2 cell left

## 2. Owner decisions before the flip
- [ ] 2.1 One undriven-switch policy (`add-kernel-simulation-object` DECISIONS 23 item 1, task 11.3 there): today pwl and dsed run all CLOSED with a warning, trbdf2 refuses, `auto` without dt refuses; under the release route `auto` would run it on the exact engine, CLOSED
- [ ] 2.2 The other per-engine blocks 2–8 of the same decision (`mmc_arms` / `on_cache`, `progress` / `live_stream`, `resume_from` time, `mode_selection='auto'` resolution site, `on_impulse='error'`, pwl-only value checks, `sweep()` recipe)
- [ ] 2.3 One voltage-sanity check (`add-kernel-simulation-object` task 6.2)
- [ ] 2.4 TR-BDF2 step grid: flip only with its option and its accuracy sign-off (`update-trbdf2-factor-reuse` 5.6, 6.1)
- [ ] 2.5 The exact route's output grid without `dt` (rehearsal section "engine='auto' under the profile")

## 3. The flip
- [ ] 3.1 Defaults in `simulate()` and `sim::kRunOptionTable` take the values of `benchmarks/tools/next_defaults.py` (`--list`)
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
