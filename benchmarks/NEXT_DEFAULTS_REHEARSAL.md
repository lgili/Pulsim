# Next-defaults rehearsal

Rehearsal of the number-changing release, rendered on `e86fee57` from the cell cache of `run`
(method and tables below the blocker list; where the cells and the pytest blast radius ran: `## Notes`).

## Ranked blocker list

### 1. Cells worse under the profile than under today's default, by class

The first three classes (fails, samples wrong, disagree without a truth) are engine
fixes before the release; the last three need a decision or a re-baseline with evidence.


**less accurate at the samples (truth, > 2x), still agrees**

1. `analytic:hard_switched_rc` on dsed: agree (7.58e-14 / ex 7.6e-14) -> agree (2.53e-10 / ex 2.5e-10); responsible: dsed:output_grid.  Repro: `with nd.builder_form(False): v1_reader.load('benchmarks/analytic:hard_switched_rc').run('dsed', output_rtol=1e-05)`

**strict gate only (disagrees; RMS error of the samples against the truth within tol)**

2. 1 cells (auto). Largest `exact` (RMS of the samples against the truth) 0. On `auto` the exact engine runs without dt: the `output_rtol` = 1e-05 output grid; the strict gate interpolates linearly between those samples. Circuits (today -> profile): `thermal/electrothermal_resistor_self_heating.yaml` agree 0 / ex 0 -> disagree 0.00178 / ex 0. Repro (any of them): `next_defaults_rehearsal._no_dt_run(v1_reader.load('benchmarks/circuits/thermal/electrothermal_resistor_self_heating.yaml'), 'dsed')`

**strict error > 2x the base's, still agrees**

3. 5 cells (auto). Largest `exact` (RMS of the samples against the truth) 4.8e-11. On `auto` the exact engine runs without dt: the `output_rtol` = 1e-05 output grid; the strict gate interpolates linearly between those samples. Circuits (today -> profile): `diodes/back_to_back_diodes_sine.yaml` agree 0 / ex 2e-16 -> agree 1.15e-05 / ex 2e-16; `switching/buck/ll06_buck_vcswitch.yaml` agree 8.21e-05 / ex 6.3e-06 -> agree 0.000171 / ex 8.8e-12; `switching/inverter/ll09_half_bridge_inverter_leg.yaml` agree 4.27e-05 / ex 3.2e-06 -> agree 0.000167 / ex 4.8e-11; `switching/multilevel/cascaded_h_bridge_5level.yaml` agree 0 / ex 6e-16 -> agree 5.72e-05 / ex 1.9e-16; `switching/multilevel/mmc_4cell_chain.yaml` agree 0 / ex 1.2e-13 -> agree 5.76e-05 / ex 1.2e-13. Repro (any of them): `next_defaults_rehearsal._no_dt_run(v1_reader.load('benchmarks/circuits/diodes/back_to_back_diodes_sine.yaml'), 'dsed')`

### 1b. Refusals the profile raises in `python/tests` (engine fixes or owner decisions)

1. 1 test, option `form='branch'`: bdf1 (retired) refuses a branch-form switch / diode by name and names pwl / trbdf2 (update-number-changing-release 1.7; ac_sweep runs them since): at the flip bdf1 keeps the conductance form or the test moves off bdf1. Repro: `python/tests/test_engine_conformance.py::test_device_cell[Switch-bdf1]` under `-p next_defaults_pytest`.
2. 72 tests, option `output_rtol`: the exact engine's sensitivity run refuses `output_rtol` by name (it keeps its own grid): with `output_rtol` in the profile every exact `sensitivities=` call without dt is refused. Engine work (the rule on the sensitivity run) or the profile leaves `output_rtol` off when `sensitivities=` is set. Repro: `python/tests/test_branch_form_period_map.py::test_transient_sensitivities_equal_central_differences[L.L]` under `-p next_defaults_pytest`.
3. 24 tests, option `_plan.DEFAULT_UNDRIVEN`: a test that relies on an undriven controlled switch (all closed today on pwl and the exact engine) is refused by name once `undriven='refuse'` is the default: it passes `undriven='closed'` or drives the switch (update-number-changing-release DECISIONS 4). Repro: `python/tests/test_cache_budget.py::test_result_total_bytes_covers_more_than_states` under `-p next_defaults_pytest`.

### 2. Accepted changes of number

162 cells better under the profile. By engine and basis: auto outcome 1, auto reference 8, auto truth 64, pwl outcome 11, pwl reference 13, pwl truth 65. Each is listed in 'Better under the full profile' below with its basis; the CHANGELOG
migration note of the release cites them.

### 3. The per-engine blocks (add-kernel-simulation-object DECISIONS 23, decided in DECISIONS 26)

Decided by the orchestrator (2026-11-19) behind options whose default is today's; the release values are `next_defaults.PLAN`, applied above by the full-profile cells and by the pytest blast radius (`update-number-changing-release` DECISIONS 3-4):
1. No `switch_fn`: one policy, `simulate(undriven=)`; the release default `'refuse'` (`_plan.DEFAULT_UNDRIVEN`) refuses an undriven controlled switch by name on every engine and through `auto`. Repro: `b = pulsim.CircuitBuilder(); b.add_voltage_source('V', 'a', '0', 1.0); b.add_switch('S', 'a', 'o', 1e3, 1e-9); b.add_resistor('R', 'o', '0', 1.0)`, then `pulsim.simulate(b, 1e-5, 1e-7, engine=e, undriven='refuse')` for e in pwl / trbdf2 / dsed.
2. `mmc_arms` / `on_cache`, `progress` / `live_stream`, the exact loss port: kept per engine (engine-inherent), refused by name elsewhere (option table).
3. `resume_from`: pwl / trbdf2 continue the snapshot's clock (`plan.resolved_from['t_start']`); the exact engine resumes on its own absolute clock.
4. `mode_selection='auto'`: resolved in one place (`_route`) for every engine; on trbdf2 it is the settle (no 'auto' rule there).
5. Count checks and `on_impulse`: one rule per option on every engine under `_plan.UNIFORM_ENGINE_RULES` (the release value); open: the exact engine refuses `on_impulse='error'` after the run, not at the event.

Also open for the owner: the composite diode drop's release (its DECISIONS OQ6); the TR-BDF2 step grid (factor-reuse 5.6: accuracy sign-off).

## Pytest blast radius under the profile

Full `python/tests` with the profile injected by `benchmarks/tools/next_defaults_pytest.py`: 5465 passed, 583 failed, 47 error, 110 skipped, 3 xfailed. Profile applications: {'branch:add_switch': 108751, 'branch:add_diode': 9065, 'applied:trbdf2': 1321, 'applied:pwl': 4912, 'applied:dsed': 1510, 'auto: exact refused -> trbdf2': 9}.

| class | failures |
|---|---:|
| numeric assertion on an old value | 199 |
| other assertion | 141 |
| new refusal by name | 107 |
| engine route / result layout | 76 |
| circuit digest (IR / YAML path not patched: harness artefact) | 50 |
| warning / raise expectation | 29 |
| exception | 25 |
| snapshot / layout | 3 |

| test file | failures | classes |
|---|---:|---|
| python/tests/test_circuit_ir.py | 67 | circuit digest (IR / YAML path not patched: harness artefact) 41, other assertion 26 |
| python/tests/test_benchmark_harness.py | 51 | numeric assertion on an old value 32, other assertion 10, circuit digest (IR / YAML path not patched: harness artefact) 9 |
| python/tests/test_switch_state_record.py | 45 | numeric assertion on an old value 29, other assertion 6, engine route / result layout 6, exception 3, warning / raise expectation 1 |
| python/tests/test_sensitivity_kpi.py | 30 | new refusal by name 30 |
| python/tests/test_sensitivity_exact.py | 29 | new refusal by name 22, other assertion 6, engine route / result layout 1 |
| python/tests/test_trbdf2_robustness.py | 22 | engine route / result layout 20, numeric assertion on an old value 1, warning / raise expectation 1 |
| python/tests/test_fixed_step_discontinuity_order.py | 18 | numeric assertion on an old value 11, other assertion 6, warning / raise expectation 1 |
| python/tests/test_diode_state_reconstruction.py | 14 | numeric assertion on an old value 7, other assertion 5, engine route / result layout 2 |
| python/tests/test_fixed_step_event_splitting.py | 13 | numeric assertion on an old value 7, other assertion 5, warning / raise expectation 1 |
| python/tests/test_diode_forward_drop.py | 12 | numeric assertion on an old value 9, other assertion 3 |
| python/tests/test_restart_rule_integration.py | 12 | other assertion 8, numeric assertion on an old value 4 |
| python/tests/test_switch_conductance_validation.py | 12 | new refusal by name 5, other assertion 3, warning / raise expectation 2, exception 2 |
| python/tests/test_fixed_step_resolution.py | 11 | numeric assertion on an old value 11 |
| python/tests/test_exact_resume.py | 10 | other assertion 7, numeric assertion on an old value 1, new refusal by name 1, warning / raise expectation 1 |
| python/tests/test_mode_selector.py | 10 | numeric assertion on an old value 10 |
| python/tests/test_sensitivity_closed_loop.py | 10 | new refusal by name 8, other assertion 2 |
| python/tests/test_mode_lcp.py | 9 | engine route / result layout 9 |
| python/tests/test_newton_roundoff_floor.py | 9 | numeric assertion on an old value 8, other assertion 1 |
| python/tests/test_release_uniform_rules.py | 9 | other assertion 7, warning / raise expectation 1, new refusal by name 1 |
| python/tests/test_engine_auto.py | 8 | engine route / result layout 6, warning / raise expectation 2 |
| python/tests/test_mode_pwl.py | 8 | engine route / result layout 4, numeric assertion on an old value 2, other assertion 2 |
| python/tests/test_circuit_scopes.py | 7 | new refusal by name 7 |
| python/tests/test_engine_conformance.py | 7 | new refusal by name 2, engine route / result layout 2, exception 2, warning / raise expectation 1 |
| python/tests/test_stubs.py | 7 | other assertion 6, numeric assertion on an old value 1 |
| python/tests/test_exact_route.py | 6 | other assertion 2, new refusal by name 1, numeric assertion on an old value 1, warning / raise expectation 1, engine route / result layout 1 |

## Method

Generated by `benchmarks/tools/next_defaults_rehearsal.py render` (do not edit above `## Notes`) from the
cell cache of `run`; the profile is `benchmarks/tools/next_defaults.py` (`--list`).
Nothing here changes a library default: every option is applied by the harness.

## Profile

| engine | option | value | change | decision |
|---|---|---|---|---|
| pwl | `discontinuity_restart` | `True` | `add-discontinuity-restart-rule` | DECISIONS 4 (default off until the release); CORE_ROADMAP decisions 2-3 (SDIRK(2,2)x3, gate mask sampled at the step start, same flag) |
| pwl | `enable_substep_state_correction` | `True` | `add-fixed-step-event-splitting` | DECISIONS 1 (flips with the restart rule, maintainer's); refused without the rule |
| pwl | `mode_selection` | `'auto'` | `add-complementarity-mode-selection` | DECISIONS F5 ('auto' is the value the release flips to) |
| pwl | `control_tick` | `'state'` | `add-compiled-control-graph` | DECISIONS G30 (default 'label' until the release; G32 queue ticks) |
| pwl | `device_path` | `'interface'` | `add-unified-device-interface` | DECISIONS 23/25 (the release flips the default and deletes the hand stamps) |
| trbdf2 | `mode_selection` | `'auto'` | `add-complementarity-mode-selection` | DECISIONS F5 (on trbdf2 'auto' resolves to the settle: no 'auto' rule there yet) |
| trbdf2 | `control_tick` | `'state'` | `add-compiled-control-graph` | DECISIONS G30 |
| trbdf2 | `device_path` | `'interface'` | `add-unified-device-interface` | DECISIONS 23/25 |
| dsed | `mode_selection` | `'auto'` | `add-complementarity-mode-selection` | DECISIONS F5 |
| dsed | `control_tick` | `'state'` | `add-compiled-control-graph` | DECISIONS G30 |
| dsed | `output_rtol` | `1e-05` | `update-number-changing-release` | DECISIONS 2 (auto route): without dt the exact engine returns every event instant plus an output grid chosen by rule so that linear interpolation is within output_rtol (orchestrator's decision; the harness gate is never loosened); 1e-5, not 1e-4: at 1e-4 24 auto cells grew past 2x the trbdf2 strict error while agreeing, at 1e-5 none does |
| all | `form` | `'branch'` | `add-branch-form-ideal-switch` | DECISIONS 23 (measurement gate: flip allowed on pwl), 39 (flip evidence); CORE_ROADMAP decision 3 (the default switch model) |
| all | `_plan.DEFAULT_UNDRIVEN` | `'refuse'` | `add-kernel-simulation-object` | DECISIONS 26.1 (task 11.3; orchestrator decision 2026-11-19: an undriven controlled switch is refused by name on every engine; simulate(undriven=) keeps 'open' / 'closed' / None reachable) |
| all | `_plan.UNIFORM_ENGINE_RULES` | `True` | `add-kernel-simulation-object` | DECISIONS 26.2 (items 6-7 of DECISIONS 23: one count check per option on every engine that reads it; on_impulse refused by name where the run cannot detect an impulse; the exact engine's impulse refusal names the device kind) |
| auto | `route` | `'exact if every segment is exact (exact_route), else trbdf2; an explicit dt -> pwl'` | `add-exact-propagator-strategy` | D11 + 'Route report and the default-flip gate' (2026-11-18); task 19.1 (open) |
| trbdf2 | `integrator (extra column)` | `'esdirk43'` | `add-high-order-stiff-integrator` | stays opt-in (CORE_ROADMAP corrections: helps resonant, costs 2.5x per step on event-limited circuits) |

Not applied (not expressible through the harness on this tree):

- trbdf2: step_grid_per_octave = 4 (`update-trbdf2-factor-reuse`): tasks 4.2 / 6.1 open: the option does not exist on this tree
- trbdf2: filter_error_estimate / sdirk_after_settle (`update-trbdf2-post-event-restart`): DECISIONS 9-10: C++-only TrBdf2Options, no simulate() keyword; both stay off
- pwl, trbdf2: composite diode V_F as a conduction drop (`update-composite-diode-forward-drop`): nothing to flip: already the default on this tree (the add_mosfet / Level-1 docstrings read V_F as the body diode's CONDUCTION DROP); a builder default, not a run option
- pwl: mmc_arms / on_cache, progress / live_stream (`add-kernel-simulation-object`): DECISIONS 26.3: kept per engine (they need the fixed engine's own cache / per-step ring); every other engine refuses them by name (option table)

## Counts, differential matrix

109 circuits.
Graded as `REFERENCE_STATUS.md` (strict gate, harness tolerances, suspect -> unverified).
`timeout`: the cell passed the per-cell limit and was killed.

| configuration | group | agree | disagree | error | refused | unverified | timeout | cells | wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pwl | a | 81 | 27 | 1 | 0 | 0 | 0 | 109 | 0.4 |
| pwl+restart | a | 89 | 19 | 1 | 0 | 0 | 0 | 109 | 0.5 |
| pwl+split | a | 91 | 17 | 1 | 0 | 0 | 0 | 109 | 0.5 |
| trbdf2 | a | 106 | 2 | 1 | 0 | 0 | 0 | 109 | 1.0 |
| dsed | a | 100 | 0 | 0 | 9 | 0 | 0 | 109 | 2.0 |
| pwl:restart | b | 89 | 19 | 1 | 0 | 0 | 0 | 109 | 0.5 |
| pwl:split | b | 91 | 17 | 1 | 0 | 0 | 0 | 109 | 0.5 |
| pwl:mode_auto | b | 81 | 27 | 1 | 0 | 0 | 0 | 109 | 0.4 |
| pwl:tick_state | b | 81 | 27 | 1 | 0 | 0 | 0 | 109 | 0.4 |
| pwl:dev_interface | b | 81 | 27 | 1 | 0 | 0 | 0 | 109 | 0.3 |
| pwl:branch | b | 81 | 27 | 1 | 0 | 0 | 0 | 109 | 1.3 |
| pwl+next | c | 91 | 17 | 1 | 0 | 0 | 0 | 109 | 0.6 |
| trbdf2:mode_auto | b | 106 | 2 | 1 | 0 | 0 | 0 | 109 | 1.0 |
| trbdf2:tick_state | b | 106 | 2 | 1 | 0 | 0 | 0 | 109 | 1.0 |
| trbdf2:dev_interface | b | 106 | 2 | 1 | 0 | 0 | 0 | 109 | 1.0 |
| trbdf2:branch | b | 106 | 2 | 1 | 0 | 0 | 0 | 109 | 1.1 |
| trbdf2+next | c | 106 | 2 | 1 | 0 | 0 | 0 | 109 | 1.1 |
| dsed:mode_auto | b | 100 | 0 | 0 | 9 | 0 | 0 | 109 | 1.9 |
| dsed:tick_state | b | 100 | 0 | 0 | 9 | 0 | 0 | 109 | 1.9 |
| dsed:output_grid | b | 100 | 0 | 0 | 9 | 0 | 0 | 109 | 1.9 |
| dsed:branch | b | 100 | 0 | 0 | 9 | 0 | 0 | 109 | 2.1 |
| dsed+next | c | 100 | 0 | 0 | 9 | 0 | 0 | 109 | 2.0 |
| trbdf2+esdirk43 | extra | 102 | 6 | 1 | 0 | 0 | 0 | 109 | 1.8 |
| trbdf2+next+esdirk43 | extra | 102 | 6 | 1 | 0 | 0 | 0 | 109 | 1.9 |
| auto | d | 106 | 2 | 1 | 0 | 0 | 0 | 109 | 1.0 |
| auto:route | b | 80 | 28 | 1 | 0 | 0 | 0 | 109 | 4.3 |
| auto+next | d | 106 | 2 | 1 | 0 | 0 | 0 | 109 | 7.7 |

Configuration (a) reproduces REFERENCE_STATUS.md's strict counts: **yes** (read from that file: {'pwl': (81, 27, 1, 0, 0), 'pwl+restart': (89, 19, 1, 0, 0), 'pwl+split': (91, 17, 1, 0, 0), 'trbdf2': (106, 2, 1, 0, 0), 'dsed': (100, 0, 0, 9, 0)}).

## Counts, analytic set

27 cases; closed-form truth, frozen ceilings.

| configuration | agree | disagree | error | refused | unverified | timeout |
|---|---:|---:|---:|---:|---:|---:|
| pwl | 26 | 1 | 0 | 0 | 0 | 0 |
| pwl+restart | 26 | 1 | 0 | 0 | 0 | 0 |
| pwl+split | 27 | 0 | 0 | 0 | 0 | 0 |
| trbdf2 | 27 | 0 | 0 | 0 | 0 | 0 |
| dsed | 22 | 0 | 0 | 5 | 0 | 0 |
| pwl:restart | 26 | 1 | 0 | 0 | 0 | 0 |
| pwl:split | 27 | 0 | 0 | 0 | 0 | 0 |
| pwl:mode_auto | 26 | 1 | 0 | 0 | 0 | 0 |
| pwl:tick_state | 26 | 1 | 0 | 0 | 0 | 0 |
| pwl:dev_interface | 26 | 1 | 0 | 0 | 0 | 0 |
| pwl:branch | 26 | 1 | 0 | 0 | 0 | 0 |
| pwl+next | 27 | 0 | 0 | 0 | 0 | 0 |
| trbdf2:mode_auto | 27 | 0 | 0 | 0 | 0 | 0 |
| trbdf2:tick_state | 27 | 0 | 0 | 0 | 0 | 0 |
| trbdf2:dev_interface | 27 | 0 | 0 | 0 | 0 | 0 |
| trbdf2:branch | 27 | 0 | 0 | 0 | 0 | 0 |
| trbdf2+next | 27 | 0 | 0 | 0 | 0 | 0 |
| dsed:mode_auto | 22 | 0 | 0 | 5 | 0 | 0 |
| dsed:tick_state | 22 | 0 | 0 | 5 | 0 | 0 |
| dsed:output_grid | 22 | 0 | 0 | 5 | 0 | 0 |
| dsed:branch | 22 | 0 | 0 | 5 | 0 | 0 |
| dsed+next | 22 | 0 | 0 | 5 | 0 | 0 |
| trbdf2+esdirk43 | 23 | 2 | 0 | 2 | 0 | 0 |
| trbdf2+next+esdirk43 | 23 | 2 | 0 | 2 | 0 | 0 |
| auto:route | 27 | 0 | 0 | 0 | 0 | 0 |
| auto+next | 27 | 0 | 0 | 0 | 0 | 0 |

`auto` (today) has no analytic rows: the cases name their engine; the route rows route them.
Configuration (a) reproduces REFERENCE_STATUS.md's analytic counts: **yes**.

## Each configuration against today's default of its engine

Same rule as below (outcome rank, else `exact` or the strict err_rms beyond 2x); `auto+next` against `auto`,
the esdirk columns against `trbdf2`. Circuits and analytic cases together.

`moved` = cells whose err_rms is not bit-equal to the base's (the option reached the
run); `mode` = the `mode_selection='auto'` choices recorded by the runs (lcp / legacy).

| configuration | base | better | worse | compared | moved | mode |
|---|---|---:|---:|---:|---:|---|
| pwl:restart | pwl | 82 | 4 | 136 | 116 |  |
| pwl:split | pwl | 88 | 0 | 136 | 114 |  |
| pwl:mode_auto | pwl | 0 | 0 | 136 | 1 | lcp 39, legacy 97 |
| pwl:tick_state | pwl | 0 | 0 | 136 | 4 |  |
| pwl:dev_interface | pwl | 0 | 0 | 136 | 2 |  |
| pwl:branch | pwl | 0 | 0 | 136 | 56 |  |
| pwl+next | pwl | 89 | 0 | 136 | 114 | lcp 39, legacy 97 |
| trbdf2:mode_auto | trbdf2 | 0 | 0 | 136 | 0 | legacy 135 |
| trbdf2:tick_state | trbdf2 | 0 | 0 | 136 | 0 |  |
| trbdf2:dev_interface | trbdf2 | 0 | 0 | 136 | 2 |  |
| trbdf2:branch | trbdf2 | 0 | 0 | 136 | 65 |  |
| trbdf2+next | trbdf2 | 0 | 0 | 136 | 67 | legacy 135 |
| dsed:mode_auto | dsed | 0 | 0 | 136 | 2 | lcp 39, legacy 83 |
| dsed:tick_state | dsed | 0 | 0 | 136 | 0 |  |
| dsed:output_grid | dsed | 0 | 1 | 136 | 18 |  |
| dsed:branch | dsed | 0 | 0 | 136 | 45 |  |
| dsed+next | dsed | 0 | 1 | 136 | 61 | lcp 39, legacy 83 |
| trbdf2+esdirk43 | trbdf2 | 68 | 21 | 136 | 116 |  |
| trbdf2+next+esdirk43 | trbdf2 | 68 | 21 | 136 | 119 | legacy 133 |
| auto | auto | 0 | 0 | 109 | 0 |  |
| auto:route | auto | 29 | 49 | 109 | 97 |  |
| auto+next | auto | 73 | 6 | 109 | 104 | lcp 39, legacy 96 |

## Worse under the full profile than under today's default for that engine

Worse = outcome rank up (agree/unverified < disagree < refused < error < timeout), or the
`exact` error (truth; for the analytic set the closed-form err_rms) more than 2x larger,
or the strict err_rms more than 2x the base's (above 1 % of the tolerance). Responsible =
the single options (b) that alone make the same cell worse; `combination` when none does.

| circuit | engine | today | full profile | responsible | class; message |
|---|---|---|---|---|---|
| analytic:hard_switched_rc | dsed | agree 7.58e-14 / ex 7.6e-14 | agree 2.53e-10 / ex 2.5e-10 | dsed:output_grid | less accurate at the samples (truth, > 2x), still agrees;  |
| diodes/back_to_back_diodes_sine.yaml | auto | agree 0 / ex 2e-16 | agree 1.15e-05 / ex 2e-16 | auto:route | strict error > 2x the base's, still agrees;  |
| switching/buck/ll06_buck_vcswitch.yaml | auto | agree 8.21e-05 / ex 6.3e-06 | agree 0.000171 / ex 8.8e-12 | auto:route | strict error > 2x the base's, still agrees;  |
| switching/inverter/ll09_half_bridge_inverter_leg.yaml | auto | agree 4.27e-05 / ex 3.2e-06 | agree 0.000167 / ex 4.8e-11 | auto:route | strict error > 2x the base's, still agrees;  |
| switching/multilevel/cascaded_h_bridge_5level.yaml | auto | agree 0 / ex 6e-16 | agree 5.72e-05 / ex 1.9e-16 | auto:route | strict error > 2x the base's, still agrees;  |
| switching/multilevel/mmc_4cell_chain.yaml | auto | agree 0 / ex 1.2e-13 | agree 5.76e-05 / ex 1.2e-13 | auto:route | strict error > 2x the base's, still agrees;  |
| thermal/electrothermal_resistor_self_heating.yaml | auto | agree 0 / ex 0 | disagree 0.00178 / ex 0 | auto:route | strict gate only (disagrees; RMS error of the samples against the truth within tol);  |

## Better under the full profile

Better = outcome rank down (basis `outcome`, the harness reference), or the `exact` error
more than 2x smaller with the strict err_rms not grown past 2x (basis `truth`), or the
strict err_rms more than 2x smaller (basis `reference`, no truth or truth unchanged).

| circuit | engine | today | full profile | responsible | basis |
|---|---|---|---|---|---|
| analytic:buck_ccm | pwl | agree 0.000256 / ex 0.00026 | agree 1.23e-08 / ex 1.2e-08 | pwl:restart, pwl:split | truth |
| analytic:buck_dcm | pwl | agree 0.000538 / ex 0.00054 | agree 1.87e-08 / ex 1.9e-08 | pwl:restart, pwl:split | truth |
| analytic:coupled_k099 | pwl | agree 0.000765 / ex 0.00076 | agree 8.27e-06 / ex 8.3e-06 | pwl:restart, pwl:split | truth |
| analytic:gapped_core_rl | pwl | agree 0.00159 / ex 0.0016 | agree 1.84e-06 / ex 1.8e-06 | pwl:restart, pwl:split | truth |
| analytic:halfwave_rl | pwl | agree 9.65e-06 / ex 9.6e-06 | agree 1.44e-06 / ex 1.4e-06 | pwl:restart, pwl:split | truth |
| analytic:hard_switched_rc | pwl | disagree 0.01 / ex 0.01 | agree 4.92e-05 / ex 4.9e-05 | pwl:restart, pwl:split | outcome |
| analytic:im_free_acceleration | pwl | agree 9.62e-05 / ex 9.6e-05 | agree 6e-06 / ex 6e-06 | pwl:restart, pwl:split | truth |
| analytic:lc_lossless | pwl | agree 0.00178 / ex 0.0018 | agree 3.31e-05 / ex 3.3e-05 | pwl:restart, pwl:split | truth |
| analytic:rc_charge | pwl | agree 0.00159 / ex 0.0016 | agree 1.84e-06 / ex 1.8e-06 | pwl:restart, pwl:split | truth |
| analytic:rc_discharge | pwl | agree 0.00159 / ex 0.0016 | agree 1.84e-06 / ex 1.8e-06 | pwl:restart, pwl:split | truth |
| analytic:rc_esr_charge | pwl | agree 0.000144 / ex 0.00014 | agree 1.35e-08 / ex 1.3e-08 | pwl:restart, pwl:split | truth |
| analytic:rl_step | pwl | agree 0.00112 / ex 0.0011 | agree 1.3e-06 / ex 1.3e-06 | pwl:restart, pwl:split | truth |
| analytic:rlc_overdamped | pwl | agree 5.31e-05 / ex 5.3e-05 | agree 4.21e-08 / ex 4.2e-08 | pwl:restart, pwl:split | truth |
| analytic:rlc_underdamped | pwl | agree 0.000613 / ex 0.00061 | agree 1.42e-05 / ex 1.4e-05 | pwl:restart, pwl:split | truth |
| analytic:three_limb_zero_seq | pwl | agree 0.00159 / ex 0.0016 | agree 1.84e-06 / ex 1.8e-06 | pwl:restart, pwl:split | truth |
| closed_loop/cl_boost_pi.yaml | pwl | disagree 0.0582 | agree 1.06e-05 | pwl:split | outcome |
| closed_loop/cl_buck_aggressive_pi.yaml | pwl | agree 0.00662 | agree 3.69e-05 | combination | reference |
| closed_loop/cl_buck_pi.yaml | pwl | agree 0.0121 | agree 1.77e-05 | pwl:split | reference |
| closed_loop/ll15_boost_closed_loop.yaml | pwl | disagree 0.22 | disagree 0.05 | pwl:restart, pwl:split | reference |
| diodes/diode_bridge_rectifier.yaml | pwl | agree 0.000216 / ex 0.0001 | agree 0.000173 / ex 2.4e-05 | pwl:restart, pwl:split | truth |
| diodes/diode_half_wave_rl_load.yaml | pwl | agree 0.0186 / ex 0.02 | agree 0.000586 / ex 5e-05 | pwl:restart, pwl:split | truth |
| diodes/diode_rectifier.yaml | pwl | agree 0.0139 / ex 0.0097 | agree 0.00134 / ex 0.00018 | pwl:restart, pwl:split | truth |
| diodes/half_wave_rectifier_sine.yaml | pwl | agree 0.000393 / ex 0.00038 | agree 4.09e-05 / ex 2.6e-06 | pwl:restart, pwl:split | truth |
| diodes/voltage_doubler.yaml | pwl | agree 0.000222 / ex 0.00022 | agree 3.01e-05 / ex 2e-07 | pwl:restart, pwl:split | truth |
| linear/capacitor_natural_discharge.yaml | pwl | agree 0.000798 / ex 0.00079 | agree 7.52e-07 / ex 4.6e-07 | pwl:restart, pwl:split | truth |
| linear/current_source_rl_step.yaml | pwl | disagree 0.00577 / ex 0.0054 | agree 3.99e-05 / ex 3.1e-05 | pwl:restart, pwl:split | outcome |
| linear/rc_dc.yaml | pwl | agree 0.000159 / ex 0.00016 | agree 4.39e-07 / ex 1.9e-08 | pwl:restart, pwl:split | truth |
| linear/rl_natural_response.yaml | pwl | disagree 0.00577 / ex 0.0054 | agree 3.98e-05 / ex 3.1e-05 | pwl:restart, pwl:split | outcome |
| linear/rl_step.yaml | pwl | agree 0.00112 | agree 1.8e-06 | pwl:restart, pwl:split | reference |
| linear/two_stage_rc_lowpass.yaml | pwl | agree 0.00145 | agree 1.01e-05 | pwl:restart, pwl:split | reference |
| magnetics/half_bridge_transformer_dc.yaml | pwl | disagree 0.101 / ex 0.1 | agree 0.000127 / ex 0.00013 | pwl:restart, pwl:split | outcome |
| magnetics/inductor_chain_step.yaml | pwl | agree 0.00231 | agree 5.34e-06 | pwl:restart, pwl:split | reference |
| motors/motor_dc_brush_step_load.yaml | pwl | agree 0.000289 / ex 0.00029 | agree 4.5e-07 / ex 1.4e-07 | pwl:restart, pwl:split | truth |
| motors/motor_pmsm_dq_open_loop.yaml | pwl | agree 0.000177 / ex 0.00018 | agree 4.81e-07 / ex 5.2e-08 | pwl:restart, pwl:split | truth |
| resonance/lc_resonant_tank.yaml | pwl | agree 0.00585 | agree 0.000312 | pwl:restart, pwl:split | reference |
| resonance/rlc_overdamped_step.yaml | pwl | agree 0.00179 | agree 1.12e-05 | pwl:restart, pwl:split | reference |
| resonance/rlc_parallel_step.yaml | pwl | agree 0.00778 | agree 0.000404 | pwl:restart, pwl:split | reference |
| resonance/rlc_step.yaml | pwl | agree 0.00043 | agree 9.23e-07 | pwl:restart, pwl:split | reference |
| stress/high_q_rlc_ringdown.yaml | pwl | disagree 0.0229 / ex 0.023 | agree 0.0149 / ex 0.015 | pwl:restart, pwl:split | outcome |
| stress/periodic_rc_pwm.yaml | pwl | agree 0.0119 / ex 0.012 | agree 0.000284 / ex 6.4e-07 | pwl:restart, pwl:split | truth |
| stress/stiff_rc_high_freq_switching.yaml | pwl | agree 0.00267 / ex 0.0027 | agree 2.03e-05 / ex 1.8e-05 | pwl:restart, pwl:split | truth |
| switching/basic/chopper_with_rc_snubber.yaml | pwl | agree 0.0187 / ex 0.0047 | agree 0.0153 / ex 3.1e-08 | pwl:restart, pwl:split | truth |
| switching/basic/pwm_voltage_rc.yaml | pwl | agree 0.000883 / ex 0.00088 | agree 9.98e-09 / ex 1.4e-08 | pwl:restart, pwl:split | truth |
| switching/basic/pwm_voltage_rl_filter.yaml | pwl | agree 0.00429 / ex 0.0042 | agree 6.93e-06 / ex 1.1e-06 | pwl:restart, pwl:split | truth |
| switching/boost/boost_classic.yaml | pwl | agree 0.000313 / ex 0.00031 | agree 3.93e-07 / ex 6.3e-07 | pwl:restart, pwl:split | truth |
| switching/boost/boost_pfc_open_loop.yaml | pwl | disagree 0.0304 / ex 0.03 | agree 1.06e-05 / ex 1.6e-06 | pwl:split | outcome |
| switching/boost/boost_switching_complex.yaml | pwl | agree 0.00192 / ex 0.0017 | agree 3.15e-05 / ex 1.9e-05 | pwl:restart, pwl:split | truth |
| switching/boost/cascaded_boost_buck.yaml | pwl | agree 0.000399 / ex 0.0004 | agree 2.24e-06 / ex 2.4e-07 | pwl:restart, pwl:split | truth |
| switching/boost/ll07_boost_vcswitch.yaml | pwl | disagree 0.0984 / ex 0.0022 | disagree 0.0997 / ex 3e-05 | pwl:split | truth |
| switching/boost/synchronous_boost.yaml | pwl | agree 6.31e-05 / ex 6.3e-05 | agree 2.93e-07 / ex 2.2e-08 | pwl:restart, pwl:split | truth |
| switching/boost/textbook_erickson_boost_ccm.yaml | pwl | agree 0.000313 / ex 0.00031 | agree 3.93e-07 / ex 6.3e-07 | pwl:restart, pwl:split | truth |
| switching/boost/two_phase_interleaved_boost.yaml | pwl | agree 0.00019 / ex 0.00019 | agree 4.23e-07 / ex 5e-07 | pwl:restart, pwl:split | truth |
| switching/buck/buck_dcm.yaml | pwl | agree 0.00183 / ex 0.0018 | agree 3.09e-06 / ex 3e-06 | pwl:restart, pwl:split | truth |
| switching/buck/buck_hard_switching_reference.yaml | pwl | agree 0.00102 / ex 0.001 | agree 7.61e-07 / ex 2.8e-07 | pwl:restart, pwl:split | truth |
| switching/buck/buck_mosfet_nonlinear.yaml | pwl | agree 0.00841 | agree 4.07e-07 | pwl:restart, pwl:split | reference |
| switching/buck/buck_switching.yaml | pwl | agree 0.00168 / ex 0.0017 | agree 9.53e-07 / ex 3.2e-07 | pwl:restart, pwl:split | truth |
| switching/buck/buck_with_current_sense.yaml | pwl | agree 0.00131 / ex 0.0013 | agree 1.05e-06 / ex 7.3e-07 | pwl:restart, pwl:split | truth |
| switching/buck/cascaded_buck_buck.yaml | pwl | agree 0.00126 / ex 0.0013 | agree 1.16e-06 / ex 1.4e-06 | pwl:restart, pwl:split | truth |
| switching/buck/high_freq_gan_buck.yaml | pwl | agree 0.00103 / ex 0.001 | agree 6.2e-06 / ex 3.6e-06 | pwl:restart, pwl:split | truth |
| switching/buck/ll06_buck_vcswitch.yaml | pwl | disagree 0.0781 / ex 0.055 | disagree 0.0704 / ex 0.013 | pwl:split | truth |
| switching/buck/ll08_buck_mosfet.yaml | pwl | disagree 0.227 | disagree 0.0843 | pwl:restart, pwl:split | reference |
| switching/buck/sync_buck_500khz.yaml | pwl | agree 0.00167 / ex 0.0017 | agree 3.77e-06 / ex 4.4e-06 | pwl:restart, pwl:split | truth |
| switching/buck/synchronous_buck.yaml | pwl | agree 0.00132 / ex 0.0013 | agree 1.05e-06 / ex 7.5e-07 | pwl:restart, pwl:split | truth |
| switching/buck/textbook_erickson_buck_ccm.yaml | pwl | agree 0.000771 / ex 0.00077 | agree 4.21e-07 / ex 1.4e-07 | pwl:restart, pwl:split | truth |
| switching/buck/triple_sync_buck.yaml | pwl | agree 0.00132 / ex 0.0013 | agree 1.02e-06 / ex 7.3e-07 | pwl:restart, pwl:split | truth |
| switching/buck_boost/bidirectional_buck_boost.yaml | pwl | agree 0.00132 / ex 0.0013 | agree 1.05e-06 / ex 7.4e-07 | pwl:restart, pwl:split | truth |
| switching/buck_boost/buck_boost.yaml | pwl | agree 0.00072 / ex 0.00072 | agree 6.34e-07 / ex 6.1e-07 | pwl:restart, pwl:split | truth |
| switching/buck_boost/cuk_converter.yaml | pwl | agree 0.000854 / ex 0.00085 | agree 6.2e-06 / ex 8.4e-06 | pwl:restart, pwl:split | truth |
| switching/buck_boost/sepic.yaml | pwl | agree 0.00186 / ex 0.0019 | agree 1.73e-06 / ex 1.3e-06 | pwl:restart, pwl:split | truth |
| switching/inverter/half_bridge_inverter.yaml | pwl | agree 0.000376 / ex 0.00038 | agree 1.16e-06 / ex 7.5e-08 | pwl:restart, pwl:split | truth |
| switching/inverter/half_bridge_inverter_lc.yaml | pwl | agree 0.00103 / ex 0.001 | agree 7.91e-06 / ex 1.2e-05 | pwl:restart, pwl:split | truth |
| switching/inverter/half_bridge_lc_filter.yaml | pwl | agree 0.00131 / ex 0.0013 | agree 1.11e-06 / ex 9.1e-07 | pwl:restart, pwl:split | truth |
| switching/inverter/lcc_resonant_inverter.yaml | pwl | agree 0.00565 / ex 0.0056 | agree 0.000232 / ex 0.00019 | pwl:restart, pwl:split | truth |
| switching/inverter/ll09_half_bridge_inverter_leg.yaml | pwl | disagree 0.0707 / ex 0.063 | disagree 0.0587 / ex 0.0098 | pwl:restart, pwl:split | truth |
| switching/inverter/ll10_full_bridge_inverter_lc.yaml | pwl | disagree 0.05 / ex 0.0018 | disagree 0.05 / ex 1.6e-07 | pwl:restart, pwl:split | truth |
| switching/isolated/dual_active_bridge.yaml | pwl | agree 0.000265 / ex 0.00027 | agree 2.4e-06 / ex 1.1e-06 | pwl:restart, pwl:split | truth |
| switching/isolated/flyback_converter.yaml | pwl | disagree 0.949 / ex 0.95 | agree 2.26e-06 / ex 2.5e-06 | pwl:restart, pwl:split | outcome |
| switching/isolated/forward_converter.yaml | pwl | disagree 0.707 / ex 0.71 | agree 5.07e-06 / ex 4.9e-06 | pwl:restart, pwl:split | outcome |
| switching/isolated/full_bridge_dc.yaml | pwl | disagree 0.0828 / ex 0.014 | disagree 0.0815 / ex 3.8e-09 | pwl:restart, pwl:split | truth |
| switching/isolated/ll13_active_clamp_forward.yaml | pwl | disagree 4.51 | disagree 0.335 | pwl:restart, pwl:split | reference |
| switching/isolated/synchronous_flyback.yaml | pwl | disagree 0.207 / ex 0.21 | agree 1.01e-06 / ex 1.3e-07 | pwl:restart, pwl:split | outcome |
| switching/multilevel/cascaded_h_bridge_5level.yaml | pwl | disagree 0.0274 / ex 0.005 | disagree 0.0269 / ex 1.9e-16 | pwl:restart, pwl:split | truth |
| switching/multilevel/flying_capacitor_3level.yaml | pwl | disagree 0.0466 / ex 0.0058 | disagree 0.05 / ex 1.5e-06 | pwl:restart, pwl:split | truth |
| switching/multilevel/mmc_4cell_chain.yaml | pwl | disagree 0.0268 / ex 0.005 | disagree 0.0264 / ex 2.2e-13 | pwl:restart, pwl:split | truth |
| switching/multilevel/npc_three_level_halfbridge.yaml | pwl | agree 0.00319 / ex 0.0032 | agree 5.06e-06 / ex 4e-07 | pwl:restart, pwl:split | truth |
| switching/multilevel/t_type_3level_halfbridge.yaml | pwl | disagree 0.0782 / ex 0.012 | disagree 0.0771 / ex 2.1e-07 | pwl:restart, pwl:split | truth |
| thermal/long_run_drift_buck.yaml | pwl | agree 0.000603 / ex 0.0006 | agree 1.12e-05 / ex 3.3e-06 | pwl:restart, pwl:split | truth |
| three_phase/interleaved_buck_3ph.yaml | pwl | agree 0.00217 / ex 0.0022 | agree 1.75e-06 / ex 1.7e-06 | pwl:split | truth |
| three_phase/three_phase_diode_rectifier.yaml | pwl | disagree 1.51 / ex 0.23 | agree 0.00631 / ex 0.001 | pwl:restart, pwl:split | outcome |
| closed_loop/cl_buck_aggressive_pi.yaml | auto | agree 3.72e-05 | agree 6.73e-06 | combination | reference |
| closed_loop/cl_buck_pi.yaml | auto | agree 1.88e-05 | agree 2.98e-06 | combination | reference |
| diodes/diode_bridge_rectifier.yaml | auto | agree 3.22e-05 / ex 3.5e-07 | agree 6.59e-06 / ex 3e-14 | auto:route | truth |
| diodes/diode_half_wave_rl_load.yaml | auto | agree 1.58e-05 / ex 5.6e-06 | agree 6.92e-06 / ex 4.9e-06 | auto:route | reference |
| diodes/diode_rectifier.yaml | auto | agree 0.000183 / ex 2.4e-07 | agree 2.78e-06 / ex 3.1e-14 | combination | truth |
| diodes/half_wave_rectifier_sine.yaml | auto | agree 2.36e-05 / ex 7.3e-08 | agree 4.63e-06 / ex 3.6e-13 | auto:route | truth |
| diodes/voltage_doubler.yaml | auto | agree 5.05e-06 / ex 3.5e-08 | agree 5.91e-07 / ex 2.5e-13 | auto:route | truth |
| linear/capacitor_natural_discharge.yaml | auto | agree 5.63e-06 / ex 5.5e-06 | agree 6.1e-07 / ex 1.1e-14 | auto:route | truth |
| linear/current_source_rl_step.yaml | auto | agree 2.14e-06 / ex 4.6e-07 | agree 1.09e-06 / ex 2e-15 | auto:route | truth |
| linear/lc_low_pass_sine.yaml | auto | agree 0.000118 / ex 0.00011 | agree 1.23e-06 / ex 8.5e-14 | auto:route | truth |
| linear/pulse_voltage_rl.yaml | auto | agree 3.04e-05 | agree 1.29e-06 | combination | reference |
| linear/rc_dc.yaml | auto | agree 6.84e-07 / ex 2.3e-07 | agree 4.28e-07 / ex 1.4e-14 | auto:route | truth |
| linear/rl_natural_response.yaml | auto | agree 6.66e-06 / ex 5.1e-06 | agree 1.23e-06 / ex 1.5e-15 | auto:route | truth |
| linear/rlc_series_resonant.yaml | auto | agree 0.000354 / ex 0.00034 | agree 4.04e-06 / ex 5.2e-13 | auto:route | truth |
| linear/sine_voltage_rc.yaml | auto | agree 0.000111 / ex 2.4e-05 | agree 2.93e-06 / ex 3.1e-15 | auto:route | truth |
| magnetics/core_loss_steinmetz_sine.yaml | auto | agree 0.000123 / ex 4.1e-05 | agree 3.7e-06 / ex 5.8e-14 | combination | truth |
| magnetics/half_bridge_transformer_dc.yaml | auto | agree 2.08e-05 / ex 1.5e-05 | agree 2.66e-06 / ex 6.7e-13 | combination | truth |
| magnetics/transformer_leakage.yaml | auto | agree 0.000231 / ex 1.8e-05 | agree 2.88e-06 / ex 2.3e-14 | combination | truth |
| motors/motor_dc_brush_step_load.yaml | auto | agree 6.06e-06 / ex 2.3e-06 | agree 3.62e-07 / ex 1e-13 | auto:route | truth |
| motors/motor_induction_locked_rotor.yaml | auto | agree 5.1e-05 / ex 1.4e-05 | agree 1.63e-06 / ex 6.4e-14 | auto:route | truth |
| motors/motor_pmsm_dq_open_loop.yaml | auto | agree 6.51e-06 / ex 2.5e-06 | agree 4.45e-07 / ex 1.3e-13 | auto:route | truth |
| resonance/driven_lc_series_resonance.yaml | auto | agree 0.000217 / ex 0.0002 | agree 3.06e-06 / ex 1.8e-12 | auto:route | truth |
| resonance/lc_resonant_tank.yaml | auto | agree 5.29e-05 | agree 6.55e-06 | combination | reference |
| resonance/ll04_rlc_underdamped.yaml | auto | agree 2.99e-05 | agree 7.39e-07 | auto:route | reference |
| resonance/rl_high_pass_sine.yaml | auto | agree 7.56e-05 / ex 9.9e-06 | agree 2.6e-06 / ex 3.4e-15 | auto:route | truth |
| resonance/rlc_parallel_step.yaml | auto | agree 1.61e-05 | agree 8.08e-07 | combination | reference |
| resonance/rlc_step.yaml | auto | agree 1.31e-05 | agree 4.41e-07 | auto:route | reference |
| stress/high_q_rlc_ringdown.yaml | auto | disagree 0.00566 / ex 0.0057 | agree 5.61e-05 / ex 9.7e-13 | auto:route | outcome |
| stress/periodic_rc_pwm.yaml | auto | agree 2e-07 / ex 5.1e-08 | agree 5.12e-06 / ex 1.3e-15 | combination | truth |
| stress/stiff_rc_high_freq_switching.yaml | auto | agree 1.43e-05 / ex 1e-05 | agree 6.89e-06 / ex 2.4e-14 | combination | truth |
| switching/basic/chopper_with_rc_snubber.yaml | auto | agree 3.81e-05 / ex 4.6e-08 | agree 2.91e-05 / ex 4.8e-15 | auto:route | truth |
| switching/basic/pwm_voltage_rc.yaml | auto | agree 8.98e-09 / ex 8.9e-09 | agree 1.57e-08 / ex 5.4e-14 | auto:route | truth |
| switching/basic/pwm_voltage_rl_filter.yaml | auto | agree 6.53e-05 / ex 4.9e-06 | agree 6.05e-06 / ex 4.9e-15 | auto:route | truth |
| switching/boost/boost_classic.yaml | auto | agree 9.53e-06 / ex 7.7e-06 | agree 5.4e-06 / ex 1.3e-14 | combination | truth |
| switching/boost/boost_pfc_open_loop.yaml | auto | agree 1.72e-07 / ex 1.7e-07 | agree 2.7e-06 / ex 5.7e-14 | combination | truth |
| switching/boost/boost_switching_complex.yaml | auto | agree 1.05e-05 / ex 7e-06 | agree 2.32e-06 / ex 7.5e-16 | combination | truth |
| switching/boost/cascaded_boost_buck.yaml | auto | agree 5.95e-05 / ex 3.2e-06 | agree 6.54e-06 / ex 1.8e-13 | combination | truth |
| switching/boost/ll07_boost_vcswitch.yaml | auto | agree 0.000117 / ex 0.00016 | agree 0.000181 / ex 1.8e-10 | combination | truth |
| switching/boost/synchronous_boost.yaml | auto | agree 8.19e-08 / ex 2.7e-07 | agree 4.45e-06 / ex 5.5e-12 | combination | truth |
| switching/boost/textbook_erickson_boost_ccm.yaml | auto | agree 9.53e-06 / ex 7.7e-06 | agree 5.4e-06 / ex 1.3e-14 | combination | truth |
| switching/boost/two_phase_interleaved_boost.yaml | auto | agree 5.56e-06 / ex 4.8e-06 | agree 6.29e-06 / ex 5.5e-15 | combination | truth |
| switching/buck/buck_dcm.yaml | auto | agree 7.7e-06 / ex 3e-06 | agree 4.82e-06 / ex 1.6e-14 | combination | truth |
| switching/buck/buck_hard_switching_reference.yaml | auto | agree 2.27e-06 / ex 8.8e-07 | agree 5.25e-06 / ex 1.3e-14 | combination | truth |
| switching/buck/buck_switching.yaml | auto | agree 1.36e-06 / ex 4.9e-07 | agree 5.32e-06 / ex 1.2e-14 | combination | truth |
| switching/buck/buck_with_current_sense.yaml | auto | agree 3.15e-06 / ex 2.3e-06 | agree 5.63e-06 / ex 5.4e-15 | combination | truth |
| switching/buck/cascaded_buck_buck.yaml | auto | agree 1.42e-05 / ex 9.8e-06 | agree 5.44e-06 / ex 1.2e-14 | combination | truth |
| switching/buck/high_freq_gan_buck.yaml | auto | agree 8.55e-05 / ex 2.7e-05 | agree 6.93e-06 / ex 9.6e-15 | combination | truth |
| switching/buck/sync_buck_500khz.yaml | auto | agree 1.71e-05 / ex 1.4e-05 | agree 6.67e-06 / ex 5.5e-15 | combination | truth |
| switching/buck/synchronous_buck.yaml | auto | agree 3.14e-06 / ex 2.3e-06 | agree 5.59e-06 / ex 2.7e-15 | combination | truth |
| switching/buck/textbook_erickson_buck_ccm.yaml | auto | agree 9.87e-06 / ex 1.7e-06 | agree 6.72e-06 / ex 1.2e-14 | combination | truth |
| switching/buck/triple_sync_buck.yaml | auto | agree 3.18e-06 / ex 2.3e-06 | agree 4.59e-06 / ex 2.2e-14 | combination | truth |
| switching/buck_boost/bidirectional_buck_boost.yaml | auto | agree 3.15e-06 / ex 2.3e-06 | agree 5.61e-06 / ex 4.4e-15 | combination | truth |
| switching/buck_boost/buck_boost.yaml | auto | agree 7.97e-06 / ex 7.6e-06 | agree 4.97e-06 / ex 6.5e-15 | combination | truth |
| switching/buck_boost/cuk_converter.yaml | auto | agree 9.25e-05 / ex 9.3e-05 | agree 4.32e-06 / ex 3.2e-14 | combination | truth |
| switching/buck_boost/sepic.yaml | auto | agree 2.96e-05 / ex 1.5e-05 | agree 6.37e-07 / ex 1.6e-13 | combination | truth |
| switching/inverter/half_bridge_inverter.yaml | auto | agree 9.06e-07 / ex 9.4e-07 | agree 5.8e-06 / ex 2.5e-15 | combination | truth |
| switching/inverter/half_bridge_inverter_lc.yaml | auto | agree 0.000303 / ex 0.00035 | agree 7.85e-06 / ex 2.7e-13 | auto:route | truth |
| switching/inverter/half_bridge_lc_filter.yaml | auto | agree 5e-06 / ex 4e-06 | agree 5.82e-06 / ex 1.2e-14 | combination | truth |
| switching/inverter/lcc_resonant_inverter.yaml | auto | agree 8.37e-05 / ex 5.4e-05 | agree 5.38e-06 / ex 1.5e-14 | combination | truth |
| switching/inverter/ll10_full_bridge_inverter_lc.yaml | auto | agree 9.18e-05 / ex 4.9e-05 | agree 0.000128 / ex 1.4e-11 | auto:route | truth |
| switching/isolated/active_clamp_forward.yaml | auto | agree 0.000117 / ex 0.00012 | agree 3.66e-06 / ex 2.5e-08 | combination | truth |
| switching/isolated/dual_active_bridge.yaml | auto | agree 3.49e-05 / ex 9.1e-06 | agree 3.68e-06 / ex 1.5e-11 | combination | truth |
| switching/isolated/flyback_converter.yaml | auto | agree 4.29e-05 / ex 3.6e-05 | agree 6.69e-06 / ex 3.9e-07 | combination | truth |
| switching/isolated/forward_converter.yaml | auto | agree 2.17e-05 / ex 1.2e-06 | agree 6.13e-06 / ex 1e-09 | combination | truth |
| switching/isolated/full_bridge_dc.yaml | auto | agree 0.000141 / ex 5e-09 | agree 2.66e-05 / ex 5.1e-17 | auto:route | truth |
| switching/isolated/synchronous_flyback.yaml | auto | agree 9.15e-06 / ex 6e-06 | agree 6.2e-06 / ex 7.2e-11 | combination | truth |
| switching/multilevel/flying_capacitor_3level.yaml | auto | agree 6.78e-05 / ex 1.2e-05 | agree 3.74e-05 / ex 6.4e-14 | combination | truth |
| switching/multilevel/npc_three_level_halfbridge.yaml | auto | agree 7.9e-06 / ex 1.8e-06 | agree 5.55e-06 / ex 4.1e-13 | combination | truth |
| switching/multilevel/t_type_3level_halfbridge.yaml | auto | agree 0.000112 / ex 9.3e-07 | agree 4.24e-05 / ex 7e-15 | combination | truth |
| thermal/long_run_drift_buck.yaml | auto | agree 2.58e-06 / ex 5.7e-07 | agree 7.04e-06 / ex 1.5e-13 | combination | truth |
| three_phase/interleaved_buck_3ph.yaml | auto | agree 6.64e-08 / ex 1.5e-07 | agree 1.28e-06 / ex 1.1e-14 | auto:route | truth |
| three_phase/three_phase_balanced_rl_load.yaml | auto | agree 4.22e-05 / ex 3.4e-06 | agree 6.04e-07 / ex 5.2e-12 | auto:route | truth |
| three_phase/three_phase_diode_rectifier.yaml | auto | agree 5.51e-05 / ex 3e-06 | agree 5.47e-06 / ex 5.2e-14 | auto:route | truth |

## engine='auto' under the profile

`auto` = today (no dt: trbdf2, or a refusal naming the blocker); `auto+next` = the
release route (no dt). The exact engine without dt samples at its 10 us default ceiling (1 us when the gate names no edge),
so its strict cell can disagree where `dsed+next` (netlist dt as the output grid) agrees.

| circuit | today: engine / outcome | next: engine / outcome | route reason |
|---|---|---|---|
| closed_loop/cl_boost_pi.yaml | trbdf2 / agree 1.31e-06 | dsed / agree 4.27e-06 |  |
| closed_loop/cl_buck_aggressive_pi.yaml | trbdf2 / agree 3.72e-05 | dsed / agree 6.73e-06 |  |
| closed_loop/cl_buck_pi.yaml | trbdf2 / agree 1.88e-05 | dsed / agree 2.98e-06 |  |
| closed_loop/ll14_buck_closed_loop.yaml | trbdf2 / agree 0.000547 | trbdf2 / agree 0.000548 | segment not exact: device |
| closed_loop/ll15_boost_closed_loop.yaml | trbdf2 / agree 0.000264 | trbdf2 / agree 0.000264 | segment not exact: device |
| closed_loop/ll16_flyback_closed_loop.yaml | trbdf2 / agree 0.000363 | trbdf2 / agree 0.000362 | segment not exact: device |
| diodes/back_to_back_diodes_sine.yaml | trbdf2 / agree 0 / ex 2e-16 | dsed / agree 1.15e-05 / ex 2e-16 |  |
| diodes/diode_bridge_rectifier.yaml | trbdf2 / agree 3.22e-05 / ex 3.5e-07 | dsed / agree 6.59e-06 / ex 3e-14 |  |
| diodes/diode_clamp_overvoltage.yaml | trbdf2 / agree 0 | dsed / agree 6.14e-12 |  |
| diodes/diode_half_wave_rl_load.yaml | trbdf2 / agree 1.58e-05 / ex 5.6e-06 | dsed / agree 6.92e-06 / ex 4.9e-06 |  |
| diodes/diode_rectifier.yaml | trbdf2 / agree 0.000183 / ex 2.4e-07 | dsed / agree 2.78e-06 / ex 3.1e-14 |  |
| diodes/half_wave_rectifier_sine.yaml | trbdf2 / agree 2.36e-05 / ex 7.3e-08 | dsed / agree 4.63e-06 / ex 3.6e-13 |  |
| diodes/voltage_doubler.yaml | trbdf2 / agree 5.05e-06 / ex 3.5e-08 | dsed / agree 5.91e-07 / ex 2.5e-13 |  |
| linear/bridge_resistive_pure.yaml | trbdf2 / agree 0 / ex 0 | dsed / agree 1.33e-13 / ex 1.3e-13 |  |
| linear/capacitor_natural_discharge.yaml | trbdf2 / agree 5.63e-06 / ex 5.5e-06 | dsed / agree 6.1e-07 / ex 1.1e-14 |  |
| linear/current_divider_3way.yaml | trbdf2 / agree 0 / ex 0 | dsed / agree 0 / ex 0 |  |
| linear/current_source_rc.yaml | trbdf2 / agree 0 / ex 1.1e-06 | dsed / agree 1.63e-10 / ex 1.1e-06 |  |
| linear/current_source_rl_step.yaml | trbdf2 / agree 2.14e-06 / ex 4.6e-07 | dsed / agree 1.09e-06 / ex 2e-15 |  |
| linear/lc_low_pass_sine.yaml | trbdf2 / agree 0.000118 / ex 0.00011 | dsed / agree 1.23e-06 / ex 8.5e-14 |  |
| linear/ll01_resistive_divider_dc.yaml | trbdf2 / agree 0 / ex 4e-07 | dsed / agree 4.52e-10 / ex 4e-07 |  |
| linear/ll02_rc_pulse_charge.yaml | trbdf2 / agree 1.48e-07 | dsed / agree 4.3e-07 |  |
| linear/ll03_rl_step.yaml | trbdf2 / agree 6.38e-06 | dsed / agree 5.57e-07 |  |
| linear/pulse_voltage_rl.yaml | trbdf2 / agree 3.04e-05 | dsed / agree 1.29e-06 |  |
| linear/rc_dc.yaml | trbdf2 / agree 6.84e-07 / ex 2.3e-07 | dsed / agree 4.28e-07 / ex 1.4e-14 |  |
| linear/rc_step.yaml | trbdf2 / agree 6.84e-07 | dsed / agree 4.27e-07 |  |
| linear/rl_natural_response.yaml | trbdf2 / agree 6.66e-06 / ex 5.1e-06 | dsed / agree 1.23e-06 / ex 1.5e-15 |  |
| linear/rl_step.yaml | trbdf2 / agree 1.75e-06 | dsed / agree 1.17e-06 |  |
| linear/rlc_series_resonant.yaml | trbdf2 / agree 0.000354 / ex 0.00034 | dsed / agree 4.04e-06 / ex 5.2e-13 |  |
| linear/sine_voltage_rc.yaml | trbdf2 / agree 0.000111 / ex 2.4e-05 | dsed / agree 2.93e-06 / ex 3.1e-15 |  |
| linear/two_stage_rc_lowpass.yaml | trbdf2 / agree 3.44e-06 | dsed / agree 4.43e-07 |  |
| linear/voltage_divider_3stage.yaml | trbdf2 / agree 0 / ex 0 | dsed / agree 0 / ex 0 |  |
| magnetics/core_loss_steinmetz_sine.yaml | trbdf2 / agree 0.000123 / ex 4.1e-05 | dsed / agree 3.7e-06 / ex 5.8e-14 |  |
| magnetics/coupled_inductor_step.yaml | trbdf2 / agree 3.76e-06 | dsed / agree 6.83e-07 |  |
| magnetics/half_bridge_transformer_dc.yaml | trbdf2 / agree 2.08e-05 / ex 1.5e-05 | dsed / agree 2.66e-06 / ex 6.7e-13 |  |
| magnetics/inductor_chain_step.yaml | trbdf2 / agree 9.05e-07 | dsed / agree 4.82e-07 |  |
| magnetics/transformer_leakage.yaml | trbdf2 / agree 0.000231 / ex 1.8e-05 | dsed / agree 2.88e-06 / ex 2.3e-14 |  |
| magnetics/transformer_step_up.yaml | trbdf2 / agree 0 / ex 4e-16 | dsed / agree 4.64e-06 / ex 4e-16 |  |
| motors/motor_dc_brush_step_load.yaml | trbdf2 / agree 6.06e-06 / ex 2.3e-06 | dsed / agree 3.62e-07 / ex 1e-13 |  |
| motors/motor_induction_locked_rotor.yaml | trbdf2 / agree 5.1e-05 / ex 1.4e-05 | dsed / agree 1.63e-06 / ex 6.4e-14 |  |
| motors/motor_pmsm_dq_open_loop.yaml | trbdf2 / agree 6.51e-06 / ex 2.5e-06 | dsed / agree 4.45e-07 / ex 1.3e-13 |  |
| resonance/bode_rlc_low_pass.yaml | trbdf2 / agree 0 / ex 0 | dsed / agree 0 / ex 0 |  |
| resonance/driven_lc_series_resonance.yaml | trbdf2 / agree 0.000217 / ex 0.0002 | dsed / agree 3.06e-06 / ex 1.8e-12 |  |
| resonance/lc_resonant_tank.yaml | trbdf2 / agree 5.29e-05 | dsed / agree 6.55e-06 |  |
| resonance/ll04_rlc_underdamped.yaml | trbdf2 / agree 2.99e-05 | dsed / agree 7.39e-07 |  |
| resonance/rl_high_pass_sine.yaml | trbdf2 / agree 7.56e-05 / ex 9.9e-06 | dsed / agree 2.6e-06 / ex 3.4e-15 |  |
| resonance/rlc_overdamped_step.yaml | trbdf2 / agree 5.77e-06 | dsed / agree 2.85e-07 |  |
| resonance/rlc_parallel_step.yaml | trbdf2 / agree 1.61e-05 | dsed / agree 8.08e-07 |  |
| resonance/rlc_step.yaml | trbdf2 / agree 1.31e-05 | dsed / agree 4.41e-07 |  |
| stress/high_q_rlc_ringdown.yaml | trbdf2 / disagree 0.00566 / ex 0.0057 | dsed / agree 5.61e-05 / ex 9.7e-13 |  |
| stress/periodic_rc_pwm.yaml | trbdf2 / agree 2e-07 / ex 5.1e-08 | dsed / agree 5.12e-06 / ex 1.3e-15 |  |
| stress/stiff_rc_high_freq_switching.yaml | trbdf2 / agree 1.43e-05 / ex 1e-05 | dsed / agree 6.89e-06 / ex 2.4e-14 |  |
| stress/stiff_rlc.yaml | trbdf2 / agree 0 / ex 2.7e-09 | dsed / agree 3.56e-09 / ex 3.2e-09 |  |
| stress/stiff_two_timescale_rc.yaml | trbdf2 / agree 7.79e-07 | dsed / agree 7.75e-07 |  |
| switching/basic/chopper_with_rc_snubber.yaml | trbdf2 / agree 3.81e-05 / ex 4.6e-08 | dsed / agree 2.91e-05 / ex 4.8e-15 |  |
| switching/basic/mosfet_nmos_dc.yaml | trbdf2 / agree 0 | trbdf2 / agree 0 | segment not exact: device |
| switching/basic/pwm_voltage_rc.yaml | trbdf2 / agree 8.98e-09 / ex 8.9e-09 | dsed / agree 1.57e-08 / ex 5.4e-14 |  |
| switching/basic/pwm_voltage_rl_filter.yaml | trbdf2 / agree 6.53e-05 / ex 4.9e-06 | dsed / agree 6.05e-06 / ex 4.9e-15 |  |
| switching/basic/snubber_rc.yaml | trbdf2 / agree 1.65e-07 | dsed / agree 2.16e-08 |  |
| switching/basic/vcswitch_resistor.yaml | trbdf2 / agree 0 / ex 2e-09 | dsed / agree 3.16e-06 / ex 2e-09 |  |
| switching/boost/boost_classic.yaml | trbdf2 / agree 9.53e-06 / ex 7.7e-06 | dsed / agree 5.4e-06 / ex 1.3e-14 |  |
| switching/boost/boost_pfc_open_loop.yaml | trbdf2 / agree 1.72e-07 / ex 1.7e-07 | dsed / agree 2.7e-06 / ex 5.7e-14 |  |
| switching/boost/boost_switching_complex.yaml | trbdf2 / agree 1.05e-05 / ex 7e-06 | dsed / agree 2.32e-06 / ex 7.5e-16 |  |
| switching/boost/cascaded_boost_buck.yaml | trbdf2 / agree 5.95e-05 / ex 3.2e-06 | dsed / agree 6.54e-06 / ex 1.8e-13 |  |
| switching/boost/ll07_boost_vcswitch.yaml | trbdf2 / agree 0.000117 / ex 0.00016 | dsed / agree 0.000181 / ex 1.8e-10 |  |
| switching/boost/ll12_pfc_boost_continuous.yaml | trbdf2 / agree 0.000115 | trbdf2 / agree 0.000115 | segment not exact: device |
| switching/boost/synchronous_boost.yaml | trbdf2 / agree 8.19e-08 / ex 2.7e-07 | dsed / agree 4.45e-06 / ex 5.5e-12 |  |
| switching/boost/textbook_erickson_boost_ccm.yaml | trbdf2 / agree 9.53e-06 / ex 7.7e-06 | dsed / agree 5.4e-06 / ex 1.3e-14 |  |
| switching/boost/two_phase_interleaved_boost.yaml | trbdf2 / agree 5.56e-06 / ex 4.8e-06 | dsed / agree 6.29e-06 / ex 5.5e-15 |  |
| switching/buck/buck_dcm.yaml | trbdf2 / agree 7.7e-06 / ex 3e-06 | dsed / agree 4.82e-06 / ex 1.6e-14 |  |
| switching/buck/buck_hard_switching_reference.yaml | trbdf2 / agree 2.27e-06 / ex 8.8e-07 | dsed / agree 5.25e-06 / ex 1.3e-14 |  |
| switching/buck/buck_mosfet_nonlinear.yaml | trbdf2 / agree 5.08e-07 | trbdf2 / agree 5.08e-07 | segment not exact: device |
| switching/buck/buck_switching.yaml | trbdf2 / agree 1.36e-06 / ex 4.9e-07 | dsed / agree 5.32e-06 / ex 1.2e-14 |  |
| switching/buck/buck_with_current_sense.yaml | trbdf2 / agree 3.15e-06 / ex 2.3e-06 | dsed / agree 5.63e-06 / ex 5.4e-15 |  |
| switching/buck/cascaded_buck_buck.yaml | trbdf2 / agree 1.42e-05 / ex 9.8e-06 | dsed / agree 5.44e-06 / ex 1.2e-14 |  |
| switching/buck/high_freq_gan_buck.yaml | trbdf2 / agree 8.55e-05 / ex 2.7e-05 | dsed / agree 6.93e-06 / ex 9.6e-15 |  |
| switching/buck/ll06_buck_vcswitch.yaml | trbdf2 / agree 8.21e-05 / ex 6.3e-06 | dsed / agree 0.000171 / ex 8.8e-12 |  |
| switching/buck/ll08_buck_mosfet.yaml | trbdf2 / agree 0.00011 | trbdf2 / agree 0.000101 | segment not exact: device |
| switching/buck/sync_buck_500khz.yaml | trbdf2 / agree 1.71e-05 / ex 1.4e-05 | dsed / agree 6.67e-06 / ex 5.5e-15 |  |
| switching/buck/synchronous_buck.yaml | trbdf2 / agree 3.14e-06 / ex 2.3e-06 | dsed / agree 5.59e-06 / ex 2.7e-15 |  |
| switching/buck/textbook_erickson_buck_ccm.yaml | trbdf2 / agree 9.87e-06 / ex 1.7e-06 | dsed / agree 6.72e-06 / ex 1.2e-14 |  |
| switching/buck/triple_sync_buck.yaml | trbdf2 / agree 3.18e-06 / ex 2.3e-06 | dsed / agree 4.59e-06 / ex 2.2e-14 |  |
| switching/buck_boost/bidirectional_buck_boost.yaml | trbdf2 / agree 3.15e-06 / ex 2.3e-06 | dsed / agree 5.61e-06 / ex 4.4e-15 |  |
| switching/buck_boost/buck_boost.yaml | trbdf2 / agree 7.97e-06 / ex 7.6e-06 | dsed / agree 4.97e-06 / ex 6.5e-15 |  |
| switching/buck_boost/cuk_converter.yaml | trbdf2 / agree 9.25e-05 / ex 9.3e-05 | dsed / agree 4.32e-06 / ex 3.2e-14 |  |
| switching/buck_boost/sepic.yaml | trbdf2 / agree 2.96e-05 / ex 1.5e-05 | dsed / agree 6.37e-07 / ex 1.6e-13 |  |
| switching/inverter/half_bridge_inverter.yaml | trbdf2 / agree 9.06e-07 / ex 9.4e-07 | dsed / agree 5.8e-06 / ex 2.5e-15 |  |
| switching/inverter/half_bridge_inverter_lc.yaml | trbdf2 / agree 0.000303 / ex 0.00035 | dsed / agree 7.85e-06 / ex 2.7e-13 |  |
| switching/inverter/half_bridge_lc_filter.yaml | trbdf2 / agree 5e-06 / ex 4e-06 | dsed / agree 5.82e-06 / ex 1.2e-14 |  |
| switching/inverter/lcc_resonant_inverter.yaml | trbdf2 / agree 8.37e-05 / ex 5.4e-05 | dsed / agree 5.38e-06 / ex 1.5e-14 |  |
| switching/inverter/ll09_half_bridge_inverter_leg.yaml | trbdf2 / agree 4.27e-05 / ex 3.2e-06 | dsed / agree 0.000167 / ex 4.8e-11 |  |
| switching/inverter/ll10_full_bridge_inverter_lc.yaml | trbdf2 / agree 9.18e-05 / ex 4.9e-05 | dsed / agree 0.000128 / ex 1.4e-11 |  |
| switching/isolated/active_clamp_forward.yaml | trbdf2 / agree 0.000117 / ex 0.00012 | dsed / agree 3.66e-06 / ex 2.5e-08 |  |
| switching/isolated/dual_active_bridge.yaml | trbdf2 / agree 3.49e-05 / ex 9.1e-06 | dsed / agree 3.68e-06 / ex 1.5e-11 |  |
| switching/isolated/flyback_converter.yaml | trbdf2 / agree 4.29e-05 / ex 3.6e-05 | dsed / agree 6.69e-06 / ex 3.9e-07 |  |
| switching/isolated/forward_converter.yaml | trbdf2 / agree 2.17e-05 / ex 1.2e-06 | dsed / agree 6.13e-06 / ex 1e-09 |  |
| switching/isolated/full_bridge_dc.yaml | trbdf2 / agree 0.000141 / ex 5e-09 | dsed / agree 2.66e-05 / ex 5.1e-17 |  |
| switching/isolated/ll11_llc_resonant_converter.yaml | - / error error | trbdf2 / error error | segment not exact: device |
| switching/isolated/ll13_active_clamp_forward.yaml | trbdf2 / disagree 0.00133 | trbdf2 / disagree 0.00155 | segment not exact: device |
| switching/isolated/synchronous_flyback.yaml | trbdf2 / agree 9.15e-06 / ex 6e-06 | dsed / agree 6.2e-06 / ex 7.2e-11 |  |
| switching/multilevel/cascaded_h_bridge_5level.yaml | trbdf2 / agree 0 / ex 6e-16 | dsed / agree 5.72e-05 / ex 1.9e-16 |  |
| switching/multilevel/flying_capacitor_3level.yaml | trbdf2 / agree 6.78e-05 / ex 1.2e-05 | dsed / agree 3.74e-05 / ex 6.4e-14 |  |
| switching/multilevel/mmc_4cell_chain.yaml | trbdf2 / agree 0 / ex 1.2e-13 | dsed / agree 5.76e-05 / ex 1.2e-13 |  |
| switching/multilevel/npc_three_level_halfbridge.yaml | trbdf2 / agree 7.9e-06 / ex 1.8e-06 | dsed / agree 5.55e-06 / ex 4.1e-13 |  |
| switching/multilevel/t_type_3level_halfbridge.yaml | trbdf2 / agree 0.000112 / ex 9.3e-07 | dsed / agree 4.24e-05 / ex 7e-15 |  |
| thermal/electrothermal_resistor_self_heating.yaml | trbdf2 / agree 0 / ex 0 | dsed / disagree 0.00178 / ex 0 |  |
| thermal/long_run_drift_buck.yaml | trbdf2 / agree 2.58e-06 / ex 5.7e-07 | dsed / agree 7.04e-06 / ex 1.5e-13 |  |
| three_phase/interleaved_buck_3ph.yaml | trbdf2 / agree 6.64e-08 / ex 1.5e-07 | dsed / agree 1.28e-06 / ex 1.1e-14 |  |
| three_phase/three_phase_balanced_rl_load.yaml | trbdf2 / agree 4.22e-05 / ex 3.4e-06 | dsed / agree 6.04e-07 / ex 5.2e-12 |  |
| three_phase/three_phase_diode_rectifier.yaml | trbdf2 / agree 5.51e-05 / ex 3e-06 | dsed / agree 5.47e-06 / ex 5.2e-14 |  |

Circuits whose engine changes under `auto`: **101**.

## Every cell (err_rms / exact, or outcome)

| circuit | pwl | pwl+restart | pwl+split | trbdf2 | dsed | pwl:restart | pwl:split | pwl:mode_auto | pwl:tick_state | pwl:dev_interface | pwl:branch | pwl+next | trbdf2:mode_auto | trbdf2:tick_state | trbdf2:dev_interface | trbdf2:branch | trbdf2+next | dsed:mode_auto | dsed:tick_state | dsed:output_grid | dsed:branch | dsed+next | trbdf2+esdirk43 | trbdf2+next+esdirk43 | auto | auto:route | auto+next |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| analytic:buck_ccm | 0.000256 / ex 0.00026 | 1.23e-08 / ex 1.2e-08 | 1.23e-08 / ex 1.2e-08 | 3.37e-07 / ex 3.4e-07 | 3.92e-15 / ex 3.9e-15 | 1.23e-08 / ex 1.2e-08 | 1.23e-08 / ex 1.2e-08 | 0.000256 / ex 0.00026 | 0.000256 / ex 0.00026 | 0.000256 / ex 0.00026 | 0.000256 / ex 0.00026 | 1.23e-08 / ex 1.2e-08 | 3.37e-07 / ex 3.4e-07 | 3.37e-07 / ex 3.4e-07 | 3.37e-07 / ex 3.4e-07 | 3.37e-07 / ex 3.4e-07 | 3.37e-07 / ex 3.4e-07 | 3.92e-15 / ex 3.9e-15 | 3.92e-15 / ex 3.9e-15 | 2.94e-14 / ex 2.9e-14 | 3.92e-15 / ex 3.9e-15 | 2.94e-14 / ex 2.9e-14 | 1.74e-08 / ex 1.7e-08 | 1.73e-08 / ex 1.7e-08 | n/a | 3.92e-15 / ex 3.9e-15 | 2.94e-14 / ex 2.9e-14 |
| analytic:buck_dcm | 0.000538 / ex 0.00054 | 5.32e-08 / ex 5.3e-08 | 1.87e-08 / ex 1.9e-08 | 5.62e-07 / ex 5.6e-07 | 7.34e-10 / ex 7.3e-10 | 5.32e-08 / ex 5.3e-08 | 1.87e-08 / ex 1.9e-08 | 0.000538 / ex 0.00054 | 0.000538 / ex 0.00054 | 0.000538 / ex 0.00054 | 0.000538 / ex 0.00054 | 1.87e-08 / ex 1.9e-08 | 5.62e-07 / ex 5.6e-07 | 5.62e-07 / ex 5.6e-07 | 5.62e-07 / ex 5.6e-07 | 5.62e-07 / ex 5.6e-07 | 5.62e-07 / ex 5.6e-07 | 7.34e-10 / ex 7.3e-10 | 7.34e-10 / ex 7.3e-10 | 7.34e-10 / ex 7.3e-10 | 7.34e-10 / ex 7.3e-10 | 7.34e-10 / ex 7.3e-10 | 3.19e-08 / ex 3.2e-08 | 3.19e-08 / ex 3.2e-08 | n/a | 7.34e-10 / ex 7.3e-10 | 7.34e-10 / ex 7.3e-10 |
| analytic:coupled_k099 | 0.000765 / ex 0.00076 | 8.27e-06 / ex 8.3e-06 | 8.27e-06 / ex 8.3e-06 | 8.13e-07 / ex 8.1e-07 | 1.32e-14 / ex 1.3e-14 | 8.27e-06 / ex 8.3e-06 | 8.27e-06 / ex 8.3e-06 | 0.000765 / ex 0.00076 | 0.000765 / ex 0.00076 | 0.000765 / ex 0.00076 | 0.000765 / ex 0.00076 | 8.27e-06 / ex 8.3e-06 | 8.13e-07 / ex 8.1e-07 | 8.13e-07 / ex 8.1e-07 | 8.13e-07 / ex 8.1e-07 | 8.13e-07 / ex 8.1e-07 | 8.13e-07 / ex 8.1e-07 | 1.32e-14 / ex 1.3e-14 | 1.32e-14 / ex 1.3e-14 | 1.35e-14 / ex 1.3e-14 | 1.32e-14 / ex 1.3e-14 | 1.35e-14 / ex 1.3e-14 | 1.04e-07 / ex 1e-07 | 1.04e-07 / ex 1e-07 | n/a | 1.32e-14 / ex 1.3e-14 | 1.35e-14 / ex 1.3e-14 |
| analytic:gapped_core_rl | 0.00159 / ex 0.0016 | 1.84e-06 / ex 1.8e-06 | 1.84e-06 / ex 1.8e-06 | 2.26e-07 / ex 2.3e-07 | 4.37e-16 / ex 4.4e-16 | 1.84e-06 / ex 1.8e-06 | 1.84e-06 / ex 1.8e-06 | 0.00159 / ex 0.0016 | 0.00159 / ex 0.0016 | 0.00159 / ex 0.0016 | 0.00159 / ex 0.0016 | 1.84e-06 / ex 1.8e-06 | 2.26e-07 / ex 2.3e-07 | 2.26e-07 / ex 2.3e-07 | 2.26e-07 / ex 2.3e-07 | 2.26e-07 / ex 2.3e-07 | 2.26e-07 / ex 2.3e-07 | 4.37e-16 / ex 4.4e-16 | 4.37e-16 / ex 4.4e-16 | 1.82e-15 / ex 1.8e-15 | 4.37e-16 / ex 4.4e-16 | 1.82e-15 / ex 1.8e-15 | 8.76e-09 / ex 8.8e-09 | 8.76e-09 / ex 8.8e-09 | n/a | 4.37e-16 / ex 4.4e-16 | 1.82e-15 / ex 1.8e-15 |
| analytic:halfwave_r | 5.38e-09 / ex 5.4e-09 | 5.38e-09 / ex 5.4e-09 | 5.38e-09 / ex 5.4e-09 | 2.59e-08 / ex 2.6e-08 | 5.38e-09 / ex 5.4e-09 | 5.38e-09 / ex 5.4e-09 | 5.38e-09 / ex 5.4e-09 | 5.38e-09 / ex 5.4e-09 | 5.38e-09 / ex 5.4e-09 | 5.38e-09 / ex 5.4e-09 | 5.38e-09 / ex 5.4e-09 | 5.38e-09 / ex 5.4e-09 | 2.59e-08 / ex 2.6e-08 | 2.59e-08 / ex 2.6e-08 | 2.59e-08 / ex 2.6e-08 | 2.59e-08 / ex 2.6e-08 | 2.59e-08 / ex 2.6e-08 | 5.38e-09 / ex 5.4e-09 | 5.38e-09 / ex 5.4e-09 | 5.38e-09 / ex 5.4e-09 | 5.38e-09 / ex 5.4e-09 | 5.38e-09 / ex 5.4e-09 | 1.41e-08 / ex 1.4e-08 | 1.41e-08 / ex 1.4e-08 | n/a | 5.38e-09 / ex 5.4e-09 | 5.38e-09 / ex 5.4e-09 |
| analytic:halfwave_rl | 9.65e-06 / ex 9.6e-06 | 1.44e-06 / ex 1.4e-06 | 1.44e-06 / ex 1.4e-06 | 3.76e-06 / ex 3.8e-06 | 5.69e-09 / ex 5.7e-09 | 1.44e-06 / ex 1.4e-06 | 1.44e-06 / ex 1.4e-06 | 9.65e-06 / ex 9.6e-06 | 9.65e-06 / ex 9.6e-06 | 9.65e-06 / ex 9.6e-06 | 9.65e-06 / ex 9.6e-06 | 1.44e-06 / ex 1.4e-06 | 3.76e-06 / ex 3.8e-06 | 3.76e-06 / ex 3.8e-06 | 3.76e-06 / ex 3.8e-06 | 3.76e-06 / ex 3.8e-06 | 3.76e-06 / ex 3.8e-06 | 5.69e-09 / ex 5.7e-09 | 5.69e-09 / ex 5.7e-09 | 5.69e-09 / ex 5.7e-09 | 5.69e-09 / ex 5.7e-09 | 5.69e-09 / ex 5.7e-09 | 5.7e-09 / ex 5.7e-09 | 5.7e-09 / ex 5.7e-09 | n/a | 5.69e-09 / ex 5.7e-09 | 5.69e-09 / ex 5.7e-09 |
| analytic:hard_switched_rc | 0.01 / ex 0.01 | 4.92e-05 / ex 4.9e-05 | 4.92e-05 / ex 4.9e-05 | 2.22e-06 / ex 2.2e-06 | 7.58e-14 / ex 7.6e-14 | 4.92e-05 / ex 4.9e-05 | 4.92e-05 / ex 4.9e-05 | 0.01 / ex 0.01 | 0.01 / ex 0.01 | 0.01 / ex 0.01 | 0.01 / ex 0.01 | 4.92e-05 / ex 4.9e-05 | 2.22e-06 / ex 2.2e-06 | 2.22e-06 / ex 2.2e-06 | 2.22e-06 / ex 2.2e-06 | 2.22e-06 / ex 2.2e-06 | 2.22e-06 / ex 2.2e-06 | 7.58e-14 / ex 7.6e-14 | 7.58e-14 / ex 7.6e-14 | 2.53e-10 / ex 2.5e-10 | 7.58e-14 / ex 7.6e-14 | 2.53e-10 / ex 2.5e-10 | 3.48e-06 / ex 3.5e-06 | 3.48e-06 / ex 3.5e-06 | n/a | 7.58e-14 / ex 7.6e-14 | 2.53e-10 / ex 2.5e-10 |
| analytic:ideal_transformer | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 3.05e-17 / ex 3e-17 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 3.05e-17 / ex 3e-17 | 3.05e-17 / ex 3e-17 | 3.05e-17 / ex 3e-17 | 3.05e-17 / ex 3e-17 | 3.05e-17 / ex 3e-17 | 0 / ex 0 | 0 / ex 0 | n/a | 3.05e-17 / ex 3e-17 | 3.05e-17 / ex 3e-17 |
| analytic:igbt_knee | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | refused | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | refused | refused | refused | refused | refused | 0 / ex 0 | 0 / ex 0 | n/a | 0 / ex 0 | 0 / ex 0 |
| analytic:im_free_acceleration | 9.62e-05 / ex 9.6e-05 | 6e-06 / ex 6e-06 | 6e-06 / ex 6e-06 | 4.25e-05 / ex 4.2e-05 | refused | 6e-06 / ex 6e-06 | 6e-06 / ex 6e-06 | 9.62e-05 / ex 9.6e-05 | 9.62e-05 / ex 9.6e-05 | 9.62e-05 / ex 9.6e-05 | 9.62e-05 / ex 9.6e-05 | 6e-06 / ex 6e-06 | 4.25e-05 / ex 4.2e-05 | 4.25e-05 / ex 4.2e-05 | 4.25e-05 / ex 4.2e-05 | 4.25e-05 / ex 4.2e-05 | 4.25e-05 / ex 4.2e-05 | refused | refused | refused | refused | refused | refused | refused | n/a | 4.25e-05 / ex 4.2e-05 | 4.25e-05 / ex 4.2e-05 |
| analytic:im_locked_rotor | 1.35e-06 / ex 1.3e-06 | 1.35e-06 / ex 1.3e-06 | 1.35e-06 / ex 1.3e-06 | 8.23e-06 / ex 8.2e-06 | refused | 1.35e-06 / ex 1.3e-06 | 1.35e-06 / ex 1.3e-06 | 1.35e-06 / ex 1.3e-06 | 1.35e-06 / ex 1.3e-06 | 1.35e-06 / ex 1.3e-06 | 1.35e-06 / ex 1.3e-06 | 1.35e-06 / ex 1.3e-06 | 8.23e-06 / ex 8.2e-06 | 8.23e-06 / ex 8.2e-06 | 8.23e-06 / ex 8.2e-06 | 8.23e-06 / ex 8.2e-06 | 8.23e-06 / ex 8.2e-06 | refused | refused | refused | refused | refused | refused | refused | n/a | 8.23e-06 / ex 8.2e-06 | 8.23e-06 / ex 8.2e-06 |
| analytic:lc_lossless | 0.00178 / ex 0.0018 | 3.31e-05 / ex 3.3e-05 | 3.31e-05 / ex 3.3e-05 | 5.5e-05 / ex 5.5e-05 | 5e-14 / ex 5e-14 | 3.31e-05 / ex 3.3e-05 | 3.31e-05 / ex 3.3e-05 | 0.00178 / ex 0.0018 | 0.00178 / ex 0.0018 | 0.00178 / ex 0.0018 | 0.00178 / ex 0.0018 | 3.31e-05 / ex 3.3e-05 | 5.5e-05 / ex 5.5e-05 | 5.5e-05 / ex 5.5e-05 | 5.5e-05 / ex 5.5e-05 | 5.5e-05 / ex 5.5e-05 | 5.5e-05 / ex 5.5e-05 | 5e-14 / ex 5e-14 | 5e-14 / ex 5e-14 | 7.95e-14 / ex 8e-14 | 5e-14 / ex 5e-14 | 7.95e-14 / ex 8e-14 | 2.21e-07 / ex 2.2e-07 | 2.21e-07 / ex 2.2e-07 | n/a | 5e-14 / ex 5e-14 | 7.95e-14 / ex 8e-14 |
| analytic:pulse_rc | 3.25e-06 / ex 3.2e-06 | 3.25e-06 / ex 3.2e-06 | 3.25e-06 / ex 3.2e-06 | 1.01e-06 / ex 1e-06 | 7.8e-14 / ex 7.8e-14 | 3.25e-06 / ex 3.2e-06 | 3.25e-06 / ex 3.2e-06 | 3.25e-06 / ex 3.2e-06 | 3.25e-06 / ex 3.2e-06 | 3.25e-06 / ex 3.2e-06 | 3.25e-06 / ex 3.2e-06 | 3.25e-06 / ex 3.2e-06 | 1.01e-06 / ex 1e-06 | 1.01e-06 / ex 1e-06 | 1.01e-06 / ex 1e-06 | 1.01e-06 / ex 1e-06 | 1.01e-06 / ex 1e-06 | 7.8e-14 / ex 7.8e-14 | 7.8e-14 / ex 7.8e-14 | 7.8e-14 / ex 7.8e-14 | 7.8e-14 / ex 7.8e-14 | 7.8e-14 / ex 7.8e-14 | 8.96e-12 / ex 9e-12 | 8.96e-12 / ex 9e-12 | n/a | 7.8e-14 / ex 7.8e-14 | 7.8e-14 / ex 7.8e-14 |
| analytic:pwl_diode_drop | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 1.85e-16 / ex 1.9e-16 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 1.85e-16 / ex 1.9e-16 | 1.85e-16 / ex 1.9e-16 | 1.85e-16 / ex 1.9e-16 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | n/a | 1.85e-16 / ex 1.9e-16 | 0 / ex 0 |
| analytic:rc_charge | 0.00159 / ex 0.0016 | 1.84e-06 / ex 1.8e-06 | 1.84e-06 / ex 1.8e-06 | 2.26e-07 / ex 2.3e-07 | 1.55e-14 / ex 1.5e-14 | 1.84e-06 / ex 1.8e-06 | 1.84e-06 / ex 1.8e-06 | 0.00159 / ex 0.0016 | 0.00159 / ex 0.0016 | 0.00159 / ex 0.0016 | 0.00159 / ex 0.0016 | 1.84e-06 / ex 1.8e-06 | 2.26e-07 / ex 2.3e-07 | 2.26e-07 / ex 2.3e-07 | 2.26e-07 / ex 2.3e-07 | 2.26e-07 / ex 2.3e-07 | 2.26e-07 / ex 2.3e-07 | 1.55e-14 / ex 1.5e-14 | 1.55e-14 / ex 1.5e-14 | 1.44e-14 / ex 1.4e-14 | 1.55e-14 / ex 1.5e-14 | 1.44e-14 / ex 1.4e-14 | 1.56e-08 / ex 1.6e-08 | 1.56e-08 / ex 1.6e-08 | n/a | 1.55e-14 / ex 1.5e-14 | 1.44e-14 / ex 1.4e-14 |
| analytic:rc_discharge | 0.00159 / ex 0.0016 | 1.84e-06 / ex 1.8e-06 | 1.84e-06 / ex 1.8e-06 | 6.6e-07 / ex 6.6e-07 | 1.18e-14 / ex 1.2e-14 | 1.84e-06 / ex 1.8e-06 | 1.84e-06 / ex 1.8e-06 | 0.00159 / ex 0.0016 | 0.00159 / ex 0.0016 | 0.00159 / ex 0.0016 | 0.00159 / ex 0.0016 | 1.84e-06 / ex 1.8e-06 | 6.6e-07 / ex 6.6e-07 | 6.6e-07 / ex 6.6e-07 | 6.6e-07 / ex 6.6e-07 | 6.6e-07 / ex 6.6e-07 | 6.6e-07 / ex 6.6e-07 | 1.18e-14 / ex 1.2e-14 | 1.18e-14 / ex 1.2e-14 | 1.05e-14 / ex 1.1e-14 | 1.18e-14 / ex 1.2e-14 | 1.05e-14 / ex 1.1e-14 | 2.51e-06 / ex 2.5e-06 | 2.51e-06 / ex 2.5e-06 | n/a | 1.18e-14 / ex 1.2e-14 | 1.05e-14 / ex 1.1e-14 |
| analytic:rc_esr_charge | 0.000144 / ex 0.00014 | 1.35e-08 / ex 1.3e-08 | 1.35e-08 / ex 1.3e-08 | 1.63e-07 / ex 1.6e-07 | 4.19e-14 / ex 4.2e-14 | 1.35e-08 / ex 1.3e-08 | 1.35e-08 / ex 1.3e-08 | 0.000144 / ex 0.00014 | 0.000144 / ex 0.00014 | 0.000144 / ex 0.00014 | 0.000144 / ex 0.00014 | 1.35e-08 / ex 1.3e-08 | 1.63e-07 / ex 1.6e-07 | 1.63e-07 / ex 1.6e-07 | 1.63e-07 / ex 1.6e-07 | 1.63e-07 / ex 1.6e-07 | 1.63e-07 / ex 1.6e-07 | 4.19e-14 / ex 4.2e-14 | 4.19e-14 / ex 4.2e-14 | 4.27e-14 / ex 4.3e-14 | 4.19e-14 / ex 4.2e-14 | 4.27e-14 / ex 4.3e-14 | 1.73e-08 / ex 1.7e-08 | 1.73e-08 / ex 1.7e-08 | n/a | 4.19e-14 / ex 4.2e-14 | 4.27e-14 / ex 4.3e-14 |
| analytic:rl_step | 0.00112 / ex 0.0011 | 1.3e-06 / ex 1.3e-06 | 1.3e-06 / ex 1.3e-06 | 6.25e-07 / ex 6.3e-07 | 1.9e-15 / ex 1.9e-15 | 1.3e-06 / ex 1.3e-06 | 1.3e-06 / ex 1.3e-06 | 0.00112 / ex 0.0011 | 0.00112 / ex 0.0011 | 0.00112 / ex 0.0011 | 0.00112 / ex 0.0011 | 1.3e-06 / ex 1.3e-06 | 6.25e-07 / ex 6.3e-07 | 6.25e-07 / ex 6.3e-07 | 6.25e-07 / ex 6.3e-07 | 6.25e-07 / ex 6.3e-07 | 6.25e-07 / ex 6.3e-07 | 1.9e-15 / ex 1.9e-15 | 1.9e-15 / ex 1.9e-15 | 2.06e-15 / ex 2.1e-15 | 1.9e-15 / ex 1.9e-15 | 2.06e-15 / ex 2.1e-15 | 1.07e-07 / ex 1.1e-07 | 1.07e-07 / ex 1.1e-07 | n/a | 1.9e-15 / ex 1.9e-15 | 2.06e-15 / ex 2.1e-15 |
| analytic:rlc_overdamped | 5.31e-05 / ex 5.3e-05 | 4.21e-08 / ex 4.2e-08 | 4.21e-08 / ex 4.2e-08 | 2.24e-07 / ex 2.2e-07 | 1.17e-13 / ex 1.2e-13 | 4.21e-08 / ex 4.2e-08 | 4.21e-08 / ex 4.2e-08 | 5.31e-05 / ex 5.3e-05 | 5.31e-05 / ex 5.3e-05 | 5.31e-05 / ex 5.3e-05 | 5.31e-05 / ex 5.3e-05 | 4.21e-08 / ex 4.2e-08 | 2.24e-07 / ex 2.2e-07 | 2.24e-07 / ex 2.2e-07 | 2.24e-07 / ex 2.2e-07 | 2.24e-07 / ex 2.2e-07 | 2.24e-07 / ex 2.2e-07 | 1.17e-13 / ex 1.2e-13 | 1.17e-13 / ex 1.2e-13 | 1.15e-13 / ex 1.1e-13 | 1.17e-13 / ex 1.2e-13 | 1.15e-13 / ex 1.1e-13 | 6.9e-09 / ex 6.9e-09 | 6.9e-09 / ex 6.9e-09 | n/a | 1.17e-13 / ex 1.2e-13 | 1.15e-13 / ex 1.1e-13 |
| analytic:rlc_underdamped | 0.000613 / ex 0.00061 | 1.42e-05 / ex 1.4e-05 | 1.42e-05 / ex 1.4e-05 | 5.78e-05 / ex 5.8e-05 | 5.1e-15 / ex 5.1e-15 | 1.42e-05 / ex 1.4e-05 | 1.42e-05 / ex 1.4e-05 | 0.000613 / ex 0.00061 | 0.000613 / ex 0.00061 | 0.000613 / ex 0.00061 | 0.000613 / ex 0.00061 | 1.42e-05 / ex 1.4e-05 | 5.78e-05 / ex 5.8e-05 | 5.78e-05 / ex 5.8e-05 | 5.78e-05 / ex 5.8e-05 | 5.78e-05 / ex 5.8e-05 | 5.78e-05 / ex 5.8e-05 | 5.1e-15 / ex 5.1e-15 | 5.1e-15 / ex 5.1e-15 | 5.75e-15 / ex 5.8e-15 | 5.1e-15 / ex 5.1e-15 | 5.75e-15 / ex 5.8e-15 | 2.04e-07 / ex 2e-07 | 2.04e-07 / ex 2e-07 | n/a | 5.1e-15 / ex 5.1e-15 | 5.75e-15 / ex 5.8e-15 |
| analytic:scr_halfwave_r | 1.46e-16 / ex 1.5e-16 | 0.0271 / ex 0.027 | 1.47e-16 / ex 1.5e-16 | 1.3e-11 / ex 1.3e-11 | refused | 0.0271 / ex 0.027 | 1.47e-16 / ex 1.5e-16 | 1.46e-16 / ex 1.5e-16 | 1.46e-16 / ex 1.5e-16 | 1.46e-16 / ex 1.5e-16 | 1.46e-16 / ex 1.5e-16 | 1.47e-16 / ex 1.5e-16 | 1.3e-11 / ex 1.3e-11 | 1.3e-11 / ex 1.3e-11 | 1.3e-11 / ex 1.3e-11 | 1.3e-11 / ex 1.3e-11 | 1.3e-11 / ex 1.3e-11 | refused | refused | refused | refused | refused | 1.3e-11 / ex 1.3e-11 | 1.3e-11 / ex 1.3e-11 | n/a | 1.3e-11 / ex 1.3e-11 | 1.3e-11 / ex 1.3e-11 |
| analytic:shockley_op | 6.67e-13 / ex 6.7e-13 | 6.67e-13 / ex 6.7e-13 | 6.67e-13 / ex 6.7e-13 | 6.67e-13 / ex 6.7e-13 | refused | 6.67e-13 / ex 6.7e-13 | 6.67e-13 / ex 6.7e-13 | 6.67e-13 / ex 6.7e-13 | 6.67e-13 / ex 6.7e-13 | 6.67e-13 / ex 6.7e-13 | 6.67e-13 / ex 6.7e-13 | 6.67e-13 / ex 6.7e-13 | 6.67e-13 / ex 6.7e-13 | 6.67e-13 / ex 6.7e-13 | 6.67e-13 / ex 6.7e-13 | 6.67e-13 / ex 6.7e-13 | 6.67e-13 / ex 6.7e-13 | refused | refused | refused | refused | refused | 6.67e-13 / ex 6.7e-13 | 6.67e-13 / ex 6.7e-13 | n/a | 6.67e-13 / ex 6.7e-13 | 6.67e-13 / ex 6.7e-13 |
| analytic:sine_rc | 9.65e-07 / ex 9.7e-07 | 9.65e-07 / ex 9.7e-07 | 9.65e-07 / ex 9.7e-07 | 5.85e-06 / ex 5.8e-06 | 5.55e-15 / ex 5.6e-15 | 9.65e-07 / ex 9.7e-07 | 9.65e-07 / ex 9.7e-07 | 9.65e-07 / ex 9.7e-07 | 9.65e-07 / ex 9.7e-07 | 9.65e-07 / ex 9.7e-07 | 9.65e-07 / ex 9.7e-07 | 9.65e-07 / ex 9.7e-07 | 5.85e-06 / ex 5.8e-06 | 5.85e-06 / ex 5.8e-06 | 5.85e-06 / ex 5.8e-06 | 5.85e-06 / ex 5.8e-06 | 5.85e-06 / ex 5.8e-06 | 5.55e-15 / ex 5.6e-15 | 5.55e-15 / ex 5.6e-15 | 6.15e-15 / ex 6.1e-15 | 5.55e-15 / ex 5.6e-15 | 6.15e-15 / ex 6.1e-15 | 1.17e-10 / ex 1.2e-10 | 1.17e-10 / ex 1.2e-10 | n/a | 5.55e-15 / ex 5.6e-15 | 6.15e-15 / ex 6.1e-15 |
| analytic:sine_rl | 9.65e-07 / ex 9.7e-07 | 9.65e-07 / ex 9.7e-07 | 9.65e-07 / ex 9.7e-07 | 6.3e-06 / ex 6.3e-06 | 7.43e-15 / ex 7.4e-15 | 9.65e-07 / ex 9.7e-07 | 9.65e-07 / ex 9.7e-07 | 9.65e-07 / ex 9.7e-07 | 9.65e-07 / ex 9.7e-07 | 9.65e-07 / ex 9.7e-07 | 9.65e-07 / ex 9.7e-07 | 9.65e-07 / ex 9.7e-07 | 6.3e-06 / ex 6.3e-06 | 6.3e-06 / ex 6.3e-06 | 6.3e-06 / ex 6.3e-06 | 6.3e-06 / ex 6.3e-06 | 6.3e-06 / ex 6.3e-06 | 7.43e-15 / ex 7.4e-15 | 7.43e-15 / ex 7.4e-15 | 6.06e-15 / ex 6.1e-15 | 7.43e-15 / ex 7.4e-15 | 6.06e-15 / ex 6.1e-15 | 1.17e-10 / ex 1.2e-10 | 1.17e-10 / ex 1.2e-10 | n/a | 7.43e-15 / ex 7.4e-15 | 6.06e-15 / ex 6.1e-15 |
| analytic:three_limb_zero_seq | 0.00159 / ex 0.0016 | 1.84e-06 / ex 1.8e-06 | 1.84e-06 / ex 1.8e-06 | 2.27e-07 / ex 2.3e-07 | 2.59e-14 / ex 2.6e-14 | 1.84e-06 / ex 1.8e-06 | 1.84e-06 / ex 1.8e-06 | 0.00159 / ex 0.0016 | 0.00159 / ex 0.0016 | 0.00159 / ex 0.0016 | 0.00159 / ex 0.0016 | 1.84e-06 / ex 1.8e-06 | 2.27e-07 / ex 2.3e-07 | 2.27e-07 / ex 2.3e-07 | 2.27e-07 / ex 2.3e-07 | 2.27e-07 / ex 2.3e-07 | 2.27e-07 / ex 2.3e-07 | 2.59e-14 / ex 2.6e-14 | 2.59e-14 / ex 2.6e-14 | 2.59e-14 / ex 2.6e-14 | 2.59e-14 / ex 2.6e-14 | 2.59e-14 / ex 2.6e-14 | 3.12e-09 / ex 3.1e-09 | 3.12e-09 / ex 3.1e-09 | n/a | 2.59e-14 / ex 2.6e-14 | 2.59e-14 / ex 2.6e-14 |
| analytic:vcvs_gain2 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | n/a | 0 / ex 0 | 0 / ex 0 |
| analytic:wheatstone | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 8.88e-17 / ex 8.9e-17 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 8.88e-17 / ex 8.9e-17 | 8.88e-17 / ex 8.9e-17 | 8.88e-17 / ex 8.9e-17 | 8.88e-17 / ex 8.9e-17 | 8.88e-17 / ex 8.9e-17 | 0 / ex 0 | 0 / ex 0 | n/a | 8.88e-17 / ex 8.9e-17 | 8.88e-17 / ex 8.9e-17 |
| closed_loop/cl_boost_pi.yaml | 0.0582 | 0.0579 | 8.55e-05 | 1.31e-06 | 1.29e-06 | 0.0579 | 8.55e-05 | 0.0582 | 0.0581 | 0.0582 | 0.0582 | 1.06e-05 | 1.31e-06 | 1.31e-06 | 1.31e-06 | 1.31e-06 | 1.31e-06 | 1.29e-06 | 1.29e-06 | 1.29e-06 | 1.29e-06 | 1.29e-06 | 1.29e-06 | 1.29e-06 | 1.31e-06 | 0.000146 | 4.27e-06 |
| closed_loop/cl_buck_aggressive_pi.yaml | 0.00662 | 0.00709 | 0.00404 | 3.72e-05 | 3.24e-05 | 0.00709 | 0.00404 | 0.00662 | 0.00695 | 0.00662 | 0.00662 | 3.69e-05 | 3.72e-05 | 3.72e-05 | 3.72e-05 | 3.72e-05 | 3.72e-05 | 3.24e-05 | 3.24e-05 | 3.24e-05 | 3.24e-05 | 3.24e-05 | 8.36e-05 | 8.36e-05 | 3.72e-05 | 0.00395 | 6.73e-06 |
| closed_loop/cl_buck_pi.yaml | 0.0121 | 0.0123 | 0.00049 | 1.88e-05 | 1.77e-05 | 0.0123 | 0.00049 | 0.0121 | 0.0122 | 0.0121 | 0.0121 | 1.77e-05 | 1.88e-05 | 1.88e-05 | 1.88e-05 | 1.88e-05 | 1.88e-05 | 1.77e-05 | 1.77e-05 | 1.77e-05 | 1.77e-05 | 1.77e-05 | 4.95e-05 | 4.95e-05 | 1.88e-05 | 0.000853 | 2.98e-06 |
| closed_loop/ll14_buck_closed_loop.yaml | 0.249 | 0.221 | 0.188 | 0.000547 | refused | 0.221 | 0.188 | 0.249 | 0.249 | 0.249 | 0.249 | 0.188 | 0.000547 | 0.000547 | 0.000547 | 0.000548 | 0.000548 | refused | refused | refused | refused | refused | 0.000375 | 0.000376 | 0.000547 | 0.000547 | 0.000548 |
| closed_loop/ll15_boost_closed_loop.yaml | 0.22 | 0.0659 | 0.05 | 0.000264 | refused | 0.0659 | 0.05 | 0.22 | 0.22 | 0.22 | 0.22 | 0.05 | 0.000264 | 0.000264 | 0.000264 | 0.000264 | 0.000264 | refused | refused | refused | refused | refused | 0.000111 | 0.000109 | 0.000264 | 0.000264 | 0.000264 |
| closed_loop/ll16_flyback_closed_loop.yaml | 0.112 | 0.208 | 0.115 | 0.000363 | refused | 0.208 | 0.115 | 0.112 | 0.112 | 0.112 | 0.112 | 0.115 | 0.000363 | 0.000363 | 0.000363 | 0.000362 | 0.000362 | refused | refused | refused | refused | refused | 0.000193 | 0.000193 | 0.000363 | 0.000363 | 0.000362 |
| diodes/back_to_back_diodes_sine.yaml | 1.14e-05 / ex 1.9e-16 | 1.14e-05 / ex 1.9e-16 | 1.14e-05 / ex 1.9e-16 | 0 / ex 2e-16 | 1.15e-05 / ex 2e-16 | 1.14e-05 / ex 1.9e-16 | 1.14e-05 / ex 1.9e-16 | 1.14e-05 / ex 5e-13 | 1.14e-05 / ex 1.9e-16 | 1.14e-05 / ex 1.9e-16 | 1.14e-05 / ex 1.9e-16 | 1.14e-05 / ex 2e-16 | 0 / ex 2e-16 | 0 / ex 2e-16 | 0 / ex 2e-16 | 5.09e-17 / ex 2e-16 | 5.09e-17 / ex 2e-16 | 1.15e-05 / ex 2e-16 | 1.15e-05 / ex 2e-16 | 1.15e-05 / ex 2e-16 | 1.15e-05 / ex 2e-16 | 1.15e-05 / ex 2e-16 | 0 / ex 2e-16 | 5.09e-17 / ex 2e-16 | 0 / ex 2e-16 | 1.15e-05 / ex 2e-16 | 1.15e-05 / ex 2e-16 |
| diodes/diode_bridge_rectifier.yaml | 0.000216 / ex 0.0001 | 0.000177 / ex 2.2e-05 | 0.000173 / ex 2.4e-05 | 3.22e-05 / ex 3.5e-07 | 6.58e-06 / ex 1.4e-14 | 0.000177 / ex 2.2e-05 | 0.000173 / ex 2.4e-05 | 0.000216 / ex 0.0001 | 0.000216 / ex 0.0001 | 0.000216 / ex 0.0001 | 0.000216 / ex 0.0001 | 0.000173 / ex 2.4e-05 | 3.22e-05 / ex 3.5e-07 | 3.22e-05 / ex 3.5e-07 | 3.22e-05 / ex 3.5e-07 | 3.22e-05 / ex 3.5e-07 | 3.22e-05 / ex 3.5e-07 | 6.58e-06 / ex 1.4e-14 | 6.58e-06 / ex 1.4e-14 | 6.58e-06 / ex 1.4e-14 | 6.58e-06 / ex 1.4e-14 | 6.58e-06 / ex 1.4e-14 | 3.29e-05 / ex 3.6e-07 | 3.29e-05 / ex 3.6e-07 | 3.22e-05 / ex 3.5e-07 | 6.59e-06 / ex 2.9e-14 | 6.59e-06 / ex 3e-14 |
| diodes/diode_clamp_overvoltage.yaml | 0.00565 | 0.00565 | 0.00565 | 0 | 6.01e-12 | 0.00565 | 0.00565 | 0.00565 | 0.00565 | 0.00565 | 0.00565 | 0.00565 | 0 | 0 | 0 | 2.71e-16 | 2.71e-16 | 6.01e-12 | 6.01e-12 | 6.01e-12 | 6.01e-12 | 6.01e-12 | 4.45e-17 | 2.71e-16 | 0 | 6.14e-12 | 6.14e-12 |
| diodes/diode_half_wave_rl_load.yaml | 0.0186 / ex 0.02 | 0.000598 / ex 0.00095 | 0.00061 / ex 0.00096 | 1.58e-05 / ex 5.6e-06 | 6.92e-06 / ex 4.9e-06 | 0.000598 / ex 0.00095 | 0.00061 / ex 0.00096 | 0.0186 / ex 0.02 | 0.0186 / ex 0.02 | 0.0186 / ex 0.02 | 0.0186 / ex 0.02 | 0.000586 / ex 5e-05 | 1.58e-05 / ex 5.6e-06 | 1.58e-05 / ex 5.6e-06 | 1.58e-05 / ex 5.6e-06 | 1.58e-05 / ex 5.6e-06 | 1.58e-05 / ex 5.6e-06 | 6.92e-06 / ex 4.9e-06 | 6.92e-06 / ex 4.9e-06 | 6.92e-06 / ex 4.9e-06 | 6.92e-06 / ex 4.9e-06 | 6.92e-06 / ex 4.9e-06 | 0.00101 / ex 1.3e-07 | 0.00101 / ex 1.3e-07 | 1.58e-05 / ex 5.6e-06 | 6.92e-06 / ex 4.9e-06 | 6.92e-06 / ex 4.9e-06 |
| diodes/diode_rectifier.yaml | 0.0139 / ex 0.0097 | 0.00157 / ex 0.00067 | 0.00134 / ex 0.00018 | 0.000183 / ex 2.4e-07 | 1.26e-05 / ex 1.6e-07 | 0.00157 / ex 0.00067 | 0.00134 / ex 0.00018 | 0.0139 / ex 0.0097 | 0.0139 / ex 0.0097 | 0.0139 / ex 0.0097 | 0.0139 / ex 0.0097 | 0.00134 / ex 0.00018 | 0.000183 / ex 2.4e-07 | 0.000183 / ex 2.4e-07 | 0.000183 / ex 2.4e-07 | 0.000183 / ex 2.4e-07 | 0.000183 / ex 2.4e-07 | 1.26e-05 / ex 1.6e-07 | 1.26e-05 / ex 1.6e-07 | 1.26e-05 / ex 1.6e-07 | 1.26e-05 / ex 1.6e-07 | 1.26e-05 / ex 1.6e-07 | 0.000316 / ex 3.2e-07 | 0.000316 / ex 3.2e-07 | 0.000183 / ex 2.4e-07 | 0.00056 / ex 1.6e-07 | 2.78e-06 / ex 3.1e-14 |
| diodes/half_wave_rectifier_sine.yaml | 0.000393 / ex 0.00038 | 0.000112 / ex 0.0001 | 4.09e-05 / ex 2.6e-06 | 2.36e-05 / ex 7.3e-08 | 4.6e-06 / ex 5.4e-14 | 0.000112 / ex 0.0001 | 4.09e-05 / ex 2.6e-06 | 0.000393 / ex 0.00038 | 0.000393 / ex 0.00038 | 0.000393 / ex 0.00038 | 0.000393 / ex 0.00038 | 4.09e-05 / ex 2.6e-06 | 2.36e-05 / ex 7.3e-08 | 2.36e-05 / ex 7.3e-08 | 2.36e-05 / ex 7.3e-08 | 2.36e-05 / ex 7.3e-08 | 2.36e-05 / ex 7.3e-08 | 4.6e-06 / ex 5.4e-14 | 4.6e-06 / ex 5.4e-14 | 4.6e-06 / ex 5.4e-14 | 4.6e-06 / ex 5.5e-14 | 4.6e-06 / ex 5.5e-14 | 2.34e-05 / ex 6.1e-08 | 2.34e-05 / ex 6.1e-08 | 2.36e-05 / ex 7.3e-08 | 4.63e-06 / ex 3.7e-13 | 4.63e-06 / ex 3.6e-13 |
| diodes/voltage_doubler.yaml | 0.000222 / ex 0.00022 | 3e-05 / ex 1.7e-06 | 3.01e-05 / ex 2.2e-07 | 5.05e-06 / ex 3.5e-08 | 5.84e-07 / ex 3.6e-10 | 3e-05 / ex 1.7e-06 | 3.01e-05 / ex 2.2e-07 | 0.000222 / ex 0.00022 | 0.000222 / ex 0.00022 | 0.000222 / ex 0.00022 | 0.000222 / ex 0.00022 | 3.01e-05 / ex 2e-07 | 5.05e-06 / ex 3.5e-08 | 5.05e-06 / ex 3.5e-08 | 5.05e-06 / ex 3.5e-08 | 5.05e-06 / ex 3.5e-08 | 5.05e-06 / ex 3.5e-08 | 5.84e-07 / ex 3.6e-10 | 5.84e-07 / ex 3.6e-10 | 5.84e-07 / ex 3.6e-10 | 5.84e-07 / ex 3.6e-10 | 5.84e-07 / ex 3.6e-10 | 5.07e-06 / ex 4.8e-09 | 5.07e-06 / ex 4.8e-09 | 5.05e-06 / ex 3.5e-08 | 5.91e-07 / ex 2.5e-13 | 5.91e-07 / ex 2.5e-13 |
| linear/bridge_resistive_pure.yaml | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 1.33e-13 / ex 1.3e-13 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 1.33e-13 / ex 1.3e-13 | 1.33e-13 / ex 1.3e-13 | 1.33e-13 / ex 1.3e-13 | 1.33e-13 / ex 1.3e-13 | 1.33e-13 / ex 1.3e-13 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 1.33e-13 / ex 1.3e-13 | 1.33e-13 / ex 1.3e-13 |
| linear/capacitor_natural_discharge.yaml | 0.000798 / ex 0.00079 | 7.52e-07 / ex 4.6e-07 | 7.52e-07 / ex 4.6e-07 | 5.63e-06 / ex 5.5e-06 | 6.11e-07 / ex 1.1e-14 | 7.52e-07 / ex 4.6e-07 | 7.52e-07 / ex 4.6e-07 | 0.000798 / ex 0.00079 | 0.000798 / ex 0.00079 | 0.000798 / ex 0.00079 | 0.000798 / ex 0.00079 | 7.52e-07 / ex 4.6e-07 | 5.63e-06 / ex 5.5e-06 | 5.63e-06 / ex 5.5e-06 | 5.63e-06 / ex 5.5e-06 | 5.63e-06 / ex 5.5e-06 | 5.63e-06 / ex 5.5e-06 | 6.11e-07 / ex 1.1e-14 | 6.11e-07 / ex 1.1e-14 | 6.11e-07 / ex 1.1e-14 | 6.11e-07 / ex 1.1e-14 | 6.11e-07 / ex 1.1e-14 | 2.84e-06 / ex 2.5e-06 | 2.84e-06 / ex 2.5e-06 | 5.63e-06 / ex 5.5e-06 | 6.12e-07 / ex 1.2e-14 | 6.1e-07 / ex 1.1e-14 |
| linear/current_divider_3way.yaml | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 |
| linear/current_source_rc.yaml | 5.66e-10 / ex 1.1e-06 | 4.07e-10 / ex 1.1e-06 | 4.07e-10 / ex 1.1e-06 | 0 / ex 1.1e-06 | 3.58e-10 / ex 1.1e-06 | 4.07e-10 / ex 1.1e-06 | 4.07e-10 / ex 1.1e-06 | 5.66e-10 / ex 1.1e-06 | 5.66e-10 / ex 1.1e-06 | 5.66e-10 / ex 1.1e-06 | 5.66e-10 / ex 1.1e-06 | 4.07e-10 / ex 1.1e-06 | 0 / ex 1.1e-06 | 0 / ex 1.1e-06 | 0 / ex 1.1e-06 | 0 / ex 1.1e-06 | 0 / ex 1.1e-06 | 3.58e-10 / ex 1.1e-06 | 3.58e-10 / ex 1.1e-06 | 3.58e-10 / ex 1.1e-06 | 3.58e-10 / ex 1.1e-06 | 3.58e-10 / ex 1.1e-06 | 3e-11 / ex 1.1e-06 | 3e-11 / ex 1.1e-06 | 0 / ex 1.1e-06 | 1.63e-10 / ex 1.1e-06 | 1.63e-10 / ex 1.1e-06 |
| linear/current_source_rl_step.yaml | 0.00577 / ex 0.0054 | 3.99e-05 / ex 3.1e-05 | 3.99e-05 / ex 3.1e-05 | 2.14e-06 / ex 4.6e-07 | 6.04e-07 / ex 1.5e-14 | 3.99e-05 / ex 3.1e-05 | 3.99e-05 / ex 3.1e-05 | 0.00577 / ex 0.0054 | 0.00577 / ex 0.0054 | 0.00577 / ex 0.0054 | 0.00577 / ex 0.0054 | 3.99e-05 / ex 3.1e-05 | 2.14e-06 / ex 4.6e-07 | 2.14e-06 / ex 4.6e-07 | 2.14e-06 / ex 4.6e-07 | 2.14e-06 / ex 4.6e-07 | 2.14e-06 / ex 4.6e-07 | 6.04e-07 / ex 1.5e-14 | 6.04e-07 / ex 1.5e-14 | 6.04e-07 / ex 1.5e-14 | 6.04e-07 / ex 1.5e-14 | 6.04e-07 / ex 1.5e-14 | 5.2e-06 / ex 3.5e-06 | 5.2e-06 / ex 3.5e-06 | 2.14e-06 / ex 4.6e-07 | 2e-06 / ex 2.2e-15 | 1.09e-06 / ex 2e-15 |
| linear/lc_low_pass_sine.yaml | 3.26e-05 / ex 2.7e-05 | 3.26e-05 / ex 2.7e-05 | 3.26e-05 / ex 2.7e-05 | 0.000118 / ex 0.00011 | 1.14e-06 / ex 1.7e-13 | 3.26e-05 / ex 2.7e-05 | 3.26e-05 / ex 2.7e-05 | 3.26e-05 / ex 2.7e-05 | 3.26e-05 / ex 2.7e-05 | 3.26e-05 / ex 2.7e-05 | 3.26e-05 / ex 2.7e-05 | 3.26e-05 / ex 2.7e-05 | 0.000118 / ex 0.00011 | 0.000118 / ex 0.00011 | 0.000118 / ex 0.00011 | 0.000118 / ex 0.00011 | 0.000118 / ex 0.00011 | 1.14e-06 / ex 1.7e-13 | 1.14e-06 / ex 1.7e-13 | 1.14e-06 / ex 1.7e-13 | 1.14e-06 / ex 1.7e-13 | 1.14e-06 / ex 1.7e-13 | 0.00014 / ex 1.3e-08 | 0.00014 / ex 1.3e-08 | 0.000118 / ex 0.00011 | 8.59e-06 / ex 8.2e-15 | 1.23e-06 / ex 8.5e-14 |
| linear/ll01_resistive_divider_dc.yaml | 2.85e-10 / ex 4e-07 | 3.41e-10 / ex 4e-07 | 3.41e-10 / ex 4e-07 | 0 / ex 4e-07 | 1.08e-09 / ex 4e-07 | 3.41e-10 / ex 4e-07 | 3.41e-10 / ex 4e-07 | 2.85e-10 / ex 4e-07 | 2.85e-10 / ex 4e-07 | 2.85e-10 / ex 4e-07 | 2.85e-10 / ex 4e-07 | 3.41e-10 / ex 4e-07 | 0 / ex 4e-07 | 0 / ex 4e-07 | 0 / ex 4e-07 | 0 / ex 4e-07 | 0 / ex 4e-07 | 1.08e-09 / ex 4e-07 | 1.08e-09 / ex 4e-07 | 1.08e-09 / ex 4e-07 | 1.08e-09 / ex 4e-07 | 1.08e-09 / ex 4e-07 | 1.01e-10 / ex 4e-07 | 1.01e-10 / ex 4e-07 | 0 / ex 4e-07 | 4.52e-10 / ex 4e-07 | 4.52e-10 / ex 4e-07 |
| linear/ll02_rc_pulse_charge.yaml | 6.92e-08 | 6.97e-08 | 6.97e-08 | 1.48e-07 | 4.54e-07 | 6.97e-08 | 6.97e-08 | 6.92e-08 | 6.92e-08 | 6.92e-08 | 6.92e-08 | 6.97e-08 | 1.48e-07 | 1.48e-07 | 1.48e-07 | 1.48e-07 | 1.48e-07 | 4.54e-07 | 4.54e-07 | 4.54e-07 | 4.54e-07 | 4.54e-07 | 1.72e-07 | 1.72e-07 | 1.48e-07 | 4.47e-07 | 4.3e-07 |
| linear/ll03_rl_step.yaml | 0.00601 | 0.00601 | 0.00601 | 6.38e-06 | 5.61e-07 | 0.00601 | 0.00601 | 0.00601 | 0.00601 | 0.00601 | 0.00601 | 0.00601 | 6.38e-06 | 6.38e-06 | 6.38e-06 | 6.38e-06 | 6.38e-06 | 5.61e-07 | 5.61e-07 | 5.61e-07 | 5.61e-07 | 5.61e-07 | 7.62e-06 | 7.62e-06 | 6.38e-06 | 5.53e-07 | 5.57e-07 |
| linear/pulse_voltage_rl.yaml | 0.0199 | 0.0199 | 0.0199 | 3.04e-05 | 5.12e-06 | 0.0199 | 0.0199 | 0.0199 | 0.0199 | 0.0199 | 0.0199 | 0.0199 | 3.04e-05 | 3.04e-05 | 3.04e-05 | 3.04e-05 | 3.04e-05 | 5.12e-06 | 5.12e-06 | 5.12e-06 | 5.12e-06 | 5.12e-06 | 4.14e-05 | 4.14e-05 | 3.04e-05 | 2.05e-05 | 1.29e-06 |
| linear/rc_dc.yaml | 0.000159 / ex 0.00016 | 4.39e-07 / ex 1.9e-08 | 4.39e-07 / ex 1.9e-08 | 6.84e-07 / ex 2.3e-07 | 4.29e-07 / ex 2.4e-13 | 4.39e-07 / ex 1.9e-08 | 4.39e-07 / ex 1.9e-08 | 0.000159 / ex 0.00016 | 0.000159 / ex 0.00016 | 0.000159 / ex 0.00016 | 0.000159 / ex 0.00016 | 4.39e-07 / ex 1.9e-08 | 6.84e-07 / ex 2.3e-07 | 6.84e-07 / ex 2.3e-07 | 6.84e-07 / ex 2.3e-07 | 6.84e-07 / ex 2.3e-07 | 6.84e-07 / ex 2.3e-07 | 4.29e-07 / ex 2.4e-13 | 4.29e-07 / ex 2.4e-13 | 4.29e-07 / ex 2.4e-13 | 4.29e-07 / ex 2.4e-13 | 4.29e-07 / ex 2.4e-13 | 5.88e-07 / ex 3e-08 | 5.88e-07 / ex 3e-08 | 6.84e-07 / ex 2.3e-07 | 4.29e-07 / ex 1.5e-14 | 4.28e-07 / ex 1.4e-14 |
| linear/rc_step.yaml | 0.000159 | 5.65e-07 | 5.65e-07 | 6.84e-07 | 4.29e-07 | 5.65e-07 | 5.65e-07 | 0.000159 | 0.000159 | 0.000159 | 0.000159 | 5.65e-07 | 6.84e-07 | 6.84e-07 | 6.84e-07 | 6.84e-07 | 6.84e-07 | 4.29e-07 | 4.29e-07 | 4.29e-07 | 4.29e-07 | 4.29e-07 | 7.19e-07 | 7.19e-07 | 6.84e-07 | 4.28e-07 | 4.27e-07 |
| linear/rl_natural_response.yaml | 0.00577 / ex 0.0054 | 3.98e-05 / ex 3.1e-05 | 3.98e-05 / ex 3.1e-05 | 6.66e-06 / ex 5.1e-06 | 6.04e-07 / ex 1.5e-15 | 3.98e-05 / ex 3.1e-05 | 3.98e-05 / ex 3.1e-05 | 0.00577 / ex 0.0054 | 0.00577 / ex 0.0054 | 0.00577 / ex 0.0054 | 0.00577 / ex 0.0054 | 3.98e-05 / ex 3.1e-05 | 6.66e-06 / ex 5.1e-06 | 6.66e-06 / ex 5.1e-06 | 6.66e-06 / ex 5.1e-06 | 6.66e-06 / ex 5.1e-06 | 6.66e-06 / ex 5.1e-06 | 6.04e-07 / ex 1.5e-15 | 6.04e-07 / ex 1.5e-15 | 6.04e-07 / ex 1.5e-15 | 6.04e-07 / ex 1.5e-15 | 6.04e-07 / ex 1.5e-15 | 5.11e-06 / ex 3.5e-06 | 5.11e-06 / ex 3.5e-06 | 6.66e-06 / ex 5.1e-06 | 1.96e-06 / ex 8.3e-16 | 1.23e-06 / ex 1.5e-15 |
| linear/rl_step.yaml | 0.00112 | 1.8e-06 | 1.8e-06 | 1.75e-06 | 4.98e-07 | 1.8e-06 | 1.8e-06 | 0.00112 | 0.00112 | 0.00112 | 0.00112 | 1.8e-06 | 1.75e-06 | 1.75e-06 | 1.75e-06 | 1.75e-06 | 1.75e-06 | 4.98e-07 | 4.98e-07 | 4.98e-07 | 4.98e-07 | 4.98e-07 | 1.97e-06 | 1.97e-06 | 1.75e-06 | 2.01e-06 | 1.17e-06 |
| linear/rlc_series_resonant.yaml | 0.00033 / ex 0.00033 | 0.00033 / ex 0.00033 | 0.00033 / ex 0.00033 | 0.000354 / ex 0.00034 | 3.41e-06 / ex 5.4e-14 | 0.00033 / ex 0.00033 | 0.00033 / ex 0.00033 | 0.00033 / ex 0.00033 | 0.00033 / ex 0.00033 | 0.00033 / ex 0.00033 | 0.00033 / ex 0.00033 | 0.00033 / ex 0.00033 | 0.000354 / ex 0.00034 | 0.000354 / ex 0.00034 | 0.000354 / ex 0.00034 | 0.000354 / ex 0.00034 | 0.000354 / ex 0.00034 | 3.41e-06 / ex 5.4e-14 | 3.41e-06 / ex 5.4e-14 | 3.41e-06 / ex 5.4e-14 | 3.41e-06 / ex 5.4e-14 | 3.41e-06 / ex 5.4e-14 | 0.000757 / ex 2.1e-06 | 0.000757 / ex 2.1e-06 | 0.000354 / ex 0.00034 | 3.02e-05 / ex 3.8e-15 | 4.04e-06 / ex 5.2e-13 |
| linear/sine_voltage_rc.yaml | 4.42e-05 / ex 2.3e-05 | 4.42e-05 / ex 2.3e-05 | 4.42e-05 / ex 2.3e-05 | 0.000111 / ex 2.4e-05 | 3.06e-06 / ex 6.3e-15 | 4.42e-05 / ex 2.3e-05 | 4.42e-05 / ex 2.3e-05 | 4.42e-05 / ex 2.3e-05 | 4.42e-05 / ex 2.3e-05 | 4.42e-05 / ex 2.3e-05 | 4.42e-05 / ex 2.3e-05 | 4.42e-05 / ex 2.3e-05 | 0.000111 / ex 2.4e-05 | 0.000111 / ex 2.4e-05 | 0.000111 / ex 2.4e-05 | 0.000111 / ex 2.4e-05 | 0.000111 / ex 2.4e-05 | 3.06e-06 / ex 6.3e-15 | 3.06e-06 / ex 6.3e-15 | 3.06e-06 / ex 6.3e-15 | 3.06e-06 / ex 6.3e-15 | 3.06e-06 / ex 6.3e-15 | 9.66e-05 / ex 1.8e-09 | 9.66e-05 / ex 1.8e-09 | 0.000111 / ex 2.4e-05 | 2.92e-06 / ex 3.4e-15 | 2.93e-06 / ex 3.1e-15 |
| linear/two_stage_rc_lowpass.yaml | 0.00145 | 1.01e-05 | 1.01e-05 | 3.44e-06 | 4.62e-07 | 1.01e-05 | 1.01e-05 | 0.00145 | 0.00145 | 0.00145 | 0.00145 | 1.01e-05 | 3.44e-06 | 3.44e-06 | 3.44e-06 | 3.44e-06 | 3.44e-06 | 4.62e-07 | 4.62e-07 | 4.62e-07 | 4.62e-07 | 4.62e-07 | 1.16e-05 | 1.16e-05 | 3.44e-06 | 6.7e-07 | 4.43e-07 |
| linear/voltage_divider_3stage.yaml | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 |
| magnetics/core_loss_steinmetz_sine.yaml | 0.000228 / ex 0.00015 | 0.000227 / ex 0.00015 | 0.000227 / ex 0.00015 | 0.000123 / ex 4.1e-05 | 3.5e-06 / ex 1.9e-14 | 0.000227 / ex 0.00015 | 0.000227 / ex 0.00015 | 0.000228 / ex 0.00015 | 0.000228 / ex 0.00015 | 0.000228 / ex 0.00015 | 0.000228 / ex 0.00015 | 0.000227 / ex 0.00015 | 0.000123 / ex 4.1e-05 | 0.000123 / ex 4.1e-05 | 0.000123 / ex 4.1e-05 | 0.000123 / ex 4.1e-05 | 0.000123 / ex 4.1e-05 | 3.5e-06 / ex 1.9e-14 | 3.5e-06 / ex 1.9e-14 | 3.5e-06 / ex 1.9e-14 | 3.5e-06 / ex 1.9e-14 | 3.5e-06 / ex 1.9e-14 | 0.000107 / ex 2.7e-09 | 0.000107 / ex 2.7e-09 | 0.000123 / ex 4.1e-05 | 0.0106 / ex 2e-15 | 3.7e-06 / ex 5.8e-14 |
| magnetics/coupled_inductor_step.yaml | 0.00032 | 0.00032 | 0.00032 | 3.76e-06 | 6.72e-06 | 0.00032 | 0.00032 | 0.00032 | 0.00032 | 0.00032 | 0.00032 | 0.00032 | 3.76e-06 | 3.76e-06 | 3.76e-06 | 3.76e-06 | 3.76e-06 | 6.72e-06 | 6.72e-06 | 6.72e-06 | 6.72e-06 | 6.72e-06 | 4.48e-05 | 4.48e-05 | 3.76e-06 | 5.11e-06 | 6.83e-07 |
| magnetics/half_bridge_transformer_dc.yaml | 0.101 / ex 0.1 | 0.000128 / ex 0.00013 | 0.000127 / ex 0.00013 | 2.08e-05 / ex 1.5e-05 | 3.41e-06 / ex 3e-06 | 0.000128 / ex 0.00013 | 0.000127 / ex 0.00013 | 0.101 / ex 0.1 | 0.101 / ex 0.1 | 0.101 / ex 0.1 | 0.101 / ex 0.1 | 0.000127 / ex 0.00013 | 2.08e-05 / ex 1.5e-05 | 2.08e-05 / ex 1.5e-05 | 2.08e-05 / ex 1.5e-05 | 2.08e-05 / ex 1.5e-05 | 2.08e-05 / ex 1.5e-05 | 3.41e-06 / ex 3e-06 | 3.41e-06 / ex 3e-06 | 3.41e-06 / ex 3e-06 | 3.41e-06 / ex 3e-06 | 3.41e-06 / ex 3e-06 | 2.81e-05 / ex 4.4e-08 | 2.81e-05 / ex 4.4e-08 | 2.08e-05 / ex 1.5e-05 | 0.00195 / ex 3e-06 | 2.66e-06 / ex 6.7e-13 |
| magnetics/inductor_chain_step.yaml | 0.00231 | 5.34e-06 | 5.34e-06 | 9.05e-07 | 4.89e-07 | 5.34e-06 | 5.34e-06 | 0.00231 | 0.00231 | 0.00231 | 0.00231 | 5.34e-06 | 9.05e-07 | 9.05e-07 | 9.05e-07 | 9.05e-07 | 9.05e-07 | 4.89e-07 | 4.89e-07 | 4.89e-07 | 4.89e-07 | 4.89e-07 | 9.85e-07 | 9.85e-07 | 9.05e-07 | 4.9e-07 | 4.82e-07 |
| magnetics/transformer_leakage.yaml | 0.0041 / ex 0.0015 | 0.00409 / ex 0.0015 | 0.00409 / ex 0.0015 | 0.000231 / ex 1.8e-05 | 3.11e-05 / ex 6e-15 | 0.00409 / ex 0.0015 | 0.00409 / ex 0.0015 | 0.0041 / ex 0.0015 | 0.0041 / ex 0.0015 | 0.0041 / ex 0.0015 | 0.0041 / ex 0.0015 | 0.00409 / ex 0.0015 | 0.000231 / ex 1.8e-05 | 0.000231 / ex 1.8e-05 | 0.000231 / ex 1.8e-05 | 0.000231 / ex 1.8e-05 | 0.000231 / ex 1.8e-05 | 3.11e-05 / ex 6e-15 | 3.11e-05 / ex 6e-15 | 3.11e-05 / ex 6e-15 | 3.11e-05 / ex 6e-15 | 3.11e-05 / ex 6e-15 | 0.000789 / ex 2.2e-07 | 0.000789 / ex 2.2e-07 | 0.000231 / ex 1.8e-05 | 0.00315 / ex 6.2e-15 | 2.88e-06 / ex 2.3e-14 |
| magnetics/transformer_step_up.yaml | 4.64e-06 / ex 4e-16 | 4.64e-06 / ex 4e-16 | 4.64e-06 / ex 4e-16 | 0 / ex 4e-16 | 5.1e-06 / ex 4e-16 | 4.64e-06 / ex 4e-16 | 4.64e-06 / ex 4e-16 | 4.64e-06 / ex 4e-16 | 4.64e-06 / ex 4e-16 | 4.64e-06 / ex 4e-16 | 4.64e-06 / ex 4e-16 | 4.64e-06 / ex 4e-16 | 0 / ex 4e-16 | 0 / ex 4e-16 | 0 / ex 4e-16 | 0 / ex 4e-16 | 0 / ex 4e-16 | 5.1e-06 / ex 4e-16 | 5.1e-06 / ex 4e-16 | 5.1e-06 / ex 4e-16 | 5.1e-06 / ex 4e-16 | 5.1e-06 / ex 4e-16 | 0 / ex 4e-16 | 0 / ex 4e-16 | 0 / ex 4e-16 | 4.64e-06 / ex 4e-16 | 4.64e-06 / ex 4e-16 |
| motors/motor_dc_brush_step_load.yaml | 0.000289 / ex 0.00029 | 4.5e-07 / ex 1.4e-07 | 4.5e-07 / ex 1.4e-07 | 6.06e-06 / ex 2.3e-06 | 3.62e-07 / ex 1.2e-13 | 4.5e-07 / ex 1.4e-07 | 4.5e-07 / ex 1.4e-07 | 0.000289 / ex 0.00029 | 0.000289 / ex 0.00029 | 0.000289 / ex 0.00029 | 0.000289 / ex 0.00029 | 4.5e-07 / ex 1.4e-07 | 6.06e-06 / ex 2.3e-06 | 6.06e-06 / ex 2.3e-06 | 6.06e-06 / ex 2.3e-06 | 6.06e-06 / ex 2.3e-06 | 6.06e-06 / ex 2.3e-06 | 3.62e-07 / ex 1.2e-13 | 3.62e-07 / ex 1.2e-13 | 3.62e-07 / ex 1.2e-13 | 3.62e-07 / ex 1.2e-13 | 3.62e-07 / ex 1.2e-13 | 7.51e-06 / ex 1.4e-08 | 7.51e-06 / ex 1.4e-08 | 6.06e-06 / ex 2.3e-06 | 3.63e-07 / ex 1e-13 | 3.62e-07 / ex 1e-13 |
| motors/motor_induction_locked_rotor.yaml | 2.25e-06 / ex 1.3e-06 | 2.25e-06 / ex 1.3e-06 | 2.25e-06 / ex 1.3e-06 | 5.1e-05 / ex 1.4e-05 | 1.63e-06 / ex 2.8e-14 | 2.25e-06 / ex 1.3e-06 | 2.25e-06 / ex 1.3e-06 | 2.25e-06 / ex 1.3e-06 | 2.25e-06 / ex 1.3e-06 | 2.25e-06 / ex 1.3e-06 | 2.25e-06 / ex 1.3e-06 | 2.25e-06 / ex 1.3e-06 | 5.1e-05 / ex 1.4e-05 | 5.1e-05 / ex 1.4e-05 | 5.1e-05 / ex 1.4e-05 | 5.1e-05 / ex 1.4e-05 | 5.1e-05 / ex 1.4e-05 | 1.63e-06 / ex 2.8e-14 | 1.63e-06 / ex 2.8e-14 | 1.63e-06 / ex 2.8e-14 | 1.63e-06 / ex 2.8e-14 | 1.63e-06 / ex 2.8e-14 | 4.21e-05 / ex 4.4e-10 | 4.21e-05 / ex 4.4e-10 | 5.1e-05 / ex 1.4e-05 | 1.63e-06 / ex 5.7e-14 | 1.63e-06 / ex 6.4e-14 |
| motors/motor_pmsm_dq_open_loop.yaml | 0.000177 / ex 0.00018 | 4.81e-07 / ex 5.2e-08 | 4.81e-07 / ex 5.2e-08 | 6.51e-06 / ex 2.5e-06 | 4.45e-07 / ex 1.2e-13 | 4.81e-07 / ex 5.2e-08 | 4.81e-07 / ex 5.2e-08 | 0.000177 / ex 0.00018 | 0.000177 / ex 0.00018 | 0.000177 / ex 0.00018 | 0.000177 / ex 0.00018 | 4.81e-07 / ex 5.2e-08 | 6.51e-06 / ex 2.5e-06 | 6.51e-06 / ex 2.5e-06 | 6.51e-06 / ex 2.5e-06 | 6.51e-06 / ex 2.5e-06 | 6.51e-06 / ex 2.5e-06 | 4.45e-07 / ex 1.2e-13 | 4.45e-07 / ex 1.2e-13 | 4.45e-07 / ex 1.2e-13 | 4.45e-07 / ex 1.2e-13 | 4.45e-07 / ex 1.2e-13 | 7.98e-06 / ex 1.2e-08 | 7.98e-06 / ex 1.2e-08 | 6.51e-06 / ex 2.5e-06 | 4.45e-07 / ex 1.2e-13 | 4.45e-07 / ex 1.3e-13 |
| resonance/bode_rlc_low_pass.yaml | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 |
| resonance/driven_lc_series_resonance.yaml | 0.000771 / ex 0.00074 | 0.000771 / ex 0.00074 | 0.000771 / ex 0.00074 | 0.000217 / ex 0.0002 | 2.13e-06 / ex 2e-14 | 0.000771 / ex 0.00074 | 0.000771 / ex 0.00074 | 0.000771 / ex 0.00074 | 0.000771 / ex 0.00074 | 0.000771 / ex 0.00074 | 0.000771 / ex 0.00074 | 0.000771 / ex 0.00074 | 0.000217 / ex 0.0002 | 0.000217 / ex 0.0002 | 0.000217 / ex 0.0002 | 0.000217 / ex 0.0002 | 0.000217 / ex 0.0002 | 2.13e-06 / ex 2e-14 | 2.13e-06 / ex 2e-14 | 2.13e-06 / ex 2e-14 | 2.13e-06 / ex 2e-14 | 2.13e-06 / ex 2e-14 | 0.00315 / ex 1.8e-05 | 0.00315 / ex 1.8e-05 | 0.000217 / ex 0.0002 | 3.17e-05 / ex 6.8e-15 | 3.06e-06 / ex 1.8e-12 |
| resonance/lc_resonant_tank.yaml | 0.00585 | 0.000312 | 0.000312 | 5.29e-05 | 4.29e-06 | 0.000312 | 0.000312 | 0.00585 | 0.00585 | 0.00585 | 0.00585 | 0.000312 | 5.29e-05 | 5.29e-05 | 5.29e-05 | 5.29e-05 | 5.29e-05 | 4.29e-06 | 4.29e-06 | 4.29e-06 | 4.29e-06 | 4.29e-06 | 1.21e-05 | 1.21e-05 | 5.29e-05 | 3.23e-05 | 6.55e-06 |
| resonance/ll04_rlc_underdamped.yaml | 0.00592 | 0.00592 | 0.00592 | 2.99e-05 | 7.34e-07 | 0.00592 | 0.00592 | 0.00592 | 0.00592 | 0.00592 | 0.00592 | 0.00592 | 2.99e-05 | 2.99e-05 | 2.99e-05 | 2.99e-05 | 2.99e-05 | 7.34e-07 | 7.34e-07 | 7.34e-07 | 7.34e-07 | 7.34e-07 | 2.26e-05 | 2.26e-05 | 2.99e-05 | 7.4e-07 | 7.39e-07 |
| resonance/rl_high_pass_sine.yaml | 0.000735 / ex 0.00022 | 0.000736 / ex 0.00022 | 0.000736 / ex 0.00022 | 7.56e-05 / ex 9.9e-06 | 7.54e-06 / ex 3.4e-16 | 0.000736 / ex 0.00022 | 0.000736 / ex 0.00022 | 0.000735 / ex 0.00022 | 0.000735 / ex 0.00022 | 0.000735 / ex 0.00022 | 0.000735 / ex 0.00022 | 0.000736 / ex 0.00022 | 7.56e-05 / ex 9.9e-06 | 7.56e-05 / ex 9.9e-06 | 7.56e-05 / ex 9.9e-06 | 7.56e-05 / ex 9.9e-06 | 7.56e-05 / ex 9.9e-06 | 7.54e-06 / ex 3.4e-16 | 7.54e-06 / ex 3.4e-16 | 7.54e-06 / ex 3.4e-16 | 7.54e-06 / ex 3.4e-16 | 7.54e-06 / ex 3.4e-16 | 0.000123 / ex 8.2e-10 | 0.000123 / ex 8.2e-10 | 7.56e-05 / ex 9.9e-06 | 3.07e-05 / ex 2.6e-16 | 2.6e-06 / ex 3.4e-15 |
| resonance/rlc_overdamped_step.yaml | 0.00179 | 1.12e-05 | 1.12e-05 | 5.77e-06 | 5.58e-07 | 1.12e-05 | 1.12e-05 | 0.00179 | 0.00179 | 0.00179 | 0.00179 | 1.12e-05 | 5.77e-06 | 5.77e-06 | 5.77e-06 | 5.77e-06 | 5.77e-06 | 5.58e-07 | 5.58e-07 | 5.58e-07 | 5.58e-07 | 5.58e-07 | 9.11e-06 | 9.11e-06 | 5.77e-06 | 2.04e-06 | 2.85e-07 |
| resonance/rlc_parallel_step.yaml | 0.00778 | 0.000404 | 0.000404 | 1.61e-05 | 1.02e-05 | 0.000404 | 0.000404 | 0.00778 | 0.00778 | 0.00778 | 0.00778 | 0.000404 | 1.61e-05 | 1.61e-05 | 1.61e-05 | 1.61e-05 | 1.61e-05 | 1.02e-05 | 1.02e-05 | 1.02e-05 | 1.02e-05 | 1.02e-05 | 0.000367 | 0.000367 | 1.61e-05 | 4.07e-05 | 8.08e-07 |
| resonance/rlc_step.yaml | 0.00043 | 9.23e-07 | 9.23e-07 | 1.31e-05 | 1.94e-07 | 9.23e-07 | 9.23e-07 | 0.00043 | 0.00043 | 0.00043 | 0.00043 | 9.23e-07 | 1.31e-05 | 1.31e-05 | 1.31e-05 | 1.31e-05 | 1.31e-05 | 1.94e-07 | 1.94e-07 | 1.94e-07 | 1.94e-07 | 1.94e-07 | 1.93e-05 | 1.93e-05 | 1.31e-05 | 7.83e-07 | 4.41e-07 |
| stress/high_q_rlc_ringdown.yaml | 0.0229 / ex 0.023 | 0.0149 / ex 0.015 | 0.0149 / ex 0.015 | 0.00566 / ex 0.0057 | 5.58e-05 / ex 2.6e-13 | 0.0149 / ex 0.015 | 0.0149 / ex 0.015 | 0.0229 / ex 0.023 | 0.0229 / ex 0.023 | 0.0229 / ex 0.023 | 0.0229 / ex 0.023 | 0.0149 / ex 0.015 | 0.00566 / ex 0.0057 | 0.00566 / ex 0.0057 | 0.00566 / ex 0.0057 | 0.00566 / ex 0.0057 | 0.00566 / ex 0.0057 | 5.58e-05 / ex 2.6e-13 | 5.58e-05 / ex 2.6e-13 | 5.58e-05 / ex 2.6e-13 | 5.58e-05 / ex 2.6e-13 | 5.58e-05 / ex 2.6e-13 | 0.00252 / ex 0.00037 | 0.00252 / ex 0.00037 | 0.00566 / ex 0.0057 | 6.14e-05 / ex 5.3e-14 | 5.61e-05 / ex 9.7e-13 |
| stress/periodic_rc_pwm.yaml | 0.0119 / ex 0.012 | 0.00378 / ex 0.0038 | 0.000284 / ex 6.4e-07 | 2e-07 / ex 5.1e-08 | 4.38e-07 / ex 4e-14 | 0.00378 / ex 0.0038 | 0.000284 / ex 6.4e-07 | 0.0119 / ex 0.012 | 0.0119 / ex 0.012 | 0.0119 / ex 0.012 | 0.0119 / ex 0.012 | 0.000284 / ex 6.4e-07 | 2e-07 / ex 5.1e-08 | 2e-07 / ex 5.1e-08 | 2e-07 / ex 5.1e-08 | 2e-07 / ex 5.1e-08 | 2e-07 / ex 5.1e-08 | 4.38e-07 / ex 4e-14 | 4.38e-07 / ex 4e-14 | 4.38e-07 / ex 4e-14 | 4.38e-07 / ex 4e-14 | 4.38e-07 / ex 4e-14 | 2.08e-07 / ex 1.1e-07 | 2.08e-07 / ex 1.1e-07 | 2e-07 / ex 5.1e-08 | 1.14e-05 / ex 1.7e-15 | 5.12e-06 / ex 1.3e-15 |
| stress/stiff_rc_high_freq_switching.yaml | 0.00267 / ex 0.0027 | 2.03e-05 / ex 1.8e-05 | 2.03e-05 / ex 1.8e-05 | 1.43e-05 / ex 1e-05 | 1.1e-05 / ex 1.9e-14 | 2.03e-05 / ex 1.8e-05 | 2.03e-05 / ex 1.8e-05 | 0.00267 / ex 0.0027 | 0.00267 / ex 0.0027 | 0.00267 / ex 0.0027 | 0.00267 / ex 0.0027 | 2.03e-05 / ex 1.8e-05 | 1.43e-05 / ex 1e-05 | 1.43e-05 / ex 1e-05 | 1.43e-05 / ex 1e-05 | 1.43e-05 / ex 1e-05 | 1.43e-05 / ex 1e-05 | 1.1e-05 / ex 1.9e-14 | 1.1e-05 / ex 1.9e-14 | 1.1e-05 / ex 1.9e-14 | 1.1e-05 / ex 1.9e-14 | 1.1e-05 / ex 1.9e-14 | 1.1e-05 / ex 5.8e-09 | 1.1e-05 / ex 5.7e-09 | 1.43e-05 / ex 1e-05 | 0.00112 / ex 3.5e-15 | 6.89e-06 / ex 2.4e-14 |
| stress/stiff_rlc.yaml | 3.97e-09 / ex 3.1e-09 | 3.72e-09 / ex 3.8e-09 | 3.72e-09 / ex 3.8e-09 | 0 / ex 2.7e-09 | 3.56e-09 / ex 3.2e-09 | 3.72e-09 / ex 3.8e-09 | 3.72e-09 / ex 3.8e-09 | 3.97e-09 / ex 3.1e-09 | 3.97e-09 / ex 3.1e-09 | 3.97e-09 / ex 3.1e-09 | 3.97e-09 / ex 3.1e-09 | 3.72e-09 / ex 3.8e-09 | 0 / ex 2.7e-09 | 0 / ex 2.7e-09 | 0 / ex 2.7e-09 | 0 / ex 2.7e-09 | 0 / ex 2.7e-09 | 3.56e-09 / ex 3.2e-09 | 3.56e-09 / ex 3.2e-09 | 3.56e-09 / ex 3.2e-09 | 3.56e-09 / ex 3.2e-09 | 3.56e-09 / ex 3.2e-09 | 3.76e-09 / ex 3.2e-09 | 3.76e-09 / ex 3.2e-09 | 0 / ex 2.7e-09 | 1.68e-09 / ex 1e-09 | 3.56e-09 / ex 3.2e-09 |
| stress/stiff_two_timescale_rc.yaml | 7.85e-05 | 1.19e-06 | 1.19e-06 | 7.79e-07 | 4.64e-07 | 1.19e-06 | 1.19e-06 | 7.85e-05 | 7.85e-05 | 7.85e-05 | 7.85e-05 | 1.19e-06 | 7.79e-07 | 7.79e-07 | 7.79e-07 | 7.79e-07 | 7.79e-07 | 4.64e-07 | 4.64e-07 | 4.64e-07 | 4.64e-07 | 4.64e-07 | 6.93e-07 | 6.93e-07 | 7.79e-07 | 6.19e-07 | 7.75e-07 |
| switching/basic/chopper_with_rc_snubber.yaml | 0.0187 / ex 0.0047 | 0.0153 / ex 2.5e-08 | 0.0153 / ex 3.1e-08 | 3.81e-05 / ex 4.6e-08 | 2.91e-05 / ex 7.7e-15 | 0.0153 / ex 2.5e-08 | 0.0153 / ex 3.1e-08 | 0.0187 / ex 0.0047 | 0.0187 / ex 0.0047 | 0.0187 / ex 0.0047 | 0.0187 / ex 0.0047 | 0.0153 / ex 3.1e-08 | 3.81e-05 / ex 4.6e-08 | 3.81e-05 / ex 4.6e-08 | 3.81e-05 / ex 4.6e-08 | 3.81e-05 / ex 4.6e-08 | 3.81e-05 / ex 4.6e-08 | 2.91e-05 / ex 7.7e-15 | 2.91e-05 / ex 7.7e-15 | 2.91e-05 / ex 7.7e-15 | 2.91e-05 / ex 7.7e-15 | 2.91e-05 / ex 7.7e-15 | 5.57e-05 / ex 1.6e-08 | 5.57e-05 / ex 1.6e-08 | 3.81e-05 / ex 4.6e-08 | 3.07e-05 / ex 3.6e-15 | 2.91e-05 / ex 4.8e-15 |
| switching/basic/mosfet_nmos_dc.yaml | 3.55e-15 | 0 | 0 | 0 | refused | 0 | 0 | 3.55e-15 | 3.55e-15 | 3.55e-15 | 3.55e-15 | 0 | 0 | 0 | 0 | 0 | 0 | refused | refused | refused | refused | refused | 0 | 0 | 0 | 0 | 0 |
| switching/basic/pwm_voltage_rc.yaml | 0.000883 / ex 0.00088 | 9.98e-09 / ex 1.4e-08 | 9.98e-09 / ex 1.4e-08 | 8.98e-09 / ex 8.9e-09 | 1.15e-07 / ex 3.7e-13 | 9.98e-09 / ex 1.4e-08 | 9.98e-09 / ex 1.4e-08 | 0.000883 / ex 0.00088 | 0.000883 / ex 0.00088 | 0.000883 / ex 0.00088 | 0.000883 / ex 0.00088 | 9.98e-09 / ex 1.4e-08 | 8.98e-09 / ex 8.9e-09 | 8.98e-09 / ex 8.9e-09 | 8.98e-09 / ex 8.9e-09 | 8.98e-09 / ex 8.9e-09 | 8.98e-09 / ex 8.9e-09 | 1.15e-07 / ex 3.7e-13 | 1.15e-07 / ex 3.7e-13 | 1.15e-07 / ex 3.7e-13 | 1.15e-07 / ex 3.7e-13 | 1.15e-07 / ex 3.7e-13 | 4.6e-08 / ex 3.8e-08 | 4.6e-08 / ex 3.8e-08 | 8.98e-09 / ex 8.9e-09 | 1.34e-08 / ex 5.4e-14 | 1.57e-08 / ex 5.4e-14 |
| switching/basic/pwm_voltage_rl_filter.yaml | 0.00429 / ex 0.0042 | 6.93e-06 / ex 1.1e-06 | 6.93e-06 / ex 1.1e-06 | 6.53e-05 / ex 4.9e-06 | 2.26e-06 / ex 6.3e-14 | 6.93e-06 / ex 1.1e-06 | 6.93e-06 / ex 1.1e-06 | 0.00429 / ex 0.0042 | 0.00429 / ex 0.0042 | 0.00429 / ex 0.0042 | 0.00429 / ex 0.0042 | 6.93e-06 / ex 1.1e-06 | 6.53e-05 / ex 4.9e-06 | 6.53e-05 / ex 4.9e-06 | 6.53e-05 / ex 4.9e-06 | 6.53e-05 / ex 4.9e-06 | 6.53e-05 / ex 4.9e-06 | 2.26e-06 / ex 6.3e-14 | 2.26e-06 / ex 6.3e-14 | 2.26e-06 / ex 6.3e-14 | 2.26e-06 / ex 6.3e-14 | 2.26e-06 / ex 6.3e-14 | 6.64e-05 / ex 1.9e-06 | 6.64e-05 / ex 1.9e-06 | 6.53e-05 / ex 4.9e-06 | 6.94e-06 / ex 6.3e-14 | 6.05e-06 / ex 4.9e-15 |
| switching/basic/snubber_rc.yaml | 0.000561 | 0.000561 | 0.000561 | 1.65e-07 | 1.73e-07 | 0.000561 | 0.000561 | 0.000561 | 0.000561 | 0.000561 | 0.000561 | 0.000561 | 1.65e-07 | 1.65e-07 | 1.65e-07 | 1.65e-07 | 1.65e-07 | 1.73e-07 | 1.73e-07 | 1.73e-07 | 1.73e-07 | 1.73e-07 | 1.66e-07 | 1.66e-07 | 1.65e-07 | 1.66e-07 | 2.16e-08 |
| switching/basic/vcswitch_resistor.yaml | 0.0332 / ex 0 | 0.0682 / ex 0.059 | 0.0332 / ex 0 | 0 / ex 2e-09 | 3.16e-06 / ex 2e-09 | 0.0682 / ex 0.059 | 0.0332 / ex 0 | 0.0332 / ex 0 | 0.0332 / ex 0 | 0.0332 / ex 0 | 0.0332 / ex 0 | 0.0332 / ex 0 | 0 / ex 2e-09 | 0 / ex 2e-09 | 0 / ex 2e-09 | 0 / ex 2e-09 | 0 / ex 2e-09 | 3.16e-06 / ex 2e-09 | 3.16e-06 / ex 2e-09 | 3.16e-06 / ex 2e-09 | 3.16e-06 / ex 2e-09 | 3.16e-06 / ex 2e-09 | 0 / ex 2e-09 | 0 / ex 2e-09 | 0 / ex 2e-09 | 3.16e-06 / ex 2e-09 | 3.16e-06 / ex 2e-09 |
| switching/boost/boost_classic.yaml | 0.000313 / ex 0.00031 | 1.03e-06 / ex 1.5e-06 | 3.93e-07 / ex 6.3e-07 | 9.53e-06 / ex 7.7e-06 | 7.02e-07 / ex 5.2e-14 | 1.03e-06 / ex 1.5e-06 | 3.93e-07 / ex 6.3e-07 | 0.000313 / ex 0.00031 | 0.000313 / ex 0.00031 | 0.000313 / ex 0.00031 | 0.000313 / ex 0.00031 | 3.93e-07 / ex 6.3e-07 | 9.53e-06 / ex 7.7e-06 | 9.53e-06 / ex 7.7e-06 | 9.53e-06 / ex 7.7e-06 | 9.52e-06 / ex 7.7e-06 | 9.52e-06 / ex 7.7e-06 | 7.02e-07 / ex 5.2e-14 | 7.02e-07 / ex 5.2e-14 | 7.02e-07 / ex 5.2e-14 | 7.02e-07 / ex 1e-13 | 7.02e-07 / ex 1e-13 | 9.39e-06 / ex 1.5e-08 | 9.39e-06 / ex 1.7e-08 | 9.53e-06 / ex 7.7e-06 | 0.000684 / ex 1e-14 | 5.4e-06 / ex 1.3e-14 |
| switching/boost/boost_pfc_open_loop.yaml | 0.0304 / ex 0.03 | 0.0304 / ex 0.03 | 1.06e-05 / ex 1.6e-06 | 1.72e-07 / ex 1.7e-07 | 4.16e-07 / ex 2.2e-07 | 0.0304 / ex 0.03 | 1.06e-05 / ex 1.6e-06 | 0.0304 / ex 0.03 | 0.0304 / ex 0.03 | 0.0304 / ex 0.03 | 0.0304 / ex 0.03 | 1.06e-05 / ex 1.6e-06 | 1.72e-07 / ex 1.7e-07 | 1.72e-07 / ex 1.7e-07 | 1.72e-07 / ex 1.7e-07 | 1.78e-07 / ex 1.6e-07 | 1.78e-07 / ex 1.6e-07 | 4.15e-07 / ex 2.2e-07 | 4.16e-07 / ex 2.2e-07 | 4.16e-07 / ex 2.2e-07 | 4.16e-07 / ex 2.2e-07 | 4.15e-07 / ex 2.2e-07 | 2.56e-07 / ex 2.5e-08 | 2.53e-07 / ex 2.4e-08 | 1.72e-07 / ex 1.7e-07 | 3.17e-06 / ex 2.2e-07 | 2.7e-06 / ex 5.7e-14 |
| switching/boost/boost_switching_complex.yaml | 0.00192 / ex 0.0017 | 0.000334 / ex 0.00031 | 3.15e-05 / ex 1.9e-05 | 1.05e-05 / ex 7e-06 | 2.99e-05 / ex 4.7e-15 | 0.000334 / ex 0.00031 | 3.15e-05 / ex 1.9e-05 | 0.00192 / ex 0.0017 | 0.00192 / ex 0.0017 | 0.00192 / ex 0.0017 | 0.00192 / ex 0.0017 | 3.15e-05 / ex 1.9e-05 | 1.05e-05 / ex 7e-06 | 1.05e-05 / ex 7e-06 | 1.05e-05 / ex 7e-06 | 1.05e-05 / ex 7e-06 | 1.05e-05 / ex 7e-06 | 2.99e-05 / ex 4.7e-15 | 2.99e-05 / ex 4.7e-15 | 2.99e-05 / ex 4.7e-15 | 2.99e-05 / ex 2.1e-15 | 2.99e-05 / ex 2.1e-15 | 0.00016 / ex 6.2e-08 | 0.00016 / ex 6.2e-08 | 1.05e-05 / ex 7e-06 | 0.018 / ex 8.9e-16 | 2.32e-06 / ex 7.5e-16 |
| switching/boost/cascaded_boost_buck.yaml | 0.000399 / ex 0.0004 | 3.97e-06 / ex 3.3e-06 | 2.24e-06 / ex 2.4e-07 | 5.95e-05 / ex 3.2e-06 | 2.24e-06 / ex 3.6e-13 | 3.97e-06 / ex 3.3e-06 | 2.24e-06 / ex 2.4e-07 | 0.000399 / ex 0.0004 | 0.000399 / ex 0.0004 | 0.000399 / ex 0.0004 | 0.000399 / ex 0.0004 | 2.24e-06 / ex 2.4e-07 | 5.95e-05 / ex 3.2e-06 | 5.95e-05 / ex 3.2e-06 | 5.95e-05 / ex 3.2e-06 | 5.95e-05 / ex 3.2e-06 | 5.95e-05 / ex 3.2e-06 | 2.24e-06 / ex 3.6e-13 | 2.24e-06 / ex 3.6e-13 | 2.24e-06 / ex 3.6e-13 | 2.24e-06 / ex 2.3e-13 | 2.24e-06 / ex 2.3e-13 | 6.06e-05 / ex 4e-08 | 6.06e-05 / ex 4.2e-08 | 5.95e-05 / ex 3.2e-06 | 0.00173 / ex 2e-14 | 6.54e-06 / ex 1.8e-13 |
| switching/boost/ll07_boost_vcswitch.yaml | 0.0984 / ex 0.0022 | 0.0997 / ex 0.018 | 0.0997 / ex 3e-05 | 0.000117 / ex 0.00016 | 0.000181 / ex 2e-10 | 0.0997 / ex 0.018 | 0.0997 / ex 3e-05 | 0.0984 / ex 0.0022 | 0.0984 / ex 0.0022 | 0.0984 / ex 0.0022 | 0.0984 / ex 0.0022 | 0.0997 / ex 3e-05 | 0.000117 / ex 0.00016 | 0.000117 / ex 0.00016 | 0.000117 / ex 0.00016 | 0.000117 / ex 0.00015 | 0.000117 / ex 0.00015 | 0.000181 / ex 2e-10 | 0.000181 / ex 2e-10 | 0.000181 / ex 2e-10 | 0.000181 / ex 1.8e-10 | 0.000181 / ex 1.8e-10 | 0.000147 / ex 3.1e-06 | 0.000147 / ex 1.2e-06 | 0.000117 / ex 0.00016 | 0.0159 / ex 1.8e-10 | 0.000181 / ex 1.8e-10 |
| switching/boost/ll12_pfc_boost_continuous.yaml | 0.083 | 0.0806 | 0.0806 | 0.000115 | refused | 0.0806 | 0.0806 | 0.083 | 0.083 | 0.083 | 0.083 | 0.0806 | 0.000115 | 0.000115 | 0.000115 | 0.000115 | 0.000115 | refused | refused | refused | refused | refused | 0.000128 | 0.000128 | 0.000115 | 0.000115 | 0.000115 |
| switching/boost/synchronous_boost.yaml | 6.31e-05 / ex 6.3e-05 | 2.93e-07 / ex 2.2e-08 | 2.93e-07 / ex 2.2e-08 | 8.19e-08 / ex 2.7e-07 | 3.07e-07 / ex 5.6e-12 | 2.93e-07 / ex 2.2e-08 | 2.93e-07 / ex 2.2e-08 | 6.31e-05 / ex 6.3e-05 | 6.31e-05 / ex 6.3e-05 | 6.31e-05 / ex 6.3e-05 | 6.31e-05 / ex 6.3e-05 | 2.93e-07 / ex 2.2e-08 | 8.19e-08 / ex 2.7e-07 | 8.19e-08 / ex 2.7e-07 | 8.19e-08 / ex 2.7e-07 | 8.19e-08 / ex 2.7e-07 | 8.19e-08 / ex 2.7e-07 | 3.07e-07 / ex 5.6e-12 | 3.07e-07 / ex 5.6e-12 | 3.07e-07 / ex 5.6e-12 | 3.07e-07 / ex 5.4e-12 | 3.07e-07 / ex 5.4e-12 | 2.71e-07 / ex 5.7e-09 | 2.71e-07 / ex 5.7e-09 | 8.19e-08 / ex 2.7e-07 | 2.24e-05 / ex 5.5e-12 | 4.45e-06 / ex 5.5e-12 |
| switching/boost/textbook_erickson_boost_ccm.yaml | 0.000313 / ex 0.00031 | 1.03e-06 / ex 1.5e-06 | 3.93e-07 / ex 6.3e-07 | 9.53e-06 / ex 7.7e-06 | 7.02e-07 / ex 5.2e-14 | 1.03e-06 / ex 1.5e-06 | 3.93e-07 / ex 6.3e-07 | 0.000313 / ex 0.00031 | 0.000313 / ex 0.00031 | 0.000313 / ex 0.00031 | 0.000313 / ex 0.00031 | 3.93e-07 / ex 6.3e-07 | 9.53e-06 / ex 7.7e-06 | 9.53e-06 / ex 7.7e-06 | 9.53e-06 / ex 7.7e-06 | 9.52e-06 / ex 7.7e-06 | 9.52e-06 / ex 7.7e-06 | 7.02e-07 / ex 5.2e-14 | 7.02e-07 / ex 5.2e-14 | 7.02e-07 / ex 5.2e-14 | 7.02e-07 / ex 1e-13 | 7.02e-07 / ex 1e-13 | 9.39e-06 / ex 1.5e-08 | 9.39e-06 / ex 1.7e-08 | 9.53e-06 / ex 7.7e-06 | 0.000684 / ex 1e-14 | 5.4e-06 / ex 1.3e-14 |
| switching/boost/two_phase_interleaved_boost.yaml | 0.00019 / ex 0.00019 | 1.02e-06 / ex 1.3e-06 | 4.23e-07 / ex 5e-07 | 5.56e-06 / ex 4.8e-06 | 6.63e-07 / ex 1.2e-13 | 1.02e-06 / ex 1.3e-06 | 4.23e-07 / ex 5e-07 | 0.00019 / ex 0.00019 | 0.00019 / ex 0.00019 | 0.00019 / ex 0.00019 | 0.00019 / ex 0.00019 | 4.23e-07 / ex 5e-07 | 5.56e-06 / ex 4.8e-06 | 5.56e-06 / ex 4.8e-06 | 5.56e-06 / ex 4.8e-06 | 5.54e-06 / ex 4.8e-06 | 5.54e-06 / ex 4.8e-06 | 6.63e-07 / ex 1.2e-13 | 6.63e-07 / ex 1.2e-13 | 6.63e-07 / ex 1.2e-13 | 6.63e-07 / ex 4.6e-14 | 6.63e-07 / ex 4.6e-14 | 6.26e-06 / ex 1.6e-08 | 6.26e-06 / ex 1.6e-08 | 5.56e-06 / ex 4.8e-06 | 0.000234 / ex 4e-14 | 6.29e-06 / ex 5.5e-15 |
| switching/buck/buck_dcm.yaml | 0.00183 / ex 0.0018 | 3.4e-06 / ex 3.3e-06 | 3.09e-06 / ex 3e-06 | 7.7e-06 / ex 3e-06 | 6e-07 / ex 1.3e-14 | 3.4e-06 / ex 3.3e-06 | 3.09e-06 / ex 3e-06 | 0.00183 / ex 0.0018 | 0.00183 / ex 0.0018 | 0.00183 / ex 0.0018 | 0.00183 / ex 0.0018 | 3.09e-06 / ex 3e-06 | 7.7e-06 / ex 3e-06 | 7.7e-06 / ex 3e-06 | 7.7e-06 / ex 3e-06 | 7.7e-06 / ex 3e-06 | 7.7e-06 / ex 3e-06 | 6e-07 / ex 1.3e-14 | 6e-07 / ex 1.3e-14 | 6e-07 / ex 1.3e-14 | 6e-07 / ex 1.3e-14 | 6e-07 / ex 1.3e-14 | 6.87e-06 / ex 1.5e-08 | 6.87e-06 / ex 1.5e-08 | 7.7e-06 / ex 3e-06 | 0.000994 / ex 2.3e-14 | 4.82e-06 / ex 1.6e-14 |
| switching/buck/buck_hard_switching_reference.yaml | 0.00102 / ex 0.001 | 1.43e-06 / ex 1.3e-06 | 7.61e-07 / ex 2.8e-07 | 2.27e-06 / ex 8.8e-07 | 8.28e-07 / ex 4.2e-14 | 1.43e-06 / ex 1.3e-06 | 7.61e-07 / ex 2.8e-07 | 0.00102 / ex 0.001 | 0.00102 / ex 0.001 | 0.00102 / ex 0.001 | 0.00102 / ex 0.001 | 7.61e-07 / ex 2.8e-07 | 2.27e-06 / ex 8.8e-07 | 2.27e-06 / ex 8.8e-07 | 2.27e-06 / ex 8.8e-07 | 2.27e-06 / ex 8.8e-07 | 2.27e-06 / ex 8.8e-07 | 8.28e-07 / ex 4.2e-14 | 8.28e-07 / ex 4.2e-14 | 8.28e-07 / ex 4.2e-14 | 8.28e-07 / ex 4.3e-14 | 8.28e-07 / ex 4.3e-14 | 2.16e-06 / ex 7.3e-09 | 2.16e-06 / ex 7.3e-09 | 2.27e-06 / ex 8.8e-07 | 0.0008 / ex 5.2e-15 | 5.25e-06 / ex 1.3e-14 |
| switching/buck/buck_mosfet_nonlinear.yaml | 0.00841 | 0.00232 | 4.07e-07 | 5.08e-07 | refused | 0.00232 | 4.07e-07 | 0.00841 | 0.00841 | 0.00841 | 0.00841 | 4.07e-07 | 5.08e-07 | 5.08e-07 | 5.08e-07 | 5.08e-07 | 5.08e-07 | refused | refused | refused | refused | refused | 5.17e-07 | 5.17e-07 | 5.08e-07 | 5.08e-07 | 5.08e-07 |
| switching/buck/buck_switching.yaml | 0.00168 / ex 0.0017 | 9.61e-07 / ex 3.6e-07 | 9.53e-07 / ex 3.2e-07 | 1.36e-06 / ex 4.9e-07 | 1.03e-06 / ex 1.6e-14 | 9.61e-07 / ex 3.6e-07 | 9.53e-07 / ex 3.2e-07 | 0.00168 / ex 0.0017 | 0.00168 / ex 0.0017 | 0.00168 / ex 0.0017 | 0.00168 / ex 0.0017 | 9.53e-07 / ex 3.2e-07 | 1.36e-06 / ex 4.9e-07 | 1.36e-06 / ex 4.9e-07 | 1.36e-06 / ex 4.9e-07 | 1.36e-06 / ex 4.9e-07 | 1.36e-06 / ex 4.9e-07 | 1.03e-06 / ex 1.6e-14 | 1.03e-06 / ex 1.6e-14 | 1.03e-06 / ex 1.6e-14 | 1.03e-06 / ex 1.4e-14 | 1.03e-06 / ex 1.4e-14 | 1.38e-06 / ex 1.8e-08 | 1.38e-06 / ex 1.8e-08 | 1.36e-06 / ex 4.9e-07 | 0.0009 / ex 2.9e-15 | 5.32e-06 / ex 1.2e-14 |
| switching/buck/buck_with_current_sense.yaml | 0.00131 / ex 0.0013 | 1.05e-06 / ex 7.3e-07 | 1.05e-06 / ex 7.3e-07 | 3.15e-06 / ex 2.3e-06 | 1.33e-06 / ex 4.9e-14 | 1.05e-06 / ex 7.3e-07 | 1.05e-06 / ex 7.3e-07 | 0.00131 / ex 0.0013 | 0.00131 / ex 0.0013 | 0.00131 / ex 0.0013 | 0.00131 / ex 0.0013 | 1.05e-06 / ex 7.3e-07 | 3.15e-06 / ex 2.3e-06 | 3.15e-06 / ex 2.3e-06 | 3.15e-06 / ex 2.3e-06 | 3.15e-06 / ex 2.3e-06 | 3.15e-06 / ex 2.3e-06 | 1.33e-06 / ex 4.9e-14 | 1.33e-06 / ex 4.9e-14 | 1.33e-06 / ex 4.9e-14 | 1.33e-06 / ex 4.9e-14 | 1.33e-06 / ex 4.9e-14 | 2.99e-06 / ex 9.4e-09 | 2.99e-06 / ex 9.4e-09 | 3.15e-06 / ex 2.3e-06 | 0.00124 / ex 6.9e-16 | 5.63e-06 / ex 5.4e-15 |
| switching/buck/cascaded_buck_buck.yaml | 0.00126 / ex 0.0013 | 1.16e-06 / ex 1.4e-06 | 1.16e-06 / ex 1.4e-06 | 1.42e-05 / ex 9.8e-06 | 1.46e-06 / ex 8.6e-14 | 1.16e-06 / ex 1.4e-06 | 1.16e-06 / ex 1.4e-06 | 0.00126 / ex 0.0013 | 0.00126 / ex 0.0013 | 0.00126 / ex 0.0013 | 0.00126 / ex 0.0013 | 1.16e-06 / ex 1.4e-06 | 1.42e-05 / ex 9.8e-06 | 1.42e-05 / ex 9.8e-06 | 1.42e-05 / ex 9.8e-06 | 1.42e-05 / ex 9.8e-06 | 1.42e-05 / ex 9.8e-06 | 1.46e-06 / ex 8.6e-14 | 1.46e-06 / ex 8.6e-14 | 1.46e-06 / ex 8.6e-14 | 1.46e-06 / ex 6.6e-14 | 1.46e-06 / ex 6.6e-14 | 1.13e-05 / ex 1.5e-08 | 1.13e-05 / ex 1.5e-08 | 1.42e-05 / ex 9.8e-06 | 0.00184 / ex 1.9e-15 | 5.44e-06 / ex 1.2e-14 |
| switching/buck/high_freq_gan_buck.yaml | 0.00103 / ex 0.001 | 6.2e-06 / ex 3.6e-06 | 6.2e-06 / ex 3.6e-06 | 8.55e-05 / ex 2.7e-05 | 5.1e-06 / ex 2.6e-14 | 6.2e-06 / ex 3.6e-06 | 6.2e-06 / ex 3.6e-06 | 0.00103 / ex 0.001 | 0.00103 / ex 0.001 | 0.00103 / ex 0.001 | 0.00103 / ex 0.001 | 6.2e-06 / ex 3.6e-06 | 8.55e-05 / ex 2.7e-05 | 8.55e-05 / ex 2.7e-05 | 8.55e-05 / ex 2.7e-05 | 8.55e-05 / ex 2.7e-05 | 8.55e-05 / ex 2.7e-05 | 5.1e-06 / ex 2.6e-14 | 5.1e-06 / ex 2.6e-14 | 5.1e-06 / ex 2.6e-14 | 5.1e-06 / ex 2.6e-14 | 5.1e-06 / ex 2.6e-14 | 0.000132 / ex 4.4e-09 | 0.000132 / ex 4.4e-09 | 8.55e-05 / ex 2.7e-05 | 0.0153 / ex 1.6e-14 | 6.93e-06 / ex 9.6e-15 |
| switching/buck/ll06_buck_vcswitch.yaml | 0.0781 / ex 0.055 | 0.118 / ex 0.12 | 0.0704 / ex 0.013 | 8.21e-05 / ex 6.3e-06 | 0.000171 / ex 9.5e-12 | 0.118 / ex 0.12 | 0.0704 / ex 0.013 | 0.0781 / ex 0.055 | 0.0781 / ex 0.055 | 0.0781 / ex 0.055 | 0.0781 / ex 0.055 | 0.0704 / ex 0.013 | 8.21e-05 / ex 6.3e-06 | 8.21e-05 / ex 6.3e-06 | 8.21e-05 / ex 6.3e-06 | 8.21e-05 / ex 6.3e-06 | 8.21e-05 / ex 6.3e-06 | 0.000171 / ex 9.5e-12 | 0.000171 / ex 9.5e-12 | 0.000171 / ex 9.5e-12 | 0.000171 / ex 9.5e-12 | 0.000171 / ex 9.5e-12 | 0.000202 / ex 5e-07 | 0.000202 / ex 5e-07 | 8.21e-05 / ex 6.3e-06 | 0.0112 / ex 0.011 | 0.000171 / ex 8.8e-12 |
| switching/buck/ll08_buck_mosfet.yaml | 0.227 | 0.0843 | 0.0843 | 0.00011 | refused | 0.0843 | 0.0843 | 0.227 | 0.227 | 0.227 | 0.227 | 0.0843 | 0.00011 | 0.00011 | 0.00011 | 0.000101 | 0.000101 | refused | refused | refused | refused | refused | 0.000178 | 0.000179 | 0.00011 | 0.00011 | 0.000101 |
| switching/buck/sync_buck_500khz.yaml | 0.00167 / ex 0.0017 | 3.77e-06 / ex 4.4e-06 | 3.77e-06 / ex 4.4e-06 | 1.71e-05 / ex 1.4e-05 | 1.98e-06 / ex 1.1e-13 | 3.77e-06 / ex 4.4e-06 | 3.77e-06 / ex 4.4e-06 | 0.00167 / ex 0.0017 | 0.00167 / ex 0.0017 | 0.00167 / ex 0.0017 | 0.00167 / ex 0.0017 | 3.77e-06 / ex 4.4e-06 | 1.71e-05 / ex 1.4e-05 | 1.71e-05 / ex 1.4e-05 | 1.71e-05 / ex 1.4e-05 | 1.71e-05 / ex 1.4e-05 | 1.71e-05 / ex 1.4e-05 | 1.98e-06 / ex 1.1e-13 | 1.98e-06 / ex 1.1e-13 | 1.98e-06 / ex 1.1e-13 | 1.98e-06 / ex 1.1e-13 | 1.98e-06 / ex 1.1e-13 | 1.13e-05 / ex 9e-09 | 1.13e-05 / ex 9e-09 | 1.71e-05 / ex 1.4e-05 | 0.00115 / ex 1.1e-15 | 6.67e-06 / ex 5.5e-15 |
| switching/buck/synchronous_buck.yaml | 0.00132 / ex 0.0013 | 1.05e-06 / ex 7.5e-07 | 1.05e-06 / ex 7.5e-07 | 3.14e-06 / ex 2.3e-06 | 1.35e-06 / ex 2.1e-14 | 1.05e-06 / ex 7.5e-07 | 1.05e-06 / ex 7.5e-07 | 0.00132 / ex 0.0013 | 0.00132 / ex 0.0013 | 0.00132 / ex 0.0013 | 0.00132 / ex 0.0013 | 1.05e-06 / ex 7.5e-07 | 3.14e-06 / ex 2.3e-06 | 3.14e-06 / ex 2.3e-06 | 3.14e-06 / ex 2.3e-06 | 3.14e-06 / ex 2.3e-06 | 3.14e-06 / ex 2.3e-06 | 1.35e-06 / ex 2.1e-14 | 1.35e-06 / ex 2.1e-14 | 1.35e-06 / ex 2.1e-14 | 1.35e-06 / ex 2.1e-14 | 1.35e-06 / ex 2.1e-14 | 2.98e-06 / ex 9.5e-09 | 2.98e-06 / ex 9.5e-09 | 3.14e-06 / ex 2.3e-06 | 0.00124 / ex 1.1e-15 | 5.59e-06 / ex 2.7e-15 |
| switching/buck/textbook_erickson_buck_ccm.yaml | 0.000771 / ex 0.00077 | 1.76e-06 / ex 1.7e-06 | 4.21e-07 / ex 1.4e-07 | 9.87e-06 / ex 1.7e-06 | 4.3e-07 / ex 1.2e-13 | 1.76e-06 / ex 1.7e-06 | 4.21e-07 / ex 1.4e-07 | 0.000771 / ex 0.00077 | 0.000771 / ex 0.00077 | 0.000771 / ex 0.00077 | 0.000771 / ex 0.00077 | 4.21e-07 / ex 1.4e-07 | 9.87e-06 / ex 1.7e-06 | 9.87e-06 / ex 1.7e-06 | 9.87e-06 / ex 1.7e-06 | 9.87e-06 / ex 1.7e-06 | 9.87e-06 / ex 1.7e-06 | 4.3e-07 / ex 1.2e-13 | 4.3e-07 / ex 1.2e-13 | 4.3e-07 / ex 1.2e-13 | 4.3e-07 / ex 1.2e-13 | 4.3e-07 / ex 1.2e-13 | 9.71e-06 / ex 1.3e-08 | 9.71e-06 / ex 1.3e-08 | 9.87e-06 / ex 1.7e-06 | 0.00092 / ex 1.2e-14 | 6.72e-06 / ex 1.2e-14 |
| switching/buck/triple_sync_buck.yaml | 0.00132 / ex 0.0013 | 1.02e-06 / ex 7.3e-07 | 1.02e-06 / ex 7.3e-07 | 3.18e-06 / ex 2.3e-06 | 1.3e-06 / ex 6e-14 | 1.02e-06 / ex 7.3e-07 | 1.02e-06 / ex 7.3e-07 | 0.00132 / ex 0.0013 | 0.00132 / ex 0.0013 | 0.00132 / ex 0.0013 | 0.00132 / ex 0.0013 | 1.02e-06 / ex 7.3e-07 | 3.18e-06 / ex 2.3e-06 | 3.18e-06 / ex 2.3e-06 | 3.18e-06 / ex 2.3e-06 | 3.18e-06 / ex 2.3e-06 | 3.18e-06 / ex 2.3e-06 | 1.3e-06 / ex 6e-14 | 1.3e-06 / ex 6e-14 | 1.3e-06 / ex 6e-14 | 1.3e-06 / ex 6e-14 | 1.3e-06 / ex 6e-14 | 2.99e-06 / ex 1.1e-08 | 2.99e-06 / ex 1.1e-08 | 3.18e-06 / ex 2.3e-06 | 0.000309 / ex 3.7e-15 | 4.59e-06 / ex 2.2e-14 |
| switching/buck_boost/bidirectional_buck_boost.yaml | 0.00132 / ex 0.0013 | 1.05e-06 / ex 7.4e-07 | 1.05e-06 / ex 7.4e-07 | 3.15e-06 / ex 2.3e-06 | 1.34e-06 / ex 2.6e-14 | 1.05e-06 / ex 7.4e-07 | 1.05e-06 / ex 7.4e-07 | 0.00132 / ex 0.0013 | 0.00132 / ex 0.0013 | 0.00132 / ex 0.0013 | 0.00132 / ex 0.0013 | 1.05e-06 / ex 7.4e-07 | 3.15e-06 / ex 2.3e-06 | 3.15e-06 / ex 2.3e-06 | 3.15e-06 / ex 2.3e-06 | 3.15e-06 / ex 2.3e-06 | 3.15e-06 / ex 2.3e-06 | 1.34e-06 / ex 2.6e-14 | 1.34e-06 / ex 2.6e-14 | 1.34e-06 / ex 2.6e-14 | 1.34e-06 / ex 3.6e-14 | 1.34e-06 / ex 3.6e-14 | 2.99e-06 / ex 9.5e-09 | 2.99e-06 / ex 9.5e-09 | 3.15e-06 / ex 2.3e-06 | 0.00124 / ex 5.4e-15 | 5.61e-06 / ex 4.4e-15 |
| switching/buck_boost/buck_boost.yaml | 0.00072 / ex 0.00072 | 9.09e-07 / ex 1.4e-06 | 6.34e-07 / ex 6.1e-07 | 7.97e-06 / ex 7.6e-06 | 1.06e-06 / ex 2.6e-13 | 9.09e-07 / ex 1.4e-06 | 6.34e-07 / ex 6.1e-07 | 0.00072 / ex 0.00072 | 0.00072 / ex 0.00072 | 0.00072 / ex 0.00072 | 0.00072 / ex 0.00072 | 6.34e-07 / ex 6.1e-07 | 7.97e-06 / ex 7.6e-06 | 7.97e-06 / ex 7.6e-06 | 7.97e-06 / ex 7.6e-06 | 7.97e-06 / ex 7.6e-06 | 7.97e-06 / ex 7.6e-06 | 1.06e-06 / ex 2.6e-13 | 1.06e-06 / ex 2.6e-13 | 1.06e-06 / ex 2.6e-13 | 1.06e-06 / ex 1e-13 | 1.06e-06 / ex 1e-13 | 9.03e-06 / ex 4e-08 | 9.03e-06 / ex 4e-08 | 7.97e-06 / ex 7.6e-06 | 0.00072 / ex 1.3e-14 | 4.97e-06 / ex 6.5e-15 |
| switching/buck_boost/cuk_converter.yaml | 0.000854 / ex 0.00085 | 7.85e-06 / ex 9.7e-06 | 6.2e-06 / ex 8.4e-06 | 9.25e-05 / ex 9.3e-05 | 2.51e-06 / ex 3.5e-13 | 7.85e-06 / ex 9.7e-06 | 6.2e-06 / ex 8.4e-06 | 0.000854 / ex 0.00085 | 0.000854 / ex 0.00085 | 0.000854 / ex 0.00085 | 0.000854 / ex 0.00085 | 6.2e-06 / ex 8.4e-06 | 9.25e-05 / ex 9.3e-05 | 9.25e-05 / ex 9.3e-05 | 9.25e-05 / ex 9.3e-05 | 9.25e-05 / ex 9.3e-05 | 9.25e-05 / ex 9.3e-05 | 2.51e-06 / ex 3.5e-13 | 2.51e-06 / ex 3.5e-13 | 2.51e-06 / ex 3.5e-13 | 2.51e-06 / ex 3.6e-13 | 2.51e-06 / ex 3.6e-13 | 2.59e-05 / ex 4.1e-08 | 2.59e-05 / ex 4.2e-08 | 9.25e-05 / ex 9.3e-05 | 0.00212 / ex 5.5e-15 | 4.32e-06 / ex 3.2e-14 |
| switching/buck_boost/sepic.yaml | 0.00186 / ex 0.0019 | 6e-06 / ex 6.5e-06 | 1.73e-06 / ex 1.3e-06 | 2.96e-05 / ex 1.5e-05 | 1.51e-06 / ex 1.3e-13 | 6e-06 / ex 6.5e-06 | 1.73e-06 / ex 1.3e-06 | 0.00186 / ex 0.0019 | 0.00186 / ex 0.0019 | 0.00186 / ex 0.0019 | 0.00186 / ex 0.0019 | 1.73e-06 / ex 1.3e-06 | 2.96e-05 / ex 1.5e-05 | 2.96e-05 / ex 1.5e-05 | 2.96e-05 / ex 1.5e-05 | 2.96e-05 / ex 1.5e-05 | 2.96e-05 / ex 1.5e-05 | 1.51e-06 / ex 1.3e-13 | 1.51e-06 / ex 1.3e-13 | 1.51e-06 / ex 1.3e-13 | 1.51e-06 / ex 9.8e-14 | 1.51e-06 / ex 9.8e-14 | 3.81e-05 / ex 2.5e-08 | 3.81e-05 / ex 2.6e-08 | 2.96e-05 / ex 1.5e-05 | 0.00193 / ex 4.6e-14 | 6.37e-07 / ex 1.6e-13 |
| switching/inverter/half_bridge_inverter.yaml | 0.000376 / ex 0.00038 | 1.16e-06 / ex 7.5e-08 | 1.16e-06 / ex 7.5e-08 | 9.06e-07 / ex 9.4e-07 | 1.2e-06 / ex 1.6e-13 | 1.16e-06 / ex 7.5e-08 | 1.16e-06 / ex 7.5e-08 | 0.000376 / ex 0.00038 | 0.000376 / ex 0.00038 | 0.000376 / ex 0.00038 | 0.000376 / ex 0.00038 | 1.16e-06 / ex 7.5e-08 | 9.06e-07 / ex 9.4e-07 | 9.06e-07 / ex 9.4e-07 | 9.06e-07 / ex 9.4e-07 | 9.06e-07 / ex 9.4e-07 | 9.06e-07 / ex 9.4e-07 | 1.2e-06 / ex 1.6e-13 | 1.2e-06 / ex 1.6e-13 | 1.2e-06 / ex 1.6e-13 | 1.2e-06 / ex 1.6e-13 | 1.2e-06 / ex 1.6e-13 | 1.1e-06 / ex 3.3e-08 | 1.1e-06 / ex 3.3e-08 | 9.06e-07 / ex 9.4e-07 | 0.000125 / ex 2.5e-15 | 5.8e-06 / ex 2.5e-15 |
| switching/inverter/half_bridge_inverter_lc.yaml | 0.00103 / ex 0.001 | 7.91e-06 / ex 1.2e-05 | 7.91e-06 / ex 1.2e-05 | 0.000303 / ex 0.00035 | 3.89e-06 / ex 2.7e-12 | 7.91e-06 / ex 1.2e-05 | 7.91e-06 / ex 1.2e-05 | 0.00103 / ex 0.001 | 0.00103 / ex 0.001 | 0.00103 / ex 0.001 | 0.00103 / ex 0.001 | 7.91e-06 / ex 1.2e-05 | 0.000303 / ex 0.00035 | 0.000303 / ex 0.00035 | 0.000303 / ex 0.00035 | 0.000303 / ex 0.00035 | 0.000303 / ex 0.00035 | 3.89e-06 / ex 2.7e-12 | 3.89e-06 / ex 2.7e-12 | 3.89e-06 / ex 2.7e-12 | 3.89e-06 / ex 2.7e-12 | 3.89e-06 / ex 2.7e-12 | 0.00296 / ex 3.4e-05 | 0.00296 / ex 3.4e-05 | 0.000303 / ex 0.00035 | 0.000122 / ex 5.6e-13 | 7.85e-06 / ex 2.7e-13 |
| switching/inverter/half_bridge_lc_filter.yaml | 0.00131 / ex 0.0013 | 1.11e-06 / ex 9.1e-07 | 1.11e-06 / ex 9.1e-07 | 5e-06 / ex 4e-06 | 1.48e-06 / ex 3.1e-14 | 1.11e-06 / ex 9.1e-07 | 1.11e-06 / ex 9.1e-07 | 0.00131 / ex 0.0013 | 0.00131 / ex 0.0013 | 0.00131 / ex 0.0013 | 0.00131 / ex 0.0013 | 1.11e-06 / ex 9.1e-07 | 5e-06 / ex 4e-06 | 5e-06 / ex 4e-06 | 5e-06 / ex 4e-06 | 5e-06 / ex 4e-06 | 5e-06 / ex 4e-06 | 1.48e-06 / ex 3.1e-14 | 1.48e-06 / ex 3.1e-14 | 1.48e-06 / ex 3.1e-14 | 1.48e-06 / ex 3.1e-14 | 1.48e-06 / ex 3.1e-14 | 4.16e-06 / ex 2.5e-08 | 4.17e-06 / ex 2.5e-08 | 5e-06 / ex 4e-06 | 0.00122 / ex 2.5e-15 | 5.82e-06 / ex 1.2e-14 |
| switching/inverter/lcc_resonant_inverter.yaml | 0.00565 / ex 0.0056 | 0.000232 / ex 0.00019 | 0.000232 / ex 0.00019 | 8.37e-05 / ex 5.4e-05 | 8.81e-05 / ex 1e-14 | 0.000232 / ex 0.00019 | 0.000232 / ex 0.00019 | 0.00565 / ex 0.0056 | 0.00565 / ex 0.0056 | 0.00565 / ex 0.0056 | 0.00565 / ex 0.0056 | 0.000232 / ex 0.00019 | 8.37e-05 / ex 5.4e-05 | 8.37e-05 / ex 5.4e-05 | 8.37e-05 / ex 5.4e-05 | 8.37e-05 / ex 5.4e-05 | 8.37e-05 / ex 5.4e-05 | 8.81e-05 / ex 1e-14 | 8.81e-05 / ex 1e-14 | 8.81e-05 / ex 1e-14 | 8.81e-05 / ex 1e-14 | 8.81e-05 / ex 1e-14 | 0.0014 / ex 3.1e-06 | 0.0014 / ex 3.1e-06 | 8.37e-05 / ex 5.4e-05 | 0.288 / ex 1.2e-15 | 5.38e-06 / ex 1.5e-14 |
| switching/inverter/ll09_half_bridge_inverter_leg.yaml | 0.0707 / ex 0.063 | 0.0587 / ex 0.0098 | 0.0587 / ex 0.0098 | 4.27e-05 / ex 3.2e-06 | 0.000167 / ex 4e-13 | 0.0587 / ex 0.0098 | 0.0587 / ex 0.0098 | 0.0707 / ex 0.063 | 0.0707 / ex 0.063 | 0.0707 / ex 0.063 | 0.0707 / ex 0.063 | 0.0587 / ex 0.0098 | 4.27e-05 / ex 3.2e-06 | 4.27e-05 / ex 3.2e-06 | 4.27e-05 / ex 3.2e-06 | 4.27e-05 / ex 3.2e-06 | 4.27e-05 / ex 3.2e-06 | 0.000167 / ex 4e-13 | 0.000167 / ex 4e-13 | 0.000167 / ex 4e-13 | 0.000167 / ex 4e-13 | 0.000167 / ex 4e-13 | 0.000261 / ex 2.9e-06 | 0.000261 / ex 2.9e-06 | 4.27e-05 / ex 3.2e-06 | 0.000167 / ex 4e-13 | 0.000167 / ex 4.8e-11 |
| switching/inverter/ll10_full_bridge_inverter_lc.yaml | 0.05 / ex 0.0018 | 0.05 / ex 1.6e-07 | 0.05 / ex 1.6e-07 | 9.18e-05 / ex 4.9e-05 | 0.000127 / ex 1.4e-11 | 0.05 / ex 1.6e-07 | 0.05 / ex 1.6e-07 | 0.05 / ex 0.0018 | 0.05 / ex 0.0018 | 0.05 / ex 0.0018 | 0.05 / ex 0.0018 | 0.05 / ex 1.6e-07 | 9.18e-05 / ex 4.9e-05 | 9.18e-05 / ex 4.9e-05 | 9.18e-05 / ex 4.9e-05 | 8.74e-05 / ex 3.1e-05 | 8.74e-05 / ex 3.1e-05 | 0.000127 / ex 1.4e-11 | 0.000127 / ex 1.4e-11 | 0.000127 / ex 1.4e-11 | 0.000127 / ex 1.4e-11 | 0.000127 / ex 1.4e-11 | 9.7e-05 / ex 5.7e-05 | 9.3e-05 / ex 3.9e-05 | 9.18e-05 / ex 4.9e-05 | 0.000128 / ex 1.4e-11 | 0.000128 / ex 1.4e-11 |
| switching/isolated/active_clamp_forward.yaml | 0.00425 / ex 0.0043 | 0.00375 / ex 0.0038 | 0.00374 / ex 0.0037 | 0.000117 / ex 0.00012 | 1.24e-06 / ex 2.5e-08 | 0.00375 / ex 0.0038 | 0.00374 / ex 0.0037 | 0.00425 / ex 0.0043 | 0.00425 / ex 0.0043 | 0.00425 / ex 0.0043 | 0.00425 / ex 0.0043 | 0.00374 / ex 0.0037 | 0.000117 / ex 0.00012 | 0.000117 / ex 0.00012 | 0.000117 / ex 0.00012 | 0.000117 / ex 0.00012 | 0.000117 / ex 0.00012 | 1.24e-06 / ex 2.5e-08 | 1.24e-06 / ex 2.5e-08 | 1.24e-06 / ex 2.5e-08 | 1.24e-06 / ex 2.5e-08 | 1.24e-06 / ex 2.5e-08 | 3.42e-05 / ex 3.7e-05 | 3.42e-05 / ex 3.7e-05 | 0.000117 / ex 0.00012 | 0.00077 / ex 2.7e-06 | 3.66e-06 / ex 2.5e-08 |
| switching/isolated/dual_active_bridge.yaml | 0.000265 / ex 0.00027 | 2.4e-06 / ex 1.1e-06 | 2.4e-06 / ex 1.1e-06 | 3.49e-05 / ex 9.1e-06 | 1.86e-06 / ex 1.5e-11 | 2.4e-06 / ex 1.1e-06 | 2.4e-06 / ex 1.1e-06 | 0.000265 / ex 0.00027 | 0.000265 / ex 0.00027 | 0.000265 / ex 0.00027 | 0.000265 / ex 0.00027 | 2.4e-06 / ex 1.1e-06 | 3.49e-05 / ex 9.1e-06 | 3.49e-05 / ex 9.1e-06 | 3.49e-05 / ex 9.1e-06 | 3.49e-05 / ex 9.1e-06 | 3.49e-05 / ex 9.1e-06 | 1.86e-06 / ex 1.5e-11 | 1.86e-06 / ex 1.5e-11 | 1.86e-06 / ex 1.5e-11 | 1.86e-06 / ex 1.5e-11 | 1.86e-06 / ex 1.5e-11 | 3.31e-05 / ex 1.8e-09 | 3.31e-05 / ex 7.3e-10 | 3.49e-05 / ex 9.1e-06 | 0.00133 / ex 1.5e-11 | 3.68e-06 / ex 1.5e-11 |
| switching/isolated/flyback_converter.yaml | 0.949 / ex 0.95 | 2.26e-06 / ex 2.5e-06 | 2.26e-06 / ex 2.5e-06 | 4.29e-05 / ex 3.6e-05 | 2.7e-06 / ex 2.1e-06 | 2.26e-06 / ex 2.5e-06 | 2.26e-06 / ex 2.5e-06 | 0.949 / ex 0.95 | 0.949 / ex 0.95 | 0.949 / ex 0.95 | 0.949 / ex 0.95 | 2.26e-06 / ex 2.5e-06 | 4.29e-05 / ex 3.6e-05 | 4.29e-05 / ex 3.6e-05 | 4.29e-05 / ex 3.6e-05 | 4.29e-05 / ex 3.6e-05 | 4.29e-05 / ex 3.6e-05 | 2.67e-06 / ex 2.1e-06 | 2.7e-06 / ex 2.1e-06 | 2.7e-06 / ex 2.1e-06 | 2.7e-06 / ex 2.1e-06 | 2.67e-06 / ex 2.1e-06 | 8.27e-05 / ex 3e-06 | 8.27e-05 / ex 3e-06 | 4.29e-05 / ex 3.6e-05 | 0.00622 / ex 2.1e-06 | 6.69e-06 / ex 3.9e-07 |
| switching/isolated/forward_converter.yaml | 0.707 / ex 0.71 | 5.14e-06 / ex 5e-06 | 5.07e-06 / ex 4.9e-06 | 2.17e-05 / ex 1.2e-06 | 1.12e-06 / ex 4.7e-08 | 5.14e-06 / ex 5e-06 | 5.07e-06 / ex 4.9e-06 | 0.707 / ex 0.71 | 0.707 / ex 0.71 | 0.707 / ex 0.71 | 0.707 / ex 0.71 | 5.07e-06 / ex 4.9e-06 | 2.17e-05 / ex 1.2e-06 | 2.17e-05 / ex 1.2e-06 | 2.17e-05 / ex 1.2e-06 | 2.17e-05 / ex 1.3e-06 | 2.17e-05 / ex 1.3e-06 | 1.12e-06 / ex 4.7e-08 | 1.12e-06 / ex 4.7e-08 | 1.12e-06 / ex 4.7e-08 | 1.12e-06 / ex 4.7e-08 | 1.12e-06 / ex 4.7e-08 | 2.16e-05 / ex 4.8e-08 | 2.16e-05 / ex 4.8e-08 | 2.17e-05 / ex 1.2e-06 | 0.00164 / ex 4.7e-08 | 6.13e-06 / ex 1e-09 |
| switching/isolated/full_bridge_dc.yaml | 0.0828 / ex 0.014 | 0.0815 / ex 3.8e-09 | 0.0815 / ex 3.9e-09 | 0.000141 / ex 5e-09 | 2.66e-05 / ex 4.6e-17 | 0.0815 / ex 3.8e-09 | 0.0815 / ex 3.9e-09 | 0.0828 / ex 0.014 | 0.0828 / ex 0.014 | 0.0828 / ex 0.014 | 0.0828 / ex 0.014 | 0.0815 / ex 3.8e-09 | 0.000141 / ex 5e-09 | 0.000141 / ex 5e-09 | 0.000141 / ex 5e-09 | 0.000141 / ex 5e-09 | 0.000141 / ex 5e-09 | 2.66e-05 / ex 4.6e-17 | 2.66e-05 / ex 4.6e-17 | 2.66e-05 / ex 4.6e-17 | 2.66e-05 / ex 5.2e-17 | 2.66e-05 / ex 5.2e-17 | 5.69e-05 / ex 9.9e-12 | 5.69e-05 / ex 1.7e-09 | 0.000141 / ex 5e-09 | 2.89e-05 / ex 5.6e-17 | 2.66e-05 / ex 5.1e-17 |
| switching/isolated/ll11_llc_resonant_converter.yaml | error | error | error | error | refused | error | error | error | error | error | error | error | error | error | error | error | error | refused | refused | refused | refused | refused | error | error | error | error | error |
| switching/isolated/ll13_active_clamp_forward.yaml | 4.51 | 0.335 | 0.335 | 0.00133 | refused | 0.335 | 0.335 | 4.51 | 4.51 | 4.51 | 4.51 | 0.335 | 0.00133 | 0.00133 | 0.00133 | 0.00155 | 0.00155 | refused | refused | refused | refused | refused | 0.00156 | 0.00155 | 0.00133 | 0.00133 | 0.00155 |
| switching/isolated/synchronous_flyback.yaml | 0.207 / ex 0.21 | 1.01e-06 / ex 1.3e-07 | 1.01e-06 / ex 1.3e-07 | 9.15e-06 / ex 6e-06 | 1.02e-06 / ex 1e-13 | 1.01e-06 / ex 1.3e-07 | 1.01e-06 / ex 1.3e-07 | 0.207 / ex 0.21 | 0.207 / ex 0.21 | 0.207 / ex 0.21 | 0.207 / ex 0.21 | 1.01e-06 / ex 1.3e-07 | 9.15e-06 / ex 6e-06 | 9.15e-06 / ex 6e-06 | 9.15e-06 / ex 6e-06 | 9.15e-06 / ex 6e-06 | 9.15e-06 / ex 6e-06 | 1.02e-06 / ex 1e-13 | 1.02e-06 / ex 1e-13 | 1.02e-06 / ex 1e-13 | 1.02e-06 / ex 6.1e-14 | 1.02e-06 / ex 6.1e-14 | 1.16e-05 / ex 2.3e-08 | 1.16e-05 / ex 2.3e-08 | 9.15e-06 / ex 6e-06 | 0.00109 / ex 1.3e-14 | 6.2e-06 / ex 7.2e-11 |
| switching/multilevel/cascaded_h_bridge_5level.yaml | 0.0274 / ex 0.005 | 0.0269 / ex 6e-16 | 0.0269 / ex 6e-16 | 0 / ex 6e-16 | 5.69e-05 / ex 5.6e-16 | 0.0269 / ex 6e-16 | 0.0269 / ex 6e-16 | 0.0274 / ex 0.005 | 0.0274 / ex 0.005 | 0.0274 / ex 0.005 | 0.0274 / ex 0.005 | 0.0269 / ex 1.9e-16 | 0 / ex 6e-16 | 0 / ex 6e-16 | 0 / ex 6e-16 | 5.34e-16 / ex 1.9e-16 | 5.34e-16 / ex 1.9e-16 | 5.69e-05 / ex 5.6e-16 | 5.69e-05 / ex 5.6e-16 | 5.69e-05 / ex 5.6e-16 | 5.69e-05 / ex 1.9e-16 | 5.69e-05 / ex 1.9e-16 | 0 / ex 6e-16 | 5.34e-16 / ex 1.9e-16 | 0 / ex 6e-16 | 5.72e-05 / ex 5.7e-16 | 5.72e-05 / ex 1.9e-16 |
| switching/multilevel/flying_capacitor_3level.yaml | 0.0466 / ex 0.0058 | 0.05 / ex 1.5e-06 | 0.05 / ex 1.5e-06 | 6.78e-05 / ex 1.2e-05 | 3.71e-05 / ex 1.3e-13 | 0.05 / ex 1.5e-06 | 0.05 / ex 1.5e-06 | 0.0466 / ex 0.0058 | 0.0466 / ex 0.0058 | 0.0466 / ex 0.0058 | 0.0466 / ex 0.0058 | 0.05 / ex 1.5e-06 | 6.78e-05 / ex 1.2e-05 | 6.78e-05 / ex 1.2e-05 | 6.78e-05 / ex 1.2e-05 | 6.78e-05 / ex 1.2e-05 | 6.78e-05 / ex 1.2e-05 | 3.71e-05 / ex 1.3e-13 | 3.71e-05 / ex 1.3e-13 | 3.71e-05 / ex 1.3e-13 | 3.71e-05 / ex 6.7e-13 | 3.71e-05 / ex 6.7e-13 | 6.7e-05 / ex 5e-08 | 6.7e-05 / ex 4.9e-08 | 6.78e-05 / ex 1.2e-05 | 0.00613 / ex 1.1e-12 | 3.74e-05 / ex 6.4e-14 |
| switching/multilevel/mmc_4cell_chain.yaml | 0.0268 / ex 0.005 | 0.0264 / ex 1.2e-13 | 0.0264 / ex 1.2e-13 | 0 / ex 1.2e-13 | 5.74e-05 / ex 1.3e-13 | 0.0264 / ex 1.2e-13 | 0.0264 / ex 1.2e-13 | 0.0268 / ex 0.005 | 0.0268 / ex 0.005 | 0.0268 / ex 0.005 | 0.0268 / ex 0.005 | 0.0264 / ex 2.2e-13 | 0 / ex 1.2e-13 | 0 / ex 1.2e-13 | 0 / ex 1.2e-13 | 2.83e-13 / ex 2.2e-13 | 2.83e-13 / ex 2.2e-13 | 5.74e-05 / ex 1.3e-13 | 5.74e-05 / ex 1.3e-13 | 5.74e-05 / ex 1.3e-13 | 5.74e-05 / ex 1.2e-13 | 5.74e-05 / ex 1.2e-13 | 0 / ex 1.2e-13 | 2.83e-13 / ex 2.2e-13 | 0 / ex 1.2e-13 | 5.76e-05 / ex 1.3e-13 | 5.76e-05 / ex 1.2e-13 |
| switching/multilevel/npc_three_level_halfbridge.yaml | 0.00319 / ex 0.0032 | 5.06e-06 / ex 4e-07 | 5.06e-06 / ex 4e-07 | 7.9e-06 / ex 1.8e-06 | 5.09e-06 / ex 7e-14 | 5.06e-06 / ex 4e-07 | 5.06e-06 / ex 4e-07 | 0.00319 / ex 0.0032 | 0.00319 / ex 0.0032 | 0.00319 / ex 0.0032 | 0.00319 / ex 0.0032 | 5.06e-06 / ex 4e-07 | 7.9e-06 / ex 1.8e-06 | 7.9e-06 / ex 1.8e-06 | 7.9e-06 / ex 1.8e-06 | 7.9e-06 / ex 1.8e-06 | 7.9e-06 / ex 1.8e-06 | 5.09e-06 / ex 7e-14 | 5.09e-06 / ex 7e-14 | 5.09e-06 / ex 7e-14 | 5.09e-06 / ex 4.9e-13 | 5.09e-06 / ex 4.9e-13 | 7.9e-06 / ex 1.3e-07 | 7.9e-06 / ex 1.3e-07 | 7.9e-06 / ex 1.8e-06 | 0.00245 / ex 1.3e-14 | 5.55e-06 / ex 4.1e-13 |
| switching/multilevel/t_type_3level_halfbridge.yaml | 0.0782 / ex 0.012 | 0.0771 / ex 2.1e-07 | 0.0771 / ex 2.1e-07 | 0.000112 / ex 9.3e-07 | 4.24e-05 / ex 1.7e-15 | 0.0771 / ex 2.1e-07 | 0.0771 / ex 2.1e-07 | 0.0782 / ex 0.012 | 0.0782 / ex 0.012 | 0.0782 / ex 0.012 | 0.0782 / ex 0.012 | 0.0771 / ex 2.1e-07 | 0.000112 / ex 9.3e-07 | 0.000112 / ex 9.3e-07 | 0.000112 / ex 9.3e-07 | 0.000112 / ex 9.3e-07 | 0.000112 / ex 9.3e-07 | 4.24e-05 / ex 1.7e-15 | 4.24e-05 / ex 1.7e-15 | 4.24e-05 / ex 1.7e-15 | 4.24e-05 / ex 5.2e-15 | 4.24e-05 / ex 5.2e-15 | 0.000111 / ex 2.8e-09 | 0.000111 / ex 2.8e-09 | 0.000112 / ex 9.3e-07 | 0.00377 / ex 5.5e-16 | 4.24e-05 / ex 7e-15 |
| thermal/electrothermal_resistor_self_heating.yaml | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0 / ex 0 | 0.00178 / ex 0 | 0.00178 / ex 0 |
| thermal/long_run_drift_buck.yaml | 0.000603 / ex 0.0006 | 1.12e-05 / ex 3.3e-06 | 1.12e-05 / ex 3.3e-06 | 2.58e-06 / ex 5.7e-07 | 1.07e-05 / ex 6.3e-14 | 1.12e-05 / ex 3.3e-06 | 1.12e-05 / ex 3.3e-06 | 0.000603 / ex 0.0006 | 0.000603 / ex 0.0006 | 0.000603 / ex 0.0006 | 0.000603 / ex 0.0006 | 1.12e-05 / ex 3.3e-06 | 2.58e-06 / ex 5.7e-07 | 2.58e-06 / ex 5.7e-07 | 2.58e-06 / ex 5.7e-07 | 2.58e-06 / ex 5.7e-07 | 2.58e-06 / ex 5.7e-07 | 1.07e-05 / ex 6.3e-14 | 1.07e-05 / ex 6.3e-14 | 1.07e-05 / ex 6.3e-14 | 1.07e-05 / ex 6.3e-14 | 1.07e-05 / ex 6.3e-14 | 2.55e-06 / ex 8.5e-10 | 2.55e-06 / ex 8.6e-10 | 2.58e-06 / ex 5.7e-07 | 0.000272 / ex 6.3e-14 | 7.04e-06 / ex 1.5e-13 |
| three_phase/interleaved_buck_3ph.yaml | 0.00217 / ex 0.0022 | 0.00249 / ex 0.0025 | 1.75e-06 / ex 1.7e-06 | 6.64e-08 / ex 1.5e-07 | 2.2e-07 / ex 4.6e-14 | 0.00249 / ex 0.0025 | 1.75e-06 / ex 1.7e-06 | 0.00217 / ex 0.0022 | 0.00217 / ex 0.0022 | 0.00217 / ex 0.0022 | 0.00217 / ex 0.0022 | 1.75e-06 / ex 1.7e-06 | 6.64e-08 / ex 1.5e-07 | 6.64e-08 / ex 1.5e-07 | 6.64e-08 / ex 1.5e-07 | 6.65e-08 / ex 1.4e-07 | 6.65e-08 / ex 1.4e-07 | 2.2e-07 / ex 4.6e-14 | 2.2e-07 / ex 4.6e-14 | 2.2e-07 / ex 4.6e-14 | 2.2e-07 / ex 4.7e-14 | 2.2e-07 / ex 4.7e-14 | 1.23e-07 / ex 3.7e-08 | 1.23e-07 / ex 3.6e-08 | 6.64e-08 / ex 1.5e-07 | 7.02e-06 / ex 4.9e-15 | 1.28e-06 / ex 1.1e-14 |
| three_phase/three_phase_balanced_rl_load.yaml | 1.82e-06 / ex 5.7e-07 | 1.83e-06 / ex 5.5e-07 | 1.83e-06 / ex 5.5e-07 | 4.22e-05 / ex 3.4e-06 | 6.01e-07 / ex 5.2e-12 | 1.83e-06 / ex 5.5e-07 | 1.83e-06 / ex 5.5e-07 | 1.82e-06 / ex 5.7e-07 | 1.82e-06 / ex 5.7e-07 | 1.82e-06 / ex 5.7e-07 | 1.82e-06 / ex 5.7e-07 | 1.83e-06 / ex 5.5e-07 | 4.22e-05 / ex 3.4e-06 | 4.22e-05 / ex 3.4e-06 | 4.22e-05 / ex 3.4e-06 | 4.22e-05 / ex 3.4e-06 | 4.22e-05 / ex 3.4e-06 | 6.01e-07 / ex 5.2e-12 | 6.01e-07 / ex 5.2e-12 | 6.01e-07 / ex 5.2e-12 | 6.01e-07 / ex 5.2e-12 | 6.01e-07 / ex 5.2e-12 | 4.77e-05 / ex 4.1e-09 | 4.77e-05 / ex 4.1e-09 | 4.22e-05 / ex 3.4e-06 | 6.04e-07 / ex 5.2e-12 | 6.04e-07 / ex 5.2e-12 |
| three_phase/three_phase_diode_rectifier.yaml | 1.51 / ex 0.23 | 0.00632 / ex 0.001 | 0.00631 / ex 0.001 | 5.51e-05 / ex 3e-06 | 0.00034 / ex 1.9e-14 | 0.00632 / ex 0.001 | 0.00631 / ex 0.001 | 1.51 / ex 0.23 | 1.51 / ex 0.23 | 1.51 / ex 0.23 | 1.51 / ex 0.23 | 0.00631 / ex 0.001 | 5.51e-05 / ex 3e-06 | 5.51e-05 / ex 3e-06 | 5.51e-05 / ex 3e-06 | 5.51e-05 / ex 3e-06 | 5.51e-05 / ex 3e-06 | 0.00034 / ex 1.9e-14 | 0.00034 / ex 1.9e-14 | 0.00034 / ex 1.9e-14 | 0.00034 / ex 1.9e-14 | 0.00034 / ex 1.9e-14 | 8.53e-05 / ex 1.3e-06 | 8.53e-05 / ex 1.3e-06 | 5.51e-05 / ex 3e-06 | 1.83e-05 / ex 5.2e-14 | 5.47e-06 / ex 5.2e-14 |

## Notes

Hand-written; kept by `render`.

- **How each option is applied.** The `simulate()` keywords through `reference_status.extra_kwargs`
  (the same injection the reference harness uses for `pwl+restart` / `pwl+split`); `form='branch'`
  by patching `CircuitBuilder.add_switch` / `add_diode` for the build (`ideal_switch_flip_evidence.branch_form`:
  `R_on = 1/g_on`, the same `g_off`; MOSFET / IGBT Level-1 helpers and Shockley diodes are not reached);
  the `auto` route by `pulsim.exact_route` (every segment exact -> the exact engine, else trbdf2), both
  without `dt`. The route is an emulation of D11 by whole run, not per segment.
- **Evidence that an option reached the run** (`moved` and `mode` columns): `mode_selection='auto'`
  chose `lcp` on the 39 units with a PWL diode on pwl and the exact engine and `legacy` elsewhere; on trbdf2 it
  always resolves to `legacy` (DECISIONS F5: no 'auto' rule there), so it cannot change a trbdf2 number.
  `control_tick='state'` moved the 4 pwl closed loops (`cl_boost_pi`, `cl_buck_aggressive_pi`,
  `cl_buck_pi`, `ll16_flyback_closed_loop`; no outcome changed) and no trbdf2 / exact cell (those engines
  land ticks at their instants; nothing shows the keyword did anything there beyond being accepted).
  `device_path='interface'` moved only the two induction-machine analytic cases, by rounding, on pwl and
  trbdf2; the Level-1 MOSFET / IGBT circuits are bit-identical under it (one law on both paths,
  device-interface DECISIONS 44; checked on `buck_mosfet_nonlinear`, states equal) and the result records
  no path, so for them the option's application is not observable. The branch form moved 56 / 65 / 45
  cells (pwl / trbdf2 / exact), no outcome. On pwl and the exact engine by rounding; on trbdf2 by more:
  44 of 65 moved more than 1e-6 relative in err_rms (e.g. `ll13` +17 %, `ll08` -8 %, `ll10` -5 %), because
  the adaptive step sequence follows the different stamp.
- **The restart rule alone is worse on 4 cells** (`pwl:restart` against `pwl`: `vcswitch_resistor`,
  `ll06_buck_vcswitch`, `ll07_boost_vcswitch` on `exact`, `analytic:scr_halfwave_r` agree -> disagree);
  with the split on (`pwl+split`, `pwl+next`) none is. The two flip together (add-fixed-step-event-splitting
  DECISIONS 1), so this is not a release blocker; it is why they must not flip apart.
- **Pytest refusal groups** (blocker section 1b) are matched by message substring over the stored
  failures (`SUITE_ROOTS`); the rest of the blast radius is classified by `next_defaults_pytest.classify`
  from the `E` lines and is a heuristic (a failure can sit in a neighbouring class).
- **`pwl:split` is `pwl+split`**: `enable_substep_state_correction=True` alone is refused by
  `run_transient` (it requires the restart rule), so the split's single column is the rule plus the split.
- **ll06 (blocker 1 of the run without `output_rtol`; class 3 above under the profile).** Graded column by column (`exact_scoreboard.grade`): `V(sw)` 1.1e-2, `V(n_l)`
  1.1e-2, `V(out)` 3.8e-7, `I(L1)` 1.6e-5. The first departure is the sample at the D1 turn-off event
  (t = 1.825982343832e-5 s, 99 % of span). Time-weighted `V(out)` over [1, 2] ms: 21.773493 V without dt,
  21.774224 V with the netlist dt (part of that is the trapezoid on sparse samples; the per-sample
  measure above is the evidence).
- **Not applied** (listed under Profile): the TR-BDF2 step grid (option missing), the C++-only TR-BDF2
  options, the composite diode drop (already the default on this tree).
- **Wall times** are single runs on a shared machine at `nice -n 15`, information only.
- **Route on the analytic set.** The analytic cases' `auto:route` / `auto+next` call `exact_route` on the
  built circuit only, without the case's own keywords (e.g. a `b_extra_fn`); the run that follows does pass
  them, and a refusal there is recorded as such.
- **Left from the review** (one fresh-context reviewer, opus, medium): a `timeout` cell is cached and not
  retried on resume (delete the cell to retry); the cache signature covers the configuration and the
  profile table, not the library build. (The header's commit is now read from git at render time.)
- **This regeneration (integration batch 14o, 2026-11-19).** Every cell re-measured from an empty cache
  (`run --jobs 1`, `nice -n 15`, 16:20-16:30, no timeout) with the extension built at the library commit
  642573e2 (`feat/core-integration-wip`: batch 14n plus the exact engine's snapshot / resume and junction loop,
  the closed-loop sensitivities and the release options merged; the evidence merge that carries this file
  changes no library file), macOS arm64, CPython 3.13. The profile now holds `output_rtol = 1e-5` (batch 14n)
  and the two `PLAN` library defaults (`_plan.DEFAULT_UNDRIVEN = 'refuse'`, `_plan.UNIFORM_ENGINE_RULES = True`);
  the full-profile cells (`<engine>+next`, `trbdf2+next+esdirk43`, `auto+next`) run under
  `next_defaults.plan_defaults()` (no tool applied `PLAN` to a cell before). No differential or analytic cell
  changed outcome under `PLAN` (no reference circuit has an undriven switch); every count equals batch 14n's
  re-render (`auto+next` 106 / 2 / 1). The ll06 entry above describes the run without `output_rtol`; under the
  profile ll06 agrees on `auto+next`. Pytest blast radius on the same tree (`-p next_defaults_pytest`, 21 min):
  583 failed / 47 errors (480 / 0 at a67d3994 without `output_rtol` and `PLAN`; 518 / 0 with `PLAN` alone,
  update-number-changing-release DECISIONS 4). The 47 errors and 25 of the failures are one cause (blocker 1b):
  the exact engine's `sensitivities=` run refuses `output_rtol` by name, and the harness injects `output_rtol`
  into every exact call without dt, fixtures included; 24 failures are the `undriven='refuse'` default. One
  test (in `test_exact_route.py`) ran about 9 minutes under the profile before failing (not identified; the
  per-test limit is 900 s).
