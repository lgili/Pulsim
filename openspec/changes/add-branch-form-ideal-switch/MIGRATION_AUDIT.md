# Migration audit for the default flips (tasks 9.1-9.3)

**This document flips no default.** It lists what the two planned flips
would touch. Date: 2026-11-18. Worktree: `/Users/lgili/pulsim_wt/phase0`.

- `add-branch-form-ideal-switch` 9.1-9.3 / design D6: the default diode and
  switch form becomes branch, ideal (`R_on = 0`, `V_f = 0`); `form='conductance'`
  warns; topology helpers follow the default.
- `add-complementarity-mode-selection` 9.1-9.3: default `mode_selection='lcp'`
  on every engine; `'legacy'` warns, later refused; settle loops deleted one
  release later.

Method: `rg` over the tree, excluding `build*/`, `.scratch/`, `node_modules`,
`.git`, `openspec/`. Counts are call sites (`rg -c`) / files. Facts carry
`file:line`. Anything marked **(inferred)** was not run, only read.

## 0. Headline findings

| # | Finding | Evidence |
|---|---|---|
| F1 | No caller in the repository passes `form='conductance'` or `mode_selection='legacy'`. Every diode/switch and every simulation relies on today's defaults for both. | `rg "form\s*=\s*['\"]conductance"` and `rg "mode_selection\s*=\s*['\"]legacy"` hit only `yaml/loader.hpp:263` (an error message) |
| F2 | The Python `add_switch` and the conductance `add_diode` have **no numeric defaults** today: `g_on`/`g_off` are required. D6 row 1 ("`add_switch(name, a, b)`: `g_on = 1e3`, `g_off = 1e-9`") does not match the code. So every existing call passes `g_on, g_off`, and after the flip each one maps to `R_on = 1/g_on` (D6 row 2), not to the ideal element. | `bindings.cpp:1230-1234`, `bindings.cpp:1191-1195`, `circuit_builder.hpp:1576-1578` |
| F3 | `add_switch` has no `form=` keyword. Today the only branch-form switch is `add_ideal_switch`. A "keep today's numbers" path for switches (`form='conductance'`) has to be added before 9.2. | `bindings.cpp:1230`, `_pulsim.pyi:544` |
| F4 | The branch form runs on `pwl` and `trbdf2` only. `dsed`/exact, `bdf1` and the C++ AC sweep refuse it by name. The flip-evidence run measured 54 of 109 reference circuits newly refused on dsed. | `device_pool.hpp:1422-1429` (`kBranchFormCaps`), `unsupported.hpp:65-87`, `bdf1.hpp:305`, `mna_sweep.hpp:85`, `dsed/builder_adapter.hpp:159`; DECISIONS #39 |
| F5 | Today `'lcp'` refuses several common cases: circuits with no PWL diode, a Newton device, `max_event_iterations=0`, `engine='trbdf2'` (hook 4.2 is still missing), a non-rk45 dsed integrator, and `sensitivities=`. If the default became a literal `'lcp'`, all of these runs would raise. **(inferred)** The default needs a fallback that a caller who passes `'lcp'` explicitly does not get. | `run_transient.hpp:946-981`, `__init__.py:2521-2527`, `__init__.py:2779-2786`, `_dsed_dispatch.py:895-903` |
| F6 | `g_on > 1e9 S` maps to `R_on < 1e-9 Ω`, which the builder refuses (`kBranchRonMin`). One call site seen. | `circuit_builder.hpp:2391-2405`; `python/tests/test_switch_conductance_validation.py:63` (`g_on = 1e12`) |

## 1. Public API

### 1a. Builder entry points (Python binding → C++)

| Entry point | file:line | Current default | After the flip (D6) |
|---|---|---|---|
| `CircuitBuilder.add_diode(name, a, k, g_on=None, g_off=None, V_th=0.0, V_f=0.0, *, form='conductance', R_on=None)` | `python/bindings.cpp:1171-1213`; stub `python/pulsim/_pulsim.pyi:395` | `form='conductance'`; g_on/g_off required (ValueError when missing, `:1191`); branch: `R_on` = given R_on, else `1/g_on`, else 0; `g_off` 0 (`:1179-1184`) | `form='branch'` default |
| C++ `CircuitBuilder::add_diode(name, a, k, g_on, g_off, V_th=0, V_f=0)` | `core/include/pulsim/builder/circuit_builder.hpp:1533` | conductance, no form parameter | not covered by D6 (D6 is the Python/YAML table): owner decision |
| C++ `add_diode_branch(name, a, k, R_on=0, g_off=0, V_th=0, V_f=0)` | `circuit_builder.hpp:1609` | branch | unchanged |
| `add_switch(name, n_pos, n_neg, g_on, g_off)` | `bindings.cpp:1230-1237`; `_pulsim.pyi:544`; C++ `circuit_builder.hpp:1576` | conductance; g_on/g_off required; no `form=` | branch, `R_on = 1/g_on` (D6); needs a `form=` keyword (F3) |
| `add_ideal_switch(name, n_pos, n_neg, R_on=0.0, g_off=0.0)` | `bindings.cpp:1221-1229`; `_pulsim.pyi:411`; C++ `circuit_builder.hpp:1593` | branch, ideal | becomes an alias of `add_switch` (D6) |
| `add_vcswitch` | not found | — | — (the `vcswitch` type exists only in the v1 reader, §1d) |
| Branch-form R_on bound | `circuit_builder.hpp:2391` `kBranchRonMin = 1e-9` | refuses `0 < R_on < 1e-9` | also catches converted `g_on > 1e9` (F6) |

### 1b. Device helpers that build a PWL diode or switch inside

| Entry point | file:line | Current default | Note |
|---|---|---|---|
| `add_mosfet(name, d, s, R_on=1e-3, R_off=1e9, body_diode=True, V_F=0.7)` | `bindings.cpp:1239-1261`; `_pulsim.pyi:450`; C++ `circuit_builder.hpp:1669` | switch `g = 1/R_on, 1/R_off` conductance; body diode via `add_mosfet_with_body_diode` | D6: "follows the default; `form=` accepted". No `form=` today |
| `add_mosfet_with_body_diode(..., R_on=1e-3, R_off=1e9, V_F=0.7, g_on_diode=1e3, g_off_diode=1e-9)` | `bindings.cpp:1262-1277`; `_pulsim.pyi:458`; C++ `circuit_builder.hpp:1695-1729` | conductance switch and conductance body diode (`V_th = 0`, `V_f = V_F`) | the parameter names `g_on_diode`/`g_off_diode` are conductance-only |
| `add_igbt(name, c, e, R_on=10e-3, R_off=1e9)` | `bindings.cpp:1278`; `_pulsim.pyi:419`; C++ `circuit_builder.hpp:1736-1747` | conductance switch via `add_switch` | follows `add_switch` |
| `add_mosfet_level1(..., with_body_diode=True, V_F=0.7)` | `bindings.cpp:991`; `_pulsim.pyi:454`; body diode hard-coded at `circuit_builder.hpp:450-453` | body diode `g_on=1e3`, `g_off=1e-9`, `V_th=0` (literals) | Newton device in the circuit: `'lcp'` refuses (F5) |
| `add_igbt_level1(..., with_fwd=False, V_F=0.7)` | `bindings.cpp:946`; `_pulsim.pyi:423`; diode hard-coded at `circuit_builder.hpp:965-968` | `g_on=1e3`, `g_off=1e-9` (literals) | same |
| `add_power_mosfet(..., body_diode=None\|dict)` | `bindings.cpp:1715`; C++ `circuit_builder.hpp:1880` | Lauritzen/Shockley body diode (Newton), no PWL diode | not affected by the form flip; Newton device → `'lcp'` refusal (F5) |

### 1c. Topology helpers (`python/pulsim/topology.py`)

| Helper | file:line | Current defaults | Note |
|---|---|---|---|
| `add_bridge_rectifier` | `topology.py:119` (defaults `:126-129`, call `:197`) | `g_on=1e3`, `g_off=1e-9`, `V_th=0.7`, `V_f=0.0` | no `form=` |
| `add_three_phase_vsi` | `topology.py:208` (defaults `:216-221`, calls `:351`, `:361`) | `R_on=1e-3`, `R_off=1e9`, `V_f=0.7`, `R_d=0.05`, `V_th=0.0` | `R_on <= 0` falls back to `g_on = 1e3` and `R_d <= 0` to `1e3` (`:338`, `:358`): today `R_on=0` quietly means 1 mΩ. After the flip it should mean ideal |
| `add_buck` | `topology.py:525` (`:536-540`, calls `:622`, `:625`) | `g_on=1e3`, `g_off=1e-9`, `diode_g_on=1e3`, `diode_g_off=1e-6`, `V_th=0.0` | no `form=` |
| `add_boost` | `topology.py:636` (`:647-651`, calls `:699`, `:702`) | same as buck | no `form=` |
| `add_flyback` | `topology.py:710` (`:723-727`, calls `:785`, `:798`) | same as buck | no `form=` |
| SPICE importer, LTspice `D(Ron Roff Vfwd)` | `python/pulsim/spice_import.py:677` | `add_diode(d, a, k, area/Ron, area/Roff, V_th=0, V_f=Vfwd)` | follows `add_diode`; `Ron < 1e-9·area` would be refused (F6) **(inferred)** |

### 1d. YAML / netlist readers

| Reader | file:line | Current default | After the flip |
|---|---|---|---|
| C++ YAML `form:` validation | `core/include/pulsim/yaml/loader.hpp:257-264` | `conductance \| branch`, switch/diode only | — |
| YAML `diode` | `loader.hpp:520-542` | no `form` → conductance, `g_on`/`g_off` **required** (`:539-540`); `form: branch` → `R_on` = given, else `1/g_on`, else 0; `g_off` 0 | `form: branch` default (D6) |
| YAML `switch` | `loader.hpp:570-586` | same as diode (`:581-586` conductance) | `form: branch` default |
| YAML `mosfet` / `mosfet_with_body_diode` / `igbt` | `loader.hpp:587`, `:596`, `:606` | conductance; `g_on_diode` 1e3, `g_off_diode` 1e-9 | follow the helpers |
| YAML `mosfet_level1` / `igbt_level1` | `loader.hpp:311`, `:419` (`with_fwd` `:430`) | hard-coded body/fwd diode (1b) | follow the helpers |
| YAML emitter | `core/include/pulsim/yaml/emitter.hpp:325`, `:608-610` | writes a `branch_form:` section | round-trips both forms; no change seen as needed |
| Benchmark v1 reader | `benchmarks/tools/v1_reader.py:279-284` | `diode`: `g_on` 1e3 / `g_off` 1e-9 when absent; `switch`: 1e6 / 1e-12; `vcswitch`: 1e3 / 1e-9; always calls `add_diode`/`add_switch` with both conductances | becomes branch `R_on = 1/g_on` through D6 row 2. Consumers: `goldens.py:75`, `counters.py:65`, `differential_runner.py:294`, `reference_status.py:82` |
| `benchmarks/analytic/__init__.py` | `:239`, `:246`, `:264`, `:292`, `:325-326` | explicit g_on/g_off | follows `add_diode`/`add_switch` |

`benchmarks/circuits/` holds 76 YAML files with PWL devices: 71 PWL diodes,
11 switches and 116 vcswitches, all with `g_on`/`ron` given; 12 Shockley
diodes are not affected (counted with a YAML parse).

### 1e. Simulation options

| Entry point | file:line | Current default | After the flip |
|---|---|---|---|
| `pulsim.simulate(..., mode_selection=None)` | `python/pulsim/__init__.py:1973`; docs `:2197-2210`; validation `:2382`; maps to the int at `:3142` (`{"shadow":1,"lcp":2}.get(..., 0)`) | `None` = `'legacy'` = settle (0) | `'lcp'`; `'legacy'` warns (9.1), refused (9.3) |
| `simulate(sensitivities=...)` + lcp | `__init__.py:2521-2527` | refused for `'shadow'`/`'lcp'` | would refuse every sensitivity run under a literal default (F5) |
| `simulate(engine='trbdf2'/'auto')` + lcp | `__init__.py:2779-2786` | refused ("whose settle has no hook yet") | needs task 4.2 first |
| `engine='auto'`, branch form, no dt | `__init__.py:2728-2747` | a trbdf2 `branch_form`/`dependent_states` refusal routes to pwl, or raises ValueError "pass dt" when no dt | **(inferred)** more auto runs may land on this path after the form flip |
| dsed dispatch `mode_selection=None` | `python/pulsim/_dsed_dispatch.py:609`, `:895-910` | settle plus guard (`MODE_GUARD = True`, `:570`) | `'lcp'` refuses non-rk45 integrators and diode-free circuits (`:895-903`) |
| pwl guard switch | `__init__.py:1918` `_PWL_MODE_GUARD = True`, `:3143` | guard on | deleted at 9.3 |
| C++ `SimulationOptions::mode_selection` | `core/include/pulsim/solver/options.hpp:189` (`int = 0`), `mode_guard = true` `:193`; binding `bindings.cpp:2420`; stub `_pulsim.pyi:1272-1277` | 0 = settle | 2 |
| C++ refusals under `'lcp'` | `core/include/pulsim/solver/run_transient.hpp:946-951` (no diodes), `:967-973` (Newton device), `:975-981` (`max_event_iterations=0`) | raise `UnsupportedError(pwl, mode_selection)` | F5 |
| dsed C++ scheduler | `core/include/pulsim/dsed/scheduler.hpp:1245`, `:1887` (`lcp_decides_`) | settle | — |

### 1f. Downstream consumers that read `g_on` (affected by `R_on = 0`)

| Consumer | file:line | Behaviour with an ideal element (`g_on = inf`) |
|---|---|---|
| `device_thermal_summary` | `python/pulsim/thermal.py:861-868` (diode), `:903-910` (switch) | `continue`: the element is left out of the thermal summary **with no warning** |
| period KPIs | `python/pulsim/_period_kpis.py:91-94` | returns a refusal string ("without finite conductances") |
| `result.i()` switch path | `python/pulsim/__init__.py:1399-1402` | reads `g_on`; `__init__.py:1223` handles the branch form separately (not traced further) |
| `device_loss_summary` | `python/pulsim/losses.py:1557-1563`, `:1975`, `:2151`, `:2251` | handled (`_ideal_entry`, DECISIONS #22) |
| `_diode_state` traces | `python/pulsim/_diode_state.py:365` | handled |
| Discovery template | `python/pulsim/__init__.py:4059` | names `add_diode(name, anode, cathode, g_on, g_off, V_th, V_f)` |

## 2. Examples, tutorials, notebooks, docs

Patterns: `add_diode(` (D), `add_switch(` (S), MOSFET/IGBT helpers with a
built-in PWL diode (M), topology helpers (T), YAML device entries (Y).
**No file in these directories passes `form=` or `mode_selection=`.**

| Directory | D | S | M | T | Y | Representative |
|---|---|---|---|---|---|---|
| `examples/scripts/` | 57 / 34 files | 16 / 13 | 20 / 20 | 0 | — | `run_buck.py:42` (`add_mosfet_with_body_diode`), `:44` (`add_diode(..., 1e3, 1e-9, V_th=0.7)`); `run_three_phase_inverter.py:51`, `:59`; `run_flyback.py:41` |
| `examples/notebooks/` (38 notebooks) | 27 / 17 | 25 / 15 | 13 / 4 | 15 / 5 | — | `05_nonlinear_devices.ipynb`, `14_api_reference.ipynb` (5 T), `vsi_inverter_design.ipynb` (4 T), `boost_pfc_vsi_design.ipynb` (4 T) |
| `examples/pulsim_tutorial.ipynb` | 1 | 2 | 0 | 0 | — | — |
| `examples/*.yaml` | — | — | — | — | 43 / 11 files, none with `form:` | `three_phase_inverter.yaml:33-103` (6 switches, 6 diodes); `phase_shift_full_bridge.yaml:51-137` (12); `three_phase_diode_rectifier.yaml:66-98`; `buck.yaml:21` (`mosfet_with_body_diode`); `flyback.yaml:25` (`mosfet`) |
| `projects/converters/` | 13 / 9 | 0 | 10 / 9 | 1 / 1 | — | `half_bridge/half_bridge_pulsim_validation.py:66-69`; `boost_pfc/boost_pfc_pulsim_validation.py` (T) |
| `projects/inverters/` | 6 / 3 | 10 / 4 | 2 / 1 | 3 / 2 | — | `npc_3phase/npc_3phase_pulsim_validation.py:134-136`; `pfc_vsi_drive/...validation.py` (2 T) |
| `docs/tutorials/` | 4 / 4 | 1 / 1 | 12 / 5 | 0 | — | `02-buck-converter.md:55`, `03-flyback-isolated.md:48`, `06-igbt-boost.md:50`, `08-design-optimisation.md:24-25` (run on `engine="dsed"`, `:86`), `04-3phase-vsi.md` (7) |
| `docs/` other (internals, guides, archive) | 25 / 12 | 14 / 10 | 19 / 6 | 4 / 2 | 1 | `migration-from-v1.md` (13), `internals/layer2-v1-mosfet-igbt.md` (11), `api-reference.md` (8); `internals/layer8-yaml-parser.md:81-85` documents `g_on`/`g_off` as **required** for YAML `diode`/`switch` |
| `python/pulsim/AI_GUIDE.md` | 8 | 1 | 7 | 5 | — | `:125`, `:219`, `:306`, `:494` (`add_diode(..., 1e3, 1e-9)`); signatures `:406-408`; executed by `python/tests/test_ai_guide.py` |
| `python/pulsim/` docstrings | 3 | 3 | 1 | — | — | `sweep.py:30-31`, `discovery.py:400-421`, `subsystem.py:20-21` |
| `scripts/` | 3 / 1 | 10 / 2 | 0 | 0 | — | `bench_dsed_vs_pwl.py` (8), `bench_vs_ngspice.py` (5) |
| `probes/` | 0 | 1 | 0 | 0 | — | — |
| `README.md` | 0 | 0 | 0 | 0 | — | — |

Docs that run PWL devices on `engine='dsed'`, which refuses the branch form
(F4): `docs/losses-and-thermal.md`, `docs/known-limitations.md`,
`docs/sensitivities.md`, `docs/tutorials/08-design-optimisation.md`,
`docs/api-reference.md`, `docs/how-pulsim-works/16-dsed-benchmarks.md`.

`mode_selection` is named only in `KNOWN_LIMITATIONS.md:410`, `:416`,
`CHANGELOG.md`, `benchmarks/manifests/v2_triage.yaml:156`,
`benchmarks/tools/ideal_switch_flip_evidence.py`,
`benchmarks/IDEAL_SWITCH_FLIP_EVIDENCE.md` and `scripts/stubs_overlay.pyi:47`.

**(inferred)** Many examples use `V_th=0.7, V_f=0` as a turn-on threshold,
for example `run_buck.py:44` and the AI_GUIDE. Under `'lcp'`, `V_th > V_f` is a
hysteresis band (complementarity design D1/D8), not a threshold. The numbers
need re-checking, not only the form.

## 3. Tests

### 3a. Python (`python/tests/`, 246 files; 141 build PWL devices or name `mode_selection`)

| Group | Files | Under the flip |
|---|---|---|
| Opt-in, explicit `form=` / `add_ideal_switch` | 13: `test_branch_form_reinit`, `_review`, `_switch`, `_trbdf2`, `test_bridge_ground_truth`, `test_circuit_ir_integration`, `test_circuit_sweep`, `test_circuit_yaml`, `test_engine_conformance`, `test_ideal_dcm_buck`, `test_ideal_loop_release`, `test_ideal_losses`, `test_impulse_losses` | unaffected for the branch cells; the conductance cells in the same files rely on the default |
| Name `mode_selection` | 11: `test_bridge_ground_truth`, `test_dsed_diode_events`, `test_engine_conformance` (7), `test_exact_route`, `test_ideal_loop_release`, `test_mode_lcp` (14), `test_mode_passivity`, `test_mode_pwl` (7), `test_mode_selection_today`, `test_mode_shadow`, `test_probe_horizon` | explicit `'lcp'`/`'shadow'` cells are unaffected. The cells parametrised with `None` as "the default" (`test_probe_horizon.py:46`, `test_bridge_ground_truth.py:114`, `test_ideal_loop_release.py:49`, `test_mode_shadow.py:26`, `test_mode_pwl`, `test_engine_conformance`) silently become `'lcp'` against `'lcp'` |
| Run PWL devices on `engine='dsed'`/`'exact'` | 60 files (heuristic: the file names dsed/exact and builds a PWL device). Largest: `test_switch_state_record` (27 calls), `test_branch_form_switch` (14), `test_mode_lcp` (14), `test_diode_forward_drop` (14), `test_dsed_exact_propagator` (12), `test_engine_conformance` (12), `test_composite_diode_forward_drop` (10) | refused under the branch default until task 8 (F4), except the cells that assert the refusal |

**Tests that pin numbers or behaviour of today's defaults (would change):**

| Test | What it pins | Changes under |
|---|---|---|
| `test_switch_conductance_today.py` | `g_off = 0` conductance switch on an RL load: pwl rings at ±397 V, trbdf2 reaches ~1e9 V ("these tests pin it until then", docstring) | form flip (the branch form re-initialises) |
| `test_switch_conductance_validation.py:62-64` | `add_diode(..., 1e12, 0.0)` builds | form flip: `R_on = 1e-12` is refused (F6) |
| `test_mode_selection_today.py` | dsed settle record (boost 1.01 V / 0.13 V low; `MODE_GUARD=False` refuses) | lcp flip; branch form (dsed refuses) |
| `test_ideal_loop_release.py:193` | `ideal_loop_releases > 0` under the default settle | lcp flip (the LCP releases none, `:194`) |
| `test_newton_stall_and_mask_chatter.py:233` | "diode-mask chatter" message of the settle; Newton device + PWL diode | lcp flip (Newton device refused, F5) |
| `test_trbdf2_stateful_with_diodes.py` (`:39`, `:161`) | event counts ("6 events in 3 cycles") on trbdf2 | lcp flip (trbdf2 refuses until 4.2) |
| `test_circuit_params.py:30`, `test_sensitivity_exact.py:364` | parameter paths `X.g_on` / `S.g_on` | form flip **(inferred)**: the `g_on` path on a branch element, plus dsed refusal and the sensitivity + lcp refusal (F5) |
| `test_mosfet_body_diode_default.py`, `test_composite_diode_forward_drop.py`, `test_mosfet_third_quadrant.py` | body-diode drop `V_F + i/1e3` | if helpers follow the default (D6) |
| `test_smps_topology_factories.py` | helper defaults (buck/boost/flyback) | if helpers follow the default |
| `test_named_lookups.py`, `test_snapshot_resume.py`, `test_dsed_end_to_end.py`, `oracles/lcp_pulsim_bridge.py` | read the unknown layout / branch variable ids | **(inferred)** the branch form adds one unknown per element |
| Settle counters / `max_event_iterations` / chatter named, no `mode_selection`: `test_branch_form_review`, `test_composite_diode_forward_drop`, `test_convergence_failure_typed`, `test_dc_operating_point`, `test_design_evaluation`, `test_diode_forward_drop`, `test_diode_state_reconstruction`, `test_dsed_exact_propagator`, `test_engine_auto`, `test_fixed_step_loop_bookkeeping`, `test_periodic_steady_state`, `test_solver_options`, `test_switch_state_record`, `test_trbdf2_error_control`, `test_trbdf2_robustness` | **(inferred, heuristic grep)** may pin settle-specific counts | lcp flip |

All other Python test files (about 75) build conductance devices through the
default with explicit `g_on`/`g_off` on pwl/trbdf2. D6 row 2 says those agree
within 1e-10 (task 3.4), and DECISIONS #39 shows the same outcome on 109/109
circuits on pwl. They need a re-run, not a rewrite **(inferred)**.

### 3b. C++ (`core/tests/`, 52 files touch switches/diodes)

| Group | Count | Under the flip |
|---|---|---|
| `DevicePool::add_switch/add_diode` directly (no form parameter) | 51 calls / 27 files | unaffected |
| `CircuitBuilder::add_switch/add_diode/add_mosfet*/add_igbt` | 103 calls / 25 files; largest `dsed/test_trbdf2_transient.cpp` (20), `layer4_v1/test_linear_time_assembly.cpp` (11), `sources/test_phase_shift_full_bridge_fn.cpp` (8) | affected only if the C++ `add_switch`/`add_diode` flip too. D6 does not say so (owner decision) |
| Explicit branch form | 1 file: `layer4_v2/test_branch_form_switch.cpp` | unaffected |
| YAML loader | `yaml/test_yaml_loader.cpp:244` (switch with `g_on`), `:314` (`switch_g_on(0)` readback) | YAML default flip **(inferred)**: the branch path stores `1/R_on`, so the readback may still match |
| Settle counters named | `dsed/test_convergence_failure.cpp`, `dsed/test_design_batch.cpp`, `dsed/test_periodic_orbit.cpp`, `layer5_v2/test_integration_boost.cpp`, `layer5_v3/test_substep_correction.cpp` | C++ `SimulationOptions::mode_selection` default flip **(inferred)** |

### 3c. Reference harness

- `benchmarks/tools/ideal_switch_flip_evidence.py` → `benchmarks/IDEAL_SWITCH_FLIP_EVIDENCE.md`:
  109 circuits × 5 configurations × {default, branch, lcp, branch+lcp}.
  Branch (finite `R_on = 1/g_on`): same outcome on 109/109 on pwl;
  trbdf2: 2 better, 106 same, 1 new error (`ll13_active_clamp_forward`);
  dsed: 54 newly refused. `'lcp'`: 2 refused by design (Newton devices), dsed 5
  errors (DECISIONS #39). Cost after DECISIONS #40: lcp/default median
  1.058-1.088× on pwl.
- Goldens and reference status (`goldens.py`, `reference_status.py`,
  `counters.py`, `differential_runner.py`) pin today's numbers through
  `v1_reader`: 9.3 / 9.2 re-baseline them.

## 4. Migration instructions (one line per category)

| Category | Instruction |
|---|---|
| `add_diode(name, a, k, g_on, g_off, ...)` | To keep today's numbers, pass `form='conductance'` (warns after the flip). To adopt the flip, drop `g_on` and give `R_on=1/g_on` (`g_off` kept), or omit both for the ideal element |
| `add_switch(name, a, b, g_on, g_off)` | Same, once `add_switch` accepts `form=` (F3). Until then `add_ideal_switch(name, a, b, R_on=1/g_on, g_off=g_off)` is the branch spelling |
| Any `g_on > 1e9` | Give `R_on=0` (ideal) or `R_on >= 1e-9`. The converted value is refused (F6) |
| `add_ideal_switch` | No change (it becomes an alias of `add_switch`) |
| `add_mosfet*` / `add_igbt` / `*_level1` body and freewheel diodes | Need a `form=` keyword (D6) and must stop hard-coding `1e3`/`1e-9` (`circuit_builder.hpp:450`, `:965`, `:1724`). To keep today's numbers, pass `form='conductance'` |
| Topology helpers | Add `form=` (D6). Replace the `R_on <= 0 → 1e3` fallback (`topology.py:338`, `:358`) with the ideal element, or refuse it. To keep today's numbers, pass `form='conductance'` |
| YAML `diode` / `switch` | To keep today's numbers, add `form: conductance`. `form: branch` with `g_on` means `R_on = 1/g_on`. Update `docs/internals/layer8-yaml-parser.md:81-85` ("required" no longer holds) |
| `benchmarks/tools/v1_reader.py` | Pass `form='conductance'` to keep the goldens, or re-baseline (9.3). The dsed rows wait for task 8 |
| Circuits run on `dsed`/exact, `bdf1`, the C++ AC sweep | Pass `form='conductance'` until task 8, or run on `engine='pwl'`/`'trbdf2'` |
| `simulate()` callers relying on the settle | Pass `mode_selection='legacy'` (warns; refused at 9.3), or accept the re-baselined numbers |
| Tests parametrised with `None` as "the default" | Replace `None` with `'legacy'` so the default-vs-lcp comparison survives the flip |
| `mode_selection='lcp'` in a call | Remove it after 9.1 (it becomes the default). Keep it where a test asserts a refusal |
| Thermal / KPI consumers | Make `device_thermal_summary` report ideal elements, or warn instead of `continue` (`thermal.py:866`, `:908`) |
| Docs and AI_GUIDE | Rewrite the `add_diode(..., 1e3, 1e-9, V_th=0.7)` pattern to the new default, and re-check `V_th`-as-threshold text against D1/D8 |

## 5. Top migration risks

1. **dsed/exact refuses the branch form (F4).** About 60 Python test files, 6
   doc pages and 54 of 109 reference circuits run PWL devices on dsed. Flipping
   before task 8 turns them into refusals.
2. **A literal `'lcp'` default raises in common runs (F5):** diode-free
   circuits, Newton devices, trbdf2 (and so `engine='auto'` without dt),
   `max_event_iterations=0`, `sensitivities=`. The default needs a fallback to
   the settle with a recorded reason, or these refusals must be lifted first
   **(inferred design requirement)**.
3. **No escape hatch exists yet for switches and helpers (F3).** `add_switch`,
   the MOSFET/IGBT helpers and the topology helpers take no `form=`, and two
   C++ helpers hard-code conductance body diodes.
4. **Silent output loss for ideal elements.** `device_thermal_summary` skips
   any element with non-finite `g_on` without a warning (`thermal.py:866`, `:908`).
   `topology.py:338` silently maps `R_on=0` to 1 mΩ today.
5. **Design/code mismatch (F2) and conversion edge cases (F6).** D6 assumes
   default `g_on`/`g_off` that do not exist. Converted `g_on > 1e9` is refused.
   The C++ `CircuitBuilder::add_switch`/`add_diode` flip (103 C++ test calls) is
   not decided in D6. D7 says the AC sweep supports the branch form, but
   `mna_sweep.hpp:85` refuses it; whether the Python `run_ac_sweep` reaches that
   code was not checked.
