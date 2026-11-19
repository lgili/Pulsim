# Decisions — update-number-changing-release

## 1. Branch-form coverage (blockers 1.4–1.8, feat/core-branch-form-coverage)

No default number changes; with today's defaults every conductance-form
result is byte-identical (goldens, stateful, engines, steps, pwl_identity,
trbdf2_identity, and a hash of every PSS / sensitivity / orbit / AC output).

1. **Period map with branch-form devices (1.4).** No fallback to the
   conductance form. The per-mode reduction is reused unchanged: the state
   stays continuous across an event, the algebraic unknowns (the branch
   currents) jump. A branch-form diode's turn-OFF predicate reads its solved
   current `V_f + x_full[var]` (as the transient's census); its saltation
   gradient is the recovery row `Rx.row(var)` of that current. Turn-ON reads
   `v_D − V_th` on the node rows as before. An ideal diode (R_on = 0) uses
   the transient's `BranchStructureChecker` hold rule. Functions changed in
   `dsed/`: `add_diode_predicates` and `period_map_from_run`
   (`core/include/pulsim/dsed/period_integrals.hpp`); nothing else in
   `dsed/`. Python: `periodic_steady_state._Solver._g`, `sensitivity`
   (`_branch_diode_vars`, the recovery rows, `_jump`, `_near_misses`, the
   mode guard passed only when a branch-form diode is present),
   `_period_kpis.spec_for` (a branch-form current is its own unknown).
   Left refused by its existing name: `dependent_states` (a capacitor loop or
   inductor cut-set, e.g. g_off = 0): the transient projects, the period map
   has no Jacobian of the projection (add-branch-form-ideal-switch 8.1/8.2
   stay open). Shortcut kept: `add_diode_predicates` duplicates the
   binding's predicate code.
2. **Tri-state restart rule and split (1.5, 1.8).** `discontinuity_restart`
   and `enable_substep_state_correction` = None resolve in the pwl block of
   the run plan from `pulsim._plan.DEFAULT_DISCONTINUITY_RESTART` /
   `DEFAULT_SUBSTEP_STATE_CORRECTION` (False today). The rule's default is
   not applied under `mmc_arms` (one `engine_route_reason` entry); an explicit
   True with `mmc_arms` stays refused. The split's default is on only where
   the rule resolved on, so `discontinuity_restart=False` alone runs.
   Explicit True / False do what they did. `add-kernel-simulation-object`
   DECISIONS 23 item 7 updated. The rehearsal plugin sets the two `_plan`
   attributes instead of injecting keywords.
3. **`bind_param` on a branch-form switch (1.6).** A parameter may drive
   `g_on` (stored as R_on = 1/g_on, g_on > 0) and `g_off` (>= 0); other
   fields stay refused by name. Parameters stay finite (a document never
   holds an infinity), so the ideal limit R_on = 0 is set with
   `set_value("<dev>.R_on", 0)`; a `set_value` drops the binding it
   overrides. `set_params` is all-or-nothing across the branch-form fields.
4. **AC sweep and bdf1 (1.7).** `run_mna_sweep` stamps branch-form devices
   with the transient's assembler (`kBranchFormCaps` gains `AcSweep`; the
   known-limitations table says yes). At the caller's mask it refuses
   `floating_node` / `ideal_voltage_loop` by the transient's names, and an ON
   ideal diode closing a loop as the dsed cache does. bdf1 keeps refusing
   branch form by name; the message says the engine is retired and names
   `engine='pwl'` / `engine='trbdf2'`. A loss port is named before branch
   form.
5. **Blast radius after 1.4–1.8** (`benchmarks/tools/next_defaults_pytest.py`,
   full `python/tests` under the profile): 615 failed / 27 errors -> 479
   failed / 0 errors (5409 passed). By class, before -> after: new refusal
   by name 96 -> 6, engine route / result layout 101 -> 71, snapshot /
   layout 27 -> 6, other assertion 118 -> 101, warning / raise expectation
   36 -> 28, numeric assertion on an old value 191 -> 193, exception 23 ->
   24, circuit digest (harness artefact) 50 -> 50. The 6 refusals left: one
   bdf1 cell (bdf1 refuses branch form by design; section 1b of the report)
   and five `test_switch_conductance_validation` message patterns (the
   harness turns `g_on` into `R_on = 1/g_on`, so the builder names R_on).
   Section 1b and the blast radius of `benchmarks/NEXT_DEFAULTS_REHEARSAL.md`
   are regenerated from this run (`suite_refusals()` / `pytest_section()`;
   the differential-matrix sections need the cell cache and were not re-run).
6. **Review** (fresh-context, read-only, on `e33e0e71..HEAD`): no
   conductance-form path or default changes; the saltation predicate and
   gradient read the same current in the pre-event mode. Left as noted:
   bdf1 names a loss port before branch form (intended); the C++ saltation
   has no dg/dt term (pre-existing; a time-varying source takes the Python
   map, which has it); `bind_param("<dev>.g_on")` on an ideal device makes it
   non-ideal at bind time (intended: the parameter is the device's g_on).

## 2. The exact route's output grid without dt (`output_rtol`), 2026-11-19

Branch `feat/core-auto-route-exact` (from `feat/core-release-rehearsal` bcadf4cd, merged with
`feat/core-integration` 061e4099). It closes blockers 1.1-1.3 and owner item 2.5.

### Causes, found by trace

- **1.1 `ll06_buck_vcswitch`** (exact engine, no dt, `exact` 1.1e-2):
  - The event trace showed the D1 turn-off right limit was already projected: `I(L1)` went from +4.8e-3 to -4.77e-3 A, and `V(sw)` from 0 to 47.6 V, in zero time.
  - `project_` cuts every mode faster than `100/dt_max`. With the default `dt_max` of 10 us, that removed a real ~100 ns transient.
  - Proof: `enable_event_projection=False` scores 8.9e-12.
- **1.2 `cl_buck_aggressive_pi`** (strict 3.95e-3): no engine defect.
  - Two independent closed-loop references agree within 8.4e-6: trbdf2 at rtol 1e-9 with `control_tick='state'`, and pwl with split at dt 2e-8.
  - The dsed samples agree with both to 1.2e-6.
  - The strict failure came from the 10 us output grid alone.
- **1.3** The 10 us grid between exact samples.

### The rule

Applies only with `simulate(engine='exact', output_rtol=r)` and no `dt`, `store_every` or recorders.

- **Event instants.** Every event instant is a sample, with both limits as before.
- **Step cut (chord bound).** Each exact step is cut so that the chord bound δ²·max|ẍ|/8 ≤ tol_j holds for every state j. The bound uses the mode's closed form:
  - ẍ(τ) = V e^{Λτ} V⁻¹ ẍ(0);
  - two rigorous bounds are computed, and the cancellation-preserving one is solved by bisection in log δ;
  - oscillator (sine) states and PWL source slopes are included.
- **Tolerance.** tol_j = r · max(range of state j so far, 1e-3·its peak, atol).
  - The range is used, not |x|: a ripple on a 24 V level is what a plot reads. `ll07_boost_vcswitch` disagreed at 5.2e-3 while the scale was |x|.
- **Event projection.** With the rule on, the projection removes only the parasitic cluster, 1/(1e-6·resolved scale), instead of everything faster than 100/dt_max.
- **Post-event points.** The geometric post-event points are the existing `exact_fast_samples` plan.
- **Cap.** 4096 cut steps per interval. Hits are counted in `diagnostics['output_grid']['n_output_capped']`.
- **Unbounded steps.** Steps without a tier-1 basis are counted in `n_output_unbounded` and keep the dt_max grid.
- **Warnings.** Both counters raise a RuntimeWarning.
- **Refused by name:** on pwl and trbdf2, with `integrator='bdf2'`, with sensitivities, and on the non-native fallback.

### The exact-zero root (`boost_classic` / `textbook_erickson_boost_ccm` under the rule)

**Symptom.** At `output_rtol=1e-4`, both runs were refused with DiodeModeInconsistentError: Drect ON carrying -8.7e-2 A at 1.06e-3 s.

**Trace.** I added temporary fprintf instrumentation, since removed:
- `locate_root_bracket` hit an exact 0 of Drect's ON current at t = 1.0594845721750604e-3 and returned that instant.
- The OFF mode read the same state as +9.98e-8 V forward. Its bound was 4.6e-14, plus 7.97e-8 of root error.
- The zero-band ladder therefore kept the diode ON.
- The two runs reached the root from different step starts. Their states differed by 5e-13 A, which is rounding.

**Fix.** An exact 0 carries no sign, so it now stays on ta's side, and the returned state is past the crossing, as the function's contract says.
- The fix is behind `zero_is_root=false`, which the exact path passes only under the output rule.
- With it on unconditionally, 19 of the 111 benchmark circuits change their default samples (byte compare). That change is left to the release flip.

### Profile value: 1e-5

Rehearsal `auto+next` against `auto` (trbdf2 today), using the tool's own `judge`:

| `output_rtol` | outcomes (circuits) | better | worse | worse that still agree |
|---|---|---:|---:|---:|
| 1e-4 | 106 / 2 / 1 | 47 | 30 | 29 |
| 1e-5 | 106 / 2 / 1 | 73 | 6 | 5 |

None of the 6 remaining worse cells comes from the output grid. The strict error is the same at 1e-4 and 1e-5:

- **`cascaded_h_bridge_5level`, `mmc_4cell_chain`, `back_to_back_diodes_sine`:**
  - the trbdf2 run is the harness reference, so `auto` scores 0;
  - the exact samples are 1e-13 to 2e-16 off the truth.
- **`ll06_buck_vcswitch` and `ll09_half_bridge_inverter_leg`:**
  - the strict error against the trbdf2 reference is 1.7e-4, below the tolerance (1e-3);
  - against the truth the exact samples are 8.8e-12 and 4.8e-11;
  - the error is the reference's.
- **`thermal/electrothermal_resistor_self_heating`:**
  - disagrees at 1.78e-3 with `exact` 0, the same as `auto:route` without the rule;
  - cause: the trbdf2 reference starts V(vin) at 0 (the uic left limit) and ramps to 5 V by 2e-5 s, and the 1 us samples of the opaque switch_fn poll put that ramp inside the gate window;
  - suggested fix: the reference reads the right limit at t=0, or the gate excludes the t=0 limit;
  - the harness gate was not loosened.

Under `dsed:output_grid` / `dsed+next` (netlist dt), the counts are unchanged at 100/0/0/9/0.

Not named above (integration batch 14n review): one analytic cell is less accurate under the rule while it
still agrees, `analytic:hard_switched_rc` on dsed, 7.6e-14 -> 2.5e-10 against its closed form (rehearsal blocker
list item 1, responsible `dsed:output_grid`). Cause not traced; the rule's projection threshold and the
relocated event roots are the candidates.

**Cost.** Measured on the 100 circuits `auto+next` routes to dsed, rule alone, wall time best of 2, all 100 run, cap hit 0 times, unbounded steps 0:

| `output_rtol` | samples | wall |
|---|---|---|
| off | 2,982,860 | 1.97 s |
| 1e-4 | 3,128,984 (+4.9 %) | 2.41 s |
| 1e-5 | 3,600,621 (+20.7 %) | 3.44 s |

The biggest sample ratios at 1e-4 were lcc ×120, t_type ×23 and steinmetz ×21.

**Rehearsal regeneration.** The cell signature hashes the whole profile, so every cell was stale after the value change.
- Only the configurations that read `output_rtol` were re-run: `auto+next`, `dsed:output_grid`, `dsed+next`.
- All other cells were re-signed from the run on this branch's code.
- This is valid because the rule is off in them: the dsed default and dt runs compare byte-equal, 222/222, base 9640041a against this tip.

### Functions changed in `core/include/pulsim/dsed/`

- `exact_lti.hpp`: new `exact_chord_step`.
- `builder_adapter.hpp`: new `BuilderAdapter::exact_output_step`.
- `event_predictor.hpp`: `locate_root_bracket` gains the parameter `zero_is_root` (default `true`, today's behaviour).
- `scheduler.hpp`, no code moved or reformatted:
  - new `set_output_rule`, `n_output_capped`, `n_output_unbounded`, and `output_step_` with its members (after `enable_event_projection_`);
  - the exact-path call site in `simulate()` (after the `fast_t_` clamp) and the run-start reset;
  - the `project_` threshold under the rule;
  - the two `locate_root_bracket` calls in `locate_event_exact_` pass `output_rtol_ == 0`.

Outside dsed:
- `sim/options.hpp`: the `output_rtol` row.
- `python/dsed_bindings.cpp`: `run_ped_native_builder` gains `output_rtol` and `output_cap`.
- Python files: `__init__.py`, `_plan.py`, `_dsed_dispatch.py`.

`loss_port.hpp` and the snapshot payload are untouched.

### Left open (reviewer items not done)

- `exact_output_step` and its `fits` allocate per step: there is a per-step augmented vector, and an `advance_` for the interval end. This is part of the +75 % wall time at 1e-5.
- Tier-2 (Padé) segments are not bounded. They keep the dt_max grid and are counted as unbounded.
- The tolerance is per state, not per observable. An observable that is a difference of large states gets that absolute error.

## 3. One undriven policy and the per-engine blocks (tasks 2.1–2.3, feat/core-release-options)

Owner decisions delegated to the orchestrator (2026-11-19), implemented as
options whose default is today's; the full record is DECISIONS 26 of
`add-kernel-simulation-object`. The release flips two library defaults, both
data in `benchmarks/tools/next_defaults.py` section `PLAN`:

| item (DECISIONS 23 of add-kernel-simulation-object) | today pwl / trbdf2 / exact | release value | mechanism | profile entry |
|---|---|---|---|---|
| 1 undriven controlled switch | closed + warning / refused by name (`switch_fn`) / closed + warning; `auto` no dt refuses | `'refuse'` on every engine and `auto`, naming each switch | `simulate(undriven=)`, `None` reads `_plan.DEFAULT_UNDRIVEN` | `PLAN["DEFAULT_UNDRIVEN"] = 'refuse'` |
| 7 count checks | pwl (after the mmc combinations) / `max_event_iterations` in the kernel's words, the rest refused by name / all refused by name | one message per count, every engine that reads it, before the mmc combinations | `_plan.UNIFORM_ENGINE_RULES` | `PLAN["UNIFORM_ENGINE_RULES"] = True` |
| 6 `on_impulse` | refused at the event (kernel) / same / after the run, `kind=''`; `integrator='bdf2'` ignores it | bdf2 refuses `on_impulse` by name; the exact refusal names the kind | `_plan.UNIFORM_ENGINE_RULES` | same entry |
| 5 `mode_selection='auto'` | one place now (`_route`) | — (done, no caller-visible change) | none | none |
| 6.2 voltage check | one function on every engine now (the exact engine had it already) | — (done) | none | none |
| 4 `resume_from` clock | snapshot's t unless t_start is set / same / refused | — (one rule, `plan.resolved_from`) | none | none |
| 2, 3, 9 `mmc_arms` / `on_cache`, `progress` / `live_stream`, the exact loss port | per engine | kept per engine (engine-inherent), refused by name elsewhere | option table | none |

C++: one row of `sim::kRunOptionTable` (`undriven`). Blast radius of the
profile on `python/tests` with the two new entries: section 3 below (hand-off).

## 4. Blast radius with `PLAN` (hand-off, feat/core-release-options)

`next_defaults_pytest.py` on full `python/tests`, re-measured on the same
day: a67d3994 (no `PLAN`) 480 failed / 0 errors (recorded 479 in section 1;
the extra one was not identified — every one of the 480 still fails after);
this branch with `DEFAULT_UNDRIVEN='refuse'` and `UNIFORM_ENGINE_RULES=True`
518 failed / 0 errors. No failure went away; 38 are new. By class, before ->
after: new refusal by name 6 -> 31, other assertion 102 -> 112, engine route
/ result layout 71 -> 74, the rest unchanged (numeric 193, digest 50,
warning / raise 28, exception 24, snapshot 6).

The 38 new: 26 rely on an undriven controlled switch (`'refuse'`), 6 pin
today's behaviour on purpose (`test_release_uniform_rules.py` 5,
`test_next_defaults.py` 1), 6 are harness artefacts (the harness's
`simulate` wrapper adds a frame, so the 4 caller-line voltage-check tests
see its file; it routes `auto` before the call, so 2 `undriven=` auto cases
take another engine). Files relying on an undriven switch (not fixed):

| file | tests |
|---|---|
| `test_circuit_scopes.py` | 7 |
| `test_cache_budget.py` | 3 |
| `test_cross_cutting_on_every_engine.py` | 3 |
| `test_undriven_policy.py` (the `None` pins) | 3 |
| `test_dsed_control_ticks.py` (refused as `switch_fn` before `step_observer`) | 2 |
| `test_exact_guard_transient.py` | 2 |
| `test_mosfet_body_diode_default.py` | 2 |
| `test_mode_auto.py` (gated: `switch_fn` before `mode_selection`) | 1 |
| `test_circuit_sweep.py` | 1 |
| `test_controller_state_isolation.py` | 1 |
| `test_named_diagnostics.py` | 1 |

The fix at the release is the same in each: drive the switch
(`switch_fn=`) or pass `undriven='closed'` where the test means today's
fixed-step / exact behaviour.

## 5. R1 — the flip (branch `release/core-next-defaults`, 2026-11-19)

Base: `feat/core-integration` a32b97ed + `feat/core-release-options` f1d2bc23 (merge 2e8c3360).

### Every place a default was flipped (file, symbol: old -> new; the explicit value that restores it)

| layer | file | symbol | old -> new | restores |
|---|---|---|---|---|
| plan | `python/pulsim/_plan.py` | `DEFAULT_DISCONTINUITY_RESTART` | False -> True (pwl; skipped under `mmc_arms` with a route reason) | `discontinuity_restart=False` |
| plan | `python/pulsim/_plan.py` | `DEFAULT_SUBSTEP_STATE_CORRECTION` | False -> True (follows the resolved rule) | `discontinuity_restart=False` / `enable_substep_state_correction=False` |
| plan | `python/pulsim/_plan.py` | `DEFAULT_MODE_SELECTION` (new) | (None = settle) -> 'auto' | `mode_selection='legacy'` |
| plan | `python/pulsim/_plan.py` | `DEFAULT_DEVICE_PATH` (new) | ('legacy') -> 'interface' (pwl, trbdf2; the exact engine keeps its own) | `device_path='legacy'` |
| plan | `python/pulsim/_plan.py` | `DEFAULT_OUTPUT_RTOL` (new) | (None) -> 1e-5 where the rule can act (exact engine, no dt / store_every / recorders / sensitivities, not `integrator='bdf2'`; dropped silently on the non-native fallback) | `output_rtol=0` (new: 0 = off) |
| plan | `python/pulsim/_plan.py` | `DEFAULT_UNDRIVEN` | None (per engine) -> 'refuse' | `undriven='closed'` (pwl / exact; trbdf2 refused before) |
| plan | `python/pulsim/_plan.py` | `UNIFORM_ENGINE_RULES` | False -> True | no keyword (messages / refusal order only; `_plan.UNIFORM_ENGINE_RULES = False`) |
| signature | `python/pulsim/__init__.py` `simulate()` | `control_tick` | 'label' -> 'state' | `control_tick='label'` |
| signature | `python/pulsim/__init__.py` `simulate()` | `device_path` | 'legacy' -> None (resolved after the route) | `device_path='legacy'` |
| route | `python/pulsim/__init__.py` `_route`, `simulate`, `_auto_exact_fallback` | `engine='auto'` without dt | trbdf2 (blockers: ValueError) -> exact when `_require_supported(..., 'exact')` and no keyword the exact engine refuses; trbdf2 otherwise; an `UnsupportedError` from the exact run (a caller `b_extra_fn` is one, refused at the run) re-plans on the variable step (`_AUTO_EXACT_OFF`), the refusal first in the route reasons | explicit `engine='trbdf2'` |
| bindings | `python/bindings.cpp` | `kDefaultSwitchForm` (`add_switch`, `add_diode` `form=`) | 'conductance' -> 'branch' | `form='conductance'` |
| YAML | `core/include/pulsim/yaml/loader.hpp` | `kDefaultSwitchForm`, `LoadOptions::default_switch_form`; `load_yaml_file/string(switch_form=)` (new) | `switch` / `diode` without `form:` conductance -> branch | `form: conductance` per device, or `switch_form='conductance'` per netlist |
| kernel | `core/include/pulsim/sim/defaults.hpp` | `Defaults::discontinuity_restart`, `substep_state_correction`, `control_tick_state`, `device_path` (new) | off / off / false / Legacy -> on / follows / true / Interface | set the `RunOptions` field |
| kernel | `core/include/pulsim/sim/options.hpp` | `RunOptions::control_tick_state` default; `RunOptions::device_path` now `optional` (unset = Interface; only an explicit non-Legacy value is a request, so the exact engine does not refuse a default); `to_simulation_options` restart / split `value_or(Defaults)` | as above | as above |
| kernel | `core/include/pulsim/sim/options.hpp` `kRunOptionTable` | default texts of enable_substep_state_correction, discontinuity_restart, mode_selection, undriven, output_rtol (`Unset::None` -> `Unset::Falsy`: 0 = off is no request), control_tick, device_path | — | — |
| docs | `docs/known-limitations.md` option table (generated by `_run_options.markdown_table`) | — | regenerated | — |
| stubs | `python/pulsim/_pulsim.pyi` | add_switch / add_diode `form` | regenerated | — |

### Could NOT be flipped by a table entry (and why)

- **The two exact-engine behaviours gated on the output rule** (only the parasitic cluster projected at an event; an exact zero of a diode's ON current is not yet a crossing). Section 2 says the second is right by the function's contract and that applying it everywhere moves 19 of 111 circuits. Both are keyed on `output_rtol_ != 0` inside `dsed/scheduler.hpp` (`project_` threshold, the two `locate_root_bracket(..., output_rtol_ == 0)` calls); making them unconditional is a code change in that file, which R1 must not touch beyond a default value, and it would need its own legacy switch (the legacy period). They are therefore on wherever the rule is on — every `engine='auto'` / `engine='exact'` run without dt, i.e. the release route — and off with an explicit dt. Hand-off to the exact-engine track (R3 or its own slice): unconditional, with a keyword for the old behaviour.
- **`mode_selection='auto'` in the kernel**: `sim::RunOptions` has no Auto (the rule needs `_lcp_circuit_refusal` and the circuit census, resolved by `simulate()`); a C++ caller of `sim::Simulation` keeps the settle unless it asks for Lcp.
- **`compute_dc_op(device_path='legacy')`** (`python/pulsim/dc_strategy.py`): a separate public entry point, not in the profile; the reference harness's DC starts go through it, so flipping it would move every DC-started reference cell beyond the rehearsal's measurement. Left for R3 / the device track, together with the hand-stamp deletion.
- **MOSFET / IGBT helpers' own `form=`** (`add_mosfet*`, `add_igbt*`, YAML `mosfet*` / `igbt*`): not in the profile (`next_defaults.py` BUILDER names add_switch / add_diode only); unchanged.
- **`UNIFORM_ENGINE_RULES = False`** is reachable only as a private attribute: it changes refusal order and messages, no number.
- **The TR-BDF2 step grid** (`update-trbdf2-factor-reuse`): its option does not exist on this tree (unchanged from the profile's NOT_APPLIED).

### The legacy profile

`benchmarks/tools/next_defaults.py`: `LEGACY` (per engine), `LEGACY_BUILDER`, `LEGACY_YAML`, `LEGACY_PLAN`, `legacy_route`, `legacy()` (a context manager the harness uses), `check_current()` (empty list = the flip is the profile; `next_defaults_pytest.py` aborts the session otherwise and injects nothing). Measured on this branch's extension with `legacy()` against a32b97ed's defaults (a32b97ed `git archive` + `/Users/lgili/pulsim_wt/so_a32b97ed`): goldens 21/21, stateful 40/40 (the stateful set has 40 cases), engines 35/35, steps 28/28, `pwl_identity` 390/390, `trbdf2_identity` 371/371 (one recorded refusal names the tree's own path: `refrigerator_compressor_cc.yaml`, identical otherwise). The first attempt found two holes, both fixed in the profile: the split's old default when a caller set only `discontinuity_restart=True` (LEGACY now sets `enable_substep_state_correction=False`) and the YAML loader (`switch_form=`).

### Orchestrator decisions on what R1 flagged (2026-11-19; the orchestrator's, recorded here)

1. **Goldens farther from the truth at rounding level: accepted.** `stiff_rlc` (mpmath truth, exact measure 3.15e-9 -> 3.76e-9, 1.19x) and `back_to_back_diodes_sine` (1.92e-16 -> 1.96e-16, 1.02x) are rounding-level moves, not defects. Both goldens are regenerated and stay LISTED in `benchmarks/RELEASE_GOLDEN_DIFF.md` under "Farther from truth, accepted: below 1e-8 / at machine precision" (`probes/golden_diff.py` `ACCEPTED`). The other 36 changed goldens are closer to their truth.
2. **The exact engine's analytic cells under the new output grid: accepted as the documented cost of `output_rtol=1e-5`.** The engine's samples at event instants are unchanged; between them the grid is chosen for a 1e-5 interpolation tolerance. Measured against the closed form (`freeze_analytic.py`, defaults vs the manifest): `hard_switched_rc` 1.071e-13 -> 2.526e-10; `sine_rc` 4.5e-16 -> 6.1e-15, `sine_rl` 5.0e-16 -> 6.1e-15, `buck_ccm` 2.6e-15 -> 2.9e-14, `rc_charge` 1.5e-15 -> 1.4e-14, `gapped_core_rl` 4.4e-16 -> 1.8e-15, `lc_lossless` 5.0e-14 -> 8.0e-14 (and `rc_esr_charge`, `three_limb_zero_seq` <= 1.02x). The OLD ceilings are kept (no exact-engine ceiling loosened; the manifest's `measured` stays the pre-release figure); every cell still passes its ceiling (the exact engine's floor is 1e-6), so no analytic cell needed the legacy pin `output_rtol=0`. Tests in buckets 1-6 whose subject is the engine's exactness at samples may pin `output_rtol=0` (rule (b)); each bucket says which. CHANGELOG migration table: one row.
3. **The `simulate()` call-overhead gate (`test_simulate_overhead.py`) fails on the flipped tree: open release item "R2-cost", ceilings NOT raised in R1.** Measured (CPython 3.13, warm second call, `sys.settrace` opcodes): pwl 3056 > ceiling 2890 (2963 with `mode_selection='legacy'`, 3054 with `device_path='legacy'`, 3057 with `control_tick='label'`, 3061 with `undriven='closed'`); the no-`dt` call now routes to the exact engine: 5456 > ceiling 3760 (the ceiling was measured on trbdf2). Handed to the agent working on the exact engine's output-rule cost.
