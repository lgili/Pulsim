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
| 4 `resume_from` clock | snapshot's t unless t_start is set / same / own absolute clock (refused before integration batch 14o) | — (one rule, `plan.resolved_from`) | none | none |
| 2, 3, 9 `mmc_arms` / `on_cache`, `progress` / `live_stream`, the exact loss port | per engine | kept per engine (engine-inherent), refused by name elsewhere | option table | none |

C++: one row of `sim::kRunOptionTable` (`undriven`). Blast radius of the
profile on `python/tests` with the two new entries: section 4 below (hand-off).

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

### R1 hand-off (2026-11-19, end of R1's budget)

**Committed on `release/core-next-defaults`** (all pushed): the merge 2e8c3360; the flip e9575e1e; harness under the legacy profile 542d3024; bucket 0 a3c05be3 (`test_next_defaults`), aa636e4c (stubs), 8fdcf9a2 (`simulate()` stays plan -> table -> finalize: the auto -> exact fallback decides in `_auto_exact_fallback`, which re-raises any other refusal; the route reads no `b_extra_fn`; exact result members in `scripts/stubs_overlay.pyi`), 9fafe25d (`ideal_switch_flip_evidence.py` keeps its five configurations); D d2a15c91 (`R2_BUCKETS.md`, goldens, `goldens/pwl_legacy`); 9b2d6ec6 (golden diff), f25e042e (analytic ceilings), 469d9bd3 (tasks.md), 7c4cc06d (orchestrator decisions, CHANGELOG row); and the hand-off commit (golden-diff "accepted" heading, `probes/r1_resolved.txt`, `test_legacy_goldens_hold_under_the_legacy_profile`).

**Runs on the flipped tree (what was actually run):** full pytest 523 failed / 5512 passed / 110 skipped / 3 xfailed (1291 s, tree = 542d3024 content); full ctest 1061/1061 passed (8.7 s; 1083 `TEST_CASE` literals in `core/tests`, the 22 not registered were not checked). Proof B(i) RUN, result in "The legacy profile" above (identical; one path-text artefact). `goldens.py --legacy` 52/52 against `goldens/pwl_legacy`; `goldens.py` 52/52 against the regenerated `goldens/pwl` (before the two were put back). `test_benchmark_harness.py -k "legacy_goldens or test_golden"`: 53 passed, 2 failed (`test_golden[stiff_rlc]`, `[back_to_back_diodes_sine]`: their old golden is on file, see below). ruff (python/, benchmarks/tools, probes), stubs `--check` rc 0, `openspec validate --all --strict` 79/79 at the time of the flip.

**NOT done (a successor does these):**
1. Regenerate the two accepted goldens (orchestrator decision 1): `goldens.py --update-goldens` (it rewrites all 52; only those two should then differ in git), then `probes/golden_diff.py` (it asserts an ACCEPTED golden is regenerated and lists them under "Farther from truth, accepted"); edit the two `NOT DONE` lines of `probes/r1_resolved.txt`; commit.
2. Perf-counter re-freeze: `benchmarks/tools/counters.py` (check) then `--freeze` (rewrites `benchmarks/manifests/perf_counters.yaml`); record each increase in this file as a cost of the new defaults. The check was started once and ran 88 min without finishing (output buffered, log empty; stopped by PID): find which circuit x engine hangs or is slow under the defaults FIRST (suspect: a no-dt / exact-engine count, cf. the ten-bridges exact test that took ~10 min in the pytest run, its time in `port_scale_full` / `node_pair_voltage_scale` of the output rule). The 9 `test_counter_gate*` failures wait on this.
3. Proof B(ii): `benchmarks/tools/reference_status.py --json OUT.json` (writes `REFERENCE_STATUS.md`; computes its reference under `next_defaults.legacy()`; configs `pwl, trbdf2, dsed, auto` + `@legacy` columns), then `probes/rehearsal_match.py OUT.json` (each cell vs the rehearsal's `+next` / pre-release column). Then migrate the triage freezes in `benchmarks/manifests/v2_triage.yaml`: the old per-config `known:` / `known_err:` keys -> the `@legacy` columns, new freezes for pwl / trbdf2 / dsed / auto.
4. Proof B(iii): `benchmarks/tools/exact_scoreboard.py run --force` then `render` (raw records in `.scratch/exact_scoreboard`; reference under `legacy()`); count pwl and dsed cells with `exact` below 1e-3 / 1e-6 / 1e-9 from `benchmarks/exact_scoreboard.json` (targets: pwl 76/41/10, dsed 81/77/70 of 81).
5. Analytic: the manifest is tightened (f25e042e); after the counters, re-run `test_benchmark_harness.py` whole (its `test_analytic*` failures should be gone with the new ceilings; not re-run after f25e042e).
6. The R1 reviewer (fresh-context, read-only) on the flip commits and the bucket map; merging the six R2 buckets (`release/r2-bucket-1..6`, made from d2a15c91: they did not see 9b2d6ec6 onward, which touched only benchmark data, tools and docs, plus `test_benchmark_harness.py`, which is in no bucket).
7. R2-cost (decision 3): `test_simulate_overhead.py` stays red until the exact engine's output-rule cost work lands; ceilings not to be raised.

**Tools (all run with `PYTHONPATH=<tree>/python:<tree>/benchmarks/tools:<tree>/benchmarks`, framework python3.13):**
- Bucket map: `probes/r2_buckets.py PYTEST_JSON CTEST_FAILED_TXT --resolved probes/r1_resolved.txt [--header ...]`; PYTEST_JSON is what `pytest python/tests -p next_defaults_pytest` writes to `.scratch/rehearsal/pytest_profile.json` (the plugin is now the no-op check: it aborts unless `next_defaults.check_current() == []`). The run used here is kept at `.scratch/logs/pytest_flipped.json` (scratch, not committed). It asserts buckets + resolved = the run's failed count.
- Golden diff: `probes/golden_diff.py` -> `benchmarks/RELEASE_GOLDEN_DIFF.md` (old = `goldens/pwl_legacy`, new = the current run; truth: closed form / mpmath / trbdf2 rtol 1e-10 marked not independent); exit 1 on a non-accepted farther golden.
- Freezes: `goldens.py [--legacy] [--update-goldens]`, `freeze_analytic.py [--write]` (careful: `--write` rewrites comments away, re-adds a looser floor for two hand-set ceilings and writes `error: ran although the case lists it as refused` for six exact-engine cells — f25e042e applied only the tighter numbers by hand), `counters.py [--freeze]`.
- Legacy profile check (B(i)): `.scratch/bc/bc.sh compare TREE OUTDIR legacy` against the a32b97ed baseline in `.scratch/bcbase` (scratch; delete it with `.scratch/base_a32b97ed` when no longer needed: 151 MB + 81 MB). `next_defaults.check_current()` = the layers agree.

**Only in my head until now:** `output_rtol=0` (not None) is the legacy value: None means "the default" (1e-5); the orchestrator's decision 2 wording "pin `output_rtol=None`" should read `output_rtol=0`. `build/python` in the worktree is a symlink to `python/` (tracebacks say `build/python/pulsim/...`: same files). The pytest plugin's class label "circuit digest (IR / YAML path not patched: harness artefact)" was renamed "circuit digest (branch-form circuit)": since the YAML loader flipped, those digests change for real. The exact-engine dispatcher drops the DEFAULT output_rtol with `integrator='bdf2'` and on the non-native fallback, and refuses only an explicit value (`output_rtol_explicit`). `origin/feat/core-integration` was a32b97ed at every check until the last one, which found 27c00520: NOT merged into this branch (the successor merges it with the six buckets).
## 6. The output rule's cost (`output_rtol`), 2026-11-19 — diagnosis

Branch `fix/core-exact-output-rule-cost` (from `feat/core-integration` a32b97ed). Trigger:
`test_exact_route.py::test_ten_paralleled_bridges_agree_with_pwl[None]` ran > 8 min with the
release profile's `output_rtol=1e-5`.

### Measured (extension of a32b97ed; shared Mac, 3 other agents building, single runs)

Paralleled diode bridges (`_paralleled_bridges(n)`; states 3n inductors + bus + 2 oscillator):

| banks | t_end | `output_rtol` | wall | samples | events | capped | unbounded |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | 20 ms | None / 1e-4 / 1e-5 | 0.27 / 0.19 / 0.20 s | 20155 / 20460 / 21889 | 14 | 0 | 0 |
| 2 | 20 ms | None / 1e-4 / 1e-5 | 0.78 / 0.77 / 1.10 s | 20315 / 22481 / 34393 | 31 | 0 | 0 |
| 5 | 20 ms | None / 1e-4 / 1e-5 | 6.1 / 8.8 / 15.9 s | 20591 / 43234 / 99925 | 65 | 0 | 0 |
| 10 | 20 ms | None | 31.3 s | 20909 | 111 | | |
| 10 | 2 ms | None / 1e-4 / 1e-5 | 5.8 / 68.5 / 202.6 s | 2227 / 83523 / 261058 | 35 | 0 | 161 |

- The cost is the SAMPLE COUNT: 10 banks at 1e-5 write 117x the samples of `None` (2 ms).
  The full 20 ms run extrapolates to > 2.5 million samples, ~30 min.
- The cost per sample does not grow with the rule (10 banks: 2.6 ms/sample `None`, 0.78 ms/sample
  under the rule — cut steps are shorter, so the event scan has fewer points).
- Samples grow with the number of banks; events grow only linearly.

### Profile (`sample` on the 10-bank run, 10 s, 8483 samples in the run)

- 86 % self time in `PEDSimulator::simulate` (inlined): the hot loop is a strided row dot
  product plus a constant, i.e. the diode predicates `row·x + c` of the event scan (60 diodes).
- 10 % `node_pair_voltage` / `node_pair_voltage_scale` / `node_pair_voltage_forced` (the same
  predicates), 1.5 % `build_time_varying_b_extra_`.
- The chord bound itself (`exact_chord_step`) and its allocations are not visible at this size:
  every extra sample pays the whole event scan.

### Cause

- Inside one inter-event interval the rule's step is CONSTANT (5 banks, 20 ms: e.g.
  1.23e-8 s for 0.3 ms, 24023 samples, while the trajectory is a slow sinusoid). A real
  curvature would give a varying step.
- The bound is `δ²/8 · Σ_k |V_jk z̈_k|` with `z̈_k = λ_k(λ_k z_k + w_k)`. A parasitic mode (the
  1 Mohm bus tie, the 1 nS OFF diodes in series with 0.2 mH: |λ| up to ~5e12 /s) is projected at
  each event, so its amplitude is rounding (~u·κ·|x|), but `λ²·noise` is large and constant:
  it sets δ for the whole interval. The bound never lets a mode's contribution saturate at
  its amplitude.
- Reference set (100 circuits `auto` routes to the exact engine, best of 5, a32b97ed): rule
  off 2.00 s, on 3.63 s (+82 %), samples 2,982,860 -> 3,600,621. Worst ratios: lcc 101x
  (41 -> 15156 samples), transformer_leakage 57x, steinmetz 51x, high_freq_gan_buck 37x,
  diode_rectifier 25x. These circuits' 10 us grid is far too coarse for 1e-5 (lcc: a sine at
  1e-5 needs ~700 samples per period), so most of their samples are needed; their extra time
  per sample is the rule's own per-step work (an extra `advance_` per step for the range,
  allocations, O(n²) `hypot` per bisection probe).

### The fix

1. **The bound (b): each mode saturates at its amplitude.** Between events, in modal coordinates
   z = V⁻¹x, a mode is z_k(τ) = α_k + β_k τ + γ_k e^{λ_k τ} with γ_k = z̈_k(0)/λ_k² (λ_k = 0: z_k is
   quadratic, z̈_k = (V⁻¹c1)_k; a piecewise-linear source's ramp keeps this form). The chord error
   is linear in the trajectory and zero on its linear part, so e_j = Σ_k Re(V_jk e_k), and per mode
   - |e_k| ≤ δ²/8 · |z̈_k| g_k, g_k = e^{max(0, Re λ_k)·δ} (the chord's Peano kernel is ≥ 0 and
     integrates to ≤ δ²/8, so this holds for a complex term);
   - |e_k| ≤ 2|γ_k| = (2/|λ_k|²)|z̈_k| when Re λ_k ≤ 0, λ_k ≠ 0 (the term and its chord both stay
     within |γ_k|). Growing and zero modes have no such bound (s_k = ∞).
   With F = the modes whose saturation level s_k = 2/|λ_k|² is below δ²/8 (a prefix of the modes
   ordered by s_k) and S the rest, m_jk = |V_jk||z̈_k|g_k:

       |e_j| ≤ Σ_{k∈F} m_jk s_k + δ²/8 · min(Σ_{k∈S} m_jk, |Σ_{k∈S} V_jk z̈_k| + δ Σ_{k∈S} m_jk|λ_k|)

   With F empty this is the a32b97ed bound (the triangle-inequality bound and the one that keeps
   the cancellation between modes at rest). Growing modes (always in S) are added per probe of δ
   with g_k(δ), as before (an overflowing g fails the probe instead of producing NaN). Complex pairs: each complex term is bounded by modulus; repeated but
   diagonalisable eigenvalues: V is a basis, nothing changes; defective modes: no tier-1 basis, the
   step is `n_output_unbounded` and keeps the dt_max grid, as before. Time-varying sources: sine
   sources are oscillator states of the augmented system (modes with Re λ = 0, s_k = 2/ω²);
   piecewise-linear sources enter as c0 + c1·τ over the step (above). The bisection keeps `lo`
   fitting, so any returned δ satisfies the bound whatever the split does to monotonicity.
   After an event the fast modes ARE alive (amplitude ≫ tol): their term s_k m_jk exceeds the
   tolerance, the curvature term rules, and the step stays short; the geometric post-event points
   (`exact_fast_samples`) are unchanged.
2. **Work per mode, not per step (a).** |V_jk| (n² `hypot`), the order of the modes by s_k and
   the sorted s_k are computed once per mode, on the mode's `ExactLTI` (the per-mode cache that
   already holds V, V⁻¹, λ; built on the rule's first call in that mode, so a run without the rule
   holds nothing more). The adapter resets `entry.exact = ExactLTI{}` wherever a mode's matrix is
   invalidated, which drops these too. The dsed mode cache has no byte budget (an unbounded
   `unordered_map`); the new member is one n×n matrix per mode visited under the rule.
   Per step: z = V⁻¹x by two gemv, then four n×(n+1) running sums over the sorted modes (O(n²));
   a probe of δ is O(n) (binary search for the split, one pass over components). No heap
   allocation in the step: `ChordWork` (owned by the scheduler) holds the augmented state, the
   source vectors and the sums. The interval-end `advance_` that widens the state range is now
   done only when the bound cuts the step (a wider range only lengthens a step, so an uncut step's
   decision is the same).
3. **A per-run budget (c).** At most 2e6 cut steps per run (`pulsim._dsed_dispatch.OUTPUT_BUDGET`,
   passed to the native runner as `output_budget`, like `output_cap`). Past it the rule stops
   cutting (the dt_max grid), `diagnostics['output_grid']['output_budget_t']` is the time it ran
   out (None otherwise) and one RuntimeWarning names it.
4. (d) — the tolerance per observable — not needed.

### Results (extension built from this branch; the machine at load 7–175 from other agents)

Paralleled bridges, 20 ms, conductance-form diodes (the test as written):

| banks | `output_rtol` | samples a32b97ed -> now | wall a32b97ed -> now | `None` |
|---|---|---|---|---|
| 1 | 1e-5 | 21889 -> 20804 | 0.20 -> 0.17 s | 20155, 0.27 s |
| 2 | 1e-5 | 34393 -> 20962 | 1.10 -> 0.63 s | 20315, 0.59 s |
| 5 | 1e-5 | 99925 -> 21337 | 15.9 -> 4.8 s | 20591, 5.7 s |
| 10 | 1e-5 | > 2.5e6 (est.; 261058 in 2 ms) -> 21857 | > 8 min -> 27 s | 20909, 31 s |
| 10 | 1e-4 | (83523 in 2 ms) -> 21113 | -> 29 s | |

- The ten-bridge test under the rule now costs about what it costs without it (< 3x target met).
- Under the WHOLE release profile (branch-form diodes, `mode_selection='auto'`, `control_tick='state'`)
  the ten-bridge run is refused at t = 19.969 ms: `DiodeModeInconsistentError`, Db8b OFF while
  forward-biased by 0.73 V. NOT caused by this change: it is the same with `output_rtol` = 1e-4,
  3e-5, 2e-5, 5e-6, with `mode_selection='lcp'`, and with `output_rtol=1e30` (the rule on, no step
  ever cut: only the rule's event semantics — the parasitic-only projection and the exact-zero
  root — differ from `None`, which runs). 5, 8 and 9 banks run under the same profile. Owner: the
  exact engine's event decisions (`zero_band_on_` / settle; not touched here). Before this change
  the same run would have reached 19.97 ms after ~30 min.
- Reference set (100 circuits `auto+next` routes to the exact engine): the samples do not move
  (3,600,621 -> 3,600,044 at 1e-5; `None` 2,982,860): no circuit there had the parasitic-noise
  defect. Their extra samples are needed by the tolerance (lcc 41 -> 15156: a 1e-5 chord on a sine
  needs ~700 samples per period), so the time ratio of those circuits stays far above 5x and the
  +25 % target is not reachable without a looser tolerance or a per-observable one; the rule's own
  per-step work is now O(n²) without allocation. Wall time, paired in one run (best of 5): a32b97ed
  2.00 -> 3.63 s (1.82x, load ~8); this branch 9.86 -> 14.41 s (1.46x, load ~175: not comparable in
  absolute terms). Worst five ratios before: lcc 101x, transformer_leakage 57x, steinmetz 51x,
  high_freq_gan_buck 37x, diode_rectifier 25x; after: lcc 122x, transformer_leakage 43x, steinmetz
  39x, synchronous_flyback 34x, t_type_3level 30x (same samples; ratio noise from the load).
- `simulate()` overhead (bytecodes, warm call, the overhead test's buck, `engine='exact'`, no dt):
  4401 without the rule, 4502 with it (a32b97ed: 4401 / 4483). The rule's per-call Python cost is
  ~100 bytecodes; its C++ set-up is lazy (per-mode data on the first cut in that mode). The release
  agent's 5456 on an `engine='auto'` call is the route: `pulsim.exact_route` alone executes 4745.

### Invariants

- `output_rtol=None`, untouched extension of a32b97ed as the baseline, against this branch:
  dsed bit-identity 36/36; exact engine with and without `dt` on every benchmark circuit
  222/222 byte-identical (`.scratch/ar/dsedid.py`, the rule's author's compare); goldens ok;
  `pwl_identity` 390/390; `trbdf2_identity` 371/371. By construction: `output_step_` runs only under
  `output_rtol_ > 0`, and the per-mode members are filled only by `exact_chord_step`.
- Rule ON (1e-5 and 1e-4, the 100 dsed-routed circuits): every event instant is still a sample,
  samples in time order, the event lists the same length; event times equal to ≤ 2.7e-16 s. The
  states AT the event instants agree to 1e-12 (of max(scale, 1)) on 94 circuits; six transformer /
  coupled-inductor circuits differ up to 6.7e-8 (flyback), 8.4e-9 (half_bridge_transformer_dc),
  3.8e-9 (forward), 1.3e-10 (active_clamp_forward), 2.7e-11 (cuk), 1.5e-12 (sepic). That is the
  grid sensitivity of these circuits' event states, not of this change: a32b97ed's own 1e-5 and
  1e-4 runs differ from each other by 5.4e-8, 2.1e-8, 3.3e-9, 1.7e-10, 3.3e-11, 4.2e-12 on the same
  circuits (and on five more above 1e-12). The 1e-12 bar of the brief is not met on them.

### Functions changed in `core/include/pulsim/dsed/`

- `exact_lti.hpp`: `ExactLTI` (three mutable per-mode members), new `ChordWork`, `exact_chord_step`
  (body and a `ChordWork&` parameter).
- `builder_adapter.hpp`: `exact_output_step` (a `ChordWork&` parameter, no locals allocated).
- `scheduler.hpp`: `set_output_rule` (a `budget` parameter, default 2e6), new `output_budget_t()`;
  the run-start reset (two lines after the existing output reset); the members after
  `n_output_capped_` (`output_budget_`, `output_total_`, `output_budget_t_`, `output_work_`);
  `output_step_`. Nothing moved or reformatted.
- Outside: `python/dsed_bindings.cpp` (`output_budget` argument, `output_budget_t` key),
  `_dsed_dispatch.py` (`OUTPUT_BUDGET`, the key, one warning), the `simulate()` docstring, stubs.

### Verification

- Rehearsal (`auto+next`, `dsed:output_grid`, `dsed+next`, `--jobs 1`, 136 units) re-run on this
  branch against the old cell cache: outcomes unchanged unit by unit (`auto+next` 133 agree /
  2 disagree / 1 error; both dsed configurations 122 agree / 14 refused, as before); `auto+next`
  vs `auto` still 73 better / 6 worse; no cell worse than 2x against the old `auto+next` cell;
  `auto+next` samples 3,794,590 -> 3,793,932.
- dsed C++ tests on Eigen 5: 139 cases / 41211 assertions pass, `[output_rule]` included (330).
- Full pytest (`python/tests`, one worker): 5971 passed, 110 skipped, 8 xfailed. After the
  growing-mode fix's extension was installed: `test_exact_output_grid.py` and both ten-bridge
  parametrisations, 12 passed.
- `ruff check python/` clean; stubs `--check` rc 0. Full ctest not run on this tip.

### Reviewer (one fresh-context pass)

- Fixed: a growing mode (Re λ > 0) used a fixed envelope that could overflow or over-cut; it now
  enters each bisection probe with exp(Re λ·d), and a non-finite probe does not fit (43a1f7c3).
- Left as noted: the per-mode cache in `ExactLTI` is `mutable` and not thread-safe (same as the
  propagator caches before it); a step floored at 64ε still counts as a cut toward the budget.

## 7. R2 merge — integrator hand-off (2026-11-19, end of the integrator's budget)

**Merged and pushed on `release/core-next-defaults`:** the six buckets (`--no-ff`, 2758dc7b..b437d7a3; no textual conflict, no bucket strayed beyond the two accepted out-of-bucket files); bucket-0 items + `FINDINGS.md` (3210315e); `feat/core-integration` 27c00520 (65f5de3b); `fix/core-exact-output-rule-cost` 1e2b90ff (3ba79d2e); stubs (967c8bfe); the PR line's tests under the release defaults (b1ed9c74); this hand-off.

**Merge resolutions (the only library edits besides the bucket-0 items):** `kRunOptionTable` and the docs option table keep the release defaults (52 rows: `undriven` 'refuse', `output_rtol` "None (1e-5 without dt; 0 = off)"), with the PR's `resume_from` row on every engine; `simulate()` keeps the auto -> exact route and its fallback, and `resume_from` left the exact engine's refused list (the PR's row); `_dsed_dispatch.run_dsed_from_builder` takes both parameter sets; the scheduler resets the output budget before the resume branch; the native run takes `output_budget` beside resume / ticks / loop feed; `exact_scoreboard.py` runs `reference_status`'s cells (with the `@legacy` columns) plus `trbdf2+esdirk43` (no `+next` cells: the profile is the default here). **One flip bug found by the merge and fixed:** `simulate(recorders=...)` re-called itself without `output_rtol`, so the release default 1e-5 ran under recorders, against the documented "recorders keep today's samples"; it now passes `output_rtol=0` (b1ed9c74). The evidence files (`EXACT_SCOREBOARD.md` / json, `IDEAL_SWITCH_FLIP_EVIDENCE.md`, `ACCURACY_SUMMARY.md`) are the PR line's PRE-flip versions (the release branch had none of its own): regenerate them (step 5).

**Suite on the merged tree (what was run):** extension built from the merged C++ (identical C++ at 967c8bfe and b1ed9c74; saved in `/Users/lgili/pulsim_wt/so_release_967c8bfe/`). Full pytest at 967c8bfe ("m1", 1759 s): 36 failed / 6018 passed / 110 skipped / 67 xfailed, no XPASS. The 36 = 13 R1 leftovers (`test_golden[stiff_rlc]`, `[back_to_back_diodes_sine]`, ten `test_counter_gate[...]`, `test_counter_gate_names_an_extra_factorisation_per_step`) + 23 new, all in the PR line's files or the merge: `test_ai_guide` 2 (size 33039 > 32768; quick reference without `switch_form`), `test_exact_auto_resume` 5, `test_exact_resume` 9, `test_exact_output_rule_combinations` 2, `test_exact_recorder_loop` 3, `test_recorders` 1 (the recorder flip bug), `test_exact_route::test_ten_paralleled_bridges_agree_with_pwl` 2 (= F-X-1). Fixed in b1ed9c74 (rule (b): the resume-continuity and 'auto'-integrator tests compare on the old grid, `output_rtol=0`; F-X-1 parked strict) and re-run per file: 145 passed, 2 xfailed, 0 failed. The full pytest was NOT re-run after b1ed9c74. The call-overhead gate does not fail: it is parked (F-4-6). F-6-7's test (`..._first_mode_matches_a_50_digit_exponential`, bound 1e-8) now passes with the cost fix merged.
**C++:** NOT run on the merged tree. The ctest build was stopped twice by PID (machine load 50–57 from other agents; 42 of 165 objects rebuilt in 36 min). Resume: `zsh .scratch/build.sh ctest`, then `cd .scratch/build-ctest && nice -n 15 ctest -j2`.

**FINDINGS.md:** 38 findings (35 of the buckets + F-X-1 ten bridges refused, F-X-2 auto routing cost, F-X-3 the output rule does not finish `ladder500_rc` on dsed — counters.py's 88-minute case, still there with the cost fix: > 29 min, `sample` in `exact_chord_step` / `ChordWork` / `ExactLTI`, stopped by PID). By severity 23 W / 7 R / 8 C; by group A 6, B 9, C 7, D 2, E 2, F 8, G 2, H 2. Generated from `.scratch/r2m_pytest_m1.xml` (59 parked tests); F-X-1's park arrived after that run, so the index lists it as "none parked" until the generator is re-run on a newer full run: `PYTHONPATH=python python3 openspec/changes/update-number-changing-release/probes/findings_index.py <junit.xml>`.

**NOT done (steps 4–6 of the integrator's brief), how to run each** (`PYTHONPATH=<tree>/python:<tree>/benchmarks/tools:<tree>/benchmarks`, framework python3.13, `nice -n 15`):
1. Goldens: `benchmarks/tools/goldens.py --update-goldens`, then `probes/golden_diff.py`; edit the two NOT DONE lines of `probes/r1_resolved.txt`.
2. Perf counters: `counters.py` cannot finish until F-X-3 is fixed (`ladder500_rc` dsed). Either wait for the fix or freeze the other cells and record `ladder500_rc` / dsed as the open cost; then `counters.py --freeze`, every rise recorded here.
3. Proof B(ii): `reference_status.py --json OUT.json`, then `probes/rehearsal_match.py OUT.json` (targets pwl 91/17/1/0/0, trbdf2 106/2/1/0/0, exact 100/0/0/9/0, auto 106/2/1); then the triage freezes in `manifests/v2_triage.yaml`.
4. Proof B(iii): `exact_scoreboard.py run --force` then `render` (targets pwl 76/41/10, exact 81/77/70 of 81 below 1e-3/1e-6/1e-9).
5. `test_benchmark_harness.py` whole (after 1–2).
6. Proof B(i) against 27c00520: the baseline tree is extracted at `.scratch/base_27c00520` (git archive + `so_27c00520`), not yet recorded: `.scratch/bc/bc.sh record .scratch/base_27c00520 .scratch/bcbase27`, then `.scratch/bc/bc.sh compare /Users/lgili/pulsim_wt/p3-simulation .scratch/bcbase27 legacy`.
7. Step 5 evidence regeneration (`REFERENCE_STATUS.md`, `EXACT_SCOREBOARD.md` + json, flip evidence, `ACCURACY_SUMMARY.md --check`) — after 3–4.
8. Verification not run: `openspec validate --all --strict`, device reference `--check`, `mkdocs build --strict`, ctest. Run: ruff (clean at b1ed9c74), stubs `--check` (clean after 967c8bfe).
9. Clean-up: delete `.scratch/bcbase`, `.scratch/base_a32b97ed`, `.scratch/base_27c00520` (and `.scratch/bcbase27` once used).

## 8. R2 merge, second integrator (2026-11-20, running note — kept current after every step)

**Merged:** `feat/core-integration` b9e46778 (fad54e51; conflicts: `dsed_bindings.cpp` takes the PR line's `output_budget` parameter order, `test_permeance_magnetic.py` keeps the release's restart-aware energy check plus the PR's 4.1 / 4.4 tests, stubs = the PR's, `--check` clean after the build); `release/h-call-overhead` 26d059b0 (1b3f7ecc, no conflict; overhead gate 3/3 pass, no ceiling raised).
**Suite on the merged tree (1b3f7ecc, extension `/Users/lgili/pulsim_wt/so_release_1b3f7ecc/`):** ctest 1071/1071 (at fad54e51, same C++); full pytest 11 failed / 6123 passed / 110 skipped / 67 xfailed, no XPASS (`test_counter_gate[ladder500_rc]` deselected: F-X-3). The 11 = the two accepted goldens + nine counter gates (re-freeze pending). One new failure fixed: `test_ai_guide` size (the PR's passives line; four trap-table phrasings shortened).
**FINDINGS.md:** 36 open + 2 closed (F-4-6, F-X-2 by group H); `probes/findings_index.py` has a `FIXED` table.
**Counters:** `counters.py` `SKIP` names `ladder500_rc` / dsed (F-X-3), written `skipped:` into the freeze file; its gate case is a strict xfail citing F-X-3.
**Step 3 (run 2026-11-20 05:00-05:17, chain `.scratch/r2m2_chain.sh`, tree eac80cbf + the extension above):**
- **Proof B(ii)** `reference_status.py --json` + `probes/rehearsal_match.py`: the DEFAULT strict counts equal the rehearsal's full profile: pwl 91/17/1/0/0, trbdf2 106/2/1/0/0, dsed 100/0/0/9/0, auto 106/2/1/0/0 (@legacy: pwl 81/27/1, pwl+restart 89/19/1, pwl+split 91/17/1, trbdf2 106/2/1, dsed 100/0/0/9). 1197 cells compared: 0 differ in kind, 12 differ in the printed number: `ll07_boost_vcswitch` trbdf2 1.17e-4 -> 2.95e-4 and trbdf2@legacy 1.17e-4 -> 2.94e-4 (agree both; the PR line's trbdf2 changes since the rehearsal, the same move in both columns = not a default); `ll07_boost_vcswitch` pwl+restart@legacy 0.0997 -> 0.111 and `ll12_pfc_boost_continuous` pwl+restart@legacy 0.0806 -> 0.114 (disagree both, legacy-only columns); `ll09_half_bridge_inverter_leg` dsed and dsed@legacy 1.67e-4 -> 2.31e-4 (agree; same in both columns: the PR line's exact engine, not the flip); auto `cuk_converter` 4.32e-6 -> 4.34e-6, `active_clamp_forward` 3.66e-6 -> 4.82e-6, `flyback_converter` 6.69e-6 -> 7.0e-6, `forward_converter` 6.13e-6 -> 6.62e-6, `synchronous_flyback` 6.2e-6 -> 6.27e-6 (agree; the exact route's output grid after the cost fix 1e2b90ff); `analytic:hard_switched_rc` dsed 2.53e-10 -> 7.76e-14 (closer to the closed form: the cost fix's per-mode chord bound). The triage freezes in `manifests/v2_triage.yaml` were NOT migrated (reference_status prints 72 `UNEXPECTED` lines against the old per-config keys) — leftover.
- **Proof B(iii)** exact scoreboard at the defaults: pwl 76/41/10, dsed 81/77/70 of 81 below 1e-3/1e-6/1e-9 = the targets; trbdf2 80/30/9, auto 81/79/73, trbdf2+esdirk43 81/65/13.
- **Proof B(i)** byte compares under `next_defaults.legacy()` against b9e46778 (`git archive` + `so_b9e46778`): goldens 22/22, stateful 41/41, engines 36/36, steps 29/29, `pwl_identity` 390/390, `trbdf2_identity` 370/371 — the one is `examples/refrigerator_compressor_cc.yaml`, a v1-netlist refusal whose text names the tree's own path (checked: same message, path differs; R1 saw the same artefact).
- **Goldens:** `goldens.py --update-goldens` changed exactly the two accepted files (`stiff_rlc`, `back_to_back_diodes_sine`); `golden_diff.py` rc 0 (38 of 52 changed, 0 worse outside the accepted list; both listed under "Farther from truth, accepted").
- **Counters** (`counters.py --freeze`, `ladder500_rc` / dsed skipped by name, F-X-3): every counter that rose, a cost of the new defaults (pwl: the restart rule's event factorisations; dsed: the output rule's samples; dsed refusals now `UnsupportedError`):

  - rc_charge / pwl / event_builds: 0 -> 1
  - rc_charge / pwl / event_hits: 0 -> 5
  - rc_charge / dsed / n_accept: 5000 -> 5008
  - ladder500_rc / pwl / event_builds: 0 -> 1
  - ladder500_rc / pwl / event_hits: 0 -> 5
  - buck_ccm / pwl / event_builds: 0 -> 4
  - buck_ccm / pwl / event_hits: 0 -> 557
  - buck_ccm / dsed / n_accept: 80 -> 1117
  - buck_dcm / pwl / event_builds: 0 -> 4
  - buck_dcm / pwl / event_hits: 0 -> 597
  - buck_dcm / pwl / event_refactors: 0 -> 75
  - buck_dcm / dsed / n_accept: 99 -> 858
  - hard_switched_rc / pwl / event_builds: 0 -> 2
  - hard_switched_rc / pwl / event_hits: 0 -> 58
  - hard_switched_rc / dsed / n_accept: 74 -> 1189
  - shockley_op / dsed: {'refused': 'ValueError'} -> {'refused': 'UnsupportedError'}
  - saturable_rl / dsed: {'refused': 'ValueError'} -> {'refused': 'UnsupportedError'}
  - half_bridge_inverter / pwl / event_builds: 0 -> 2
  - half_bridge_inverter / pwl / event_hits: 0 -> 1198
  - half_bridge_inverter / dsed / n_accept: 206 -> 1321
  - three_phase_diode_rectifier / pwl / event_builds: 0 -> 12
  - three_phase_diode_rectifier / pwl / event_hits: 0 -> 428
  - three_phase_diode_rectifier / pwl / event_refactors: 0 -> 249
  - three_phase_diode_rectifier / pwl / n_commutations: 54 -> 98
  - three_phase_diode_rectifier / trbdf2 / n_accept: 1250 -> 1251
  - three_phase_diode_rectifier / dsed / n_accept: 80042 -> 80284
  - buck_mosfet_nonlinear / dsed: {'refused': 'ValueError'} -> {'refused': 'UnsupportedError'}
  - flyback_converter / pwl / event_builds: 0 -> 4
  - flyback_converter / pwl / event_hits: 0 -> 1399
  - flyback_converter / pwl / event_refactors: 0 -> 200
  - flyback_converter / dsed / n_accept: 1500 -> 8885
  - flyback_converter / dsed / n_events: 199 -> 296
  - diode_bridge_rectifier / pwl / event_builds: 0 -> 3
  - diode_bridge_rectifier / pwl / event_hits: 0 -> 96
  - diode_bridge_rectifier / pwl / event_refactors: 0 -> 56
  - diode_bridge_rectifier / pwl / n_commutations: 38 -> 40
  - diode_bridge_rectifier / dsed / n_accept: 50010 -> 50347

- `test_benchmark_harness.py` whole: 379 passed, 1 xfailed (`test_counter_gate[ladder500_rc]`, F-X-3).
- **Step 5 evidence:** `REFERENCE_STATUS.md`, `EXACT_SCOREBOARD.md` + json, `IDEAL_SWITCH_FLIP_EVIDENCE.md`, `ACCURACY_SUMMARY.md` (`--check` rc 0) regenerated at the defaults, `@legacy` columns kept.

## 9. Triage freeze migration (2026-11-20, R1.6 leftover)

`reference_status.py` printed 72 `UNEXPECTED` lines at 4c6ba3cb: the `known:` / `known_err:` freezes of `benchmarks/manifests/v2_triage.yaml` were keyed by the pre-release configurations (`pwl`, `pwl+restart`, `pwl+split`, `trbdf2`), so every legacy-profile cell (`*@legacy`) and three `auto` cells read as new. Migrated one entry at a time from the cells of the step-3 run (`reference_status.py --json`, tree eac80cbf, extension `so_release_1b3f7ecc`; C++ unchanged since), re-annotated with `--from-json` (no re-run): 0 `UNEXPECTED`, exit 0, `REFERENCE_STATUS.md` regenerated.

Classes: **a** the same failure under a renamed key (old key `K` -> `K@legacy`, or the engine `auto` lands on); the recorded cause comment kept; **a\*** the same failure, measured more than 2x below its old freeze (the tool's "re-freeze" prompt): frozen at the measured error; **b** the cell passes now: freeze removed; **c** new failure: a release finding.

Counts over the 72 lines: a 69, a* 2, b 0 (none of the 72 passes: an `UNEXPECTED` line is a failing cell), c 1 (F-X-4).

| # | circuit | cell | old key: frozen | measured | class | new key |
|---:|---|---|---|---|---|---|
| 1 | closed_loop/cl_boost_pi.yaml | pwl@legacy | `pwl`: disagree (0.0582) | disagree (0.0582) | a | `pwl@legacy` |
| 2 | closed_loop/cl_boost_pi.yaml | pwl+restart@legacy | `pwl+restart`: disagree (0.0579) | disagree (0.0579) | a | `pwl+restart@legacy` |
| 3 | closed_loop/ll14_buck_closed_loop.yaml | pwl@legacy | `pwl`: disagree (0.29) | disagree (0.249) | a | `pwl@legacy` |
| 4 | closed_loop/ll14_buck_closed_loop.yaml | pwl+restart@legacy | `pwl+restart`: disagree (0.213) | disagree (0.221) | a | `pwl+restart@legacy` |
| 5 | closed_loop/ll14_buck_closed_loop.yaml | pwl+split@legacy | `pwl+split`: disagree (0.178) | disagree (0.188) | a | `pwl+split@legacy` |
| 6 | closed_loop/ll15_boost_closed_loop.yaml | pwl@legacy | `pwl`: disagree (0.3) | disagree (0.22) | a | `pwl@legacy` |
| 7 | closed_loop/ll15_boost_closed_loop.yaml | pwl+restart@legacy | `pwl+restart`: disagree (0.0663) | disagree (0.0659) | a | `pwl+restart@legacy` |
| 8 | closed_loop/ll15_boost_closed_loop.yaml | pwl+split@legacy | `pwl+split`: disagree (0.0484) | disagree (0.05) | a | `pwl+split@legacy` |
| 9 | closed_loop/ll16_flyback_closed_loop.yaml | pwl@legacy | `pwl`: disagree (0.261) | disagree (0.112) | a* | `pwl@legacy` |
| 10 | closed_loop/ll16_flyback_closed_loop.yaml | pwl+restart@legacy | `pwl+restart`: disagree (0.251) | disagree (0.208) | a | `pwl+restart@legacy` |
| 11 | closed_loop/ll16_flyback_closed_loop.yaml | pwl+split@legacy | `pwl+split`: disagree (0.245) | disagree (0.115) | a* | `pwl+split@legacy` |
| 12 | linear/current_source_rl_step.yaml | pwl@legacy | `pwl`: disagree (0.00577) | disagree (0.00577) | a | `pwl@legacy` |
| 13 | linear/rl_natural_response.yaml | pwl@legacy | `pwl`: disagree (0.00577) | disagree (0.00577) | a | `pwl@legacy` |
| 14 | magnetics/half_bridge_transformer_dc.yaml | pwl@legacy | `pwl`: disagree (0.151) | disagree (0.101) | a | `pwl@legacy` |
| 15 | magnetics/transformer_leakage.yaml | pwl@legacy | `pwl`: disagree (0.0041) | disagree (0.0041) | a | `pwl@legacy` |
| 16 | magnetics/transformer_leakage.yaml | pwl+restart@legacy | `pwl+restart`: disagree (0.00409) | disagree (0.00409) | a | `pwl+restart@legacy` |
| 17 | magnetics/transformer_leakage.yaml | pwl+split@legacy | `pwl+split`: disagree (0.00409) | disagree (0.00409) | a | `pwl+split@legacy` |
| 18 | stress/high_q_rlc_ringdown.yaml | pwl@legacy | `pwl`: disagree (0.0229) | disagree (0.0229) | a | `pwl@legacy` |
| 19 | stress/high_q_rlc_ringdown.yaml | trbdf2@legacy | `trbdf2`: disagree (0.00566) | disagree (0.00566) | a | `trbdf2@legacy` |
| 20 | switching/basic/vcswitch_resistor.yaml | pwl@legacy | `pwl`: disagree (0.0332) | disagree (0.0332) | a | `pwl@legacy` |
| 21 | switching/basic/vcswitch_resistor.yaml | pwl+restart@legacy | `pwl+restart`: disagree (0.0682) | disagree (0.0682) | a | `pwl+restart@legacy` |
| 22 | switching/basic/vcswitch_resistor.yaml | pwl+split@legacy | `pwl+split`: disagree (0.0332) | disagree (0.0332) | a | `pwl+split@legacy` |
| 23 | switching/boost/boost_pfc_open_loop.yaml | pwl@legacy | `pwl`: disagree (0.0304) | disagree (0.0304) | a | `pwl@legacy` |
| 24 | switching/boost/boost_pfc_open_loop.yaml | pwl+restart@legacy | `pwl+restart`: disagree (0.0304) | disagree (0.0304) | a | `pwl+restart@legacy` |
| 25 | switching/boost/ll07_boost_vcswitch.yaml | pwl@legacy | `pwl`: disagree (0.0984) | disagree (0.0984) | a | `pwl@legacy` |
| 26 | switching/boost/ll07_boost_vcswitch.yaml | pwl+restart@legacy | `pwl+restart`: disagree (0.0997) | disagree (0.111) | a | `pwl+restart@legacy` |
| 27 | switching/boost/ll07_boost_vcswitch.yaml | pwl+split@legacy | `pwl+split`: disagree (0.0997) | disagree (0.0997) | a | `pwl+split@legacy` |
| 28 | switching/boost/ll12_pfc_boost_continuous.yaml | pwl@legacy | `pwl`: disagree (0.083) | disagree (0.083) | a | `pwl@legacy` |
| 29 | switching/boost/ll12_pfc_boost_continuous.yaml | pwl+restart@legacy | `pwl+restart`: disagree (0.0806) | disagree (0.114) | a | `pwl+restart@legacy` |
| 30 | switching/boost/ll12_pfc_boost_continuous.yaml | pwl+split@legacy | `pwl+split`: disagree (0.0806) | disagree (0.0806) | a | `pwl+split@legacy` |
| 31 | switching/buck/ll06_buck_vcswitch.yaml | pwl@legacy | `pwl`: disagree (0.0781) | disagree (0.0781) | a | `pwl@legacy` |
| 32 | switching/buck/ll06_buck_vcswitch.yaml | pwl+restart@legacy | `pwl+restart`: disagree (0.118) | disagree (0.118) | a | `pwl+restart@legacy` |
| 33 | switching/buck/ll06_buck_vcswitch.yaml | pwl+split@legacy | `pwl+split`: disagree (0.0704) | disagree (0.0704) | a | `pwl+split@legacy` |
| 34 | switching/buck/ll08_buck_mosfet.yaml | pwl@legacy | `pwl`: disagree (0.227) | disagree (0.227) | a | `pwl@legacy` |
| 35 | switching/buck/ll08_buck_mosfet.yaml | pwl+restart@legacy | `pwl+restart`: disagree (0.0842) | disagree (0.0843) | a | `pwl+restart@legacy` |
| 36 | switching/buck/ll08_buck_mosfet.yaml | pwl+split@legacy | `pwl+split`: disagree (0.0842) | disagree (0.0843) | a | `pwl+split@legacy` |
| 37 | switching/inverter/ll09_half_bridge_inverter_leg.yaml | pwl@legacy | `pwl`: disagree (0.0707) | disagree (0.0707) | a | `pwl@legacy` |
| 38 | switching/inverter/ll09_half_bridge_inverter_leg.yaml | pwl+restart@legacy | `pwl+restart`: disagree (0.0587) | disagree (0.0587) | a | `pwl+restart@legacy` |
| 39 | switching/inverter/ll09_half_bridge_inverter_leg.yaml | pwl+split@legacy | `pwl+split`: disagree (0.0587) | disagree (0.0587) | a | `pwl+split@legacy` |
| 40 | switching/inverter/ll10_full_bridge_inverter_lc.yaml | pwl@legacy | `pwl`: disagree (0.05) | disagree (0.05) | a | `pwl@legacy` |
| 41 | switching/inverter/ll10_full_bridge_inverter_lc.yaml | pwl+restart@legacy | `pwl+restart`: disagree (0.05) | disagree (0.05) | a | `pwl+restart@legacy` |
| 42 | switching/inverter/ll10_full_bridge_inverter_lc.yaml | pwl+split@legacy | `pwl+split`: disagree (0.05) | disagree (0.05) | a | `pwl+split@legacy` |
| 43 | switching/isolated/flyback_converter.yaml | pwl@legacy | `pwl`: disagree (0.949) | disagree (0.949) | a | `pwl@legacy` |
| 44 | switching/isolated/forward_converter.yaml | pwl@legacy | `pwl`: disagree (0.707) | disagree (0.707) | a | `pwl@legacy` |
| 45 | switching/isolated/full_bridge_dc.yaml | pwl@legacy | `pwl`: disagree (0.0828) | disagree (0.0828) | a | `pwl@legacy` |
| 46 | switching/isolated/full_bridge_dc.yaml | pwl+restart@legacy | `pwl+restart`: disagree (0.0815) | disagree (0.0815) | a | `pwl+restart@legacy` |
| 47 | switching/isolated/full_bridge_dc.yaml | pwl+split@legacy | `pwl+split`: disagree (0.0815) | disagree (0.0815) | a | `pwl+split@legacy` |
| 48 | switching/isolated/ll11_llc_resonant_converter.yaml | auto | `trbdf2`: error | error | a | `auto` |
| 49 | switching/isolated/ll11_llc_resonant_converter.yaml | pwl@legacy | `pwl`: error | error | a | `pwl@legacy` |
| 50 | switching/isolated/ll11_llc_resonant_converter.yaml | pwl+restart@legacy | `pwl+restart`: error | error | a | `pwl+restart@legacy` |
| 51 | switching/isolated/ll11_llc_resonant_converter.yaml | pwl+split@legacy | `pwl+split`: error | error | a | `pwl+split@legacy` |
| 52 | switching/isolated/ll11_llc_resonant_converter.yaml | trbdf2@legacy | `trbdf2`: error | error | a | `trbdf2@legacy` |
| 53 | switching/isolated/ll13_active_clamp_forward.yaml | auto | `trbdf2`: disagree (0.00133) | disagree (0.00155) | a | `auto` |
| 54 | switching/isolated/ll13_active_clamp_forward.yaml | pwl@legacy | `pwl`: disagree (4.51) | disagree (4.51) | a | `pwl@legacy` |
| 55 | switching/isolated/ll13_active_clamp_forward.yaml | pwl+restart@legacy | `pwl+restart`: disagree (0.335) | disagree (0.335) | a | `pwl+restart@legacy` |
| 56 | switching/isolated/ll13_active_clamp_forward.yaml | pwl+split@legacy | `pwl+split`: disagree (0.335) | disagree (0.335) | a | `pwl+split@legacy` |
| 57 | switching/isolated/ll13_active_clamp_forward.yaml | trbdf2@legacy | `trbdf2`: disagree (0.00133) | disagree (0.00133) | a | `trbdf2@legacy` |
| 58 | switching/isolated/synchronous_flyback.yaml | pwl@legacy | `pwl`: disagree (0.207) | disagree (0.207) | a | `pwl@legacy` |
| 59 | switching/multilevel/cascaded_h_bridge_5level.yaml | pwl@legacy | `pwl`: disagree (0.0274) | disagree (0.0274) | a | `pwl@legacy` |
| 60 | switching/multilevel/cascaded_h_bridge_5level.yaml | pwl+restart@legacy | `pwl+restart`: disagree (0.0269) | disagree (0.0269) | a | `pwl+restart@legacy` |
| 61 | switching/multilevel/cascaded_h_bridge_5level.yaml | pwl+split@legacy | `pwl+split`: disagree (0.0269) | disagree (0.0269) | a | `pwl+split@legacy` |
| 62 | switching/multilevel/flying_capacitor_3level.yaml | pwl@legacy | `pwl`: disagree (0.0466) | disagree (0.0466) | a | `pwl@legacy` |
| 63 | switching/multilevel/flying_capacitor_3level.yaml | pwl+restart@legacy | `pwl+restart`: disagree (0.05) | disagree (0.05) | a | `pwl+restart@legacy` |
| 64 | switching/multilevel/flying_capacitor_3level.yaml | pwl+split@legacy | `pwl+split`: disagree (0.05) | disagree (0.05) | a | `pwl+split@legacy` |
| 65 | switching/multilevel/mmc_4cell_chain.yaml | pwl@legacy | `pwl`: disagree (0.0268) | disagree (0.0268) | a | `pwl@legacy` |
| 66 | switching/multilevel/mmc_4cell_chain.yaml | pwl+restart@legacy | `pwl+restart`: disagree (0.0264) | disagree (0.0264) | a | `pwl+restart@legacy` |
| 67 | switching/multilevel/mmc_4cell_chain.yaml | pwl+split@legacy | `pwl+split`: disagree (0.0264) | disagree (0.0264) | a | `pwl+split@legacy` |
| 68 | switching/multilevel/t_type_3level_halfbridge.yaml | pwl@legacy | `pwl`: disagree (0.0749) | disagree (0.0782) | a | `pwl@legacy` |
| 69 | switching/multilevel/t_type_3level_halfbridge.yaml | pwl+restart@legacy | `pwl+restart`: disagree (0.0738) | disagree (0.0771) | a | `pwl+restart@legacy` |
| 70 | switching/multilevel/t_type_3level_halfbridge.yaml | pwl+split@legacy | `pwl+split`: disagree (0.0738) | disagree (0.0771) | a | `pwl+split@legacy` |
| 71 | thermal/electrothermal_resistor_self_heating.yaml | auto | `dsed`: none | disagree (0.00178) | c | `auto` (F-X-4) |
| 72 | three_phase/three_phase_diode_rectifier.yaml | pwl@legacy | `pwl`: disagree (1.51) | disagree (1.51) | a | `pwl@legacy` |

The default-profile keys of the same entries (not among the 72: none was `UNEXPECTED`) were migrated with them:
- **b**, removed (the default cell agrees; the tool printed "now agree: drop the freeze"): `pwl` on `cl_boost_pi.yaml`, `current_source_rl_step.yaml`, `rl_natural_response.yaml`, `half_bridge_transformer_dc.yaml`, `high_q_rlc_ringdown.yaml`, `boost_pfc_open_loop.yaml`, `flyback_converter.yaml`, `forward_converter.yaml`, `synchronous_flyback.yaml`, `three_phase_diode_rectifier.yaml`.
- kept, `pwl` frozen at the old `pwl+split` error (the release `pwl` default is the old `pwl+split` run; measured within 2x) on `ll14_buck_closed_loop.yaml`, `ll15_boost_closed_loop.yaml`, `transformer_leakage.yaml`, `vcswitch_resistor.yaml`, `ll07_boost_vcswitch.yaml`, `ll12_pfc_boost_continuous.yaml`, `ll06_buck_vcswitch.yaml`, `ll08_buck_mosfet.yaml`, `ll09_half_bridge_inverter_leg.yaml`, `ll10_full_bridge_inverter_lc.yaml`, `full_bridge_dc.yaml`, `ll13_active_clamp_forward.yaml`, `cascaded_h_bridge_5level.yaml`, `flying_capacitor_3level.yaml`, `mmc_4cell_chain.yaml`, `t_type_3level_halfbridge.yaml`; `ll16_flyback_closed_loop` `pwl` 0.245 -> 0.115 (measured, more than 2x below); `trbdf2` kept as is on `high_q_rlc_ringdown` (0.00566) and `ll13_active_clamp_forward` (0.00133, measured 0.00155); `ll11_llc_resonant_converter` `pwl` / `trbdf2` `error` kept.
- The old `pwl+restart` / `pwl+split` keys are gone: those configurations exist only as `@legacy` columns now.
- ll11 `auto`: the no-dt route aborts inside trbdf2 (`run_transient_trbdf2: Newton failed at t = 1e-5`), the same failure as the frozen `trbdf2` error: class a. ll13 `auto` lands on trbdf2 (1.55e-3, the `trbdf2` default cell measures the same): class a, frozen at the recorded trbdf2 error 1.33e-3.
- **F-X-4** (class c): `electrothermal_resistor_self_heating` `auto` disagrees 1.78e-3 (tol 1e-3, err_max 1.0). The run is right (V(vin) = 5 V); without dt, `auto` now lands on dsed, whose output rule writes 2,000,002 samples (1 us) on this 2 s DC resistor (trbdf2 route before: 1004 samples), and the reference's uic sample V(vin)(0) = 0, interpolated to 5 V at 2e-5 s, spans 20 of them. dsed with the netlist dt agrees (err 0, 4002 samples). Group F (output rule), severity C; frozen `auto: disagree (0.00178)` citing the id.

## 10. The rehearsal's bar for cells under the output rule (orchestrator, 2026-11-20)

- KEPT: F-X-4 ("a step end the rule's chord already covers is not stored"). A DC circuit under
  the rule stores its two ends; a slow stretch keeps few samples.
- The bar "no cell worse than 2x the base" (section 5's verification) is REPLACED, for every cell
  that ran under the output rule (the exact engine without `dt`, `diagnostics['output_grid']`;
  the rehearsal writes `rule_ratio`), by: no cell changes class (outcome rank), and every cell
  within what the rule promises -- the linear interpolation of its samples within
  `output_rtol`*range + `atol` of the reference at every interval midpoint between events
  (`metrics.rule_ratio` <= 1; range = the reference's range, at least 1e-3 of its peak). The base
  looked better only because the 1 us gate poll (`dt_max / 10`, kept: decision of the same day)
  stored extra points. Other cells keep the 2x rules.
- The harness compares a short trace: dropping the initial sample is skipped when it would
  leave fewer than two (`metrics.first_kept`); the reference is then read from its own first
  kept sample. `undersampled` is reserved for a trace that cannot be compared, with the reason.
- The opaque-gate poll stays at `dt_max / 10` (1 us with the defaults); the documents that said
  10 us now say so. No number moves.

## 11. The exact scoreboard's measure in three parts, and rule (1), 2026-11-20

Branch `feat/core-scoreboard-measure`. Evidence only: no library file changes, no default number
moves. The measure, the old one kept beside it, the tests and the triage are in
`fixes/restart-scoreboard.md` ("Measure v2"); the scoreboard's `## The measure` states it.

1. **Definition.** `exact` = the largest of three parts of the worst column, each a fraction of
   the truth's span: smooth (the engine's stored rows, minus the initial instant, the 0+ row and
   the rows inside a matched event's displacement), instants (sqrt(Σ jump²·|δ| / T), a missing
   event charged as displaced to the device's next event, at most 2 h), limits (max distance of
   every recorded one-sided value from the truth's). The previous definition is kept as
   `exact_old` for this release (json and table), and a `--previous` json without the measure
   tag is compared on `exact_old`.
2. **Rule (1), a projected transient (decided by the orchestrator).** At a matched event where
   the engine REPORTS a projection (a `reinit_records` entry within 4 ulp of its instant) and
   records a right value the truth does not have at its instant, the truth is followed to
   32 W (W = the engine's declared parasitic window, `PARASITIC`: dt/100 for the exact engine,
   from `project_`'s threshold 100/dt_max with dt_max = dt in the harness; h/32 for pwl, so
   followed to h), on the offsets 32 W·2^-k (k < 60) cut at irrational fractions, never past the
   next truth event. If it settles (one doubling moves it ≤ max(1e-9 span, 1e-6 jump)) and every
   value from its settling offset tau_eff on is within the cell's tol of the engine's value, the
   right value is a transient collapsed in time: priced in instants by its true L2 cost
   ∫(truth − engine)² over [t_e, t_e + tau_eff] (once per truth instant), its residual against
   the settled truth kept in limits, counted in `n_projected`. Either condition failing leaves the limit as it is. A fixed-engine impulse
   record is told apart by the data: the truth makes that jump too (its own right limit is
   already the recorded value, carrying the jump energy dW the record books), so nothing is
   collapsed and the rule does not apply; a projection's record would carry only the transient's
   dissipation, and the truth reaches its value only after tau_eff.
3. **What rule (1) changed: no cell.** No engine writes a record at the events it was made
   for. The exact engine's slow-manifold projection (`scheduler.hpp` `project_` →
   `project_slow`) writes none; `reinit_records` comes only from the constraint re-init
   (`builder_adapter.hpp`), and only above 1e-9 of the circuit's scale. On
   `diode_half_wave_rl_load` dsed the rule WOULD apply (forced in a probe: 0.030 limits →
   4.8e-6 instants, limits 3.0e-7 = the frozen-source error of the projected value, −3.1e-6 V
   against the settled truth's −6.3e-6 V; 3 turn-offs). Finding F-SB-3.
   Reviewer (fresh context, adversarial): the first version (cost jump²·tau_eff on the bare
   dyadic offsets W·2^-k) hid an overshoot of 184x the span (7.1e-4 against an honest 2.5e-2), a
   ringing of 10x the span whose period was a dyadic fraction of W (3.5e-4 against 2.9e-3), and
   charged a transient once per device; fixed in 6b85d9b5 as above, each case a test. Left as
   written: the projection flag is per instant, not per state (any record at that instant arms
   the rule; condition (ii) still gates it); a narrow pulse between two evaluated offsets can
   still be missed, but the truth is one LTI segment there (no truth event inside), a sum of
   exponentials, so only a ringing can hide and the irrational cut defeats the aliasing.
4. **Triage of the cells the one-sided-limits part set.** See `fixes/restart-scoreboard.md`
   ("Triage"). F-SB-1's test (each right-limit row at a restart-window step with no event
   replaced by the engine's own left value, re-graded) moved none of them: none is F-SB-1.
   New findings, strict xfails in `python/tests/test_restart_scoreboard_findings.py`: F-SB-2
   (a switch_fn edge inside a step: split applies it from the step start, restart from the next
   step), F-SB-3 (a diode turn-off behind a sub-step transient: a right limit the truth does not
   have, no projection record), F-SB-4 (the split rule locates diode_rectifier's D1 turn-on
   0.227 h late).

## 12. R3 merge of the PR head 03c8b67c (orchestrator, inline, 2026-11-20 — running note)

**Merged:** `feat/core-integration` 03c8b67c (batches 14q–14w) into `release/core-next-defaults` (9e051abe). Sixteen conflicts; the library ones: `kRunOptionTable` (sensitivities on every engine + the release's `mode_selection` / `device_path` defaults); `simulate()` (the step-engine sensitivity dispatch inside the auto → exact fallback; the `t_start` relabel around the exact run, with `output_rtol_explicit`; recorders KEEP the output rule under the release default, as the PR line decided with the closed-form statistics); `v2_triage.yaml` (the release's migrated keys and values + the PR line's cause ids; the two new ZVS circuits keyed `pwl@legacy`); the PR line's two DECISIONS sections are 10 and 11 here.

**Proofs on the merged tree (extension built from it):**
- B(i) byte compare under `next_defaults.legacy()` against 03c8b67c (`git archive` + the 7a9ac912 extension, same engine code): goldens 22/22, stateful 41/41, engines 36/36, steps 29/29, `pwl_identity` 396/396, `trbdf2_identity` 376/377 — the one is the same v1-netlist refusal text as in section 8.
- B(ii) `reference_status` exit 0, no `UNEXPECTED` line. `rehearsal_match` exits 1 on one rounding-level cell (`analytic:gapped_core_rl` dsed: 1.82e-15 in the rehearsal, 4.53e-16 now).
- B(iii) exact scoreboard regenerated under the three-part measure; `accuracy_summary --check` clean.
- Goldens: 5 files changed, `RELEASE_GOLDEN_DIFF.md` 38 changed / 0 worse. `test_benchmark_harness.py`: 384 passed, 1 xfailed.
- Counters re-frozen. RISES, recorded and NOT explained yet: the fixed engine's `event_hits` about doubled on every switching case (buck_ccm 557 → 1037, buck_dcm 597 → 1191, hard_switched_rc 58 → 123, half_bridge_inverter 1198 → 2398, flyback_converter 1399 → 2999, diode_bridge_rectifier 96 → 336, three_phase_diode_rectifier 428 → 1576, rc_charge / ladder500_rc 5 → 11). Falls: the exact engine's `n_accept` (buck_ccm 1117 → 80, flyback 8885 → 1597: the output rule stores only what its tolerance needs).

**Suite:** full pytest on 9e051abe: 164 failed / 6916 passed / 136 skipped / 62 xfailed. Sixteen were the regenerated evidence above. Of the other 148, a diagnostic run with the default output rule switched off (never a fix) splits them: 62 are the rule's sampling (a test interpolating or averaging the exact engine's default samples against a bound tighter than `1e-5 × range`: since batch 14s the rule does not store a step end its chord covers) and 86 are the other defaults or merge interactions. Done so far: 2e69f0b0 (about 25 of the 62, `output_rtol=0` where the subject is the step grid or the time base). Two groups of the 86 are release findings, not test edits: step-engine sensitivities are refused at every event because event splitting is the default (needs `feat/core-sensitivity-events`), and the restart rule refuses an event that moves a saturable inductor (the flux-form twin of the nonlinear-capacitor jump). NOT run on the merged tree: ctest, Eigen 3.4.0, the rehearsal, `findings_index.py`.
