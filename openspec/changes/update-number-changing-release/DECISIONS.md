# Decisions — update-number-changing-release

## 1. The exact route's output grid without dt (`output_rtol`), 2026-11-19

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

## 2. Branch-form coverage (blockers 1.4–1.8, feat/core-branch-form-coverage)

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

## 3. Owner decisions (2.1–2.4), 2026-11-24

Closes tasks 2.1–2.4 before the release flip while preserving 100 %
bit-identical and warning-identical behaviour under today's defaults.

### 3.1 One undriven-switch policy (task 2.1; `add-kernel-simulation-object` DECISIONS 23 item 1, task 11.3)

- **Decision:** at the release flip, any circuit with controlled switches and
  no `switch_fn` (nor `closed_loops` / control graph driving them) is **refused
  on every engine** (`pwl`, `trbdf2`, `dsed`/`exact`, `auto` with or without
  `dt`) with `pulsim.UnsupportedError(engine, "undriven_switch", ...)` naming
  the undriven controlled switch(es). Passive circuits and diode-only circuits
  have no controlled switches and are never refused.
- **Pre-flip default:** `pulsim._plan.DEFAULT_UNDRIVEN_SWITCH_POLICY = "warn_closed"`
  (allowed values: `"warn_closed"`, `"refuse"`). Today's `"warn_closed"` keeps
  existing per-engine behaviour, call order, warnings and bytecodes unchanged.
- **Rehearsal wiring:** `benchmarks/tools/next_defaults.py` records
  `POLICY["DEFAULT_UNDRIVEN_SWITCH_POLICY"] = "refuse"` (plus a
  `policy_defaults()` context manager); `benchmarks/tools/next_defaults_pytest.py`
  sets `_plan.DEFAULT_UNDRIVEN_SWITCH_POLICY = "refuse"` during the rehearsal
  session (re-raising `feature == "undriven_switch"` immediately instead of
  falling back `exact -> trbdf2`) and restores all patched `_plan` defaults and
  wrappers in `pytest_unconfigure()`.
- **Tests:** `python/tests/test_undriven_switch_policy.py`.

### 3.2 RunPlan blocks 2–8 (task 2.2; `add-kernel-simulation-object` DECISIONS 23 items 2–8)

Each position-dependent block recorded in `add-kernel-simulation-object`
DECISIONS 23 is decided explicitly for this release:

- **Block 2 (`mmc_arms` / `on_cache`):** keep in `_run_pwl`. Both are pwl-only
  (refused by `sim::kRunOptionTable` on `trbdf2` and `dsed`) and require the
  run's own `Simulation.cache`; constructing `Simulation` during `_plan_run`
  would violate planning purity. Deferred to Stage B (`add-kernel-simulation-object`
  11.1 / 12.2).
- **Block 3 (`progress` / `live_stream`):** keep in `_run_pwl`. Both are
  pwl-only, and `live_stream.attach()` mutates the caller's stream object,
  which must happen at run start rather than during planning.
- **Block 4 (`resume_from` time):** keep the `engine != "dsed"` guard in
  `_plan_run`. Following `core-a-complementarity` (`add-exact-propagator-strategy`
  sessions 24–26), `dsed` natively supports `resume_from` on its own absolute
  clock (`t_end` is absolute; `_dsed_dispatch._check_resume_snapshot` refuses
  relabelling `t_start != 0.0`). Rewriting `t_start = float(resume_from.t)` on
  `dsed` would break exact-engine resume; keeping it for `pwl` and `trbdf2`
  only (when `t_start == 0.0`) matches all three engines' contracts.
- **Block 5 (`mode_selection='auto'`):** keep current resolution sites
  (`_route` for `pwl`/`trbdf2`, `_dsed_dispatch` for `dsed`, `_plan_run` for
  `sensitivities`). Unify only when the exact engine moves onto
  `sim::Simulation` (`add-kernel-simulation-object` DECISIONS 16).
- **Block 6 (`on_impulse='error'`):** keep as-is (`pwl` aborts at the event in
  C++, `dsed` raises `UnsupportedError("dsed", "impulse", ...)` in `_finalize`
  from `reinit_records`, `trbdf2` never records impulses). All three honour the
  contract; moving `dsed`'s abort into the C++ scheduler is a post-release
  fail-fast optimization.
- **Block 7 (pwl-only option value checks & tri-state restart/split defaults):**
  keep in `_plan_run`'s `pwl` branch. Unsupported keywords are refused on
  `trbdf2` / `dsed` by `sim::kRunOptionTable` and kernel validation, and the
  tri-state `discontinuity_restart` / `enable_substep_state_correction` default
  resolution already lives here (blockers 1.5 / 1.8).
- **Block 8 (`sweep()` recipe `_PWL_RECIPE`):** keep reading once during
  `_plan_run` (`plan.recipe`) and populating in `_run_pwl`. Moving `sweep()`
  replay onto `RunOptions` / `Simulation` is post-release
  (`add-kernel-simulation-object` task 12.2).

### 3.3 One voltage-sanity check (task 2.3; `add-kernel-simulation-object` task 6.2)

- **Decision:** unify all three engines (`pwl`, `trbdf2`, `dsed`) directly onto
  `_auto_check_voltage_sanity(res, builder, voltage_sanity_factor)` in
  `_finalize`.
- **Verification:** `dsed` and `trbdf2` already ran `_auto_check_voltage_sanity`
  in `_finalize`. Replacing `pwl`'s duplicated inline `find_implausible_voltage`
  block at the same site in `_finalize` is 100 % warning-identical
  (`stacklevel=4` inside `_auto_check_voltage_sanity` resolves to the exact
  same caller frame as `stacklevel=3` inline in `_finalize`; same message,
  category, order, `res._implausible_voltage`, and `-W error` behaviour) and
  bit-identical, while removing the last swallowing broad handler from
  `_SIMULATE_BODY`. No feature toggle is needed.

### 3.4 TR-BDF2 step grid (task 2.4; `update-trbdf2-factor-reuse` Part 3)

- **Decision:** **defer** `step_grid_per_octave` (Part 3 of
  `update-trbdf2-factor-reuse`, tasks 4.1–6.3) to a post-release PR. Keep
  continuous step sizing as the default in this release so the two-pillar
  release is not coupled to an unimplemented performance feature.
  `benchmarks/tools/next_defaults.py` `NOT_APPLIED` updated accordingly.
