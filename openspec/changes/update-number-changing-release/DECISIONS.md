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

## 2. One undriven policy and the per-engine blocks (tasks 2.1–2.3, feat/core-release-options)

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

## 3. Blast radius with `PLAN` (hand-off, feat/core-release-options)

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
