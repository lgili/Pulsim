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
