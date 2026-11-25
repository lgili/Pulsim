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

## 5. The output rule's cost (`output_rtol`), 2026-11-19 — diagnosis

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

## 6. The rehearsal's bar for cells under the output rule (orchestrator, 2026-11-20)

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

## 7. The exact scoreboard's measure in three parts, and rule (1), 2026-11-20

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

## 8. The rehearsal's analytic gate reads the projected flow (owner's delegate, 2026-11-25)

- The output rule's promise is stated against the flow with the parasitic cluster projected at
  events (the `simulate` docstring), and every projection is recorded
  (`diagnostics['output_grid']['projections']`: `t`, `amplitude`, `column`, `tau`). So the
  rehearsal's analytic gate (err_rms and `rule_ratio` of an analytic case) reads the PROJECTED
  closed form: `metrics.projected_reference(case.exact, records)` removes, on each record's window
  `(t, t + 40 tau]`, the closed form's own fast part `D exp(-(s - t)/tau)`.
- Only `t` and `tau` come from the engine. `D` is the closed form's fast part (the closed form
  minus its slow part, extrapolated linearly from `t + 40 tau` and `t + 80 tau`); a record whose
  `tau` the closed form does not decay at (within 1e-3 at `t + tau`) is not applied, so the gate
  never takes the engine's projected values as truth. One mode per instant; records past the cap
  (1000) are not applied (those windows stay graded against the unprojected form).
- F-X-7's sample 2 ps past each event stays. The engine does not change.
- A run with no record is graded by `case.exact` itself (the same object): its cells are
  byte-identical. A cell with records carries `n_projected`.
