# Decisions taken while implementing (owner decisions delegated)

1. **The one declaration is a C++ table.** `sim::kRunOptionTable`
   (`core/include/pulsim/sim/options.hpp`) has one row per `simulate()`
   keyword (plus the `c_blocks` circuit feature the router reads): the
   engines that honour it, which value means "not requested", its
   documented default and the mechanism the other engines lack. Python
   reads it (`pulsim._pulsim.run_option_table()`, `_run_options.py`); the
   dsed refusal, the post-routing refusal, the fixed-only blockers of
   `_trbdf2_blockers` and `pulsim.option_capabilities()` are all derived
   from it. A test asserts its names equal `simulate()`'s parameters, so a
   new keyword cannot be added without a row.

2. **`sim::Defaults` is its own header** (`sim/defaults.hpp`) so that the
   driver structs (`SimulationOptions`, `TrBdf2Options`) can initialise
   from it without including the facade. Open question 2 answered: two
   pairs, by strategy (1e-5 / 1e-8 MNA variable step, 1e-6 / 1e-9 exact),
   because they are the values `simulate()` has always used.

3. **`RunOptions` is flat with optionals, not D2's nested variant.** It
   maps one-to-one onto the `simulate()` keywords the kernel reads, so the
   wrapper is a field copy and the table check covers it; the variant's
   structural refusal (a fixed-step field cannot be written on a variable
   step) is replaced by the table check, which also covers the keywords
   that are Python attachments (observers, streams, MMC arms). "Not set" is
   `std::optional` / `None` everywhere; no sentinel.

4. **Silent ignores removed (decided per the proposal: refuse by name).**
   Before, warned-then-ignored or ignored outright:
   - `rtol`, `atol`, `dt_init` on the fixed step (warning);
   - `integrator`, `stiffness_threshold`, `h_bdf2` on pwl / trbdf2 / auto
     (warning);
   - on the exact engine, ignored with no word: `enable_nonlinear_refresh`,
     `max_newton_iterations`, `tol_newton_dx`, `tol_newton_res`,
     `enable_newton_line_search`, `enable_newton_lm`,
     `max_event_iterations`, `control_tick='state'`;
   - `max_event_iterations=0` on trbdf2 (replaced by 16);
   - `tol_newton_dx=0` / `tol_newton_res=0` on trbdf2 (replaced by 1e-9).
   The first eight are refusals now; the last is honoured (zero is a value,
   as on pwl). Only `max_newton_iterations=0` follows the proposal's
   warn-then-refuse (a `DeprecationWarning` for one release).

5. **Loosened, not tightened:** the exact engine accepted no
   `max_dt_halvings` at all; `max_dt_halvings=0` (no retry) is accepted now
   on every engine, as on trbdf2 before — no engine lacks "no retry".

6. **Stage A (`_plan_run`, `RunPlan` in Python) was not built.** The Python
   `RunPlan` of `add-immutable-circuit-ir` (circuit, ties, preflight)
   exists already; the options are resolved by the kernel from one
   `RunOptions` instead (Stage B, task 9), which is what removes the
   scattered defaults. The composition still lives in `simulate()`.

7. **The facade runs pwl and trbdf2; the exact engine is refused by name.**
   It is composed in Python (`_dsed_dispatch.run_dsed_from_builder`); moving
   it is not a facade job. C blocks and graph-driven sources are refused by
   name too (they are composed by `_control_plan` in Python).

8. **Undriven switches: today's per-engine behaviour kept** (pwl
   all-CLOSED + a diagnostic, trbdf2 all-OPEN, 'auto' without `dt` refuses).
   Open question 1 (one policy) is a BREAKING change for a later PR
   (task 11.3), not taken here.

9. **`simulate()` does not yet run through `Simulation`** (task 11.1). It
   resolves its options through the same kernel functions, so its results
   are the facade's by construction; the parity test
   (`test_simulation_object_equals_simulate_bit_for_bit`) compares both on
   pwl and trbdf2.

10. **`PulsimError` is created in the extension, not in `errors.py`.** The
    kernel's exception translators must name the Python class at throw
    time, so the base and the kernel-raised classes (`UnsupportedError`,
    `ConvergenceError`, `OptionsError`) are made in `bindings.cpp` with two
    bases (the standard one and `PulsimError`); the Python-raised ones
    (`Cancelled`, `NameNotFoundError`) add `PulsimError` to their bases. No
    `errors.py` re-export module: `pulsim.<Class>` is the one name.
    `NetlistError` and `SingularError` are not added: no raise site uses
    them yet (task 8.2), and a class nothing raises is a promise. The C++
    side is a mix-in (`sim/error.hpp`, `PulsimError::message()`), not a
    `std::exception` subclass, so `catch (const std::exception&)` stays
    unambiguous.

11. **Raw `run_transient_trbdf2` sentinels removed.** `max_event_iterations`,
    `max_newton_iterations`, `newton_roundoff_budget` are `None` by
    default; 0 is a value (the first two refuse it, as `to_trbdf2_options`
    and `to_simulation_options` do).

12. **Batch-13 options get a row when they land, not before.** Adding a
    `simulate()` keyword is one `kRunOptionTable` row (engines, unset
    kind, default, why) plus, when the kernel reads it, one `RunOptions`
    optional and a line in `to_simulation_options` / `to_trbdf2_options`;
    `test_the_table_covers_every_simulate_parameter` fails until the row
    exists. Planned: `device_path` ('legacy' = not requested; a new
    `Unset` kind like `Legacy`), `mode_selection` gaining `kTrBdf2` (a
    one-bit change), a monitor tick kind (an `add_tick` variant on the
    facade), controller-state / tick-origin snapshot blocks (fields of
    `SolverSnapshot`, which `Simulation::snapshot()` already returns
    whole). No placeholder field is added now: a field nothing reads
    would be the silent ignore this change removes.

13. **simulate()'s fixed overhead is held by construction, not by timing.**
    The shared Mac's load swings timings by +-50%, so the gate was checked
    with a deterministic count of Python bytecodes executed per call
    (`sys.settrace` opcode events, 6-branch buck): base 2308 (pwl, 20
    steps) / 3033 (trbdf2), this change 2266 / 3026. To get there the
    table check filters "not set" values in C (`itertools.compress`), the
    options are filled in C++ (`_simulation_options_kw`,
    `_trbdf2_options_kw`: no Python attribute writes), the option snapshot
    is a literal (test-checked against the table) instead of `locals()`,
    and five in-function `from . import` on the hot path became module
    handles. The kernel time is unchanged (same drivers, same options;
    goldens identical).

14. **simulate() on the kernel object, engine by engine.** The `'pwl'` and
    `'trbdf2'` runs go through `sim::Simulation` (goldens, stateful,
    engines and options compares identical). Choices that keep the bits:
    the object owns the cache (`Simulation.cache`, built on first use), so
    the run's cache is the object's; the initial state and
    `start_from_dc_op` apply to the first run call only (a continuation
    is the snapshot's); the first call's tick grid has a NaN origin (the
    run's t0, as the raw drivers get it) and continuations keep the
    original t0; a Python schedule's `next_edge_after` is read on the
    fixed step only (`run_transient_trbdf2` never read it); `restore(s,
    at_snapshot_time=False)` keeps simulate()'s "an explicit t_start
    wins"; the preflight is simulate()'s (`auto_regularize=False` on the
    object, which receives the tied circuit); C blocks are composed by
    simulate() and passed as attachments (`controls_composed=True`).
    `pulsim._pulsim._run_simulation(sim)` is a module function so a
    profiler sees the kernel run as one C call, as it saw `run_transient`.
    Still direct: `mmc_arms` / `on_cache` (they need the cache before the
    options exist: `mmc_arms` forces `max_dt_halvings = 0`), `sweep()`'s
    replay, the C++ chain fast path, and the exact engine (its driver is
    composed in `_dsed_dispatch.py`, owned by the event-queue and exact
    tracks). Measured in one process, the object path is not slower than
    the direct one (43.7 vs 44.2 us per call through a builder; 39.8 vs
    42.8 us through a Circuit).

15. **Integration batch 14a merged (656b894f); every new keyword is a row.**
    `device_path` is row 49 (pwl, trbdf2; `Unset::Legacy`; a
    `RunOptions` field passed to both drivers). The exact engine now runs
    controllers on its event queue, so `step_observer`, `closed_loops`,
    `controller_period` and `control_tick` became `kEvery` /
    `Unset::Always` (no engine refuses them by the table; an every-step
    observer on a variable-step engine keeps its own refusal, feature
    `step_observer`); `Unset::Label` had no row left and was deleted.
    Monitors are `add_tick(..., monitor=True)` (trbdf2 reads them; pwl
    ignores the flag, as `run_transient` does). The object's resume reads
    the v5 snapshot's `tick_origin` / `tick_next_k` when it has one per
    tick source (simulate()'s `_resume_tick_origins` rule), on a
    `restore` and on a continuation alike; the pickle itself is the
    binding's (`SolverSnapshot`), so the object writes v5 / 17 fields.
    `test_the_table_covers_every_simulate_parameter` now names its
    exclusions (`_NOT_A_ROW = {"builder"}`) and fails on a keyword in
    neither. `mode_selection='auto'` and the exact-route options are not
    in 656b894f's `simulate()`; they become rows when they land (12).

16. **The exact engine is not on the object yet (hand-off).** Its run is
    composed in two places: `_dsed_dispatch.run_dsed_from_builder`
    (integrator routing by diode count and ticks, the shadow/guard event
    step `_shadow_h_e`, IC mapping, the default mask, the poll warning,
    the bridge fallbacks) and `python/dsed_bindings.cpp`
    (`run_*_native_builder`: diode predicates, ticks, the mode guard,
    `EdgeSearch` / `poll_window`, the full-state reconstruction). Putting
    it on `sim::Simulation` byte-identically means moving the binding's
    composition into a core header (`dsed/builder_run.hpp`) that the
    binding and the object both call, then porting the dispatch's rules.
    Not started here: `dsed_bindings.cpp` is being edited on
    `feat/core-a-complementarity` (11 commits ahead), and moving ~400 of
    its lines now would hand that track a large conflict. Do it right
    after that branch is integrated. The object keeps refusing
    `engine='exact'` by name (DECISIONS 7).

17. **`mmc_arms` / `on_cache` on the object: the open question is the
    pool.** Both hooks take the run's cache before the run (`Simulation
    .cache` gives it: built on first use, kept). But the object stamps
    into its OWN builder's pool (`circuit.to_builder()`), while
    `make_mmc_thevenin_driver(mmc_arms, _caller_builder, cache, ...)` and
    the electro-thermal `on_cache` hooks read and refactor through the
    caller's builder. On the direct path the cache is built on that same
    builder, so their writes reach the solve; through the object they
    would reach a different pool. Checked for `mmc_arms`: its driver
    writes only through `cache.refactor_parametric` (mmc_thevenin.py
    ~line 206, "the caller's builder is not touched"), but it reads the
    caller builder's `branch_id` / `node_from` / `node_to` and checks
    `a._builder is not builder`, so it can move first only once the
    object's builder (`circuit.to_builder()`, after the preflight ties)
    is shown to keep that numbering; the `on_cache` hooks are user code and the shipped ones are
    unchecked. The successor must route their pool access through
    `sim.builder` (or prove they only touch the cache) before `_via_sim` drops its `not mmc_arms and on_cache is
    None` clause, then compare `test_mmc*`, `test_electrothermal*` and
    the goldens bit for bit. `mmc_arms` also needs `max_dt_halvings = 0`
    in the `RunOptions` (set before the object is built: the refusals
    above it stay as they are).
    Trap for that change: the C++ chain fast path (`cxx_chain`, decided
    after the hooks) calls `run_transient_with_chain` on the LOCAL
    builder's graph and pool; a cache taken from a pre-built object
    belongs to the object's builder, so that path must keep its own
    cache (or the chain must move onto the object) before `mmc_arms`
    goes through `Simulation`.

18. **Hand-off list (2026-11-18).** Next, in order: (a) the exact engine on
    the object (16), after `feat/core-a-complementarity` is integrated;
    (b) `mmc_arms` through `Simulation` (17: builder numbering, the chain
    fast path); (c) `on_cache` (17: user hooks and the pool); (d) Stage A
    (tasks 1-7). Seen in passing, not fixed: `losses.impulse_losses`
    reads `rec.t` but the exact engine's reinit records reach Python as
    `SimpleNamespace` from dicts (`_dsed_dispatch._result_from_native`),
    which is fine; a raw `run_*_native_builder` dict record would not be.
    Not verified here: Eigen 3.4.0 (the C++ touched is option plumbing,
    no sparse code).
    State at hand-off: 9fbbdb2a + 09b5ff34 + this commit are LOCAL, not
    pushed: the full ctest build (156 steps after the merge) did not
    finish in this session, and no tip is pushed without it. Next agent:
    `zsh .scratch/build.sh ctest`, run ctest, then push. Bytecode count
    per simulate() call (`.scratch/opcount.py`, 6-branch buck): pwl 2462
    vs 2398 at 656b894f (+64, above the DECISIONS 13 gate of "not above
    base"; not investigated), trbdf2 3028 vs 3159.

19. **The +64 bytecodes on pwl (18) were the object route's own cost, not
    a merge item; removed, not re-based.** Counted per line with
    `.scratch/opcount.py` (sys.settrace opcode events; 6-branch buck) on
    656b894f and 3835a4ac and diffed by line text: the base's
    `_validate_engine_kwargs` work and option attribute writes were gone,
    but the object route added `_opt_values`, two `refused()` calls (the
    second inside `_trbdf2_blockers`, which built its own 13-entry dict),
    the `_run_on_simulation` setter probes and a 5-clause `_via_sim`
    boolean. Fixes, all behaviour-free: `_trbdf2_blockers` reads
    simulate()'s `_opt_values` through the pseudo-engine row
    `_ro.TRBDF2_ROUTE` (the 13 routing rows only: `mode_selection` on a
    dt-less 'auto' run stays a refusal, not a route; tested);
    `refused()` skips its comprehension when no non-falsy row is set;
    `_run_on_simulation` returns before the setter probes when `kw` has
    only its three base keys; `_via_sim` is an `if`; the pwl integrator
    check tests the (usually None) integrator first. Result: pwl 2394
    (base 2398), trbdf2 3012 (base 3159), after the hooks change below.
    The count stays a local tool: it depends on the interpreter version,
    so no CI test asserts it (STATE.md CI lessons).

20. **`mmc_arms` and `on_cache` run on the object: one cache, numbering
    checked once.** The object is built first for a hooked run (on the
    same `RunOptions`; mmc's `max_dt_halvings = 0` is set before them),
    and `builder` / `cache` become `Simulation.builder` /
    `Simulation.cache`, so the MMC driver's and the hook's
    `refactor_parametric` reach the pool the run solves. The C++ chain
    fast path keeps that cache and builder when a hook took it (a hook
    that returns None leaves the chain observer in place; tested bit for
    bit against the hook-free run). The hooks resolve ids on the CALLER's
    builder: `_check_run_numbering` compares node count, names and the
    caller's branch records + names against the run's builder (ties may
    only append) and refuses by keyword otherwise; a `Circuit` caller is
    its own lowering. `on_cache` receives `_OnCacheView`: the run's cache
    with `build` / `build_lazy` refused by name (they re-key every factor
    on another dt); `refactor_parametric` is kind-checked in C++. What
    cannot be made safe and is not: a hook that writes through the
    caller's builder (`b.pool.update_*`) changes a builder no run reads,
    on either route since add-immutable-circuit-ir; documented in the
    view's docstring, not detectable without a pool fingerprint. Options
    are now validated before the hooks run (a refused option no longer
    calls the user's hook first). Oracle: every simulate() result of
    test_mmc_thevenin, test_mmc_arm_builder,
    test_bidirectional_electrothermal and the MMC example (26) is
    byte-identical to 656b894f (`.scratch/hooks_bitid.py`); golden /
    stateful / engine / option compares identical.
    Still off the object: `sweep()`'s recipe replay (`_PWL_RECIPE`) and
    the exact engine (16).

21. **Two behaviour notes of 19/20 (from the fresh-context review).**
    (a) simulate()'s broad handlers around `builder.initial_state()` and
    `state_var_names()` now catch `AttributeError` only, and the one
    around `_switch_census` `(AttributeError, TypeError)`: they existed
    for builder stand-ins (test doubles, older binaries). A real builder
    raising anything else there is a kernel error and now reaches the
    caller instead of being swallowed (the ICs one would have run the
    circuit from zero). Only the `_run_circuit is None` stand-in path can
    see the difference; no in-repo caller does.
    (b) On the paths that still build their own cache (`_kernel_run`,
    `sweep()`'s replay, stand-ins), the cache is now built after the mmc
    refusals and the count checks (it used to be built before them): a
    run with a bad option AND a circuit whose cache cannot be built now
    reports the option. The view also refuses `share_symbolic`.

22. **Hand-off (2026-11-18, 22:15).** Done this session (all Python;
    no C++ touched since 9fbbdb2a): 19 (gate under base), 20 (mmc_arms /
    on_cache / chain on the object), 21, Stage A 1.4 and the simulate()
    part of 1.3. LOCAL, not pushed: 73adff8b, 0a45b8a4, 832e11d5,
    ae449172 and this commit, on 3835a4ac. The full ctest build
    (`zsh .scratch/build.sh ctest`, log `.scratch/build-ctest.log`) stood
    at 28/146 under a load average of ~80; the Python verification set
    (`zsh .scratch/dod.sh fin2`, logs `.scratch/*_fin2.log`) was running.
    Next agent: when both are green, push the tip; if the build is still
    running, let it finish (it is incremental). Then, in order: (a) the
    exact engine on the object (16) once `feat/core-a-complementarity` is
    integrated; (b) `sweep()`'s native replay (`_run_pwl_points` with a
    `SimulationOptions` recipe, sweep.py ~359-386) onto `RunOptions` /
    the object, a C++ binding job; (c) Stage A 1.3's one-level-down
    broad-handler scan; (d) Stage A 2-7 only if the owner still wants a
    Python plan (DECISIONS 6: the kernel resolves options; the route,
    compose and finalize still live in simulate()).

23. **Stage A built (tasks 1-7): plan -> engine table -> finalize.** DECISIONS 6
    said Stage A would not be built; the owner asked for it on 2026-11-19, as a
    Python-only refactor with an absolute invariant (results, warnings,
    refusals and result attributes unchanged). `simulate()` is now
    `plan = _plan_run(**locals())` and `_finalize(_ENGINES[plan.engine](plan),
    plan)`; the pieces, in `__init__.py`: `_plan_run` (keywords, `solver=`,
    `LoadedCircuit`, validation, preflight via `_apply_ties`, `resume_from`,
    the initial state, `_compose`, the sensitivity refusals, `_route`,
    `_compose_for`, the engine's kernel options), the runners `_run_pwl` /
    `_run_trbdf2` / `_run_dsed` (they read only `plan`), `_finalize(out,
    plan)`; the dataclasses in `python/pulsim/_plan.py`.
    - **Name.** The plan class is `SimPlan`, not D1's `RunPlan`: `pulsim.RunPlan`
      is public (the circuit record `result.plan` carries) and is not renamed;
      the plan holds it as `circuit_plan`.
    - **`_finalize(out, plan)`, not `(result, plan)`.** The pwl run composes
      part of its drive at run time (below), so a runner returns a
      `RunOutcome` (result + the drive as finally composed + the kernel
      object's builder).
    - **Warnings keep their location.** Every `warnings.warn` that moved one
      frame deeper got `stacklevel + 1` (each helper is called from one
      depth only), so a warning still names the caller's line, and Python's
      once-per-location filter counts the same warnings. Oracle: a pytest
      plugin records (test, category, message, file, line outside pulsim)
      for the whole suite; identical to the base after every group. Two
      tests call `_trbdf2_warn_stats` directly: their warnings now point one
      pytest frame deeper (inside pytest either way).
    - **The plan is frozen, not pure.** It holds mutable objects (builder,
      control plan, the user's callables), and planning keeps today's side
      effects in today's order: `_apply_ties` regularises a builder stand-in
      in place (test doubles); pwl's `for_fixed_step` sets the control
      graphs' solver dt / horizon. Purity comes with
      `add-immutable-circuit-ir` / Stage B.
    - **Position-dependent blocks found (kept per engine, bit for bit; for the
      number-changing release):**
      1. *No `switch_fn` (`_undriven`, one function).* trbdf2 decides while
         planning (all open, silent). pwl decides in the run AFTER the cache
         hooks: the `on_cache` hook (user code), the MMC driver and the
         `Simulation` constructor can raise or warn before the
         "ALL CLOSED" warning. dsed decides inside `run_dsed_from_builder`
         after `_require_supported`, the tick refusals and the cache build.
         Lifting either into the plan would reorder that warning against
         those refusals. (Open question 1 / task 11.3 decides the policy.)
      2. *`mmc_arms` / `on_cache` (pwl).* Their composition needs the run's
         own cache (the `Simulation` object's), so it stays in `_run_pwl`;
         `plan.drive` holds the drive before them, `RunOutcome` after.
      3. *`progress` and `live_stream` (pwl).* Wired in the run after (1):
         `live_stream.attach()` is a side effect on the caller's object and
         can supply `should_continue`.
      4. *`resume_from` continues time on pwl / trbdf2 only.* It sat after
         the dsed return; dsed refuses `resume_from` earlier, so the planner
         guards it with `engine != "dsed"` (same behaviour, now explicit).
      5. *`mode_selection='auto'`.* pwl / trbdf2: resolved by `_route` (a
         warning for `max_event_iterations=0`, ordered before the integrator
         and table refusals, so `_route` keeps it inside). dsed: resolved in
         the dispatcher, except with `sensitivities` (planned).
      6. *`on_impulse='error'`.* pwl's kernel refuses at the event; dsed
         refuses in `_finalize` after the run; trbdf2 never records.
      7. *Option value checks.* `store_every < 1`, `max_event_iterations < 0`,
         `max_dt_halvings < 0` and the six `mmc_arms` combinations are
         checked in Python for pwl only (the table / kernel refuse them on
         the other engines). Since feat/core-branch-form-coverage the same
         pwl block first resolves the tri-state defaults of
         `discontinuity_restart` and the step split (`_plan.DEFAULT_*`, False
         today): the rule's default is not applied under `mmc_arms` (one
         `route_reasons` entry says so; an explicit True is still refused),
         and the split's default is on only where the rule resolved on
         (update-number-changing-release blockers 1.5 / 1.8).
      8. *`sweep()`'s recipe.* `_PWL_RECIPE` is now read once while planning
         (`plan.recipe`) instead of again in the run; equal unless a hook
         sets that context variable mid-call.
      9. *The exact engine's loss port (batch 14k, merged at fe75ee01).* A
         circuit with `set_device_losses` on engine 'dsed' refuses
         `integrator='bdf2'` and `t_start != 0` by name (feature
         `device_losses`) inside `run_dsed_from_builder`: below the
         dispatch, after the diode census's own `bdf2` refusal and before
         `_require_supported`; pwl and trbdf2 honour both. Kept there (an
         upstream hunk in the dispatcher, not in the old `simulate()` body;
         lifting it into `_route` would order it before the census refusal
         and the tick checks). A candidate for the plan when the exact
         engine runs through it (16).
      10. *`resume_from` on the exact engine (add-exact-propagator-strategy
         sessions 24-25, merged after 061e4099).* Item 4's guard stays (dsed
         resumes on its own absolute clock, so `t_start` is not rewritten).
         In the plan, above the route: an invalid snapshot whose run named why
         it kept none (`invalid_reason`) is refused by name (feature
         `resume_from`), and an exact-engine snapshot (controller kind 3) is
         refused by name on pwl / trbdf2. `_route` no longer lists
         `resume_from` among dsed's refused keywords (its table row is every
         engine). Engine-specific, in `_compose_for`'s dsed branch:
         `sensitivities` with `resume_from`, a tick grid the snapshot does not
         carry, and the `t_start` tick refusal skipped on a resume; in
         `run_dsed_from_builder`: bdf2, a relabelling `t_start`, recorder
         state, `initial_state`, an explicit `integrator='auto'` that would not
         reach the event-driven integrator, stateful-device histories; in
         `_run_dsed`: the snapshot's tick grid passed down and the
         controllers' state saved into the result's snapshot.
      11. *The exact engine's in-loop recorders, 'auto' snapshot, loss port
         with t_start (add-exact-propagator-strategy session 26).* Engine-
         specific, in `run_dsed_from_builder` (rk45 branch): the recorder feed
         is attached through `_recorders.ExactFeedSlot` (`attach_loop_feed(slot,
         "dsed")`; a fresh `t_start` run keeps the stored pass); the field-18
         refusal is gone (the feed continues the record; `simulate(recorders=)`
         refuses a snapshot's recorder state when this run cannot feed it); an
         'auto'-written snapshot (payload tag 101) resumes on 'auto' only (an
         explicit other integrator or control ticks are refused by name);
         `_mark_no_snapshot` no longer names 'auto'; the loss port's `t_start`
         refusal is gone (`t_offset` shifts its event times).
    - **6.2 left open.** `_finalize` already runs `_auto_check_voltage_sanity`
      on dsed results today (the arrays path); what differs is pwl's own
      inline check (`find_implausible_voltage` on the result, one broad
      handler). Unifying them changes warnings: for the release.
    - **Not done:** `plan.undriven` (D1's tuple of undriven switches; (1)
      decides it in the run), C++ `Simulation` for the exact engine (16).
    - **Fresh-context review (one reviewer, read-only, diff from f220a1da).**
      No engine sees a different option value, initial state, drive or
      refusal, or a different order. Fixed: `plan.features` listed
      `recorders` (never reachable: `recorders=` runs through
      `simulate_with_recorders` first) and raised `KeyError` on a plan built
      without `resolved_from`. Left, recorded:
      1. With `sensitivities=` (dsed), the near-miss warning of
         `sensitivity._compute` (`stacklevel=3`, public module, not moved)
         now names `_run_dsed`'s line instead of `simulate`'s. Both are in
         `__init__.py`, one call site each, so the user's view and the
         once-per-location count are unchanged. A warning the C++ engine
         raises during a run is likewise attributed to the runner's line.
      2. `_warn_newton_roundoff` via `_trbdf2_warn_stats` names `simulate`'s
         frame, not the user's (as before: an existing off-by-one, kept).
      3. `resolved_from` reads `explicit` for an inert value that is not
         the default (`progress=None`, `closed_loops=[]`), and the
         `yaml` / `solver` marks compare by identity (a YAML `t_start: 0.0`
         reads `yaml`). Metadata only; no engine reads it.
    - **Overhead (information; task 7.2 asks "not above 0.2").** Bytecodes
      per call (`.scratch/opcount.py`, 2-switch buck): pwl 2480 -> 3538,
      trbdf2 3317 -> 4351; ~670 of that is the frozen dataclasses'
      `__init__`. Best of 300, RC, 2 steps: pwl 34.4 -> 49.1 us, trbdf2
      194 -> 210 us. 7.2 stays open on that clause.

24. **Hand-off (2026-11-19, Stage A).** Built: tasks 2, 3, 4, 5, 6.1, 7 (one
    commit or more per group on `feat/core-p3-simulation-object`, each a
    pure move named in its message). Oracles after every group against the
    f220a1da base: full pytest (5705 -> 5710 passed, the new tests), the
    suite's warning set (414 = 414, per test, category, message, file, user
    line), golden / stateful / engines / options / hooks byte compares,
    `pwl_identity` 390/390, `trbdf2_identity` 371/371, `-m reference` 382,
    `reference_status` (file unchanged), examples 137, ruff, openspec,
    stubs. Open, for the number-changing release: 6.2 (one voltage check),
    11.3 (one undriven policy: item 1 of 23), the per-engine blocks 2-8
    of 23, `plan.undriven`, 7.2's overhead clause. `simulate()` is 375
    lines (306 docstring, 69 code; was 1846). Not run: `ctest` (no C++
    changed), mkdocs build.

25. **Task 7.2's overhead clause: most of Stage A's cost removed, not all.**
    Measured per function (`sys.settrace` opcode events, warm call, 2-switch
    buck, same `.so`; `.scratch/opcount_all.py`): pre-Stage A (214947c1)
    pwl 2551 / trbdf2 3380 / dsed 3993 bytecodes; Stage A as landed
    (b348f331) 3609 / 4413 / 5051; now 2834 / 3683 / 4348. Where Stage A's
    +1058 (pwl) went: the frozen dataclasses' `__init__` (one
    `object.__setattr__` per field) +502; `**locals()` into `_plan_run`
    and `dict(locals())` for `resolved_from` (1 bytecode each, but ~1 us
    each: `locals()` of a 50-parameter frame); the 45-keyword `_route`
    call and the 22-keyword plan constructor (CPython compiles a call with
    more than 30 stack slots to a dict built name by name: ~2 bytecodes per
    keyword); the runners' and `_finalize`'s prologues re-reading the plan;
    the eager `plan.diagnostics` tuple.
    - **Removed:** the records are `typing.NamedTuple`s built with
      `tuple.__new__` and every field in order (`_planning.new`): no Python
      frame, no keyword dict. Immutability: a tuple has no `__dict__`, a
      field cannot be set or deleted, `AttributeError` (frozen dataclasses
      raised `FrozenInstanceError`, a subclass of it). `simulate()` passes
      its arguments to `_plan_run`, and `_plan_run` to `_route`,
      positionally as two tuples of <= 30 names concatenated and starred;
      `resolved_from` keeps the values as a tuple (captured at call time,
      compared lazily). `plan.diagnostics` is a property over
      `route_reasons` (a tuple of strings). Fields read on rare paths are
      read where used in `_run_pwl` / `_finalize`. On the path but not
      Stage A's: six function-level `from . import` hoisted to module level
      (each costs `importlib._bootstrap.parent` / `_handle_fromlist`),
      `np.any(a != 0)` -> `(a != 0).any()` (numpy's Python wrapper).
      Checks: `test_positional_plan_calls_follow_the_signatures` (each name
      list equals the callee's parameters, by `ast`),
      `test_every_positional_record_has_its_arity`,
      `test_plan_fields_and_immutability`, `test_run_outcome_fields`.
    - **What remains (+283 pwl, +303 trbdf2, +355 dsed):** the cost of the
      layout itself: five calls instead of one body, the plan built and
      read back field by field. Removing it means merging the functions
      back; not done. Wall time (information only; RC, 2 steps, best of
      42 x 9 x 300 calls, the three trees interleaved, same `.so`): pwl
      43.1 -> 57.4 -> 43.3 us, trbdf2 236.4 -> 255.2 -> 220.9 us: the wall
      regression is gone within the machine's noise; the bytecodes that
      remain are cheap ones.
    - **Gate:** `python/tests/test_simulate_overhead.py`, a LOCAL-only
      ratchet (CPython 3.13 on darwin, skipped when `CI` is set): pwl <=
      2890, trbdf2 <= 3760, counted in a fresh interpreter (inside the
      suite, state earlier tests leave behind adds ~190). Lower the
      ceilings when the count drops.
    - Oracles vs b348f331 (clean `git archive` extract, same `.so`): pytest
      5784 -> 5788 passed (new tests; one skip fewer: `test_no_home_paths`
      skips outside a git checkout), warning set 418 = 418, goldens /
      stateful / engines / options / hooks identical, `pwl_identity`
      390/390, `trbdf2_identity` 370/371 (the one difference is the
      extract's absolute path inside a YAML loader message; equal with the
      path substituted).
    - **Merged fe75ee01 (batch 14k) after this, no conflict** (the PR head
      did not touch `__init__.py`). Oracles of the merged tip vs a clean
      `git archive` extract of fe75ee01 with its `.so`: pytest 5819 -> 5831
      passed (this branch's tests), warning set 425 = 425, goldens /
      stateful / engines / options / hooks identical, `pwl_identity`
      390/390, `trbdf2_identity` 370/371 (the same extract-path artifact),
      `-m reference` 383. The loss-port refusals are item 9 of 23.
      Bytecodes on the merged tip: 2834 / 3683 / 4374 (dsed: the PR head's
      own +26, 3993 -> 4019).
