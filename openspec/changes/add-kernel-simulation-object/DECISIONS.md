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
    ~line 201, "the caller's builder is not touched"), so it can move
    first; the `on_cache` hooks are user code and the shipped ones are
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
