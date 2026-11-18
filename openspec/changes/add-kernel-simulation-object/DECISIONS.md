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
