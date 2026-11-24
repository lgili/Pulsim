## Context

Pulsim has four transient drivers: the fixed-step trapezoidal engine
(`engine='pwl'`, `solver/run_transient.hpp`), the variable-step TR-BDF2
engine (`'trbdf2'`, `solver/trbdf2_transient.hpp`), the event-driven
exact-propagator engine (`'dsed'`, `dsed/`), and the kernel-only
`run_transient_bdf1` (`solver/bdf1.hpp`). The device pool is a 20-way
variant (`DevicePool::StoredKind`, `pwl/device_pool.hpp:59-80`).

"Can engine E run device kind K" is answered in four unrelated ways:

| Site | Rule |
|---|---|
| `_dsed_dispatch.py:578`, `bdf1.hpp:297` | `has_nonlinear_devices()`: 7 kinds listed by hand, `NonlinearCapacitor` missing |
| `trbdf2_transient.hpp:322-342` | `BranchKind::Nonlinear` on the graph, minus `NonlinearCapacitor` |
| `__init__.py:2286, 2677`, `bindings.cpp:2964` | the same hand list decides whether the Newton stamps exist at all |
| `_dsed_dispatch.py:686, 853` | whatever the LTI extractor happens to reject, as `NotImplementedError` |

Refusals are raised as `ValueError` (Python `raise`, and C++
`std::invalid_argument` through pybind11), except the two extractor
sites above. The repository's tests contain 196 `pytest.raises(ValueError`
and 8 `pytest.raises(NotImplementedError`.

Probe scripts: `/Users/lgili/pulsim_wt/core_review_2026-11/arch/`
(`matrix.py`, `probe3.py`, `probe4.py`, `probe5.py`) and, for the
multi-mask measurements made while writing this proposal,
`probes/d10_multimask.py` and `probes/d10b_narrow.py` in this change
folder. All run read-only against the built extension
(`PYTHONPATH=<worktree>/python python3 <script>`).

## Goals / Non-Goals

- Goals: one table; one exception type; the check inside the kernel
  drivers; a test that fails when a cell is silent; fix the four cells
  listed in proposal.md.
- Non-Goals:
  - Making any engine support a new device. A refusal is the fix.
  - Moving keyword resolution out of `simulate()`
    (`add-kernel-simulation-object`).
  - A `caps` member on each device type
    (`add-unified-device-interface` replaces the table with it).
  - Tick semantics (`fix-control-tick-semantics`).
  - `result.i()` for Newton devices, which raises `NotImplementedError`
    for a different reason (`add-unified-device-interface`).

## Decisions

### 1. The table

```cpp
// pwl/device_pool.hpp, next to StoredKind
enum class Engine : std::uint8_t { FixedTrap, TrBdf2, Dsed, Bdf1 };
enum class SolveClass : std::uint8_t {
    Linear, Source, Switch, PwlDiode, NewtonStatic, NewtonStateful };
struct KindCaps {
    std::string_view name;      // "NonlinearCapacitor"
    SolveClass       cls;
    std::uint8_t     engines;   // bit per Engine
};
inline constexpr std::array<KindCaps, 20> kKindCaps = { ... };
static_assert(kKindCaps.size() == std::variant_size_v<Entry>);
```

Derived queries, the only ones callers may use:

- `pool.requires_newton()` — any `NewtonStatic` or `NewtonStateful`.
- `pool.requires_static_refresh()` — any `NewtonStatic`.
- `require_supported(graph, pool, Engine)` — throws `UnsupportedError`
  for the first branch whose kind lacks the engine bit.
- `has_nonlinear_devices()` stays as an alias of `requires_newton()`
  (bound in Python, used by three tests).

Circuit-level limits that are not a property of one kind stay where they
are but raise the typed refusal: coupled inductors on `dsed`
(`pool.transformer_couplings()` non-empty), PWL diodes under
`integrator='bdf2'`, builder initial conditions on `dsed`.

Alternatives considered:
- *A `caps` constant on each `models::X`* — the right end state, but it
  touches 20 model headers and is what `add-unified-device-interface`
  does. Rejected here: Phase 0 must be small.
- *Use `BranchKind::Nonlinear` from the graph* — covers the Newton kinds
  but cannot express "PWL diode on bdf1" or "coupled inductors on dsed",
  and it is set by the builder, a second place to forget.
- *Add `NonlinearCapacitor` to the hand list* — fixes one cell and leaves
  the mechanism that produced it.

### 2. Initial expected table (device kind × engine)

OK = runs and agrees; REF = refused; "n/c" = not probed in the audit:
the implementation runs the cell and the table records the outcome (OK
only if it agrees with the reference within the tolerance of decision 5).

| StoredKind | Class | pwl | trbdf2 | dsed | bdf1 |
|---|---|---|---|---|---|
| Resistor, Capacitor, Inductor | Linear | OK | OK | OK | OK |
| VoltageSource | Source | OK | OK | OK | OK |
| CurrentSource | Source | OK | OK | OK | n/c |
| PWMVoltageSource | Source | OK | OK | OK | n/c |
| SineVoltageSource, PulseVoltageSource | Source | OK | OK | n/c | n/c |
| VCVS, IdealTransformer | Linear | OK | OK | OK | n/c |
| Switch | Switch | OK | OK | OK | OK |
| Diode (PWL) | PwlDiode | OK | OK | OK (RK45 scheduler; `bdf2` REF) | REF |
| NonlinearDiode, ShockleyDiode | NewtonStatic | OK | OK | REF | REF |
| MosfetLevel1, IgbtLevel1 | NewtonStatic | OK | OK | REF | REF |
| SaturableInductor | NewtonStateful | OK | OK | REF | REF |
| LauritzenDiode, PmsmMna | NewtonStateful | OK | OK | REF | REF |
| NonlinearCapacitor | NewtonStateful | OK | OK | **REF (was silent)** | **REF (was silent)** |
| coupled inductors (`add_transformer`) | circuit-level | OK | OK | REF | n/c |

`NonlinearDiode` on `pwl` aborts on a 20 V clamp today because of its
law, not of the engine; `fix-nonlinear-diode-passivity` owns that.

### 3. Initial expected table (feature × engine)

| Feature | pwl | trbdf2 | dsed |
|---|---|---|---|
| Python `switch_fn` | OK | OK | OK |
| `SwitchSchedule`, `NativePwm2Switch` | OK | OK | OK |
| `NativeMultiMaskPwm` | OK | **OK (was sampled)** | OK |
| `enable_nonlinear_refresh=False` with Newton devices | **REF (was silent)** | REF | REF |
| C blocks | OK | REF | **REF (was a warning)** |
| `step_observer`, no cadence | OK | REF | REF |
| `step_observer` + `controller_period` | xfail → `fix-control-tick-semantics` | OK | REF |
| one `ClosedLoop` | OK | OK | REF |
| two `ClosedLoop`s; loops at two rates | OK | xfail → `fix-control-tick-semantics` | REF |
| `b_extra_fn` | OK | OK | OK |
| builder initial conditions | OK | OK | REF |
| `start_from_dc_op`, `store_every`, `live_stream`, `max_dt_halvings`, `enable_substep_state_correction`, inductor guards, `mmc_arms` | OK | REF | REF |
| `on_cache` | OK | REF | n/c |
| `should_continue` | OK | OK | REF |
| `progress` | OK | REF | n/c |
| `resume_from` | OK | OK | n/c |

An undriven controlled switch (all-closed with a warning on `pwl` and
`dsed`, refused on `trbdf2`) already satisfies the invariant and is left
to `add-kernel-simulation-object`.

### 4. The exception

- C++: `class UnsupportedError : public std::invalid_argument` with
  `engine()`, `feature()`, `device()`, `kind()` (empty string when not
  applicable), in `solver/unsupported.hpp`.
- Python: the class is created in `bindings.cpp` with base
  `PyExc_ValueError` and filled by an exception translator, the pattern
  already used for `SimulationAborted` (`bindings.cpp:112-130`). It is
  re-exported as `pulsim.UnsupportedError`; Python-side refusals raise
  the same class.
- The two sites that raise `NotImplementedError` today
  (`_dsed_dispatch.py:686, 853`) raise a private subclass that derives
  from both `UnsupportedError` and `NotImplementedError`, so
  `test_dsed_end_to_end.py:344` and any user `except NotImplementedError`
  keep working. The subclass is removed in the next major.

Alternatives considered:
- *`UnsupportedError(RuntimeError)`* — breaks every `except ValueError`
  and 196 test assertions.
- *One class deriving from both `ValueError` and `NotImplementedError`*
  — keeps both patterns but makes every refusal catchable by
  `except RuntimeError`, which is how callers catch convergence
  failures; that is the handler class that hid refusals in PulsimGUI.
- *Keep `ValueError` and add attributes by convention* — a caller cannot
  tell a refusal from a bad argument, which is what the GUI needs.

### 5. The conformance test

`python/tests/test_engine_conformance.py`, parametrised from
`pulsim.engine_capabilities()` (kinds) and from the feature table above
(kept in the test).

- **Device cell.** One small circuit per kind, source-driven, no gate
  edges, built so the device matters. Reference: `engine='pwl'` at a
  `dt` for which halving `dt` moves the observable by less than a
  quarter of the tolerance. Comparison at 20 fixed instants, each
  engine's trace linearly interpolated in time (never by index, never an
  unweighted mean). Tolerance: 1 % of the reference observable's span.
- **Presence control.** Each device circuit is also run on the reference
  engine without the device; the observable must move by at least 10 %
  of the span. This is what makes a deleted device visible: for the
  `Coss` circuit the two values are 1.02 V and 6.30 V.
- **Outcome.** Agreement, or `UnsupportedError` with `engine` equal to
  the engine asked for and `device` equal to the branch name. Any other
  exception type, and any disagreement, fails. The outcome must equal the
  table: a kind the table marks supported may not refuse.
- **Feature cell.** Same rule, with `feature` equal to the keyword name.
  Features with a countable effect assert the count (observer calls,
  gate events) rather than a waveform.
- **`bdf1`** is exercised through `pulsim._pulsim.run_transient_bdf1`.
- Cells owned by another change are `xfail(strict=True)` naming the
  change id, so they fail the suite the day they start passing.

### 6. Refresh ownership

`run_transient`, `run_transient_trbdf2` and
`compute_dc_op_with_strategy`: if `nl_refresh` is empty and
`pool.requires_static_refresh()`, use
`pwl::make_combined_diode_mosfet_refresh()`. The stateful wrappers
(Coss, Lauritzen, IGBT tail, PMSM, saturable inductor) are already built
in the kernel (`run_transient.hpp:471-600`) and must keep capturing the
effective chain. The binding keyword becomes `Optional[bool]`:
`None` = kernel decides; `True` = unchanged; `False` with Newton devices
= `UnsupportedError` on the transient drivers.

The DC operating point keeps its documented opt-out
(`docs/gotchas.md:63`, `test_dc_operating_point.py:98`): there `False`
returns the linear network on purpose.

### 7. Switch-function adapter

One helper in the binding layer returns `{SwitchScheduleFn,
next_edge_fn}` for `SwitchSchedule`, `NativePwm2Switch`,
`NativeMultiMaskPwm` or a Python callable, and replaces the detection at
`bindings.cpp:2598-2636`, `bindings.cpp:4016-4047` and
`dsed_bindings.cpp:749-815`.

## Risks / Trade-offs

- A circuit that ran (wrongly) now refuses → the CHANGELOG lists the
  three cases with the engine to use instead.
- `requires_newton()` now includes `NonlinearCapacitor`; three Python
  sites used the old list to decide the refresh. They stop deciding
  (decision 6), so a Coss-only circuit keeps its current path. Gate: the
  Coss-only `pwl` and `trbdf2` results are bit-identical before and
  after.
- The reference engine is first order at gate edges, so device cells
  avoid gate edges; switch and schedule behaviour is covered by the
  feature cells with counts.
- Table and reality can drift for "n/c" cells → the test runs every
  cell, so a wrong entry fails.

## Migration Plan

1. Table, exception and translator, with the old guards rewritten on top
   (no behaviour change except `NonlinearCapacitor`).
2. Kernel-side checks and refresh ownership.
3. Adapter and C-block refusal.
4. Conformance test, docs, CHANGELOG.

Rollback is per step; step 1 alone already fixes the two silent kinds.

## Open Questions

1. `UnsupportedError` base: `ValueError` plus a private
   `NotImplementedError` bridge at two sites (proposed), or one class
   deriving from both?
2. `enable_nonlinear_refresh=False` on the DC operating point: keep it
   as the documented "linear network" request (proposed), rename it, or
   refuse it too?
3. `run_transient_bdf1`: give it a column in the public table
   (proposed, since it is bound), or hide it until
   `refactor-unified-transient-stepper` retires it?
4. Tolerance of a device cell: 1 % of span with a 10 % presence control
   (proposed). Tighter values belong to
   `restore-reference-benchmark-harness`; confirm the split.
5. Should `pulsim.engine_capabilities()` be public API in 2.x, or
   private until the device interface replaces the table?
