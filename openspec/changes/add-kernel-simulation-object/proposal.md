## Why

What a simulation *is* (which engine, from which state, with which gate
drive, observers, defaults and refusals) is decided inside one Python
function, in three branches that return early. Counted on this branch
(PY = `python/pulsim/__init__.py`, DD = `python/pulsim/_dsed_dispatch.py`,
`core/…` = `core/include/pulsim/…`):

- **`simulate()` is PY:1572-3006: 1435 lines, 40 parameters** (3
  positional, 37 keyword-only; `ast` count; the audit's "46" could not
  be reproduced), **three direct `return`s** at PY:2116 (`dsed`),
  PY:2363 (`trbdf2`) and PY:3006 (`pwl`). The fixed-step tail alone is
  PY:2365-3006, 642 lines.
- **Decisions taken more than once, differently:**
  - loop / observer composition: PY:2215-2283 and PY:2420-2455;
  - the mask of an undriven controlled switch: all-OPEN at PY:2304,
    all-CLOSED with a warning at PY:2664-2673 and at DD:617-630;
  - `resume_from` validation: PY:1956-1975, `solver/run_transient.hpp:364-390`
    and `solver/trbdf2_transient.hpp:434-445`;
  - the default relative tolerance: 1e-4 (`solver/trbdf2_transient.hpp:119`),
    1e-5 (`solver/options.hpp:229`, PY:2296), 1e-6 (DD:71);
  - "not set": `None` for most keywords, `0` for
    `max_newton_iterations` (PY:1593), `0.0` after `or` for the Newton
    tolerances (PY:2329-2330).
- **The kernel has no entry point that is a simulation.** A C++ caller
  gets `run_transient` with 12 parameters
  (`solver/run_transient.hpp:284-298`) or `run_transient_trbdf2` with 13
  (`solver/trbdf2_transient.hpp:264-298`), must build the cache, and
  must know to pass the Newton refresh. Engine routing, initial-condition
  synthesis, preflight, the capability check, the default mask, loop
  composition, C-block hold, the MMC driver and the voltage-sanity check
  exist only in Python.
- **No error model.** `python/pulsim` raises `ValueError` 406 times and
  has 68 broad `except Exception` handlers (13 in PY); the kernel
  registers two exception types (`Cancelled`, `SimulationAborted`,
  `python/bindings.cpp:101-114`). Warnings are 44 `warnings.warn` call
  sites (14 in PY) plus attributes set on the result
  (`_preflight`, `_trbdf2_stats`, `_implausible_voltage`).
- **`SimulationOptions` carries four fields no engine reads**
  (`integrator`, `rtol`, `atol`, `dt_init`, `solver/options.hpp:225-237`).
- **`simulate()` mutates the caller's builder** (the preflight appends
  reference ties, PY:1904-1932).

Greps: `grep -rn "raise ValueError" python/pulsim --include="*.py" | wc -l`,
`grep -rnE "except Exception|except:\s*$|except BaseException" …`,
`grep -rn "warnings.warn(" …`.

Each duplication has produced a wrong answer or a crash:

| Symptom | Cause | Stop-gap |
|---|---|---|
| Builder ICs (`i0=5.0`) ignored on `trbdf2` and `dsed` until 2026-09-04: the run started at 0 A with no error | the synthesis sat below both early returns (owner note *simulate-engine-dispatch-order*) | moved above the dispatch |
| D5: two closed loops without `dt` raise `TypeError` at PY:2281 | second composition | `fix-control-tick-semantics` |
| D6, D7: multi-rate ticks wrong on trbdf2; `controller_period` ignored on pwl | composition per branch | `fix-control-tick-semantics` |
| Voltage-sanity check skipped on `dsed` | it runs after the `dsed` return | — |
| C blocks on `dsed`: a warning, then an open-loop run (PY:1996-2003) | refusal written per branch | `add-engine-conformance-matrix` |
| D4: the raw `run_transient` binding defaults to no Newton stamps | the stamps are an argument | `add-engine-conformance-matrix` |
| PulsimGUI turned 2.0's refusals into silent wrong results through 48 broad handlers | refusals are untyped `ValueError` / `RuntimeError` | GUI-side `BackendRefusal` |

Probes: `core_review_2026-11/arch/{matrix,probe4,probe6,probe7}.py`.

## What Changes

- **Stage A, Python only: `RunPlan`.** A pure function
  `_plan_run(circuit, **kwargs) -> RunPlan` is called once, above the
  dispatch. It resolves every keyword exactly once (explicit value,
  `solver=` bundle, YAML, default), validates, runs the capability
  check, routes the engine, synthesises the initial condition, composes
  gate drives and tick handlers by index, applies the undriven-switch
  policy and collects diagnostics. The three engine branches become
  `_run_dsed(plan)`, `_run_trbdf2(plan)`, `_run_pwl(plan)` and read
  nothing but the plan. `_finalize(result, plan)` runs for every engine.
- **Stage B, kernel: `Simulation`.** `sim::RunOptions` (nested,
  `std::optional`, one defaults table) and
  `sim::Simulation{Circuit, RunOptions}` with `plan()`, `run()`,
  `snapshot()`, `restore()`, `diagnostics()`, and `step()` once the
  unified stepper exists. The constructor does what `_plan_run` did.
  It is a facade over today's drivers first; the drivers behind it are
  replaced by `refactor-unified-transient-stepper` without changing the
  surface.
- **Typed errors.** `PulsimError` → `NetlistError`, `OptionsError`,
  `UnsupportedError{engine, feature, device, kind}` (the class
  `add-engine-conformance-matrix` introduces), `ConvergenceError`,
  `SingularError`; `Cancelled` and `SimulationAborted` join the
  hierarchy. In Python each class also derives from the builtin its
  call sites raise today, and in C++ each keeps its present `std` base,
  so existing handlers keep working.
- **Structured diagnostics.** `Diagnostic{severity, code, message,
  device, t}` on the plan and on the result. The Python layer emits one
  `warnings.warn` per diagnostic from one place.
- **`simulate()` becomes a wrapper** of about 100 lines:
  keywords → `RunOptions` → `Simulation` → result.
- **BREAKING:** `max_newton_iterations=0` no longer means "default"
  (use `None`; `0` warns for one release).
- **BREAKING:** the default tolerances come from one table; the raw
  `run_transient_trbdf2` binding's default `rtol` / `atol` change from
  1e-4 / 1e-6 to the `simulate()` values 1e-5 / 1e-8.
- **BREAKING:** an undriven controlled switch follows one policy on
  every engine (proposed: refuse, naming the switches; see Open
  Questions), replacing all-CLOSED on `pwl` / `dsed` and all-OPEN on
  `trbdf2`.
- **BREAKING:** the raw bindings `run_transient` and
  `run_transient_trbdf2` and the C++ free functions are deprecated for
  one release in favour of `Simulation`.
- **BREAKING:** the voltage-sanity check and its warning now run on
  `dsed` too.
- **Deleted at the end:** PY:1815-3006 except the wrapper,
  `_trbdf2_blockers`, `_validate_engine_kwargs`, the per-branch default
  masks, the four recorded-only option fields, `TrBdf2Options` and
  `SimulationOptions` as public types.

## Impact

- Affected specs: `simulation-kernel` (new), `python-bindings`.
- Affected code: new `core/include/pulsim/sim/`;
  `solver/{options,result,run_transient,trbdf2_transient,voltage_sanity}.hpp`;
  `pwl/preflight.hpp`; `builder/circuit_builder.hpp`;
  `python/bindings.cpp`, `python/dsed_bindings.cpp`;
  `python/pulsim/{__init__,_dsed_dispatch,control,c_block,mmc,steady_state,sweep,adaptive}.py`
  (callers of `simulate`); stubs; `AI_GUIDE.md`; docs; PulsimGUI's
  `backend_adapter.py` (downstream, informed).
- Depends on: `add-engine-conformance-matrix` (its feature × engine
  table is the capability check, and its `UnsupportedError` is the first
  member of the hierarchy).
- Uses, when present: `add-unified-event-queue` (composition registers
  gate sources and rates on the queue), `add-unified-device-interface`
  (the kernel builds the device stamps), `fix-control-tick-semantics`.
- Unblocks: `add-immutable-circuit-ir` (the `Circuit` argument),
  `refactor-unified-transient-stepper` (`step()`),
  `add-probes-and-recorders` (the result and `RecordOptions`).
- Performance constraints: planning runs once per simulation; the
  fixed overhead of `simulate()` (0.3–0.5 ms for n = 5, audit row 21)
  must not rise; nothing in the step loop changes in this change. A run
  with native gate sources and no observers must not take the GIL.
- Results: Stage A is a pure refactor (goldens bit-identical, same
  warnings). Stage B changes results only through the four **BREAKING**
  items above, each behind its own PR and migration note.
