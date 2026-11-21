# Decisions taken while implementing

The proposal has no Open Questions section. These are the choices the
implementation had to make; each says how to reverse it.

## 1. Native schedules cannot reach a process pool

- **Question:** the spec scenario "Backend can be forced" runs
  `backend="process"` with a native schedule.
- **Found:** `SwitchSchedule`, `NativePwm2Switch` and
  `NativeMultiMaskPwm` are pybind11 objects without pickle support
  (`SwitchSchedule` wraps a `std::function`), so no native schedule can
  be sent to a worker process. This was already true of
  `parallel=True` before this change: a sweep driven by
  `make_pwm_switch_fn` never worked on the process pool.
- **Option taken:** the forced-process test uses a switchless RC
  circuit (nothing to pickle but module-level functions); the
  docstring and `docs/api-reference.md` state the limit; the pickling
  `TypeError` reaches the caller unchanged.
- **Why:** pickling a schedule needs a C++ change outside this
  proposal; `"auto"` never sends a native run to processes anyway.
- **Reverse:** add `py::pickle` to the three native schedule classes
  (the two `Native*` classes have plain fields; `SwitchSchedule` would
  need its builder arguments recorded) and switch the test back to the
  buck.

## 2. The process backend still raises on a failing job

- **Option taken:** serial and thread runs record `(index, message)`
  in `failed`; the process run propagates the first exception, as the
  proposal says ("The process backend keeps its current behaviour").
- **Reverse:** wrap `_run_one_simulation` in the worker with the same
  capture `run_captured` uses.

## 3. Process results are now in job order

- **Found:** the old process path did `enumerate(pool.imap_unordered(...))`
  and sorted on the enumeration index, which is the ARRIVAL order, so
  results could be attached to the wrong parameter row.
- **Option taken:** `pool.imap` (ordered). Required by "Results SHALL
  be ordered by job index" on every backend.

## 4. `pulsim.scope` is protected by a module subclass

- **Found:** with the eager import gone, the first
  `import pulsim.scope` / `from pulsim.scope import LiveScope` happens
  after `__init__` has finished, so the import system's
  `setattr(pulsim, "scope", <module>)` can no longer be undone by a
  later line of `__init__`.
- **Option taken:** `sys.modules["pulsim"].__class__ = _PulsimModule`,
  a `ModuleType` subclass whose `__setattr__` ignores a module assigned
  to `scope` (the pattern the Python docs give for customising module
  attribute access). Everything else is assigned normally.
- **Reverse:** delete `_PulsimModule` and restore the eager import
  (and its import cost).

## 5. Runs without a `switch_fn` report `"switch_fn"`

- Every engine installs a Python default mask when no `switch_fn` is
  given (`pwl`: all closed; `trbdf2`: all open; `dsed`: all closed in
  `_dsed_dispatch`). The report names it, so `"auto"` keeps such
  sweeps on processes, as the proposal anticipates until
  `update-fixed-step-loop-bookkeeping` lands a native default.

## 6. What counts as native

- A `switch_fn` is native when it is a `NativePwm2Switch`,
  `NativeMultiMaskPwm`, or a `SwitchSchedule` with `has_edges` (a
  `make_combined_switch_fn` result that holds a Python callable has no
  edge oracle, and is reported). Every other non-`None` hook
  (`b_extra_fn`, `step_observer`, `should_continue`) is reported as
  Python, even when it wraps a C++ object: conservative, it can only
  send a run to processes. The fixed engine's C++ chain fast path
  (`_cxx_chain`) is not reported as a `step_observer`.

## 7. `"auto"` pickles after the first job

- Question: job 0 runs in the calling thread before the rest are sent
  to the pool, so a callable that caches an unpicklable object on its
  first call (a `SwitchStateMask`) stops pickling. Snapshot the
  arguments before job 0, or document?
- Option: document it in the `sweep` docstring (`__getstate__`).
- Why: a snapshot pickles every argument up front, which fails for
  native schedules on the thread path and costs a copy per job; the
  measurement script hit the case and a three-line `__getstate__`
  fixed it.
- Reverse: `pickle.dumps` the job arguments before job 0 when the
  backend is `"auto"` and send the snapshot to the pool.
