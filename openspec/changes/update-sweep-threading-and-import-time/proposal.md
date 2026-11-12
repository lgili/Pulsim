## Why

Two costs sit outside the kernel and are paid by every sweep and every
script. Core review 2026-11, `raw/perf.md` (throttled machine; trust the
ratios):

| Cost today | Measurement | Probe |
|---|---|---|
| 8-job buck sweep, sequential | 695 ms | `perf/p8_par.py` |
| Same on a thread pool | 2 threads 1.7–2.0×; 4 threads 3.0–3.1×; results identical to sequential | `perf/p8_par.py` |
| Same on `multiprocessing.Pool(4)` (what `parallel=True` does) | 18.9 s, 27× slower than sequential | `perf/p8_par.py` |
| Thread pool when `switch_fn` is a Python callable | 0.21× of sequential | `perf/p8_par.py` |
| `import pulsim` | 10–15 s under this throttling, about half in pyqtgraph / PySide6 | `perf/p8b.py` |

A single `python -X importtime -c "import pulsim"` on this branch (same
throttling, one run, file cache not controlled) gave 14.0 s in total,
of which `pulsim.scope` was 9.6 s (pyqtgraph 9.3 s).

Where:

- `sweep` and `monte_carlo` use `multiprocessing.Pool` when
  `parallel=True` (`python/pulsim/sweep.py:185`, `:290`). Each worker
  process pays the import above and pickles its arguments. The kernel
  already releases the GIL around a run (`python/bindings.cpp:2694`) and
  its solve workspaces are per thread, so threads scale whenever the
  step loop makes no Python call.
- `pulsim/__init__.py:450` imports `LiveScope` from `pulsim.scope`,
  which imports pyqtgraph and its Qt binding at module load
  (`python/pulsim/scope.py:53-54`), for every user, with or without a
  GUI.

## What Changes

- **Thread backend for sweeps.** `sweep` and `monte_carlo` gain
  `backend="auto" | "thread" | "process"` (default `"auto"`), used when
  `parallel=True`.
  - `"auto"` runs the first job in the calling thread, reads from its
    result whether the step loop installed any Python callable, and
    then runs the remaining jobs on a thread pool (none installed) or
    on the process pool (some installed).
  - Results are collected by job index; KPIs are identical to the
    sequential run.
  - The thread backend isolates failures per job, like the sequential
    path. The process backend keeps its current behaviour.
  - `SweepResult.backend` records what ran.
  - **BREAKING (execution model):** with `parallel=True`,
    `builder_factory` and `kpi_fn` may now run in threads of one
    interpreter instead of separate processes. `backend="process"`
    restores the previous model.
- **Step-loop callback report.** `simulate()` attaches
  `res.step_loop_python_callbacks`, a tuple naming the Python callables
  it installed in the step loop (`"switch_fn"`, `"b_extra_fn"`,
  `"step_observer"`, `"should_continue"`, `"closed_loops"`, …); empty
  when the loop is fully native.
- **Lazy scope import.** `import pulsim` no longer imports
  `pulsim.scope`. `pulsim.LiveScope` is resolved on first access
  through the module's existing `__getattr__`; `pulsim.scope` stays the
  plotting function before and after that access; availability is
  tested with `importlib.util.find_spec`, without importing.

Not in this change: a native default schedule for runs without
`switch_fn` (`update-fixed-step-loop-bookkeeping`; until it lands such
runs report `"switch_fn"` and stay on processes); SIMD batching of
scenarios; other import costs (`pulsim.c_block` was 1.1 s in the same
`importtime` run).

## Impact

- Affected specs: `parameter-sweep`, `python-bindings`.
- Affected code: `python/pulsim/sweep.py`, `python/pulsim/__init__.py`
  (`simulate()` engine branches, the `LiveScope` import and
  `__getattr__`), stubs, `docs/`.
- Numerical results: unchanged. Sweep KPIs are identical to sequential
  execution on every backend.
- No dependency on the reference harness; no kernel change.
- Expected speed-up: about 3× on 4 threads for native-callback sweeps
  (measured ratio between two existing code paths, not a prototype of
  this change) with no process start-up cost; `import pulsim` at about
  half its current time (**estimate** from the import split).
- The existing `parameter-sweep` spec names `serial` / `joblib` /
  `dask` executors and a `pulsim.sweep(circuit_factory, parameters,
  metrics, executor, …)` signature that the code does not have. This
  change adds a requirement in the terms of the shipped API and leaves
  that drift to the owner.
