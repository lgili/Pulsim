Throttle every run on the shared Mac: `nice -n 15 taskpolicy -b`, one
process at a time (the process-pool probe starts four interpreters;
run it alone).

Thresholds below are targets from the review's measurements. A missed
threshold is reported to the owner; it is not waived.

## 1. Baseline (before any change)
- [x] 1.1 (re-measured through `pulsim.sweep`, numbers in CHANGELOG; the raw `ThreadPool` / `Pool(4)` table was not re-run) Record `perf/p8_par.py`: sequential, 2 and 4 threads, `multiprocessing.Pool(4)`, threads with a Python `switch_fn` (expected: 695 ms; 1.7–2.0×; 3.0–3.1×; 18.9 s; 0.21×)
- [x] 1.2 Record `import pulsim` with `perf/p8b.py` (best of 5, warm file cache) and the split with `python -X importtime -c "import pulsim"`; note the cumulative time of `pulsim.scope`
- [x] 1.3 Record `"pyqtgraph" in sys.modules` after `import pulsim` (expected: `True` where pyqtgraph is installed)

## 2. Step-loop callback report
- [x] 2.1 List every way `simulate()` puts a Python callable in the step loop, per engine branch of the dispatch: `switch_fn`, `b_extra_fn`, `step_observer`, `should_continue`, closed loops, control blocks, MMC arms, auto observers
- [x] 2.2 Attach `res.step_loop_python_callbacks` in every branch (the three early returns of the dispatch included); a branch that cannot tell reports `("unknown",)`
- [x] 2.3 Tests: native PWM schedule → `()`; Python lambda → `("switch_fn",)`; a `step_observer` → contains `"step_observer"`; one test per engine branch

## 3. Thread backend
- [x] 3.1 `backend="auto" | "thread" | "process"` on `sweep` and `monte_carlo`; ignored when `parallel=False`
- [x] 3.2 `"auto"`: first job in the calling thread, then thread pool or process pool for the rest according to its report
- [x] 3.3 Thread backend: results by job index; per-job failure capture into `SweepResult.failed` as on the sequential path
- [x] 3.4 `SweepResult.backend`
- [x] 3.5 Tests (deterministic, no timing): native schedule + `parallel=True` → `backend == "thread"` and KPI arrays `np.array_equal` to `parallel=False`; Python `switch_fn` → `backend == "process"`; forced `"thread"` and `"process"` honoured; a failing job is reported and the others complete; `monte_carlo` with a seed gives identical samples and KPIs on every backend
- [x] 3.6 Test: an unpicklable `builder_factory` (a closure) works on the thread backend

## 4. Lazy scope import
- [x] 4.1 Remove the eager `from .scope import LiveScope`; resolve `LiveScope` in the module `__getattr__`; `__all__` membership from `importlib.util.find_spec("pyqtgraph")`
- [x] 4.2 Keep `pulsim.scope` bound to the plotting function after `LiveScope` is first resolved (the submodule import rebinds the attribute)
- [x] 4.3 Tests in a subprocess: after `import pulsim`, `"pyqtgraph"` and `"pulsim.scope"` are not in `sys.modules`; `pulsim.LiveScope` resolves when pyqtgraph is installed; without it `hasattr(pulsim, "LiveScope")` is `False` and the `AttributeError` names pyqtgraph; `callable(pulsim.scope)` before and after; `from pulsim.scope import LiveScope` still works
- [x] 4.4 Check the GUI entry points and examples that use `LiveScope` still start

## 5. Docs
- [x] 5.1 CHANGELOG: sweep backend and the execution-model note; import time
- [x] 5.2 `docs/performance-tuning.md` and the sweep section of the API reference: when threads are used, `backend="process"`, thread-safety expectation for `kpi_fn` and `builder_factory`
- [x] 5.3 `AI_GUIDE.md` / `docs/ai-guide.md` if they describe `parallel=True`; stubs for `backend`, `SweepResult.backend`, `res.step_loop_python_callbacks`

## 6. Verification and acceptance
- [x] 6.1 `ruff`, `pytest`, `ctest`, `openspec validate --all --strict`, stub check
- [x] 6.2 CI regression tests are the deterministic ones of 2.3, 3.5 and 4.3; no wall-clock assertion
- [ ] 6.3 (measured, not accepted: the shared Mac ran at load ~9 on 10 cores and the 8-job native ratio swung 0.41–1.48 × serial, best 0.53 ×; the Python-`switch_fn` sweep runs on processes as before) Re-run 1.1 through `pulsim.sweep(..., parallel=True, n_workers=4)`. Accept when the 8-job native sweep takes ≤ 0.5 × the sequential time and the Python-`switch_fn` sweep is not slower than its baseline
- [x] 6.4 Re-run 1.2. Accept when `import pulsim` takes ≤ 0.6 × baseline with a warm file cache and `pulsim.scope` no longer appears in the `importtime` output
