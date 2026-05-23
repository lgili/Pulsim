## Why

Pulsim v2's transient solver runs at a single fixed `dt` for the whole simulation. That's fine for switching power converters where the fastest dynamics (the PWM period) are known a priori, but it's wasteful or inaccurate for circuits with:

* **Stiff regions** where dt must be tiny for stability but the system spends most of its time in a slow regime (e.g. settling tails, low-current freewheel, soft-switched resonant tanks).
* **Events** where a sub-step would resolve an intersection but the fixed dt overshoots (the existing `max_event_iterations` bisection helps, but with a true LTE estimate the solver can pick the right step in one shot).
* **Validation** against analytical solutions where the user wants to guarantee `||x_numerical − x_exact|| ≤ ε` without picking dt by hand.

The standard fix is **adaptive time-stepping with local truncation-error (LTE) estimation**. The most robust off-the-shelf scheme for second-order trapezoidal is **step doubling + Richardson extrapolation**: take one step of size `dt` (call the result `x_big`), take two steps of size `dt/2` (call the result `x_small`), and use `||x_small − x_big|| / 3` as the LTE estimate. Accept or reject the step based on `LTE / (atol + rtol·||x||)`, then re-scale dt for the next trial.

This proposal adds adaptive stepping to the kernel + a Python API.

## What Changes

* **New C++ header** `core/include/pulsim/v2/solver/adaptive_step.hpp` exposes:
  * `AdaptiveOptions` (atol, rtol, dt_min, dt_max, safety, growth_max).
  * `AdaptiveResult` (parallel `times` + `states` + `dt_history` + `n_accepted` + `n_rejected`).
  * `run_transient_adaptive(cache, graph, pool, t_start, t_end, dt_init, opts, switch_fn, b_extra_fn, step_observer)`.
* Internally the driver uses the existing cache's `lookup(mask, dt)` (which supports per-step dt rebuilding via the multi-dt cache feature already shipped) so no new sparse infrastructure is needed.
* **Python binding** `p.run_transient_adaptive(builder, t_start, t_end, dt_init=...)` returns the same `SimulationResult` shape, with an extra `dt_history` array.
* **Validation** — RL step response with R/L ≈ 10 µs and a long simulation (500 ms) demonstrates dt growing from 100 ns initial to ~50 µs at steady state (500× speedup), with `||x − analytic||_∞ < rtol·||x||`.

## Impact

* Affected specs: `pulsim-v2-solver` (new requirement: "Adaptive time-stepping").
* Affected code:
  * New header `core/include/pulsim/v2/solver/adaptive_step.hpp`.
  * New Python module `python/pulsim/v2_adaptive.py` (thin wrapper).
  * Re-export `run_transient_adaptive` from `pulsim.v2`.
  * Python binding entry in `python/bindings_v2_kernel.cpp`.
  * New example `examples/v2/scripts/run_adaptive_rl_settle.py`.
* No breaking changes — existing `simulate(...)` continues to use fixed dt.
