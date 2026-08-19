"""Pulsim — parameter sweep + Monte Carlo helpers (Phase E.3).

Turns ``simulate()`` into a design-exploration tool. The user supplies:
  * A `builder_factory(**params) -> CircuitBuilder` that constructs a
    populated builder from named parameter values,
  * A parameter grid (`dict[str, sequence]`) OR Monte Carlo
    distributions (`dict[str, callable]`),
  * A KPI extractor `(SimulationResult, params) -> dict[str, float]`
    that pulls the metrics the user cares about (output ripple,
    efficiency, settling time, peak current, …),
  * Standard `simulate()` kwargs (t_end, dt, switch_fn, etc.).

Returns a `SweepResult` containing:
  * `params`: dict of parameter arrays (parallel to KPIs),
  * `kpis`: dict of KPI arrays,
  * `n_simulations`, `wall_time_s`.

Optional parallel execution via `multiprocessing`. For small parameter
spaces (≤ 100 points) the serial path is fast enough; for grid
sweeps over 1000+ points, set `parallel=True`.

Typical usage — design buck output ripple vs (L, C):

    import pulsim as p
    import numpy as np

    def build(L, C):
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "vin", "gnd", 24.0)
        b.add_switch("Q1", "vin", "sw", g_on=1e3, g_off=1e-9)
        b.add_diode("D", "gnd", "sw", g_on=1e3, g_off=1e-9, V_th=0.7)
        b.add_inductor("L", "sw", "vout", L)
        b.add_capacitor("C", "vout", "gnd", C)
        b.add_resistor("R", "vout", "gnd", 5.0)
        return b

    def ripple_kpi(res, params):
        # Output-voltage peak-to-peak ripple in the last 0.5 ms.
        import numpy as np
        t = np.asarray(res.times)
        v = np.asarray(res.states)[:, 3]  # vout index
        tail = v[t >= t[-1] - 0.5e-3]
        return {"v_ripple_pp": tail.max() - tail.min(),
                "v_mean": tail.mean()}

    out = p.sweep(
        build,
        params={"L": np.geomspace(50e-6, 500e-6, 5),
                "C":  np.geomspace(10e-6, 200e-6, 5)},
        kpi_fn=ripple_kpi,
        t_end=2e-3, dt=2e-7,
        switch_fn=p.make_pwm_switch_fn(100e3, 0.5, 0, 3),
    )

For Monte Carlo (random distributions instead of a regular grid),
pass `distributions={"L": lambda rng: rng.uniform(50e-6, 500e-6)}`
and `n_samples=500`.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np


__all__ = [
    "SweepResult",
    "sweep",
    "monte_carlo",
    # v1.4.0 — path-aware variants exploiting cache.refactor_parametric
    "sweep_path_aware",
    "monte_carlo_path_aware",
]


@dataclass
class SweepResult:
    """Result of a parameter sweep or Monte Carlo run.

    `params` and `kpis` are dicts of equal-length arrays. Each index
    corresponds to one simulation: ``params[name][i]`` is the value
    of parameter ``name`` for run ``i``, and ``kpis[name][i]`` is the
    extracted KPI value.

    `shape` is the original grid shape (for `sweep` only — empty tuple
    for Monte Carlo). Reshape any 1-D array via
    ``kpis[name].reshape(shape)`` to get a meshgrid-style 2-D array.
    """
    params: Dict[str, np.ndarray]
    kpis: Dict[str, np.ndarray]
    shape: tuple = ()
    n_simulations: int = 0
    wall_time_s: float = 0.0
    failed: list = field(default_factory=list)  # indices that raised

    def to_dataframe(self):
        """Convert to a pandas DataFrame (one row per simulation)."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "to_dataframe() requires pandas — "
                "pip install pandas") from exc
        all_cols = {**self.params, **self.kpis}
        return pd.DataFrame(all_cols)


def _run_one_simulation(args):
    """Worker — runs one simulation. Top-level so multiprocessing can
    pickle it."""
    (builder_factory, kpi_fn, params,
      simulate_kwargs) = args
    # Lazy import inside the worker to support multiprocessing.
    import pulsim as _v2
    builder = builder_factory(**params)
    res = _v2.simulate(builder, **simulate_kwargs)
    return kpi_fn(res, params)


def sweep(builder_factory: Callable[..., Any],
            *,
            params: Dict[str, Sequence[float]],
            kpi_fn: Callable[..., Dict[str, float]],
            parallel: bool = False,
            n_workers: Optional[int] = None,
            verbose: bool = True,
            **simulate_kwargs,
            ) -> SweepResult:
    """Iterate `simulate()` over a parameter grid.

    Parameters
    ----------
    builder_factory
        Function `(**params) -> CircuitBuilder` that constructs a
        populated builder.
    params
        Dict mapping parameter names to value sequences. The sweep
        runs over the Cartesian product (every combination).
    kpi_fn
        `(SimulationResult, params_dict) -> dict[str, float]`. Called
        once per simulation to extract scalar metrics from the result.
    parallel
        If True, use `multiprocessing.Pool` (one process per
        simulation in flight). Default False — serial is faster for
        small grids due to process-startup cost.
    n_workers
        Pool size when parallel. Defaults to `os.cpu_count() − 1`.
    **simulate_kwargs
        Forwarded to `simulate(...)` for each run (t_end, dt,
        switch_fn, etc.).

    Returns
    -------
    SweepResult
        Parallel arrays of parameters and KPIs.
    """
    keys = list(params.keys())
    vals = [list(params[k]) for k in keys]
    shape = tuple(len(v) for v in vals)
    grid_iter = list(itertools.product(*vals))
    n_runs = len(grid_iter)

    if verbose:
        print(f"  sweep: {n_runs} simulations "
               f"(grid shape {shape})  "
               f"params: {keys}")

    args_iter = []
    grid_params = []
    for combo in grid_iter:
        p_dict = dict(zip(keys, combo))
        grid_params.append(p_dict)
        args_iter.append(
            (builder_factory, kpi_fn, p_dict, simulate_kwargs))

    t0 = time.perf_counter()
    failed = []
    kpi_records = []

    if parallel:
        from multiprocessing import Pool
        import os
        nw = n_workers if n_workers is not None else \
              max(1, (os.cpu_count() or 2) - 1)
        if verbose:
            print(f"    parallel: {nw} workers")
        with Pool(nw) as pool:
            try:
                for i, kpis in enumerate(
                        pool.imap_unordered(_run_one_simulation,
                                              args_iter)):
                    kpi_records.append((i, kpis))
                    if verbose and (i + 1) % max(1, n_runs // 10) == 0:
                        print(f"    {i+1}/{n_runs} done")
            except Exception:
                pool.terminate()
                raise
        # Imap is unordered → sort to match the grid index.
        kpi_records.sort(key=lambda x: x[0])
        kpi_list = [k for _, k in kpi_records]
    else:
        kpi_list = []
        for i, args in enumerate(args_iter):
            try:
                kpi_list.append(_run_one_simulation(args))
            except Exception as exc:
                failed.append((i, str(exc)))
                # Push placeholder NaN KPIs for this row.
                kpi_list.append({})
            if verbose and (i + 1) % max(1, n_runs // 10) == 0:
                print(f"    {i+1}/{n_runs} done")

    wall = time.perf_counter() - t0
    if verbose:
        print(f"  sweep finished in {wall:.2f} s "
               f"({wall*1000/n_runs:.1f} ms / sim, "
               f"{len(failed)} failed)")

    # Stack into parallel arrays.
    params_arr = {k: np.array([p[k] for p in grid_params])
                    for k in keys}
    all_kpi_names = set()
    for d in kpi_list:
        all_kpi_names.update(d.keys())
    kpis_arr = {}
    for name in sorted(all_kpi_names):
        kpis_arr[name] = np.array(
            [d.get(name, np.nan) for d in kpi_list], dtype=float)

    return SweepResult(
        params=params_arr, kpis=kpis_arr, shape=shape,
        n_simulations=n_runs, wall_time_s=wall, failed=failed,
    )


def monte_carlo(builder_factory: Callable[..., Any],
                  *,
                  distributions: Dict[str, Callable[..., float]],
                  kpi_fn: Callable[..., Dict[str, float]],
                  n_samples: int = 100,
                  seed: Optional[int] = None,
                  parallel: bool = False,
                  n_workers: Optional[int] = None,
                  verbose: bool = True,
                  **simulate_kwargs,
                  ) -> SweepResult:
    """Run a Monte Carlo simulation over random parameter draws.

    Parameters
    ----------
    builder_factory
        `(**params) -> CircuitBuilder`.
    distributions
        Dict mapping parameter names to draw functions. Each draw
        function is called as ``rng_draw(np_default_rng)`` and must
        return a single float. Example:
        ``{"R": lambda r: r.normal(10.0, 1.0)}``
    kpi_fn
        `(res, params) -> dict[str, float]`.
    n_samples
        Number of random draws.
    seed
        RNG seed (default: non-reproducible).

    Other parameters
    ----------------
    Same as :func:`sweep`.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_samples):
        s = {name: float(draw(rng))
              for name, draw in distributions.items()}
        samples.append(s)
    if verbose:
        print(f"  monte_carlo: {n_samples} random draws  "
               f"params: {list(distributions)}")

    args_iter = [(builder_factory, kpi_fn, s, simulate_kwargs)
                   for s in samples]

    t0 = time.perf_counter()
    failed = []
    kpi_list = []
    if parallel:
        from multiprocessing import Pool
        import os
        nw = n_workers if n_workers is not None else \
              max(1, (os.cpu_count() or 2) - 1)
        if verbose:
            print(f"    parallel: {nw} workers")
        with Pool(nw) as pool:
            records = list(enumerate(
                pool.imap_unordered(_run_one_simulation, args_iter)))
        records.sort(key=lambda x: x[0])
        kpi_list = [k for _, k in records]
    else:
        for i, args in enumerate(args_iter):
            try:
                kpi_list.append(_run_one_simulation(args))
            except Exception as exc:
                failed.append((i, str(exc)))
                kpi_list.append({})
            if verbose and (i + 1) % max(1, n_samples // 10) == 0:
                print(f"    {i+1}/{n_samples} done")

    wall = time.perf_counter() - t0
    if verbose:
        print(f"  monte_carlo finished in {wall:.2f} s "
               f"({wall*1000/n_samples:.1f} ms / sim, "
               f"{len(failed)} failed)")

    keys = list(distributions.keys())
    params_arr = {k: np.array([s[k] for s in samples]) for k in keys}
    all_kpi_names = set()
    for d in kpi_list:
        all_kpi_names.update(d.keys())
    kpis_arr = {}
    for name in sorted(all_kpi_names):
        kpis_arr[name] = np.array(
            [d.get(name, np.nan) for d in kpi_list], dtype=float)

    return SweepResult(
        params=params_arr, kpis=kpis_arr, shape=(),
        n_simulations=n_samples, wall_time_s=wall, failed=failed,
    )


# =============================================================================
# v1.4.0 — path-aware variants exploiting cache.refactor_parametric
# =============================================================================
#
# `sweep_path_aware` / `monte_carlo_path_aware` are drop-in
# replacements for `sweep` / `monte_carlo` that exploit the v1.4.0
# `PwlStateSpaceCache::refactor_parametric` API to avoid rebuilding
# the cache from scratch at every sweep point. Captured speedup on
# the synthetic buck fixture: 3.0–3.7× (see
# `artigos/02_tpel_methods/benchmarks/PARAMETRIC_RESULTS.md`).
#
# Key API differences from the legacy variants:
#   * Takes a SINGLE pre-populated `builder` instead of a
#     `builder_factory(**params)` callable. The cache + pool are
#     built once upfront; the sweep mutates the pool in place via
#     `cache.refactor_parametric`.
#   * `params_dict` keys are **component names** (the first argument
#     of every `add_*` call, e.g. `"R_load"`, `"L_out"`). The
#     device kind is auto-detected via `builder.pool.kind_of(...)`.
#   * Auto-fallback: if any parameter name is unknown or maps to an
#     unsupported device kind (CurrentSource, Diode, MOSFET, …), the
#     helper falls back to the legacy `sweep` / `monte_carlo` path
#     transparently. No user-facing breakage; the user sees a
#     warning and the same `SweepResult` shape.
#   * `parallel=True` is NOT supported — the path-aware variant
#     reuses one cache across all sweep points, which is incompatible
#     with multiprocessing's separate-process model. Use
#     `parallel=True` on the legacy `sweep` for embarrassingly
#     parallel grids; use `sweep_path_aware` for serial sweeps where
#     the per-point speedup matters.
#
# Both helpers return the same `SweepResult` as their legacy
# counterparts so downstream code (DataFrame conversion, plotting,
# etc.) is unchanged.


# Component-kind dispatch table for the in-place builder updaters.
# Maps `DevicePool::StoredKind` enum values to the matching
# `CircuitBuilder.update_<kind>(name, value)` method name. Keys are
# integers (the enum's underlying value); we look up the integer
# from `pool.kind_of(branch_id)` which returns the StoredKind.
#
# The kind enum is exposed to Python as `_pulsim.StoredKind` —
# we don't import it eagerly to keep this module importable without
# the kernel extension (e.g. for docs builds). Lookup happens
# inside the helper via duck-typing on `pool.kind_of()`'s return.
_KIND_TO_UPDATE_METHOD = {
    "Resistor":      "update_resistor_R",
    "Inductor":      "update_inductor_L",
    "Capacitor":     "update_capacitor_C",
    "VoltageSource": "update_voltage_source_V",
}


def _resolve_param_kind(builder, name: str) -> Optional[str]:
    """Look up the device kind for a builder-time component name.

    Returns one of the strings in `_KIND_TO_UPDATE_METHOD`, or None
    if the name is unknown / the kind isn't parametric-refactor-supported.
    """
    try:
        b_id = builder.branch_id_of(name)
    except Exception:
        return None
    try:
        kind = builder.pool.kind_of(b_id)
    except Exception:
        return None
    # kind comes back as the `StoredKind` enum; its `.name` attribute
    # is the canonical string ("Resistor", "Inductor", etc.).
    kind_str = getattr(kind, "name", str(kind))
    if kind_str in _KIND_TO_UPDATE_METHOD:
        return kind_str
    return None


def _check_path_aware_eligible(builder,
                                  param_names: Sequence[str]
                                  ) -> tuple[bool, list[str]]:
    """Pre-flight every param name. Returns (eligible, unknown_names).

    If `eligible` is False, the caller should fall back to the legacy
    sweep path. `unknown_names` is the list of names that failed
    resolution (for the warning message).
    """
    unknown = [n for n in param_names if _resolve_param_kind(builder, n) is None]
    return (len(unknown) == 0, unknown)


def sweep_path_aware(
    builder,
    *,
    params: Dict[str, Sequence[float]],
    kpi_fn: Callable[..., Dict[str, float]],
    t_end: float,
    dt: float,
    verbose: bool = True,
    **simulate_kwargs,
) -> SweepResult:
    """Path-aware grid sweep over named component parameters (v1.4.0).

    Drop-in replacement for ``sweep`` that exploits the in-house
    `PwlStateSpaceCache::refactor_parametric` API. Builds the cache
    ONCE upfront, then mutates the pool's parameter values and
    re-uses the cached symbolic + most of the L+U factor for every
    sweep point.

    Captured wall-clock speedup: 3.0–3.7× vs legacy ``sweep`` on the
    synthetic buck fixture (see
    `artigos/02_tpel_methods/benchmarks/PARAMETRIC_RESULTS.md`).

    Parameters
    ----------
    builder
        A pre-populated :class:`CircuitBuilder`. NOT a factory —
        the path-aware variant mutates this builder's pool in place.
    params
        Dict mapping **component names** (the first arg of
        `add_*` calls, e.g. ``"R_load"``, ``"L_out"``) to value
        sequences. The sweep runs over the Cartesian product.
    kpi_fn
        ``(SimulationResult, params_dict) -> dict[str, float]``.
        Called once per sweep point to extract scalar KPIs.
    t_end, dt
        Time window and fixed step. Same semantics as ``simulate``.
    verbose
        Print per-row progress + final timing summary.
    **simulate_kwargs
        Forwarded to ``simulate(...)`` for each run (``switch_fn``,
        ``b_extra_fn``, ``start_from_dc_op``, etc.).

    Returns
    -------
    SweepResult
        Same shape as :func:`sweep`. Downstream code is unchanged.

    Notes
    -----
    * **Auto-fallback**: if any param name is unknown to the builder
      or maps to an unsupported device kind (CurrentSource, Diode,
      MOSFET, etc.), the helper logs a warning and silently routes
      to the legacy ``sweep`` path. The user sees the same
      ``SweepResult`` and the same KPI numbers — just no speedup.
    * **Multi-bit refactor**: when the grid covers ≥ 2 changed
      parameters per row, the cache uses
      ``refactor_parametric_batch`` to apply them in one batch.
    * **No parallel mode**: the cache is shared across all sweep
      points. Use legacy ``sweep(parallel=True)`` for that.
    """
    import pulsim as _v2

    keys = list(params.keys())
    vals = [list(params[k]) for k in keys]
    shape = tuple(len(v) for v in vals)
    grid_iter = list(itertools.product(*vals))
    n_runs = len(grid_iter)

    # Pre-flight: are all param names path-aware-eligible?
    eligible, unknown = _check_path_aware_eligible(builder, keys)
    if not eligible:
        import warnings
        warnings.warn(
            f"sweep_path_aware: parameter name(s) {unknown!r} "
            "are not registered as Resistor / Inductor / Capacitor / "
            "VoltageSource components in the builder; falling back "
            "to the legacy sweep() path. To get the v1.4.0 speedup, "
            "rename the affected components or use a path-aware-"
            "compatible device kind.",
            RuntimeWarning, stacklevel=2)
        # Legacy path needs a builder_factory; wrap the single
        # builder + apply updates via update_* before each sim.
        def _legacy_factory(**p):
            # Mutate the builder's pool, then return it. The legacy
            # sweep() copies the result implicitly via the
            # builder_factory contract — but our single-builder
            # design needs to be careful: legacy sweep runs serially
            # by default, so in-place mutation per call is safe.
            for n, v in p.items():
                kind = _resolve_param_kind(builder, n)
                # kind is None for unsupported names — skip silently
                # so unknown params can be consumed by the user's
                # builder_factory pattern (e.g. a derived value
                # injected into KPIs but not a real pool entry).
                if kind is None:
                    continue
                method = getattr(builder, _KIND_TO_UPDATE_METHOD[kind])
                method(n, v)
            return builder
        return sweep(_legacy_factory, params=params,
                     kpi_fn=kpi_fn, parallel=False,
                     verbose=verbose, t_end=t_end, dt=dt,
                     **simulate_kwargs)

    # Resolve every param name → (branch_id, update_method_name) once.
    resolved = []
    for n in keys:
        kind = _resolve_param_kind(builder, n)
        assert kind is not None, "pre-flight guarantees kind"
        resolved.append((n, builder.branch_id_of(n),
                         _KIND_TO_UPDATE_METHOD[kind]))

    # Build the cache ONCE at the initial values.
    cache = _v2.PwlStateSpaceCache(builder.graph, builder.pool)
    cache.build(dt)

    # Default switch_fn: all switches closed.
    user_switch_fn = simulate_kwargs.pop("switch_fn", None)
    if user_switch_fn is None:
        n_sw = builder.graph.num_switches
        from pulsim import SwitchStateMask
        all_closed = SwitchStateMask(n_sw)
        for i in range(n_sw):
            all_closed.set(i, True)
        user_switch_fn = lambda _t: all_closed  # noqa: E731

    # Sweep
    if verbose:
        print(f"  sweep_path_aware: {n_runs} simulations "
               f"(grid shape {shape})  params: {keys}")

    t0 = time.perf_counter()
    failed: list = []
    kpi_list: list = []
    grid_params: list = []
    total_refactor_us = 0.0

    for i, combo in enumerate(grid_iter):
        p_dict = dict(zip(keys, combo))
        grid_params.append(p_dict)

        # 1. Push pool updates through builder convenience setters.
        for n, b_id, method_name in resolved:
            getattr(builder, method_name)(n, p_dict[n])

        # 2. Refactor every active mask's cached factor via the
        #    path-based update. Batch overload pushes all updates
        #    in one call → single mask-traversal per sweep point.
        if len(resolved) == 1:
            res = cache.refactor_parametric(
                resolved[0][1], p_dict[resolved[0][0]])
        else:
            updates = [(b_id, p_dict[n]) for n, b_id, _ in resolved]
            res = cache.refactor_parametric_batch(updates)
        total_refactor_us += res.wall_time_us

        # 3. Run the transient on the refactored cache.
        try:
            sim = _v2.run_transient(
                cache, builder.graph, builder.pool,
                _make_options(t_end, dt, **simulate_kwargs),
                switch_fn=user_switch_fn,
                **{k: v for k, v in simulate_kwargs.items()
                    if k in {"b_extra_fn", "start_from_dc_op",
                              "step_observer", "initial_state",
                              "should_continue"}})
            _warn_if_breached(sim, f"sweep point {i} {p_dict}")
            kpi_list.append(kpi_fn(sim, p_dict))
        except Exception as exc:
            failed.append((i, str(exc)))
            kpi_list.append({})

        if verbose and (i + 1) % max(1, n_runs // 10) == 0:
            print(f"    {i+1}/{n_runs} done")

    wall = time.perf_counter() - t0
    if verbose:
        print(f"  sweep_path_aware finished in {wall:.2f} s "
               f"({wall*1000/n_runs:.1f} ms / sim, "
               f"refactor share: {total_refactor_us/1000/wall*100:.1f}%, "
               f"{len(failed)} failed)")

    params_arr = {k: np.array([p[k] for p in grid_params])
                    for k in keys}
    all_kpi_names: set = set()
    for d in kpi_list:
        all_kpi_names.update(d.keys())
    kpis_arr = {}
    for name in sorted(all_kpi_names):
        kpis_arr[name] = np.array(
            [d.get(name, np.nan) for d in kpi_list], dtype=float)

    return SweepResult(
        params=params_arr, kpis=kpis_arr, shape=shape,
        n_simulations=n_runs, wall_time_s=wall, failed=failed,
    )


def monte_carlo_path_aware(
    builder,
    *,
    distributions: Dict[str, Callable[..., float]],
    kpi_fn: Callable[..., Dict[str, float]],
    n_samples: int,
    t_end: float,
    dt: float,
    seed: Optional[int] = None,
    verbose: bool = True,
    **simulate_kwargs,
) -> SweepResult:
    """Path-aware Monte Carlo (v1.4.0).

    Drop-in replacement for ``monte_carlo`` that exploits
    ``cache.refactor_parametric`` to avoid rebuilding the cache per
    sample. Same auto-fallback semantics as :func:`sweep_path_aware`.

    Parameters
    ----------
    builder
        Pre-populated :class:`CircuitBuilder`.
    distributions
        Dict mapping component names to callables
        ``rng -> float``. Each sample draws ``distributions[name](rng)``.
    kpi_fn
        ``(SimulationResult, params_dict) -> dict[str, float]``.
    n_samples
        Number of Monte Carlo draws.
    t_end, dt
        Time window + fixed step.
    seed
        RNG seed for reproducibility (passed to ``np.random.default_rng``).
    verbose, **simulate_kwargs
        Same as :func:`sweep_path_aware`.
    """
    import pulsim as _v2

    keys = list(distributions.keys())
    eligible, unknown = _check_path_aware_eligible(builder, keys)
    if not eligible:
        import warnings
        warnings.warn(
            f"monte_carlo_path_aware: parameter name(s) {unknown!r} "
            "are not parametric-refactor-supported; falling back to "
            "the legacy monte_carlo() path.",
            RuntimeWarning, stacklevel=2)
        def _legacy_factory(**p):
            for n, v in p.items():
                kind = _resolve_param_kind(builder, n)
                if kind is not None:
                    method = getattr(builder, _KIND_TO_UPDATE_METHOD[kind])
                    method(n, v)
            return builder
        return monte_carlo(
            _legacy_factory, distributions=distributions,
            kpi_fn=kpi_fn, n_samples=n_samples, seed=seed,
            parallel=False, verbose=verbose,
            t_end=t_end, dt=dt, **simulate_kwargs)

    resolved = []
    for n in keys:
        kind = _resolve_param_kind(builder, n)
        assert kind is not None, "pre-flight guarantees kind"
        resolved.append((n, builder.branch_id_of(n),
                         _KIND_TO_UPDATE_METHOD[kind]))

    cache = _v2.PwlStateSpaceCache(builder.graph, builder.pool)
    cache.build(dt)

    user_switch_fn = simulate_kwargs.pop("switch_fn", None)
    if user_switch_fn is None:
        n_sw = builder.graph.num_switches
        from pulsim import SwitchStateMask
        all_closed = SwitchStateMask(n_sw)
        for i in range(n_sw):
            all_closed.set(i, True)
        user_switch_fn = lambda _t: all_closed  # noqa: E731

    rng = np.random.default_rng(seed)
    samples: list = []
    for _ in range(n_samples):
        sample = {n: float(distributions[n](rng)) for n in keys}
        samples.append(sample)

    if verbose:
        print(f"  monte_carlo_path_aware: {n_samples} samples  "
               f"params: {keys}  seed: {seed}")

    t0 = time.perf_counter()
    failed: list = []
    kpi_list: list = []
    total_refactor_us = 0.0

    for i, sample in enumerate(samples):
        for n, b_id, method_name in resolved:
            getattr(builder, method_name)(n, sample[n])
        if len(resolved) == 1:
            res = cache.refactor_parametric(
                resolved[0][1], sample[resolved[0][0]])
        else:
            updates = [(b_id, sample[n]) for n, b_id, _ in resolved]
            res = cache.refactor_parametric_batch(updates)
        total_refactor_us += res.wall_time_us

        try:
            sim = _v2.run_transient(
                cache, builder.graph, builder.pool,
                _make_options(t_end, dt, **simulate_kwargs),
                switch_fn=user_switch_fn,
                **{k: v for k, v in simulate_kwargs.items()
                    if k in {"b_extra_fn", "start_from_dc_op",
                              "step_observer", "initial_state",
                              "should_continue"}})
            _warn_if_breached(sim, f"MC sample {i}")
            kpi_list.append(kpi_fn(sim, sample))
        except Exception as exc:
            failed.append((i, str(exc)))
            kpi_list.append({})

        if verbose and (i + 1) % max(1, n_samples // 10) == 0:
            print(f"    {i+1}/{n_samples} done")

    wall = time.perf_counter() - t0
    if verbose:
        print(f"  monte_carlo_path_aware finished in {wall:.2f} s "
               f"({wall*1000/n_samples:.1f} ms / sim, "
               f"refactor share: "
               f"{total_refactor_us/1000/wall*100:.1f}%, "
               f"{len(failed)} failed)")

    params_arr = {k: np.array([s[k] for s in samples]) for k in keys}
    all_kpi_names: set = set()
    for d in kpi_list:
        all_kpi_names.update(d.keys())
    kpis_arr = {}
    for name in sorted(all_kpi_names):
        kpis_arr[name] = np.array(
            [d.get(name, np.nan) for d in kpi_list], dtype=float)

    return SweepResult(
        params=params_arr, kpis=kpis_arr, shape=(),
        n_simulations=n_samples, wall_time_s=wall, failed=failed,
    )


def _make_options(t_end: float, dt: float, **simulate_kwargs):
    """Build a SimulationOptions struct from the simulate-style kwargs
    we accept. Anything not consumed here gets forwarded to
    run_transient via the caller's filter on the kwargs dict."""
    import pulsim as _v2
    opts = _v2.SimulationOptions()
    opts.t_start = float(simulate_kwargs.get("t_start", 0.0))
    opts.t_end   = float(t_end)
    opts.dt      = float(dt)
    # Pass-through Newton / event knobs if present.
    for k in ("max_newton_iterations", "max_event_iterations"):
        if k in simulate_kwargs:
            setattr(opts, k, int(simulate_kwargs[k]))
    for k in ("tol_newton_dx", "tol_newton_res"):
        if k in simulate_kwargs and simulate_kwargs[k] is not None:
            setattr(opts, k, float(simulate_kwargs[k]))
    for k in ("enable_newton_line_search", "enable_newton_lm",
                "enable_substep_state_correction"):
        if k in simulate_kwargs and simulate_kwargs[k] is not None:
            setattr(opts, k, bool(simulate_kwargs[k]))
    # review finding PY-1: without this, sweeps could neither opt back
    # into the strict throw nor see it — breaches were fully silent.
    if simulate_kwargs.get("strict_event_iterations"):
        opts.strict_event_iterations = True
    return opts


def _warn_if_breached(sim, combo_desc: str) -> None:
    """review finding PY-1: run_transient no longer throws on diode
    event-iteration breaches, and sweep bypasses simulate()'s loud
    warning — so surface breaches here or sweep KPIs from breached
    runs would be silently plausible-but-degraded."""
    breaches = getattr(sim, "event_iteration_breaches", None)
    if breaches:
        import warnings
        warnings.warn(
            f"sweep: diode event iteration failed to settle on "
            f"{len(breaches)} step(s) for {combo_desc} (first at "
            f"t={breaches[0].t:.6g}). KPIs for this point carry a "
            "one-flip diode-state error near those instants; pass "
            "strict_event_iterations=True to fail such points "
            "instead.",
            stacklevel=4)
