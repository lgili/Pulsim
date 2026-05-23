"""Pulsim v2 — parameter sweep + Monte Carlo helpers (Phase E.3).

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
