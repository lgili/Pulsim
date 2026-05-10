"""Sweep harness: builds the circuit per sample, runs the transient,
applies metrics, collects results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .distributions import (
    ParameterSpec,
    SamplingStrategy,
    sample,
)
from .metrics import Metric


__all__ = ["SweepResult", "run"]


@dataclass
class SweepResult:
    """Outcome of a parameter sweep.

    `parameters[k]` is the input dict for sample k; `metrics[k]` is the
    metric dict for the same sample (or empty if the sample failed).
    `failed[k]` is `None` for successful samples or an error string for
    failed ones."""
    parameters: list[dict[str, float]] = field(default_factory=list)
    metrics: list[dict[str, float]] = field(default_factory=list)
    failed: list[str | None] = field(default_factory=list)
    strategy: str = ""
    n_samples: int = 0
    seed: int | None = None
    wall_seconds: float = 0.0

    @property
    def n_succeeded(self) -> int:
        return sum(1 for f in self.failed if f is None)

    @property
    def n_failed(self) -> int:
        return self.n_samples - self.n_succeeded

    def to_pandas(self):
        """Wide-format DataFrame: one row per sample, columns =
        parameter names + metric names + 'failed' string."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "to_pandas() requires pandas. Install with: pip install pandas"
            ) from exc
        rows = []
        for k in range(self.n_samples):
            row = {}
            row.update(self.parameters[k])
            row.update(self.metrics[k])
            row["__failed__"] = self.failed[k] if self.failed[k] else ""
            rows.append(row)
        return pd.DataFrame(rows)

    def percentile(self, metric_name: str, q: float) -> float:
        """The q-th percentile (q ∈ [0, 100]) of a named metric across
        the successful samples. Useful for yield analysis (e.g. P95)."""
        values = [
            m[metric_name] for m, f in zip(self.metrics, self.failed)
            if f is None and metric_name in m
        ]
        if not values:
            return float("nan")
        import numpy as np
        return float(np.percentile(values, q))


def _run_one_sample(
    sample_idx: int,
    params: dict[str, float],
    circuit_factory: Callable[..., Any],
    metrics: Sequence[Metric],
    sim_options_factory: Callable[[], Any] | None,
) -> tuple[int, dict[str, float], str | None]:
    """Build the circuit, simulate, apply metrics, return one row.
    Catches all exceptions per sample so a single bad apple doesn't
    abort the sweep — the failure reason is recorded in the result."""
    try:
        circuit = circuit_factory(**params)
        from .. import _pulsim as _pl
        opts = sim_options_factory() if sim_options_factory else _pl.SimulationOptions()
        sim = _pl.Simulator(circuit, opts)
        dc = sim.dc_operating_point()
        if not dc.success:
            return sample_idx, {}, f"DC OP failed: {dc.message}"
        run = sim.run_transient(dc.newton_result.solution)
        if not run.success:
            return sample_idx, {}, f"transient failed: {run.message}"
        out: dict[str, float] = {}
        for m in metrics:
            out[m.name] = float(m(run, circuit))
        return sample_idx, out, None
    except Exception as exc:
        return sample_idx, {}, f"{type(exc).__name__}: {exc}"


def run(
    *,
    circuit_factory: Callable[..., Any],
    parameters: dict[str, ParameterSpec],
    metrics: Sequence[Metric],
    n_samples: int = 64,
    strategy: SamplingStrategy = "monte_carlo",
    seed: int | None = None,
    executor: str = "serial",
    n_workers: int = 0,
    sim_options_factory: Callable[[], Any] | None = None,
) -> SweepResult:
    """Run a parameter sweep.

    Args:
        circuit_factory: callable producing a `pulsim.Circuit` from a
            parameter dict — `circuit_factory(**params)`.
        parameters: per-parameter spec (Distribution / Cartesian / list).
        metrics: list of `Metric` instances applied to every successful
            sample.
        n_samples: number of samples to draw (ignored for `strategy=
            "cartesian"` which enumerates the full product).
        strategy: `"cartesian" | "monte_carlo" | "lhs" | "sobol" | "halton"`.
        seed: RNG seed for bit-identical reruns.
        executor: `"serial"` (default) or `"joblib"` for parallel.
        n_workers: process count for joblib executor.
            0 = `cpu_count() - 1`.
        sim_options_factory: optional callable returning a
            `SimulationOptions` per sample. Defaults to a fresh
            default-constructed SimulationOptions.
    """
    import time
    t0 = time.perf_counter()

    samples = sample(parameters, n_samples=n_samples,
                      strategy=strategy, seed=seed)
    n_actual = len(samples)
    out = SweepResult(
        strategy=strategy, n_samples=n_actual, seed=seed,
        parameters=samples,
        metrics=[{} for _ in range(n_actual)],
        failed=[None] * n_actual,
    )

    if executor == "serial":
        for k in range(n_actual):
            _, m, err = _run_one_sample(
                k, samples[k], circuit_factory, metrics, sim_options_factory)
            out.metrics[k] = m
            out.failed[k] = err
    elif executor == "joblib":
        try:
            from joblib import Parallel, delayed
        except ImportError as exc:
            raise ImportError(
                "executor='joblib' requires joblib. "
                "Install with: pip install joblib"
            ) from exc
        if n_workers <= 0:
            import os
            n_workers = max(1, (os.cpu_count() or 2) - 1)
        rows = Parallel(n_jobs=n_workers)(
            delayed(_run_one_sample)(
                k, samples[k], circuit_factory, metrics, sim_options_factory)
            for k in range(n_actual)
        )
        for r in rows:
            if r is None:
                continue
            k, m, err = r
            out.metrics[k] = m
            out.failed[k] = err
    else:
        raise ValueError(
            f"unknown executor {executor!r}. "
            "Expected 'serial' or 'joblib'.")

    out.wall_seconds = time.perf_counter() - t0
    return out
