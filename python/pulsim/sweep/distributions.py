"""Parameter distributions and the cartesian / quasi-MC sampling
strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


__all__ = [
    "Distribution",
    "Cartesian",
    "ParameterSpec",
    "sample",
    "SamplingStrategy",
]


@dataclass
class Distribution:
    """A parametric continuous distribution.

    Constructed via the class methods (`normal`, `uniform`,
    `loguniform`, `triangular`); not normally instantiated directly.
    Each instance carries an `inverse_cdf(u)` callable that maps a
    uniform-`(0,1)` variate to the distribution's quantile — that lets
    quasi-Monte-Carlo samplers (LHS / Sobol) drive every distribution
    type uniformly.
    """
    name: str
    inverse_cdf: Callable[[np.ndarray], np.ndarray]

    @classmethod
    def normal(cls, mean: float, std: float) -> "Distribution":
        from scipy.stats import norm
        return cls(
            name=f"normal(mean={mean}, std={std})",
            inverse_cdf=lambda u: norm.ppf(np.clip(u, 1e-12, 1 - 1e-12),
                                            loc=mean, scale=std),
        )

    @classmethod
    def uniform(cls, low: float, high: float) -> "Distribution":
        if not (high > low):
            raise ValueError(f"uniform: high ({high}) must exceed low ({low})")
        return cls(
            name=f"uniform(low={low}, high={high})",
            inverse_cdf=lambda u: low + (high - low) * np.asarray(u, dtype=float),
        )

    @classmethod
    def loguniform(cls, low: float, high: float) -> "Distribution":
        if not (low > 0 and high > low):
            raise ValueError(
                f"loguniform: require 0 < low ({low}) < high ({high})")
        log_low, log_high = math.log(low), math.log(high)
        return cls(
            name=f"loguniform(low={low}, high={high})",
            inverse_cdf=lambda u: np.exp(
                log_low + (log_high - log_low) * np.asarray(u, dtype=float)),
        )

    @classmethod
    def triangular(cls, low: float, mode: float, high: float) -> "Distribution":
        if not (low < mode < high):
            raise ValueError(
                f"triangular: require low ({low}) < mode ({mode}) < high ({high})")

        def _inv(u):
            u = np.asarray(u, dtype=float)
            c = (mode - low) / (high - low)
            return np.where(
                u < c,
                low + np.sqrt(u * (high - low) * (mode - low)),
                high - np.sqrt((1 - u) * (high - low) * (high - mode)),
            )
        return cls(
            name=f"triangular(low={low}, mode={mode}, high={high})",
            inverse_cdf=_inv,
        )


@dataclass
class Cartesian:
    """Discrete enumeration over an explicit value list."""
    values: Sequence[float]


# ParameterSpec: either a Distribution (continuous) or a Cartesian list
# (discrete). The sampler dispatches on the type.
ParameterSpec = Distribution | Cartesian | Sequence[float]


SamplingStrategy = str   # "cartesian" | "monte_carlo" | "lhs" | "sobol" | "halton"


def _coerce_spec(spec: ParameterSpec) -> Distribution | Cartesian:
    if isinstance(spec, Distribution) or isinstance(spec, Cartesian):
        return spec
    if isinstance(spec, (list, tuple)):
        return Cartesian(values=tuple(spec))
    raise TypeError(
        f"Unsupported ParameterSpec type: {type(spec).__name__}. "
        "Expected Distribution, Cartesian, or list/tuple of values.")


def sample(
    parameters: dict[str, ParameterSpec],
    *,
    n_samples: int,
    strategy: SamplingStrategy = "monte_carlo",
    seed: int | None = None,
) -> list[dict[str, float]]:
    """Generate `n_samples` parameter dicts under the given strategy.

    For `"cartesian"` the `n_samples` request is ignored; the function
    returns the full Cartesian product of the value lists. All other
    strategies honor `n_samples` exactly.

    `seed=None` defers to NumPy's default random state — non-
    reproducible. Pass an explicit `seed` for bit-identical reruns.
    """
    rng = np.random.default_rng(seed)
    if not parameters:
        return []

    coerced = {name: _coerce_spec(s) for name, s in parameters.items()}

    if strategy == "cartesian":
        # Require all specs to be Cartesian for explicit enumeration.
        from itertools import product
        names = list(coerced.keys())
        all_values = []
        for n in names:
            v = coerced[n]
            if not isinstance(v, Cartesian):
                raise ValueError(
                    f"strategy='cartesian' requires every parameter to be "
                    f"a Cartesian or list — got {type(v).__name__} for {n!r}")
            all_values.append(list(v.values))
        rows = []
        for combo in product(*all_values):
            rows.append({names[i]: float(combo[i]) for i in range(len(names))})
        return rows

    # For non-cartesian strategies, every continuous Distribution is
    # driven by a uniform-(0,1) variate from the chosen sequence.
    n_dim = len(coerced)
    if strategy == "monte_carlo":
        u = rng.random(size=(n_samples, n_dim))
    elif strategy in ("lhs", "sobol", "halton"):
        # scipy renamed `seed` → `rng` in 1.10. Handle both.
        from scipy.stats import qmc
        cls_map = {
            "lhs":    qmc.LatinHypercube,
            "sobol":  qmc.Sobol,
            "halton": qmc.Halton,
        }
        cls = cls_map[strategy]
        try:
            sampler = cls(d=n_dim, rng=rng)
        except TypeError:
            sampler = cls(d=n_dim, seed=rng)
        u = sampler.random(n=n_samples)
    else:
        raise ValueError(
            f"unknown sampling strategy {strategy!r}. "
            "Expected one of: cartesian, monte_carlo, lhs, sobol, halton")

    rows = []
    names = list(coerced.keys())
    for k in range(n_samples):
        row: dict[str, float] = {}
        for i, name in enumerate(names):
            spec = coerced[name]
            if isinstance(spec, Cartesian):
                # Map u ∈ [0, 1) to a discrete index.
                idx = int(min(len(spec.values) - 1,
                              math.floor(u[k, i] * len(spec.values))))
                row[name] = float(spec.values[idx])
            else:
                row[name] = float(spec.inverse_cdf(u[k, i]))
        rows.append(row)
    return rows
