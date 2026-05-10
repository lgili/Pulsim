"""Monte-Carlo / quasi-Monte-Carlo / Cartesian parameter sweep.

`add-monte-carlo-parameter-sweep` Phase 1: ergonomic high-level
harness that runs a Pulsim simulation across a parameter space and
collects user-defined metrics.

Usage::

    import pulsim

    def make_buck(L, C):
        return pulsim.templates.buck(
            Vin=24, Vout=5, Iout=2, fsw=100e3,
            L=L, C=C,
        ).circuit

    result = pulsim.sweep.run(
        circuit_factory=make_buck,
        parameters={
            "L": pulsim.sweep.Distribution.uniform(40e-6, 80e-6),
            "C": pulsim.sweep.Distribution.uniform(10e-6, 30e-6),
        },
        metrics=[
            pulsim.sweep.steady_state("out"),
            pulsim.sweep.peak("out"),
        ],
        n_samples=128,
        strategy="lhs",
        seed=42,
        sim_options=lambda: pulsim.SimulationOptions(),  # per-sample
    )
    df = result.to_pandas()
    print(df.describe())
    print(f"Failed: {len(result.failed)}")
"""

from .distributions import (
    Cartesian,
    Distribution,
    ParameterSpec,
    SamplingStrategy,
    sample,
)
from .metrics import (
    Metric,
    custom,
    peak,
    rms,
    settling_time,
    steady_state,
)
from .runner import (
    SweepResult,
    run,
)

__all__ = [
    "Cartesian",
    "Distribution",
    "Metric",
    "ParameterSpec",
    "SamplingStrategy",
    "SweepResult",
    "custom",
    "peak",
    "rms",
    "run",
    "sample",
    "settling_time",
    "steady_state",
]
