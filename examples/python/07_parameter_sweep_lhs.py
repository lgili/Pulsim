"""Design exploration via Latin-Hypercube parameter sweep.

A power-electronics designer picks ``L`` and ``C`` once and lives with the
choice. This script sweeps the (L, C) plane via a 64-point LHS draw,
simulates each candidate buck, and reports the ripple / settling-time
trade-off across the sample.

Useful idiom: pick the corner of the (L, C) plane that minimizes ripple
*subject to* board area and cost — drive the rest of the workflow from
the resulting Pareto.

Run::

    python 07_parameter_sweep_lhs.py

See also: docs/parameter-sweep.md
"""

from __future__ import annotations

import os

import pulsim


def make_buck(L: float, C: float):
    """Factory consumed by ``pulsim.sweep.run``.

    The sweep harness calls ``circuit_factory(**params)``; each parameter
    name in the spec dict must match a kwarg here.
    """
    exp = pulsim.templates.buck(
        Vin=24.0,
        Vout=5.0,
        Iout=2.0,
        fsw=100_000.0,
        L=L,
        C=C,
    )
    exp.circuit.set_switching_mode_for_all(pulsim.SwitchingMode.Ideal)
    exp.circuit.set_pwl_state("Q1", False)
    exp.circuit.set_pwl_state("D1", False)
    return exp.circuit


def make_options() -> pulsim.SimulationOptions:
    opts = pulsim.SimulationOptions()
    opts.tstop = 2e-3
    opts.dt = 1e-7
    opts.adaptive_timestep = False
    opts.switching_mode = pulsim.SwitchingMode.Ideal
    return opts


def main() -> None:
    spec = {
        "L": pulsim.sweep.Distribution.uniform(40e-6, 120e-6),
        "C": pulsim.sweep.Distribution.uniform(10e-6, 60e-6),
    }
    metrics = [
        pulsim.sweep.steady_state("out"),
        pulsim.sweep.peak("out"),
        pulsim.sweep.settling_time("out", target=5.0, tolerance=0.02),
    ]

    print("Running 64-point LHS sweep over (L, C) ...")
    result = pulsim.sweep.run(
        circuit_factory=make_buck,
        parameters=spec,
        metrics=metrics,
        n_samples=64,
        strategy="lhs",
        seed=42,
        sim_options_factory=make_options,
    )
    print(f"  succeeded: {result.n_succeeded} / {result.n_samples}    "
          f"wall: {result.wall_seconds:.2f} s")     # SweepResult.wall_seconds
    if result.n_failed:
        print(f"  failed:    {result.n_failed} (e.g. {result.failed[0]!r})")

    # Yield analysis: P5 / P50 / P95 of each metric across the LHS.
    print()
    print(f"{'metric':<48}  {'P5':>10}  {'P50':>10}  {'P95':>10}")
    print("-" * 85)
    for m in metrics:
        p5 = result.percentile(m.name, 5)
        p50 = result.percentile(m.name, 50)
        p95 = result.percentile(m.name, 95)
        print(f"{m.name:<48}  {p5:>10.4f}  {p50:>10.4f}  {p95:>10.4f}")

    if os.environ.get("PULSIM_EXAMPLE_NOPLOT"):
        return
    try:
        import matplotlib.pyplot as plt

        df = result.to_pandas()
        df_ok = df[df["__failed__"] == ""]
        # 2-D scatter: (L, C) coloured by ripple proxy = peak - steady-state
        ripple_proxy = df_ok["peak[out]"] - df_ok["steady_state[out]"]
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(df_ok["L"] * 1e6, df_ok["C"] * 1e6,
                        c=ripple_proxy * 1000, s=44, cmap="viridis")
        ax.set_xlabel("L (µH)")
        ax.set_ylabel("C (µF)")
        ax.set_title("Buck (L, C) sweep — peak-to-steady-state ripple (mV)")
        plt.colorbar(sc, ax=ax, label="ripple proxy (mV)")
        out_path = os.path.join(os.path.dirname(__file__), "07_lhs_pareto.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Saved Pareto: {out_path}")
    except ImportError:
        print("(matplotlib / pandas not installed — skipping plot)")


if __name__ == "__main__":
    main()
