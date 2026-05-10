"""Monte-Carlo yield analysis under component tolerance.

Real R / L / C have ±1–10 % tolerance. This script runs 256 Monte-Carlo
draws over (R, L, C) ~ N(nominal, σ) and reports the fraction of designs
that meet a steady-state-Vout window — the classic six-sigma "yield"
question.

The tolerances chosen here (5 % on L / C, 1 % on R) are aggressive enough
to push some samples *just* outside the ±2 % output spec, so the run
produces a non-trivial yield number.

Run::

    python 08_monte_carlo_yield.py

See also: docs/parameter-sweep.md
"""

from __future__ import annotations

import os

import pulsim


def make_buck(L: float, C: float, R: float):
    """Buck factory consumed by the sweep harness."""
    exp = pulsim.templates.buck(
        Vin=24.0,
        Vout=5.0,
        Iout=2.0,
        fsw=100_000.0,
        L=L,
        C=C,
        Rload=R,
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
    L_nom = 80e-6
    C_nom = 30e-6
    R_nom = 2.5

    # 5 % L/C tolerance, 1 % R tolerance — typical for off-the-shelf parts.
    spec = {
        "L": pulsim.sweep.Distribution.normal(L_nom, 0.05 * L_nom),
        "C": pulsim.sweep.Distribution.normal(C_nom, 0.05 * C_nom),
        "R": pulsim.sweep.Distribution.normal(R_nom, 0.01 * R_nom),
    }
    metrics = [pulsim.sweep.steady_state("out")]

    print("Running 256-point Monte-Carlo yield sweep ...")
    result = pulsim.sweep.run(
        circuit_factory=make_buck,
        parameters=spec,
        metrics=metrics,
        n_samples=256,
        strategy="monte_carlo",
        seed=2026,
        sim_options_factory=make_options,
    )
    print(f"  succeeded: {result.n_succeeded} / {result.n_samples}    "
          f"wall: {result.wall_seconds:.2f} s")     # SweepResult.wall_seconds

    # Spec: Vout in 5 V ± 2 %.
    Vlo, Vhi = 4.9, 5.1
    in_spec = 0
    total_ok = 0
    for k in range(result.n_samples):
        if result.failed[k] is not None:
            continue
        v = result.metrics[k]["steady_state[out]"]
        total_ok += 1
        if Vlo <= v <= Vhi:
            in_spec += 1
    yield_pct = 100.0 * in_spec / max(total_ok, 1)
    print()
    print(f"Spec window: {Vlo} V ≤ Vout ≤ {Vhi} V")
    print(f"In-spec: {in_spec} / {total_ok}    yield = {yield_pct:.2f} %")
    print(f"Vout P5 = {result.percentile('steady_state[out]',  5):.4f} V")
    print(f"Vout P50 = {result.percentile('steady_state[out]', 50):.4f} V")
    print(f"Vout P95 = {result.percentile('steady_state[out]', 95):.4f} V")

    if os.environ.get("PULSIM_EXAMPLE_NOPLOT"):
        return
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        df = result.to_pandas()
        df_ok = df[df["__failed__"] == ""]
        v = df_ok["steady_state[out]"].values
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(v, bins=32, color="steelblue", alpha=0.85)
        ax.axvline(Vlo, color="red", linestyle="--", label="spec ±2 %")
        ax.axvline(Vhi, color="red", linestyle="--")
        ax.axvline(np.mean(v), color="orange", linestyle="-", label=f"mean = {np.mean(v):.3f} V")
        ax.set_xlabel("Vout steady-state (V)")
        ax.set_ylabel("samples")
        ax.set_title(f"Monte-Carlo yield over component tolerance — "
                     f"yield = {yield_pct:.1f} %")
        ax.legend()
        out_path = os.path.join(os.path.dirname(__file__), "08_yield_histogram.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Saved histogram: {out_path}")
    except ImportError:
        print("(matplotlib / pandas not installed — skipping plot)")


if __name__ == "__main__":
    main()
