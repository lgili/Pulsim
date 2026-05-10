"""Boost and Buck-Boost templates — auto-design and pipeline check.

Companion to ``03_buck_template.py``: prints the auto-designed L / C / R
for each topology and confirms the build → DC OP → transient pipeline
runs cleanly (matches the Phase 8.1 contract from
``add-converter-templates``).

The open-loop transient is bounded but not settled at the target Vout —
closing the loop with a PI compensator is the production workflow,
documented in ``docs/converter-templates.md``.

Run::

    python 04_boost_buckboost_templates.py

See also: docs/converter-templates.md
"""

from __future__ import annotations

import math
import os

import pulsim


def simulate_topology(exp, label: str, *, tstop: float = 1e-3, dt: float = 1e-7):
    """Run a transient on a TemplateExpansion and return (time, vout)."""
    exp.circuit.set_switching_mode_for_all(pulsim.SwitchingMode.Ideal)
    exp.circuit.set_pwl_state("Q1", False)
    exp.circuit.set_pwl_state("D1", False)

    opts = pulsim.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = tstop
    opts.dt = dt
    opts.dt_min = 1e-12
    opts.dt_max = dt
    opts.adaptive_timestep = False
    opts.integrator = pulsim.Integrator.BDF1
    opts.switching_mode = pulsim.SwitchingMode.Ideal
    opts.newton_options.num_nodes = exp.circuit.num_nodes()
    opts.newton_options.num_branches = exp.circuit.num_branches()

    sim = pulsim.Simulator(exp.circuit, opts)
    dc = sim.dc_operating_point()
    if not dc.success:
        raise SystemExit(f"[{label}] DC OP failed: {dc.message}")
    result = sim.run_transient(dc.newton_result.solution)
    if not result.success:
        raise SystemExit(f"[{label}] transient failed: {result.message}")
    out_idx = exp.circuit.get_node("out")
    time = list(result.time)
    vout = [s[out_idx] for s in result.states]
    return time, vout, result.total_time_seconds


def main() -> None:
    Vin = 12.0
    Iout = 1.0
    fsw = 100e3

    runs = [
        ("buck",       pulsim.templates.buck(Vin=Vin,  Vout=5.0,  Iout=Iout, fsw=fsw)),
        ("boost",      pulsim.templates.boost(Vin=Vin, Vout=24.0, Iout=Iout, fsw=fsw)),
        ("buck_boost", pulsim.templates.buck_boost(Vin=Vin, Vout=15.0, Iout=Iout, fsw=fsw)),
    ]

    print(f"{'topology':<12}  {'Vin':>5} → {'Vout':>5}  {'D':>6}  {'L (H)':>10}  "
          f"{'C (F)':>10}  {'Rload (Ω)':>10}")
    print("-" * 70)
    traces: list[tuple[str, list[float], list[float], float]] = []
    for label, exp in runs:
        p = exp.parameters
        print(f"{label:<12}  {p['Vin']:>5.1f}  {p['Vout']:>5.1f}  "
              f"{p['D']:>6.3f}  {p['L']:>10.3e}  {p['C']:>10.3e}  {p['Rload']:>10.3f}")
        time, vout, wall = simulate_topology(exp, label)
        V_final = vout[-1]
        assert math.isfinite(V_final), f"[{label}] V_final not finite"
        print(f"  V_final = {V_final:>+10.4f} V    "
              f"|V_final| < 100 V: {abs(V_final) < 100}    "
              f"wall = {wall*1e3:.1f} ms")
        traces.append((label, time, vout, p["Vout"]))

    print()
    print("Note: open-loop V_final is bounded but not settled at the target.")
    print("Production workflow closes the loop with the PI compensator")
    print("documented in docs/converter-templates.md.")

    if os.environ.get("PULSIM_EXAMPLE_NOPLOT"):
        return
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for ax, (label, t, v, target) in zip(axes, traces):
            ax.plot(t, v, lw=0.6)
            ax.axhline(target, color="orange", linestyle="--",
                       label=f"target {target} V (closed-loop only)")
            ax.set_ylabel(f"{label}\nV_out (V)")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.5)
        axes[-1].set_xlabel("time (s)")
        fig.suptitle("Three converter templates — auto-designed, OPEN-LOOP transient")
        out_path = os.path.join(os.path.dirname(__file__), "04_three_templates.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Saved: {out_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
