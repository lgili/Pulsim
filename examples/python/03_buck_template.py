"""Buck converter from a high-level template.

`pulsim.templates.buck(Vin=24, Vout=5, Iout=2, fsw=100e3)` returns a fully
wired Circuit with auto-sized L / C / R, a PWM voltage-controlled switch,
free-wheel diode, and resistive load. The duty cycle is pre-set to the
steady-state operating point.

This script demonstrates two things:

1. **Auto-design** — the template prints the resolved L / C / Rload and
   the design-decision notes so the user can sanity-check the heuristics
   against datasheet expectations.
2. **End-to-end pipeline** — DC OP → transient runs cleanly to completion
   on the segment-primary engine, with switching modelled in PWL Ideal
   mode (the contract from ``add-converter-templates`` Phase 8.1).

The open-loop output is not expected to settle exactly at the target —
that's what the PI compensator in ``docs/converter-templates.md`` is for.
The example just confirms the build → simulate path is healthy.

Run::

    python 03_buck_template.py

See also: docs/converter-templates.md
"""

from __future__ import annotations

import math
import os

import pulsim


def main() -> None:
    exp = pulsim.templates.buck(
        Vin=24.0,
        Vout=5.0,
        Iout=2.0,
        fsw=100_000.0,
        # ripple_pct=0.30, vout_ripple_pct=0.01 (defaults)
    )

    print(f"Topology: {exp.topology}")
    print(f"Auto-designed parameters:")
    for k in ("D", "L", "C", "Rload"):
        v = exp.parameters[k]
        note = exp.notes.get(k, "")
        unit = {"D": "", "L": " H", "C": " F", "Rload": " Ω"}[k]
        print(f"  {k:>5} = {v:.6g}{unit}    {note}")

    # Switch the converter to PWL Ideal mode so the segment-primary engine
    # resolves switch transitions analytically (much faster than Newton-DAE
    # on every PWM edge).
    exp.circuit.set_switching_mode_for_all(pulsim.SwitchingMode.Ideal)
    exp.circuit.set_pwl_state("Q1", False)
    exp.circuit.set_pwl_state("D1", False)

    opts = pulsim.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = 1e-3
    opts.dt = 1e-7
    opts.dt_min = 1e-12
    opts.dt_max = 1e-7
    opts.adaptive_timestep = False
    opts.integrator = pulsim.Integrator.BDF1
    opts.switching_mode = pulsim.SwitchingMode.Ideal
    opts.newton_options.num_nodes = exp.circuit.num_nodes()
    opts.newton_options.num_branches = exp.circuit.num_branches()

    sim = pulsim.Simulator(exp.circuit, opts)
    dc = sim.dc_operating_point()
    if not dc.success:
        raise SystemExit(f"DC OP failed: {dc.message}")

    result = sim.run_transient(dc.newton_result.solution)
    if not result.success:
        raise SystemExit(f"transient failed: {result.message}")

    out_idx = exp.circuit.get_node("out")
    V_final = result.states[-1][out_idx]
    print()
    print(f"Transient finished:")
    print(f"  samples written: {len(result.states)}")
    print(f"  V_final at out:  {V_final:+.4f} V")
    print(f"  |V_final| ≤ Vin: {abs(V_final) < exp.parameters['Vin'] + 5}  "
          f"(open-loop bound — closing the loop with a PI compensator settles to target)")
    print(f"  wallclock:       {result.total_time_seconds*1e3:.1f} ms")
    assert math.isfinite(V_final), "V_final is not finite"

    if os.environ.get("PULSIM_EXAMPLE_NOPLOT"):
        return
    try:
        import matplotlib.pyplot as plt

        time = list(result.time)
        vout = [s[out_idx] for s in result.states]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(time, vout, lw=0.6)
        ax.axhline(exp.parameters["Vout"], color="orange", linestyle="--",
                   label=f"target {exp.parameters['Vout']} V (closed-loop only)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("V_out (V)")
        ax.set_title(f"Buck — auto-designed, OPEN-LOOP transient "
                     f"(D={exp.parameters['D']:.3f}, "
                     f"L={exp.parameters['L']*1e6:.1f} µH, "
                     f"C={exp.parameters['C']*1e6:.1f} µF)")
        ax.legend(loc="best")
        ax.grid(True, linestyle=":", alpha=0.5)
        out_path = os.path.join(os.path.dirname(__file__), "03_buck_template.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Saved waveform: {out_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
