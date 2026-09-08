"""Figure 3.2 — Buck inductor current at four values of Δt.

Simulate an idealised buck inductor under a square-wave drive
using closed-form analytic RL response within each half-period,
resampled with the trapezoidal rule at four step sizes.

Shows convergence as Δt → 0. The Δt=1 µs (red) trace clearly
overshoots the converged Δt=1 ns (dashed blue) reference at the
ripple peaks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def trapezoidal_RL_step(i_L, V_in, V_out, R, L, dt):
    return ((L - dt * R / 2.0) * i_L + dt * (V_in - V_out)) \
            / (L + dt * R / 2.0)


def simulate_buck_inductor(dt: float, n_periods: int = 2):
    f_sw    = 100e3
    T_sw    = 1.0 / f_sw
    duty    = 0.5
    L       = 100e-6
    R       = 1.0
    V_in    = 12.0
    V_out   = 5.0
    t_end   = n_periods * T_sw

    n_steps = int(np.ceil(t_end / dt)) + 1
    t  = np.linspace(0.0, t_end, n_steps)
    iL = np.zeros(n_steps)

    for k in range(n_steps - 1):
        ph = (t[k] % T_sw) / T_sw
        V_drive = V_in if ph < duty else 0.0
        iL[k + 1] = trapezoidal_RL_step(iL[k], V_drive, V_out,
                                         R, L, dt)

    return t, iL


def render(output_dir: Path) -> None:
    # Two-panel side-by-side: full waveform on the left, zoom on
    # the right. Much cleaner than the busy inline-inset approach.
    fig, (ax_full, ax_zoom) = plt.subplots(
        1, 2, figsize=(8.4, 3.6),
        gridspec_kw={"width_ratios": [1.6, 1.0]},
    )

    dts    = [1e-6, 1e-7, 1e-8, 1e-9]
    labels = [r"$\Delta t = 1\;\mu\mathrm{s}$",
              r"$\Delta t = 100\;\mathrm{ns}$",
              r"$\Delta t = 10\;\mathrm{ns}$",
              r"$\Delta t = 1\;\mathrm{ns}$ (reference)"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
    widths = [1.6, 1.6, 1.6, 1.0]
    styles = ["-", "-", "-", "--"]

    for dt, label, c, w, st in zip(dts, labels, colors, widths, styles):
        t, iL = simulate_buck_inductor(dt, n_periods=2)
        ax_full.plot(t * 1e6, iL, color=c, lw=w, ls=st, label=label)
        ax_zoom.plot(t * 1e6, iL, color=c, lw=w, ls=st)

    # Left panel: full waveform
    ax_full.set_xlabel(r"Time  ($\mu$s)")
    ax_full.set_ylabel(r"Inductor current $i_L$  (A)")
    ax_full.set_title("Full 2-period waveform", fontsize=10, pad=4)
    ax_full.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax_full.grid(True, alpha=0.3)
    ax_full.spines["top"].set_visible(False)
    ax_full.spines["right"].set_visible(False)

    # Right panel: zoom on the ripple peak around the first
    # half-period crossover
    ax_zoom.set_xlim(4.0, 6.0)
    ax_zoom.set_ylim(0.30, 0.55)
    ax_zoom.set_xlabel(r"Time  ($\mu$s)")
    ax_zoom.set_title("Zoom: first ripple peak (4–6 µs)",
                       fontsize=10, pad=4)
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.spines["top"].set_visible(False)
    ax_zoom.spines["right"].set_visible(False)

    # Connector lines between the full plot's zoom region and the
    # zoom panel — gives the reader a visual cue
    ax_full.axvspan(4.0, 6.0, color="#eaeaea", alpha=0.5,
                      zorder=0)

    fig.suptitle(
        r"Trapezoidal-discretised buck inductor current under PWM"
        "\n"
        r"(buck @ $f_{sw} = 100\,\mathrm{kHz}$, $L = 100\,\mu\mathrm{H}$,"
        r" $V_{in} = 12\,$V, $V_{out} = 5\,$V, duty $0.5$)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "fig04_trapezoidal_dt_sweep.png")
    fig.savefig(output_dir / "fig04_trapezoidal_dt_sweep.pdf")
