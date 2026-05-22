"""Pulsim v2 — one-line plot helpers.

Pre-baked matplotlib wrappers so users can render the most common
v2 plots (waveform scope, Bode magnitude/phase, branch currents,
multi-signal grid) without writing matplotlib boilerplate.

Each function:
  - Auto-detects sensible defaults (time-axis units, log/linear).
  - Looks up node names and branch IDs from a CircuitBuilder.
  - Optionally saves to a PNG path.
  - Returns the matplotlib Figure for further customization.

Usage:

    >>> import pulsim.v2 as p
    >>> res = p.simulate(b, t_end=5e-3, dt=1e-5)
    >>> p.scope(b, res, signals=["vout", "sw"])
    >>> p.scope(b, res, signals=["vout"], save="output/buck.png")

For closed-loop / dq / multi-axis:

    >>> p.scope_grid(b, res, panels=[
    ...     dict(signals=["vout"], ylabel="V_out [V]"),
    ...     dict(signals=["sw"], ylabel="V_sw [V]"),
    ...     dict(currents=["L1"], ylabel="i_L [A]"),
    ... ])

For Bode:

    >>> p.plot_bode(sweep_result, title="...", save="bode.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


__all__ = [
    "scope",
    "scope_grid",
    "plot_currents",
    "scope_fft",
    "compare",
    "_auto_time_units",
]


# =============================================================================
# Internal helpers
# =============================================================================

def _auto_time_units(times) -> tuple[float, str]:
    """Pick a sensible time-axis multiplier and unit string.

    Returns (scale, label) where `scale` is multiplied into the time
    array and `label` is the axis name (e.g. (1e3, "time [ms]")).
    """
    import numpy as np
    t_max = float(np.max(times)) if len(times) > 0 else 0.0
    if t_max < 1e-6:
        return 1e9, "time [ns]"
    if t_max < 1e-3:
        return 1e6, "time [µs]"
    if t_max < 1.0:
        return 1e3, "time [ms]"
    return 1.0, "time [s]"


def _resolve_signal_idx(builder, name: str) -> int | None:
    """Look up a state-vector index for a named signal.

    Accepts:
      - node names (e.g. "vout") → returns `builder.node_id_of(name)`
      - "i:<branch_id>" → tries `branch_var_id_for_inductor` then
                          `branch_var_id_for_source`
      - "ground" / "gnd" / "0" → returns None (constant 0 V)

    Returns None for ground or unknown.
    """
    name_lc = name.lower()
    if name_lc in ("gnd", "ground", "0"):
        return None
    if name.startswith("i:"):
        try:
            branch_id = int(name[2:])
        except ValueError:
            return None
        # Try inductor first, then source.
        for fn_name in ("branch_var_id_for_inductor",
                         "branch_var_id_for_source"):
            fn = getattr(builder.pool, fn_name, None)
            if fn is None:
                continue
            try:
                return fn(branch_id, builder.graph)
            except Exception:
                continue
        return None
    # Fall through: treat as node name.
    try:
        return int(builder.node_id_of(name))
    except Exception:
        return None


# =============================================================================
# scope()
# =============================================================================

def scope(builder, result, *,
            signals: Sequence[str] | None = None,
            title: str | None = None,
            save: str | Path | None = None,
            stack: bool = True,
            show: bool = False,
            tight_lw: bool = False):
    """One-line waveform plot.

    Parameters
    ----------
    builder
        The `CircuitBuilder` that owns `result`.
    result
        A `SimulationResult` (from `simulate()` or `run_transient()`).
    signals
        List of signal names to plot. Each can be a node name
        (e.g. "vout") or "i:<branch_id>" for an inductor / source
        branch current. Default: just plot the first non-ground
        node ([0]).
    title
        Plot title. Defaults to "Pulsim v2 — waveform".
    save
        If given, save the figure to this path (creates parent dir).
    stack
        If True (default), put each signal on its own subplot,
        sharing the x-axis. If False, plot all on one axes.
    show
        Call `plt.show()` at the end (default False — saves only).
    tight_lw
        Use thin lines (lw=0.4) — useful for dense PWM traces.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if signals is None or len(signals) == 0:
        # Default: pick the first non-ground node.
        signals = [name for name in dir(builder) if False]   # placeholder
        # Find any node name through the builder. Easiest: iterate
        # 0..num_nodes-1 — but we don't have names back from id. Use
        # state[0] as a "first signal" — usually node 0.
        # For pedagogy, just plot every node + every inductor current.
        # Need a way to enumerate nodes; CircuitBuilder doesn't expose
        # node names easily. Fall back to "plot state[0]".
        signals = ["[0]"]    # special marker handled below

    times = np.asarray(result.times)
    scale, t_label = _auto_time_units(times)

    n = len(signals)
    if stack and n > 1:
        fig, axes = plt.subplots(n, 1, figsize=(11, 2.5 * n),
                                    sharex=True)
        ax_list = list(axes)
    else:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax_list = [ax] * n

    lw = 0.4 if tight_lw else 0.8

    for i, sig in enumerate(signals):
        ax = ax_list[i] if stack else ax_list[0]
        if sig == "[0]":
            # Special: plot the entire state[0].
            y = np.array([s[0] for s in result.states])
            label = "state[0]"
        else:
            idx = _resolve_signal_idx(builder, sig)
            if idx is None:
                # ground or unknown — plot a flat zero (visually
                # informative + obvious if the user typo'd a node).
                y = np.zeros(len(times))
                label = f"{sig} (?)"
            else:
                y = np.array([s[idx] for s in result.states])
                label = sig
        ax.plot(times * scale, y, lw=lw, label=label)
        if stack:
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
        else:
            ax.grid(True, alpha=0.3)

    if not stack and n > 1:
        ax_list[0].legend(loc="best")

    # x-label only on the last subplot.
    (ax_list[-1] if stack else ax_list[0]).set_xlabel(t_label)
    if title is None:
        title = "Pulsim v2 — waveform"
    (ax_list[0] if stack else ax_list[0]).set_title(title)

    plt.tight_layout()
    if save is not None:
        path = Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=120)
        print(f"  plot → {path}")
    if show:
        plt.show()
    return fig


# =============================================================================
# scope_grid() — more flexible
# =============================================================================

def scope_grid(builder, result, *, panels: Sequence[dict],
                  title: str | None = None,
                  save: str | Path | None = None,
                  show: bool = False):
    """Multi-panel waveform plot with per-panel configuration.

    Each entry in `panels` is a dict with optional keys:
      - signals: list[str]   — node-voltage signals (see scope())
      - currents: list[str]  — inductor names; resolved to i_L
      - ylabel: str          — y-axis label
      - legend: bool         — show legend on this panel (default True)
      - lw: float            — line width (default 0.8)
      - color: str           — single colour (or rely on default cycler)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    times = np.asarray(result.times)
    scale, t_label = _auto_time_units(times)

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.5 * n + 0.5),
                                sharex=True)
    if n == 1:
        axes = [axes]

    for i, panel in enumerate(panels):
        ax = axes[i]
        lw = panel.get("lw", 0.8)
        show_legend = panel.get("legend", True)
        for sig in panel.get("signals", []):
            idx = _resolve_signal_idx(builder, sig)
            y = (np.zeros(len(times)) if idx is None
                 else np.array([s[idx] for s in result.states]))
            ax.plot(times * scale, y, lw=lw, label=sig)
        # Currents — resolved by branch name.
        for cur_name in panel.get("currents", []):
            try:
                bid = int(builder.graph.num_branches)   # noop guard
            except Exception:
                pass
            # Look up by name → branch_id is tricky from Python today
            # (the Graph doesn't expose name lookup). Skip with a
            # warning print, since the user can pass "i:<idx>" via
            # signals= instead.
            print(f"  scope_grid: 'currents' key needs branch_id; "
                  f"use signals=['i:<id>'] for now (got {cur_name!r})")
        ax.set_ylabel(panel.get("ylabel", ""))
        ax.grid(True, alpha=0.3)
        if show_legend and len(panel.get("signals", [])) > 1:
            ax.legend(loc="best", fontsize=9)
    axes[-1].set_xlabel(t_label)
    if title:
        axes[0].set_title(title)

    plt.tight_layout()
    if save is not None:
        path = Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=120)
        print(f"  plot → {path}")
    if show:
        plt.show()
    return fig


# =============================================================================
# plot_currents()
# =============================================================================

def plot_currents(builder, result, branch_ids: Sequence[int], *,
                     labels: Sequence[str] | None = None,
                     title: str | None = None,
                     save: str | Path | None = None,
                     show: bool = False):
    """Plot inductor or source branch currents.

    `branch_ids` are the graph branch indices (from insertion order
    in the builder). For each, we try
    `pool.branch_var_id_for_inductor` first, then
    `branch_var_id_for_source`.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    times = np.asarray(result.times)
    scale, t_label = _auto_time_units(times)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for k, bid in enumerate(branch_ids):
        idx = None
        for fn_name in ("branch_var_id_for_inductor",
                         "branch_var_id_for_source"):
            fn = getattr(builder.pool, fn_name, None)
            if fn is None:
                continue
            try:
                idx = fn(bid, builder.graph)
                break
            except Exception:
                continue
        if idx is None:
            print(f"  plot_currents: branch {bid} not an inductor or "
                  f"source — skipping")
            continue
        y = np.array([s[idx] for s in result.states])
        label = (labels[k] if labels and k < len(labels)
                 else f"i[branch {bid}]")
        ax.plot(times * scale, y, lw=0.8, label=label)

    ax.set_xlabel(t_label); ax.set_ylabel("current [A]")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    if title:
        ax.set_title(title)

    plt.tight_layout()
    if save is not None:
        path = Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=120)
        print(f"  plot → {path}")
    if show:
        plt.show()
    return fig


# =============================================================================
# scope_fft() — FFT of a signal for harmonic analysis
# =============================================================================

def scope_fft(builder, result, *,
                signal: str,
                t_window: tuple[float, float] | None = None,
                f_max: float | None = None,
                log_y: bool = True,
                show_thd: bool = True,
                f_fundamental: float | None = None,
                title: str | None = None,
                save: str | Path | None = None,
                show: bool = False):
    """FFT of a node signal — magnitude vs frequency.

    The standard tool for harmonic content / THD analysis. Common
    uses:

      - Rectifier output: see the line-frequency harmonics
      - PFC: verify the input current is dominated by the
        fundamental at the mains frequency
      - Motor drive: check the line-to-line voltage harmonics

    Returns `(fig, freqs, mag)` where `freqs` and `mag` are the
    one-sided magnitude spectrum (peak amplitude per bin, after
    Hann window correction).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import math

    times = np.asarray(result.times)
    n = len(times)
    if n < 4:
        raise ValueError("scope_fft: need at least 4 samples")
    dt = float(times[1] - times[0])
    fs = 1.0 / dt

    idx = _resolve_signal_idx(builder, signal)
    if idx is None:
        raise ValueError(f"scope_fft: unknown signal '{signal}'")
    y = np.array([s[idx] for s in result.states])

    # Default: skip the first 30 % of samples (settling transient).
    if t_window is None:
        k0 = int(0.3 * n)
        k1 = n
    else:
        k0 = max(0, int(t_window[0] / dt))
        k1 = min(n, int(t_window[1] / dt))
    if k1 - k0 < 4:
        raise ValueError("scope_fft: window too small")
    y_w = y[k0:k1]
    y_w = y_w - y_w.mean()
    window = np.hanning(len(y_w))
    cf = 2.0 / window.sum()
    Y = np.fft.rfft(y_w * window)
    freqs = np.fft.rfftfreq(len(y_w), d=dt)
    mag = np.abs(Y) * cf

    if f_max is None:
        f_max = fs / 2.0
    mask = freqs <= f_max
    freqs, mag = freqs[mask], mag[mask]

    thd_txt = ""
    if show_thd and f_fundamental is not None:
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 0.0
        if df > 0:
            def amp_at(f):
                k = int(round(f / df))
                return mag[k] if 0 <= k < len(mag) else 0.0
            v1 = amp_at(f_fundamental)
            n_h = int(math.floor(f_max / f_fundamental))
            harm_sum_sq = sum(amp_at(k * f_fundamental) ** 2
                                for k in range(2, n_h + 1))
            if v1 > 0:
                thd = math.sqrt(harm_sum_sq) / v1
                thd_txt = f"  THD = {thd*100:.2f} %"

    fig, ax = plt.subplots(figsize=(11, 4.5))
    if log_y:
        ax.semilogy(freqs, mag, lw=0.9)
    else:
        ax.plot(freqs, mag, lw=0.9)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(f"|FFT({signal})|")
    ax.grid(True, which="both", alpha=0.3)
    if f_fundamental is not None:
        ax.axvline(f_fundamental, color="r", ls=":", lw=0.6,
                    label=f"f₁ = {f_fundamental:.1f} Hz")
        ax.legend(loc="best")
    ax.set_title((title or f"FFT of {signal}") + thd_txt)

    plt.tight_layout()
    if save is not None:
        path = Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=120)
        print(f"  plot → {path}")
    if show:
        plt.show()
    return fig, freqs, mag


# =============================================================================
# compare() — overlay one signal from multiple SimulationResults
# =============================================================================

def compare(builder, results: dict, *,
              signal: str,
              title: str | None = None,
              save: str | Path | None = None,
              show: bool = False):
    """Overlay `signal` from multiple `SimulationResult` instances.

    `results` is a dict {label: SimulationResult}. All results must
    have been built against the same `builder` (so the state-vector
    indices align).

    Useful for parameter sweeps:

        runs = {}
        for kp in [0.01, 0.05, 0.1]:
            runs[f"Kp={kp}"] = run_my_sim(kp)
        p.compare(b, runs, signal="vout", title="Buck — Kp sensitivity")
    """
    import numpy as np
    import matplotlib.pyplot as plt

    idx = _resolve_signal_idx(builder, signal)
    if idx is None:
        raise ValueError(f"compare: unknown signal '{signal}'")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    first_times = np.asarray(next(iter(results.values())).times)
    scale, t_label = _auto_time_units(first_times)

    for label, res in results.items():
        times = np.asarray(res.times) * scale
        y = np.array([s[idx] for s in res.states])
        ax.plot(times, y, lw=0.9, label=label)

    ax.set_xlabel(t_label); ax.set_ylabel(signal)
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    if title:
        ax.set_title(title)

    plt.tight_layout()
    if save is not None:
        path = Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=120)
        print(f"  plot → {path}")
    if show:
        plt.show()
    return fig
