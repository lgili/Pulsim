"""Frequency-domain analysis plotting helpers.

Phase 5 of `add-frequency-domain-analysis`. Wraps the C++-bound
`AcSweepResult` and `FraResult` types in matplotlib-friendly plotting
helpers so the user can go from an AC sweep to a Bode plot in one call.

Usage::

    import pulsim
    sim = pulsim.Simulator(circuit, opts)
    ac = pulsim.AcSweepOptions()
    ac.f_start = 1.0
    ac.f_stop = 1e6
    ac.points_per_decade = 30
    ac.perturbation_source = "Vin"
    ac.measurement_nodes = ["vout"]
    result = sim.run_ac_sweep(ac)

    fig, axes = pulsim.bode_plot(result)        # one Bode pair per node
    pulsim.nyquist_plot(result, ax=ax)          # all nodes on one Nyquist
    pulsim.fra_overlay(ac_result, fra_result)   # AC vs FRA side-by-side

All helpers accept an optional `ax` / `axes` parameter so the user can
embed plots in custom figures. matplotlib is imported lazily so this
module doesn't drag matplotlib into the import path of users who only
want time-domain simulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


__all__ = [
    "bode_plot",
    "nyquist_plot",
    "fra_overlay",
    "export_ac_csv",
    "export_fra_csv",
    "export_ac_json",
    "export_fra_json",
    "load_ac_result_csv",
]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        return plt
    except ImportError as exc:
        raise ImportError(
            "pulsim.frequency_analysis plotting helpers require matplotlib. "
            "Install with: pip install matplotlib"
        ) from exc


def _measurement_label(measurement) -> str:
    """Best-effort human-readable label for an AC/FRA measurement."""
    node = getattr(measurement, "node", "") or ""
    if node:
        return node
    idx = getattr(measurement, "state_index", -1)
    return f"state[{idx}]" if idx >= 0 else "<unnamed>"


def bode_plot(
    result,
    measurements: Sequence[str] | None = None,
    ax_mag: "Axes | None" = None,
    ax_phase: "Axes | None" = None,
    *,
    title: str | None = None,
    legend: bool = True,
    grid: bool = True,
    color=None,
    linestyle: str = "-",
    label_prefix: str = "",
):
    """Bode magnitude / phase plot of an `AcSweepResult` or `FraResult`.

    Args:
        result: `AcSweepResult` or `FraResult` from `Simulator.run_ac_sweep`
            / `Simulator.run_fra`.
        measurements: optional list of node names to plot. If None, plot
            every measurement contained in the result.
        ax_mag, ax_phase: matplotlib Axes for magnitude / phase. If None,
            a new figure with two stacked axes is created.
        title: figure title (only used when creating a new figure).
        legend: draw a per-node legend.
        grid: enable grid lines.
        color, linestyle, label_prefix: styling knobs forwarded to
            matplotlib's `plot` for every measurement curve. `label_prefix`
            is prepended to the measurement label so users can overlay
            multiple results on the same axes (e.g. AC vs FRA).

    Returns:
        `(fig, (ax_mag, ax_phase))` so users can further customize.
    """
    plt = _require_matplotlib()

    if not getattr(result, "success", False):
        raise ValueError(
            f"Cannot plot: result is not successful "
            f"(failure_reason={result.failure_reason!r})"
        )

    freqs = list(result.frequencies)
    if not freqs:
        raise ValueError("Cannot plot: result has no frequency points")

    selected = list(result.measurements)
    if measurements is not None:
        wanted = set(measurements)
        selected = [m for m in selected if _measurement_label(m) in wanted]
        if not selected:
            raise ValueError(
                f"None of the requested nodes match: {sorted(wanted)}"
            )

    fig: "Figure | None" = None
    if ax_mag is None or ax_phase is None:
        fig, axes_pair = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax_mag, ax_phase = axes_pair
        if title is not None:
            fig.suptitle(title)
    assert ax_mag is not None and ax_phase is not None  # narrowing for type-checkers

    for m in selected:
        label = f"{label_prefix}{_measurement_label(m)}"
        kwargs: dict = {"label": label, "linestyle": linestyle}
        if color is not None:
            kwargs["color"] = color
        ax_mag.semilogx(freqs, m.magnitude_db, **kwargs)
        ax_phase.semilogx(freqs, m.phase_deg, **kwargs)

    ax_mag.set_ylabel("Magnitude (dB)")
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.set_xlabel("Frequency (Hz)")

    if grid:
        ax_mag.grid(True, which="both", linestyle=":", alpha=0.5)
        ax_phase.grid(True, which="both", linestyle=":", alpha=0.5)
    if legend and (measurements is not None or len(selected) > 1):
        ax_mag.legend(loc="best")

    return fig, (ax_mag, ax_phase)


def nyquist_plot(
    result,
    measurements: Sequence[str] | None = None,
    ax: "Axes | None" = None,
    *,
    title: str | None = None,
    unit_circle: bool = False,
    legend: bool = True,
    grid: bool = True,
    color=None,
    linestyle: str = "-",
    label_prefix: str = "",
):
    """Nyquist plot of an `AcSweepResult` or `FraResult`.

    Plots Im(H) vs Re(H) for each measurement. Use `unit_circle=True` to
    overlay the |H|=1 reference (useful for stability margin reading).
    Returns `(fig, ax)`.
    """
    plt = _require_matplotlib()

    if not getattr(result, "success", False):
        raise ValueError(
            f"Cannot plot: result is not successful "
            f"(failure_reason={result.failure_reason!r})"
        )

    selected = list(result.measurements)
    if measurements is not None:
        wanted = set(measurements)
        selected = [m for m in selected if _measurement_label(m) in wanted]
        if not selected:
            raise ValueError(
                f"None of the requested nodes match: {sorted(wanted)}"
            )

    fig: "Figure | None" = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
        if title is not None:
            fig.suptitle(title)
    assert ax is not None  # narrowing for type-checkers

    for m in selected:
        label = f"{label_prefix}{_measurement_label(m)}"
        kwargs: dict = {"label": label, "linestyle": linestyle}
        if color is not None:
            kwargs["color"] = color
        ax.plot(m.real_part, m.imag_part, **kwargs)

    if unit_circle:
        import math

        theta = [i * 2 * math.pi / 200 for i in range(201)]
        ax.plot(
            [math.cos(t) for t in theta],
            [math.sin(t) for t in theta],
            linestyle="--",
            color="gray",
            alpha=0.5,
            label="|H|=1",
        )

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Re H(jω)")
    ax.set_ylabel("Im H(jω)")
    ax.set_aspect("equal", adjustable="datalim")
    if grid:
        ax.grid(True, linestyle=":", alpha=0.5)
    if legend:
        ax.legend(loc="best")

    return fig, ax


def fra_overlay(
    ac_result,
    fra_result,
    measurements: Sequence[str] | None = None,
    *,
    title: str = "AC sweep vs FRA",
    legend: bool = True,
):
    """Side-by-side Bode overlay of an `AcSweepResult` and a matching
    `FraResult`. Useful for the regression contract "FRA agrees with AC
    sweep within 1 dB / 5°" — visualizes the gap.

    Returns `(fig, (ax_mag, ax_phase))`.
    """
    plt = _require_matplotlib()

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    fig.suptitle(title)

    bode_plot(
        ac_result,
        measurements=measurements,
        ax_mag=ax_mag,
        ax_phase=ax_phase,
        legend=False,
        linestyle="-",
        label_prefix="AC ",
    )
    bode_plot(
        fra_result,
        measurements=measurements,
        ax_mag=ax_mag,
        ax_phase=ax_phase,
        legend=False,
        linestyle="--",
        label_prefix="FRA ",
    )

    if legend:
        ax_mag.legend(loc="best")

    return fig, (ax_mag, ax_phase)


def export_ac_csv(result, path: str, *, format: str = "magphase") -> None:
    """Export an `AcSweepResult` to CSV.

    `format`:
        - `"magphase"`: `frequency_hz, mag_db_<node1>, phase_deg_<node1>, ...`
        - `"complex"` : `frequency_hz, real_<node1>, imag_<node1>, ...`
    """
    if not getattr(result, "success", False):
        raise ValueError(
            f"Cannot export: result is not successful "
            f"(failure_reason={result.failure_reason!r})"
        )
    if format not in ("magphase", "complex"):
        raise ValueError(f"format must be 'magphase' or 'complex', got {format!r}")

    import csv

    rows = []
    headers = ["frequency_hz"]
    for m in result.measurements:
        label = _measurement_label(m)
        if format == "magphase":
            headers += [f"mag_db_{label}", f"phase_deg_{label}"]
        else:
            headers += [f"real_{label}", f"imag_{label}"]

    freqs = list(result.frequencies)
    for i, f in enumerate(freqs):
        row = [f]
        for m in result.measurements:
            if format == "magphase":
                row += [m.magnitude_db[i], m.phase_deg[i]]
            else:
                row += [m.real_part[i], m.imag_part[i]]
        rows.append(row)

    with open(path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(headers)
        w.writerows(rows)


def export_fra_csv(result, path: str, *, format: str = "magphase") -> None:
    """Export a `FraResult` to CSV. Same `format` semantics as
    `export_ac_csv`."""
    # FraResult has the same shape as AcSweepResult for plotting/export
    # purposes — measurements carry magnitude_db / phase_deg / real_part /
    # imag_part — so we can dispatch through the same code path.
    export_ac_csv(result, path, format=format)


def _result_to_dict(result) -> dict:
    """Lower an AcSweepResult / FraResult to a JSON-serializable dict."""
    payload: dict = {
        "kind": "AcSweepResult" if hasattr(result, "total_factorizations")
                                  else "FraResult",
        "success": bool(result.success),
        "failure_reason": result.failure_reason,
        "frequencies": list(result.frequencies),
        "wall_seconds": float(result.wall_seconds),
        "measurements": [],
    }
    if hasattr(result, "total_factorizations"):
        payload["total_factorizations"] = int(result.total_factorizations)
        payload["total_solves"]         = int(result.total_solves)
    if hasattr(result, "total_transient_steps"):
        payload["total_transient_steps"] = int(result.total_transient_steps)

    for m in result.measurements:
        entry = {
            "node":          m.node,
            "state_index":   int(m.state_index),
            "magnitude_db":  list(m.magnitude_db),
            "phase_deg":     list(m.phase_deg),
            "real_part":     list(m.real_part),
            "imag_part":     list(m.imag_part),
        }
        # Phase 4 multi-input matrix: only emitted when the source label is
        # set (single-source sweeps from Phases 2/3 leave it empty).
        src = getattr(m, "perturbation_source", "")
        if src:
            entry["perturbation_source"] = src
        payload["measurements"].append(entry)
    return payload


def export_ac_json(result, path: str) -> None:
    """Export an `AcSweepResult` to JSON.

    The schema matches the C++ struct field-for-field. Round-trips with
    `load_ac_result_csv` is not needed since the Python consumer can
    `json.load(...)` the file and access fields directly.
    """
    if not getattr(result, "success", False):
        raise ValueError(
            f"Cannot export: result is not successful "
            f"(failure_reason={result.failure_reason!r})"
        )
    import json
    with open(path, "w") as fp:
        json.dump(_result_to_dict(result), fp, indent=2)


def export_fra_json(result, path: str) -> None:
    """Export a `FraResult` to JSON. Same schema as `export_ac_json`."""
    export_ac_json(result, path)


class _LoadedMeasurement:
    """Plain-Python stand-in for the C++ `AcMeasurement` returned by
    `load_ac_result_csv`. Carries the same fields plus an empty
    `perturbation_source` for single-source sweeps."""
    __slots__ = (
        "node", "state_index", "perturbation_source",
        "magnitude_db", "phase_deg", "real_part", "imag_part",
    )

    def __init__(self) -> None:
        self.node = ""
        self.state_index = -1
        self.perturbation_source = ""
        self.magnitude_db: list = []
        self.phase_deg: list = []
        self.real_part: list = []
        self.imag_part: list = []


class LoadedAcResult:
    """Plain-Python stand-in for the C++ `AcSweepResult` returned by
    `load_ac_result_csv`. Quacks like the binding type for the plotting
    helpers (`bode_plot`, `nyquist_plot`)."""
    __slots__ = ("success", "failure_reason", "frequencies", "measurements")

    def __init__(self) -> None:
        self.success = False
        self.failure_reason = ""
        self.frequencies: list = []
        self.measurements: list = []


def load_ac_result_csv(path: str, *, format: str = "magphase") -> LoadedAcResult:
    """Read a CSV produced by `export_ac_csv` back into a
    `LoadedAcResult`. The returned object is a plain Python container
    that the plotting helpers (`bode_plot`, `nyquist_plot`) accept
    transparently — useful for replaying a sweep result from disk
    without re-running the simulation.
    """
    if format not in ("magphase", "complex"):
        raise ValueError(f"format must be 'magphase' or 'complex', got {format!r}")

    import csv

    with open(path) as fp:
        reader = csv.reader(fp)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV file is empty: {path}")
    header = rows[0]
    data_rows = rows[1:]

    if header[0] != "frequency_hz":
        raise ValueError(
            f"CSV header missing 'frequency_hz' as first column (got {header[0]!r})"
        )

    out = LoadedAcResult()
    out.success = True

    if format == "magphase":
        # Each measurement consumes 2 columns: mag_db_<node>, phase_deg_<node>
        n_meas = (len(header) - 1) // 2
        for k in range(n_meas):
            mag_col = header[1 + 2 * k]
            phase_col = header[1 + 2 * k + 1]
            assert mag_col.startswith("mag_db_"), mag_col
            assert phase_col.startswith("phase_deg_"), phase_col
            m = _LoadedMeasurement()
            m.node = mag_col[len("mag_db_"):]
            out.measurements.append(m)
    else:
        n_meas = (len(header) - 1) // 2
        for k in range(n_meas):
            real_col = header[1 + 2 * k]
            # imag_col header (next column) is just paired bookkeeping; we
            # extract the node label from `real_col` and walk both columns
            # together when the data rows are read below.
            m = _LoadedMeasurement()
            m.node = real_col[len("real_"):]
            out.measurements.append(m)

    for row in data_rows:
        out.frequencies.append(float(row[0]))
        for k, m in enumerate(out.measurements):
            if format == "magphase":
                m.magnitude_db.append(float(row[1 + 2 * k]))
                m.phase_deg.append(float(row[1 + 2 * k + 1]))
            else:
                re = float(row[1 + 2 * k])
                im = float(row[1 + 2 * k + 1])
                m.real_part.append(re)
                m.imag_part.append(im)

    # Phase / mag fill-in for the case the user wants to plot a CSV
    # written in `complex` format: derive magnitude_db / phase_deg from
    # real / imag.
    if format == "complex":
        import math
        for m in out.measurements:
            for re, im in zip(m.real_part, m.imag_part):
                mag = math.sqrt(re * re + im * im)
                m.magnitude_db.append(20.0 * math.log10(mag) if mag > 1e-300 else -300.0)
                m.phase_deg.append(math.degrees(math.atan2(im, re)))

    return out
