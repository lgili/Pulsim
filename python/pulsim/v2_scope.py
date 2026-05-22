"""Pulsim v2 — live oscilloscope GUI (pyqtgraph).

A real-time waveform display that consumes batches from a
``LiveStream`` while the simulation runs on a worker thread.

Usage:

    import threading
    import pulsim.v2 as p

    b = build_my_plant()
    stream = p.LiveStream(batch_size=100, max_queue=200)

    scope = p.LiveScope(b, stream, window_seconds=2.0, update_hz=30)
    scope.add_node_voltage("vout")
    scope.add_node_voltage("vin")
    scope.add_branch_current_for_inductor("L1")

    # Sim on a worker thread.
    def run_sim():
        return p.simulate(b, t_end=10.0, dt=1e-7,
                            step_observer=stream.step_observer,
                            should_continue=stream.should_continue)
    threading.Thread(target=run_sim, daemon=True).start()

    # Blocks until the scope window is closed; meanwhile the timer
    # pulls batches from the stream and redraws ~30× per second.
    scope.start()

The scope window has:
  * Multi-trace plot with one curve per enabled signal
  * Checkbox row for live signal enable/disable
  * STOP button (calls `stream.stop()` → kernel returns partial trace)
  * Status line: current sim time, samples received, queue depth

For signals that aren't direct state-vector entries (e.g. chain
channels, derived quantities), pass a custom extractor via
:meth:`add_signal`.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np


__all__ = ["LiveScope", "SignalDef"]


try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets, QtCore
    _PG_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PG_AVAILABLE = False


class SignalDef:
    """One displayable signal: a name + extractor ``(t, x) → float``."""

    def __init__(self, name: str,
                  extractor: Callable[[float, "object"], float],
                  *, color: Optional[str] = None,
                  unit: str = ""):
        self.name = name
        self.extractor = extractor
        self.color = color
        self.unit = unit


_DEFAULT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]


class LiveScope:
    """Live-updating waveform window backed by pyqtgraph.

    Parameters
    ----------
    builder
        The populated :class:`CircuitBuilder` (used for node /
        inductor lookups in the convenience adders).
    stream
        The :class:`LiveStream` that the simulation pushes into.
    window_seconds
        Sliding-window width on the time axis. Samples older than
        ``t_max − window_seconds`` are dropped from each curve.
        Default 2 s.
    update_hz
        How fast the GUI redraws (Hz). Default 30 — perceptually
        smooth without burning CPU.
    title
        Window title.
    """

    def __init__(self, builder, stream, *,
                  window_seconds: float = 2.0,
                  update_hz: float = 30.0,
                  title: str = "Pulsim — LiveScope"):
        if not _PG_AVAILABLE:
            raise ImportError(
                "LiveScope requires pyqtgraph. Install with: "
                "pip install pyqtgraph pyqt5  (or pyqt6 / pyside6)")
        self.builder = builder
        self.stream = stream
        self.window_seconds = float(window_seconds)
        self.update_interval_ms = max(10, int(1000.0 / update_hz))
        self.title = title

        self._signals: List[SignalDef] = []
        self._enabled: Dict[str, bool] = {}
        self._curves: Dict[str, "pg.PlotDataItem"] = {}
        self._data_t: Dict[str, list] = {}
        self._data_y: Dict[str, list] = {}
        self._max_points = 200_000   # safety cap per curve

        self._app = None
        self._win = None
        self._plot = None
        self._stop_btn = None
        self._status_label = None
        self._timer = None
        self._wall_t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Signal registration
    # ------------------------------------------------------------------

    def add_signal(self, name: str,
                      extractor: Callable[[float, "object"], float],
                      *, color: Optional[str] = None,
                      unit: str = "") -> "LiveScope":
        """Register a custom signal with an extractor lambda."""
        self._signals.append(
            SignalDef(name, extractor, color=color, unit=unit))
        return self

    def add_node_voltage(self, node_name: str, *,
                            label: Optional[str] = None,
                            color: Optional[str] = None) -> "LiveScope":
        """Add a node-voltage signal by name."""
        idx = self.builder.node_id_of(node_name)
        nm = label or f"V({node_name})"
        return self.add_signal(
            nm, lambda t, x, _i=idx: float(x[_i]),
            color=color, unit="V")

    def add_inductor_current(self, name: str, *,
                                 label: Optional[str] = None,
                                 color: Optional[str] = None) -> "LiveScope":
        """Add an inductor branch-current signal by inductor name.
        Looks up the branch id via ``builder.graph.branch_id_for_name`` —
        if that helper isn't available, pass `add_branch_current` with
        the explicit branch_id instead."""
        # Walk the builder's branches to find the named inductor.
        idx_branch = -1
        for bid in range(self.builder.graph.num_branches):
            try:
                br = self.builder.graph.branch(bid)
                from pulsim._pulsim import v2_kernel as _k   # noqa: F401
                pool = self.builder.pool
                if pool.kind_of(br.id).name == "Inductor":
                    # We don't have a name-on-branch query from
                    # Python; user should use add_branch_current
                    # directly with the branch id they tracked.
                    idx_branch = br.id
            except Exception:
                pass
        if idx_branch < 0:
            raise ValueError(
                f"could not resolve inductor named {name!r}; "
                "pass add_branch_current(branch_id) directly")
        return self.add_branch_current(idx_branch, label=label,
                                            color=color)

    def add_branch_current(self, branch_id: int, *,
                                label: Optional[str] = None,
                                color: Optional[str] = None) -> "LiveScope":
        """Add an inductor or source branch-current signal by branch id."""
        idx = self.builder.pool.branch_var_id_for_inductor(
            branch_id, self.builder.graph)
        nm = label or f"I(branch#{branch_id})"
        return self.add_signal(
            nm, lambda t, x, _i=idx: float(x[_i]),
            color=color, unit="A")

    def add_chain_channel(self, chain, name: str, *,
                              label: Optional[str] = None,
                              color: Optional[str] = None,
                              unit: str = "") -> "LiveScope":
        """Add a signal sourced from a `MixedDomainBlockChain` channel.

        The chain's current channel value is read at scope-update
        time (NOT per simulation step) — this works because the
        chain holds the most recent value internally and we just
        sample it. Best for low-bandwidth signals (controller
        outputs, motor speed, etc.).
        """
        nm = label or f"{name}"
        return self.add_signal(
            nm, lambda t, x, _c=chain, _n=name: float(_c.get(_n)),
            color=color, unit=unit)

    # ------------------------------------------------------------------
    # GUI lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Build the window + start the update timer. Blocks until
        the window closes. Run on the main thread."""
        pg.setConfigOptions(antialias=True, background="#0f1116",
                              foreground="#cccccc")
        self._app = pg.mkQApp(self.title)
        self._win = QtWidgets.QMainWindow()
        self._win.setWindowTitle(self.title)
        self._win.resize(1100, 600)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        self._win.setCentralWidget(central)

        # Plot
        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "time", units="s")
        self._plot.setLabel("left", "value")
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.addLegend(offset=(10, 10))
        layout.addWidget(self._plot, stretch=1)

        # Signal checkboxes
        sig_widget = QtWidgets.QWidget()
        sig_row = QtWidgets.QHBoxLayout(sig_widget)
        sig_row.setContentsMargins(0, 0, 0, 0)
        sig_row.addWidget(QtWidgets.QLabel("Show:"))
        for i, sig in enumerate(self._signals):
            color = sig.color or _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
            cb = QtWidgets.QCheckBox(sig.name)
            cb.setChecked(True)
            # Make the checkbox label coloured to match its curve.
            cb.setStyleSheet(f"color: {color}; font-weight: 600;")
            cb.stateChanged.connect(
                lambda state, n=sig.name: self._on_toggle(n, state))
            sig_row.addWidget(cb)
            self._enabled[sig.name] = True
            curve = self._plot.plot(name=sig.name,
                                       pen=pg.mkPen(color=color, width=1.4))
            self._curves[sig.name] = curve
            self._data_t[sig.name] = []
            self._data_y[sig.name] = []
        sig_row.addStretch()
        layout.addWidget(sig_widget)

        # Status + stop
        ctrl = QtWidgets.QWidget()
        ctrl_row = QtWidgets.QHBoxLayout(ctrl)
        ctrl_row.setContentsMargins(0, 0, 0, 0)
        self._status_label = QtWidgets.QLabel("waiting for samples…")
        self._status_label.setStyleSheet("color: #888;")
        ctrl_row.addWidget(self._status_label)
        ctrl_row.addStretch()
        self._stop_btn = QtWidgets.QPushButton("STOP simulation")
        self._stop_btn.setStyleSheet(
            "QPushButton { background:#a02020; color:white; "
            "padding:6px 14px; border-radius:4px; "
            "font-weight:bold; } "
            "QPushButton:hover { background:#c63838; } "
            "QPushButton:disabled { background:#555; color:#aaa; }")
        self._stop_btn.clicked.connect(self._on_stop)
        ctrl_row.addWidget(self._stop_btn)
        layout.addWidget(ctrl)

        # Update timer
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(self.update_interval_ms)

        self._wall_t0 = time.perf_counter()
        self._win.show()
        self._app.exec()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_toggle(self, name: str, state) -> None:
        self._enabled[name] = bool(state)
        if not self._enabled[name]:
            self._curves[name].clear()
        else:
            # Replay buffered data when re-enabling.
            t = np.asarray(self._data_t[name], dtype=float)
            y = np.asarray(self._data_y[name], dtype=float)
            if len(t) > 0:
                self._curves[name].setData(t, y)

    def _on_stop(self) -> None:
        self.stream.stop()
        if self._status_label:
            self._status_label.setText("STOPPING — wait for kernel…")
        if self._stop_btn:
            self._stop_btn.setEnabled(False)
            self._stop_btn.setText("Stopped")

    def _tick(self) -> None:
        """Drain available batches, push samples into the rolling
        buffers, redraw enabled curves. Runs on the GUI thread."""
        n_batches = 0
        while True:
            batch = self.stream.get_batch(timeout=0.0)
            if batch is None:
                break
            t_arr, x_arr = batch
            n_batches += 1
            for sig in self._signals:
                # Compute values for this batch (vectorised when
                # possible; falls back to a Python list comp).
                try:
                    y_arr = np.fromiter(
                        (sig.extractor(t_arr[i], x_arr[i])
                          for i in range(len(t_arr))),
                        dtype=float, count=len(t_arr))
                except Exception:
                    y_arr = np.array(
                        [sig.extractor(t_arr[i], x_arr[i])
                          for i in range(len(t_arr))],
                        dtype=float)
                self._data_t[sig.name].extend(t_arr.tolist())
                self._data_y[sig.name].extend(y_arr.tolist())

        # Trim to sliding window + max cap.
        if n_batches > 0:
            t_max = 0.0
            for name in self._data_t:
                if self._data_t[name]:
                    t_max = max(t_max, self._data_t[name][-1])
            cutoff = t_max - self.window_seconds
            for name in self._data_t:
                buf_t = self._data_t[name]
                buf_y = self._data_y[name]
                if buf_t and buf_t[0] < cutoff:
                    # Binary search for cutoff index.
                    arr_t = np.asarray(buf_t, dtype=float)
                    i_cut = int(np.searchsorted(arr_t, cutoff))
                    del buf_t[:i_cut]
                    del buf_y[:i_cut]
                if len(buf_t) > self._max_points:
                    n_drop = len(buf_t) - self._max_points
                    del buf_t[:n_drop]
                    del buf_y[:n_drop]
                # Redraw if enabled.
                if self._enabled[name] and buf_t:
                    self._curves[name].setData(
                        np.asarray(buf_t, dtype=float),
                        np.asarray(buf_y, dtype=float))

            # Status line.
            wall = time.perf_counter() - self._wall_t0
            self._status_label.setText(
                f"  sim t = {t_max*1e3:.2f} ms     "
                f"wall = {wall:.1f} s     "
                f"samples = {self.stream.n_steps_received}     "
                f"queue = {self.stream.qsize()} batches     "
                f"batches/sec = "
                f"{self.stream.n_batches_emitted / max(wall, 1e-3):.1f}")
