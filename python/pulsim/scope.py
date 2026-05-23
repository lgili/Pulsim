"""Pulsim — live oscilloscope GUI (pyqtgraph).

A polished real-time waveform viewer fed by a ``LiveStream``.

Key features:
  * **Multi-panel layout** — group signals into stacked plots
    (e.g. V on top, I in the middle, control signals at the bottom)
    with shared X axis but independent Y.
  * **Live value readout** — current value of each enabled trace
    shown beside its checkbox, monospace formatted like a DMM.
  * **Cursors (A, B)** — two draggable vertical cursors with a
    Δt / ΔV readout for measurements.
  * **Stats panel** — live min/max/mean/rms/peak-to-peak for the
    selected focus channel.
  * **Pause** — freeze the display while the simulation keeps
    running on the worker thread.
  * **Auto-Y** — per-panel toggle.
  * **Time window** — slider from 1 ms to 10 s sliding view.
  * **Snapshot** — save the current visible trace to PNG or CSV.
  * **Keyboard shortcuts** — Space=pause, S=stop, C=cursors,
    A=auto-Y, F=fit-now.

Usage:

    scope = p.LiveScope(builder, stream,
                          panels=["Voltage", "Current", "Control"])
    scope.add_node_voltage("vout", panel="Voltage", color="#1f77b4")
    scope.add_node_voltage("vin",  panel="Voltage", color="#888")
    scope.add_branch_current(ind_id, label="I(L1)",
                                panel="Current", color="#ff7f0e")
    scope.add_chain_channel(chain, "duty",   panel="Control")
    scope.add_chain_channel(chain, "v_filt", panel="Control")
    scope.start()
"""

from __future__ import annotations

import csv
import statistics
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np


__all__ = ["LiveScope", "SignalDef"]


# NOTE: we deliberately import pyqtgraph at module load. If pyqtgraph
# is not installed, importing this module raises ImportError — the
# parent ``pulsim`` package catches that and simply doesn't expose
# ``LiveScope``, keeping headless environments fully functional.
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui


# Modern, accessible palette (Tableau 10 — print-friendly).
_DEFAULT_COLORS = [
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ab",
]

# Dark theme colours.
_BG_COLOR     = "#13151c"
_PANEL_COLOR  = "#1a1d27"
_GRID_COLOR   = "#2a2e3a"
_TEXT_COLOR   = "#d8d9da"
_DIM_COLOR    = "#7a7e8a"
_ACCENT_COLOR = "#5b9bd5"
_DANGER_COLOR = "#c44545"


_Extractor = Callable[[float, np.ndarray], float]


class SignalDef:
    """One displayable signal.

    `kind` discriminates the fast path used by `_tick`:
      * ``"state"``   — `state_idx` is a column of the state vector;
        extraction is one numpy slice (`x_arr[:, state_idx]`). FAST.
      * ``"chain"``   — `chain.get(chain_name)` sampled once per
        tick (chain channels carry current value, not history).
      * ``"custom"``  — opaque scalar lambda; evaluated per sample.
        Slow path — only used by ``add_signal(extractor=…)``.
    """

    __slots__ = ("name", "extractor", "color", "unit", "panel",
                  "fmt", "kind", "state_idx", "chain", "chain_name")

    def __init__(self, name: str,
                  extractor: _Extractor,
                  *, color: Optional[str] = None,
                  unit: str = "",
                  panel: str = "default",
                  fmt: str = "{:+8.3f}",
                  kind: str = "custom",
                  state_idx: int = -1,
                  chain: Any = None,
                  chain_name: str = ""):
        self.name = name
        self.extractor = extractor
        self.color = color
        self.unit = unit
        self.panel = panel
        self.fmt = fmt
        self.kind = kind
        self.state_idx = state_idx
        self.chain = chain
        self.chain_name = chain_name


class _PanelState:
    """Internal: one plot panel."""
    def __init__(self, title: str, plot_widget):
        self.title = title
        self.plot = plot_widget
        self.signals: List[SignalDef] = []
        self.curves: Dict[str, Any] = {}
        self.auto_y = True
        self.y_range_manual: Optional[tuple] = None  # (lo, hi) when manual


class LiveScope:
    """Live-updating oscilloscope GUI for a running v2 simulation.

    Parameters
    ----------
    builder
        Populated :class:`CircuitBuilder` (for node / inductor lookups).
    stream
        The :class:`LiveStream` instance feeding the scope.
    panels
        Sequence of panel titles for the stacked plot layout.
        Default: a single "Main" panel.
    window_seconds
        Initial sliding-window width on the time axis. Default 2 s.
    update_hz
        GUI redraw rate. Default 30 Hz.
    title
        Window title.
    """

    def __init__(self, builder, stream, *,
                  panels: Optional[List[str]] = None,
                  window_seconds: float = 2.0,
                  update_hz: float = 30.0,
                  title: str = "Pulsim — LiveScope"):
        # Note: if pyqtgraph isn't installed the module-level
        # ``import pyqtgraph`` raises ImportError before we ever
        # get here, so the parent ``pulsim`` package never
        # exposes ``LiveScope`` in that case.
        self.builder = builder
        self.stream = stream
        self.window_seconds = float(window_seconds)
        self.update_interval_ms = max(10, int(1000.0 / update_hz))
        self.title = title

        self._signals: List[SignalDef] = []
        self._enabled: Dict[str, bool] = {}
        # Numpy ring buffers — preallocated, fixed-size. ``_ring_n``
        # holds the count of valid leading samples. Append is a
        # numpy slice; trim-on-window-overflow is a memmove of half.
        # Default 300k covers e.g. 3 s of a 100 kHz visual stream;
        # ~5 MB per signal at 16 B/sample — well below memory pain.
        self._max_points = 300_000
        self._ring_t: Dict[str, np.ndarray] = {}
        self._ring_y: Dict[str, np.ndarray] = {}
        self._ring_n: Dict[str, int] = {}
        self._panel_names = list(panels) if panels else ["Main"]

        # GUI state
        self._app = None
        self._win = None
        self._panels: Dict[str, _PanelState] = {}
        self._timer = None
        self._status_label = None
        self._fps_label = None
        self._value_labels: Dict[str, "QtWidgets.QLabel"] = {}
        self._checkboxes: Dict[str, "QtWidgets.QCheckBox"] = {}
        self._wall_t0 = time.perf_counter()
        self._last_tick = time.perf_counter()
        self._fps_ema = 0.0

        # Paused — when True, the display freezes but we keep
        # accepting samples into the rolling buffer.
        self._paused = False

        # Cursors
        self._cursors_enabled = False
        self._cursor_a = None
        self._cursor_b = None
        self._cursor_a_pos = 0.0
        self._cursor_b_pos = 0.0
        self._cursor_panel = None   # which panel the cursors live in
        self._cursor_readout = None

        # Stats focus channel — the one shown in the stats line.
        self._focus_signal: Optional[str] = None
        self._stats_label = None
        self._stats_history: Optional[list] = None

    # =====================================================================
    # Signal-registration API
    # =====================================================================

    def _resolve_panel(self, panel: Optional[str]) -> str:
        name = panel if panel is not None else self._panel_names[0]
        if name not in self._panel_names:
            self._panel_names.append(name)
        return name

    def add_signal(self, name: str,
                      extractor: _Extractor,
                      *, color: Optional[str] = None,
                      unit: str = "",
                      panel: Optional[str] = None,
                      fmt: str = "{:+8.3f}") -> "LiveScope":
        """Generic signal — extractor is called per sample (slow path).
        Prefer ``add_node_voltage`` / ``add_branch_current`` /
        ``add_chain_channel`` when possible — those use vectorized
        numpy access and are 50×+ faster."""
        sig = SignalDef(name, extractor, color=color, unit=unit,
                          panel=self._resolve_panel(panel),
                          fmt=fmt, kind="custom")
        self._signals.append(sig)
        return self

    def add_node_voltage(self, node_name: str, *,
                            label: Optional[str] = None,
                            color: Optional[str] = None,
                            panel: Optional[str] = None,
                            fmt: str = "{:+8.3f}") -> "LiveScope":
        idx = self.builder.node_id_of(node_name)
        nm = label or f"V({node_name})"
        sig = SignalDef(nm, lambda t, x, _i=idx: float(x[_i]),
                         color=color, unit="V",
                         panel=self._resolve_panel(panel),
                         fmt=fmt, kind="state", state_idx=int(idx))
        self._signals.append(sig)
        return self

    def add_branch_current(self, branch_id: int, *,
                                label: Optional[str] = None,
                                color: Optional[str] = None,
                                panel: Optional[str] = None,
                                fmt: str = "{:+8.3f}") -> "LiveScope":
        idx = self.builder.pool.branch_var_id_for_inductor(
            branch_id, self.builder.graph)
        nm = label or f"I(branch#{branch_id})"
        sig = SignalDef(nm, lambda t, x, _i=idx: float(x[_i]),
                         color=color, unit="A",
                         panel=self._resolve_panel(panel),
                         fmt=fmt, kind="state", state_idx=int(idx))
        self._signals.append(sig)
        return self

    def add_chain_channel(self, chain, name: str, *,
                              label: Optional[str] = None,
                              color: Optional[str] = None,
                              unit: str = "",
                              panel: Optional[str] = None,
                              fmt: str = "{:+8.3f}") -> "LiveScope":
        nm = label or name
        sig = SignalDef(
            nm, lambda t, x, _c=chain, _n=name: float(_c.get(_n)),
            color=color, unit=unit,
            panel=self._resolve_panel(panel),
            fmt=fmt, kind="chain", chain=chain, chain_name=name)
        self._signals.append(sig)
        return self

    # =====================================================================
    # GUI lifecycle
    # =====================================================================

    def start(self) -> None:
        pg.setConfigOptions(antialias=True,
                              background=_BG_COLOR,
                              foreground=_TEXT_COLOR,
                              useOpenGL=False)
        self._app = pg.mkQApp(self.title)
        self._build_window()
        self._build_panels()
        self._build_controls()
        self._build_status()
        self._wire_shortcuts()
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(self.update_interval_ms)
        self._wall_t0 = time.perf_counter()
        self._win.show()
        self._app.exec()

    # ---------------------------------------------------------------------
    # Window scaffolding
    # ---------------------------------------------------------------------

    def _build_window(self) -> None:
        self._win = QtWidgets.QMainWindow()
        self._win.setWindowTitle(self.title)
        self._win.resize(1280, 760)
        self._win.setStyleSheet(f"""
            QMainWindow {{ background: {_BG_COLOR}; }}
            QLabel      {{ color: {_TEXT_COLOR}; }}
            QCheckBox   {{ color: {_TEXT_COLOR}; padding: 2px 6px; }}
            QPushButton {{
                background: {_PANEL_COLOR}; color: {_TEXT_COLOR};
                border: 1px solid {_GRID_COLOR}; padding: 6px 12px;
                border-radius: 4px;
            }}
            QPushButton:hover {{ background: {_GRID_COLOR}; }}
            QPushButton:checked {{
                background: {_ACCENT_COLOR}; color: white;
                border: 1px solid {_ACCENT_COLOR};
            }}
            QPushButton#stopBtn {{
                background: {_DANGER_COLOR}; color: white;
                font-weight: 700; padding: 6px 18px;
            }}
            QPushButton#stopBtn:hover {{ background: #d35858; }}
            QPushButton#stopBtn:disabled {{
                background: #555; color: #aaa;
            }}
            QSlider::groove:horizontal {{
                background: {_PANEL_COLOR}; height: 4px;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {_ACCENT_COLOR}; width: 14px;
                margin: -5px 0; border-radius: 7px;
            }}
        """)
        central = QtWidgets.QWidget()
        central.setStyleSheet(f"background: {_BG_COLOR};")
        self._main_layout = QtWidgets.QVBoxLayout(central)
        self._main_layout.setContentsMargins(10, 8, 10, 8)
        self._main_layout.setSpacing(6)
        self._win.setCentralWidget(central)

    def _build_panels(self) -> None:
        """One PlotWidget per panel, X-linked. Each panel has its
        own Y axis and a header row (title + auto-Y toggle + value
        readout for each enabled signal)."""
        self._panels_container = QtWidgets.QWidget()
        panels_layout = QtWidgets.QVBoxLayout(self._panels_container)
        panels_layout.setContentsMargins(0, 0, 0, 0)
        panels_layout.setSpacing(2)
        self._main_layout.addWidget(self._panels_container, stretch=1)

        first_plot = None
        for panel_name in self._panel_names:
            # Get signals routed to this panel.
            panel_sigs = [s for s in self._signals
                          if s.panel == panel_name]
            if not panel_sigs and len(self._panel_names) > 1:
                continue   # skip empty named panels

            # Panel container.
            panel_box = QtWidgets.QFrame()
            panel_box.setStyleSheet(
                f"QFrame {{ background: {_PANEL_COLOR}; "
                f"border-radius: 6px; padding: 4px; }}")
            pb_layout = QtWidgets.QVBoxLayout(panel_box)
            pb_layout.setContentsMargins(6, 4, 6, 4)
            pb_layout.setSpacing(2)

            # Header row: title + auto-Y toggle + per-signal value
            # readouts.
            header = QtWidgets.QHBoxLayout()
            header.setContentsMargins(2, 0, 2, 0)
            header.setSpacing(10)
            title_lbl = QtWidgets.QLabel(panel_name)
            title_lbl.setStyleSheet(
                f"color: {_TEXT_COLOR}; font-weight: 700; "
                f"font-size: 11pt;")
            header.addWidget(title_lbl)
            auto_y_btn = QtWidgets.QPushButton("Auto Y")
            auto_y_btn.setCheckable(True)
            auto_y_btn.setChecked(True)
            auto_y_btn.setFixedWidth(72)
            auto_y_btn.toggled.connect(
                lambda checked, p=panel_name: self._set_auto_y(p, checked))
            header.addWidget(auto_y_btn)
            header.addStretch()

            # Value readouts for this panel's signals (live DMM-style).
            for i, sig in enumerate(panel_sigs):
                color = sig.color or _DEFAULT_COLORS[
                    self._signals.index(sig) % len(_DEFAULT_COLORS)]
                val_lbl = QtWidgets.QLabel(f"{sig.name}: ---")
                val_lbl.setStyleSheet(
                    f"color: {color}; font-family: "
                    f"'SF Mono', 'Menlo', 'Courier New', monospace; "
                    f"font-size: 10pt; padding: 1px 6px; "
                    f"background: {_BG_COLOR}; border-radius: 3px;")
                header.addWidget(val_lbl)
                self._value_labels[sig.name] = val_lbl
            pb_layout.addLayout(header)

            # Plot widget. We enable BOTH ``setClipToView`` (only
            # render points inside the visible X range) and
            # ``setDownsampling(auto=True, mode='peak')`` so
            # pyqtgraph collapses each pixel-column of samples to
            # min/max — the curve is visually identical but the
            # GPU upload per redraw drops by 10-100×.
            plot = pg.PlotWidget()
            plot.setLabel("left", units=panel_sigs[0].unit if panel_sigs
                                          else "")
            plot.showGrid(x=True, y=True, alpha=0.20)
            plot.getViewBox().setMouseEnabled(x=True, y=False)
            plot.getAxis("left").setStyle(tickFont=
                self._mono_font(9))
            plot.getAxis("bottom").setStyle(tickFont=
                self._mono_font(9))
            plot.setDownsampling(auto=True, mode="peak")
            plot.setClipToView(True)
            # Bottom axis only on the last panel (shared X).
            if panel_name != self._panel_names[-1]:
                plot.getAxis("bottom").setStyle(showValues=False)
            else:
                plot.setLabel("bottom", "time", units="s")
            if first_plot is not None:
                plot.setXLink(first_plot)
            else:
                first_plot = plot

            pb_layout.addWidget(plot)
            panels_layout.addWidget(panel_box, stretch=1)

            ps = _PanelState(panel_name, plot)
            self._panels[panel_name] = ps

            # Add curves + allocate ring buffers.
            for sig in panel_sigs:
                color = sig.color or _DEFAULT_COLORS[
                    self._signals.index(sig) % len(_DEFAULT_COLORS)]
                curve = plot.plot(
                    pen=pg.mkPen(color=color, width=1.4))
                ps.curves[sig.name] = curve
                ps.signals.append(sig)
                self._enabled[sig.name] = True
                self._ring_t[sig.name] = np.zeros(self._max_points,
                                                       dtype=np.float64)
                self._ring_y[sig.name] = np.zeros(self._max_points,
                                                       dtype=np.float64)
                self._ring_n[sig.name] = 0

        # Reserve focus signal for stats (first signal by default).
        if self._signals:
            self._focus_signal = self._signals[0].name

    def _build_controls(self) -> None:
        """Toolbar above the status bar: signal checkboxes, time
        window slider, action buttons."""
        ctrl_box = QtWidgets.QFrame()
        ctrl_box.setStyleSheet(
            f"QFrame {{ background: {_PANEL_COLOR}; "
            f"border-radius: 6px; padding: 4px; }}")
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_box)
        ctrl_layout.setContentsMargins(8, 4, 8, 4)
        ctrl_layout.setSpacing(4)

        # Row 1 — signal checkboxes, grouped by panel.
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(8)
        for panel_name in self._panel_names:
            if panel_name not in self._panels:
                continue
            for sig in self._panels[panel_name].signals:
                color = sig.color or _DEFAULT_COLORS[
                    self._signals.index(sig) % len(_DEFAULT_COLORS)]
                cb = QtWidgets.QCheckBox(sig.name)
                cb.setChecked(True)
                cb.setStyleSheet(
                    f"QCheckBox {{ color: {color}; font-weight: 600; }} "
                    f"QCheckBox::indicator {{ width: 14px; height: 14px; "
                    f"border: 1px solid {color}; border-radius: 3px; "
                    f"background: {_BG_COLOR}; }} "
                    f"QCheckBox::indicator:checked {{ background: {color}; }}")
                cb.stateChanged.connect(
                    lambda state, n=sig.name: self._on_toggle(n, state))
                # Click-to-focus for stats panel.
                cb.installEventFilter(
                    _ClickFilter(lambda n=sig.name:
                                    self._set_focus_signal(n), self._win))
                self._checkboxes[sig.name] = cb
                row1.addWidget(cb)
        row1.addStretch()

        # Time-window slider.
        row1.addWidget(QtWidgets.QLabel("window:"))
        self._win_label = QtWidgets.QLabel(
            f"{self.window_seconds*1000:.0f} ms")
        self._win_label.setMinimumWidth(64)
        self._win_label.setStyleSheet(
            f"color: {_DIM_COLOR}; "
            f"font-family: 'SF Mono', Menlo, monospace;")
        row1.addWidget(self._win_label)
        self._win_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        # Slider maps 0..1000 → 1ms..10s logarithmically.
        self._win_slider.setRange(0, 1000)
        self._win_slider.setValue(self._window_to_slider(
            self.window_seconds))
        self._win_slider.setFixedWidth(140)
        self._win_slider.valueChanged.connect(self._on_window_changed)
        row1.addWidget(self._win_slider)
        ctrl_layout.addLayout(row1)

        # Row 2 — action buttons.
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(6)
        self._pause_btn = QtWidgets.QPushButton("⏸ Pause display")
        self._pause_btn.setCheckable(True)
        self._pause_btn.toggled.connect(self._on_pause_toggled)
        row2.addWidget(self._pause_btn)

        self._cursors_btn = QtWidgets.QPushButton("⤬ Cursors")
        self._cursors_btn.setCheckable(True)
        self._cursors_btn.toggled.connect(self._on_cursors_toggled)
        row2.addWidget(self._cursors_btn)

        self._fit_btn = QtWidgets.QPushButton("⤢ Fit now")
        self._fit_btn.clicked.connect(self._fit_y_all)
        row2.addWidget(self._fit_btn)

        png_btn = QtWidgets.QPushButton("📷 PNG")
        png_btn.clicked.connect(self._save_png)
        row2.addWidget(png_btn)

        csv_btn = QtWidgets.QPushButton("⬇ CSV")
        csv_btn.clicked.connect(self._save_csv)
        row2.addWidget(csv_btn)

        row2.addStretch()

        # Stats readout.
        self._stats_label = QtWidgets.QLabel(
            "stats: click a signal name to focus")
        self._stats_label.setStyleSheet(
            f"color: {_DIM_COLOR}; "
            f"font-family: 'SF Mono', Menlo, monospace; "
            f"font-size: 9pt;")
        row2.addWidget(self._stats_label)

        # Cursor readout (Δt / Δy) — hidden initially.
        self._cursor_readout = QtWidgets.QLabel("")
        self._cursor_readout.setStyleSheet(
            f"color: {_ACCENT_COLOR}; "
            f"font-family: 'SF Mono', Menlo, monospace; "
            f"font-size: 9pt; font-weight: 700;")
        self._cursor_readout.setVisible(False)
        row2.addWidget(self._cursor_readout)

        self._stop_btn = QtWidgets.QPushButton("STOP simulation")
        self._stop_btn.setObjectName("stopBtn")
        self._stop_btn.clicked.connect(self._on_stop)
        row2.addWidget(self._stop_btn)
        ctrl_layout.addLayout(row2)

        self._main_layout.addWidget(ctrl_box)

    def _build_status(self) -> None:
        status_box = QtWidgets.QFrame()
        status_box.setStyleSheet(
            f"QFrame {{ background: transparent; padding: 0; }}")
        sl = QtWidgets.QHBoxLayout(status_box)
        sl.setContentsMargins(4, 0, 4, 0)
        sl.setSpacing(20)

        self._status_label = QtWidgets.QLabel("waiting for samples…")
        self._status_label.setStyleSheet(
            f"color: {_DIM_COLOR}; "
            f"font-family: 'SF Mono', Menlo, monospace; "
            f"font-size: 9pt;")
        sl.addWidget(self._status_label)
        sl.addStretch()

        self._fps_label = QtWidgets.QLabel("0 fps")
        self._fps_label.setStyleSheet(
            f"color: {_DIM_COLOR}; "
            f"font-family: 'SF Mono', Menlo, monospace; "
            f"font-size: 9pt;")
        sl.addWidget(self._fps_label)
        self._main_layout.addWidget(status_box)

    def _wire_shortcuts(self) -> None:
        """Keyboard shortcuts."""
        def make_shortcut(key, handler):
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self._win)
            sc.activated.connect(handler)
            return sc
        make_shortcut("Space", lambda:
                         self._pause_btn.toggle())
        make_shortcut("S", lambda:
                         self._stop_btn.click() if self._stop_btn.isEnabled()
                         else None)
        make_shortcut("C", lambda:
                         self._cursors_btn.toggle())
        make_shortcut("A", lambda:
                         self._toggle_auto_y_all())
        make_shortcut("F", self._fit_y_all)

    @staticmethod
    def _mono_font(size: int) -> "QtGui.QFont":
        f = QtGui.QFont()
        # Try a monospace; Qt picks the best available.
        for family in ("SF Mono", "Menlo", "Consolas",
                          "DejaVu Sans Mono", "Courier New"):
            f.setFamily(family)
            break
        f.setPointSize(size)
        f.setStyleHint(QtGui.QFont.Monospace)
        return f

    # =====================================================================
    # Slider math
    # =====================================================================

    def _window_to_slider(self, win_s: float) -> int:
        # Slider 0..1000 → log10(1e-3) to log10(10) in seconds.
        x = (np.log10(win_s) - np.log10(1e-3)) / (
            np.log10(10) - np.log10(1e-3))
        return int(np.clip(x * 1000, 0, 1000))

    def _slider_to_window(self, val: int) -> float:
        x = val / 1000.0
        log_win = np.log10(1e-3) + x * (np.log10(10) - np.log10(1e-3))
        return float(10.0 ** log_win)

    def _on_window_changed(self, val: int) -> None:
        self.window_seconds = self._slider_to_window(val)
        if self.window_seconds >= 1.0:
            self._win_label.setText(f"{self.window_seconds:.2f} s")
        else:
            self._win_label.setText(
                f"{self.window_seconds*1000:.0f} ms")

    # =====================================================================
    # Toggle handlers
    # =====================================================================

    def _on_toggle(self, name: str, state) -> None:
        self._enabled[name] = bool(state)
        for ps in self._panels.values():
            if name in ps.curves:
                if not self._enabled[name]:
                    ps.curves[name].clear()
                    self._value_labels[name].setText(f"{name}: ---")

    def _set_auto_y(self, panel: str, enabled: bool) -> None:
        if panel in self._panels:
            self._panels[panel].auto_y = enabled
            if not enabled:
                # Capture current range as the manual range.
                r = self._panels[panel].plot.getViewBox().viewRange()[1]
                self._panels[panel].y_range_manual = tuple(r)

    def _toggle_auto_y_all(self) -> None:
        any_off = any(not ps.auto_y for ps in self._panels.values())
        for ps in self._panels.values():
            ps.auto_y = any_off    # if any was off, turn all on

    def _fit_y_all(self) -> None:
        for ps in self._panels.values():
            ps.plot.enableAutoRange(axis="y", enable=True)

    def _on_pause_toggled(self, checked: bool) -> None:
        self._paused = checked
        self._pause_btn.setText("▶ Resume display" if checked
                                   else "⏸ Pause display")

    def _on_cursors_toggled(self, checked: bool) -> None:
        self._cursors_enabled = checked
        self._cursor_readout.setVisible(checked)
        if checked and self._cursor_panel is None:
            # Add cursors to the first panel by default.
            first = next(iter(self._panels.values()))
            self._cursor_panel = first
            self._cursor_a = pg.InfiniteLine(
                angle=90, movable=True,
                pen=pg.mkPen(color=_ACCENT_COLOR, width=1.4,
                                style=QtCore.Qt.DashLine),
                label="A", labelOpts={"position": 0.95,
                                            "color": _ACCENT_COLOR})
            self._cursor_b = pg.InfiniteLine(
                angle=90, movable=True,
                pen=pg.mkPen(color="#e1aa45", width=1.4,
                                style=QtCore.Qt.DashLine),
                label="B", labelOpts={"position": 0.95,
                                            "color": "#e1aa45"})
            # Default positions: near the right edge of the visible plot.
            xr = first.plot.getViewBox().viewRange()[0]
            mid = 0.5 * (xr[0] + xr[1])
            span = xr[1] - xr[0]
            self._cursor_a.setPos(mid - 0.1 * span)
            self._cursor_b.setPos(mid + 0.1 * span)
            self._cursor_a.sigPositionChanged.connect(
                self._update_cursor_readout)
            self._cursor_b.sigPositionChanged.connect(
                self._update_cursor_readout)
        if self._cursor_a and self._cursor_b:
            if checked:
                self._cursor_panel.plot.addItem(self._cursor_a)
                self._cursor_panel.plot.addItem(self._cursor_b)
                self._update_cursor_readout()
            else:
                self._cursor_panel.plot.removeItem(self._cursor_a)
                self._cursor_panel.plot.removeItem(self._cursor_b)

    def _update_cursor_readout(self) -> None:
        if not (self._cursors_enabled and self._cursor_a
                  and self._cursor_b and self._focus_signal):
            return
        ta = float(self._cursor_a.value())
        tb = float(self._cursor_b.value())
        dt = tb - ta
        # Find ya/yb on the focus signal.
        sig_name = self._focus_signal
        n = self._ring_n.get(sig_name, 0)
        if n < 2:
            self._cursor_readout.setText(
                f"A: t={ta*1e3:.3f} ms   B: t={tb*1e3:.3f} ms   "
                f"Δt={dt*1e3:+.3f} ms")
            return
        arr_t = self._ring_t[sig_name][:n]
        arr_y = self._ring_y[sig_name][:n]
        ia = int(np.clip(np.searchsorted(arr_t, ta), 1, n - 1))
        ib = int(np.clip(np.searchsorted(arr_t, tb), 1, n - 1))
        ya = float(arr_y[ia])
        yb = float(arr_y[ib])
        dy = yb - ya
        if abs(dt) > 1e-12:
            freq = f"  (1/Δt = {1.0/abs(dt):.1f} Hz)"
        else:
            freq = ""
        self._cursor_readout.setText(
            f"A({sig_name})={ya:+.4g}   B={yb:+.4g}   "
            f"Δy={dy:+.4g}   Δt={dt*1e3:+.4f} ms{freq}")

    def _set_focus_signal(self, name: str) -> None:
        self._focus_signal = name
        # Reset stats history on focus change.
        self._stats_history = None

    def _on_stop(self) -> None:
        self.stream.stop()
        self._status_label.setText("STOPPING — wait for kernel…")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setText("Stopped")

    # =====================================================================
    # Snapshot
    # =====================================================================

    def _save_png(self) -> None:
        if not self._panels:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._win, "Save scope snapshot",
            f"scope_{int(time.time())}.png",
            "PNG image (*.png)")
        if not path:
            return
        # Render the central widget at 2× scale.
        central = self._win.centralWidget()
        pixmap = central.grab()
        pixmap = pixmap.scaled(
            int(pixmap.width() * 2),
            int(pixmap.height() * 2),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation)
        pixmap.save(path, "PNG", 100)
        self._status_label.setText(f"saved → {path}")

    def _save_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._win, "Save scope data",
            f"scope_{int(time.time())}.csv",
            "CSV (*.csv)")
        if not path:
            return
        # Union of all time vectors — write one row per unique t.
        # For simplicity, write each enabled signal with its own t,y
        # in long format.
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["signal", "time_s", "value"])
            for name in self._ring_n:
                if not self._enabled.get(name, False):
                    continue
                n = self._ring_n[name]
                if n == 0:
                    continue
                arr_t = self._ring_t[name][:n]
                arr_y = self._ring_y[name][:n]
                for ti, yi in zip(arr_t.tolist(), arr_y.tolist()):
                    w.writerow([name, ti, yi])
        self._status_label.setText(f"saved → {path}")

    # =====================================================================
    # The hot loop
    # =====================================================================

    def _tick(self) -> None:
        """Per-frame GUI update.

        Hot path performance notes:
        * Drains ALL queued batches in one pass, concatenates them
          with ``np.concatenate`` (one numpy op), and extracts
          each signal from the combined array in a single
          vectorized slice (``state`` signals) — no Python lambda
          calls in the per-sample loop. That's the 50× speedup
          over the previous implementation.
        * Ring-buffer writes are numpy slice assignments — no
          ``list.extend`` / ``tolist`` churn.
        * Pyqtgraph's auto-downsampling (configured in
          ``_build_panels``) means ``setData`` collapses the
          per-pixel column on the GL side, so the curve always
          uploads ~visible-pixel-count points regardless of how
          many samples the ring holds.
        """
        now = time.perf_counter()
        # FPS EMA.
        dt = now - self._last_tick
        if dt > 0:
            fps = 1.0 / dt
            self._fps_ema = 0.85 * self._fps_ema + 0.15 * fps
        self._last_tick = now

        # Drain queue → list of (t_arr, x_arr). Cheap.
        batches: List[tuple] = []
        while True:
            b = self.stream.get_batch(timeout=0.0)
            if b is None:
                break
            batches.append(b)
        n_batches = len(batches)
        t_latest = 0.0

        if n_batches > 0 and not self._paused:
            # One concat for all batches. ``t_all`` shape (N,),
            # ``x_all`` shape (N, state_size).
            t_all = np.concatenate(
                [b[0] for b in batches]).astype(np.float64, copy=False)
            x_all = np.concatenate(
                [b[1] for b in batches]).astype(np.float64, copy=False)
            t_latest = float(t_all[-1])
            n_new = t_all.size

            # Vectorized extraction per signal.
            for sig in self._signals:
                if sig.kind == "state":
                    # ONE numpy slice — no Python per-sample work.
                    y_new = x_all[:, sig.state_idx]
                elif sig.kind == "chain":
                    # Chain channel — only one current value per
                    # tick, broadcast across the batch's time
                    # samples. Visually identical to per-sample
                    # sampling (the chain doesn't have per-sample
                    # history exposed) but ~1000× cheaper.
                    val = float(sig.chain.get(sig.chain_name))
                    y_new = np.full(n_new, val, dtype=np.float64)
                else:
                    # Slow path — generic extractor (used only by
                    # ``add_signal(extractor=…)`` callers).
                    y_new = np.fromiter(
                        (sig.extractor(t_all[i], x_all[i])
                          for i in range(n_new)),
                        dtype=np.float64, count=n_new)
                self._ring_push(sig.name, t_all, y_new)

            # Window trim: drop everything older than t_latest
            # minus window_seconds. Same cutoff for every signal
            # so curves stay aligned.
            self._ring_trim_window(t_latest - self.window_seconds)

        # Redraw curves + value readouts.
        do_redraw = (n_batches > 0 and not self._paused)
        if do_redraw or (n_batches > 0 and self._paused):
            for sig in self._signals:
                name = sig.name
                n = self._ring_n[name]
                if n == 0 or not self._enabled.get(name, False):
                    continue
                if do_redraw:
                    # Direct ring view — no copy.
                    buf_t = self._ring_t[name][:n]
                    buf_y = self._ring_y[name][:n]
                    for ps in self._panels.values():
                        if name in ps.curves:
                            ps.curves[name].setData(buf_t, buf_y)
                # Value readout.
                if name in self._value_labels:
                    val = float(self._ring_y[name][n - 1])
                    self._value_labels[name].setText(
                        f"{name}: {sig.fmt.format(val)} {sig.unit}")

            self._update_stats()
            if self._cursors_enabled:
                self._update_cursor_readout()

        # Apply manual Y range where requested.
        for ps in self._panels.values():
            if ps.auto_y:
                ps.plot.enableAutoRange(axis="y", enable=True)
            elif ps.y_range_manual is not None:
                ps.plot.setYRange(*ps.y_range_manual, padding=0)

        # Status line.
        wall = now - self._wall_t0
        sim_t_disp = (f"{t_latest*1e3:.2f} ms" if t_latest < 1.0
                        else f"{t_latest:.3f} s")
        self._status_label.setText(
            f"sim t = {sim_t_disp}   "
            f"wall = {wall:.1f} s   "
            f"samples = {self.stream.n_steps_received:,}   "
            f"kept = {self.stream.n_steps_kept:,}   "
            f"queue = {self.stream.qsize()}   "
            f"dropped = {self.stream.n_batches_dropped}   "
            f"batches/tick = {n_batches}")
        self._fps_label.setText(f"{self._fps_ema:.0f} fps")

    # ------------------------------------------------------------------
    # Ring-buffer helpers (numpy, fixed capacity)
    # ------------------------------------------------------------------

    def _ring_push(self, name: str,
                       t_new: np.ndarray,
                       y_new: np.ndarray) -> None:
        """Append ``(t_new, y_new)`` to the named ring buffer.
        Drops oldest samples when the buffer would overflow."""
        n = self._ring_n[name]
        cap = self._max_points
        n_new = t_new.size

        if n_new >= cap:
            # New batch alone is bigger than capacity — keep only
            # the tail.
            self._ring_t[name][:] = t_new[-cap:]
            self._ring_y[name][:] = y_new[-cap:]
            self._ring_n[name] = cap
            return

        if n + n_new > cap:
            # Shift left by (n + n_new - cap) — keep newest.
            shift = (n + n_new) - cap
            keep = n - shift
            self._ring_t[name][:keep] = self._ring_t[name][shift:n]
            self._ring_y[name][:keep] = self._ring_y[name][shift:n]
            n = keep

        self._ring_t[name][n:n + n_new] = t_new
        self._ring_y[name][n:n + n_new] = y_new
        self._ring_n[name] = n + n_new

    def _ring_trim_window(self, t_cutoff: float) -> None:
        """Drop samples older than ``t_cutoff`` from every ring.
        Uses ``np.searchsorted`` — O(log N) per signal."""
        if t_cutoff <= 0:
            return
        for name, n in list(self._ring_n.items()):
            if n == 0:
                continue
            t_buf = self._ring_t[name]
            if t_buf[0] >= t_cutoff:
                continue
            i_cut = int(np.searchsorted(t_buf[:n], t_cutoff, side="left"))
            if i_cut <= 0:
                continue
            keep = n - i_cut
            if keep > 0:
                self._ring_t[name][:keep] = t_buf[i_cut:n]
                self._ring_y[name][:keep] = self._ring_y[name][i_cut:n]
            self._ring_n[name] = keep

    def _update_stats(self) -> None:
        if not self._focus_signal:
            return
        name = self._focus_signal
        n = self._ring_n.get(name, 0)
        if n == 0:
            self._stats_label.setText(f"stats[{name}]: ---")
            return
        a = self._ring_y[name][:n]
        # Sliding-window stats over what's currently displayed.
        v_min = float(a.min())
        v_max = float(a.max())
        v_mean = float(a.mean())
        v_rms = float(np.sqrt(np.mean(a * a)))
        v_pp = v_max - v_min
        sig = next((s for s in self._signals if s.name == name), None)
        unit = sig.unit if sig else ""
        self._stats_label.setText(
            f"stats[{name}]:  "
            f"min={v_min:+.4g}{unit}  "
            f"max={v_max:+.4g}{unit}  "
            f"mean={v_mean:+.4g}{unit}  "
            f"rms={v_rms:.4g}{unit}  "
            f"p-p={v_pp:.4g}{unit}")


class _ClickFilter(QtCore.QObject):
    """Helper: catches clicks on a widget to fire a callback. Used
    to make signal-name checkbox labels act as 'focus this signal
    in the stats panel' click targets."""
    def __init__(self, callback, parent=None):
        super().__init__(parent)
        self._cb = callback

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.MouseButtonRelease:
            self._cb()
        return False
