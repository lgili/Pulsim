"""Jupyter widgets for streaming waveform visualization."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .client import PulsimClient

# Check for optional dependencies
_HAS_IPYWIDGETS = False
_HAS_MATPLOTLIB = False
_HAS_PLOTLY = False

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    _HAS_IPYWIDGETS = True
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    pass

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    go = None  # type: ignore[assignment]


def _check_dependencies(plotly: bool = False) -> None:
    """Check if required dependencies are available."""
    if not _HAS_IPYWIDGETS:
        raise ImportError(
            "ipywidgets is required for Jupyter widgets. "
            "Install with: pip install ipywidgets"
        )
    if plotly and not _HAS_PLOTLY:
        raise ImportError(
            "plotly is required for interactive plots. "
            "Install with: pip install plotly"
        )
    if not plotly and not _HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plots. "
            "Install with: pip install matplotlib"
        )


class StreamingPlot:
    """Matplotlib-based streaming waveform plot for Jupyter notebooks.

    Displays live waveform data from a streaming simulation with
    auto-updating plots.

    Example:
        >>> from pulsim.grpc import PulsimClient
        >>> from pulsim.grpc.widgets import StreamingPlot
        >>>
        >>> client = PulsimClient("localhost:50051")
        >>> with client:
        ...     session = client.create_session(...)
        ...     client.start_simulation(session.session_id)
        ...     plot = StreamingPlot(client, session.session_id)
        ...     plot.start()
    """

    def __init__(
        self,
        client: "PulsimClient",
        session_id: str,
        signals: Optional[List[str]] = None,
        decimation: int = 1,
        max_points: int = 10000,
        update_interval: float = 0.1,
        figsize: Tuple[int, int] = (12, 6),
    ):
        """Initialize the streaming plot.

        Args:
            client: Connected PulsimClient instance.
            session_id: Session to stream from.
            signals: Signals to plot (all if None).
            decimation: Decimation factor for streaming.
            max_points: Maximum points to display (older points are dropped).
            update_interval: Update interval in seconds.
            figsize: Figure size (width, height) in inches.
        """
        _check_dependencies()

        self._client = client
        self._session_id = session_id
        self._signals = signals
        self._decimation = decimation
        self._max_points = max_points
        self._update_interval = update_interval
        self._figsize = figsize

        self._times: List[float] = []
        self._values: Dict[str, List[float]] = {}
        self._signal_names: List[str] = []
        self._running = False
        self._stream = None
        self._fig = None
        self._axes = None
        self._lines: Dict[str, Any] = {}
        self._output = None
        self._status_label = None
        self._progress_bar = None

    def _create_ui(self) -> None:
        """Create the Jupyter widget UI."""
        # Status label
        self._status_label = widgets.HTML(
            value="<b>Status:</b> Initializing..."
        )

        # Progress bar
        self._progress_bar = widgets.FloatProgress(
            value=0,
            min=0,
            max=100,
            description="Progress:",
            bar_style="info",
            style={"bar_color": "#2196F3"},
        )

        # Control buttons
        self._stop_button = widgets.Button(
            description="Stop",
            button_style="danger",
            icon="stop",
        )
        self._stop_button.on_click(lambda _: self.stop())

        self._pause_button = widgets.Button(
            description="Pause",
            button_style="warning",
            icon="pause",
        )
        self._pause_button.on_click(lambda _: self._toggle_pause())

        controls = widgets.HBox([self._stop_button, self._pause_button])

        # Output area for the plot
        self._output = widgets.Output()

        # Layout
        self._container = widgets.VBox([
            self._status_label,
            self._progress_bar,
            controls,
            self._output,
        ])

    def _init_plot(self) -> None:
        """Initialize the matplotlib figure."""
        with self._output:
            clear_output(wait=True)
            self._fig, self._axes = plt.subplots(figsize=self._figsize)
            self._fig.tight_layout()
            plt.ion()
            plt.show()

    def _update_plot(self) -> None:
        """Update the plot with new data."""
        if not self._times:
            return

        with self._output:
            times_arr = np.array(self._times[-self._max_points:])

            for sig_name in self._signal_names:
                values = self._values.get(sig_name, [])[-self._max_points:]
                values_arr = np.array(values)

                if sig_name not in self._lines:
                    line, = self._axes.plot(times_arr, values_arr, label=sig_name)
                    self._lines[sig_name] = line
                else:
                    self._lines[sig_name].set_data(times_arr, values_arr)

            self._axes.relim()
            self._axes.autoscale_view()
            self._axes.set_xlabel("Time (s)")
            self._axes.set_ylabel("Value")
            self._axes.legend(loc="upper right")
            self._axes.grid(True, alpha=0.3)

            self._fig.canvas.draw()
            self._fig.canvas.flush_events()

    def _stream_data(self) -> None:
        """Stream data from the server."""
        try:
            self._stream = self._client.stream_waveforms(
                self._session_id,
                self._signals,
                self._decimation,
            )

            for sample in self._stream:
                if not self._running:
                    break

                self._times.append(sample.time)
                for sig_name, value in sample.values.items():
                    if sig_name not in self._signal_names:
                        self._signal_names.append(sig_name)
                        self._values[sig_name] = []
                    self._values[sig_name].append(value)

            self._status_label.value = "<b>Status:</b> Completed"
            self._progress_bar.bar_style = "success"
            self._progress_bar.value = 100

        except Exception as e:
            self._status_label.value = f"<b>Status:</b> Error - {e}"
            self._progress_bar.bar_style = "danger"

        finally:
            self._running = False

    def _toggle_pause(self) -> None:
        """Toggle pause state."""
        if self._pause_button.description == "Pause":
            self._pause_button.description = "Resume"
            self._pause_button.icon = "play"
            self._client.pause_simulation(self._session_id)
        else:
            self._pause_button.description = "Pause"
            self._pause_button.icon = "pause"
            self._client.resume_simulation(self._session_id)

    def start(self) -> widgets.VBox:
        """Start streaming and display the widget.

        Returns:
            The Jupyter widget container.
        """
        self._create_ui()
        display(self._container)

        self._init_plot()
        self._running = True
        self._status_label.value = "<b>Status:</b> Streaming..."

        # Start streaming in background thread
        self._stream_thread = threading.Thread(target=self._stream_data)
        self._stream_thread.start()

        # Start update loop
        self._update_loop()

        return self._container

    def _update_loop(self) -> None:
        """Periodic update loop."""
        if self._running:
            self._update_plot()
            # Schedule next update
            loop = asyncio.get_event_loop()
            loop.call_later(self._update_interval, self._update_loop)

    def stop(self) -> None:
        """Stop streaming."""
        self._running = False
        self._client.stop_simulation(self._session_id)
        self._status_label.value = "<b>Status:</b> Stopped"
        self._progress_bar.bar_style = "warning"

    def get_data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Get the collected data.

        Returns:
            Tuple of (times, {signal_name: values}).
        """
        times = np.array(self._times)
        signals = {name: np.array(vals) for name, vals in self._values.items()}
        return times, signals


class InteractivePlot:
    """Plotly-based interactive streaming plot for Jupyter notebooks.

    Provides a more interactive experience with zoom, pan, and hover.

    Example:
        >>> from pulsim.grpc.widgets import InteractivePlot
        >>>
        >>> plot = InteractivePlot(client, session_id, signals=["V(out)"])
        >>> plot.start()
    """

    def __init__(
        self,
        client: "PulsimClient",
        session_id: str,
        signals: Optional[List[str]] = None,
        decimation: int = 1,
        max_points: int = 10000,
        update_interval: float = 0.2,
        height: int = 500,
    ):
        """Initialize the interactive plot.

        Args:
            client: Connected PulsimClient instance.
            session_id: Session to stream from.
            signals: Signals to plot.
            decimation: Decimation factor.
            max_points: Maximum points to display.
            update_interval: Update interval in seconds.
            height: Plot height in pixels.
        """
        _check_dependencies(plotly=True)

        self._client = client
        self._session_id = session_id
        self._signals = signals
        self._decimation = decimation
        self._max_points = max_points
        self._update_interval = update_interval
        self._height = height

        self._times: List[float] = []
        self._values: Dict[str, List[float]] = {}
        self._signal_names: List[str] = []
        self._running = False
        self._stream = None
        self._fig_widget = None
        self._output = None

    def _create_ui(self) -> None:
        """Create the UI components."""
        # Status
        self._status_label = widgets.HTML(value="<b>Status:</b> Initializing...")

        # Control buttons
        self._stop_button = widgets.Button(
            description="Stop",
            button_style="danger",
            icon="stop",
        )
        self._stop_button.on_click(lambda _: self.stop())

        controls = widgets.HBox([self._stop_button])

        # Figure widget
        self._fig = go.FigureWidget()
        self._fig.update_layout(
            height=self._height,
            xaxis_title="Time (s)",
            yaxis_title="Value",
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        )

        self._container = widgets.VBox([
            self._status_label,
            controls,
            self._fig,
        ])

    def _stream_data(self) -> None:
        """Stream data from the server."""
        try:
            self._stream = self._client.stream_waveforms(
                self._session_id,
                self._signals,
                self._decimation,
            )

            for sample in self._stream:
                if not self._running:
                    break

                self._times.append(sample.time)
                for sig_name, value in sample.values.items():
                    if sig_name not in self._signal_names:
                        self._signal_names.append(sig_name)
                        self._values[sig_name] = []
                        # Add trace
                        self._fig.add_trace(
                            go.Scatter(x=[], y=[], name=sig_name, mode="lines")
                        )
                    self._values[sig_name].append(value)

            self._status_label.value = "<b>Status:</b> Completed"

        except Exception as e:
            self._status_label.value = f"<b>Status:</b> Error - {e}"

        finally:
            self._running = False

    def _update_plot(self) -> None:
        """Update the plot with new data."""
        if not self._times or not self._fig.data:
            return

        times = self._times[-self._max_points:]

        with self._fig.batch_update():
            for i, sig_name in enumerate(self._signal_names):
                if i < len(self._fig.data):
                    values = self._values.get(sig_name, [])[-self._max_points:]
                    self._fig.data[i].x = times
                    self._fig.data[i].y = values

    def start(self) -> widgets.VBox:
        """Start streaming and display."""
        self._create_ui()
        display(self._container)

        self._running = True
        self._status_label.value = "<b>Status:</b> Streaming..."

        # Start streaming in background
        self._stream_thread = threading.Thread(target=self._stream_data)
        self._stream_thread.start()

        # Start update loop
        self._update_loop()

        return self._container

    def _update_loop(self) -> None:
        """Periodic update."""
        if self._running:
            self._update_plot()
            loop = asyncio.get_event_loop()
            loop.call_later(self._update_interval, self._update_loop)

    def stop(self) -> None:
        """Stop streaming."""
        self._running = False
        self._client.stop_simulation(self._session_id)
        self._status_label.value = "<b>Status:</b> Stopped"

    def get_data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Get collected data."""
        times = np.array(self._times)
        signals = {name: np.array(vals) for name, vals in self._values.items()}
        return times, signals


def plot_waveforms(
    client: "PulsimClient",
    session_id: str,
    signals: Optional[List[str]] = None,
    interactive: bool = True,
    **kwargs,
) -> StreamingPlot:
    """Convenience function to create a streaming plot.

    Args:
        client: PulsimClient instance.
        session_id: Session identifier.
        signals: Signals to plot.
        interactive: Use Plotly (True) or Matplotlib (False).
        **kwargs: Additional arguments for the plot class.

    Returns:
        The plot widget (started).
    """
    if interactive and _HAS_PLOTLY:
        plot = InteractivePlot(client, session_id, signals, **kwargs)
    else:
        plot = StreamingPlot(client, session_id, signals, **kwargs)

    return plot.start()
