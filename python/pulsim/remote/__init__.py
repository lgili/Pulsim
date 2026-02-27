"""Pulsim remote client for distributed simulation via gRPC."""

from .client import (
    PulsimClient,
    Session,
    SessionStatus,
    SimulationOptions,
    WaveformStream,
    AsyncWaveformStream,
    WaveformSample,
    WaveformHeader,
    HealthStatus,
    HealthCheckResult,
)

__all__ = [
    "PulsimClient",
    "Session",
    "SessionStatus",
    "SimulationOptions",
    "WaveformStream",
    "AsyncWaveformStream",
    "WaveformSample",
    "WaveformHeader",
    "HealthStatus",
    "HealthCheckResult",
]

# Optional widget imports (require ipywidgets)
try:
    from .widgets import StreamingPlot as StreamingPlot
    from .widgets import InteractivePlot as InteractivePlot
    from .widgets import plot_waveforms as plot_waveforms
    __all__.extend(["StreamingPlot", "InteractivePlot", "plot_waveforms"])
except ImportError:
    pass
