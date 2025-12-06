"""SpiceLab remote client for distributed simulation via gRPC."""

from .client import (
    SpiceLabClient,
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
    "SpiceLabClient",
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
    from .widgets import StreamingPlot, InteractivePlot, plot_waveforms
    __all__.extend(["StreamingPlot", "InteractivePlot", "plot_waveforms"])
except ImportError:
    pass
