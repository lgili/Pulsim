"""Pulsim gRPC client implementation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

import grpc
import numpy as np

from . import simulator_pb2 as pb
from . import simulator_pb2_grpc as pb_grpc

logger = logging.getLogger(__name__)


class HealthStatus(IntEnum):
    """Server health status."""

    UNSPECIFIED = 0
    OK = 1
    DEGRADED = 2
    ERROR = 3


class SessionStatus(IntEnum):
    """Simulation session status."""

    UNSPECIFIED = 0
    CREATED = 1
    VALIDATING = 2
    READY = 3
    RUNNING = 4
    PAUSED = 5
    COMPLETED = 6
    STOPPED = 7
    FAILED = 8


@dataclass
class SimulationOptions:
    """Simulation configuration options."""

    tstart: Optional[float] = None
    tstop: Optional[float] = None
    dt: Optional[float] = None
    dtmin: Optional[float] = None
    dtmax: Optional[float] = None
    abstol: Optional[float] = None
    reltol: Optional[float] = None
    max_newton_iterations: Optional[int] = None
    damping_factor: Optional[float] = None
    use_ic: Optional[bool] = None
    output_signals: List[str] = field(default_factory=list)

    def to_proto(self) -> pb.SimulationOptions:
        """Convert to protobuf message."""
        opts = pb.SimulationOptions()
        if self.tstart is not None:
            opts.tstart.value = self.tstart
        if self.tstop is not None:
            opts.tstop.value = self.tstop
        if self.dt is not None:
            opts.dt.value = self.dt
        if self.dtmin is not None:
            opts.dtmin.value = self.dtmin
        if self.dtmax is not None:
            opts.dtmax.value = self.dtmax
        if self.abstol is not None:
            opts.abstol.value = self.abstol
        if self.reltol is not None:
            opts.reltol.value = self.reltol
        if self.max_newton_iterations is not None:
            opts.max_newton_iterations.value = self.max_newton_iterations
        if self.damping_factor is not None:
            opts.damping_factor.value = self.damping_factor
        if self.use_ic is not None:
            opts.use_ic.value = self.use_ic
        opts.output_signals.extend(self.output_signals)
        return opts


@dataclass
class WaveformSample:
    """A single waveform sample point."""

    time: float
    values: Dict[str, float]


@dataclass
class WaveformHeader:
    """Waveform stream header with metadata."""

    session_id: str
    signals: List[str]
    tstart: float
    tstop: float
    total_samples: int


class WaveformStream:
    """Iterator for streaming waveform data.

    Supports both synchronous and asynchronous iteration, with optional
    conversion to pandas DataFrame or xarray Dataset.
    """

    def __init__(
        self,
        response_iterator: Iterator[pb.WaveformStreamResponse],
        signals: List[str],
    ):
        self._iterator = response_iterator
        self._signals = signals
        self._header: Optional[WaveformHeader] = None
        self._samples: List[WaveformSample] = []
        self._complete = False
        self._error: Optional[str] = None
        self._final_status: Optional[SessionStatus] = None

    @property
    def header(self) -> Optional[WaveformHeader]:
        """Get the stream header (available after first iteration)."""
        return self._header

    @property
    def signals(self) -> List[str]:
        """Get the list of signal names."""
        return self._signals

    @property
    def is_complete(self) -> bool:
        """Check if the stream is complete."""
        return self._complete

    @property
    def error(self) -> Optional[str]:
        """Get error message if stream failed."""
        return self._error

    @property
    def final_status(self) -> Optional[SessionStatus]:
        """Get the final session status."""
        return self._final_status

    def __iter__(self) -> Iterator[WaveformSample]:
        """Iterate over waveform samples."""
        for response in self._iterator:
            payload = response.WhichOneof("payload")
            if payload == "header":
                hdr = response.header
                self._header = WaveformHeader(
                    session_id=hdr.session_id,
                    signals=list(hdr.signals),
                    tstart=hdr.tstart,
                    tstop=hdr.tstop,
                    total_samples=hdr.total_samples,
                )
                self._signals = self._header.signals
            elif payload == "sample":
                sample = WaveformSample(
                    time=response.sample.time,
                    values={
                        sig: val
                        for sig, val in zip(self._signals, response.sample.values)
                    },
                )
                self._samples.append(sample)
                yield sample
            elif payload == "complete":
                self._complete = True
                self._final_status = SessionStatus(response.complete.final_status)
                if response.complete.error_message:
                    self._error = response.complete.error_message

    def collect(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Collect all samples into numpy arrays.

        Returns:
            Tuple of (time_array, {signal_name: values_array})
        """
        # Consume remaining samples if not already done
        if not self._complete:
            for _ in self:
                pass

        if not self._samples:
            return np.array([]), {}

        times = np.array([s.time for s in self._samples])
        signals = {
            sig: np.array([s.values[sig] for s in self._samples])
            for sig in self._signals
        }
        return times, signals

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert collected samples to pandas DataFrame.

        Returns:
            DataFrame with time index and signal columns.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for DataFrame conversion. "
                "Install with: pip install pandas"
            )

        times, signals = self.collect()
        df = pd.DataFrame(signals, index=times)
        df.index.name = "time"
        return df

    def to_xarray(self) -> "xr.Dataset":
        """Convert collected samples to xarray Dataset.

        Returns:
            xarray Dataset with time coordinate and signal data variables.
        """
        try:
            import xarray as xr
        except ImportError:
            raise ImportError(
                "xarray is required for Dataset conversion. "
                "Install with: pip install xarray"
            )

        times, signals = self.collect()
        data_vars = {
            name: (["time"], values) for name, values in signals.items()
        }
        return xr.Dataset(data_vars, coords={"time": times})


class AsyncWaveformStream:
    """Async iterator for streaming waveform data."""

    def __init__(
        self,
        response_iterator: AsyncIterator[pb.WaveformStreamResponse],
        signals: List[str],
    ):
        self._iterator = response_iterator
        self._signals = signals
        self._header: Optional[WaveformHeader] = None
        self._samples: List[WaveformSample] = []
        self._complete = False
        self._error: Optional[str] = None
        self._final_status: Optional[SessionStatus] = None

    @property
    def header(self) -> Optional[WaveformHeader]:
        """Get the stream header."""
        return self._header

    @property
    def signals(self) -> List[str]:
        """Get the list of signal names."""
        return self._signals

    @property
    def is_complete(self) -> bool:
        """Check if the stream is complete."""
        return self._complete

    def __aiter__(self) -> AsyncIterator[WaveformSample]:
        return self

    async def __anext__(self) -> WaveformSample:
        async for response in self._iterator:
            payload = response.WhichOneof("payload")
            if payload == "header":
                hdr = response.header
                self._header = WaveformHeader(
                    session_id=hdr.session_id,
                    signals=list(hdr.signals),
                    tstart=hdr.tstart,
                    tstop=hdr.tstop,
                    total_samples=hdr.total_samples,
                )
                self._signals = self._header.signals
            elif payload == "sample":
                sample = WaveformSample(
                    time=response.sample.time,
                    values={
                        sig: val
                        for sig, val in zip(self._signals, response.sample.values)
                    },
                )
                self._samples.append(sample)
                return sample
            elif payload == "complete":
                self._complete = True
                self._final_status = SessionStatus(response.complete.final_status)
                if response.complete.error_message:
                    self._error = response.complete.error_message
                raise StopAsyncIteration
        raise StopAsyncIteration

    async def collect(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Collect all samples into numpy arrays asynchronously."""
        if not self._complete:
            async for _ in self:
                pass

        if not self._samples:
            return np.array([]), {}

        times = np.array([s.time for s in self._samples])
        signals = {
            sig: np.array([s.values[sig] for s in self._samples])
            for sig in self._signals
        }
        return times, signals

    async def to_dataframe(self) -> "pd.DataFrame":
        """Convert to pandas DataFrame asynchronously."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for DataFrame conversion.")

        times, signals = await self.collect()
        df = pd.DataFrame(signals, index=times)
        df.index.name = "time"
        return df

    async def to_xarray(self) -> "xr.Dataset":
        """Convert to xarray Dataset asynchronously."""
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("xarray is required for Dataset conversion.")

        times, signals = await self.collect()
        data_vars = {
            name: (["time"], values) for name, values in signals.items()
        }
        return xr.Dataset(data_vars, coords={"time": times})


@dataclass
class Session:
    """Represents a simulation session on the server."""

    session_id: str
    name: str
    status: SessionStatus
    model_id: Optional[str] = None
    active_signals: List[str] = field(default_factory=list)
    owner: Optional[str] = None

    @classmethod
    def from_proto(cls, desc: pb.SessionDescriptor) -> Session:
        """Create from protobuf message."""
        return cls(
            session_id=desc.session_id,
            name=desc.name,
            status=SessionStatus(desc.status),
            model_id=desc.model_id if desc.model_id else None,
            active_signals=list(desc.active_signals),
            owner=desc.owner if desc.owner else None,
        )


@dataclass
class HealthCheckResult:
    """Server health check result."""

    status: HealthStatus
    version: str
    active_sessions: int
    completed_sessions: int
    authentication_enabled: bool


class PulsimClient:
    """gRPC client for Pulsim simulation server.

    Provides both synchronous and asynchronous interfaces for running
    circuit simulations on a remote server.

    Example:
        >>> client = PulsimClient("localhost:50051")
        >>> with client:
        ...     session = client.create_session(
        ...         name="RC Circuit",
        ...         netlist={"name": "RC", "components": [...]},
        ...         options=SimulationOptions(tstop=0.01, dt=1e-6)
        ...     )
        ...     client.start_simulation(session.session_id)
        ...     stream = client.stream_waveforms(session.session_id)
        ...     df = stream.to_dataframe()
    """

    def __init__(
        self,
        address: str = "localhost:50051",
        secure: bool = False,
        credentials: Optional[grpc.ChannelCredentials] = None,
        timeout: float = 30.0,
    ):
        """Initialize the client.

        Args:
            address: Server address in host:port format.
            secure: Use TLS/SSL connection.
            credentials: Optional custom channel credentials.
            timeout: Default timeout for RPC calls in seconds.
        """
        self._address = address
        self._secure = secure
        self._credentials = credentials
        self._timeout = timeout
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[pb_grpc.SimulatorServiceStub] = None
        self._async_channel: Optional[grpc.aio.Channel] = None
        self._async_stub: Optional[pb_grpc.SimulatorServiceStub] = None

    def connect(self) -> None:
        """Establish connection to the server."""
        if self._channel is not None:
            return

        if self._secure:
            if self._credentials:
                self._channel = grpc.secure_channel(self._address, self._credentials)
            else:
                self._channel = grpc.secure_channel(
                    self._address, grpc.ssl_channel_credentials()
                )
        else:
            self._channel = grpc.insecure_channel(self._address)

        self._stub = pb_grpc.SimulatorServiceStub(self._channel)
        logger.info(f"Connected to Pulsim server at {self._address}")

    def close(self) -> None:
        """Close the connection."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None
            logger.info("Disconnected from Pulsim server")

    def __enter__(self) -> PulsimClient:
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    @property
    def stub(self) -> pb_grpc.SimulatorServiceStub:
        """Get the gRPC stub, connecting if necessary."""
        if self._stub is None:
            self.connect()
        return self._stub

    def health_check(self) -> HealthCheckResult:
        """Check server health.

        Returns:
            HealthCheckResult with server status information.
        """
        request = pb.HealthCheckRequest()
        response = self.stub.HealthCheck(request, timeout=self._timeout)
        return HealthCheckResult(
            status=HealthStatus(response.status),
            version=response.version,
            active_sessions=response.active_sessions,
            completed_sessions=response.completed_sessions,
            authentication_enabled=response.authentication_enabled,
        )

    def create_session(
        self,
        name: str,
        netlist: Union[str, Dict[str, Any]],
        options: Optional[SimulationOptions] = None,
        model_id: Optional[str] = None,
    ) -> Session:
        """Create a new simulation session.

        Args:
            name: Session name.
            netlist: Circuit netlist as JSON string or dict.
            options: Simulation options.
            model_id: Optional existing model ID to use.

        Returns:
            Created Session object.
        """
        request = pb.CreateSessionRequest(name=name)

        if model_id:
            request.model_id = model_id
        else:
            model = pb.CircuitModel()
            model.name = name
            if isinstance(netlist, dict):
                model.model_json = json.dumps(netlist)
            else:
                model.model_json = netlist
            request.inline_model.CopyFrom(model)

        if options:
            request.options.CopyFrom(options.to_proto())

        response = self.stub.CreateSession(request, timeout=self._timeout)
        return Session.from_proto(response.session)

    def get_session(self, session_id: str) -> Session:
        """Get session information.

        Args:
            session_id: Session identifier.

        Returns:
            Session object with current status.
        """
        request = pb.GetSessionRequest(session_id=session_id)
        response = self.stub.GetSession(request, timeout=self._timeout)
        return Session.from_proto(response.session)

    def list_sessions(self) -> List[Session]:
        """List all sessions.

        Returns:
            List of Session objects.
        """
        request = pb.ListSessionsRequest()
        response = self.stub.ListSessions(request, timeout=self._timeout)
        return [Session.from_proto(s) for s in response.sessions]

    def start_simulation(
        self,
        session_id: str,
        overrides: Optional[SimulationOptions] = None,
    ) -> Session:
        """Start a simulation.

        Args:
            session_id: Session identifier.
            overrides: Optional options to override.

        Returns:
            Updated Session object.
        """
        request = pb.StartSimulationRequest(session_id=session_id)
        if overrides:
            request.overrides.CopyFrom(overrides.to_proto())

        response = self.stub.StartSimulation(request, timeout=self._timeout)
        return Session.from_proto(response.session)

    def pause_simulation(self, session_id: str) -> Session:
        """Pause a running simulation.

        Args:
            session_id: Session identifier.

        Returns:
            Updated Session object.
        """
        request = pb.PauseSimulationRequest(session_id=session_id)
        response = self.stub.PauseSimulation(request, timeout=self._timeout)
        return Session.from_proto(response.session)

    def resume_simulation(self, session_id: str) -> Session:
        """Resume a paused simulation.

        Args:
            session_id: Session identifier.

        Returns:
            Updated Session object.
        """
        request = pb.ResumeSimulationRequest(session_id=session_id)
        response = self.stub.ResumeSimulation(request, timeout=self._timeout)
        return Session.from_proto(response.session)

    def stop_simulation(self, session_id: str) -> Session:
        """Stop a simulation.

        Args:
            session_id: Session identifier.

        Returns:
            Updated Session object.
        """
        request = pb.StopSimulationRequest(session_id=session_id)
        response = self.stub.StopSimulation(request, timeout=self._timeout)
        return Session.from_proto(response.session)

    def stream_waveforms(
        self,
        session_id: str,
        signals: Optional[List[str]] = None,
        decimation: int = 1,
        start_time: Optional[float] = None,
    ) -> WaveformStream:
        """Stream waveform data from a simulation.

        Args:
            session_id: Session identifier.
            signals: List of signal names to stream (all if None).
            decimation: Decimation factor (1 = all samples).
            start_time: Start time for streaming.

        Returns:
            WaveformStream iterator.
        """
        request = pb.StreamWaveformsRequest(
            session_id=session_id,
            decimation=decimation,
        )
        if signals:
            request.signals.extend(signals)
        if start_time is not None:
            request.start_time.value = start_time

        response_iterator = self.stub.StreamWaveforms(request)
        return WaveformStream(response_iterator, signals or [])

    def get_result(
        self,
        session_id: str,
        signals: Optional[List[str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Get simulation result metadata.

        Args:
            session_id: Session identifier.
            signals: List of signal names.
            start_time: Start time filter.
            end_time: End time filter.

        Returns:
            Result metadata dictionary.
        """
        request = pb.GetResultRequest(session_id=session_id)
        if signals:
            request.signals.extend(signals)
        if start_time is not None:
            request.start_time.value = start_time
        if end_time is not None:
            request.end_time.value = end_time

        response = self.stub.GetResult(request, timeout=self._timeout)
        meta = response.metadata
        return {
            "start_time": meta.start_time,
            "end_time": meta.end_time,
            "sample_count": meta.sample_count,
            "signals": list(meta.signals),
            "status": SessionStatus(meta.status),
            "error_message": meta.error_message,
        }

    def download_result_csv(
        self,
        session_id: str,
        signals: Optional[List[str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> bytes:
        """Download simulation results as CSV.

        Args:
            session_id: Session identifier.
            signals: List of signal names.
            start_time: Start time filter.
            end_time: End time filter.

        Returns:
            CSV data as bytes.
        """
        request = pb.DownloadResultRequest(
            session_id=session_id,
            format=pb.SIMULATION_RESULT_FORMAT_CSV,
        )
        if signals:
            request.signals.extend(signals)
        if start_time is not None:
            request.start_time.value = start_time
        if end_time is not None:
            request.end_time.value = end_time

        chunks = []
        for response in self.stub.DownloadResult(request):
            chunks.append(response.chunk)
        return b"".join(chunks)

    def run_simulation(
        self,
        name: str,
        netlist: Union[str, Dict[str, Any]],
        options: Optional[SimulationOptions] = None,
        signals: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Convenience method to run a complete simulation.

        Creates a session, runs the simulation, and returns results.

        Args:
            name: Session name.
            netlist: Circuit netlist.
            options: Simulation options.
            signals: Signals to capture.

        Returns:
            Tuple of (time_array, {signal_name: values_array})
        """
        session = self.create_session(name, netlist, options)
        self.start_simulation(session.session_id)

        stream = self.stream_waveforms(session.session_id, signals)
        return stream.collect()

    # Async interface

    async def aconnect(self) -> None:
        """Establish async connection to the server."""
        if self._async_channel is not None:
            return

        if self._secure:
            if self._credentials:
                self._async_channel = grpc.aio.secure_channel(
                    self._address, self._credentials
                )
            else:
                self._async_channel = grpc.aio.secure_channel(
                    self._address, grpc.ssl_channel_credentials()
                )
        else:
            self._async_channel = grpc.aio.insecure_channel(self._address)

        self._async_stub = pb_grpc.SimulatorServiceStub(self._async_channel)
        logger.info(f"Async connected to Pulsim server at {self._address}")

    async def aclose(self) -> None:
        """Close the async connection."""
        if self._async_channel is not None:
            await self._async_channel.close()
            self._async_channel = None
            self._async_stub = None
            logger.info("Async disconnected from Pulsim server")

    async def __aenter__(self) -> PulsimClient:
        """Async context manager entry."""
        await self.aconnect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.aclose()

    @property
    def async_stub(self) -> pb_grpc.SimulatorServiceStub:
        """Get the async gRPC stub."""
        if self._async_stub is None:
            raise RuntimeError("Async connection not established. Use aconnect().")
        return self._async_stub

    async def ahealth_check(self) -> HealthCheckResult:
        """Async health check."""
        request = pb.HealthCheckRequest()
        response = await self.async_stub.HealthCheck(request, timeout=self._timeout)
        return HealthCheckResult(
            status=HealthStatus(response.status),
            version=response.version,
            active_sessions=response.active_sessions,
            completed_sessions=response.completed_sessions,
            authentication_enabled=response.authentication_enabled,
        )

    async def acreate_session(
        self,
        name: str,
        netlist: Union[str, Dict[str, Any]],
        options: Optional[SimulationOptions] = None,
        model_id: Optional[str] = None,
    ) -> Session:
        """Async create session."""
        request = pb.CreateSessionRequest(name=name)

        if model_id:
            request.model_id = model_id
        else:
            model = pb.CircuitModel()
            model.name = name
            if isinstance(netlist, dict):
                model.model_json = json.dumps(netlist)
            else:
                model.model_json = netlist
            request.inline_model.CopyFrom(model)

        if options:
            request.options.CopyFrom(options.to_proto())

        response = await self.async_stub.CreateSession(request, timeout=self._timeout)
        return Session.from_proto(response.session)

    async def aget_session(self, session_id: str) -> Session:
        """Async get session."""
        request = pb.GetSessionRequest(session_id=session_id)
        response = await self.async_stub.GetSession(request, timeout=self._timeout)
        return Session.from_proto(response.session)

    async def alist_sessions(self) -> List[Session]:
        """Async list sessions."""
        request = pb.ListSessionsRequest()
        response = await self.async_stub.ListSessions(request, timeout=self._timeout)
        return [Session.from_proto(s) for s in response.sessions]

    async def astart_simulation(
        self,
        session_id: str,
        overrides: Optional[SimulationOptions] = None,
    ) -> Session:
        """Async start simulation."""
        request = pb.StartSimulationRequest(session_id=session_id)
        if overrides:
            request.overrides.CopyFrom(overrides.to_proto())

        response = await self.async_stub.StartSimulation(request, timeout=self._timeout)
        return Session.from_proto(response.session)

    async def apause_simulation(self, session_id: str) -> Session:
        """Async pause simulation."""
        request = pb.PauseSimulationRequest(session_id=session_id)
        response = await self.async_stub.PauseSimulation(request, timeout=self._timeout)
        return Session.from_proto(response.session)

    async def aresume_simulation(self, session_id: str) -> Session:
        """Async resume simulation."""
        request = pb.ResumeSimulationRequest(session_id=session_id)
        response = await self.async_stub.ResumeSimulation(
            request, timeout=self._timeout
        )
        return Session.from_proto(response.session)

    async def astop_simulation(self, session_id: str) -> Session:
        """Async stop simulation."""
        request = pb.StopSimulationRequest(session_id=session_id)
        response = await self.async_stub.StopSimulation(request, timeout=self._timeout)
        return Session.from_proto(response.session)

    async def astream_waveforms(
        self,
        session_id: str,
        signals: Optional[List[str]] = None,
        decimation: int = 1,
        start_time: Optional[float] = None,
    ) -> AsyncWaveformStream:
        """Async stream waveforms."""
        request = pb.StreamWaveformsRequest(
            session_id=session_id,
            decimation=decimation,
        )
        if signals:
            request.signals.extend(signals)
        if start_time is not None:
            request.start_time.value = start_time

        response_iterator = self.async_stub.StreamWaveforms(request)
        return AsyncWaveformStream(response_iterator, signals or [])

    async def aget_result(
        self,
        session_id: str,
        signals: Optional[List[str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Async get result."""
        request = pb.GetResultRequest(session_id=session_id)
        if signals:
            request.signals.extend(signals)
        if start_time is not None:
            request.start_time.value = start_time
        if end_time is not None:
            request.end_time.value = end_time

        response = await self.async_stub.GetResult(request, timeout=self._timeout)
        meta = response.metadata
        return {
            "start_time": meta.start_time,
            "end_time": meta.end_time,
            "sample_count": meta.sample_count,
            "signals": list(meta.signals),
            "status": SessionStatus(meta.status),
            "error_message": meta.error_message,
        }

    async def arun_simulation(
        self,
        name: str,
        netlist: Union[str, Dict[str, Any]],
        options: Optional[SimulationOptions] = None,
        signals: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Async convenience method to run a complete simulation."""
        session = await self.acreate_session(name, netlist, options)
        await self.astart_simulation(session.session_id)

        stream = await self.astream_waveforms(session.session_id, signals)
        return await stream.collect()
