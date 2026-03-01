"""Tests for the Pulsim gRPC Python client."""

import json
import pytest
from unittest.mock import MagicMock, patch

import numpy as np

# Skip all tests in this module if grpc is not installed
pytest.importorskip("grpc")


class TestSimulationOptions:
    """Tests for SimulationOptions dataclass."""

    def test_default_options(self):
        """Test default options creation."""
        from pulsim.remote.client import SimulationOptions

        opts = SimulationOptions()
        assert opts.tstart is None
        assert opts.tstop is None
        assert opts.dt is None
        assert opts.output_signals == []

    def test_options_with_values(self):
        """Test options with custom values."""
        from pulsim.remote.client import SimulationOptions

        opts = SimulationOptions(
            tstart=0.0,
            tstop=0.01,
            dt=1e-6,
            abstol=1e-9,
            reltol=1e-6,
            output_signals=["V(out)", "I(R1)"],
        )

        assert opts.tstart == 0.0
        assert opts.tstop == 0.01
        assert opts.dt == 1e-6
        assert opts.abstol == 1e-9
        assert opts.reltol == 1e-6
        assert opts.output_signals == ["V(out)", "I(R1)"]

    def test_to_proto(self):
        """Test conversion to protobuf message."""
        from pulsim.remote.client import SimulationOptions
        from pulsim.remote import simulator_pb2 as pb

        opts = SimulationOptions(
            tstop=0.01,
            dt=1e-6,
            output_signals=["V(out)"],
        )

        proto = opts.to_proto()

        assert isinstance(proto, pb.SimulationOptions)
        assert proto.tstop.value == 0.01
        assert proto.dt.value == 1e-6
        assert list(proto.output_signals) == ["V(out)"]


class TestSession:
    """Tests for Session dataclass."""

    def test_session_creation(self):
        """Test session creation."""
        from pulsim.remote.client import Session, SessionStatus

        session = Session(
            session_id="test-123",
            name="Test Session",
            status=SessionStatus.READY,
            active_signals=["V(out)"],
        )

        assert session.session_id == "test-123"
        assert session.name == "Test Session"
        assert session.status == SessionStatus.READY
        assert session.active_signals == ["V(out)"]

    def test_from_proto(self):
        """Test creation from protobuf message."""
        from pulsim.remote.client import Session, SessionStatus
        from pulsim.remote import simulator_pb2 as pb

        proto = pb.SessionDescriptor(
            session_id="proto-456",
            name="Proto Session",
            status=pb.SESSION_STATUS_RUNNING,
        )
        proto.active_signals.extend(["V(in)", "V(out)"])

        session = Session.from_proto(proto)

        assert session.session_id == "proto-456"
        assert session.name == "Proto Session"
        assert session.status == SessionStatus.RUNNING
        assert session.active_signals == ["V(in)", "V(out)"]


class TestSessionStatus:
    """Tests for SessionStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        from pulsim.remote.client import SessionStatus

        assert SessionStatus.CREATED == 1
        assert SessionStatus.READY == 3
        assert SessionStatus.RUNNING == 4
        assert SessionStatus.COMPLETED == 6
        assert SessionStatus.FAILED == 8


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_health_values(self):
        """Test health enum values."""
        from pulsim.remote.client import HealthStatus

        assert HealthStatus.OK == 1
        assert HealthStatus.DEGRADED == 2
        assert HealthStatus.ERROR == 3


class TestWaveformSample:
    """Tests for WaveformSample dataclass."""

    def test_sample_creation(self):
        """Test sample creation."""
        from pulsim.remote.client import WaveformSample

        sample = WaveformSample(
            time=0.001,
            values={"V(out)": 3.3, "I(R1)": 0.001},
        )

        assert sample.time == 0.001
        assert sample.values["V(out)"] == 3.3
        assert sample.values["I(R1)"] == 0.001


class TestWaveformStream:
    """Tests for WaveformStream."""

    def _create_mock_responses(self):
        """Create mock response iterator."""
        from pulsim.remote import simulator_pb2 as pb

        # Header
        header_resp = pb.WaveformStreamResponse()
        header_resp.header.session_id = "test-123"
        header_resp.header.signals.extend(["V(out)", "I(R1)"])
        header_resp.header.tstart = 0.0
        header_resp.header.tstop = 0.01
        header_resp.header.total_samples = 100

        # Samples
        samples = []
        for i in range(5):
            sample_resp = pb.WaveformStreamResponse()
            sample_resp.sample.time = i * 0.001
            sample_resp.sample.values.extend([3.3 * (1 - np.exp(-i)), 0.001 * i])
            samples.append(sample_resp)

        # Complete
        complete_resp = pb.WaveformStreamResponse()
        complete_resp.complete.final_status = pb.SESSION_STATUS_COMPLETED

        return [header_resp] + samples + [complete_resp]

    def test_iterate_samples(self):
        """Test iterating over samples."""
        from pulsim.remote.client import WaveformStream

        responses = self._create_mock_responses()
        stream = WaveformStream(iter(responses), ["V(out)", "I(R1)"])

        samples = list(stream)

        assert len(samples) == 5
        assert samples[0].time == 0.0
        assert "V(out)" in samples[0].values
        assert "I(R1)" in samples[0].values

    def test_header_property(self):
        """Test header property."""
        from pulsim.remote.client import WaveformStream

        responses = self._create_mock_responses()
        stream = WaveformStream(iter(responses), [])

        # Consume first sample to get header
        next(iter(stream))

        assert stream.header is not None
        assert stream.header.session_id == "test-123"
        assert stream.header.signals == ["V(out)", "I(R1)"]

    def test_collect(self):
        """Test collecting all samples."""
        from pulsim.remote.client import WaveformStream

        responses = self._create_mock_responses()
        stream = WaveformStream(iter(responses), ["V(out)", "I(R1)"])

        times, signals = stream.collect()

        assert len(times) == 5
        assert "V(out)" in signals
        assert "I(R1)" in signals
        assert len(signals["V(out)"]) == 5

    def test_is_complete(self):
        """Test completion detection."""
        from pulsim.remote.client import WaveformStream, SessionStatus

        responses = self._create_mock_responses()
        stream = WaveformStream(iter(responses), [])

        assert not stream.is_complete

        # Consume all
        list(stream)

        assert stream.is_complete
        assert stream.final_status == SessionStatus.COMPLETED

    def test_to_dataframe(self):
        """Test DataFrame conversion."""
        pytest.importorskip("pandas")
        from pulsim.remote.client import WaveformStream

        responses = self._create_mock_responses()
        stream = WaveformStream(iter(responses), ["V(out)", "I(R1)"])

        df = stream.to_dataframe()

        assert df.index.name == "time"
        assert "V(out)" in df.columns
        assert "I(R1)" in df.columns
        assert len(df) == 5

    def test_to_xarray(self):
        """Test xarray conversion."""
        pytest.importorskip("xarray")
        from pulsim.remote.client import WaveformStream

        responses = self._create_mock_responses()
        stream = WaveformStream(iter(responses), ["V(out)", "I(R1)"])

        ds = stream.to_xarray()

        assert "time" in ds.coords
        assert "V(out)" in ds.data_vars
        assert "I(R1)" in ds.data_vars


class TestPulsimClient:
    """Tests for PulsimClient."""

    def test_client_creation(self):
        """Test client instantiation."""
        from pulsim.remote.client import PulsimClient

        client = PulsimClient("localhost:50051")

        assert client._address == "localhost:50051"
        assert client._timeout == 30.0
        assert not client._secure

    def test_client_with_options(self):
        """Test client with custom options."""
        from pulsim.remote.client import PulsimClient

        client = PulsimClient(
            address="server:50052",
            secure=True,
            timeout=60.0,
        )

        assert client._address == "server:50052"
        assert client._secure
        assert client._timeout == 60.0

    def test_connect(self):
        """Test connection establishment."""
        from pulsim.remote.client import PulsimClient

        with patch("pulsim.remote.client.grpc") as mock_grpc:
            mock_channel = MagicMock()
            mock_grpc.insecure_channel.return_value = mock_channel

            client = PulsimClient("localhost:50051")
            client.connect()

            mock_grpc.insecure_channel.assert_called_once_with("localhost:50051")
            assert client._channel is not None

    def test_context_manager(self):
        """Test context manager usage."""
        from pulsim.remote.client import PulsimClient

        with patch("pulsim.remote.client.grpc") as mock_grpc:
            mock_channel = MagicMock()
            mock_grpc.insecure_channel.return_value = mock_channel

            with PulsimClient("localhost:50051") as client:
                assert client._channel is not None

            mock_channel.close.assert_called_once()

    def test_health_check(self):
        """Test health check call."""
        from pulsim.remote.client import PulsimClient, HealthStatus
        from pulsim.remote import simulator_pb2 as pb

        mock_stub = MagicMock()
        mock_response = pb.HealthCheckResponse(
            status=pb.HEALTH_STATUS_OK,
            version="1.0.0",
            active_sessions=5,
            completed_sessions=100,
            authentication_enabled=False,
        )
        mock_stub.HealthCheck.return_value = mock_response

        client = PulsimClient("localhost:50051")
        client._stub = mock_stub

        result = client.health_check()

        assert result.status == HealthStatus.OK
        assert result.version == "1.0.0"
        assert result.active_sessions == 5
        assert result.completed_sessions == 100
        assert not result.authentication_enabled

    def test_create_session_with_dict(self):
        """Test session creation with dict netlist."""
        from pulsim.remote.client import PulsimClient, SimulationOptions
        from pulsim.remote import simulator_pb2 as pb

        mock_stub = MagicMock()
        mock_response = pb.CreateSessionResponse()
        mock_response.session.session_id = "new-session"
        mock_response.session.name = "RC Circuit"
        mock_response.session.status = pb.SESSION_STATUS_READY
        mock_stub.CreateSession.return_value = mock_response

        client = PulsimClient("localhost:50051")
        client._stub = mock_stub

        netlist = {
            "name": "RC Circuit",
            "components": [
                {"name": "V1", "type": "V", "n1": "in", "n2": "0", "waveform": 5.0},
                {"name": "R1", "type": "R", "n1": "in", "n2": "out", "value": 1000},
                {"name": "C1", "type": "C", "n1": "out", "n2": "0", "value": 1e-6},
            ],
        }

        session = client.create_session(
            name="RC Circuit",
            netlist=netlist,
            options=SimulationOptions(tstop=0.01, dt=1e-6),
        )

        assert session.session_id == "new-session"
        assert session.name == "RC Circuit"

        # Verify the request
        call_args = mock_stub.CreateSession.call_args
        request = call_args[0][0]
        assert request.name == "RC Circuit"
        assert json.loads(request.inline_model.model_json) == netlist

    def test_start_simulation(self):
        """Test starting simulation."""
        from pulsim.remote.client import PulsimClient, SessionStatus
        from pulsim.remote import simulator_pb2 as pb

        mock_stub = MagicMock()
        mock_response = pb.StartSimulationResponse()
        mock_response.session.session_id = "test-123"
        mock_response.session.status = pb.SESSION_STATUS_RUNNING
        mock_stub.StartSimulation.return_value = mock_response

        client = PulsimClient("localhost:50051")
        client._stub = mock_stub

        session = client.start_simulation("test-123")

        assert session.status == SessionStatus.RUNNING

    def test_list_sessions(self):
        """Test listing sessions."""
        from pulsim.remote.client import PulsimClient
        from pulsim.remote import simulator_pb2 as pb

        mock_stub = MagicMock()
        mock_response = pb.ListSessionsResponse()

        s1 = mock_response.sessions.add()
        s1.session_id = "session-1"
        s1.name = "Session 1"
        s1.status = pb.SESSION_STATUS_COMPLETED

        s2 = mock_response.sessions.add()
        s2.session_id = "session-2"
        s2.name = "Session 2"
        s2.status = pb.SESSION_STATUS_RUNNING

        mock_stub.ListSessions.return_value = mock_response

        client = PulsimClient("localhost:50051")
        client._stub = mock_stub

        sessions = client.list_sessions()

        assert len(sessions) == 2
        assert sessions[0].session_id == "session-1"
        assert sessions[1].session_id == "session-2"


class TestDataConversion:
    """Tests for data conversion utilities."""

    def test_numpy_array_conversion(self):
        """Test conversion to numpy arrays."""
        from pulsim.remote.client import WaveformSample

        samples = [
            WaveformSample(time=0.0, values={"V": 1.0, "I": 0.001}),
            WaveformSample(time=0.001, values={"V": 2.0, "I": 0.002}),
            WaveformSample(time=0.002, values={"V": 3.0, "I": 0.003}),
        ]

        times = np.array([s.time for s in samples])
        v_values = np.array([s.values["V"] for s in samples])
        i_values = np.array([s.values["I"] for s in samples])

        np.testing.assert_array_equal(times, [0.0, 0.001, 0.002])
        np.testing.assert_array_equal(v_values, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(i_values, [0.001, 0.002, 0.003])
