import datetime

from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf import duration_pb2 as _duration_pb2
from google.protobuf import wrappers_pb2 as _wrappers_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class HealthStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    HEALTH_STATUS_UNSPECIFIED: _ClassVar[HealthStatus]
    HEALTH_STATUS_OK: _ClassVar[HealthStatus]
    HEALTH_STATUS_DEGRADED: _ClassVar[HealthStatus]
    HEALTH_STATUS_ERROR: _ClassVar[HealthStatus]

class SessionStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SESSION_STATUS_UNSPECIFIED: _ClassVar[SessionStatus]
    SESSION_STATUS_CREATED: _ClassVar[SessionStatus]
    SESSION_STATUS_VALIDATING: _ClassVar[SessionStatus]
    SESSION_STATUS_READY: _ClassVar[SessionStatus]
    SESSION_STATUS_RUNNING: _ClassVar[SessionStatus]
    SESSION_STATUS_PAUSED: _ClassVar[SessionStatus]
    SESSION_STATUS_COMPLETED: _ClassVar[SessionStatus]
    SESSION_STATUS_STOPPED: _ClassVar[SessionStatus]
    SESSION_STATUS_FAILED: _ClassVar[SessionStatus]

class SimulationResultFormat(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SIMULATION_RESULT_FORMAT_UNSPECIFIED: _ClassVar[SimulationResultFormat]
    SIMULATION_RESULT_FORMAT_CSV: _ClassVar[SimulationResultFormat]
    SIMULATION_RESULT_FORMAT_HDF5: _ClassVar[SimulationResultFormat]
    SIMULATION_RESULT_FORMAT_PARQUET: _ClassVar[SimulationResultFormat]

class SweepStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SWEEP_STATUS_UNSPECIFIED: _ClassVar[SweepStatus]
    SWEEP_STATUS_PENDING: _ClassVar[SweepStatus]
    SWEEP_STATUS_RUNNING: _ClassVar[SweepStatus]
    SWEEP_STATUS_COMPLETED: _ClassVar[SweepStatus]
    SWEEP_STATUS_FAILED: _ClassVar[SweepStatus]
    SWEEP_STATUS_CANCELLED: _ClassVar[SweepStatus]
HEALTH_STATUS_UNSPECIFIED: HealthStatus
HEALTH_STATUS_OK: HealthStatus
HEALTH_STATUS_DEGRADED: HealthStatus
HEALTH_STATUS_ERROR: HealthStatus
SESSION_STATUS_UNSPECIFIED: SessionStatus
SESSION_STATUS_CREATED: SessionStatus
SESSION_STATUS_VALIDATING: SessionStatus
SESSION_STATUS_READY: SessionStatus
SESSION_STATUS_RUNNING: SessionStatus
SESSION_STATUS_PAUSED: SessionStatus
SESSION_STATUS_COMPLETED: SessionStatus
SESSION_STATUS_STOPPED: SessionStatus
SESSION_STATUS_FAILED: SessionStatus
SIMULATION_RESULT_FORMAT_UNSPECIFIED: SimulationResultFormat
SIMULATION_RESULT_FORMAT_CSV: SimulationResultFormat
SIMULATION_RESULT_FORMAT_HDF5: SimulationResultFormat
SIMULATION_RESULT_FORMAT_PARQUET: SimulationResultFormat
SWEEP_STATUS_UNSPECIFIED: SweepStatus
SWEEP_STATUS_PENDING: SweepStatus
SWEEP_STATUS_RUNNING: SweepStatus
SWEEP_STATUS_COMPLETED: SweepStatus
SWEEP_STATUS_FAILED: SweepStatus
SWEEP_STATUS_CANCELLED: SweepStatus

class SimulationOptions(_message.Message):
    __slots__ = ("tstart", "tstop", "dt", "dtmin", "dtmax", "abstol", "reltol", "max_newton_iterations", "damping_factor", "use_ic", "output_signals")
    TSTART_FIELD_NUMBER: _ClassVar[int]
    TSTOP_FIELD_NUMBER: _ClassVar[int]
    DT_FIELD_NUMBER: _ClassVar[int]
    DTMIN_FIELD_NUMBER: _ClassVar[int]
    DTMAX_FIELD_NUMBER: _ClassVar[int]
    ABSTOL_FIELD_NUMBER: _ClassVar[int]
    RELTOL_FIELD_NUMBER: _ClassVar[int]
    MAX_NEWTON_ITERATIONS_FIELD_NUMBER: _ClassVar[int]
    DAMPING_FACTOR_FIELD_NUMBER: _ClassVar[int]
    USE_IC_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SIGNALS_FIELD_NUMBER: _ClassVar[int]
    tstart: _wrappers_pb2.DoubleValue
    tstop: _wrappers_pb2.DoubleValue
    dt: _wrappers_pb2.DoubleValue
    dtmin: _wrappers_pb2.DoubleValue
    dtmax: _wrappers_pb2.DoubleValue
    abstol: _wrappers_pb2.DoubleValue
    reltol: _wrappers_pb2.DoubleValue
    max_newton_iterations: _wrappers_pb2.Int32Value
    damping_factor: _wrappers_pb2.DoubleValue
    use_ic: _wrappers_pb2.BoolValue
    output_signals: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, tstart: _Optional[_Union[_wrappers_pb2.DoubleValue, _Mapping]] = ..., tstop: _Optional[_Union[_wrappers_pb2.DoubleValue, _Mapping]] = ..., dt: _Optional[_Union[_wrappers_pb2.DoubleValue, _Mapping]] = ..., dtmin: _Optional[_Union[_wrappers_pb2.DoubleValue, _Mapping]] = ..., dtmax: _Optional[_Union[_wrappers_pb2.DoubleValue, _Mapping]] = ..., abstol: _Optional[_Union[_wrappers_pb2.DoubleValue, _Mapping]] = ..., reltol: _Optional[_Union[_wrappers_pb2.DoubleValue, _Mapping]] = ..., max_newton_iterations: _Optional[_Union[_wrappers_pb2.Int32Value, _Mapping]] = ..., damping_factor: _Optional[_Union[_wrappers_pb2.DoubleValue, _Mapping]] = ..., use_ic: _Optional[_Union[_wrappers_pb2.BoolValue, _Mapping]] = ..., output_signals: _Optional[_Iterable[str]] = ...) -> None: ...

class CircuitModel(_message.Message):
    __slots__ = ("name", "description", "model_json", "created_at", "created_by")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    MODEL_JSON_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    CREATED_BY_FIELD_NUMBER: _ClassVar[int]
    name: str
    description: str
    model_json: str
    created_at: _timestamp_pb2.Timestamp
    created_by: str
    def __init__(self, name: _Optional[str] = ..., description: _Optional[str] = ..., model_json: _Optional[str] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., created_by: _Optional[str] = ...) -> None: ...

class ModelDescriptor(_message.Message):
    __slots__ = ("model_id", "model")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    model: CircuitModel
    def __init__(self, model_id: _Optional[str] = ..., model: _Optional[_Union[CircuitModel, _Mapping]] = ...) -> None: ...

class SessionDescriptor(_message.Message):
    __slots__ = ("session_id", "model_id", "name", "status", "created_at", "updated_at", "active_signals", "retention", "owner")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_SIGNALS_FIELD_NUMBER: _ClassVar[int]
    RETENTION_FIELD_NUMBER: _ClassVar[int]
    OWNER_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    model_id: str
    name: str
    status: SessionStatus
    created_at: _timestamp_pb2.Timestamp
    updated_at: _timestamp_pb2.Timestamp
    active_signals: _containers.RepeatedScalarFieldContainer[str]
    retention: _duration_pb2.Duration
    owner: str
    def __init__(self, session_id: _Optional[str] = ..., model_id: _Optional[str] = ..., name: _Optional[str] = ..., status: _Optional[_Union[SessionStatus, str]] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., active_signals: _Optional[_Iterable[str]] = ..., retention: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., owner: _Optional[str] = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("status", "version", "uptime", "active_sessions", "completed_sessions", "authentication_enabled")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    UPTIME_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_SESSIONS_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_SESSIONS_FIELD_NUMBER: _ClassVar[int]
    AUTHENTICATION_ENABLED_FIELD_NUMBER: _ClassVar[int]
    status: HealthStatus
    version: str
    uptime: _duration_pb2.Duration
    active_sessions: int
    completed_sessions: int
    authentication_enabled: bool
    def __init__(self, status: _Optional[_Union[HealthStatus, str]] = ..., version: _Optional[str] = ..., uptime: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., active_sessions: _Optional[int] = ..., completed_sessions: _Optional[int] = ..., authentication_enabled: bool = ...) -> None: ...

class CreateSessionRequest(_message.Message):
    __slots__ = ("name", "model_id", "inline_model", "options")
    NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    INLINE_MODEL_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    name: str
    model_id: str
    inline_model: CircuitModel
    options: SimulationOptions
    def __init__(self, name: _Optional[str] = ..., model_id: _Optional[str] = ..., inline_model: _Optional[_Union[CircuitModel, _Mapping]] = ..., options: _Optional[_Union[SimulationOptions, _Mapping]] = ...) -> None: ...

class CreateSessionResponse(_message.Message):
    __slots__ = ("session",)
    SESSION_FIELD_NUMBER: _ClassVar[int]
    session: SessionDescriptor
    def __init__(self, session: _Optional[_Union[SessionDescriptor, _Mapping]] = ...) -> None: ...

class ListSessionsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListSessionsResponse(_message.Message):
    __slots__ = ("sessions",)
    SESSIONS_FIELD_NUMBER: _ClassVar[int]
    sessions: _containers.RepeatedCompositeFieldContainer[SessionDescriptor]
    def __init__(self, sessions: _Optional[_Iterable[_Union[SessionDescriptor, _Mapping]]] = ...) -> None: ...

class GetSessionRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class GetSessionResponse(_message.Message):
    __slots__ = ("session",)
    SESSION_FIELD_NUMBER: _ClassVar[int]
    session: SessionDescriptor
    def __init__(self, session: _Optional[_Union[SessionDescriptor, _Mapping]] = ...) -> None: ...

class StartSimulationRequest(_message.Message):
    __slots__ = ("session_id", "overrides")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    OVERRIDES_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    overrides: SimulationOptions
    def __init__(self, session_id: _Optional[str] = ..., overrides: _Optional[_Union[SimulationOptions, _Mapping]] = ...) -> None: ...

class StartSimulationResponse(_message.Message):
    __slots__ = ("session",)
    SESSION_FIELD_NUMBER: _ClassVar[int]
    session: SessionDescriptor
    def __init__(self, session: _Optional[_Union[SessionDescriptor, _Mapping]] = ...) -> None: ...

class PauseSimulationRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class PauseSimulationResponse(_message.Message):
    __slots__ = ("session",)
    SESSION_FIELD_NUMBER: _ClassVar[int]
    session: SessionDescriptor
    def __init__(self, session: _Optional[_Union[SessionDescriptor, _Mapping]] = ...) -> None: ...

class ResumeSimulationRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class ResumeSimulationResponse(_message.Message):
    __slots__ = ("session",)
    SESSION_FIELD_NUMBER: _ClassVar[int]
    session: SessionDescriptor
    def __init__(self, session: _Optional[_Union[SessionDescriptor, _Mapping]] = ...) -> None: ...

class StopSimulationRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class StopSimulationResponse(_message.Message):
    __slots__ = ("session",)
    SESSION_FIELD_NUMBER: _ClassVar[int]
    session: SessionDescriptor
    def __init__(self, session: _Optional[_Union[SessionDescriptor, _Mapping]] = ...) -> None: ...

class StreamWaveformsRequest(_message.Message):
    __slots__ = ("session_id", "signals", "decimation", "start_time")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SIGNALS_FIELD_NUMBER: _ClassVar[int]
    DECIMATION_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    signals: _containers.RepeatedScalarFieldContainer[str]
    decimation: int
    start_time: _wrappers_pb2.DoubleValue
    def __init__(self, session_id: _Optional[str] = ..., signals: _Optional[_Iterable[str]] = ..., decimation: _Optional[int] = ..., start_time: _Optional[_Union[_wrappers_pb2.DoubleValue, _Mapping]] = ...) -> None: ...

class WaveformStreamHeader(_message.Message):
    __slots__ = ("session_id", "signals", "tstart", "tstop", "total_samples")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SIGNALS_FIELD_NUMBER: _ClassVar[int]
    TSTART_FIELD_NUMBER: _ClassVar[int]
    TSTOP_FIELD_NUMBER: _ClassVar[int]
    TOTAL_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    signals: _containers.RepeatedScalarFieldContainer[str]
    tstart: float
    tstop: float
    total_samples: int
    def __init__(self, session_id: _Optional[str] = ..., signals: _Optional[_Iterable[str]] = ..., tstart: _Optional[float] = ..., tstop: _Optional[float] = ..., total_samples: _Optional[int] = ...) -> None: ...

class WaveformSample(_message.Message):
    __slots__ = ("time", "values")
    TIME_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    time: float
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, time: _Optional[float] = ..., values: _Optional[_Iterable[float]] = ...) -> None: ...

class WaveformStreamComplete(_message.Message):
    __slots__ = ("final_status", "error_message")
    FINAL_STATUS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    final_status: SessionStatus
    error_message: str
    def __init__(self, final_status: _Optional[_Union[SessionStatus, str]] = ..., error_message: _Optional[str] = ...) -> None: ...

class WaveformStreamResponse(_message.Message):
    __slots__ = ("header", "sample", "complete")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_FIELD_NUMBER: _ClassVar[int]
    COMPLETE_FIELD_NUMBER: _ClassVar[int]
    header: WaveformStreamHeader
    sample: WaveformSample
    complete: WaveformStreamComplete
    def __init__(self, header: _Optional[_Union[WaveformStreamHeader, _Mapping]] = ..., sample: _Optional[_Union[WaveformSample, _Mapping]] = ..., complete: _Optional[_Union[WaveformStreamComplete, _Mapping]] = ...) -> None: ...

class GetResultRequest(_message.Message):
    __slots__ = ("session_id", "signals", "start_time", "end_time")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SIGNALS_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    signals: _containers.RepeatedScalarFieldContainer[str]
    start_time: _wrappers_pb2.DoubleValue
    end_time: _wrappers_pb2.DoubleValue
    def __init__(self, session_id: _Optional[str] = ..., signals: _Optional[_Iterable[str]] = ..., start_time: _Optional[_Union[_wrappers_pb2.DoubleValue, _Mapping]] = ..., end_time: _Optional[_Union[_wrappers_pb2.DoubleValue, _Mapping]] = ...) -> None: ...

class ResultMetadata(_message.Message):
    __slots__ = ("start_time", "end_time", "sample_count", "signals", "status", "error_message")
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_COUNT_FIELD_NUMBER: _ClassVar[int]
    SIGNALS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    start_time: float
    end_time: float
    sample_count: int
    signals: _containers.RepeatedScalarFieldContainer[str]
    status: SessionStatus
    error_message: str
    def __init__(self, start_time: _Optional[float] = ..., end_time: _Optional[float] = ..., sample_count: _Optional[int] = ..., signals: _Optional[_Iterable[str]] = ..., status: _Optional[_Union[SessionStatus, str]] = ..., error_message: _Optional[str] = ...) -> None: ...

class GetResultResponse(_message.Message):
    __slots__ = ("metadata",)
    METADATA_FIELD_NUMBER: _ClassVar[int]
    metadata: ResultMetadata
    def __init__(self, metadata: _Optional[_Union[ResultMetadata, _Mapping]] = ...) -> None: ...

class DownloadResultRequest(_message.Message):
    __slots__ = ("session_id", "signals", "start_time", "end_time", "format")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SIGNALS_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    signals: _containers.RepeatedScalarFieldContainer[str]
    start_time: _wrappers_pb2.DoubleValue
    end_time: _wrappers_pb2.DoubleValue
    format: SimulationResultFormat
    def __init__(self, session_id: _Optional[str] = ..., signals: _Optional[_Iterable[str]] = ..., start_time: _Optional[_Union[_wrappers_pb2.DoubleValue, _Mapping]] = ..., end_time: _Optional[_Union[_wrappers_pb2.DoubleValue, _Mapping]] = ..., format: _Optional[_Union[SimulationResultFormat, str]] = ...) -> None: ...

class DownloadResultResponse(_message.Message):
    __slots__ = ("chunk", "last_chunk", "content_type", "filename")
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    LAST_CHUNK_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    chunk: bytes
    last_chunk: bool
    content_type: str
    filename: str
    def __init__(self, chunk: _Optional[bytes] = ..., last_chunk: bool = ..., content_type: _Optional[str] = ..., filename: _Optional[str] = ...) -> None: ...

class UploadModelRequest(_message.Message):
    __slots__ = ("model",)
    MODEL_FIELD_NUMBER: _ClassVar[int]
    model: CircuitModel
    def __init__(self, model: _Optional[_Union[CircuitModel, _Mapping]] = ...) -> None: ...

class UploadModelResponse(_message.Message):
    __slots__ = ("model",)
    MODEL_FIELD_NUMBER: _ClassVar[int]
    model: ModelDescriptor
    def __init__(self, model: _Optional[_Union[ModelDescriptor, _Mapping]] = ...) -> None: ...

class ListModelsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListModelsResponse(_message.Message):
    __slots__ = ("models",)
    MODELS_FIELD_NUMBER: _ClassVar[int]
    models: _containers.RepeatedCompositeFieldContainer[ModelDescriptor]
    def __init__(self, models: _Optional[_Iterable[_Union[ModelDescriptor, _Mapping]]] = ...) -> None: ...

class GetModelRequest(_message.Message):
    __slots__ = ("model_id",)
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    def __init__(self, model_id: _Optional[str] = ...) -> None: ...

class GetModelResponse(_message.Message):
    __slots__ = ("model",)
    MODEL_FIELD_NUMBER: _ClassVar[int]
    model: ModelDescriptor
    def __init__(self, model: _Optional[_Union[ModelDescriptor, _Mapping]] = ...) -> None: ...

class DeleteModelRequest(_message.Message):
    __slots__ = ("model_id", "force")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    FORCE_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    force: bool
    def __init__(self, model_id: _Optional[str] = ..., force: bool = ...) -> None: ...

class DeleteModelResponse(_message.Message):
    __slots__ = ("deleted",)
    DELETED_FIELD_NUMBER: _ClassVar[int]
    deleted: bool
    def __init__(self, deleted: bool = ...) -> None: ...

class SweepParameter(_message.Message):
    __slots__ = ("name", "start", "stop", "steps", "logarithmic")
    NAME_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    STOP_FIELD_NUMBER: _ClassVar[int]
    STEPS_FIELD_NUMBER: _ClassVar[int]
    LOGARITHMIC_FIELD_NUMBER: _ClassVar[int]
    name: str
    start: float
    stop: float
    steps: int
    logarithmic: bool
    def __init__(self, name: _Optional[str] = ..., start: _Optional[float] = ..., stop: _Optional[float] = ..., steps: _Optional[int] = ..., logarithmic: bool = ...) -> None: ...

class CreateSweepRequest(_message.Message):
    __slots__ = ("model_id", "base_options", "parameters", "signals")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    BASE_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    SIGNALS_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    base_options: SimulationOptions
    parameters: _containers.RepeatedCompositeFieldContainer[SweepParameter]
    signals: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, model_id: _Optional[str] = ..., base_options: _Optional[_Union[SimulationOptions, _Mapping]] = ..., parameters: _Optional[_Iterable[_Union[SweepParameter, _Mapping]]] = ..., signals: _Optional[_Iterable[str]] = ...) -> None: ...

class CreateSweepResponse(_message.Message):
    __slots__ = ("sweep_id",)
    SWEEP_ID_FIELD_NUMBER: _ClassVar[int]
    sweep_id: str
    def __init__(self, sweep_id: _Optional[str] = ...) -> None: ...

class RunSweepRequest(_message.Message):
    __slots__ = ("sweep_id", "parallel")
    SWEEP_ID_FIELD_NUMBER: _ClassVar[int]
    PARALLEL_FIELD_NUMBER: _ClassVar[int]
    sweep_id: str
    parallel: bool
    def __init__(self, sweep_id: _Optional[str] = ..., parallel: bool = ...) -> None: ...

class RunSweepResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: SweepStatus
    def __init__(self, status: _Optional[_Union[SweepStatus, str]] = ...) -> None: ...

class GetSweepResultsRequest(_message.Message):
    __slots__ = ("sweep_id",)
    SWEEP_ID_FIELD_NUMBER: _ClassVar[int]
    sweep_id: str
    def __init__(self, sweep_id: _Optional[str] = ...) -> None: ...

class ParameterValue(_message.Message):
    __slots__ = ("name", "value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: float
    def __init__(self, name: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...

class SweepResult(_message.Message):
    __slots__ = ("parameters", "metadata", "final_values")
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    FINAL_VALUES_FIELD_NUMBER: _ClassVar[int]
    parameters: _containers.RepeatedCompositeFieldContainer[ParameterValue]
    metadata: ResultMetadata
    final_values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, parameters: _Optional[_Iterable[_Union[ParameterValue, _Mapping]]] = ..., metadata: _Optional[_Union[ResultMetadata, _Mapping]] = ..., final_values: _Optional[_Iterable[float]] = ...) -> None: ...

class GetSweepResultsResponse(_message.Message):
    __slots__ = ("status", "results")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    status: SweepStatus
    results: _containers.RepeatedCompositeFieldContainer[SweepResult]
    def __init__(self, status: _Optional[_Union[SweepStatus, str]] = ..., results: _Optional[_Iterable[_Union[SweepResult, _Mapping]]] = ...) -> None: ...
